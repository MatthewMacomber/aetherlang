// Executable linking pipeline for Aether
// Handles linking of object files with runtime libraries and system dependencies

use crate::runtime::runtime_linking::{RuntimeLinkingConfig, LTOConfig, DebugConfig};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::fs;
use std::env;
use std::ffi::OsStr;

/// Executable linking pipeline
pub struct ExecutableLinker {
    /// Target configuration
    target_triple: String,
    /// Runtime linking configuration
    linking_config: RuntimeLinkingConfig,
    /// Linker executable path
    linker_path: PathBuf,
    /// Additional linker arguments
    extra_args: Vec<String>,
    /// Environment variables for linking
    env_vars: HashMap<String, String>,
    /// Verbose output
    verbose: bool,
}

/// Linking result information
#[derive(Debug, Clone)]
pub struct LinkingResult {
    /// Output executable path
    pub executable_path: PathBuf,
    /// Linking was successful
    pub success: bool,
    /// Linker output
    pub output: String,
    /// Linker errors
    pub errors: String,
    /// Linking statistics
    pub statistics: LinkingStatistics,
}

/// Linking statistics
#[derive(Debug, Clone, Default)]
pub struct LinkingStatistics {
    /// Number of object files linked
    pub object_files_count: usize,
    /// Number of static libraries linked
    pub static_libraries_count: usize,
    /// Number of dynamic libraries linked
    pub dynamic_libraries_count: usize,
    /// Total linking time in milliseconds
    pub linking_time_ms: u64,
    /// Final executable size in bytes
    pub executable_size_bytes: u64,
    /// Debug information included
    pub debug_info_included: bool,
    /// LTO applied
    pub lto_applied: bool,
}

/// Linking errors
#[derive(Debug, Clone)]
pub enum LinkingError {
    /// Linker not found
    LinkerNotFound(String),
    /// Object file not found
    ObjectFileNotFound(PathBuf),
    /// Library not found
    LibraryNotFound(String),
    /// Linking failed
    LinkingFailed { exit_code: i32, stderr: String },
    /// I/O error
    IOError(String),
    /// Configuration error
    ConfigurationError(String),
    /// Unsupported target
    UnsupportedTarget(String),
}

impl std::fmt::Display for LinkingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkingError::LinkerNotFound(linker) => write!(f, "Linker not found: {}", linker),
            LinkingError::ObjectFileNotFound(path) => write!(f, "Object file not found: {}", path.display()),
            LinkingError::LibraryNotFound(lib) => write!(f, "Library not found: {}", lib),
            LinkingError::LinkingFailed { exit_code, stderr } => {
                write!(f, "Linking failed with exit code {}: {}", exit_code, stderr)
            }
            LinkingError::IOError(msg) => write!(f, "I/O error: {}", msg),
            LinkingError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            LinkingError::UnsupportedTarget(target) => write!(f, "Unsupported target: {}", target),
        }
    }
}

impl std::error::Error for LinkingError {}

impl From<std::io::Error> for LinkingError {
    fn from(err: std::io::Error) -> Self {
        LinkingError::IOError(err.to_string())
    }
}

impl ExecutableLinker {
    /// Create new executable linker
    pub fn new(target_triple: String, linking_config: RuntimeLinkingConfig) -> Result<Self, LinkingError> {
        let linker_path = Self::find_linker(&target_triple)?;
        
        Ok(ExecutableLinker {
            target_triple,
            linking_config,
            linker_path,
            extra_args: Vec::new(),
            env_vars: HashMap::new(),
            verbose: false,
        })
    }

    /// Find appropriate linker for target
    fn find_linker(target_triple: &str) -> Result<PathBuf, LinkingError> {
        // Parse target triple to determine appropriate linker
        let parts: Vec<&str> = target_triple.split('-').collect();
        let arch = parts.get(0).unwrap_or(&"unknown");
        let os = parts.get(2).unwrap_or(&"unknown");

        // Try target-specific linkers first
        let target_linkers = match *os {
            "windows" => vec!["link.exe", "lld-link.exe", "x86_64-w64-mingw32-gcc"],
            "darwin" | "macos" => vec!["ld", "lld", "clang"],
            "linux" => vec!["ld", "lld", "gcc", "clang"],
            "wasm32" => vec!["wasm-ld", "lld"],
            _ => vec!["ld", "gcc", "clang"],
        };

        // Add cross-compilation prefixes
        let mut linker_candidates = Vec::new();
        for linker in &target_linkers {
            // Try target-prefixed version first
            linker_candidates.push(format!("{}-{}", target_triple, linker));
            linker_candidates.push(format!("{}-{}", arch, linker));
            // Then try plain version
            linker_candidates.push(linker.to_string());
        }

        // Search for linker in PATH
        for candidate in linker_candidates {
            if let Ok(path) = which::which(&candidate) {
                return Ok(path);
            }
        }

        // Try common system locations
        let system_paths = vec![
            "/usr/bin",
            "/usr/local/bin",
            "/opt/local/bin",
            "/mingw64/bin",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.30.30705\\bin\\Hostx64\\x64",
            "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64",
        ];

        for sys_path in system_paths {
            for linker in &target_linkers {
                let full_path = Path::new(sys_path).join(linker);
                if full_path.exists() {
                    return Ok(full_path);
                }
            }
        }

        Err(LinkingError::LinkerNotFound(format!(
            "No suitable linker found for target {}", target_triple
        )))
    }

    /// Set verbose output
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Add extra linker argument
    pub fn add_extra_arg(&mut self, arg: String) {
        self.extra_args.push(arg);
    }

    /// Set environment variable for linking
    pub fn set_env_var(&mut self, key: String, value: String) {
        self.env_vars.insert(key, value);
    }

    /// Link object files into executable
    pub fn link_executable(
        &self,
        object_files: &[PathBuf],
        output_path: &Path,
    ) -> Result<LinkingResult, LinkingError> {
        let start_time = std::time::Instant::now();
        let mut statistics = LinkingStatistics::default();

        // Validate input files
        self.validate_object_files(object_files)?;
        statistics.object_files_count = object_files.len();

        // Generate linker arguments
        let mut args = self.generate_linker_arguments(object_files, output_path)?;
        
        // Add extra arguments
        args.extend(self.extra_args.clone());

        // Count libraries for statistics
        statistics.static_libraries_count = self.linking_config.static_libraries.len();
        statistics.dynamic_libraries_count = self.linking_config.dynamic_libraries.len();
        statistics.debug_info_included = self.linking_config.debug_config.enabled;
        statistics.lto_applied = self.linking_config.lto_config.enabled;

        if self.verbose {
            println!("Linking with command: {} {}", self.linker_path.display(), args.join(" "));
        }

        // Execute linker
        let mut command = Command::new(&self.linker_path);
        command.args(&args);
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        // Set environment variables
        for (key, value) in &self.env_vars {
            command.env(key, value);
        }

        // Add system environment variables for linking
        self.setup_linking_environment(&mut command)?;

        let output = command.output()?;
        let linking_time = start_time.elapsed();
        statistics.linking_time_ms = linking_time.as_millis() as u64;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let success = output.status.success();
        
        if success {
            // Get executable size
            if let Ok(metadata) = fs::metadata(output_path) {
                statistics.executable_size_bytes = metadata.len();
            }

            // Generate debug information if requested
            if self.linking_config.debug_config.enabled && self.linking_config.debug_config.separate_debug_info {
                self.generate_separate_debug_info(output_path)?;
            }
        }

        Ok(LinkingResult {
            executable_path: output_path.to_path_buf(),
            success,
            output: stdout,
            errors: stderr,
            statistics,
        })
    }

    /// Validate that all object files exist and are readable
    fn validate_object_files(&self, object_files: &[PathBuf]) -> Result<(), LinkingError> {
        for obj_file in object_files {
            if !obj_file.exists() {
                return Err(LinkingError::ObjectFileNotFound(obj_file.clone()));
            }
            
            // Check if file is readable
            if let Err(e) = fs::File::open(obj_file) {
                return Err(LinkingError::IOError(format!(
                    "Cannot read object file {}: {}", obj_file.display(), e
                )));
            }
        }
        Ok(())
    }

    /// Generate linker arguments based on configuration
    fn generate_linker_arguments(
        &self,
        object_files: &[PathBuf],
        output_path: &Path,
    ) -> Result<Vec<String>, LinkingError> {
        let mut args = Vec::new();

        // Determine linker type
        let linker_name = self.linker_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        match linker_name {
            name if name.contains("link.exe") || name.contains("lld-link") => {
                // Microsoft linker or LLD in link.exe mode
                self.generate_msvc_linker_args(&mut args, object_files, output_path)?;
            }
            name if name.contains("ld") || name.contains("lld") => {
                // GNU ld or LLD in GNU mode
                self.generate_gnu_linker_args(&mut args, object_files, output_path)?;
            }
            name if name.contains("gcc") || name.contains("clang") => {
                // GCC or Clang as linker driver
                self.generate_compiler_linker_args(&mut args, object_files, output_path)?;
            }
            name if name.contains("wasm-ld") => {
                // WebAssembly linker
                self.generate_wasm_linker_args(&mut args, object_files, output_path)?;
            }
            _ => {
                // Default to GNU-style arguments
                self.generate_gnu_linker_args(&mut args, object_files, output_path)?;
            }
        }

        Ok(args)
    }

    /// Generate MSVC linker arguments
    fn generate_msvc_linker_args(
        &self,
        args: &mut Vec<String>,
        object_files: &[PathBuf],
        output_path: &Path,
    ) -> Result<(), LinkingError> {
        // Output file
        args.push(format!("/OUT:{}", output_path.display()));

        // Object files
        for obj_file in object_files {
            args.push(obj_file.to_string_lossy().to_string());
        }

        // Library paths
        for lib_path in &self.linking_config.library_paths {
            args.push(format!("/LIBPATH:{}", lib_path.display()));
        }

        // Static libraries
        for lib in &self.linking_config.static_libraries {
            args.push(format!("{}.lib", lib));
        }

        // Dynamic libraries
        for lib in &self.linking_config.dynamic_libraries {
            args.push(format!("{}.lib", lib));
        }

        // System libraries
        for lib in &self.linking_config.system_libraries {
            args.push(format!("{}.lib", lib));
        }

        // Debug information
        if self.linking_config.debug_config.enabled {
            args.push("/DEBUG".to_string());
            if self.linking_config.debug_config.separate_debug_info {
                args.push("/DEBUG:FULL".to_string());
            }
        }

        // LTO
        if self.linking_config.lto_config.enabled {
            args.push("/LTCG".to_string());
        }

        // Custom linker flags
        args.extend(self.linking_config.linker_flags.clone());

        Ok(())
    }

    /// Generate GNU linker arguments
    fn generate_gnu_linker_args(
        &self,
        args: &mut Vec<String>,
        object_files: &[PathBuf],
        output_path: &Path,
    ) -> Result<(), LinkingError> {
        // Output file
        args.push("-o".to_string());
        args.push(output_path.to_string_lossy().to_string());

        // Object files
        for obj_file in object_files {
            args.push(obj_file.to_string_lossy().to_string());
        }

        // Library paths
        for lib_path in &self.linking_config.library_paths {
            args.push(format!("-L{}", lib_path.display()));
        }

        // Static libraries (order matters for GNU ld)
        for lib in &self.linking_config.static_libraries {
            args.push(format!("-l{}", lib));
        }

        // Dynamic libraries
        for lib in &self.linking_config.dynamic_libraries {
            args.push(format!("-l{}", lib));
        }

        // System libraries
        for lib in &self.linking_config.system_libraries {
            args.push(format!("-l{}", lib));
        }

        // Debug information
        if self.linking_config.debug_config.enabled {
            args.push(format!("-g{}", self.linking_config.debug_config.dwarf_version));
        }

        // Strip symbols
        if self.linking_config.debug_config.strip_symbols {
            args.push("-s".to_string());
        }

        // Custom linker flags
        args.extend(self.linking_config.linker_flags.clone());

        Ok(())
    }

    /// Generate compiler-as-linker arguments (GCC/Clang)
    fn generate_compiler_linker_args(
        &self,
        args: &mut Vec<String>,
        object_files: &[PathBuf],
        output_path: &Path,
    ) -> Result<(), LinkingError> {
        // Output file
        args.push("-o".to_string());
        args.push(output_path.to_string_lossy().to_string());

        // Object files
        for obj_file in object_files {
            args.push(obj_file.to_string_lossy().to_string());
        }

        // Library paths
        for lib_path in &self.linking_config.library_paths {
            args.push(format!("-L{}", lib_path.display()));
        }

        // Static libraries
        for lib in &self.linking_config.static_libraries {
            args.push(format!("-l{}", lib));
        }

        // Dynamic libraries
        for lib in &self.linking_config.dynamic_libraries {
            args.push(format!("-l{}", lib));
        }

        // System libraries
        for lib in &self.linking_config.system_libraries {
            args.push(format!("-l{}", lib));
        }

        // Frameworks (macOS)
        for framework in &self.linking_config.frameworks {
            args.push("-framework".to_string());
            args.push(framework.clone());
        }

        // Framework paths (macOS)
        for framework_path in &self.linking_config.framework_paths {
            args.push(format!("-F{}", framework_path.display()));
        }

        // Debug information
        if self.linking_config.debug_config.enabled {
            args.push(format!("-g{}", self.linking_config.debug_config.dwarf_version));
        }

        // Strip symbols
        if self.linking_config.debug_config.strip_symbols {
            args.push("-s".to_string());
        }

        // LTO
        if self.linking_config.lto_config.enabled {
            match self.linking_config.lto_config.lto_type {
                crate::runtime::runtime_linking::LTOType::Thin => args.push("-flto=thin".to_string()),
                crate::runtime::runtime_linking::LTOType::Full => args.push("-flto".to_string()),
                crate::runtime::runtime_linking::LTOType::None => {}
            }
        }

        // Custom linker flags
        args.extend(self.linking_config.linker_flags.clone());

        Ok(())
    }

    /// Generate WebAssembly linker arguments
    fn generate_wasm_linker_args(
        &self,
        args: &mut Vec<String>,
        object_files: &[PathBuf],
        output_path: &Path,
    ) -> Result<(), LinkingError> {
        // Output file
        args.push("-o".to_string());
        args.push(output_path.to_string_lossy().to_string());

        // Object files
        for obj_file in object_files {
            args.push(obj_file.to_string_lossy().to_string());
        }

        // WebAssembly-specific flags
        args.push("--no-entry".to_string());
        args.push("--export-all".to_string());
        args.push("--allow-undefined".to_string());

        // Custom linker flags
        args.extend(self.linking_config.linker_flags.clone());

        Ok(())
    }

    /// Setup environment variables for linking
    fn setup_linking_environment(&self, command: &mut Command) -> Result<(), LinkingError> {
        // Set up library paths
        let mut library_path_env = String::new();
        for lib_path in &self.linking_config.library_paths {
            if !library_path_env.is_empty() {
                library_path_env.push(':');
            }
            library_path_env.push_str(&lib_path.to_string_lossy());
        }

        if !library_path_env.is_empty() {
            // Set appropriate environment variable based on platform
            let parts: Vec<&str> = self.target_triple.split('-').collect();
            let os = parts.get(2).unwrap_or(&"unknown");
            
            match *os {
                "windows" => command.env("LIB", &library_path_env),
                "darwin" | "macos" => command.env("DYLD_LIBRARY_PATH", &library_path_env),
                _ => command.env("LD_LIBRARY_PATH", &library_path_env),
            };
        }

        // Set up additional environment variables for specific toolchains
        if self.linking_config.has_cuda() {
            if let Ok(cuda_path) = env::var("CUDA_PATH") {
                command.env("CUDA_PATH", cuda_path);
            }
        }

        if self.linking_config.has_intel_mkl() {
            if let Ok(mkl_root) = env::var("MKLROOT") {
                command.env("MKLROOT", mkl_root);
            }
        }

        Ok(())
    }

    /// Generate separate debug information file
    fn generate_separate_debug_info(&self, executable_path: &Path) -> Result<(), LinkingError> {
        let debug_path = executable_path.with_extension("debug");
        
        // Use objcopy to extract debug information
        let objcopy_result = Command::new("objcopy")
            .args(&[
                "--only-keep-debug",
                &executable_path.to_string_lossy(),
                &debug_path.to_string_lossy(),
            ])
            .output();

        match objcopy_result {
            Ok(output) if output.status.success() => {
                // Strip debug info from executable and add debug link
                let _strip_result = Command::new("objcopy")
                    .args(&[
                        "--strip-debug",
                        &format!("--add-gnu-debuglink={}", debug_path.display()),
                        &executable_path.to_string_lossy(),
                    ])
                    .output();
                
                if self.verbose {
                    println!("Generated separate debug info: {}", debug_path.display());
                }
            }
            Ok(output) => {
                if self.verbose {
                    eprintln!("Failed to generate debug info: {}", 
                        String::from_utf8_lossy(&output.stderr));
                }
            }
            Err(_) => {
                if self.verbose {
                    eprintln!("objcopy not available, skipping separate debug info generation");
                }
            }
        }

        Ok(())
    }

    /// Link with custom library search paths
    pub fn link_with_search_paths(
        &mut self,
        object_files: &[PathBuf],
        output_path: &Path,
        additional_lib_paths: &[PathBuf],
    ) -> Result<LinkingResult, LinkingError> {
        // Temporarily add search paths
        let original_paths = self.linking_config.library_paths.clone();
        self.linking_config.library_paths.extend_from_slice(additional_lib_paths);
        
        let result = self.link_executable(object_files, output_path);
        
        // Restore original paths
        self.linking_config.library_paths = original_paths;
        
        result
    }

    /// Get linking configuration
    pub fn get_linking_config(&self) -> &RuntimeLinkingConfig {
        &self.linking_config
    }

    /// Get mutable linking configuration
    pub fn get_linking_config_mut(&mut self) -> &mut RuntimeLinkingConfig {
        &mut self.linking_config
    }

    /// Get target triple
    pub fn get_target_triple(&self) -> &str {
        &self.target_triple
    }

    /// Get linker path
    pub fn get_linker_path(&self) -> &Path {
        &self.linker_path
    }

    /// Test linking configuration
    pub fn test_linking_config(&self) -> Result<(), LinkingError> {
        // Check if linker is accessible
        let output = Command::new(&self.linker_path)
            .arg("--version")
            .output();

        match output {
            Ok(result) if result.status.success() => {
                if self.verbose {
                    println!("Linker test successful: {}", 
                        String::from_utf8_lossy(&result.stdout));
                }
                Ok(())
            }
            Ok(result) => {
                Err(LinkingError::ConfigurationError(format!(
                    "Linker test failed: {}", 
                    String::from_utf8_lossy(&result.stderr)
                )))
            }
            Err(e) => {
                Err(LinkingError::ConfigurationError(format!(
                    "Cannot execute linker: {}", e
                )))
            }
        }
    }
}

impl Drop for ExecutableLinker {
    fn drop(&mut self) {
        // Cleanup any temporary files or resources if needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_executable_linker_creation() {
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        let result = ExecutableLinker::new("x86_64-unknown-linux-gnu".to_string(), config);
        
        // This might fail if no linker is available, which is okay for testing
        match result {
            Ok(linker) => {
                assert_eq!(linker.get_target_triple(), "x86_64-unknown-linux-gnu");
            }
            Err(LinkingError::LinkerNotFound(_)) => {
                // Expected in environments without development tools
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_linker_argument_generation() {
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        if let Ok(linker) = ExecutableLinker::new("x86_64-unknown-linux-gnu".to_string(), config) {
            let temp_dir = TempDir::new().unwrap();
            let obj_file = temp_dir.path().join("test.o");
            let output_file = temp_dir.path().join("test_output");
            
            // Create a dummy object file
            std::fs::write(&obj_file, b"dummy object file").unwrap();
            
            let result = linker.generate_linker_arguments(&[obj_file], &output_file);
            match result {
                Ok(args) => {
                    assert!(args.contains(&"-o".to_string()) || args.contains(&"/OUT:test_output".to_string()));
                }
                Err(e) => println!("Argument generation failed (expected in test environment): {}", e),
            }
        }
    }

    #[test]
    fn test_object_file_validation() {
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        if let Ok(linker) = ExecutableLinker::new("x86_64-unknown-linux-gnu".to_string(), config) {
            let temp_dir = TempDir::new().unwrap();
            let existing_file = temp_dir.path().join("exists.o");
            let missing_file = temp_dir.path().join("missing.o");
            
            // Create one file but not the other
            std::fs::write(&existing_file, b"dummy").unwrap();
            
            // Test with existing file
            assert!(linker.validate_object_files(&[existing_file]).is_ok());
            
            // Test with missing file
            assert!(matches!(
                linker.validate_object_files(&[missing_file]),
                Err(LinkingError::ObjectFileNotFound(_))
            ));
        }
    }

    #[test]
    fn test_linking_statistics() {
        let mut stats = LinkingStatistics::default();
        stats.object_files_count = 5;
        stats.static_libraries_count = 3;
        stats.dynamic_libraries_count = 2;
        stats.linking_time_ms = 1500;
        stats.executable_size_bytes = 1024 * 1024; // 1MB
        stats.debug_info_included = true;
        stats.lto_applied = false;
        
        assert_eq!(stats.object_files_count, 5);
        assert_eq!(stats.static_libraries_count, 3);
        assert_eq!(stats.dynamic_libraries_count, 2);
        assert!(stats.debug_info_included);
        assert!(!stats.lto_applied);
    }
}