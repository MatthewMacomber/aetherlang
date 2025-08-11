// Runtime library linking configuration for Aether
// Handles linking of runtime libraries with generated code

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::env;

/// Runtime library linking configuration
#[derive(Debug, Clone)]
pub struct RuntimeLinkingConfig {
    /// Target triple (e.g., "x86_64-unknown-linux-gnu")
    pub target_triple: String,
    /// Runtime library paths
    pub library_paths: Vec<PathBuf>,
    /// Static libraries to link
    pub static_libraries: Vec<String>,
    /// Dynamic libraries to link
    pub dynamic_libraries: Vec<String>,
    /// System libraries to link
    pub system_libraries: Vec<String>,
    /// Linker flags
    pub linker_flags: Vec<String>,
    /// Framework paths (macOS)
    pub framework_paths: Vec<PathBuf>,
    /// Frameworks to link (macOS)
    pub frameworks: Vec<String>,
    /// Link-time optimization settings
    pub lto_config: LTOConfig,
    /// Debug information settings
    pub debug_config: DebugConfig,
}

/// Link-time optimization configuration
#[derive(Debug, Clone)]
pub struct LTOConfig {
    /// Enable LTO
    pub enabled: bool,
    /// LTO type (thin, full)
    pub lto_type: LTOType,
    /// Optimization level for LTO
    pub optimization_level: u8,
}

/// LTO types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LTOType {
    None,
    Thin,
    Full,
}

/// Debug information configuration
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// Include debug information
    pub enabled: bool,
    /// DWARF version
    pub dwarf_version: u32,
    /// Strip symbols after linking
    pub strip_symbols: bool,
    /// Generate separate debug info file
    pub separate_debug_info: bool,
}

impl RuntimeLinkingConfig {
    /// Create new runtime linking configuration
    pub fn new(target_triple: String) -> Self {
        let mut config = RuntimeLinkingConfig {
            target_triple: target_triple.clone(),
            library_paths: Vec::new(),
            static_libraries: Vec::new(),
            dynamic_libraries: Vec::new(),
            system_libraries: Vec::new(),
            linker_flags: Vec::new(),
            framework_paths: Vec::new(),
            frameworks: Vec::new(),
            lto_config: LTOConfig::default(),
            debug_config: DebugConfig::default(),
        };

        // Configure for target
        config.configure_for_target(&target_triple);
        config
    }

    /// Configure linking for specific target
    fn configure_for_target(&mut self, target_triple: &str) {
        // Parse target triple
        let parts: Vec<&str> = target_triple.split('-').collect();
        let arch = parts.get(0).unwrap_or(&"unknown");
        let os = parts.get(2).unwrap_or(&"unknown");

        // Add Aether runtime library
        self.static_libraries.push("aether_runtime".to_string());

        match *os {
            "linux" => self.configure_linux(),
            "windows" => self.configure_windows(),
            "darwin" | "macos" => self.configure_macos(),
            "wasm32" => self.configure_wasm(),
            _ => self.configure_generic(),
        }

        // Architecture-specific configuration
        match *arch {
            "x86_64" => self.configure_x86_64(),
            "aarch64" => self.configure_aarch64(),
            "wasm32" => self.configure_wasm32(),
            _ => {}
        }
    }

    /// Configure for Linux
    fn configure_linux(&mut self) {
        // System libraries
        self.system_libraries.extend_from_slice(&[
            "c".to_string(),
            "m".to_string(),
            "pthread".to_string(),
            "dl".to_string(),
        ]);

        // Math libraries for AI operations
        self.dynamic_libraries.extend_from_slice(&[
            "blas".to_string(),
            "lapack".to_string(),
        ]);

        // Optional GPU libraries
        if self.has_cuda() {
            self.dynamic_libraries.extend_from_slice(&[
                "cuda".to_string(),
                "cudart".to_string(),
                "cublas".to_string(),
                "curand".to_string(),
            ]);
        }

        // OpenMP for parallel execution
        if self.has_openmp() {
            self.dynamic_libraries.push("gomp".to_string());
        }

        // Linker flags
        self.linker_flags.extend_from_slice(&[
            "-Wl,--as-needed".to_string(),
            "-Wl,--gc-sections".to_string(),
        ]);
    }

    /// Configure for Windows
    fn configure_windows(&mut self) {
        // System libraries
        self.system_libraries.extend_from_slice(&[
            "kernel32".to_string(),
            "user32".to_string(),
            "msvcrt".to_string(),
        ]);

        // Math libraries
        if self.has_intel_mkl() {
            self.dynamic_libraries.extend_from_slice(&[
                "mkl_core".to_string(),
                "mkl_intel_lp64".to_string(),
                "mkl_sequential".to_string(),
            ]);
        }

        // CUDA libraries
        if self.has_cuda() {
            self.dynamic_libraries.extend_from_slice(&[
                "cuda".to_string(),
                "cudart".to_string(),
                "cublas".to_string(),
            ]);
        }

        // Linker flags
        self.linker_flags.extend_from_slice(&[
            "/SUBSYSTEM:CONSOLE".to_string(),
            "/DYNAMICBASE".to_string(),
            "/NXCOMPAT".to_string(),
        ]);
    }

    /// Configure for macOS
    fn configure_macos(&mut self) {
        // System libraries
        self.system_libraries.extend_from_slice(&[
            "c".to_string(),
            "m".to_string(),
            "pthread".to_string(),
        ]);

        // Frameworks
        self.frameworks.extend_from_slice(&[
            "Foundation".to_string(),
            "Accelerate".to_string(), // Apple's optimized BLAS/LAPACK
        ]);

        // Metal for GPU compute
        if self.has_metal() {
            self.frameworks.extend_from_slice(&[
                "Metal".to_string(),
                "MetalKit".to_string(),
            ]);
        }

        // Linker flags
        self.linker_flags.extend_from_slice(&[
            "-dead_strip".to_string(),
            "-no_compact_unwind".to_string(),
        ]);
    }

    /// Configure for WebAssembly
    fn configure_wasm(&mut self) {
        // WebAssembly doesn't use traditional linking
        self.linker_flags.extend_from_slice(&[
            "--export-dynamic".to_string(),
            "--allow-undefined".to_string(),
            "--import-memory".to_string(),
        ]);
    }

    /// Configure for generic target
    fn configure_generic(&mut self) {
        self.system_libraries.extend_from_slice(&[
            "c".to_string(),
            "m".to_string(),
        ]);
    }

    /// Configure for x86_64 architecture
    fn configure_x86_64(&mut self) {
        // x86_64 specific optimizations
        self.linker_flags.push("-march=x86-64".to_string());
    }

    /// Configure for AArch64 architecture
    fn configure_aarch64(&mut self) {
        // ARM64 specific optimizations
        self.linker_flags.push("-march=armv8-a".to_string());
    }

    /// Configure for WebAssembly 32-bit
    fn configure_wasm32(&mut self) {
        // WebAssembly specific settings
        self.linker_flags.extend_from_slice(&[
            "--no-entry".to_string(),
            "--export-all".to_string(),
        ]);
    }

    /// Check if CUDA is available
    pub fn has_cuda(&self) -> bool {
        // Check for CUDA installation
        env::var("CUDA_PATH").is_ok() || 
        Path::new("/usr/local/cuda").exists() ||
        Command::new("nvcc").arg("--version").output().is_ok()
    }

    /// Check if OpenMP is available
    fn has_openmp(&self) -> bool {
        // Check for OpenMP support
        Command::new("gcc").args(&["-fopenmp", "--version"]).output().is_ok() ||
        Command::new("clang").args(&["-fopenmp", "--version"]).output().is_ok()
    }

    /// Check if Intel MKL is available
    pub fn has_intel_mkl(&self) -> bool {
        env::var("MKLROOT").is_ok() ||
        Path::new("/opt/intel/mkl").exists()
    }

    /// Check if Metal is available (macOS)
    fn has_metal(&self) -> bool {
        self.target_triple.contains("darwin") || self.target_triple.contains("macos")
    }

    /// Generate linker command line arguments
    pub fn generate_linker_args(&self, object_files: &[PathBuf], output_path: &Path) -> Vec<String> {
        let mut args = Vec::new();

        // Output file
        args.push("-o".to_string());
        args.push(output_path.to_string_lossy().to_string());

        // Object files
        for obj_file in object_files {
            args.push(obj_file.to_string_lossy().to_string());
        }

        // Library paths
        for lib_path in &self.library_paths {
            args.push(format!("-L{}", lib_path.display()));
        }

        // Static libraries
        for lib in &self.static_libraries {
            args.push(format!("-l{}", lib));
        }

        // Dynamic libraries
        for lib in &self.dynamic_libraries {
            args.push(format!("-l{}", lib));
        }

        // System libraries
        for lib in &self.system_libraries {
            args.push(format!("-l{}", lib));
        }

        // Framework paths (macOS)
        for framework_path in &self.framework_paths {
            args.push(format!("-F{}", framework_path.display()));
        }

        // Frameworks (macOS)
        for framework in &self.frameworks {
            args.push("-framework".to_string());
            args.push(framework.clone());
        }

        // LTO configuration
        if self.lto_config.enabled {
            match self.lto_config.lto_type {
                LTOType::Thin => args.push("-flto=thin".to_string()),
                LTOType::Full => args.push("-flto".to_string()),
                LTOType::None => {}
            }
        }

        // Debug configuration
        if self.debug_config.enabled {
            args.push(format!("-gdwarf-{}", self.debug_config.dwarf_version));
        }
        if self.debug_config.strip_symbols {
            args.push("-s".to_string());
        }

        // Custom linker flags
        args.extend(self.linker_flags.clone());

        args
    }

    /// Generate CMake configuration for runtime linking
    pub fn generate_cmake_config(&self) -> String {
        let mut cmake = String::new();
        
        cmake.push_str("# Aether Runtime Linking Configuration\n");
        cmake.push_str("# Generated automatically - do not edit\n\n");

        // Target configuration
        cmake.push_str(&format!("set(AETHER_TARGET_TRIPLE \"{}\")\n", self.target_triple));

        // Library paths
        if !self.library_paths.is_empty() {
            cmake.push_str("set(AETHER_LIBRARY_PATHS\n");
            for path in &self.library_paths {
                cmake.push_str(&format!("    \"{}\"\n", path.display()));
            }
            cmake.push_str(")\n");
            cmake.push_str("link_directories(${AETHER_LIBRARY_PATHS})\n\n");
        }

        // Static libraries
        if !self.static_libraries.is_empty() {
            cmake.push_str("set(AETHER_STATIC_LIBRARIES\n");
            for lib in &self.static_libraries {
                cmake.push_str(&format!("    {}\n", lib));
            }
            cmake.push_str(")\n\n");
        }

        // Dynamic libraries
        if !self.dynamic_libraries.is_empty() {
            cmake.push_str("set(AETHER_DYNAMIC_LIBRARIES\n");
            for lib in &self.dynamic_libraries {
                cmake.push_str(&format!("    {}\n", lib));
            }
            cmake.push_str(")\n\n");
        }

        // System libraries
        if !self.system_libraries.is_empty() {
            cmake.push_str("set(AETHER_SYSTEM_LIBRARIES\n");
            for lib in &self.system_libraries {
                cmake.push_str(&format!("    {}\n", lib));
            }
            cmake.push_str(")\n\n");
        }

        // Frameworks (macOS)
        if !self.frameworks.is_empty() {
            cmake.push_str("set(AETHER_FRAMEWORKS\n");
            for framework in &self.frameworks {
                cmake.push_str(&format!("    {}\n", framework));
            }
            cmake.push_str(")\n\n");
        }

        // Linker flags
        if !self.linker_flags.is_empty() {
            cmake.push_str("set(AETHER_LINKER_FLAGS\n");
            for flag in &self.linker_flags {
                cmake.push_str(&format!("    \"{}\"\n", flag));
            }
            cmake.push_str(")\n");
            cmake.push_str("string(JOIN \" \" AETHER_LINKER_FLAGS_STR ${AETHER_LINKER_FLAGS})\n");
            cmake.push_str("set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} ${AETHER_LINKER_FLAGS_STR}\")\n\n");
        }

        // Target link libraries function
        cmake.push_str("function(aether_target_link_libraries target)\n");
        cmake.push_str("    target_link_libraries(${target}\n");
        cmake.push_str("        ${AETHER_STATIC_LIBRARIES}\n");
        cmake.push_str("        ${AETHER_DYNAMIC_LIBRARIES}\n");
        cmake.push_str("        ${AETHER_SYSTEM_LIBRARIES}\n");
        cmake.push_str("    )\n");
        
        if !self.frameworks.is_empty() {
            cmake.push_str("    if(APPLE)\n");
            for framework in &self.frameworks {
                cmake.push_str(&format!("        target_link_libraries(${{target}} \"-framework {}\")\n", framework));
            }
            cmake.push_str("    endif()\n");
        }
        
        cmake.push_str("endfunction()\n\n");

        // LTO configuration
        if self.lto_config.enabled {
            cmake.push_str("# Link-time optimization\n");
            cmake.push_str("set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)\n");
            match self.lto_config.lto_type {
                LTOType::Thin => cmake.push_str("set(CMAKE_C_FLAGS \"${CMAKE_C_FLAGS} -flto=thin\")\n"),
                LTOType::Full => cmake.push_str("set(CMAKE_C_FLAGS \"${CMAKE_C_FLAGS} -flto\")\n"),
                LTOType::None => {}
            }
            cmake.push('\n');
        }

        // Debug configuration
        if self.debug_config.enabled {
            cmake.push_str("# Debug configuration\n");
            cmake.push_str(&format!("set(CMAKE_C_FLAGS_DEBUG \"${{CMAKE_C_FLAGS_DEBUG}} -gdwarf-{}\")\n", self.debug_config.dwarf_version));
            cmake.push_str(&format!("set(CMAKE_CXX_FLAGS_DEBUG \"${{CMAKE_CXX_FLAGS_DEBUG}} -gdwarf-{}\")\n", self.debug_config.dwarf_version));
        }

        cmake
    }

    /// Generate pkg-config file for runtime library
    pub fn generate_pkgconfig(&self, name: &str, version: &str, description: &str) -> String {
        let mut pc = String::new();
        
        pc.push_str(&format!("Name: {}\n", name));
        pc.push_str(&format!("Description: {}\n", description));
        pc.push_str(&format!("Version: {}\n", version));
        pc.push('\n');

        // Library paths
        if !self.library_paths.is_empty() {
            let lib_paths: Vec<String> = self.library_paths.iter()
                .map(|p| format!("-L{}", p.display()))
                .collect();
            pc.push_str(&format!("Libs: {}\n", lib_paths.join(" ")));
        }

        // Libraries
        let mut libs = Vec::new();
        libs.extend(self.static_libraries.iter().map(|l| format!("-l{}", l)));
        libs.extend(self.dynamic_libraries.iter().map(|l| format!("-l{}", l)));
        libs.extend(self.system_libraries.iter().map(|l| format!("-l{}", l)));
        
        if !libs.is_empty() {
            pc.push_str(&format!("Libs: {}\n", libs.join(" ")));
        }

        // Linker flags
        if !self.linker_flags.is_empty() {
            pc.push_str(&format!("Libs.private: {}\n", self.linker_flags.join(" ")));
        }

        pc
    }

    /// Add library path
    pub fn add_library_path<P: AsRef<Path>>(&mut self, path: P) {
        self.library_paths.push(path.as_ref().to_path_buf());
    }

    /// Add static library
    pub fn add_static_library(&mut self, name: String) {
        self.static_libraries.push(name);
    }

    /// Add dynamic library
    pub fn add_dynamic_library(&mut self, name: String) {
        self.dynamic_libraries.push(name);
    }

    /// Add system library
    pub fn add_system_library(&mut self, name: String) {
        self.system_libraries.push(name);
    }

    /// Add linker flag
    pub fn add_linker_flag(&mut self, flag: String) {
        self.linker_flags.push(flag);
    }

    /// Add framework (macOS)
    pub fn add_framework(&mut self, name: String) {
        self.frameworks.push(name);
    }
}

impl Default for LTOConfig {
    fn default() -> Self {
        LTOConfig {
            enabled: false,
            lto_type: LTOType::None,
            optimization_level: 2,
        }
    }
}

impl Default for DebugConfig {
    fn default() -> Self {
        DebugConfig {
            enabled: false,
            dwarf_version: 4,
            strip_symbols: false,
            separate_debug_info: false,
        }
    }
}

/// Runtime library manager
pub struct RuntimeLibraryManager {
    configs: HashMap<String, RuntimeLinkingConfig>,
}

impl RuntimeLibraryManager {
    /// Create new runtime library manager
    pub fn new() -> Self {
        RuntimeLibraryManager {
            configs: HashMap::new(),
        }
    }

    /// Add configuration for target
    pub fn add_target_config(&mut self, target: String, config: RuntimeLinkingConfig) {
        self.configs.insert(target, config);
    }

    /// Get configuration for target
    pub fn get_target_config(&self, target: &str) -> Option<&RuntimeLinkingConfig> {
        self.configs.get(target)
    }

    /// Get or create configuration for target
    pub fn get_or_create_config(&mut self, target: &str) -> &mut RuntimeLinkingConfig {
        self.configs.entry(target.to_string())
            .or_insert_with(|| RuntimeLinkingConfig::new(target.to_string()))
    }

    /// Generate all configurations
    pub fn generate_all_configs(&self) -> HashMap<String, String> {
        let mut configs = HashMap::new();
        
        for (target, config) in &self.configs {
            configs.insert(
                format!("{}_cmake", target),
                config.generate_cmake_config()
            );
            configs.insert(
                format!("{}_pkgconfig", target),
                config.generate_pkgconfig("aether-runtime", "1.0.0", "Aether Runtime Library")
            );
        }
        
        configs
    }
}

impl Default for RuntimeLibraryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_linking_config_creation() {
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        assert_eq!(config.target_triple, "x86_64-unknown-linux-gnu");
        assert!(config.static_libraries.contains(&"aether_runtime".to_string()));
    }

    #[test]
    fn test_generate_linker_args() {
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        let object_files = vec![PathBuf::from("test.o")];
        let output_path = Path::new("test_output");
        
        let args = config.generate_linker_args(&object_files, output_path);
        assert!(args.contains(&"-o".to_string()));
        assert!(args.contains(&"test_output".to_string()));
        assert!(args.contains(&"test.o".to_string()));
    }

    #[test]
    fn test_generate_cmake_config() {
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        let cmake = config.generate_cmake_config();
        assert!(cmake.contains("AETHER_TARGET_TRIPLE"));
        assert!(cmake.contains("aether_target_link_libraries"));
    }

    #[test]
    fn test_runtime_library_manager() {
        let mut manager = RuntimeLibraryManager::new();
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        
        manager.add_target_config("linux-x64".to_string(), config);
        assert!(manager.get_target_config("linux-x64").is_some());
    }
}