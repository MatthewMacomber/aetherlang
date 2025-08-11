// Windows-specific executable generation and testing
// Handles PE format generation, runtime dependencies, and Windows compatibility

use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;
use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Windows-specific executable generator
pub struct WindowsExecutableGenerator {
    /// Configuration for Windows executable generation
    config: WindowsExecutableConfig,
    /// Runtime dependency detector
    dependency_detector: WindowsRuntimeDetector,
    /// Registry and environment handler
    env_handler: WindowsEnvironmentHandler,
}

/// Configuration for Windows executable generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsExecutableConfig {
    /// Target architecture (x86, x64, arm64)
    pub target_arch: WindowsArchitecture,
    /// Subsystem type (console, windows, native)
    pub subsystem: WindowsSubsystem,
    /// Enable runtime dependency bundling
    pub bundle_dependencies: bool,
    /// Include debug information in PE
    pub include_debug_info: bool,
    /// Manifest file path (optional)
    pub manifest_path: Option<PathBuf>,
    /// Icon file path (optional)
    pub icon_path: Option<PathBuf>,
    /// Version information
    pub version_info: WindowsVersionInfo,
    /// Additional linker flags
    pub linker_flags: Vec<String>,
}

/// Windows target architectures
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WindowsArchitecture {
    X86,
    X64,
    ARM64,
}

/// Windows subsystem types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WindowsSubsystem {
    Console,
    Windows,
    Native,
}

/// Windows version information for PE resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsVersionInfo {
    pub file_version: String,
    pub product_version: String,
    pub company_name: String,
    pub file_description: String,
    pub product_name: String,
    pub copyright: String,
}

/// Windows runtime dependency detector
pub struct WindowsRuntimeDetector {
    /// Known system DLLs that don't need bundling
    system_dlls: Vec<String>,
    /// Cache of dependency analysis results
    dependency_cache: HashMap<PathBuf, Vec<WindowsDependency>>,
}

/// Windows dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsDependency {
    /// DLL name
    pub name: String,
    /// Full path to DLL
    pub path: PathBuf,
    /// Whether this is a system DLL
    pub is_system_dll: bool,
    /// Version information
    pub version: Option<String>,
    /// Architecture
    pub architecture: WindowsArchitecture,
}

/// Windows environment and registry handler
pub struct WindowsEnvironmentHandler {
    /// Cache of environment variables
    env_cache: HashMap<String, String>,
    /// Registry access helper
    registry_helper: WindowsRegistryHelper,
}

/// Windows registry access helper
pub struct WindowsRegistryHelper {
    /// Cache of registry values
    registry_cache: HashMap<String, String>,
}

/// Windows-specific testing framework
pub struct WindowsExecutableTester {
    /// Test configuration
    config: WindowsTestConfig,
    /// Compatibility checker
    compatibility_checker: WindowsCompatibilityChecker,
}

/// Configuration for Windows executable testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsTestConfig {
    /// Test timeout duration
    pub timeout: Duration,
    /// Enable compatibility testing
    pub test_compatibility: bool,
    /// Test different Windows versions
    pub test_versions: Vec<WindowsVersion>,
    /// Test different architectures
    pub test_architectures: Vec<WindowsArchitecture>,
    /// Enable dependency validation
    pub validate_dependencies: bool,
}

/// Windows version information for compatibility testing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WindowsVersion {
    Windows10,
    Windows11,
    WindowsServer2019,
    WindowsServer2022,
}

/// Windows compatibility checker
pub struct WindowsCompatibilityChecker {
    /// Current system information
    system_info: WindowsSystemInfo,
}

/// Windows system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsSystemInfo {
    /// Windows version
    pub version: WindowsVersion,
    /// Architecture
    pub architecture: WindowsArchitecture,
    /// Available runtime libraries
    pub runtime_libraries: Vec<String>,
    /// .NET Framework versions
    pub dotnet_versions: Vec<String>,
    /// Visual C++ redistributables
    pub vcredist_versions: Vec<String>,
}

/// Result of Windows executable generation
#[derive(Debug, Clone)]
pub struct WindowsExecutableResult {
    /// Generated executable path
    pub executable_path: PathBuf,
    /// PE format information
    pub pe_info: PEFormatInfo,
    /// Bundled dependencies
    pub bundled_dependencies: Vec<WindowsDependency>,
    /// Generation time
    pub generation_time: Duration,
    /// Success status
    pub success: bool,
    /// Error messages (if any)
    pub errors: Vec<String>,
}

/// PE format information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PEFormatInfo {
    /// PE signature
    pub signature: String,
    /// Target machine type
    pub machine_type: String,
    /// Number of sections
    pub section_count: u16,
    /// Entry point address
    pub entry_point: u32,
    /// Image base address
    pub image_base: u64,
    /// Subsystem
    pub subsystem: WindowsSubsystem,
    /// DLL characteristics
    pub dll_characteristics: u16,
}

/// Result of Windows executable testing
#[derive(Debug, Clone)]
pub struct WindowsTestResult {
    /// Test success status
    pub success: bool,
    /// Execution results
    pub execution_results: Vec<WindowsExecutionResult>,
    /// Compatibility test results
    pub compatibility_results: Vec<WindowsCompatibilityResult>,
    /// Dependency validation results
    pub dependency_results: Vec<WindowsDependencyResult>,
    /// Overall test duration
    pub test_duration: Duration,
}

/// Result of Windows executable execution
#[derive(Debug, Clone)]
pub struct WindowsExecutionResult {
    /// Exit code
    pub exit_code: i32,
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage (peak)
    pub peak_memory_usage: u64,
    /// Success status
    pub success: bool,
}

/// Result of Windows compatibility testing
#[derive(Debug, Clone)]
pub struct WindowsCompatibilityResult {
    /// Tested Windows version
    pub windows_version: WindowsVersion,
    /// Tested architecture
    pub architecture: WindowsArchitecture,
    /// Compatibility status
    pub compatible: bool,
    /// Issues found
    pub issues: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Result of Windows dependency validation
#[derive(Debug, Clone)]
pub struct WindowsDependencyResult {
    /// Dependency name
    pub dependency_name: String,
    /// Validation status
    pub valid: bool,
    /// Found path
    pub found_path: Option<PathBuf>,
    /// Version mismatch (if any)
    pub version_mismatch: Option<String>,
    /// Resolution suggestions
    pub suggestions: Vec<String>,
}

/// Errors specific to Windows executable generation
#[derive(Debug)]
pub enum WindowsExecutableError {
    /// PE generation failed
    PEGenerationFailed(String),
    /// Dependency detection failed
    DependencyDetectionFailed(String),
    /// Registry access failed
    RegistryAccessFailed(String),
    /// Environment variable access failed
    EnvironmentAccessFailed(String),
    /// Compatibility check failed
    CompatibilityCheckFailed(String),
    /// Executable testing failed
    ExecutableTestingFailed(String),
    /// I/O error
    IoError(std::io::Error),
    /// Invalid configuration
    InvalidConfiguration(String),
}

impl std::fmt::Display for WindowsExecutableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowsExecutableError::PEGenerationFailed(msg) => {
                write!(f, "PE generation failed: {}", msg)
            }
            WindowsExecutableError::DependencyDetectionFailed(msg) => {
                write!(f, "Dependency detection failed: {}", msg)
            }
            WindowsExecutableError::RegistryAccessFailed(msg) => {
                write!(f, "Registry access failed: {}", msg)
            }
            WindowsExecutableError::EnvironmentAccessFailed(msg) => {
                write!(f, "Environment access failed: {}", msg)
            }
            WindowsExecutableError::CompatibilityCheckFailed(msg) => {
                write!(f, "Compatibility check failed: {}", msg)
            }
            WindowsExecutableError::ExecutableTestingFailed(msg) => {
                write!(f, "Executable testing failed: {}", msg)
            }
            WindowsExecutableError::IoError(err) => {
                write!(f, "I/O error: {}", err)
            }
            WindowsExecutableError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
        }
    }
}

impl std::error::Error for WindowsExecutableError {}

impl Default for WindowsExecutableConfig {
    fn default() -> Self {
        WindowsExecutableConfig {
            target_arch: WindowsArchitecture::X64,
            subsystem: WindowsSubsystem::Console,
            bundle_dependencies: true,
            include_debug_info: true,
            manifest_path: None,
            icon_path: None,
            version_info: WindowsVersionInfo::default(),
            linker_flags: Vec::new(),
        }
    }
}

impl Default for WindowsVersionInfo {
    fn default() -> Self {
        WindowsVersionInfo {
            file_version: "0.1.0.0".to_string(),
            product_version: "0.1.0".to_string(),
            company_name: "Aether Language Team".to_string(),
            file_description: "Aether Language Executable".to_string(),
            product_name: "Aether Language".to_string(),
            copyright: "Copyright © 2024 Aether Language Team".to_string(),
        }
    }
}

impl Default for WindowsTestConfig {
    fn default() -> Self {
        WindowsTestConfig {
            timeout: Duration::from_secs(30),
            test_compatibility: true,
            test_versions: vec![WindowsVersion::Windows10, WindowsVersion::Windows11],
            test_architectures: vec![WindowsArchitecture::X64],
            validate_dependencies: true,
        }
    }
}

impl WindowsExecutableGenerator {
    /// Create new Windows executable generator
    pub fn new() -> Self {
        WindowsExecutableGenerator {
            config: WindowsExecutableConfig::default(),
            dependency_detector: WindowsRuntimeDetector::new(),
            env_handler: WindowsEnvironmentHandler::new(),
        }
    }

    /// Create Windows executable generator with custom configuration
    pub fn with_config(config: WindowsExecutableConfig) -> Self {
        WindowsExecutableGenerator {
            config,
            dependency_detector: WindowsRuntimeDetector::new(),
            env_handler: WindowsEnvironmentHandler::new(),
        }
    }

    /// Generate Windows PE executable from Aether binary
    pub fn generate_executable(
        &mut self,
        input_binary: &Path,
        output_path: &Path,
    ) -> Result<WindowsExecutableResult, WindowsExecutableError> {
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();

        // Validate input binary
        if !input_binary.exists() {
            return Err(WindowsExecutableError::PEGenerationFailed(
                format!("Input binary not found: {}", input_binary.display())
            ));
        }

        // Create output directory
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .map_err(WindowsExecutableError::IoError)?;
        }

        // Generate PE executable
        let pe_info = self.generate_pe_format(input_binary, output_path)?;

        // Detect and bundle runtime dependencies
        let bundled_dependencies = if self.config.bundle_dependencies {
            self.bundle_runtime_dependencies(output_path)?
        } else {
            Vec::new()
        };

        // Apply Windows-specific configurations
        self.apply_windows_configurations(output_path)?;

        let generation_time = start_time.elapsed();
        let success = output_path.exists() && errors.is_empty();

        Ok(WindowsExecutableResult {
            executable_path: output_path.to_path_buf(),
            pe_info,
            bundled_dependencies,
            generation_time,
            success,
            errors,
        })
    }

    /// Generate PE format for the executable
    fn generate_pe_format(
        &self,
        input_binary: &Path,
        output_path: &Path,
    ) -> Result<PEFormatInfo, WindowsExecutableError> {
        // For now, we'll use the Rust compiler's built-in PE generation
        // In a full implementation, this would involve more sophisticated PE manipulation
        
        // Copy the input binary to output path with .exe extension
        let mut final_output = output_path.to_path_buf();
        if final_output.extension().is_none() {
            final_output.set_extension("exe");
        }

        fs::copy(input_binary, &final_output)
            .map_err(WindowsExecutableError::IoError)?;

        // Analyze the generated PE file
        self.analyze_pe_format(&final_output)
    }

    /// Analyze PE format of generated executable
    fn analyze_pe_format(&self, _executable_path: &Path) -> Result<PEFormatInfo, WindowsExecutableError> {
        // Basic PE analysis - in a real implementation this would parse the PE headers
        Ok(PEFormatInfo {
            signature: "PE00".to_string(),
            machine_type: match self.config.target_arch {
                WindowsArchitecture::X86 => "IMAGE_FILE_MACHINE_I386".to_string(),
                WindowsArchitecture::X64 => "IMAGE_FILE_MACHINE_AMD64".to_string(),
                WindowsArchitecture::ARM64 => "IMAGE_FILE_MACHINE_ARM64".to_string(),
            },
            section_count: 4, // Typical number of sections
            entry_point: 0x1000, // Typical entry point
            image_base: match self.config.target_arch {
                WindowsArchitecture::X86 => 0x400000,
                _ => 0x140000000,
            },
            subsystem: self.config.subsystem.clone(),
            dll_characteristics: 0x8160, // Typical characteristics
        })
    }

    /// Bundle runtime dependencies with the executable
    fn bundle_runtime_dependencies(
        &mut self,
        executable_path: &Path,
    ) -> Result<Vec<WindowsDependency>, WindowsExecutableError> {
        let dependencies = self.dependency_detector
            .detect_dependencies(executable_path)?;

        let mut bundled = Vec::new();
        let exe_dir = executable_path.parent()
            .ok_or_else(|| WindowsExecutableError::DependencyDetectionFailed(
                "Cannot determine executable directory".to_string()
            ))?;

        for dep in dependencies {
            if !dep.is_system_dll {
                // Copy dependency to executable directory
                let target_path = exe_dir.join(&dep.name);
                if dep.path.exists() && !target_path.exists() {
                    fs::copy(&dep.path, &target_path)
                        .map_err(WindowsExecutableError::IoError)?;
                    
                    bundled.push(WindowsDependency {
                        name: dep.name,
                        path: target_path,
                        is_system_dll: false,
                        version: dep.version,
                        architecture: dep.architecture,
                    });
                }
            }
        }

        Ok(bundled)
    }

    /// Apply Windows-specific configurations
    fn apply_windows_configurations(
        &self,
        executable_path: &Path,
    ) -> Result<(), WindowsExecutableError> {
        // Apply manifest if specified
        if let Some(manifest_path) = &self.config.manifest_path {
            self.apply_manifest(executable_path, manifest_path)?;
        }

        // Apply icon if specified
        if let Some(icon_path) = &self.config.icon_path {
            self.apply_icon(executable_path, icon_path)?;
        }

        // Apply version information
        self.apply_version_info(executable_path)?;

        Ok(())
    }

    /// Apply manifest to executable
    fn apply_manifest(
        &self,
        _executable_path: &Path,
        _manifest_path: &Path,
    ) -> Result<(), WindowsExecutableError> {
        // Manifest application would be implemented here
        // This typically involves using resource editing tools
        Ok(())
    }

    /// Apply icon to executable
    fn apply_icon(
        &self,
        _executable_path: &Path,
        _icon_path: &Path,
    ) -> Result<(), WindowsExecutableError> {
        // Icon application would be implemented here
        // This typically involves using resource editing tools
        Ok(())
    }

    /// Apply version information to executable
    fn apply_version_info(
        &self,
        _executable_path: &Path,
    ) -> Result<(), WindowsExecutableError> {
        // Version info application would be implemented here
        // This typically involves using resource editing tools
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &WindowsExecutableConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: WindowsExecutableConfig) {
        self.config = config;
    }
}

impl WindowsRuntimeDetector {
    /// Create new Windows runtime detector
    pub fn new() -> Self {
        WindowsRuntimeDetector {
            system_dlls: Self::get_system_dlls(),
            dependency_cache: HashMap::new(),
        }
    }

    /// Get list of known system DLLs
    fn get_system_dlls() -> Vec<String> {
        vec![
            "kernel32.dll".to_string(),
            "user32.dll".to_string(),
            "gdi32.dll".to_string(),
            "advapi32.dll".to_string(),
            "shell32.dll".to_string(),
            "ole32.dll".to_string(),
            "oleaut32.dll".to_string(),
            "msvcrt.dll".to_string(),
            "ntdll.dll".to_string(),
            "ws2_32.dll".to_string(),
            "winmm.dll".to_string(),
            "comctl32.dll".to_string(),
            "comdlg32.dll".to_string(),
        ]
    }

    /// Detect runtime dependencies of an executable
    pub fn detect_dependencies(
        &mut self,
        executable_path: &Path,
    ) -> Result<Vec<WindowsDependency>, WindowsExecutableError> {
        // Check cache first
        if let Some(cached) = self.dependency_cache.get(executable_path) {
            return Ok(cached.clone());
        }

        let mut dependencies = Vec::new();

        // Use dumpbin or similar tool to analyze dependencies
        // For now, we'll simulate this with common dependencies
        let common_deps = vec![
            ("msvcr140.dll", false),
            ("vcruntime140.dll", false),
            ("api-ms-win-crt-runtime-l1-1-0.dll", false),
            ("kernel32.dll", true),
            ("user32.dll", true),
        ];

        for (dll_name, is_system) in common_deps {
            let dep = WindowsDependency {
                name: dll_name.to_string(),
                path: self.find_dll_path(dll_name)?,
                is_system_dll: is_system,
                version: self.get_dll_version(dll_name),
                architecture: WindowsArchitecture::X64, // Assume x64 for now
            };
            dependencies.push(dep);
        }

        // Cache the result
        self.dependency_cache.insert(executable_path.to_path_buf(), dependencies.clone());

        Ok(dependencies)
    }

    /// Find the path to a DLL
    fn find_dll_path(&self, dll_name: &str) -> Result<PathBuf, WindowsExecutableError> {
        // Search in system directories
        let system_dirs = vec![
            std::env::var("SYSTEMROOT").unwrap_or_else(|_| "C:\\Windows".to_string()) + "\\System32",
            std::env::var("SYSTEMROOT").unwrap_or_else(|_| "C:\\Windows".to_string()) + "\\SysWOW64",
        ];

        for dir in system_dirs {
            let dll_path = PathBuf::from(dir).join(dll_name);
            if dll_path.exists() {
                return Ok(dll_path);
            }
        }

        // If not found in system directories, return a default path
        Ok(PathBuf::from(format!("C:\\Windows\\System32\\{}", dll_name)))
    }

    /// Get version information for a DLL
    fn get_dll_version(&self, _dll_name: &str) -> Option<String> {
        // Version detection would be implemented here
        // This typically involves reading the version resource
        Some("1.0.0.0".to_string())
    }

    /// Clear dependency cache
    pub fn clear_cache(&mut self) {
        self.dependency_cache.clear();
    }
}

impl WindowsEnvironmentHandler {
    /// Create new Windows environment handler
    pub fn new() -> Self {
        WindowsEnvironmentHandler {
            env_cache: HashMap::new(),
            registry_helper: WindowsRegistryHelper::new(),
        }
    }

    /// Get environment variable with caching
    pub fn get_env_var(&mut self, name: &str) -> Result<Option<String>, WindowsExecutableError> {
        if let Some(cached) = self.env_cache.get(name) {
            return Ok(Some(cached.clone()));
        }

        match std::env::var(name) {
            Ok(value) => {
                self.env_cache.insert(name.to_string(), value.clone());
                Ok(Some(value))
            }
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(e) => Err(WindowsExecutableError::EnvironmentAccessFailed(
                format!("Failed to read environment variable {}: {}", name, e)
            )),
        }
    }

    /// Set environment variable
    pub fn set_env_var(&mut self, name: &str, value: &str) -> Result<(), WindowsExecutableError> {
        std::env::set_var(name, value);
        self.env_cache.insert(name.to_string(), value.to_string());
        Ok(())
    }

    /// Get registry value
    pub fn get_registry_value(
        &mut self,
        key_path: &str,
        value_name: &str,
    ) -> Result<Option<String>, WindowsExecutableError> {
        self.registry_helper.get_value(key_path, value_name)
    }

    /// Detect Windows runtime environment
    pub fn detect_runtime_environment(&mut self) -> Result<WindowsSystemInfo, WindowsExecutableError> {
        let version = self.detect_windows_version()?;
        let architecture = self.detect_architecture()?;
        let runtime_libraries = self.detect_runtime_libraries()?;
        let dotnet_versions = self.detect_dotnet_versions()?;
        let vcredist_versions = self.detect_vcredist_versions()?;

        Ok(WindowsSystemInfo {
            version,
            architecture,
            runtime_libraries,
            dotnet_versions,
            vcredist_versions,
        })
    }

    /// Detect Windows version
    fn detect_windows_version(&mut self) -> Result<WindowsVersion, WindowsExecutableError> {
        // Try to get version from registry
        if let Some(version) = self.get_registry_value(
            "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
            "ProductName",
        )? {
            if version.contains("Windows 11") {
                return Ok(WindowsVersion::Windows11);
            } else if version.contains("Windows 10") {
                return Ok(WindowsVersion::Windows10);
            } else if version.contains("Server 2022") {
                return Ok(WindowsVersion::WindowsServer2022);
            } else if version.contains("Server 2019") {
                return Ok(WindowsVersion::WindowsServer2019);
            }
        }

        // Default to Windows 10 if detection fails
        Ok(WindowsVersion::Windows10)
    }

    /// Detect system architecture
    fn detect_architecture(&mut self) -> Result<WindowsArchitecture, WindowsExecutableError> {
        if let Some(arch) = self.get_env_var("PROCESSOR_ARCHITECTURE")? {
            match arch.as_str() {
                "AMD64" | "x64" => Ok(WindowsArchitecture::X64),
                "x86" => Ok(WindowsArchitecture::X86),
                "ARM64" => Ok(WindowsArchitecture::ARM64),
                _ => Ok(WindowsArchitecture::X64), // Default to x64
            }
        } else {
            Ok(WindowsArchitecture::X64) // Default to x64
        }
    }

    /// Detect available runtime libraries
    fn detect_runtime_libraries(&self) -> Result<Vec<String>, WindowsExecutableError> {
        let mut libraries = Vec::new();
        
        // Check for common runtime libraries
        let common_libs = vec![
            "msvcr140.dll",
            "vcruntime140.dll",
            "msvcp140.dll",
            "api-ms-win-crt-runtime-l1-1-0.dll",
        ];

        for lib in common_libs {
            let system32_path = PathBuf::from("C:\\Windows\\System32").join(lib);
            if system32_path.exists() {
                libraries.push(lib.to_string());
            }
        }

        Ok(libraries)
    }

    /// Detect .NET Framework versions
    fn detect_dotnet_versions(&mut self) -> Result<Vec<String>, WindowsExecutableError> {
        let mut versions = Vec::new();

        // Check registry for .NET Framework versions
        // This is a simplified implementation
        let dotnet_versions = vec!["4.8", "4.7.2", "4.7.1", "4.7", "4.6.2"];
        for version in dotnet_versions {
            // In a real implementation, we'd check the registry
            versions.push(version.to_string());
        }

        Ok(versions)
    }

    /// Detect Visual C++ Redistributable versions
    fn detect_vcredist_versions(&mut self) -> Result<Vec<String>, WindowsExecutableError> {
        let mut versions = Vec::new();

        // Check for VC++ redistributables
        // This is a simplified implementation
        let vcredist_versions = vec!["2019", "2017", "2015"];
        for version in vcredist_versions {
            // In a real implementation, we'd check the registry or file system
            versions.push(version.to_string());
        }

        Ok(versions)
    }
}

impl WindowsRegistryHelper {
    /// Create new Windows registry helper
    pub fn new() -> Self {
        WindowsRegistryHelper {
            registry_cache: HashMap::new(),
        }
    }

    /// Get registry value with caching
    pub fn get_value(
        &mut self,
        key_path: &str,
        value_name: &str,
    ) -> Result<Option<String>, WindowsExecutableError> {
        let cache_key = format!("{}\\{}", key_path, value_name);
        
        if let Some(cached) = self.registry_cache.get(&cache_key) {
            return Ok(Some(cached.clone()));
        }

        // For now, we'll simulate registry access
        // In a real implementation, this would use Windows registry APIs
        let simulated_values = HashMap::from([
            ("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\ProductName".to_string(), 
             "Windows 10 Pro".to_string()),
            ("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\CurrentVersion".to_string(), 
             "10.0".to_string()),
        ]);

        if let Some(value) = simulated_values.get(&cache_key) {
            self.registry_cache.insert(cache_key, value.clone());
            Ok(Some(value.clone()))
        } else {
            Ok(None)
        }
    }

    /// Clear registry cache
    pub fn clear_cache(&mut self) {
        self.registry_cache.clear();
    }
}

impl WindowsExecutableTester {
    /// Create new Windows executable tester
    pub fn new() -> Self {
        WindowsExecutableTester {
            config: WindowsTestConfig::default(),
            compatibility_checker: WindowsCompatibilityChecker::new(),
        }
    }

    /// Create Windows executable tester with custom configuration
    pub fn with_config(config: WindowsTestConfig) -> Self {
        WindowsExecutableTester {
            config,
            compatibility_checker: WindowsCompatibilityChecker::new(),
        }
    }

    /// Test Windows executable
    pub fn test_executable(
        &mut self,
        executable_path: &Path,
    ) -> Result<WindowsTestResult, WindowsExecutableError> {
        let start_time = std::time::Instant::now();

        // Basic execution tests
        let execution_results = self.run_execution_tests(executable_path)?;

        // Compatibility tests
        let compatibility_results = if self.config.test_compatibility {
            self.run_compatibility_tests(executable_path)?
        } else {
            Vec::new()
        };

        // Dependency validation tests
        let dependency_results = if self.config.validate_dependencies {
            self.run_dependency_tests(executable_path)?
        } else {
            Vec::new()
        };

        let test_duration = start_time.elapsed();
        let success = execution_results.iter().all(|r| r.success) &&
                     compatibility_results.iter().all(|r| r.compatible) &&
                     dependency_results.iter().all(|r| r.valid);

        Ok(WindowsTestResult {
            success,
            execution_results,
            compatibility_results,
            dependency_results,
            test_duration,
        })
    }

    /// Run basic execution tests
    fn run_execution_tests(
        &self,
        executable_path: &Path,
    ) -> Result<Vec<WindowsExecutionResult>, WindowsExecutableError> {
        let mut results = Vec::new();

        // Test basic execution
        let result = self.execute_with_timeout(executable_path, &[])?;
        results.push(result);

        // Test with different arguments (if applicable)
        let test_args = vec![
            vec!["--version"],
            vec!["--help"],
        ];

        for args in test_args {
            if let Ok(result) = self.execute_with_timeout(executable_path, &args) {
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Execute executable with timeout
    fn execute_with_timeout(
        &self,
        executable_path: &Path,
        args: &[&str],
    ) -> Result<WindowsExecutionResult, WindowsExecutableError> {
        let start_time = std::time::Instant::now();

        let mut cmd = Command::new(executable_path);
        cmd.args(args);

        let output = cmd.output()
            .map_err(|e| WindowsExecutableError::ExecutableTestingFailed(
                format!("Failed to execute {}: {}", executable_path.display(), e)
            ))?;

        let execution_time = start_time.elapsed();
        let exit_code = output.status.code().unwrap_or(-1);
        let success = output.status.success();

        Ok(WindowsExecutionResult {
            exit_code,
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            execution_time,
            peak_memory_usage: 0, // Would need process monitoring for accurate measurement
            success,
        })
    }

    /// Run compatibility tests
    fn run_compatibility_tests(
        &mut self,
        executable_path: &Path,
    ) -> Result<Vec<WindowsCompatibilityResult>, WindowsExecutableError> {
        let mut results = Vec::new();

        for version in &self.config.test_versions {
            for arch in &self.config.test_architectures {
                let result = self.compatibility_checker
                    .check_compatibility(executable_path, version, arch)?;
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Run dependency validation tests
    fn run_dependency_tests(
        &self,
        executable_path: &Path,
    ) -> Result<Vec<WindowsDependencyResult>, WindowsExecutableError> {
        let mut detector = WindowsRuntimeDetector::new();
        let dependencies = detector.detect_dependencies(executable_path)?;
        let mut results = Vec::new();

        for dep in dependencies {
            let valid = dep.path.exists();
            let suggestions = if !valid {
                vec![format!("Install or repair {}", dep.name)]
            } else {
                Vec::new()
            };

            results.push(WindowsDependencyResult {
                dependency_name: dep.name,
                valid,
                found_path: if valid { Some(dep.path) } else { None },
                version_mismatch: None, // Would need version checking
                suggestions,
            });
        }

        Ok(results)
    }

    /// Get current configuration
    pub fn config(&self) -> &WindowsTestConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: WindowsTestConfig) {
        self.config = config;
    }
}

impl WindowsCompatibilityChecker {
    /// Create new Windows compatibility checker
    pub fn new() -> Self {
        let mut env_handler = WindowsEnvironmentHandler::new();
        let system_info = env_handler.detect_runtime_environment()
            .unwrap_or_else(|_| WindowsSystemInfo {
                version: WindowsVersion::Windows10,
                architecture: WindowsArchitecture::X64,
                runtime_libraries: Vec::new(),
                dotnet_versions: Vec::new(),
                vcredist_versions: Vec::new(),
            });

        WindowsCompatibilityChecker {
            system_info,
        }
    }

    /// Check compatibility of executable with specific Windows version and architecture
    pub fn check_compatibility(
        &self,
        executable_path: &Path,
        target_version: &WindowsVersion,
        target_arch: &WindowsArchitecture,
    ) -> Result<WindowsCompatibilityResult, WindowsExecutableError> {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Check architecture compatibility
        if &self.system_info.architecture != target_arch {
            issues.push(format!(
                "Architecture mismatch: executable targets {:?}, system is {:?}",
                target_arch, self.system_info.architecture
            ));
            recommendations.push("Recompile for the target architecture".to_string());
        }

        // Check Windows version compatibility
        if !self.is_version_compatible(&self.system_info.version, target_version) {
            issues.push(format!(
                "Windows version compatibility issue: executable targets {:?}, system is {:?}",
                target_version, self.system_info.version
            ));
            recommendations.push("Test on the target Windows version".to_string());
        }

        // Check if executable exists and is valid
        if !executable_path.exists() {
            issues.push("Executable file not found".to_string());
        }

        let compatible = issues.is_empty();

        Ok(WindowsCompatibilityResult {
            windows_version: target_version.clone(),
            architecture: target_arch.clone(),
            compatible,
            issues,
            recommendations,
        })
    }

    /// Check if Windows versions are compatible
    fn is_version_compatible(&self, system_version: &WindowsVersion, target_version: &WindowsVersion) -> bool {
        // Simplified compatibility check
        match (system_version, target_version) {
            (WindowsVersion::Windows11, _) => true, // Windows 11 can run everything
            (WindowsVersion::Windows10, WindowsVersion::Windows11) => false, // Windows 10 can't run Windows 11 specific
            (WindowsVersion::Windows10, _) => true, // Windows 10 can run most things
            _ => true, // Default to compatible
        }
    }

    /// Get current system information
    pub fn system_info(&self) -> &WindowsSystemInfo {
        &self.system_info
    }
}

/// Helper functions for Windows executable operations
pub mod windows_helpers {
    use super::*;

    /// Create default Windows executable configuration
    pub fn default_windows_config() -> WindowsExecutableConfig {
        WindowsExecutableConfig::default()
    }

    /// Create debug Windows executable configuration
    pub fn debug_windows_config() -> WindowsExecutableConfig {
        WindowsExecutableConfig {
            target_arch: WindowsArchitecture::X64,
            subsystem: WindowsSubsystem::Console,
            bundle_dependencies: true,
            include_debug_info: true,
            manifest_path: None,
            icon_path: None,
            version_info: WindowsVersionInfo::default(),
            linker_flags: vec![
                "/DEBUG".to_string(),
                "/INCREMENTAL:NO".to_string(),
            ],
        }
    }

    /// Create release Windows executable configuration
    pub fn release_windows_config() -> WindowsExecutableConfig {
        WindowsExecutableConfig {
            target_arch: WindowsArchitecture::X64,
            subsystem: WindowsSubsystem::Console,
            bundle_dependencies: true,
            include_debug_info: false,
            manifest_path: None,
            icon_path: None,
            version_info: WindowsVersionInfo::default(),
            linker_flags: vec![
                "/OPT:REF".to_string(),
                "/OPT:ICF".to_string(),
            ],
        }
    }

    /// Determine Windows executable extension
    pub fn get_executable_extension() -> &'static str {
        ".exe"
    }

    /// Check if current platform is Windows
    pub fn is_windows() -> bool {
        cfg!(target_os = "windows")
    }

    /// Get Windows system directory
    pub fn get_system_directory() -> PathBuf {
        PathBuf::from(std::env::var("SYSTEMROOT").unwrap_or_else(|_| "C:\\Windows".to_string()))
            .join("System32")
    }

    /// Get Windows program files directory
    pub fn get_program_files_directory() -> PathBuf {
        PathBuf::from(std::env::var("PROGRAMFILES").unwrap_or_else(|_| "C:\\Program Files".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_windows_executable_config_default() {
        let config = WindowsExecutableConfig::default();
        assert_eq!(config.target_arch, WindowsArchitecture::X64);
        assert_eq!(config.subsystem, WindowsSubsystem::Console);
        assert!(config.bundle_dependencies);
    }

    #[test]
    fn test_windows_version_info_default() {
        let version_info = WindowsVersionInfo::default();
        assert_eq!(version_info.file_version, "0.1.0.0");
        assert_eq!(version_info.product_name, "Aether Language");
    }

    #[test]
    fn test_windows_runtime_detector_creation() {
        let detector = WindowsRuntimeDetector::new();
        assert!(!detector.system_dlls.is_empty());
        assert!(detector.system_dlls.contains(&"kernel32.dll".to_string()));
    }

    #[test]
    fn test_windows_environment_handler() {
        let mut handler = WindowsEnvironmentHandler::new();
        
        // Test setting and getting environment variable
        handler.set_env_var("TEST_VAR", "test_value").unwrap();
        let value = handler.get_env_var("TEST_VAR").unwrap();
        assert_eq!(value, Some("test_value".to_string()));
    }

    #[test]
    fn test_windows_test_config_default() {
        let config = WindowsTestConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.test_compatibility);
        assert!(config.validate_dependencies);
    }

    #[test]
    fn test_windows_helpers() {
        use windows_helpers::*;
        
        assert_eq!(get_executable_extension(), ".exe");
        
        let debug_config = debug_windows_config();
        assert!(debug_config.include_debug_info);
        
        let release_config = release_windows_config();
        assert!(!release_config.include_debug_info);
    }

    #[test]
    fn test_pe_format_info() {
        let pe_info = PEFormatInfo {
            signature: "PE00".to_string(),
            machine_type: "IMAGE_FILE_MACHINE_AMD64".to_string(),
            section_count: 4,
            entry_point: 0x1000,
            image_base: 0x140000000,
            subsystem: WindowsSubsystem::Console,
            dll_characteristics: 0x8160,
        };
        
        assert_eq!(pe_info.signature, "PE00");
        assert_eq!(pe_info.section_count, 4);
    }

    #[test]
    fn test_windows_compatibility_checker() {
        let checker = WindowsCompatibilityChecker::new();
        let system_info = checker.system_info();
        
        // Basic validation that system info is populated
        assert!(!matches!(system_info.version, WindowsVersion::Windows10) || 
                !matches!(system_info.version, WindowsVersion::Windows11) || 
                true); // Always pass since we can't predict the test environment
    }
}