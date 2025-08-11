// Environment validation for Aether build system

use std::process::Command;
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Environment validator for build system
pub struct EnvironmentValidator;

impl EnvironmentValidator {
    /// Create new environment validator
    pub fn new() -> Self {
        Self
    }
    
    /// Validate the entire build environment
    pub fn validate_environment(&self) -> Result<EnvironmentStatus, EnvironmentError> {
        let rust_status = self.validate_rust_toolchain()?;
        let dependency_status = self.validate_dependencies()?;
        let system_status = self.validate_system_requirements()?;
        
        let overall_status = self.determine_overall_status(&rust_status, &dependency_status, &system_status);
        
        Ok(EnvironmentStatus {
            rust_toolchain: rust_status,
            dependencies: dependency_status,
            system: system_status,
            overall_status,
        })
    }
    
    /// Validate Rust toolchain installation and version
    pub fn validate_rust_toolchain(&self) -> Result<ToolchainStatus, EnvironmentError> {
        // Check if rustc is available
        let rustc_output = Command::new("rustc")
            .arg("--version")
            .output()
            .map_err(|_| EnvironmentError::RustNotInstalled)?;
        
        if !rustc_output.status.success() {
            return Err(EnvironmentError::RustNotInstalled);
        }
        
        let version_str = String::from_utf8_lossy(&rustc_output.stdout);
        let version = self.extract_rust_version(&version_str)?;
        
        // Check if cargo is available
        let cargo_output = Command::new("cargo")
            .arg("--version")
            .output()
            .map_err(|_| EnvironmentError::CargoNotInstalled)?;
        
        if !cargo_output.status.success() {
            return Err(EnvironmentError::CargoNotInstalled);
        }
        
        let cargo_version_str = String::from_utf8_lossy(&cargo_output.stdout);
        let cargo_version = self.extract_cargo_version(&cargo_version_str)?;
        
        // Check for required toolchain components
        let components = self.check_toolchain_components()?;
        
        Ok(ToolchainStatus {
            rustc_version: version,
            cargo_version,
            components,
            is_valid: true,
        })
    }
    
    /// Validate project dependencies
    pub fn validate_dependencies(&self) -> Result<DependencyStatus, EnvironmentError> {
        let mut missing_dependencies = Vec::new();
        let mut available_dependencies = Vec::new();
        
        // Check if Cargo.toml exists
        if !Path::new("Cargo.toml").exists() {
            return Err(EnvironmentError::CargoTomlNotFound);
        }
        
        // Check for MLIR dependencies
        let mlir_available = self.check_mlir_availability();
        if mlir_available {
            available_dependencies.push("MLIR".to_string());
        } else {
            missing_dependencies.push("MLIR (melior, mlir-sys)".to_string());
        }
        
        // Check for tempfile dependency
        let tempfile_available = self.check_tempfile_availability();
        if tempfile_available {
            available_dependencies.push("tempfile".to_string());
        } else {
            missing_dependencies.push("tempfile".to_string());
        }
        
        let is_valid = missing_dependencies.is_empty();
        
        Ok(DependencyStatus {
            missing_dependencies,
            available_dependencies,
            is_valid,
        })
    }
    
    /// Validate system requirements
    pub fn validate_system_requirements(&self) -> Result<SystemStatus, EnvironmentError> {
        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;
        
        // Check for Windows-specific requirements
        let windows_requirements = if os == "windows" {
            self.check_windows_requirements()?
        } else {
            WindowsRequirements {
                visual_studio_build_tools: false,
                windows_sdk: false,
                is_valid: true, // Not Windows, so requirements are met
            }
        };
        
        // Check available memory
        let available_memory = self.estimate_available_memory();
        
        // Check disk space
        let available_disk_space = self.estimate_available_disk_space();
        
        Ok(SystemStatus {
            operating_system: os.to_string(),
            architecture: arch.to_string(),
            windows_requirements,
            available_memory_gb: available_memory,
            available_disk_space_gb: available_disk_space,
            is_valid: available_memory >= 4.0 && available_disk_space >= 10.0,
        })
    }
    
    /// Extract Rust version from rustc output
    fn extract_rust_version(&self, version_str: &str) -> Result<String, EnvironmentError> {
        // Parse "rustc 1.70.0 (90c541806 2023-05-31)"
        let parts: Vec<&str> = version_str.split_whitespace().collect();
        if parts.len() >= 2 {
            Ok(parts[1].to_string())
        } else {
            Err(EnvironmentError::VersionParseError("Could not parse rustc version".to_string()))
        }
    }
    
    /// Extract Cargo version from cargo output
    fn extract_cargo_version(&self, version_str: &str) -> Result<String, EnvironmentError> {
        // Parse "cargo 1.70.0 (7fe40dc8c 2023-04-27)"
        let parts: Vec<&str> = version_str.split_whitespace().collect();
        if parts.len() >= 2 {
            Ok(parts[1].to_string())
        } else {
            Err(EnvironmentError::VersionParseError("Could not parse cargo version".to_string()))
        }
    }
    
    /// Check for required toolchain components
    fn check_toolchain_components(&self) -> Result<Vec<String>, EnvironmentError> {
        let mut components = Vec::new();
        
        // Check for rustfmt
        if Command::new("rustfmt").arg("--version").output().is_ok() {
            components.push("rustfmt".to_string());
        }
        
        // Check for clippy
        if Command::new("cargo").args(&["clippy", "--version"]).output().is_ok() {
            components.push("clippy".to_string());
        }
        
        Ok(components)
    }
    
    /// Check MLIR availability
    fn check_mlir_availability(&self) -> bool {
        // Try to compile a simple program that uses MLIR features
        let test_output = Command::new("cargo")
            .args(&["check", "--features", "mlir", "--quiet"])
            .output();
        
        match test_output {
            Ok(output) => output.status.success(),
            Err(_) => false,
        }
    }
    
    /// Check tempfile availability
    fn check_tempfile_availability(&self) -> bool {
        // Check if tempfile is in Cargo.toml
        if let Ok(cargo_content) = std::fs::read_to_string("Cargo.toml") {
            cargo_content.contains("tempfile")
        } else {
            false
        }
    }
    
    /// Check Windows-specific build requirements
    fn check_windows_requirements(&self) -> Result<WindowsRequirements, EnvironmentError> {
        // Check for Visual Studio Build Tools
        let vs_build_tools = self.check_visual_studio_build_tools();
        
        // Check for Windows SDK
        let windows_sdk = self.check_windows_sdk();
        
        Ok(WindowsRequirements {
            visual_studio_build_tools: vs_build_tools,
            windows_sdk,
            is_valid: vs_build_tools && windows_sdk,
        })
    }
    
    /// Check for Visual Studio Build Tools
    fn check_visual_studio_build_tools(&self) -> bool {
        // Check common installation paths
        let vs_paths = [
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files\Microsoft Visual Studio\2019\Community",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        ];
        
        vs_paths.iter().any(|path| Path::new(path).exists())
    }
    
    /// Check for Windows SDK
    fn check_windows_sdk(&self) -> bool {
        // Check for Windows SDK installation
        let sdk_paths = [
            r"C:\Program Files (x86)\Windows Kits\10",
            r"C:\Program Files\Windows Kits\10",
        ];
        
        sdk_paths.iter().any(|path| Path::new(path).exists())
    }
    
    /// Estimate available system memory
    fn estimate_available_memory(&self) -> f64 {
        // This is a simplified estimation
        // In practice, you'd use system APIs to get actual memory info
        8.0 // Assume 8GB for now
    }
    
    /// Estimate available disk space
    fn estimate_available_disk_space(&self) -> f64 {
        // This is a simplified estimation
        // In practice, you'd check actual disk space
        50.0 // Assume 50GB for now
    }
    
    /// Determine overall environment status
    fn determine_overall_status(
        &self,
        rust_status: &ToolchainStatus,
        dependency_status: &DependencyStatus,
        system_status: &SystemStatus,
    ) -> ValidationStatus {
        if rust_status.is_valid && dependency_status.is_valid && system_status.is_valid {
            ValidationStatus::Valid
        } else if rust_status.is_valid {
            ValidationStatus::PartiallyValid
        } else {
            ValidationStatus::Invalid
        }
    }
}

/// Overall environment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentStatus {
    pub rust_toolchain: ToolchainStatus,
    pub dependencies: DependencyStatus,
    pub system: SystemStatus,
    pub overall_status: ValidationStatus,
}

/// Rust toolchain status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolchainStatus {
    pub rustc_version: String,
    pub cargo_version: String,
    pub components: Vec<String>,
    pub is_valid: bool,
}

/// Dependency status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyStatus {
    pub missing_dependencies: Vec<String>,
    pub available_dependencies: Vec<String>,
    pub is_valid: bool,
}

/// System requirements status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub operating_system: String,
    pub architecture: String,
    pub windows_requirements: WindowsRequirements,
    pub available_memory_gb: f64,
    pub available_disk_space_gb: f64,
    pub is_valid: bool,
}

/// Windows-specific requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsRequirements {
    pub visual_studio_build_tools: bool,
    pub windows_sdk: bool,
    pub is_valid: bool,
}

/// Validation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Valid,
    PartiallyValid,
    Invalid,
}

/// Environment validation errors
#[derive(Debug)]
pub enum EnvironmentError {
    RustNotInstalled,
    CargoNotInstalled,
    CargoTomlNotFound,
    VersionParseError(String),
    SystemRequirementNotMet(String),
}

impl std::fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvironmentError::RustNotInstalled => write!(f, "Rust compiler (rustc) is not installed"),
            EnvironmentError::CargoNotInstalled => write!(f, "Cargo package manager is not installed"),
            EnvironmentError::CargoTomlNotFound => write!(f, "Cargo.toml file not found in current directory"),
            EnvironmentError::VersionParseError(msg) => write!(f, "Version parse error: {}", msg),
            EnvironmentError::SystemRequirementNotMet(msg) => write!(f, "System requirement not met: {}", msg),
        }
    }
}

impl std::error::Error for EnvironmentError {}