// Rust compiler interface and dependency management for Aether build system
// Handles toolchain detection, dependency installation, and compilation orchestration

use crate::build_system::{BuildConfig, Dependency, ToolchainConfig, BuildSystemError};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Rust compiler interface for managing toolchain and dependencies
#[derive(Debug, Clone)]
pub struct RustCompiler {
    toolchain_version: String,
    feature_flags: Vec<String>,
    target_triple: String,
    cargo_path: PathBuf,
    rustc_path: PathBuf,
}

impl RustCompiler {
    /// Create new RustCompiler with default configuration
    pub fn new() -> Result<Self, RustCompilerError> {
        let toolchain_info = Self::detect_toolchain()?;
        
        Ok(Self {
            toolchain_version: toolchain_info.version,
            feature_flags: Vec::new(),
            target_triple: toolchain_info.target_triple,
            cargo_path: toolchain_info.cargo_path,
            rustc_path: toolchain_info.rustc_path,
        })
    }
    
    /// Create RustCompiler with specific configuration
    pub fn with_config(config: &ToolchainConfig) -> Result<Self, RustCompilerError> {
        let mut compiler = Self::new()?;
        compiler.toolchain_version = config.version.clone();
        compiler.feature_flags = config.features.clone();
        Ok(compiler)
    }
    
    /// Check Rust toolchain installation and get information
    pub fn check_installation(&self) -> Result<ToolchainInfo, RustCompilerError> {
        // Check if cargo is available
        let cargo_version = self.get_cargo_version()?;
        
        // Check if rustc is available
        let rustc_version = self.get_rustc_version()?;
        
        // Get installed targets
        let installed_targets = self.get_installed_targets()?;
        
        // Get installed components
        let installed_components = self.get_installed_components()?;
        
        Ok(ToolchainInfo {
            version: self.toolchain_version.clone(),
            cargo_version,
            rustc_version,
            target_triple: self.target_triple.clone(),
            cargo_path: self.cargo_path.clone(),
            rustc_path: self.rustc_path.clone(),
            installed_targets,
            installed_components,
        })
    }
    
    /// Compile with specified features
    pub fn compile_with_features(&self, features: &[String]) -> Result<CompilationResult, RustCompilerError> {
        let start_time = Instant::now();
        
        let mut cmd = Command::new(&self.cargo_path);
        cmd.arg("build");
        
        // Add feature flags
        if !features.is_empty() {
            cmd.arg("--features");
            cmd.arg(features.join(","));
        }
        
        // Add any additional flags
        for flag in &self.feature_flags {
            if !features.contains(flag) {
                cmd.arg("--features");
                cmd.arg(flag);
            }
        }
        
        // Set target if specified
        if !self.target_triple.is_empty() && self.target_triple != "default" {
            cmd.arg("--target");
            cmd.arg(&self.target_triple);
        }
        
        let output = cmd.output()
            .map_err(|e| RustCompilerError::CompilationFailed(
                format!("Failed to execute cargo build: {}", e)
            ))?;
        
        let duration = start_time.elapsed();
        
        let result = CompilationResult {
            success: output.status.success(),
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            duration,
            features_used: features.to_vec(),
        };
        
        // Analyze compilation result for errors
        if !result.success {
            let analysis = self.analyze_compilation_errors(&result)?;
            return Err(RustCompilerError::CompilationFailedWithAnalysis(result, analysis));
        }
        
        Ok(result)
    }
    
    /// Install missing dependencies automatically
    pub fn install_missing_dependencies(&self, deps: &[Dependency]) -> Result<DependencyInstallResult, RustCompilerError> {
        let mut install_results = Vec::new();
        let mut overall_success = true;
        
        for dep in deps {
            println!("Installing dependency: {} v{}", dep.name, dep.version);
            
            let result = self.install_single_dependency(dep)?;
            overall_success &= result.success;
            install_results.push(result);
        }
        
        Ok(DependencyInstallResult {
            overall_success,
            individual_results: install_results,
        })
    }
    
    /// Manage feature flags for optional dependencies
    pub fn manage_feature_flags(&mut self, config: &BuildConfig) -> Result<FeatureFlagResult, RustCompilerError> {
        let mut enabled_features = Vec::new();
        let mut disabled_features = Vec::new();
        
        // Check which optional dependencies are available
        for dep in &config.rust_toolchain.dependencies {
            if dep.optional {
                if self.is_dependency_available(&dep.name)? {
                    // Enable the feature if dependency is available
                    let feature_name = self.dependency_to_feature_name(&dep.name);
                    if !self.feature_flags.contains(&feature_name) {
                        self.feature_flags.push(feature_name.clone());
                        enabled_features.push(feature_name);
                    }
                } else {
                    // Disable the feature if dependency is not available
                    let feature_name = self.dependency_to_feature_name(&dep.name);
                    if let Some(pos) = self.feature_flags.iter().position(|x| x == &feature_name) {
                        self.feature_flags.remove(pos);
                        disabled_features.push(feature_name);
                    }
                }
            }
        }
        
        // Add explicitly requested features
        for feature in &config.rust_toolchain.features {
            if !self.feature_flags.contains(feature) {
                self.feature_flags.push(feature.clone());
                enabled_features.push(feature.clone());
            }
        }
        
        Ok(FeatureFlagResult {
            enabled_features,
            disabled_features,
            current_features: self.feature_flags.clone(),
        })
    }
    
    /// Analyze compilation result and extract error information
    pub fn analyze_compilation_result(&self, result: &CompilationResult) -> Result<CompilationAnalysis, RustCompilerError> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut missing_dependencies = Vec::new();
        let mut feature_issues = Vec::new();
        
        // Parse stderr for error patterns
        for line in result.stderr.lines() {
            if line.contains("error:") {
                errors.push(self.parse_error_line(line)?);
            } else if line.contains("warning:") {
                warnings.push(self.parse_warning_line(line)?);
            } else if line.contains("could not find") && line.contains("in registry") {
                if let Some(dep_name) = self.extract_dependency_name(line) {
                    missing_dependencies.push(dep_name);
                }
            } else if line.contains("feature") && (line.contains("not found") || line.contains("does not exist")) {
                if let Some(feature_name) = self.extract_feature_name(line) {
                    feature_issues.push(feature_name);
                }
            }
        }
        
        Ok(CompilationAnalysis {
            errors,
            warnings,
            missing_dependencies,
            feature_issues,
            compilation_time: result.duration,
            success: result.success,
        })
    }
    
    /// Get current feature flags
    pub fn get_feature_flags(&self) -> &[String] {
        &self.feature_flags
    }
    
    /// Set feature flags
    pub fn set_feature_flags(&mut self, features: Vec<String>) {
        self.feature_flags = features;
    }
    
    /// Add feature flag
    pub fn add_feature_flag(&mut self, feature: String) {
        if !self.feature_flags.contains(&feature) {
            self.feature_flags.push(feature);
        }
    }
    
    /// Remove feature flag
    pub fn remove_feature_flag(&mut self, feature: &str) {
        self.feature_flags.retain(|f| f != feature);
    }
    
    /// Get toolchain information
    pub fn get_toolchain_info(&self) -> ToolchainInfo {
        ToolchainInfo {
            version: self.toolchain_version.clone(),
            cargo_version: self.get_cargo_version().unwrap_or_else(|_| "unknown".to_string()),
            rustc_version: self.get_rustc_version().unwrap_or_else(|_| "unknown".to_string()),
            target_triple: self.target_triple.clone(),
            cargo_path: self.cargo_path.clone(),
            rustc_path: self.rustc_path.clone(),
            installed_targets: self.get_installed_targets().unwrap_or_default(),
            installed_components: self.get_installed_components().unwrap_or_default(),
        }
    }
    
    // Private helper methods
    
    /// Detect installed Rust toolchain
    fn detect_toolchain() -> Result<ToolchainInfo, RustCompilerError> {
        // Try to find cargo
        let cargo_path = Self::find_executable("cargo")
            .ok_or_else(|| RustCompilerError::ToolchainNotFound("cargo not found in PATH".to_string()))?;
        
        // Try to find rustc
        let rustc_path = Self::find_executable("rustc")
            .ok_or_else(|| RustCompilerError::ToolchainNotFound("rustc not found in PATH".to_string()))?;
        
        // Get version information
        let cargo_version = Self::get_version_from_path(&cargo_path, "cargo")?;
        let rustc_version = Self::get_version_from_path(&rustc_path, "rustc")?;
        
        // Get target triple
        let target_triple = Self::get_default_target_triple(&rustc_path)?;
        
        Ok(ToolchainInfo {
            version: "stable".to_string(), // Default assumption
            cargo_version,
            rustc_version,
            target_triple,
            cargo_path,
            rustc_path,
            installed_targets: Vec::new(),
            installed_components: Vec::new(),
        })
    }
    
    /// Find executable in PATH
    fn find_executable(name: &str) -> Option<PathBuf> {
        // On Windows, try with .exe extension
        let exe_name = if cfg!(windows) {
            format!("{}.exe", name)
        } else {
            name.to_string()
        };
        
        if let Ok(path_var) = std::env::var("PATH") {
            for path in std::env::split_paths(&path_var) {
                let full_path = path.join(&exe_name);
                if full_path.exists() {
                    return Some(full_path);
                }
            }
        }
        
        None
    }
    
    /// Get version from executable path
    fn get_version_from_path(path: &Path, tool_name: &str) -> Result<String, RustCompilerError> {
        let output = Command::new(path)
            .arg("--version")
            .output()
            .map_err(|e| RustCompilerError::ToolchainDetectionFailed(
                format!("Failed to get {} version: {}", tool_name, e)
            ))?;
        
        if !output.status.success() {
            return Err(RustCompilerError::ToolchainDetectionFailed(
                format!("{} --version failed", tool_name)
            ));
        }
        
        let version_output = String::from_utf8_lossy(&output.stdout);
        Ok(version_output.lines().next().unwrap_or("unknown").to_string())
    }
    
    /// Get default target triple
    fn get_default_target_triple(rustc_path: &Path) -> Result<String, RustCompilerError> {
        let output = Command::new(rustc_path)
            .arg("-vV")
            .output()
            .map_err(|e| RustCompilerError::ToolchainDetectionFailed(
                format!("Failed to get rustc target info: {}", e)
            ))?;
        
        if !output.status.success() {
            return Ok("unknown".to_string());
        }
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        for line in output_str.lines() {
            if line.starts_with("host: ") {
                return Ok(line.strip_prefix("host: ").unwrap_or("unknown").to_string());
            }
        }
        
        Ok("unknown".to_string())
    }
    
    /// Get cargo version
    fn get_cargo_version(&self) -> Result<String, RustCompilerError> {
        Self::get_version_from_path(&self.cargo_path, "cargo")
    }
    
    /// Get rustc version
    fn get_rustc_version(&self) -> Result<String, RustCompilerError> {
        Self::get_version_from_path(&self.rustc_path, "rustc")
    }
    
    /// Get installed targets
    fn get_installed_targets(&self) -> Result<Vec<String>, RustCompilerError> {
        let output = Command::new("rustup")
            .args(&["target", "list", "--installed"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                let targets = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                Ok(targets)
            }
            _ => Ok(vec![self.target_triple.clone()]), // Fallback to default target
        }
    }
    
    /// Get installed components
    fn get_installed_components(&self) -> Result<Vec<String>, RustCompilerError> {
        let output = Command::new("rustup")
            .args(&["component", "list", "--installed"])
            .output();
        
        match output {
            Ok(output) if output.status.success() => {
                let components = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                Ok(components)
            }
            _ => Ok(vec!["rust-std".to_string(), "rustc".to_string()]), // Basic components
        }
    }
    
    /// Install a single dependency
    fn install_single_dependency(&self, dep: &Dependency) -> Result<SingleDependencyResult, RustCompilerError> {
        // For Rust dependencies, we don't install them directly
        // Instead, we add them to Cargo.toml and let cargo handle it
        // This is a placeholder for the actual implementation
        
        let start_time = Instant::now();
        
        // Check if dependency is already in Cargo.toml
        if self.is_dependency_in_cargo_toml(&dep.name)? {
            return Ok(SingleDependencyResult {
                dependency_name: dep.name.clone(),
                success: true,
                already_installed: true,
                installation_time: start_time.elapsed(),
                error_message: None,
            });
        }
        
        // Add dependency to Cargo.toml
        match self.add_dependency_to_cargo_toml(dep) {
            Ok(_) => Ok(SingleDependencyResult {
                dependency_name: dep.name.clone(),
                success: true,
                already_installed: false,
                installation_time: start_time.elapsed(),
                error_message: None,
            }),
            Err(e) => Ok(SingleDependencyResult {
                dependency_name: dep.name.clone(),
                success: false,
                already_installed: false,
                installation_time: start_time.elapsed(),
                error_message: Some(e.to_string()),
            }),
        }
    }
    
    /// Check if dependency is available
    fn is_dependency_available(&self, dep_name: &str) -> Result<bool, RustCompilerError> {
        // Check if dependency is in Cargo.toml
        self.is_dependency_in_cargo_toml(dep_name)
    }
    
    /// Convert dependency name to feature name
    pub fn dependency_to_feature_name(&self, dep_name: &str) -> String {
        match dep_name {
            "melior" | "mlir-sys" => "mlir".to_string(),
            name => name.replace("-", "_"),
        }
    }
    
    /// Check if dependency is in Cargo.toml
    fn is_dependency_in_cargo_toml(&self, dep_name: &str) -> Result<bool, RustCompilerError> {
        let cargo_toml_path = Path::new("Cargo.toml");
        if !cargo_toml_path.exists() {
            return Ok(false);
        }
        
        let content = std::fs::read_to_string(cargo_toml_path)
            .map_err(|e| RustCompilerError::FileOperationFailed(
                format!("Failed to read Cargo.toml: {}", e)
            ))?;
        
        // Simple check - look for dependency name in the file
        Ok(content.contains(&format!("{} =", dep_name)) || 
           content.contains(&format!("\"{}\"", dep_name)))
    }
    
    /// Add dependency to Cargo.toml
    fn add_dependency_to_cargo_toml(&self, dep: &Dependency) -> Result<(), RustCompilerError> {
        let cargo_toml_path = Path::new("Cargo.toml");
        let content = std::fs::read_to_string(cargo_toml_path)
            .map_err(|e| RustCompilerError::FileOperationFailed(
                format!("Failed to read Cargo.toml: {}", e)
            ))?;
        
        // Find the [dependencies] section and add the dependency
        let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
        let mut in_dependencies = false;
        let mut insert_index = None;
        
        for (i, line) in lines.iter().enumerate() {
            if line.trim() == "[dependencies]" {
                in_dependencies = true;
            } else if line.starts_with('[') && line.trim() != "[dependencies]" && in_dependencies {
                insert_index = Some(i);
                break;
            }
        }
        
        // If we didn't find where to insert, add at the end
        if insert_index.is_none() && in_dependencies {
            insert_index = Some(lines.len());
        }
        
        if let Some(index) = insert_index {
            let dep_line = if dep.optional {
                format!("{} = {{ version = \"{}\", optional = true }}", dep.name, dep.version)
            } else {
                format!("{} = \"{}\"", dep.name, dep.version)
            };
            lines.insert(index, dep_line);
        } else {
            // Add [dependencies] section if it doesn't exist
            lines.push(String::new());
            lines.push("[dependencies]".to_string());
            let dep_line = if dep.optional {
                format!("{} = {{ version = \"{}\", optional = true }}", dep.name, dep.version)
            } else {
                format!("{} = \"{}\"", dep.name, dep.version)
            };
            lines.push(dep_line);
        }
        
        let new_content = lines.join("\n");
        std::fs::write(cargo_toml_path, new_content)
            .map_err(|e| RustCompilerError::FileOperationFailed(
                format!("Failed to write Cargo.toml: {}", e)
            ))?;
        
        Ok(())
    }
    
    /// Analyze compilation errors
    fn analyze_compilation_errors(&self, result: &CompilationResult) -> Result<CompilationAnalysis, RustCompilerError> {
        self.analyze_compilation_result(result)
    }
    
    /// Parse error line from compiler output
    fn parse_error_line(&self, line: &str) -> Result<CompilationError, RustCompilerError> {
        // Simple error parsing - in practice this would be more sophisticated
        Ok(CompilationError {
            message: line.to_string(),
            file: None,
            line_number: None,
            column: None,
            error_code: None,
        })
    }
    
    /// Parse warning line from compiler output
    fn parse_warning_line(&self, line: &str) -> Result<CompilationWarning, RustCompilerError> {
        Ok(CompilationWarning {
            message: line.to_string(),
            file: None,
            line_number: None,
            column: None,
        })
    }
    
    /// Extract dependency name from error message
    pub fn extract_dependency_name(&self, line: &str) -> Option<String> {
        // Look for patterns like "could not find `dependency_name` in registry"
        if let Some(start) = line.find('`') {
            if let Some(end) = line[start + 1..].find('`') {
                return Some(line[start + 1..start + 1 + end].to_string());
            }
        }
        None
    }
    
    /// Extract feature name from error message
    pub fn extract_feature_name(&self, line: &str) -> Option<String> {
        // Look for patterns like "feature `feature_name` does not exist"
        if let Some(start) = line.find('`') {
            if let Some(end) = line[start + 1..].find('`') {
                return Some(line[start + 1..start + 1 + end].to_string());
            }
        }
        None
    }
}

impl Default for RustCompiler {
    fn default() -> Self {
        Self::new().expect("Failed to create default RustCompiler")
    }
}
/// Toolchain information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolchainInfo {
    pub version: String,
    pub cargo_version: String,
    pub rustc_version: String,
    pub target_triple: String,
    pub cargo_path: PathBuf,
    pub rustc_path: PathBuf,
    pub installed_targets: Vec<String>,
    pub installed_components: Vec<String>,
}

/// Compilation result from Rust compiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationResult {
    pub success: bool,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration: Duration,
    pub features_used: Vec<String>,
}

/// Result of dependency installation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInstallResult {
    pub overall_success: bool,
    pub individual_results: Vec<SingleDependencyResult>,
}

/// Result of installing a single dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleDependencyResult {
    pub dependency_name: String,
    pub success: bool,
    pub already_installed: bool,
    pub installation_time: Duration,
    pub error_message: Option<String>,
}

/// Result of feature flag management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlagResult {
    pub enabled_features: Vec<String>,
    pub disabled_features: Vec<String>,
    pub current_features: Vec<String>,
}

/// Analysis of compilation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationAnalysis {
    pub errors: Vec<CompilationError>,
    pub warnings: Vec<CompilationWarning>,
    pub missing_dependencies: Vec<String>,
    pub feature_issues: Vec<String>,
    pub compilation_time: Duration,
    pub success: bool,
}

/// Compilation error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationError {
    pub message: String,
    pub file: Option<PathBuf>,
    pub line_number: Option<usize>,
    pub column: Option<usize>,
    pub error_code: Option<String>,
}

/// Compilation warning information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationWarning {
    pub message: String,
    pub file: Option<PathBuf>,
    pub line_number: Option<usize>,
    pub column: Option<usize>,
}

/// Rust compiler specific errors
#[derive(Debug)]
pub enum RustCompilerError {
    ToolchainNotFound(String),
    ToolchainDetectionFailed(String),
    CompilationFailed(String),
    CompilationFailedWithAnalysis(CompilationResult, CompilationAnalysis),
    DependencyInstallationFailed(String),
    FeatureManagementFailed(String),
    FileOperationFailed(String),
    ConfigurationError(String),
}

impl std::fmt::Display for RustCompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RustCompilerError::ToolchainNotFound(msg) => write!(f, "Rust toolchain not found: {}", msg),
            RustCompilerError::ToolchainDetectionFailed(msg) => write!(f, "Toolchain detection failed: {}", msg),
            RustCompilerError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            RustCompilerError::CompilationFailedWithAnalysis(_result, analysis) => {
                write!(f, "Compilation failed with {} errors and {} warnings", 
                       analysis.errors.len(), analysis.warnings.len())
            }
            RustCompilerError::DependencyInstallationFailed(msg) => write!(f, "Dependency installation failed: {}", msg),
            RustCompilerError::FeatureManagementFailed(msg) => write!(f, "Feature management failed: {}", msg),
            RustCompilerError::FileOperationFailed(msg) => write!(f, "File operation failed: {}", msg),
            RustCompilerError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for RustCompilerError {}

impl From<RustCompilerError> for BuildSystemError {
    fn from(error: RustCompilerError) -> Self {
        BuildSystemError::ConfigurationError(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_system::{Dependency, ToolchainConfig};
    
    #[test]
    fn test_rust_compiler_creation() {
        // This test might fail if Rust toolchain is not installed
        // In a real environment, we'd mock the toolchain detection
        match RustCompiler::new() {
            Ok(compiler) => {
                assert!(!compiler.toolchain_version.is_empty());
                assert!(!compiler.target_triple.is_empty());
            }
            Err(_) => {
                // If toolchain is not available, that's also a valid test result
                println!("Rust toolchain not available for testing");
            }
        }
    }
    
    #[test]
    fn test_feature_flag_management() {
        let mut compiler = RustCompiler {
            toolchain_version: "stable".to_string(),
            feature_flags: Vec::new(),
            target_triple: "x86_64-pc-windows-msvc".to_string(),
            cargo_path: PathBuf::from("cargo"),
            rustc_path: PathBuf::from("rustc"),
        };
        
        // Test adding feature flags
        compiler.add_feature_flag("mlir".to_string());
        assert!(compiler.get_feature_flags().contains(&"mlir".to_string()));
        
        // Test removing feature flags
        compiler.remove_feature_flag("mlir");
        assert!(!compiler.get_feature_flags().contains(&"mlir".to_string()));
    }
    
    #[test]
    fn test_dependency_to_feature_name() {
        let compiler = RustCompiler {
            toolchain_version: "stable".to_string(),
            feature_flags: Vec::new(),
            target_triple: "x86_64-pc-windows-msvc".to_string(),
            cargo_path: PathBuf::from("cargo"),
            rustc_path: PathBuf::from("rustc"),
        };
        
        assert_eq!(compiler.dependency_to_feature_name("melior"), "mlir");
        assert_eq!(compiler.dependency_to_feature_name("mlir-sys"), "mlir");
        assert_eq!(compiler.dependency_to_feature_name("some-crate"), "some_crate");
    }
    
    #[test]
    fn test_extract_dependency_name() {
        let compiler = RustCompiler {
            toolchain_version: "stable".to_string(),
            feature_flags: Vec::new(),
            target_triple: "x86_64-pc-windows-msvc".to_string(),
            cargo_path: PathBuf::from("cargo"),
            rustc_path: PathBuf::from("rustc"),
        };
        
        let error_line = "error: could not find `tempfile` in registry";
        assert_eq!(compiler.extract_dependency_name(error_line), Some("tempfile".to_string()));
        
        let error_line2 = "error: could not find `melior` in registry";
        assert_eq!(compiler.extract_dependency_name(error_line2), Some("melior".to_string()));
    }
    
    #[test]
    fn test_extract_feature_name() {
        let compiler = RustCompiler {
            toolchain_version: "stable".to_string(),
            feature_flags: Vec::new(),
            target_triple: "x86_64-pc-windows-msvc".to_string(),
            cargo_path: PathBuf::from("cargo"),
            rustc_path: PathBuf::from("rustc"),
        };
        
        let error_line = "error: feature `mlir` does not exist";
        assert_eq!(compiler.extract_feature_name(error_line), Some("mlir".to_string()));
    }
    
    #[test]
    fn test_compilation_result_analysis() {
        let compiler = RustCompiler {
            toolchain_version: "stable".to_string(),
            feature_flags: Vec::new(),
            target_triple: "x86_64-pc-windows-msvc".to_string(),
            cargo_path: PathBuf::from("cargo"),
            rustc_path: PathBuf::from("rustc"),
        };
        
        let result = CompilationResult {
            success: false,
            exit_code: 1,
            stdout: String::new(),
            stderr: "error: could not find `tempfile` in registry\nwarning: unused import".to_string(),
            duration: Duration::from_secs(1),
            features_used: Vec::new(),
        };
        
        let analysis = compiler.analyze_compilation_result(&result).unwrap();
        assert!(!analysis.success);
        assert_eq!(analysis.errors.len(), 1);
        assert_eq!(analysis.warnings.len(), 1);
        assert_eq!(analysis.missing_dependencies.len(), 1);
        assert_eq!(analysis.missing_dependencies[0], "tempfile");
    }
}