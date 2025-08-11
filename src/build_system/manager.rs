// Build system manager for Aether compiler
// Orchestrates compilation, error handling, and environment validation

use crate::build_system::{
    BuildConfig, ErrorHandler, AutoFixEngine, EnvironmentValidator,
    EnvironmentStatus, BuildError, AetherCompiler, AetherCompilationConfig,
    AetherCompilationResult, AetherCompilerError, SyntaxValidation, DiagnosticReport
};
use crate::build_system::rust_compiler::{
    RustCompiler, ToolchainInfo, CompilationAnalysis
};
use std::path::{Path, PathBuf};
use std::process::Command;
use serde::{Deserialize, Serialize};

/// Main build system manager
pub struct BuildSystemManager {
    config: BuildConfig,
    error_handler: ErrorHandler,
    fix_engine: AutoFixEngine,
    environment_validator: EnvironmentValidator,
    rust_compiler: RustCompiler,
    aether_compiler: Option<AetherCompiler>,
}

impl BuildSystemManager {
    /// Create new build system manager with default configuration
    pub fn new() -> Self {
        let config = BuildConfig::default();
        let rust_compiler = RustCompiler::with_config(&config.rust_toolchain)
            .unwrap_or_else(|_| RustCompiler::default());
        
        Self {
            config,
            error_handler: ErrorHandler::new(),
            fix_engine: AutoFixEngine::new(),
            environment_validator: EnvironmentValidator::new(),
            rust_compiler,
            aether_compiler: None,
        }
    }
    
    /// Create build system manager with custom configuration
    pub fn with_config(config: BuildConfig) -> Self {
        let rust_compiler = RustCompiler::with_config(&config.rust_toolchain)
            .unwrap_or_else(|_| RustCompiler::default());
        
        Self {
            config,
            error_handler: ErrorHandler::new(),
            fix_engine: AutoFixEngine::new(),
            environment_validator: EnvironmentValidator::new(),
            rust_compiler,
            aether_compiler: None,
        }
    }
    
    /// Validate the build environment
    pub fn validate_environment(&self) -> Result<EnvironmentStatus, BuildSystemError> {
        self.environment_validator
            .validate_environment()
            .map_err(BuildSystemError::EnvironmentValidation)
    }
    
    /// Compile the Aether compiler from source
    pub fn compile_aether_compiler(&mut self) -> Result<CompilerBinary, BuildSystemError> {
        println!("Starting Aether compiler compilation...");
        
        // First validate environment
        let env_status = self.validate_environment()?;
        if !matches!(env_status.overall_status, crate::build_system::ValidationStatus::Valid) {
            return Err(BuildSystemError::EnvironmentNotReady(
                "Environment validation failed. Please fix environment issues first.".to_string()
            ));
        }
        
        // Remove mock implementations first
        self.remove_mock_implementations()?;
        
        // Run cargo build
        let build_result = self.run_cargo_build()?;
        
        // Check for errors and attempt fixes
        if !build_result.success {
            self.handle_build_errors(&build_result)?;
            
            // Retry build after fixes
            let retry_result = self.run_cargo_build()?;
            if !retry_result.success {
                return Err(BuildSystemError::CompilationFailed(
                    "Build failed even after applying fixes".to_string()
                ));
            }
        }
        
        // Locate the compiled binary
        let binary_path = self.locate_compiler_binary()?;
        
        Ok(CompilerBinary {
            path: binary_path,
            version: self.get_compiler_version()?,
            features: self.config.rust_toolchain.features.clone(),
        })
    }
    
    /// Compile Aether source code to executable
    pub fn compile_aether_source(&mut self, source_path: &Path) -> Result<Executable, BuildSystemError> {
        if !source_path.exists() {
            return Err(BuildSystemError::SourceFileNotFound(
                source_path.to_string_lossy().to_string()
            ));
        }
        
        // Ensure we have a working Rust compiler first
        let compiler_binary = self.compile_aether_compiler()?;
        
        // Initialize Aether compiler if not already done
        if self.aether_compiler.is_none() {
            let aether_config = AetherCompilationConfig {
                target: crate::build_system::CompilationTarget::WindowsNative,
                optimization_level: 2,
                debug_info: true,
                compiler_flags: Vec::new(),
                output_directory: self.config.aether_config.output_directory.clone(),
                verbose: false,
            };
            
            match AetherCompiler::with_config(compiler_binary.path.clone(), aether_config) {
                Ok(aether_compiler) => {
                    self.aether_compiler = Some(aether_compiler);
                }
                Err(e) => {
                    return Err(BuildSystemError::AetherCompilationFailed(
                        format!("Failed to initialize Aether compiler: {}", e)
                    ));
                }
            }
        }
        
        // Determine output path
        let output_path = self.determine_output_path(source_path);
        
        // Compile using AetherCompiler
        let compile_result = self.aether_compiler.as_mut().unwrap()
            .compile_to_executable(source_path, &output_path)
            .map_err(|e| BuildSystemError::AetherCompilationFailed(e.to_string()))?;
        
        if !compile_result.success {
            return Err(BuildSystemError::AetherCompilationFailed(
                format!("Failed to compile {}: {}", source_path.display(), compile_result.stderr)
            ));
        }
        
        Ok(Executable {
            path: compile_result.executable_path.unwrap_or(output_path),
            source_file: source_path.to_path_buf(),
            compilation_time: compile_result.compilation_time,
        })
    }
    
    /// Run verification tests on compiled executable
    pub fn run_verification_tests(&self, executable: &Path) -> Result<TestResults, BuildSystemError> {
        if !executable.exists() {
            return Err(BuildSystemError::ExecutableNotFound(
                executable.to_string_lossy().to_string()
            ));
        }
        
        // Run basic execution test
        let execution_result = Command::new(executable)
            .output()
            .map_err(|e| BuildSystemError::TestExecutionFailed(format!("Failed to run executable: {}", e)))?;
        
        let basic_test = BasicTestResult {
            execution_success: execution_result.status.success(),
            exit_code: execution_result.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&execution_result.stdout).to_string(),
            stderr: String::from_utf8_lossy(&execution_result.stderr).to_string(),
        };
        
        let overall_success = basic_test.execution_success;
        
        Ok(TestResults {
            basic_test,
            advanced_tests: None, // TODO: Implement advanced tests
            overall_success,
        })
    }
    
    /// Remove mock implementations from the codebase
    fn remove_mock_implementations(&mut self) -> Result<(), BuildSystemError> {
        println!("Removing mock implementations...");
        
        // Remove mock exports from MLIR module
        let mlir_mod_path = Path::new("src/compiler/mlir/mod.rs");
        if mlir_mod_path.exists() {
            let content = std::fs::read_to_string(mlir_mod_path)
                .map_err(|e| BuildSystemError::FileOperationFailed(format!("Failed to read MLIR mod.rs: {}", e)))?;
            
            // Remove the mock exports line
            let new_content = content.replace(
                "pub use test_utils::{MockMLIRContext, MockMLIRModule, MockOperation, TestMLIRPipeline};",
                "// Mock implementations removed - using real MLIR implementations"
            );
            
            std::fs::write(mlir_mod_path, new_content)
                .map_err(|e| BuildSystemError::FileOperationFailed(format!("Failed to write MLIR mod.rs: {}", e)))?;
        }
        
        // Remove or update test files that use mock implementations
        self.update_test_files_to_use_real_implementations()?;
        
        println!("Mock implementations removed successfully.");
        Ok(())
    }
    
    /// Update test files to use real implementations instead of mocks
    fn update_test_files_to_use_real_implementations(&self) -> Result<(), BuildSystemError> {
        let test_files = [
            "tests/gpu_kernel_tests.rs",
            "tests/mlir_tests.rs",
            "tests/native_compilation_tests.rs",
            "tests/wasm_compilation_tests.rs",
        ];
        
        for test_file in &test_files {
            let path = Path::new(test_file);
            if path.exists() {
                let content = std::fs::read_to_string(path)
                    .map_err(|e| BuildSystemError::FileOperationFailed(format!("Failed to read {}: {}", test_file, e)))?;
                
                // Replace mock usage with real implementations
                let new_content = content
                    .replace("MockMLIRContext::new()", "MLIRContext::new().expect(\"Failed to create MLIR context\")")
                    .replace("MockMLIRModule::new()", "context.create_module(\"test_module\").expect(\"Failed to create module\")")
                    .replace("TestMLIRPipeline::new()", "MLIRPipeline::new().expect(\"Failed to create pipeline\")");
                
                std::fs::write(path, new_content)
                    .map_err(|e| BuildSystemError::FileOperationFailed(format!("Failed to write {}: {}", test_file, e)))?;
            }
        }
        
        Ok(())
    }
    
    /// Run cargo build and capture output using RustCompiler
    fn run_cargo_build(&self) -> Result<BuildResult, BuildSystemError> {
        // Use RustCompiler to compile with features
        let compilation_result = self.rust_compiler
            .compile_with_features(&self.config.rust_toolchain.features)
            .map_err(|e| BuildSystemError::CompilationFailed(e.to_string()))?;
        
        Ok(BuildResult {
            success: compilation_result.success,
            stdout: compilation_result.stdout,
            stderr: compilation_result.stderr,
            duration: compilation_result.duration,
        })
    }
    
    /// Handle build errors by detecting and applying fixes
    fn handle_build_errors(&mut self, build_result: &BuildResult) -> Result<(), BuildSystemError> {
        println!("Handling build errors...");
        
        // Detect errors from build output
        if let Some(error) = self.error_handler.detect_error(&build_result.stderr) {
            println!("Detected error: {}", error.message);
            
            // Get suggested fixes
            let fixes = self.error_handler.get_suggested_fixes(&error);
            
            // Apply fixes
            for fix in fixes {
                println!("Applying fix: {:?}", fix);
                let result = self.fix_engine.apply_fix(&fix);
                if let Err(fix_error) = result.application_result {
                    println!("Warning: Failed to apply fix: {}", fix_error);
                } else {
                    println!("Fix applied successfully");
                }
            }
        }
        
        Ok(())
    }
    
    /// Locate the compiled Aether compiler binary
    fn locate_compiler_binary(&self) -> Result<PathBuf, BuildSystemError> {
        let target_dir = Path::new("target");
        
        // Check debug build first
        let debug_binary = target_dir.join("debug").join("aetherc.exe");
        if debug_binary.exists() {
            return Ok(debug_binary);
        }
        
        // Check release build
        let release_binary = target_dir.join("release").join("aetherc.exe");
        if release_binary.exists() {
            return Ok(release_binary);
        }
        
        // Check without .exe extension (for non-Windows)
        let debug_binary_no_ext = target_dir.join("debug").join("aetherc");
        if debug_binary_no_ext.exists() {
            return Ok(debug_binary_no_ext);
        }
        
        let release_binary_no_ext = target_dir.join("release").join("aetherc");
        if release_binary_no_ext.exists() {
            return Ok(release_binary_no_ext);
        }
        
        Err(BuildSystemError::CompilerBinaryNotFound(
            "Could not locate aetherc binary in target directory".to_string()
        ))
    }
    
    /// Get compiler version
    fn get_compiler_version(&self) -> Result<String, BuildSystemError> {
        // For now, return a default version
        // In practice, this would query the actual compiler
        Ok("0.1.0".to_string())
    }
    
    /// Determine output path for Aether source compilation
    fn determine_output_path(&self, source_path: &Path) -> PathBuf {
        let mut output_path = self.config.aether_config.output_directory.clone();
        
        if let Some(stem) = source_path.file_stem() {
            output_path.push(format!("{}.exe", stem.to_string_lossy()));
        } else {
            output_path.push("output.exe");
        }
        
        output_path
    }
    
    /// Run the Aether compiler on source file
    fn run_aether_compiler(
        &self,
        compiler_path: &Path,
        source_path: &Path,
        output_path: &Path,
    ) -> Result<CompilationResult, BuildSystemError> {
        let start_time = std::time::Instant::now();
        
        // Create output directory if it doesn't exist
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| BuildSystemError::FileOperationFailed(format!("Failed to create output directory: {}", e)))?;
        }
        
        let output = Command::new(compiler_path)
            .arg(source_path)
            .arg("-o")
            .arg(output_path)
            .output()
            .map_err(|e| BuildSystemError::CommandExecutionFailed(format!("Failed to run Aether compiler: {}", e)))?;
        
        Ok(CompilationResult {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            duration: start_time.elapsed(),
        })
    }
    
    /// Get current configuration
    pub fn config(&self) -> &BuildConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: BuildConfig) {
        self.config = config;
    }
    
    /// Get error history
    pub fn get_error_history(&self) -> &[BuildError] {
        self.error_handler.get_error_history()
    }
    
    /// Clear error history
    pub fn clear_error_history(&mut self) {
        self.error_handler.clear_history();
    }
    
    /// Install missing dependencies automatically
    pub fn install_missing_dependencies(&self) -> Result<crate::build_system::rust_compiler::DependencyInstallResult, BuildSystemError> {
        self.rust_compiler
            .install_missing_dependencies(&self.config.rust_toolchain.dependencies)
            .map_err(|e| BuildSystemError::ConfigurationError(e.to_string()))
    }
    
    /// Manage feature flags for optional dependencies
    pub fn manage_feature_flags(&mut self) -> Result<crate::build_system::rust_compiler::FeatureFlagResult, BuildSystemError> {
        self.rust_compiler
            .manage_feature_flags(&self.config)
            .map_err(|e| BuildSystemError::ConfigurationError(e.to_string()))
    }
    
    /// Get Rust toolchain information
    pub fn get_toolchain_info(&self) -> ToolchainInfo {
        self.rust_compiler.get_toolchain_info()
    }
    
    /// Check Rust toolchain installation
    pub fn check_rust_installation(&self) -> Result<ToolchainInfo, BuildSystemError> {
        self.rust_compiler
            .check_installation()
            .map_err(|e| BuildSystemError::ConfigurationError(e.to_string()))
    }
    
    /// Analyze compilation result for errors and suggestions
    pub fn analyze_compilation_result(&self, result: &BuildResult) -> Result<CompilationAnalysis, BuildSystemError> {
        let compilation_result = crate::build_system::rust_compiler::CompilationResult {
            success: result.success,
            exit_code: if result.success { 0 } else { 1 },
            stdout: result.stdout.clone(),
            stderr: result.stderr.clone(),
            duration: result.duration,
            features_used: self.config.rust_toolchain.features.clone(),
        };
        
        self.rust_compiler
            .analyze_compilation_result(&compilation_result)
            .map_err(|e| BuildSystemError::ConfigurationError(e.to_string()))
    }
    
    /// Get current Rust compiler feature flags
    pub fn get_rust_feature_flags(&self) -> &[String] {
        self.rust_compiler.get_feature_flags()
    }
    
    /// Add Rust compiler feature flag
    pub fn add_rust_feature_flag(&mut self, feature: String) {
        self.rust_compiler.add_feature_flag(feature);
    }
    
    /// Remove Rust compiler feature flag
    pub fn remove_rust_feature_flag(&mut self, feature: &str) {
        self.rust_compiler.remove_feature_flag(feature);
    }

    /// Validate Aether source syntax
    pub fn validate_aether_syntax(&mut self, source_path: &Path) -> Result<SyntaxValidation, BuildSystemError> {
        // Ensure we have a working compiler
        let compiler_binary = self.compile_aether_compiler()?;
        
        // Initialize Aether compiler if not already done
        if self.aether_compiler.is_none() {
            let aether_config = AetherCompilationConfig::default();
            match AetherCompiler::with_config(compiler_binary.path.clone(), aether_config) {
                Ok(aether_compiler) => {
                    self.aether_compiler = Some(aether_compiler);
                }
                Err(e) => {
                    return Err(BuildSystemError::AetherCompilationFailed(
                        format!("Failed to initialize Aether compiler: {}", e)
                    ));
                }
            }
        }
        
        self.aether_compiler.as_mut().unwrap()
            .validate_syntax(source_path)
            .map_err(|e| BuildSystemError::AetherCompilationFailed(e.to_string()))
    }

    /// Generate AST from Aether source
    pub fn generate_aether_ast(&mut self, source_path: &Path) -> Result<crate::compiler::AST, BuildSystemError> {
        let validation = self.validate_aether_syntax(source_path)?;
        
        if !validation.is_valid {
            return Err(BuildSystemError::AetherCompilationFailed(
                format!("Syntax validation failed for {}", source_path.display())
            ));
        }
        
        validation.ast.ok_or_else(|| BuildSystemError::AetherCompilationFailed(
            "AST generation failed despite valid syntax".to_string()
        ))
    }

    /// Get Aether compiler diagnostics
    pub fn get_aether_diagnostics(&self, error: &AetherCompilerError) -> DiagnosticReport {
        if let Some(compiler) = &self.aether_compiler {
            compiler.generate_diagnostics(error)
        } else {
            DiagnosticReport {
                diagnostics: Vec::new(),
                error_count: 1,
                warning_count: 0,
                formatted_output: format!("Aether compiler not initialized: {}", error),
            }
        }
    }

    /// Update Aether compiler configuration
    pub fn update_aether_config(&mut self, config: AetherCompilationConfig) -> Result<(), BuildSystemError> {
        if let Some(compiler) = &mut self.aether_compiler {
            compiler.update_config(config);
            Ok(())
        } else {
            Err(BuildSystemError::ConfigurationError(
                "Aether compiler not initialized".to_string()
            ))
        }
    }

    /// Get current Aether compiler configuration
    pub fn get_aether_config(&self) -> Option<&AetherCompilationConfig> {
        self.aether_compiler.as_ref().map(|c| c.config())
    }

    /// Verify Aether compiler is working
    pub fn verify_aether_compiler(&self) -> Result<(), BuildSystemError> {
        if let Some(compiler) = &self.aether_compiler {
            compiler.verify_compiler()
                .map_err(|e| BuildSystemError::ConfigurationError(e.to_string()))
        } else {
            Err(BuildSystemError::ConfigurationError(
                "Aether compiler not initialized".to_string()
            ))
        }
    }

    /// Get Aether compiler version
    pub fn get_aether_compiler_version(&self) -> Result<String, BuildSystemError> {
        if let Some(compiler) = &self.aether_compiler {
            compiler.get_version()
                .map_err(|e| BuildSystemError::ConfigurationError(e.to_string()))
        } else {
            Err(BuildSystemError::ConfigurationError(
                "Aether compiler not initialized".to_string()
            ))
        }
    }

    /// List supported Aether compilation targets
    pub fn get_aether_supported_targets(&self) -> Vec<crate::build_system::CompilationTarget> {
        if let Some(compiler) = &self.aether_compiler {
            compiler.supported_targets()
        } else {
            Vec::new()
        }
    }

    /// Compile Aether source with custom configuration
    pub fn compile_aether_source_with_config(
        &mut self,
        source_path: &Path,
        output_path: &Path,
        config: AetherCompilationConfig,
    ) -> Result<AetherCompilationResult, BuildSystemError> {
        // Ensure we have a working compiler
        let compiler_binary = self.compile_aether_compiler()?;
        
        // Create or update Aether compiler with new config
        match AetherCompiler::with_config(compiler_binary.path.clone(), config) {
            Ok(mut aether_compiler) => {
                aether_compiler.compile_to_executable(source_path, output_path)
                    .map_err(|e| BuildSystemError::AetherCompilationFailed(e.to_string()))
            }
            Err(e) => {
                Err(BuildSystemError::AetherCompilationFailed(
                    format!("Failed to initialize Aether compiler: {}", e)
                ))
            }
        }
    }
}

/// Compiled Aether compiler binary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerBinary {
    pub path: PathBuf,
    pub version: String,
    pub features: Vec<String>,
}

/// Compiled executable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Executable {
    pub path: PathBuf,
    pub source_file: PathBuf,
    pub compilation_time: std::time::Duration,
}

/// Build result from cargo build
#[derive(Debug, Clone)]
pub struct BuildResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub duration: std::time::Duration,
}

/// Compilation result from Aether compiler
#[derive(Debug, Clone)]
pub struct CompilationResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub duration: std::time::Duration,
}

/// Test results from verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    pub basic_test: BasicTestResult,
    pub advanced_tests: Option<AdvancedTestResults>,
    pub overall_success: bool,
}

/// Basic test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicTestResult {
    pub execution_success: bool,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Advanced test results (placeholder for future implementation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedTestResults {
    pub tensor_operations: bool,
    pub automatic_differentiation: bool,
    pub probabilistic_programming: bool,
}

/// Build system errors
#[derive(Debug)]
pub enum BuildSystemError {
    EnvironmentValidation(crate::build_system::EnvironmentError),
    EnvironmentNotReady(String),
    CompilationFailed(String),
    AetherCompilationFailed(String),
    SourceFileNotFound(String),
    ExecutableNotFound(String),
    CompilerBinaryNotFound(String),
    TestExecutionFailed(String),
    FileOperationFailed(String),
    CommandExecutionFailed(String),
    ConfigurationError(String),
}

impl std::fmt::Display for BuildSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildSystemError::EnvironmentValidation(err) => write!(f, "Environment validation error: {}", err),
            BuildSystemError::EnvironmentNotReady(msg) => write!(f, "Environment not ready: {}", msg),
            BuildSystemError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            BuildSystemError::AetherCompilationFailed(msg) => write!(f, "Aether compilation failed: {}", msg),
            BuildSystemError::SourceFileNotFound(path) => write!(f, "Source file not found: {}", path),
            BuildSystemError::ExecutableNotFound(path) => write!(f, "Executable not found: {}", path),
            BuildSystemError::CompilerBinaryNotFound(msg) => write!(f, "Compiler binary not found: {}", msg),
            BuildSystemError::TestExecutionFailed(msg) => write!(f, "Test execution failed: {}", msg),
            BuildSystemError::FileOperationFailed(msg) => write!(f, "File operation failed: {}", msg),
            BuildSystemError::CommandExecutionFailed(msg) => write!(f, "Command execution failed: {}", msg),
            BuildSystemError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for BuildSystemError {}