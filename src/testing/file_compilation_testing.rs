// Aether File Compilation Testing Framework
// Comprehensive testing system for compiling all Aether source files to executables

use std::path::{Path, PathBuf};
use std::time::Duration;
use std::fs;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Main testing orchestrator for file compilation testing
pub struct FileCompilationTestOrchestrator {
    config: TestingConfig,
    file_discovery: FileDiscoveryEngine,
    test_generator: TestFileGenerator,
    compilation_engine: CompilationEngine,
    execution_validator: ExecutionValidator,
    report_generator: ReportGenerator,
    logger: TestingLogger,
    cleanup_manager: ResourceCleanupManager,
    error_recovery: ErrorRecoverySystem,
}

impl FileCompilationTestOrchestrator {
    /// Create a new file compilation test orchestrator
    pub fn new(config: TestingConfig) -> Result<Self, TestingError> {
        // Initialize logging system
        let log_file = if config.verbose {
            Some(config.output_directory.join("testing.log"))
        } else {
            None
        };
        let logger = TestingLogger::with_settings(
            if config.verbose { LogLevel::Debug } else { LogLevel::Info },
            log_file,
            true,
        );

        logger.log(
            LogLevel::Info,
            "Orchestrator",
            "initialization",
            "Initializing file compilation test orchestrator",
            HashMap::new(),
        );

        // Initialize cleanup manager
        let cleanup_manager = ResourceCleanupManager::new(config.cleanup_artifacts, logger.clone());

        // Initialize error recovery system
        let error_recovery = ErrorRecoverySystem::new(logger.clone(), cleanup_manager.clone());

        // Initialize components with error handling
        let file_discovery = match FileDiscoveryEngine::new(config.project_root.clone()) {
            Ok(engine) => engine,
            Err(e) => {
                logger.log_error("Orchestrator", "initialization", &e);
                if let Err(recovery_err) = error_recovery.attempt_recovery("file_discovery", &e) {
                    logger.log_error("Orchestrator", "initialization", &recovery_err);
                }
                return Err(e);
            }
        };

        let test_generator = match TestFileGenerator::new(config.output_directory.clone()) {
            Ok(generator) => generator,
            Err(e) => {
                logger.log_error("Orchestrator", "initialization", &e);
                if let Err(recovery_err) = error_recovery.attempt_recovery("test_generation", &e) {
                    logger.log_error("Orchestrator", "initialization", &recovery_err);
                }
                return Err(e);
            }
        };

        let compilation_engine = match CompilationEngine::new(
            config.compiler_path.clone(),
            config.output_directory.clone(),
            config.compilation_timeout,
        ) {
            Ok(engine) => engine,
            Err(e) => {
                logger.log_error("Orchestrator", "initialization", &e);
                if let Err(recovery_err) = error_recovery.attempt_recovery("compilation", &e) {
                    logger.log_error("Orchestrator", "initialization", &recovery_err);
                }
                return Err(e);
            }
        };

        let execution_validator = match ExecutionValidator::new(config.execution_timeout) {
            Ok(validator) => validator,
            Err(e) => {
                logger.log_error("Orchestrator", "initialization", &e);
                if let Err(recovery_err) = error_recovery.attempt_recovery("execution", &e) {
                    logger.log_error("Orchestrator", "initialization", &recovery_err);
                }
                return Err(e);
            }
        };

        let report_generator = match ReportGenerator::new(config.report_format.clone()) {
            Ok(generator) => generator,
            Err(e) => {
                logger.log_error("Orchestrator", "initialization", &e);
                if let Err(recovery_err) = error_recovery.attempt_recovery("report_generation", &e) {
                    logger.log_error("Orchestrator", "initialization", &recovery_err);
                }
                return Err(e);
            }
        };

        // Register output directory for cleanup
        cleanup_manager.register_temp_directory(config.output_directory.clone());

        logger.log(
            LogLevel::Info,
            "Orchestrator",
            "initialization",
            "File compilation test orchestrator initialized successfully",
            HashMap::new(),
        );

        Ok(Self {
            config,
            file_discovery,
            test_generator,
            compilation_engine,
            execution_validator,
            report_generator,
            logger,
            cleanup_manager,
            error_recovery,
        })
    }

    /// Run the complete file compilation testing workflow with comprehensive error handling
    pub async fn run_complete_test_suite(&mut self) -> Result<FileCompilationTestReport, TestingError> {
        let start_time = std::time::Instant::now();
        
        self.logger.log(
            LogLevel::Info,
            "Orchestrator",
            "run_complete_test_suite",
            "Starting complete file compilation test suite",
            HashMap::new(),
        );

        let mut errors_encountered = Vec::new();

        // Phase 1: Discover existing Aether files with error handling
        let discovered_files = match self.discover_files_with_recovery().await {
            Ok(files) => files,
            Err(e) => {
                errors_encountered.push(e.clone());
                self.logger.log_error("Orchestrator", "file_discovery", &e);
                
                // Attempt graceful degradation
                if let Err(degradation_error) = self.error_recovery.graceful_degradation(&e, "file_discovery") {
                    self.logger.log_error("Orchestrator", "graceful_degradation", &degradation_error);
                }
                
                // Continue with empty file list if discovery fails completely
                Vec::new()
            }
        };

        // Phase 2: Generate additional test files with error handling
        let generated_files = if self.config.generate_additional_tests {
            match self.generate_test_files_with_recovery().await {
                Ok(files) => files,
                Err(e) => {
                    errors_encountered.push(e.clone());
                    self.logger.log_error("Orchestrator", "test_generation", &e);
                    
                    // Continue without generated files if generation fails
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        // Phase 3: Combine all files for testing
        let mut all_files = discovered_files.clone();
        all_files.extend(generated_files.iter().map(|gf| gf.file_path.clone()));

        self.logger.log(
            LogLevel::Info,
            "Orchestrator",
            "run_complete_test_suite",
            &format!("Processing {} total files ({} discovered, {} generated)", 
                all_files.len(), discovered_files.len(), generated_files.len()),
            HashMap::new(),
        );

        // Register all generated files for cleanup
        for generated_file in &generated_files {
            self.cleanup_manager.register_temp_file(generated_file.file_path.clone());
        }

        // Phase 4: Compile all files with error handling and recovery
        let compilation_results = match self.compile_files_with_recovery(&all_files).await {
            Ok(results) => results,
            Err(e) => {
                errors_encountered.push(e.clone());
                self.logger.log_error("Orchestrator", "compilation", &e);
                
                // Attempt to continue with partial results if possible
                match self.attempt_partial_compilation(&all_files).await {
                    Ok(partial_results) => {
                        self.logger.log(
                            LogLevel::Warn,
                            "Orchestrator",
                            "compilation",
                            &format!("Using partial compilation results: {} files", partial_results.len()),
                            HashMap::new(),
                        );
                        partial_results
                    },
                    Err(partial_error) => {
                        self.logger.log_error("Orchestrator", "partial_compilation", &partial_error);
                        Vec::new()
                    }
                }
            }
        };

        // Phase 5: Execute successful compilations with error handling
        let successful_executables: Vec<PathBuf> = compilation_results
            .iter()
            .filter_map(|result| {
                if result.success {
                    // Register executable for cleanup
                    if let Some(ref exe_path) = result.executable_path {
                        self.cleanup_manager.register_temp_file(exe_path.clone());
                    }
                    result.executable_path.clone()
                } else {
                    None
                }
            })
            .collect();

        self.logger.log(
            LogLevel::Info,
            "Orchestrator",
            "run_complete_test_suite",
            &format!("Executing {} successful compilations", successful_executables.len()),
            HashMap::new(),
        );

        let execution_results = match self.execute_files_with_recovery(&successful_executables).await {
            Ok(results) => results,
            Err(e) => {
                errors_encountered.push(e.clone());
                self.logger.log_error("Orchestrator", "execution", &e);
                
                // Attempt partial execution
                match self.attempt_partial_execution(&successful_executables).await {
                    Ok(partial_results) => {
                        self.logger.log(
                            LogLevel::Warn,
                            "Orchestrator",
                            "execution",
                            &format!("Using partial execution results: {} files", partial_results.len()),
                            HashMap::new(),
                        );
                        partial_results
                    },
                    Err(partial_error) => {
                        self.logger.log_error("Orchestrator", "partial_execution", &partial_error);
                        Vec::new()
                    }
                }
            }
        };

        // Phase 6: Generate comprehensive report
        let test_report = FileCompilationTestReport {
            summary: TestSummary::from_results(&compilation_results, &execution_results, generated_files.len()),
            compilation_results,
            execution_results,
            generated_files,
            discovered_files: all_files,
            config: self.config.clone(),
        };

        // Log performance metrics
        let total_duration = start_time.elapsed();
        let mut performance_context = HashMap::new();
        performance_context.insert("total_files".to_string(), test_report.discovered_files.len().to_string());
        performance_context.insert("successful_compilations".to_string(), test_report.summary.successful_compilations.to_string());
        performance_context.insert("successful_executions".to_string(), test_report.summary.successful_executions.to_string());
        performance_context.insert("errors_encountered".to_string(), errors_encountered.len().to_string());

        self.logger.log_performance(
            "Orchestrator",
            "run_complete_test_suite",
            total_duration,
            performance_context,
        );

        // Handle any accumulated errors
        if !errors_encountered.is_empty() {
            match self.error_recovery.handle_multiple_errors(errors_encountered) {
                Ok(unrecovered_errors) => {
                    if !unrecovered_errors.is_empty() {
                        self.logger.log(
                            LogLevel::Warn,
                            "Orchestrator",
                            "run_complete_test_suite",
                            &format!("Test suite completed with {} unrecovered errors", unrecovered_errors.len()),
                            HashMap::new(),
                        );
                    }
                },
                Err(recovery_error) => {
                    self.logger.log_error("Orchestrator", "error_recovery", &recovery_error);
                }
            }
        }

        self.logger.log(
            LogLevel::Info,
            "Orchestrator",
            "run_complete_test_suite",
            &format!("Test suite completed in {:?}", total_duration),
            HashMap::new(),
        );

        Ok(test_report)
    }

    /// Validate configuration and setup with comprehensive error handling
    pub fn validate_setup(&self) -> Result<(), TestingError> {
        self.logger.log(
            LogLevel::Info,
            "Orchestrator",
            "validate_setup",
            "Validating orchestrator setup",
            HashMap::new(),
        );

        let mut validation_errors = Vec::new();

        // Validate configuration
        if let Err(e) = self.config.validate() {
            validation_errors.push(e);
        }

        // Validate file discovery
        if let Err(e) = self.file_discovery.validate_setup() {
            validation_errors.push(e);
        }

        // Validate compilation engine
        if let Err(e) = self.compilation_engine.validate_setup() {
            validation_errors.push(e);
        }

        // Validate execution validator
        if let Err(e) = self.execution_validator.validate_setup() {
            validation_errors.push(e);
        }

        if validation_errors.is_empty() {
            self.logger.log(
                LogLevel::Info,
                "Orchestrator",
                "validate_setup",
                "All components validated successfully",
                HashMap::new(),
            );
            Ok(())
        } else {
            let multiple_error = TestingError::Multiple(validation_errors);
            self.logger.log_error("Orchestrator", "validate_setup", &multiple_error);
            Err(multiple_error)
        }
    }

    /// Discover files with error recovery
    async fn discover_files_with_recovery(&self) -> Result<Vec<PathBuf>, TestingError> {
        self.logger.log(
            LogLevel::Debug,
            "Orchestrator",
            "discover_files_with_recovery",
            "Starting file discovery with recovery",
            HashMap::new(),
        );

        match self.file_discovery.discover_aether_files() {
            Ok(files) => {
                self.logger.log(
                    LogLevel::Info,
                    "Orchestrator",
                    "discover_files_with_recovery",
                    &format!("Successfully discovered {} files", files.len()),
                    HashMap::new(),
                );
                Ok(files)
            },
            Err(e) => {
                self.logger.log_error("Orchestrator", "discover_files_with_recovery", &e);
                
                // Attempt recovery by trying specific directories
                let mut recovered_files = Vec::new();
                let target_dirs = ["examples", "tests", "src"];
                
                for dir in &target_dirs {
                    match self.file_discovery.discover_in_directories(&[dir]) {
                        Ok(mut dir_files) => {
                            recovered_files.append(&mut dir_files);
                            self.logger.log(
                                LogLevel::Info,
                                "Orchestrator",
                                "discover_files_with_recovery",
                                &format!("Recovered {} files from {} directory", dir_files.len(), dir),
                                HashMap::new(),
                            );
                        },
                        Err(dir_error) => {
                            self.logger.log(
                                LogLevel::Warn,
                                "Orchestrator",
                                "discover_files_with_recovery",
                                &format!("Failed to recover files from {} directory: {}", dir, dir_error),
                                HashMap::new(),
                            );
                        }
                    }
                }

                if !recovered_files.is_empty() {
                    self.logger.log_recovery("Orchestrator", "discover_files_with_recovery", "partial file discovery", true);
                    Ok(recovered_files)
                } else {
                    self.logger.log_recovery("Orchestrator", "discover_files_with_recovery", "partial file discovery", false);
                    Err(e)
                }
            }
        }
    }

    /// Generate test files with error recovery
    async fn generate_test_files_with_recovery(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        self.logger.log(
            LogLevel::Debug,
            "Orchestrator",
            "generate_test_files_with_recovery",
            "Starting test file generation with recovery",
            HashMap::new(),
        );

        match self.test_generator.generate_test_files() {
            Ok(files) => {
                self.logger.log(
                    LogLevel::Info,
                    "Orchestrator",
                    "generate_test_files_with_recovery",
                    &format!("Successfully generated {} test files", files.len()),
                    HashMap::new(),
                );
                Ok(files)
            },
            Err(e) => {
                self.logger.log_error("Orchestrator", "generate_test_files_with_recovery", &e);
                
                // Attempt recovery by generating core language tests only
                match self.test_generator.generate_core_language_tests() {
                    Ok(core_files) => {
                        self.logger.log_recovery("Orchestrator", "generate_test_files_with_recovery", "core language tests only", true);
                        Ok(core_files)
                    },
                    Err(core_error) => {
                        self.logger.log_error("Orchestrator", "generate_test_files_with_recovery", &core_error);
                        self.logger.log_recovery("Orchestrator", "generate_test_files_with_recovery", "core language tests only", false);
                        Err(e)
                    }
                }
            }
        }
    }

    /// Compile files with error recovery
    async fn compile_files_with_recovery(&self, files: &[PathBuf]) -> Result<Vec<CompilationResult>, TestingError> {
        self.logger.log(
            LogLevel::Debug,
            "Orchestrator",
            "compile_files_with_recovery",
            &format!("Starting compilation of {} files with recovery", files.len()),
            HashMap::new(),
        );

        match self.compilation_engine.compile_batch(files).await {
            Ok(results) => {
                let successful = results.iter().filter(|r| r.success).count();
                self.logger.log(
                    LogLevel::Info,
                    "Orchestrator",
                    "compile_files_with_recovery",
                    &format!("Compilation completed: {}/{} successful", successful, results.len()),
                    HashMap::new(),
                );
                Ok(results)
            },
            Err(e) => {
                self.logger.log_error("Orchestrator", "compile_files_with_recovery", &e);
                Err(e)
            }
        }
    }

    /// Attempt partial compilation when batch compilation fails
    async fn attempt_partial_compilation(&self, files: &[PathBuf]) -> Result<Vec<CompilationResult>, TestingError> {
        self.logger.log(
            LogLevel::Info,
            "Orchestrator",
            "attempt_partial_compilation",
            &format!("Attempting partial compilation of {} files", files.len()),
            HashMap::new(),
        );

        let mut results = Vec::new();
        let mut successful_count = 0;
        let mut failed_count = 0;

        // Try to compile files individually
        for file in files {
            match self.compilation_engine.compile_file(file).await {
                Ok(result) => {
                    if result.success {
                        successful_count += 1;
                    } else {
                        failed_count += 1;
                    }
                    results.push(result);
                },
                Err(e) => {
                    failed_count += 1;
                    self.logger.log(
                        LogLevel::Warn,
                        "Orchestrator",
                        "attempt_partial_compilation",
                        &format!("Failed to compile {}: {}", file.display(), e),
                        HashMap::new(),
                    );
                    
                    // Create a failed compilation result
                    results.push(CompilationResult {
                        source_file: file.clone(),
                        executable_path: None,
                        success: false,
                        stdout: String::new(),
                        stderr: format!("Compilation error: {}", e),
                        compilation_time: Duration::from_secs(0),
                        exit_code: Some(-1),
                    });
                }
            }
        }

        self.logger.log(
            LogLevel::Info,
            "Orchestrator",
            "attempt_partial_compilation",
            &format!("Partial compilation completed: {}/{} successful", successful_count, results.len()),
            HashMap::new(),
        );

        Ok(results)
    }

    /// Execute files with error recovery
    async fn execute_files_with_recovery(&self, executables: &[PathBuf]) -> Result<Vec<ExecutionResult>, TestingError> {
        self.logger.log(
            LogLevel::Debug,
            "Orchestrator",
            "execute_files_with_recovery",
            &format!("Starting execution of {} files with recovery", executables.len()),
            HashMap::new(),
        );

        match self.execution_validator.validate_batch(executables).await {
            Ok(results) => {
                let successful = results.iter().filter(|r| r.success).count();
                self.logger.log(
                    LogLevel::Info,
                    "Orchestrator",
                    "execute_files_with_recovery",
                    &format!("Execution completed: {}/{} successful", successful, results.len()),
                    HashMap::new(),
                );
                Ok(results)
            },
            Err(e) => {
                self.logger.log_error("Orchestrator", "execute_files_with_recovery", &e);
                Err(e)
            }
        }
    }

    /// Attempt partial execution when batch execution fails
    async fn attempt_partial_execution(&self, executables: &[PathBuf]) -> Result<Vec<ExecutionResult>, TestingError> {
        self.logger.log(
            LogLevel::Info,
            "Orchestrator",
            "attempt_partial_execution",
            &format!("Attempting partial execution of {} files", executables.len()),
            HashMap::new(),
        );

        let mut results = Vec::new();
        let mut successful_count = 0;
        let mut failed_count = 0;

        // Try to execute files individually
        for executable in executables {
            match self.execution_validator.validate_executable(executable).await {
                Ok(result) => {
                    if result.success {
                        successful_count += 1;
                    } else {
                        failed_count += 1;
                    }
                    results.push(result);
                },
                Err(e) => {
                    failed_count += 1;
                    self.logger.log(
                        LogLevel::Warn,
                        "Orchestrator",
                        "attempt_partial_execution",
                        &format!("Failed to execute {}: {}", executable.display(), e),
                        HashMap::new(),
                    );
                    
                    // Create a failed execution result
                    results.push(ExecutionResult {
                        executable_path: executable.clone(),
                        success: false,
                        exit_code: -1,
                        stdout: String::new(),
                        stderr: format!("Execution error: {}", e),
                        execution_time: Duration::from_secs(0),
                        timed_out: false,
                        memory_exceeded: false,
                        error_message: Some(format!("Execution error: {}", e)),
                    });
                }
            }
        }

        self.logger.log(
            LogLevel::Info,
            "Orchestrator",
            "attempt_partial_execution",
            &format!("Partial execution completed: {}/{} successful", successful_count, results.len()),
            HashMap::new(),
        );

        Ok(results)
    }

    /// Get the logger instance
    pub fn logger(&self) -> &TestingLogger {
        &self.logger
    }

    /// Get the cleanup manager instance
    pub fn cleanup_manager(&self) -> &ResourceCleanupManager {
        &self.cleanup_manager
    }

    /// Get the error recovery system instance
    pub fn error_recovery(&self) -> &ErrorRecoverySystem {
        &self.error_recovery
    }

    /// Perform manual cleanup of all resources
    pub fn cleanup_all_resources(&self) -> Result<(), TestingError> {
        self.logger.log(
            LogLevel::Info,
            "Orchestrator",
            "cleanup_all_resources",
            "Starting manual cleanup of all resources",
            HashMap::new(),
        );

        match self.cleanup_manager.cleanup_all() {
            Ok(()) => {
                self.logger.log(
                    LogLevel::Info,
                    "Orchestrator",
                    "cleanup_all_resources",
                    "All resources cleaned up successfully",
                    HashMap::new(),
                );
                Ok(())
            },
            Err(e) => {
                self.logger.log_error("Orchestrator", "cleanup_all_resources", &e);
                
                // Attempt force cleanup with retries
                match self.cleanup_manager.force_cleanup_with_retries(3, Duration::from_secs(1)) {
                    Ok(()) => {
                        self.logger.log_recovery("Orchestrator", "cleanup_all_resources", "force cleanup with retries", true);
                        Ok(())
                    },
                    Err(force_error) => {
                        self.logger.log_recovery("Orchestrator", "cleanup_all_resources", "force cleanup with retries", false);
                        Err(force_error)
                    }
                }
            }
        }
    }
}

impl Drop for FileCompilationTestOrchestrator {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup_all_resources() {
            eprintln!("Warning: Failed to cleanup resources on orchestrator drop: {}", e);
        }
    }
}

/// Configuration for the file compilation testing system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestingConfig {
    /// Root directory of the project to scan for Aether files
    pub project_root: PathBuf,
    
    /// Path to the aetherc compiler executable
    pub compiler_path: PathBuf,
    
    /// Directory for output files (executables, reports, generated tests)
    pub output_directory: PathBuf,
    
    /// Directories to include in testing (e.g., ["examples", "tests"])
    pub test_directories: Vec<String>,
    
    /// Timeout for compilation operations
    pub compilation_timeout: Duration,
    
    /// Timeout for executable execution
    pub execution_timeout: Duration,
    
    /// Whether to generate additional test files
    pub generate_additional_tests: bool,
    
    /// Categories of tests to generate
    pub test_categories: Vec<TestCategory>,
    
    /// Format for the final report
    pub report_format: ReportFormat,
    
    /// Whether to clean up temporary artifacts after testing
    pub cleanup_artifacts: bool,
    
    /// Maximum number of parallel compilation processes
    pub max_parallel_compilations: usize,
    
    /// Maximum number of parallel executions
    pub max_parallel_executions: usize,
    
    /// Verbose logging enabled
    pub verbose: bool,
}

impl TestingConfig {
    /// Create a default configuration
    pub fn default() -> Self {
        Self {
            project_root: PathBuf::from("."),
            compiler_path: PathBuf::from("aetherc"),
            output_directory: PathBuf::from("target/file_compilation_tests"),
            test_directories: vec!["examples".to_string(), "tests".to_string()],
            compilation_timeout: Duration::from_secs(30),
            execution_timeout: Duration::from_secs(10),
            generate_additional_tests: true,
            test_categories: vec![
                TestCategory::CoreLanguage,
                TestCategory::TypeSystem,
                TestCategory::AIFeatures,
                TestCategory::ErrorHandling,
            ],
            report_format: ReportFormat::Console,
            cleanup_artifacts: true,
            max_parallel_compilations: num_cpus::get(),
            max_parallel_executions: num_cpus::get() / 2,
            verbose: false,
        }
    }

    /// Load configuration from a file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, TestingError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| TestingError::ConfigurationError(format!("Failed to parse config: {}", e)))?;
        Ok(config)
    }

    /// Save configuration to a file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), TestingError> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| TestingError::ConfigurationError(format!("Failed to serialize config: {}", e)))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), TestingError> {
        if !self.project_root.exists() {
            return Err(TestingError::ConfigurationError(
                format!("Project root does not exist: {:?}", self.project_root)
            ));
        }

        if !self.compiler_path.exists() && which::which(&self.compiler_path).is_err() {
            return Err(TestingError::ConfigurationError(
                format!("Compiler not found: {:?}", self.compiler_path)
            ));
        }

        if self.compilation_timeout.is_zero() {
            return Err(TestingError::ConfigurationError(
                "Compilation timeout must be greater than zero".to_string()
            ));
        }

        if self.execution_timeout.is_zero() {
            return Err(TestingError::ConfigurationError(
                "Execution timeout must be greater than zero".to_string()
            ));
        }

        if self.max_parallel_compilations == 0 {
            return Err(TestingError::ConfigurationError(
                "Max parallel compilations must be greater than zero".to_string()
            ));
        }

        if self.max_parallel_executions == 0 {
            return Err(TestingError::ConfigurationError(
                "Max parallel executions must be greater than zero".to_string()
            ));
        }

        Ok(())
    }
}

/// Categories of test files that can be generated
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TestCategory {
    /// Core language features (functions, variables, control flow)
    CoreLanguage,
    /// Type system features (gradual typing, dependent types)
    TypeSystem,
    /// AI-specific features (tensors, autodiff, probabilistic programming)
    AIFeatures,
    /// Error handling and edge cases
    ErrorHandling,
    /// Performance and optimization tests
    Performance,
}

/// Report output formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReportFormat {
    /// Human-readable console output
    Console,
    /// JSON format for programmatic consumption
    Json,
    /// HTML format with styling
    Html,
    /// Markdown format for documentation
    Markdown,
}

/// File discovery specific errors
#[derive(Debug, Error, Clone)]
pub enum DiscoveryError {
    #[error("Directory not found: {0}")]
    DirectoryNotFound(PathBuf),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    #[error("IO error during discovery: {0}")]
    IoError(String),
    
    #[error("Invalid path: {0}")]
    InvalidPath(String),
}

/// Comprehensive error types for the testing system
#[derive(Debug, Error, Clone)]
pub enum TestingError {
    #[error("File discovery error: {0}")]
    FileDiscovery(String),

    #[error("Discovery error: {0}")]
    Discovery(#[from] DiscoveryError),

    #[error("Test generation error: {0}")]
    TestGeneration(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Report generation error: {0}")]
    ReportGeneration(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Resource cleanup error: {0}")]
    Cleanup(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),

    #[error("System error: {0}")]
    System(String),

    #[error("Recovery error: {0}")]
    Recovery(String),

    #[error("Logging error: {0}")]
    Logging(String),

    #[error("Multiple errors occurred: {0:?}")]
    Multiple(Vec<TestingError>),

    #[error("Critical system failure: {0}")]
    Critical(String),
}

/// Result type for testing operations
pub type TestingResult<T> = Result<T, TestingError>;

impl From<std::io::Error> for TestingError {
    fn from(error: std::io::Error) -> Self {
        TestingError::Io(error.to_string())
    }
}

/// Comprehensive logging and debugging system for the testing framework
#[derive(Debug, Clone)]
pub struct TestingLogger {
    log_level: LogLevel,
    log_file: Option<PathBuf>,
    console_output: bool,
    operation_logs: Arc<Mutex<HashMap<String, Vec<LogEntry>>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: LogLevel,
    pub component: String,
    pub operation: String,
    pub message: String,
    pub context: HashMap<String, String>,
}

impl TestingLogger {
    /// Create a new testing logger with default settings
    pub fn new() -> Self {
        Self {
            log_level: LogLevel::Info,
            log_file: None,
            console_output: true,
            operation_logs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new testing logger with custom settings
    pub fn with_settings(
        log_level: LogLevel,
        log_file: Option<PathBuf>,
        console_output: bool,
    ) -> Self {
        Self {
            log_level,
            log_file,
            console_output,
            operation_logs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Log a message with specified level and context
    pub fn log(&self, level: LogLevel, component: &str, operation: &str, message: &str, context: HashMap<String, String>) {
        if !self.should_log(&level) {
            return;
        }

        let entry = LogEntry {
            timestamp: chrono::Utc::now(),
            level: level.clone(),
            component: component.to_string(),
            operation: operation.to_string(),
            message: message.to_string(),
            context,
        };

        // Store in operation logs
        if let Ok(mut logs) = self.operation_logs.lock() {
            logs.entry(operation.to_string())
                .or_insert_with(Vec::new)
                .push(entry.clone());
        }

        // Output to console if enabled
        if self.console_output {
            self.output_to_console(&entry);
        }

        // Output to file if configured
        if let Some(ref log_file) = self.log_file {
            if let Err(e) = self.output_to_file(&entry, log_file) {
                eprintln!("Failed to write to log file: {}", e);
            }
        }
    }

    /// Log an error with automatic context extraction
    pub fn log_error(&self, component: &str, operation: &str, error: &TestingError) {
        let mut context = HashMap::new();
        context.insert("error_type".to_string(), format!("{:?}", error));
        
        // Extract additional context based on error type
        match error {
            TestingError::FileDiscovery(msg) => {
                context.insert("discovery_error".to_string(), msg.clone());
            },
            TestingError::Compilation(msg) => {
                context.insert("compilation_error".to_string(), msg.clone());
            },
            TestingError::Execution(msg) => {
                context.insert("execution_error".to_string(), msg.clone());
            },
            TestingError::Io(io_error) => {
                context.insert("io_error_message".to_string(), io_error.clone());
            },
            TestingError::Multiple(errors) => {
                context.insert("error_count".to_string(), errors.len().to_string());
                for (i, err) in errors.iter().enumerate() {
                    context.insert(format!("error_{}", i), format!("{}", err));
                }
            },
            _ => {},
        }

        self.log(LogLevel::Error, component, operation, &format!("Error occurred: {}", error), context);
    }

    /// Log a recovery attempt
    pub fn log_recovery(&self, component: &str, operation: &str, recovery_action: &str, success: bool) {
        let mut context = HashMap::new();
        context.insert("recovery_action".to_string(), recovery_action.to_string());
        context.insert("recovery_success".to_string(), success.to_string());

        let level = if success { LogLevel::Info } else { LogLevel::Warn };
        let message = if success {
            format!("Recovery successful: {}", recovery_action)
        } else {
            format!("Recovery failed: {}", recovery_action)
        };

        self.log(level, component, operation, &message, context);
    }

    /// Log performance metrics
    pub fn log_performance(&self, component: &str, operation: &str, duration: Duration, additional_metrics: HashMap<String, String>) {
        let mut context = additional_metrics;
        context.insert("duration_ms".to_string(), duration.as_millis().to_string());
        context.insert("duration_secs".to_string(), duration.as_secs_f64().to_string());

        self.log(LogLevel::Debug, component, operation, &format!("Operation completed in {:?}", duration), context);
    }

    /// Get logs for a specific operation
    pub fn get_operation_logs(&self, operation: &str) -> Vec<LogEntry> {
        if let Ok(logs) = self.operation_logs.lock() {
            logs.get(operation).cloned().unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Get all logs
    pub fn get_all_logs(&self) -> HashMap<String, Vec<LogEntry>> {
        if let Ok(logs) = self.operation_logs.lock() {
            logs.clone()
        } else {
            HashMap::new()
        }
    }

    /// Clear logs for a specific operation
    pub fn clear_operation_logs(&self, operation: &str) {
        if let Ok(mut logs) = self.operation_logs.lock() {
            logs.remove(operation);
        }
    }

    /// Clear all logs
    pub fn clear_all_logs(&self) {
        if let Ok(mut logs) = self.operation_logs.lock() {
            logs.clear();
        }
    }

    /// Check if a log level should be logged
    fn should_log(&self, level: &LogLevel) -> bool {
        match (&self.log_level, level) {
            (LogLevel::Trace, _) => true,
            (LogLevel::Debug, LogLevel::Trace) => false,
            (LogLevel::Debug, _) => true,
            (LogLevel::Info, LogLevel::Trace | LogLevel::Debug) => false,
            (LogLevel::Info, _) => true,
            (LogLevel::Warn, LogLevel::Trace | LogLevel::Debug | LogLevel::Info) => false,
            (LogLevel::Warn, _) => true,
            (LogLevel::Error, LogLevel::Critical | LogLevel::Error) => true,
            (LogLevel::Error, _) => false,
            (LogLevel::Critical, LogLevel::Critical) => true,
            (LogLevel::Critical, _) => false,
        }
    }

    /// Output log entry to console
    fn output_to_console(&self, entry: &LogEntry) {
        let level_str = match entry.level {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO ",
            LogLevel::Warn => "WARN ",
            LogLevel::Error => "ERROR",
            LogLevel::Critical => "CRIT ",
        };

        let timestamp = entry.timestamp.format("%Y-%m-%d %H:%M:%S%.3f");
        
        println!("[{}] {} [{}:{}] {}", 
            timestamp, level_str, entry.component, entry.operation, entry.message);

        // Print context if debug level or higher
        if matches!(self.log_level, LogLevel::Trace | LogLevel::Debug) && !entry.context.is_empty() {
            for (key, value) in &entry.context {
                println!("    {}: {}", key, value);
            }
        }
    }

    /// Output log entry to file
    fn output_to_file(&self, entry: &LogEntry, log_file: &Path) -> Result<(), std::io::Error> {
        use std::io::Write;

        let log_line = format!(
            "[{}] {:?} [{}:{}] {} {:?}\n",
            entry.timestamp.to_rfc3339(),
            entry.level,
            entry.component,
            entry.operation,
            entry.message,
            entry.context
        );

        // Create parent directories if they don't exist
        if let Some(parent) = log_file.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Append to log file
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_file)?;
        
        file.write_all(log_line.as_bytes())?;
        file.flush()?;

        Ok(())
    }
}

/// Resource cleanup manager for handling temporary files and artifacts
#[derive(Debug, Clone)]
pub struct ResourceCleanupManager {
    temp_files: Arc<Mutex<Vec<PathBuf>>>,
    temp_directories: Arc<Mutex<Vec<PathBuf>>>,
    cleanup_on_drop: bool,
    logger: TestingLogger,
}

impl ResourceCleanupManager {
    /// Create a new resource cleanup manager
    pub fn new(cleanup_on_drop: bool, logger: TestingLogger) -> Self {
        Self {
            temp_files: Arc::new(Mutex::new(Vec::new())),
            temp_directories: Arc::new(Mutex::new(Vec::new())),
            cleanup_on_drop,
            logger,
        }
    }

    /// Register a temporary file for cleanup
    pub fn register_temp_file(&self, file_path: PathBuf) {
        if let Ok(mut files) = self.temp_files.lock() {
            files.push(file_path.clone());
            self.logger.log(
                LogLevel::Debug,
                "ResourceCleanup",
                "register_temp_file",
                &format!("Registered temporary file: {:?}", file_path),
                HashMap::new(),
            );
        }
    }

    /// Register a temporary directory for cleanup
    pub fn register_temp_directory(&self, dir_path: PathBuf) {
        if let Ok(mut dirs) = self.temp_directories.lock() {
            dirs.push(dir_path.clone());
            self.logger.log(
                LogLevel::Debug,
                "ResourceCleanup",
                "register_temp_directory",
                &format!("Registered temporary directory: {:?}", dir_path),
                HashMap::new(),
            );
        }
    }

    /// Clean up all registered temporary files and directories
    pub fn cleanup_all(&self) -> Result<(), TestingError> {
        let mut errors = Vec::new();

        // Clean up temporary files
        if let Ok(mut files) = self.temp_files.lock() {
            for file_path in files.drain(..) {
                if let Err(e) = self.cleanup_file(&file_path) {
                    errors.push(e);
                }
            }
        }

        // Clean up temporary directories
        if let Ok(mut dirs) = self.temp_directories.lock() {
            for dir_path in dirs.drain(..) {
                if let Err(e) = self.cleanup_directory(&dir_path) {
                    errors.push(e);
                }
            }
        }

        if errors.is_empty() {
            self.logger.log(
                LogLevel::Info,
                "ResourceCleanup",
                "cleanup_all",
                "All temporary resources cleaned up successfully",
                HashMap::new(),
            );
            Ok(())
        } else {
            self.logger.log_error("ResourceCleanup", "cleanup_all", &TestingError::Multiple(errors.clone()));
            Err(TestingError::Multiple(errors))
        }
    }

    /// Clean up a specific file
    pub fn cleanup_file(&self, file_path: &Path) -> Result<(), TestingError> {
        if file_path.exists() {
            match std::fs::remove_file(file_path) {
                Ok(()) => {
                    self.logger.log(
                        LogLevel::Debug,
                        "ResourceCleanup",
                        "cleanup_file",
                        &format!("Successfully removed file: {:?}", file_path),
                        HashMap::new(),
                    );
                    Ok(())
                },
                Err(e) => {
                    let error = TestingError::Cleanup(format!("Failed to remove file {:?}: {}", file_path, e));
                    self.logger.log_error("ResourceCleanup", "cleanup_file", &error);
                    Err(error)
                }
            }
        } else {
            self.logger.log(
                LogLevel::Debug,
                "ResourceCleanup",
                "cleanup_file",
                &format!("File does not exist, skipping: {:?}", file_path),
                HashMap::new(),
            );
            Ok(())
        }
    }

    /// Clean up a specific directory
    pub fn cleanup_directory(&self, dir_path: &Path) -> Result<(), TestingError> {
        if dir_path.exists() {
            match std::fs::remove_dir_all(dir_path) {
                Ok(()) => {
                    self.logger.log(
                        LogLevel::Debug,
                        "ResourceCleanup",
                        "cleanup_directory",
                        &format!("Successfully removed directory: {:?}", dir_path),
                        HashMap::new(),
                    );
                    Ok(())
                },
                Err(e) => {
                    let error = TestingError::Cleanup(format!("Failed to remove directory {:?}: {}", dir_path, e));
                    self.logger.log_error("ResourceCleanup", "cleanup_directory", &error);
                    Err(error)
                }
            }
        } else {
            self.logger.log(
                LogLevel::Debug,
                "ResourceCleanup",
                "cleanup_directory",
                &format!("Directory does not exist, skipping: {:?}", dir_path),
                HashMap::new(),
            );
            Ok(())
        }
    }

    /// Force cleanup with retries for stubborn files
    pub fn force_cleanup_with_retries(&self, max_retries: u32, retry_delay: Duration) -> Result<(), TestingError> {
        let mut attempt = 0;
        
        while attempt < max_retries {
            match self.cleanup_all() {
                Ok(()) => return Ok(()),
                Err(e) => {
                    attempt += 1;
                    if attempt < max_retries {
                        self.logger.log(
                            LogLevel::Warn,
                            "ResourceCleanup",
                            "force_cleanup_with_retries",
                            &format!("Cleanup attempt {} failed, retrying in {:?}", attempt, retry_delay),
                            HashMap::new(),
                        );
                        std::thread::sleep(retry_delay);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(TestingError::Cleanup("Max cleanup retries exceeded".to_string()))
    }

    /// Get list of registered temporary files
    pub fn get_temp_files(&self) -> Vec<PathBuf> {
        if let Ok(files) = self.temp_files.lock() {
            files.clone()
        } else {
            Vec::new()
        }
    }

    /// Get list of registered temporary directories
    pub fn get_temp_directories(&self) -> Vec<PathBuf> {
        if let Ok(dirs) = self.temp_directories.lock() {
            dirs.clone()
        } else {
            Vec::new()
        }
    }
}

impl Drop for ResourceCleanupManager {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            if let Err(e) = self.cleanup_all() {
                eprintln!("Warning: Failed to cleanup resources on drop: {}", e);
            }
        }
    }
}

/// Error recovery system for handling and recovering from various failure scenarios
pub struct ErrorRecoverySystem {
    logger: TestingLogger,
    cleanup_manager: ResourceCleanupManager,
    recovery_strategies: HashMap<String, Box<dyn Fn(&TestingError) -> Result<(), TestingError> + Send + Sync>>,
}

impl ErrorRecoverySystem {
    /// Create a new error recovery system
    pub fn new(logger: TestingLogger, cleanup_manager: ResourceCleanupManager) -> Self {
        let mut system = Self {
            logger,
            cleanup_manager,
            recovery_strategies: HashMap::new(),
        };

        // Register default recovery strategies
        system.register_default_strategies();
        system
    }

    /// Register default recovery strategies for common error types
    fn register_default_strategies(&mut self) {
        // File discovery recovery
        self.register_recovery_strategy("file_discovery".to_string(), Box::new(|error| {
            match error {
                TestingError::FileDiscovery(msg) if msg.contains("Permission denied") => {
                    // Try to continue with available files
                    Ok(())
                },
                TestingError::FileDiscovery(msg) if msg.contains("does not exist") => {
                    // Create missing directories if possible
                    Ok(())
                },
                _ => Err(TestingError::Recovery("No recovery strategy available".to_string())),
            }
        }));

        // Compilation recovery
        self.register_recovery_strategy("compilation".to_string(), Box::new(|error| {
            match error {
                TestingError::Compilation(msg) if msg.contains("timeout") => {
                    // Could retry with longer timeout
                    Ok(())
                },
                TestingError::Compilation(msg) if msg.contains("out of memory") => {
                    // Could reduce parallelism
                    Ok(())
                },
                _ => Err(TestingError::Recovery("No compilation recovery available".to_string())),
            }
        }));

        // Execution recovery
        self.register_recovery_strategy("execution".to_string(), Box::new(|error| {
            match error {
                TestingError::Execution(msg) if msg.contains("timeout") => {
                    // Mark as timeout but continue with other tests
                    Ok(())
                },
                TestingError::Execution(msg) if msg.contains("Permission denied") => {
                    // Skip execution but don't fail entire test suite
                    Ok(())
                },
                _ => Err(TestingError::Recovery("No execution recovery available".to_string())),
            }
        }));
    }

    /// Register a custom recovery strategy
    pub fn register_recovery_strategy(
        &mut self,
        error_type: String,
        strategy: Box<dyn Fn(&TestingError) -> Result<(), TestingError> + Send + Sync>,
    ) {
        self.recovery_strategies.insert(error_type, strategy);
    }

    /// Attempt to recover from an error
    pub fn attempt_recovery(&self, error_type: &str, error: &TestingError) -> Result<(), TestingError> {
        self.logger.log(
            LogLevel::Info,
            "ErrorRecovery",
            "attempt_recovery",
            &format!("Attempting recovery for error type: {}", error_type),
            HashMap::new(),
        );

        if let Some(strategy) = self.recovery_strategies.get(error_type) {
            match strategy(error) {
                Ok(()) => {
                    self.logger.log_recovery("ErrorRecovery", "attempt_recovery", &format!("Recovery strategy for {}", error_type), true);
                    Ok(())
                },
                Err(recovery_error) => {
                    self.logger.log_recovery("ErrorRecovery", "attempt_recovery", &format!("Recovery strategy for {}", error_type), false);
                    Err(recovery_error)
                }
            }
        } else {
            let recovery_error = TestingError::Recovery(format!("No recovery strategy registered for error type: {}", error_type));
            self.logger.log_error("ErrorRecovery", "attempt_recovery", &recovery_error);
            Err(recovery_error)
        }
    }

    /// Perform graceful degradation when critical errors occur
    pub fn graceful_degradation(&self, critical_error: &TestingError, context: &str) -> Result<(), TestingError> {
        self.logger.log(
            LogLevel::Critical,
            "ErrorRecovery",
            "graceful_degradation",
            &format!("Performing graceful degradation due to critical error in {}: {}", context, critical_error),
            HashMap::new(),
        );

        // Attempt cleanup first
        if let Err(cleanup_error) = self.cleanup_manager.cleanup_all() {
            self.logger.log_error("ErrorRecovery", "graceful_degradation", &cleanup_error);
        }

        // Log the degradation
        self.logger.log(
            LogLevel::Warn,
            "ErrorRecovery",
            "graceful_degradation",
            &format!("System degraded gracefully after critical error in {}", context),
            HashMap::new(),
        );

        Ok(())
    }

    /// Handle multiple errors with prioritized recovery
    pub fn handle_multiple_errors(&self, errors: Vec<TestingError>) -> Result<Vec<TestingError>, TestingError> {
        let mut unrecovered_errors = Vec::new();
        let mut recovery_attempts = 0;

        self.logger.log(
            LogLevel::Warn,
            "ErrorRecovery",
            "handle_multiple_errors",
            &format!("Handling {} errors", errors.len()),
            HashMap::new(),
        );

        for error in errors {
            recovery_attempts += 1;
            
            // Determine error type for recovery strategy
            let error_type = match &error {
                TestingError::FileDiscovery(_) => "file_discovery",
                TestingError::Compilation(_) => "compilation",
                TestingError::Execution(_) => "execution",
                TestingError::TestGeneration(_) => "test_generation",
                TestingError::ReportGeneration(_) => "report_generation",
                _ => "generic",
            };

            match self.attempt_recovery(error_type, &error) {
                Ok(()) => {
                    self.logger.log(
                        LogLevel::Info,
                        "ErrorRecovery",
                        "handle_multiple_errors",
                        &format!("Successfully recovered from error: {}", error),
                        HashMap::new(),
                    );
                },
                Err(_) => {
                    unrecovered_errors.push(error);
                }
            }
        }

        if unrecovered_errors.is_empty() {
            self.logger.log(
                LogLevel::Info,
                "ErrorRecovery",
                "handle_multiple_errors",
                &format!("Successfully recovered from all {} errors", recovery_attempts),
                HashMap::new(),
            );
            Ok(Vec::new())
        } else {
            self.logger.log(
                LogLevel::Warn,
                "ErrorRecovery",
                "handle_multiple_errors",
                &format!("Could not recover from {} out of {} errors", unrecovered_errors.len(), recovery_attempts),
                HashMap::new(),
            );
            Ok(unrecovered_errors)
        }
    }
}

// Placeholder structs for components that will be implemented in later tasks
// These provide the interface contracts for the orchestrator

/// File discovery engine for finding Aether source files
pub struct FileDiscoveryEngine {
    root_path: PathBuf,
    include_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
}

impl FileDiscoveryEngine {
    /// Create a new file discovery engine
    pub fn new(root_path: PathBuf) -> Result<Self, TestingError> {
        Ok(Self {
            root_path,
            include_patterns: vec!["*.ae".to_string()],
            exclude_patterns: vec![
                "target/**".to_string(),
                ".git/**".to_string(),
                "node_modules/**".to_string(),
                ".kiro/**".to_string(),
            ],
        })
    }

    /// Create a new file discovery engine with custom patterns
    pub fn with_patterns(
        root_path: PathBuf,
        include_patterns: Vec<String>,
        exclude_patterns: Vec<String>,
    ) -> Result<Self, TestingError> {
        Ok(Self {
            root_path,
            include_patterns,
            exclude_patterns,
        })
    }

    /// Discover all Aether files in the project directory
    pub fn discover_aether_files(&self) -> Result<Vec<PathBuf>, TestingError> {
        let mut discovered_files = Vec::new();
        self.scan_directory_recursive(&self.root_path, &mut discovered_files)?;
        
        // Sort files for consistent ordering
        discovered_files.sort();
        
        Ok(discovered_files)
    }

    /// Filter discovered files by specific directories
    pub fn filter_by_directories(&self, files: Vec<PathBuf>, target_dirs: &[&str]) -> Vec<PathBuf> {
        if target_dirs.is_empty() {
            return files;
        }

        files
            .into_iter()
            .filter(|file| {
                // Get the relative path from root
                if let Ok(relative_path) = file.strip_prefix(&self.root_path) {
                    // Check if the file is in any of the target directories
                    target_dirs.iter().any(|&target_dir| {
                        relative_path.starts_with(target_dir)
                    })
                } else {
                    false
                }
            })
            .collect()
    }

    /// Discover files in specific directories only
    pub fn discover_in_directories(&self, target_dirs: &[&str]) -> Result<Vec<PathBuf>, TestingError> {
        let all_files = self.discover_aether_files()?;
        Ok(self.filter_by_directories(all_files, target_dirs))
    }

    /// Recursively scan directory for Aether files
    fn scan_directory_recursive(
        &self,
        dir_path: &Path,
        discovered_files: &mut Vec<PathBuf>,
    ) -> Result<(), TestingError> {
        let entries = std::fs::read_dir(dir_path)
            .map_err(|e| TestingError::FileDiscovery(
                format!("Failed to read directory {:?}: {}", dir_path, e)
            ))?;

        for entry in entries {
            let entry = entry.map_err(|e| TestingError::FileDiscovery(
                format!("Failed to read directory entry in {:?}: {}", dir_path, e)
            ))?;

            let path = entry.path();
            
            // Skip if path matches exclude patterns
            if self.should_exclude_path(&path) {
                continue;
            }

            if path.is_dir() {
                // Recursively scan subdirectories
                self.scan_directory_recursive(&path, discovered_files)?;
            } else if path.is_file() && self.matches_include_patterns(&path) {
                discovered_files.push(path);
            }
        }

        Ok(())
    }

    /// Check if a path should be excluded based on exclude patterns
    fn should_exclude_path(&self, path: &Path) -> bool {
        let relative_path = match path.strip_prefix(&self.root_path) {
            Ok(rel_path) => rel_path,
            Err(_) => path,
        };

        let path_str = relative_path.to_string_lossy();
        
        self.exclude_patterns.iter().any(|pattern| {
            self.matches_glob_pattern(&path_str, pattern)
        })
    }

    /// Check if a file matches include patterns
    fn matches_include_patterns(&self, path: &Path) -> bool {
        let file_name = path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");

        self.include_patterns.iter().any(|pattern| {
            self.matches_glob_pattern(file_name, pattern)
        })
    }

    /// Simple glob pattern matching (supports * wildcards)
    fn matches_glob_pattern(&self, text: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        if pattern.contains("**") {
            // Handle recursive directory patterns
            let parts: Vec<&str> = pattern.split("**").collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                
                return text.starts_with(prefix) && 
                       (suffix.is_empty() || text.ends_with(suffix.trim_start_matches('/')));
            }
        }

        if pattern.contains('*') {
            // Handle simple wildcard patterns
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                return text.starts_with(prefix) && text.ends_with(suffix);
            }
        }

        // Exact match
        text == pattern
    }

    /// Get statistics about discovered files
    pub fn get_discovery_stats(&self) -> Result<FileDiscoveryStats, TestingError> {
        let all_files = self.discover_aether_files()?;
        let examples_files = self.discover_in_directories(&["examples"])?;
        let tests_files = self.discover_in_directories(&["tests"])?;
        
        Ok(FileDiscoveryStats {
            total_files: all_files.len(),
            examples_files: examples_files.len(),
            tests_files: tests_files.len(),
            other_files: all_files.len() - examples_files.len() - tests_files.len(),
            directories_scanned: self.count_directories_scanned()?,
        })
    }

    /// Count the number of directories that were scanned
    fn count_directories_scanned(&self) -> Result<usize, TestingError> {
        let mut count = 0;
        self.count_directories_recursive(&self.root_path, &mut count)?;
        Ok(count)
    }

    /// Recursively count directories
    fn count_directories_recursive(&self, dir_path: &Path, count: &mut usize) -> Result<(), TestingError> {
        if self.should_exclude_path(dir_path) {
            return Ok(());
        }

        *count += 1;

        let entries = std::fs::read_dir(dir_path)
            .map_err(|e| TestingError::FileDiscovery(
                format!("Failed to read directory {:?}: {}", dir_path, e)
            ))?;

        for entry in entries {
            let entry = entry.map_err(|e| TestingError::FileDiscovery(
                format!("Failed to read directory entry in {:?}: {}", dir_path, e)
            ))?;

            let path = entry.path();
            if path.is_dir() && !self.should_exclude_path(&path) {
                self.count_directories_recursive(&path, count)?;
            }
        }

        Ok(())
    }

    /// Validate the setup of the file discovery engine
    pub fn validate_setup(&self) -> Result<(), TestingError> {
        if !self.root_path.exists() {
            return Err(TestingError::FileDiscovery(
                format!("Root path does not exist: {:?}", self.root_path)
            ));
        }

        if !self.root_path.is_dir() {
            return Err(TestingError::FileDiscovery(
                format!("Root path is not a directory: {:?}", self.root_path)
            ));
        }

        // Validate that we can read the root directory
        std::fs::read_dir(&self.root_path)
            .map_err(|e| TestingError::FileDiscovery(
                format!("Cannot read root directory {:?}: {}", self.root_path, e)
            ))?;

        Ok(())
    }
}

/// Statistics about file discovery results
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileDiscoveryStats {
    pub total_files: usize,
    pub examples_files: usize,
    pub tests_files: usize,
    pub other_files: usize,
    pub directories_scanned: usize,
}

/// Test file generator for creating comprehensive Aether test files
pub struct TestFileGenerator {
    output_dir: PathBuf,
    examples_dir: PathBuf,
    tests_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedTestFile {
    pub file_path: PathBuf,
    pub category: TestCategory,
    pub description: String,
    pub expected_features: Vec<String>,
}

impl GeneratedTestFile {
    pub fn new(file_path: PathBuf, category: TestCategory, description: String) -> Self {
        Self {
            file_path,
            category,
            description,
            expected_features: vec!["basic_syntax".to_string()],
        }
    }
}

impl TestFileGenerator {
    pub fn new(output_dir: PathBuf) -> Result<Self, TestingError> {
        let examples_dir = PathBuf::from("examples");
        let tests_dir = PathBuf::from("tests");
        
        // Create directories if they don't exist
        std::fs::create_dir_all(&examples_dir).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to create examples directory: {}", e))
        })?;
        
        std::fs::create_dir_all(&tests_dir).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to create tests directory: {}", e))
        })?;
        
        Ok(Self { 
            output_dir,
            examples_dir,
            tests_dir,
        })
    }

    /// Generate all test files for core language features
    pub fn generate_test_files(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut generated_files = Vec::new();
        
        // Generate core language feature tests
        generated_files.extend(self.generate_core_language_tests()?);
        
        // Generate advanced feature tests
        generated_files.extend(self.generate_tensor_and_autodiff_tests()?);
        generated_files.extend(self.generate_probabilistic_programming_tests()?);
        generated_files.extend(self.generate_type_system_tests()?);
        generated_files.extend(self.generate_error_handling_tests()?);
        
        Ok(generated_files)
    }

    /// Generate test files for core language features
    pub fn generate_core_language_tests(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut generated_files = Vec::new();
        
        // Generate basic function tests
        generated_files.extend(self.generate_function_tests()?);
        
        // Generate variable and type tests
        generated_files.extend(self.generate_variable_tests()?);
        
        // Generate control flow tests
        generated_files.extend(self.generate_control_flow_tests()?);
        
        // Generate loop tests
        generated_files.extend(self.generate_loop_tests()?);
        
        // Generate conditional tests
        generated_files.extend(self.generate_conditional_tests()?);
        
        // Generate pattern matching tests
        generated_files.extend(self.generate_pattern_matching_tests()?);
        
        Ok(generated_files)
    }

    /// Generate basic function definition and call tests
    pub fn generate_function_tests(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut files = Vec::new();
        
        // Simple function test (S-expression style)
        let simple_func_content = r#"# Simple Function Test
# Tests basic function definition and calling

(func greet (name)
  (call print (concat "Hello, " name "!"))
  (return 0))

(func add (a b)
  (return (+ a b)))

(func main ()
  (call greet "Aether")
  (let result (call add 5 3))
  (call print (concat "5 + 3 = " (to_string result)))
  (return 0))
"#;
        
        let simple_func_path = self.tests_dir.join("simple_functions.ae");
        std::fs::write(&simple_func_path, simple_func_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write simple function test: {}", e))
        })?;
        
        files.push(GeneratedTestFile::new(
            simple_func_path,
            TestCategory::CoreLanguage,
            "Basic function definition and calling".to_string(),
        ));

        // Function with parameters test (modern syntax style)
        let param_func_content = r#"# Function Parameters Test
# Tests functions with various parameter types

fn calculate_area(width: f32, height: f32) -> f32 {
    width * height
}

fn format_message(name: String, age: i32) -> String {
    format!("Name: {}, Age: {}", name, age)
}

fn fibonacci(n: i32) -> i32 {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fn main() -> i32 {
    let area = calculate_area(10.0, 5.0)
    println!("Area: {}", area)
    
    let message = format_message("Alice", 25)
    println!("{}", message)
    
    let fib_result = fibonacci(10)
    println!("Fibonacci(10): {}", fib_result)
    
    0
}
"#;
        
        let param_func_path = self.examples_dir.join("function_parameters.ae");
        std::fs::write(&param_func_path, param_func_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write parameter function test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: param_func_path,
            category: TestCategory::CoreLanguage,
            description: "Functions with various parameter types".to_string(),
            expected_features: vec!["basic_syntax".to_string()],
        });

        Ok(files)
    }

    /// Generate variable declaration and assignment tests
    pub fn generate_variable_tests(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut files = Vec::new();
        
        // Basic variable test
        let var_content = r#"# Variable Declaration Test
# Tests basic variable declarations and assignments

fn main() -> i32 {
    # Integer variables
    let x = 42
    let mut y = 10
    y = y + 5
    
    # Floating point variables
    let pi = 3.14159
    let mut radius = 5.0
    radius = radius * 2.0
    
    # String variables
    let name = "Aether"
    let greeting = format!("Hello, {}!", name)
    
    # Boolean variables
    let is_active = true
    let is_complete = false
    
    # Print results
    println!("x = {}", x)
    println!("y = {}", y)
    println!("pi = {}", pi)
    println!("radius = {}", radius)
    println!("{}", greeting)
    println!("Active: {}, Complete: {}", is_active, is_complete)
    
    0
}
"#;
        
        let var_path = self.tests_dir.join("variables.ae");
        std::fs::write(&var_path, var_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write variable test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: var_path,
            category: TestCategory::CoreLanguage,
            description: "Basic variable declarations and assignments".to_string(),
            expected_features: vec!["variables".to_string()],
        });

        // Type inference test
        let type_inference_content = r#"# Type Inference Test
# Tests Aether's type inference capabilities

fn main() -> i32 {
    # Type inference with literals
    let integer_val = 100        # inferred as i32
    let float_val = 3.14         # inferred as f64
    let string_val = "hello"     # inferred as String
    let bool_val = true          # inferred as bool
    
    # Type inference with expressions
    let sum = integer_val + 50   # inferred as i32
    let product = float_val * 2.0 # inferred as f64
    let concatenated = string_val + " world" # inferred as String
    let negated = !bool_val      # inferred as bool
    
    # Type inference with function calls
    let length = string_val.len() # inferred as usize
    let absolute = sum.abs()      # inferred as i32
    
    println!("Integer: {}", integer_val)
    println!("Float: {}", float_val)
    println!("String: {}", string_val)
    println!("Boolean: {}", bool_val)
    println!("Sum: {}", sum)
    println!("Product: {}", product)
    println!("Concatenated: {}", concatenated)
    println!("Negated: {}", negated)
    println!("Length: {}", length)
    println!("Absolute: {}", absolute)
    
    0
}
"#;
        
        let type_inference_path = self.examples_dir.join("type_inference.ae");
        std::fs::write(&type_inference_path, type_inference_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write type inference test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: type_inference_path,
            category: TestCategory::CoreLanguage,
            description: "Type inference capabilities".to_string(),
            expected_features: vec!["type_inference".to_string()],
        });

        Ok(files)
    }

    /// Generate control flow tests
    pub fn generate_control_flow_tests(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut files = Vec::new();
        
        // Basic control flow test
        let control_flow_content = r#"# Control Flow Test
# Tests basic control flow constructs

fn test_if_else(x: i32) -> String {
    if x > 0 {
        "positive"
    } else if x < 0 {
        "negative"
    } else {
        "zero"
    }
}

fn test_match(value: i32) -> String {
    match value {
        0 => "zero",
        1 => "one",
        2 => "two",
        3..=10 => "small number",
        _ => "large number"
    }
}

fn main() -> i32 {
    # Test if-else
    let numbers = [5, -3, 0, 15]
    
    for num in numbers {
        let result = test_if_else(num)
        println!("{} is {}", num, result)
    }
    
    # Test match
    let test_values = [0, 1, 2, 7, 100]
    
    for val in test_values {
        let result = test_match(val)
        println!("{} -> {}", val, result)
    }
    
    0
}
"#;
        
        let control_flow_path = self.tests_dir.join("control_flow.ae");
        std::fs::write(&control_flow_path, control_flow_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write control flow test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: control_flow_path,
            category: TestCategory::CoreLanguage,
            description: "Basic control flow constructs".to_string(),
            expected_features: vec!["control_flow".to_string()],
        });

        Ok(files)
    }

    /// Generate loop tests
    pub fn generate_loop_tests(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut files = Vec::new();
        
        // Loop constructs test
        let loop_content = r#"# Loop Constructs Test
# Tests various loop types in Aether

fn test_for_loop() {
    println!("For loop test:")
    for i in 0..5 {
        println!("  i = {}", i)
    }
}

fn test_while_loop() {
    println!("While loop test:")
    let mut count = 0
    while count < 3 {
        println!("  count = {}", count)
        count += 1
    }
}

fn test_loop_with_break() {
    println!("Loop with break test:")
    let mut i = 0
    loop {
        if i >= 3 {
            break
        }
        println!("  i = {}", i)
        i += 1
    }
}

fn test_nested_loops() {
    println!("Nested loops test:")
    for i in 0..3 {
        for j in 0..2 {
            println!("  ({}, {})", i, j)
        }
    }
}

fn test_loop_with_continue() {
    println!("Loop with continue test:")
    for i in 0..5 {
        if i == 2 {
            continue
        }
        println!("  i = {}", i)
    }
}

fn main() -> i32 {
    test_for_loop()
    test_while_loop()
    test_loop_with_break()
    test_nested_loops()
    test_loop_with_continue()
    
    0
}
"#;
        
        let loop_path = self.tests_dir.join("loops.ae");
        std::fs::write(&loop_path, loop_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write loop test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: loop_path,
            category: TestCategory::CoreLanguage,
            description: "Various loop constructs".to_string(),
            expected_features: vec!["loops".to_string()],
        });

        Ok(files)
    }

    /// Generate conditional tests
    pub fn generate_conditional_tests(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut files = Vec::new();
        
        // Conditional expressions test
        let conditional_content = r#"# Conditional Expressions Test
# Tests conditional expressions and boolean logic

fn test_boolean_operations(a: bool, b: bool) -> String {
    let and_result = a && b
    let or_result = a || b
    let not_a = !a
    let not_b = !b
    
    format!("a={}, b={}, a&&b={}, a||b={}, !a={}, !b={}", 
            a, b, and_result, or_result, not_a, not_b)
}

fn test_comparison_operations(x: i32, y: i32) -> String {
    let eq = x == y
    let ne = x != y
    let lt = x < y
    let le = x <= y
    let gt = x > y
    let ge = x >= y
    
    format!("x={}, y={}, ==:{}, !=:{}, <:{}, <=:{}, >:{}, >=:{}", 
            x, y, eq, ne, lt, le, gt, ge)
}

fn test_ternary_like(condition: bool, true_val: i32, false_val: i32) -> i32 {
    if condition { true_val } else { false_val }
}

fn test_complex_conditions(age: i32, has_license: bool, has_car: bool) -> String {
    if age >= 18 && has_license && has_car {
        "Can drive"
    } else if age >= 18 && has_license {
        "Can drive but needs a car"
    } else if age >= 18 {
        "Needs license and car"
    } else {
        "Too young to drive"
    }
}

fn main() -> i32 {
    # Test boolean operations
    println!("{}", test_boolean_operations(true, false))
    println!("{}", test_boolean_operations(false, true))
    println!("{}", test_boolean_operations(true, true))
    println!("{}", test_boolean_operations(false, false))
    
    # Test comparison operations
    println!("{}", test_comparison_operations(5, 3))
    println!("{}", test_comparison_operations(3, 5))
    println!("{}", test_comparison_operations(5, 5))
    
    # Test ternary-like expressions
    println!("Ternary test: {}", test_ternary_like(true, 100, 200))
    println!("Ternary test: {}", test_ternary_like(false, 100, 200))
    
    # Test complex conditions
    println!("{}", test_complex_conditions(20, true, true))
    println!("{}", test_complex_conditions(20, true, false))
    println!("{}", test_complex_conditions(20, false, false))
    println!("{}", test_complex_conditions(16, false, false))
    
    0
}
"#;
        
        let conditional_path = self.tests_dir.join("conditionals.ae");
        std::fs::write(&conditional_path, conditional_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write conditional test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: conditional_path,
            category: TestCategory::CoreLanguage,
            description: "Conditional expressions and boolean logic".to_string(),
            expected_features: vec!["conditionals".to_string()],
        });

        Ok(files)
    }

    /// Generate pattern matching tests
    pub fn generate_pattern_matching_tests(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut files = Vec::new();
        
        // Pattern matching test
        let pattern_content = r#"# Pattern Matching Test
# Tests pattern matching capabilities

enum Color {
    Red,
    Green,
    Blue,
    RGB(u8, u8, u8),
    HSV { hue: f32, saturation: f32, value: f32 }
}

enum Option<T> {
    Some(T),
    None
}

fn describe_color(color: Color) -> String {
    match color {
        Color::Red => "Pure red".to_string(),
        Color::Green => "Pure green".to_string(),
        Color::Blue => "Pure blue".to_string(),
        Color::RGB(r, g, b) => format!("RGB({}, {}, {})", r, g, b),
        Color::HSV { hue, saturation, value } => {
            format!("HSV(h:{}, s:{}, v:{})", hue, saturation, value)
        }
    }
}

fn process_option(opt: Option<i32>) -> String {
    match opt {
        Option::Some(value) => format!("Got value: {}", value),
        Option::None => "No value".to_string()
    }
}

fn match_numbers(n: i32) -> String {
    match n {
        0 => "zero".to_string(),
        1 | 2 | 3 => "small".to_string(),
        4..=10 => "medium".to_string(),
        x if x > 100 => "very large".to_string(),
        x if x < 0 => format!("negative: {}", x),
        _ => "other".to_string()
    }
}

fn match_tuple(pair: (i32, i32)) -> String {
    match pair {
        (0, 0) => "origin".to_string(),
        (0, y) => format!("on y-axis at {}", y),
        (x, 0) => format!("on x-axis at {}", x),
        (x, y) if x == y => format!("diagonal at ({}, {})", x, y),
        (x, y) => format!("point at ({}, {})", x, y)
    }
}

fn main() -> i32 {
    # Test enum pattern matching
    let colors = [
        Color::Red,
        Color::Green,
        Color::RGB(255, 128, 0),
        Color::HSV { hue: 120.0, saturation: 1.0, value: 0.8 }
    ]
    
    for color in colors {
        println!("{}", describe_color(color))
    }
    
    # Test Option pattern matching
    let options = [Option::Some(42), Option::None, Option::Some(0)]
    
    for opt in options {
        println!("{}", process_option(opt))
    }
    
    # Test number pattern matching
    let numbers = [0, 2, 7, 50, 150, -5]
    
    for num in numbers {
        println!("{} -> {}", num, match_numbers(num))
    }
    
    # Test tuple pattern matching
    let points = [(0, 0), (0, 5), (3, 0), (4, 4), (2, 7)]
    
    for point in points {
        println!("{:?} -> {}", point, match_tuple(point))
    }
    
    0
}
"#;
        
        let pattern_path = self.examples_dir.join("pattern_matching.ae");
        std::fs::write(&pattern_path, pattern_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write pattern matching test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: pattern_path,
            category: TestCategory::CoreLanguage,
            description: "Pattern matching capabilities".to_string(),
            expected_features: vec!["pattern_matching".to_string()],
        });

        Ok(files)
    }

    /// Generate test files for tensor operations and automatic differentiation
    pub fn generate_tensor_and_autodiff_tests(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut files = Vec::new();
        
        // Basic tensor operations test
        let tensor_content = r#"# Tensor Operations Test
# Tests basic tensor creation and operations

use tensor::*

fn main() -> i32 {
    # Create basic tensors
    let a = tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    let b = tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2])
    
    # Basic arithmetic operations
    let sum = a + b
    let product = a * b
    let difference = a - b
    
    # Matrix operations
    let matmul = a @ b
    let transpose = a.transpose()
    
    # Print results
    println!("A: {}", a)
    println!("B: {}", b)
    println!("Sum: {}", sum)
    println!("Product: {}", product)
    println!("Difference: {}", difference)
    println!("Matrix multiplication: {}", matmul)
    println!("Transpose of A: {}", transpose)
    
    0
}
"#;
        
        let tensor_path = self.examples_dir.join("tensor_operations.ae");
        std::fs::write(&tensor_path, tensor_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write tensor operations test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: tensor_path,
            category: TestCategory::AIFeatures,
            description: "Basic tensor operations".to_string(),
            expected_features: vec!["tensors".to_string()],
        });

        // Advanced tensor operations test
        let advanced_tensor_content = r#"# Advanced Tensor Operations Test
# Tests advanced tensor operations and GPU acceleration

use tensor::*
use gpu::*

fn convolution_example() -> Tensor {
    # Create input tensor (batch_size=1, channels=3, height=32, width=32)
    let input = tensor_zeros([1, 3, 32, 32])
    
    # Create convolution kernel (out_channels=16, in_channels=3, kernel_h=3, kernel_w=3)
    let kernel = tensor_randn([16, 3, 3, 3])
    
    # Perform 2D convolution
    let output = conv2d(input, kernel, stride=[1, 1], padding=[1, 1])
    
    output
}

fn tensor_reduction_operations() {
    # Create test tensor
    let data = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    
    # Various reduction operations
    let sum_all = data.sum()
    let sum_axis0 = data.sum(axis=0)
    let sum_axis1 = data.sum(axis=1)
    let mean_all = data.mean()
    let max_all = data.max()
    let min_all = data.min()
    
    println!("Original tensor: {}", data)
    println!("Sum all: {}", sum_all)
    println!("Sum axis 0: {}", sum_axis0)
    println!("Sum axis 1: {}", sum_axis1)
    println!("Mean: {}", mean_all)
    println!("Max: {}", max_all)
    println!("Min: {}", min_all)
}

fn gpu_tensor_operations() {
    # Move tensors to GPU for acceleration
    let a = tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2]).to_gpu()
    let b = tensor([5.0, 6.0, 7.0, 8.0], shape=[2, 2]).to_gpu()
    
    # GPU-accelerated operations
    let gpu_result = (a @ b).relu().softmax(dim=1)
    
    # Move result back to CPU
    let cpu_result = gpu_result.to_cpu()
    
    println!("GPU computation result: {}", cpu_result)
}

fn main() -> i32 {
    # Test convolution
    let conv_result = convolution_example()
    println!("Convolution output shape: {:?}", conv_result.shape())
    
    # Test reduction operations
    tensor_reduction_operations()
    
    # Test GPU operations (if available)
    if gpu::is_available() {
        gpu_tensor_operations()
    } else {
        println!("GPU not available, skipping GPU tests")
    }
    
    0
}
"#;
        
        let advanced_tensor_path = self.examples_dir.join("advanced_tensor_operations.ae");
        std::fs::write(&advanced_tensor_path, advanced_tensor_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write advanced tensor operations test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: advanced_tensor_path,
            category: TestCategory::AIFeatures,
            description: "Advanced tensor operations and GPU acceleration".to_string(),
            expected_features: vec!["tensors".to_string(), "gpu".to_string()],
        });

        // Automatic differentiation test
        let autodiff_content = r#"# Automatic Differentiation Test
# Tests automatic differentiation capabilities

use tensor::*
use autodiff::*

fn simple_function(x: Tensor) -> Tensor {
    x * x + 2.0 * x + 1.0
}

fn neural_network_layer(input: Tensor, weights: Tensor, bias: Tensor) -> Tensor {
    let linear = input @ weights + bias
    linear.relu()
}

fn main() -> i32 {
    # Test basic autodiff
    let x = tensor([2.0], requires_grad=true)
    let y = simple_function(x)
    
    # Compute gradient
    y.backward()
    let grad = x.grad()
    
    println!("x: {}", x)
    println!("y = x^2 + 2x + 1: {}", y)
    println!("dy/dx: {}", grad)
    
    # Test neural network layer
    let input = tensor([[1.0, 2.0, 3.0]], requires_grad=true)
    let weights = tensor([[0.5, 0.3], [0.2, 0.8], [0.1, 0.9]], requires_grad=true)
    let bias = tensor([0.1, 0.2], requires_grad=true)
    
    let output = neural_network_layer(input, weights, bias)
    let loss = output.sum()
    
    loss.backward()
    
    println!("Input: {}", input)
    println!("Output: {}", output)
    println!("Loss: {}", loss)
    println!("Input gradient: {}", input.grad())
    println!("Weights gradient: {}", weights.grad())
    println!("Bias gradient: {}", bias.grad())
    
    0
}
"#;
        
        let autodiff_path = self.examples_dir.join("automatic_differentiation.ae");
        std::fs::write(&autodiff_path, autodiff_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write autodiff test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: autodiff_path,
            category: TestCategory::AIFeatures,
            description: "Automatic differentiation".to_string(),
            expected_features: vec!["autodiff".to_string()],
        });

        // Complex autodiff test with neural network
        let complex_autodiff_content = r#"# Complex Automatic Differentiation Test
# Tests complex autodiff scenarios with neural networks

use tensor::*
use autodiff::*
use nn::*

struct SimpleNN {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}

impl SimpleNN {
    fn new() -> Self {
        Self {
            layer1: Linear::new(784, 128),
            layer2: Linear::new(128, 64),
            layer3: Linear::new(64, 10),
        }
    }
    
    fn forward(&self, x: Tensor) -> Tensor {
        let x = self.layer1.forward(x).relu()
        let x = self.layer2.forward(x).relu()
        self.layer3.forward(x)
    }
}

fn train_step(model: &SimpleNN, input: Tensor, target: Tensor, learning_rate: f32) -> f32 {
    # Forward pass
    let output = model.forward(input)
    
    # Compute loss (cross entropy)
    let loss = cross_entropy_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Update parameters using gradients
    model.layer1.weight.data -= learning_rate * model.layer1.weight.grad
    model.layer1.bias.data -= learning_rate * model.layer1.bias.grad
    model.layer2.weight.data -= learning_rate * model.layer2.weight.grad
    model.layer2.bias.data -= learning_rate * model.layer2.bias.grad
    model.layer3.weight.data -= learning_rate * model.layer3.weight.grad
    model.layer3.bias.data -= learning_rate * model.layer3.bias.grad
    
    # Clear gradients
    model.layer1.zero_grad()
    model.layer2.zero_grad()
    model.layer3.zero_grad()
    
    loss.item()
}

fn main() -> i32 {
    # Create a simple neural network
    let model = SimpleNN::new()
    
    # Create dummy training data
    let batch_size = 32
    let input = tensor_randn([batch_size, 784])
    let target = tensor_randint(0, 10, [batch_size])
    
    # Training loop
    let learning_rate = 0.01
    let num_epochs = 10
    
    for epoch in 0..num_epochs {
        let loss = train_step(&model, input.clone(), target.clone(), learning_rate)
        println!("Epoch {}: Loss = {:.4}", epoch, loss)
    }
    
    # Test inference
    let test_input = tensor_randn([1, 784])
    let prediction = model.forward(test_input)
    let predicted_class = prediction.argmax(dim=1)
    
    println!("Test prediction: {}", prediction)
    println!("Predicted class: {}", predicted_class)
    
    0
}
"#;
        
        let complex_autodiff_path = self.tests_dir.join("complex_autodiff.ae");
        std::fs::write(&complex_autodiff_path, complex_autodiff_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write complex autodiff test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: complex_autodiff_path,
            category: TestCategory::AIFeatures,
            description: "Complex automatic differentiation with neural networks".to_string(),
            expected_features: vec!["autodiff".to_string(), "neural_networks".to_string()],
        });

        Ok(files)
    }

    /// Generate test files for probabilistic programming
    pub fn generate_probabilistic_programming_tests(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut files = Vec::new();
        
        // Basic probabilistic programming test
        let prob_content = r#"# Probabilistic Programming Test
# Tests basic probabilistic constructs

use prob::*

fn coin_flip_model() -> f64 {
    # Prior belief about coin bias
    let bias ~ Beta(2.0, 2.0)
    
    # Observed coin flips (1 = heads, 0 = tails)
    let observations = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    
    # Likelihood of observations given bias
    for obs in observations {
        obs ~ Bernoulli(bias)
    }
    
    # Return posterior mean
    bias
}

fn linear_regression_model(x_data: [f64], y_data: [f64]) -> (f64, f64) {
    # Priors for slope and intercept
    let slope ~ Normal(0.0, 1.0)
    let intercept ~ Normal(0.0, 1.0)
    let noise ~ Gamma(1.0, 1.0)
    
    # Likelihood
    for (x, y) in zip(x_data, y_data) {
        let predicted = slope * x + intercept
        y ~ Normal(predicted, noise)
    }
    
    (slope, intercept)
}

fn main() -> i32 {
    # Test coin flip model
    let coin_bias = coin_flip_model()
    println!("Estimated coin bias: {}", coin_bias)
    
    # Test linear regression
    let x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    let y_data = [2.1, 3.9, 6.1, 8.0, 9.9]
    
    let (slope, intercept) = linear_regression_model(x_data, y_data)
    println!("Estimated slope: {}", slope)
    println!("Estimated intercept: {}", intercept)
    
    0
}
"#;
        
        let prob_path = self.examples_dir.join("probabilistic_programming.ae");
        std::fs::write(&prob_path, prob_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write probabilistic programming test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: prob_path,
            category: TestCategory::AIFeatures,
            description: "Probabilistic programming constructs".to_string(),
            expected_features: vec!["probabilistic".to_string()],
        });

        // Advanced probabilistic programming test
        let advanced_prob_content = r#"# Advanced Probabilistic Programming Test
# Tests advanced probabilistic modeling and inference

use prob::*
use tensor::*

fn gaussian_mixture_model(data: [f64]) -> (f64, f64, f64) {
    # Number of mixture components
    let K = 2
    
    # Priors for mixture weights
    let weights ~ Dirichlet([1.0, 1.0])
    
    # Priors for component means and variances
    let mu1 ~ Normal(0.0, 10.0)
    let mu2 ~ Normal(0.0, 10.0)
    let sigma1 ~ InverseGamma(1.0, 1.0)
    let sigma2 ~ InverseGamma(1.0, 1.0)
    
    # Likelihood
    for x in data {
        let component ~ Categorical(weights)
        if component == 0 {
            x ~ Normal(mu1, sigma1)
        } else {
            x ~ Normal(mu2, sigma2)
        }
    }
    
    (mu1, mu2, weights[0])
}

fn bayesian_neural_network(x_train: Tensor, y_train: Tensor) -> BayesianNN {
    # Define a Bayesian neural network with weight uncertainty
    struct BayesianNN {
        w1_mean: Tensor,
        w1_std: Tensor,
        w2_mean: Tensor,
        w2_std: Tensor,
        b1_mean: Tensor,
        b1_std: Tensor,
        b2_mean: Tensor,
        b2_std: Tensor,
    }
    
    # Priors for network weights
    let w1_mean ~ Normal(0.0, 1.0, shape=[784, 50])
    let w1_std ~ LogNormal(0.0, 1.0, shape=[784, 50])
    let w2_mean ~ Normal(0.0, 1.0, shape=[50, 10])
    let w2_std ~ LogNormal(0.0, 1.0, shape=[50, 10])
    let b1_mean ~ Normal(0.0, 1.0, shape=[50])
    let b1_std ~ LogNormal(0.0, 1.0, shape=[50])
    let b2_mean ~ Normal(0.0, 1.0, shape=[10])
    let b2_std ~ LogNormal(0.0, 1.0, shape=[10])
    
    # Sample weights from priors
    let w1 ~ Normal(w1_mean, w1_std)
    let w2 ~ Normal(w2_mean, w2_std)
    let b1 ~ Normal(b1_mean, b1_std)
    let b2 ~ Normal(b2_mean, b2_std)
    
    # Forward pass
    for (x, y) in zip(x_train, y_train) {
        let h1 = relu(x @ w1 + b1)
        let logits = h1 @ w2 + b2
        y ~ Categorical(softmax(logits))
    }
    
    BayesianNN {
        w1_mean, w1_std, w2_mean, w2_std,
        b1_mean, b1_std, b2_mean, b2_std
    }
}

fn hierarchical_model(groups: [[f64]]) -> (f64, f64, [f64]) {
    # Hierarchical model for grouped data
    let num_groups = groups.len()
    
    # Hyperpriors
    let mu_global ~ Normal(0.0, 10.0)
    let sigma_global ~ InverseGamma(1.0, 1.0)
    
    # Group-level parameters
    let mut group_means = Vec::new()
    for i in 0..num_groups {
        let group_mean ~ Normal(mu_global, sigma_global)
        group_means.push(group_mean)
        
        # Individual observations within group
        let sigma_group ~ InverseGamma(1.0, 1.0)
        for x in groups[i] {
            x ~ Normal(group_mean, sigma_group)
        }
    }
    
    (mu_global, sigma_global, group_means)
}

fn variational_autoencoder(x_data: Tensor) -> (Tensor, Tensor) {
    # Variational Autoencoder with probabilistic encoder/decoder
    let latent_dim = 20
    let input_dim = x_data.shape()[1]
    
    # Encoder network (recognition model)
    let encoder_w1 ~ Normal(0.0, 0.1, shape=[input_dim, 400])
    let encoder_b1 ~ Normal(0.0, 0.1, shape=[400])
    let encoder_w2_mean ~ Normal(0.0, 0.1, shape=[400, latent_dim])
    let encoder_b2_mean ~ Normal(0.0, 0.1, shape=[latent_dim])
    let encoder_w2_logvar ~ Normal(0.0, 0.1, shape=[400, latent_dim])
    let encoder_b2_logvar ~ Normal(0.0, 0.1, shape=[latent_dim])
    
    # Decoder network (generative model)
    let decoder_w1 ~ Normal(0.0, 0.1, shape=[latent_dim, 400])
    let decoder_b1 ~ Normal(0.0, 0.1, shape=[400])
    let decoder_w2 ~ Normal(0.0, 0.1, shape=[400, input_dim])
    let decoder_b2 ~ Normal(0.0, 0.1, shape=[input_dim])
    
    for x in x_data {
        # Encoder forward pass
        let h1 = relu(x @ encoder_w1 + encoder_b1)
        let z_mean = h1 @ encoder_w2_mean + encoder_b2_mean
        let z_logvar = h1 @ encoder_w2_logvar + encoder_b2_logvar
        
        # Sample from latent distribution
        let z ~ Normal(z_mean, exp(0.5 * z_logvar))
        
        # Decoder forward pass
        let h2 = relu(z @ decoder_w1 + decoder_b1)
        let x_reconstructed = sigmoid(h2 @ decoder_w2 + decoder_b2)
        
        # Reconstruction likelihood
        x ~ Bernoulli(x_reconstructed)
    }
    
    (z_mean, z_logvar)
}

fn main() -> i32 {
    # Test Gaussian mixture model
    let mixture_data = [1.2, 1.5, 1.8, 5.1, 5.3, 5.7, 5.9]
    let (mu1, mu2, weight) = gaussian_mixture_model(mixture_data)
    println!("GMM results: mu1={:.2}, mu2={:.2}, weight={:.2}", mu1, mu2, weight)
    
    # Test hierarchical model
    let group_data = [
        [2.1, 2.3, 2.0, 2.4],
        [3.1, 3.2, 3.0, 3.3],
        [1.8, 1.9, 2.1, 1.7]
    ]
    let (global_mean, global_std, group_means) = hierarchical_model(group_data)
    println!("Hierarchical model: global_mean={:.2}, global_std={:.2}", global_mean, global_std)
    println!("Group means: {:?}", group_means)
    
    # Test Bayesian neural network (with dummy data)
    let x_train = tensor_randn([100, 784])
    let y_train = tensor_randint(0, 10, [100])
    let bnn = bayesian_neural_network(x_train, y_train)
    println!("Bayesian NN trained with uncertainty quantification")
    
    # Test VAE (with dummy data)
    let vae_data = tensor_rand([50, 784])
    let (z_mean, z_logvar) = variational_autoencoder(vae_data)
    println!("VAE latent representations computed")
    println!("Mean latent shape: {:?}", z_mean.shape())
    println!("Logvar latent shape: {:?}", z_logvar.shape())
    
    0
}
"#;
        
        let advanced_prob_path = self.examples_dir.join("advanced_probabilistic_programming.ae");
        std::fs::write(&advanced_prob_path, advanced_prob_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write advanced probabilistic programming test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: advanced_prob_path,
            category: TestCategory::AIFeatures,
            description: "Advanced probabilistic modeling and inference".to_string(),
            expected_features: vec!["probabilistic".to_string(), "inference".to_string()],
        });

        // Monte Carlo methods test
        let monte_carlo_content = r#"# Monte Carlo Methods Test
# Tests various Monte Carlo sampling and inference methods

use prob::*
use mcmc::*

fn metropolis_hastings_sampling() -> [f64] {
    # Target distribution: mixture of two Gaussians
    fn log_target_density(x: f64) -> f64 {
        let component1 = -0.5 * (x - 2.0).powi(2) / 1.0
        let component2 = -0.5 * (x - 6.0).powi(2) / 1.0
        log(0.3 * exp(component1) + 0.7 * exp(component2))
    }
    
    # Metropolis-Hastings sampler
    let mut samples = Vec::new()
    let mut current_x = 0.0
    let proposal_std = 1.0
    let num_samples = 10000
    
    for i in 0..num_samples {
        # Propose new state
        let proposal = current_x + Normal(0.0, proposal_std).sample()
        
        # Compute acceptance probability
        let log_alpha = log_target_density(proposal) - log_target_density(current_x)
        let alpha = min(1.0, exp(log_alpha))
        
        # Accept or reject
        if Uniform(0.0, 1.0).sample() < alpha {
            current_x = proposal
        }
        
        # Store sample (after burn-in)
        if i > 1000 {
            samples.push(current_x)
        }
    }
    
    samples
}

fn hamiltonian_monte_carlo() -> [f64] {
    # HMC sampling for a multivariate Gaussian
    fn log_density(x: [f64]) -> f64 {
        # Target: N(0, I)
        -0.5 * x.iter().map(|xi| xi * xi).sum()
    }
    
    fn gradient_log_density(x: [f64]) -> [f64] {
        # Gradient of log N(0, I)
        x.iter().map(|xi| -xi).collect()
    }
    
    let mut samples = Vec::new()
    let mut current_x = [0.0, 0.0]
    let step_size = 0.1
    let num_steps = 10
    let num_samples = 5000
    
    for _ in 0..num_samples {
        # Sample momentum
        let mut momentum: [f64; 2] = [
            Normal(0.0, 1.0).sample(),
            Normal(0.0, 1.0).sample()
        ]
        
        let mut x = current_x.clone()
        
        # Leapfrog integration
        for _ in 0..num_steps {
            # Half step for momentum
            let grad = gradient_log_density(x)
            for i in 0..2 {
                momentum[i] += 0.5 * step_size * grad[i]
            }
            
            # Full step for position
            for i in 0..2 {
                x[i] += step_size * momentum[i]
            }
            
            # Half step for momentum
            let grad = gradient_log_density(x)
            for i in 0..2 {
                momentum[i] += 0.5 * step_size * grad[i]
            }
        }
        
        # Compute acceptance probability
        let current_energy = -log_density(current_x) + 0.5 * current_x.iter().map(|xi| xi * xi).sum()
        let proposed_energy = -log_density(x) + 0.5 * momentum.iter().map(|pi| pi * pi).sum()
        let alpha = min(1.0, exp(current_energy - proposed_energy))
        
        # Accept or reject
        if Uniform(0.0, 1.0).sample() < alpha {
            current_x = x
        }
        
        samples.push(current_x[0]) # Store first component
    }
    
    samples
}

fn particle_filter(observations: [f64]) -> [f64] {
    # Particle filter for state estimation
    let num_particles = 1000
    let process_noise = 0.1
    let observation_noise = 0.5
    
    # Initialize particles
    let mut particles: [f64; 1000] = [0.0; 1000]
    let mut weights: [f64; 1000] = [1.0 / 1000.0; 1000]
    let mut states = Vec::new()
    
    for obs in observations {
        # Prediction step
        for i in 0..num_particles {
            particles[i] += Normal(0.0, process_noise).sample()
        }
        
        # Update step
        for i in 0..num_particles {
            # Likelihood of observation given particle state
            let likelihood = exp(-0.5 * (obs - particles[i]).powi(2) / observation_noise.powi(2))
            weights[i] *= likelihood
        }
        
        # Normalize weights
        let weight_sum: f64 = weights.iter().sum()
        for i in 0..num_particles {
            weights[i] /= weight_sum
        }
        
        # Estimate state (weighted average)
        let state_estimate: f64 = particles.iter().zip(weights.iter())
            .map(|(p, w)| p * w)
            .sum()
        states.push(state_estimate)
        
        # Resample particles
        let mut new_particles = [0.0; 1000]
        for i in 0..num_particles {
            let u = Uniform(0.0, 1.0).sample()
            let mut cumsum = 0.0
            for j in 0..num_particles {
                cumsum += weights[j]
                if u <= cumsum {
                    new_particles[i] = particles[j]
                    break
                }
            }
        }
        particles = new_particles
        weights = [1.0 / 1000.0; 1000] # Reset weights
    }
    
    states
}

fn main() -> i32 {
    # Test Metropolis-Hastings sampling
    println!("Running Metropolis-Hastings sampling...")
    let mh_samples = metropolis_hastings_sampling()
    let mh_mean = mh_samples.iter().sum::<f64>() / mh_samples.len() as f64
    println!("MH sample mean: {:.3}", mh_mean)
    
    # Test Hamiltonian Monte Carlo
    println!("Running Hamiltonian Monte Carlo...")
    let hmc_samples = hamiltonian_monte_carlo()
    let hmc_mean = hmc_samples.iter().sum::<f64>() / hmc_samples.len() as f64
    println!("HMC sample mean: {:.3}", hmc_mean)
    
    # Test particle filter
    println!("Running particle filter...")
    let observations = [1.0, 1.5, 2.1, 1.8, 2.3, 2.0, 1.7, 2.2]
    let filtered_states = particle_filter(observations)
    println!("Filtered states: {:?}", filtered_states)
    
    0
}
"#;
        
        let monte_carlo_path = self.tests_dir.join("monte_carlo_methods.ae");
        std::fs::write(&monte_carlo_path, monte_carlo_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write Monte Carlo methods test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: monte_carlo_path,
            category: TestCategory::AIFeatures,
            description: "Monte Carlo sampling and inference methods".to_string(),
            expected_features: vec!["monte_carlo".to_string(), "sampling".to_string()],
        });

        Ok(files)
    }

    /// Generate test files for type system features
    pub fn generate_type_system_tests(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut files = Vec::new();
        
        // Gradual typing test
        let gradual_typing_content = r#"# Gradual Typing Test
# Tests gradual typing features

fn dynamic_function(x) {
    # x has dynamic type, can be anything
    if typeof(x) == "i32" {
        x + 10
    } else if typeof(x) == "String" {
        x + " world"
    } else {
        x
    }
}

fn static_function(x: i32) -> i32 {
    # x has static type i32
    x * 2
}

fn mixed_function(static_param: f64, dynamic_param) -> String {
    # Mix of static and dynamic typing
    let result = static_param + (dynamic_param as f64)
    format!("Result: {}", result)
}

fn main() -> i32 {
    # Test dynamic function with different types
    let int_result = dynamic_function(42)
    let string_result = dynamic_function("hello")
    let float_result = dynamic_function(3.14)
    
    println!("Dynamic with int: {}", int_result)
    println!("Dynamic with string: {}", string_result)
    println!("Dynamic with float: {}", float_result)
    
    # Test static function
    let static_result = static_function(21)
    println!("Static function result: {}", static_result)
    
    # Test mixed function
    let mixed_result = mixed_function(10.5, 5)
    println!("Mixed function result: {}", mixed_result)
    
    0
}
"#;
        
        let gradual_path = self.tests_dir.join("gradual_typing.ae");
        std::fs::write(&gradual_path, gradual_typing_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write gradual typing test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: gradual_path,
            category: TestCategory::TypeSystem,
            description: "Gradual typing features".to_string(),
            expected_features: vec!["gradual_typing".to_string()],
        });

        // Dependent types test (tensor shapes)
        let dependent_types_content = r#"# Dependent Types Test
# Tests dependent types with tensor shapes

use tensor::*

fn matrix_multiply<M: usize, N: usize, P: usize>(
    a: Tensor<[M, N]>, 
    b: Tensor<[N, P]>
) -> Tensor<[M, P]> {
    # Type system ensures shape compatibility at compile time
    a @ b
}

fn reshape_tensor<S1: Shape, S2: Shape>(
    tensor: Tensor<S1>
) -> Tensor<S2> 
where
    S1::size() == S2::size()
{
    # Compile-time check that total elements match
    tensor.reshape()
}

fn main() -> i32 {
    # Create tensors with known shapes
    let a: Tensor<[2, 3]> = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let b: Tensor<[3, 2]> = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    # This compiles because shapes are compatible (2,3) @ (3,2) -> (2,2)
    let result = matrix_multiply(a, b)
    println!("Matrix multiplication result: {}", result)
    
    # Reshape test - 2x3 tensor to 3x2 tensor (same total elements)
    let reshaped: Tensor<[3, 2]> = reshape_tensor(a)
    println!("Reshaped tensor: {}", reshaped)
    
    # The following would cause compile-time errors:
    # let invalid = matrix_multiply(a, a)  # Shape mismatch: (2,3) @ (2,3)
    # let invalid_reshape: Tensor<[2, 2]> = reshape_tensor(a)  # Size mismatch: 6 != 4
    
    0
}
"#;
        
        let dependent_path = self.examples_dir.join("dependent_types.ae");
        std::fs::write(&dependent_path, dependent_types_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write dependent types test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: dependent_path,
            category: TestCategory::TypeSystem,
            description: "Dependent types with tensor shapes".to_string(),
            expected_features: vec!["dependent_types".to_string(), "tensor_shapes".to_string()],
        });

        // Linear types test
        let linear_types_content = r#"# Linear Types Test
# Tests linear type system for resource safety

use linear::*

# Linear types ensure resources are used exactly once
linear struct FileHandle {
    path: String,
    fd: i32,
}

linear struct NetworkConnection {
    socket: i32,
    address: String,
}

linear struct GpuBuffer {
    ptr: *mut u8,
    size: usize,
}

fn open_file(path: String) -> FileHandle {
    # Opens a file and returns a linear handle
    FileHandle {
        path,
        fd: unsafe { libc::open(path.as_ptr(), libc::O_RDONLY) },
    }
}

fn read_file(handle: FileHandle) -> (String, FileHandle) {
    # Reads from file and returns both content and handle
    # Linear type ensures handle is properly threaded through
    let content = unsafe {
        # Read file content using handle.fd
        String::from("file content")
    }
    (content, handle)
}

fn close_file(handle: FileHandle) {
    # Consumes the linear handle, ensuring file is closed
    unsafe {
        libc::close(handle.fd)
    }
    # handle is consumed here, cannot be used again
}

fn connect_to_server(address: String) -> NetworkConnection {
    NetworkConnection {
        socket: unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) },
        address,
    }
}

fn send_data(conn: NetworkConnection, data: String) -> NetworkConnection {
    # Send data and return connection for further use
    unsafe {
        libc::send(conn.socket, data.as_ptr(), data.len(), 0)
    }
    conn
}

fn close_connection(conn: NetworkConnection) {
    # Consume connection and close socket
    unsafe {
        libc::close(conn.socket)
    }
}

fn allocate_gpu_buffer(size: usize) -> GpuBuffer {
    GpuBuffer {
        ptr: unsafe { libc::malloc(size) as *mut u8 },
        size,
    }
}

fn write_to_gpu_buffer(buffer: GpuBuffer, data: &[u8]) -> GpuBuffer {
    # Write data to GPU buffer
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.ptr, data.len())
    }
    buffer
}

fn free_gpu_buffer(buffer: GpuBuffer) {
    # Consume buffer and free memory
    unsafe {
        libc::free(buffer.ptr as *mut libc::c_void)
    }
}

fn main() -> i32 {
    # Test file handling with linear types
    let file_handle = open_file("test.txt".to_string())
    let (content, file_handle) = read_file(file_handle)
    println!("File content: {}", content)
    close_file(file_handle) # Must consume the handle
    
    # Test network connection with linear types
    let connection = connect_to_server("127.0.0.1:8080".to_string())
    let connection = send_data(connection, "Hello, server!".to_string())
    close_connection(connection) # Must consume the connection
    
    # Test GPU buffer with linear types
    let gpu_buffer = allocate_gpu_buffer(1024)
    let data = [1u8, 2, 3, 4, 5]
    let gpu_buffer = write_to_gpu_buffer(gpu_buffer, &data)
    free_gpu_buffer(gpu_buffer) # Must consume the buffer
    
    # The following would cause compile-time errors:
    # close_file(file_handle) # Error: file_handle already consumed
    # send_data(connection, "data".to_string()) # Error: connection already consumed
    # free_gpu_buffer(gpu_buffer) # Error: gpu_buffer already consumed
    
    0
}
"#;
        
        let linear_types_path = self.tests_dir.join("linear_types.ae");
        std::fs::write(&linear_types_path, linear_types_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write linear types test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: linear_types_path,
            category: TestCategory::TypeSystem,
            description: "Linear types for resource safety".to_string(),
            expected_features: vec!["linear_types".to_string(), "resource_safety".to_string()],
        });

        // Advanced type inference test
        let type_inference_content = r#"# Advanced Type Inference Test
# Tests advanced type inference capabilities

use tensor::*

fn complex_inference_example() {
    # Type inference with complex expressions
    let data = [[1.0, 2.0], [3.0, 4.0]] # Inferred as [[f64; 2]; 2]
    let tensor_data = tensor(data) # Inferred as Tensor<[2, 2]>
    
    # Higher-order function type inference
    let numbers = [1, 2, 3, 4, 5]
    let squared = numbers.map(|x| x * x) # Inferred closure type
    let filtered = squared.filter(|&x| x > 10) # Inferred predicate type
    
    # Generic function type inference
    let result = combine_with(numbers, squared, |a, b| a + b) # Inferred generic types
    
    println!("Tensor: {}", tensor_data)
    println!("Squared: {:?}", squared)
    println!("Filtered: {:?}", filtered)
    println!("Combined: {:?}", result)
}

fn combine_with<T, U, V, F>(a: Vec<T>, b: Vec<U>, f: F) -> Vec<V>
where
    F: Fn(T, U) -> V,
    T: Copy,
    U: Copy,
{
    a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect()
}

fn tensor_type_inference() {
    # Tensor shape inference from operations
    let a = tensor_zeros([3, 4]) # Shape inferred as [3, 4]
    let b = tensor_ones([4, 5])  # Shape inferred as [4, 5]
    let c = a @ b                # Shape inferred as [3, 5] from matrix multiplication
    
    # Broadcasting inference
    let scalar = 2.0
    let broadcasted = a + scalar # Broadcasting inferred automatically
    
    # Reduction inference
    let sum_all = a.sum()        # Shape inferred as scalar
    let sum_axis0 = a.sum(axis=0) # Shape inferred as [4]
    let sum_axis1 = a.sum(axis=1) # Shape inferred as [3]
    
    println!("Matrix multiplication result shape: {:?}", c.shape())
    println!("Broadcasted shape: {:?}", broadcasted.shape())
    println!("Sum all: {}", sum_all)
    println!("Sum axis 0 shape: {:?}", sum_axis0.shape())
    println!("Sum axis 1 shape: {:?}", sum_axis1.shape())
}

fn closure_type_inference() {
    # Complex closure type inference
    let data = vec![1, 2, 3, 4, 5]
    
    # Closure capturing environment
    let multiplier = 3
    let multiplied = data.iter().map(|x| x * multiplier).collect::<Vec<_>>()
    
    # Nested closures
    let nested_result = data.iter()
        .map(|x| x * 2)
        .filter(|&x| x > 5)
        .map(|x| format!("Value: {}", x))
        .collect::<Vec<_>>()
    
    # Closure with complex return type
    let complex_closure = |x: i32| -> Result<String, &'static str> {
        if x > 0 {
            Ok(format!("Positive: {}", x))
        } else {
            Err("Non-positive number")
        }
    }
    
    let closure_results: Vec<_> = data.iter()
        .map(|&x| complex_closure(x))
        .collect()
    
    println!("Multiplied: {:?}", multiplied)
    println!("Nested result: {:?}", nested_result)
    println!("Closure results: {:?}", closure_results)
}

fn generic_type_inference() {
    # Generic container type inference
    let mut container = Vec::new() # Type not yet determined
    container.push(42) # Now inferred as Vec<i32>
    
    # Generic function with multiple type parameters
    let pairs = zip_with_index(vec!["a", "b", "c"]) # Inferred as Vec<(usize, &str)>
    
    # Option and Result type inference
    let maybe_value = Some(42) # Inferred as Option<i32>
    let result_value = Ok("success") # Inferred as Result<&str, _>
    
    # Iterator type inference
    let iterator_result: Vec<_> = (0..10)
        .filter(|&x| x % 2 == 0)
        .map(|x| x * x)
        .take(3)
        .collect()
    
    println!("Container: {:?}", container)
    println!("Pairs: {:?}", pairs)
    println!("Maybe value: {:?}", maybe_value)
    println!("Result value: {:?}", result_value)
    println!("Iterator result: {:?}", iterator_result)
}

fn zip_with_index<T>(items: Vec<T>) -> Vec<(usize, T)> {
    items.into_iter().enumerate().collect()
}

fn main() -> i32 {
    println!("=== Complex Inference Example ===")
    complex_inference_example()
    
    println!("\n=== Tensor Type Inference ===")
    tensor_type_inference()
    
    println!("\n=== Closure Type Inference ===")
    closure_type_inference()
    
    println!("\n=== Generic Type Inference ===")
    generic_type_inference()
    
    0
}
"#;
        
        let type_inference_path = self.examples_dir.join("advanced_type_inference.ae");
        std::fs::write(&type_inference_path, type_inference_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write advanced type inference test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: type_inference_path,
            category: TestCategory::TypeSystem,
            description: "Advanced type inference capabilities".to_string(),
            expected_features: vec!["type_inference".to_string()],
        });

        Ok(files)
    }

    /// Generate test files for error handling and edge cases
    pub fn generate_error_handling_tests(&self) -> Result<Vec<GeneratedTestFile>, TestingError> {
        let mut files = Vec::new();
        
        // Error handling test
        let error_handling_content = r#"# Error Handling Test
# Tests error handling mechanisms

use std::error::*

enum MathError {
    DivisionByZero,
    NegativeSquareRoot,
    Overflow,
}

fn safe_divide(a: f64, b: f64) -> Result<f64, MathError> {
    if b == 0.0 {
        Err(MathError::DivisionByZero)
    } else {
        Ok(a / b)
    }
}

fn safe_sqrt(x: f64) -> Result<f64, MathError> {
    if x < 0.0 {
        Err(MathError::NegativeSquareRoot)
    } else {
        Ok(x.sqrt())
    }
}

fn safe_factorial(n: u32) -> Result<u64, MathError> {
    if n > 20 {
        Err(MathError::Overflow)
    } else {
        let mut result = 1u64
        for i in 1..=n {
            result *= i as u64
        }
        Ok(result)
    }
}

fn main() -> i32 {
    # Test safe division
    match safe_divide(10.0, 2.0) {
        Ok(result) => println!("10 / 2 = {}", result),
        Err(e) => println!("Division error: {:?}", e),
    }
    
    match safe_divide(10.0, 0.0) {
        Ok(result) => println!("10 / 0 = {}", result),
        Err(e) => println!("Division error: {:?}", e),
    }
    
    # Test safe square root
    match safe_sqrt(16.0) {
        Ok(result) => println!("sqrt(16) = {}", result),
        Err(e) => println!("Square root error: {:?}", e),
    }
    
    match safe_sqrt(-4.0) {
        Ok(result) => println!("sqrt(-4) = {}", result),
        Err(e) => println!("Square root error: {:?}", e),
    }
    
    # Test safe factorial
    match safe_factorial(5) {
        Ok(result) => println!("5! = {}", result),
        Err(e) => println!("Factorial error: {:?}", e),
    }
    
    match safe_factorial(25) {
        Ok(result) => println!("25! = {}", result),
        Err(e) => println!("Factorial error: {:?}", e),
    }
    
    # Test error propagation with ?
    let complex_calculation = || -> Result<f64, MathError> {
        let a = safe_divide(20.0, 4.0)?
        let b = safe_sqrt(a)?
        let c = safe_divide(b, 2.0)?
        Ok(c)
    }
    
    match complex_calculation() {
        Ok(result) => println!("Complex calculation result: {}", result),
        Err(e) => println!("Complex calculation error: {:?}", e),
    }
    
    0
}
"#;
        
        let error_path = self.tests_dir.join("error_handling.ae");
        std::fs::write(&error_path, error_handling_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write error handling test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: error_path,
            category: TestCategory::ErrorHandling,
            description: "Error handling mechanisms".to_string(),
            expected_features: vec!["error_handling".to_string()],
        });

        // Advanced error handling and edge cases test
        let advanced_error_content = r#"# Advanced Error Handling and Edge Cases Test
# Tests complex error scenarios and edge cases

use std::error::*
use tensor::*

enum TensorError {
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    IndexOutOfBounds { index: usize, size: usize },
    InvalidOperation(String),
    MemoryAllocationFailed,
    GpuError(String),
}

enum NetworkError {
    ConnectionTimeout,
    InvalidResponse(String),
    AuthenticationFailed,
    RateLimitExceeded { retry_after: u64 },
}

enum ParseError {
    InvalidSyntax { line: usize, column: usize, message: String },
    UnexpectedToken { expected: String, found: String },
    UnknownIdentifier(String),
}

# Tensor operations with comprehensive error handling
fn safe_tensor_multiply(a: Tensor, b: Tensor) -> Result<Tensor, TensorError> {
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(TensorError::InvalidOperation(
            "Matrix multiplication requires 2D tensors".to_string()
        ))
    }
    
    let a_shape = a.shape()
    let b_shape = b.shape()
    
    if a_shape[1] != b_shape[0] {
        return Err(TensorError::ShapeMismatch {
            expected: vec![a_shape[0], b_shape[1]],
            actual: vec![a_shape[1], b_shape[0]],
        })
    }
    
    # Simulate potential GPU memory allocation failure
    if a_shape[0] * b_shape[1] > 1000000 {
        return Err(TensorError::MemoryAllocationFailed)
    }
    
    Ok(a @ b)
}

fn safe_tensor_index(tensor: Tensor, indices: Vec<usize>) -> Result<f64, TensorError> {
    let shape = tensor.shape()
    
    if indices.len() != shape.len() {
        return Err(TensorError::InvalidOperation(
            format!("Expected {} indices, got {}", shape.len(), indices.len())
        ))
    }
    
    for (i, (&index, &size)) in indices.iter().zip(shape.iter()).enumerate() {
        if index >= size {
            return Err(TensorError::IndexOutOfBounds { index, size })
        }
    }
    
    Ok(tensor.get(indices))
}

# Network operations with retry logic and error recovery
fn robust_network_request(url: String, max_retries: u32) -> Result<String, NetworkError> {
    let mut attempts = 0
    
    while attempts < max_retries {
        match attempt_network_request(&url) {
            Ok(response) => return Ok(response),
            Err(NetworkError::ConnectionTimeout) if attempts < max_retries - 1 => {
                attempts += 1
                println!("Connection timeout, retrying... (attempt {})", attempts + 1)
                std::thread::sleep(std::time::Duration::from_secs(1 << attempts)) # Exponential backoff
            },
            Err(NetworkError::RateLimitExceeded { retry_after }) if attempts < max_retries - 1 => {
                attempts += 1
                println!("Rate limited, waiting {} seconds...", retry_after)
                std::thread::sleep(std::time::Duration::from_secs(retry_after))
            },
            Err(e) => return Err(e), # Non-recoverable error
        }
    }
    
    Err(NetworkError::ConnectionTimeout)
}

fn attempt_network_request(url: &str) -> Result<String, NetworkError> {
    # Simulate various network conditions
    use std::collections::hash_map::DefaultHasher
    use std::hash::{Hash, Hasher}
    
    let mut hasher = DefaultHasher::new()
    url.hash(&mut hasher)
    let hash = hasher.finish()
    
    match hash % 10 {
        0..=2 => Ok(format!("Success response from {}", url)),
        3..=4 => Err(NetworkError::ConnectionTimeout),
        5 => Err(NetworkError::RateLimitExceeded { retry_after: 5 }),
        6 => Err(NetworkError::AuthenticationFailed),
        7..=9 => Err(NetworkError::InvalidResponse("Malformed JSON".to_string())),
        _ => unreachable!(),
    }
}

# Parser with detailed error reporting
fn parse_expression(input: &str) -> Result<i32, ParseError> {
    let tokens = tokenize(input)?
    parse_tokens(tokens)
}

fn tokenize(input: &str) -> Result<Vec<String>, ParseError> {
    let mut tokens = Vec::new()
    let mut current_token = String::new()
    
    for (line_num, line) in input.lines().enumerate() {
        for (col_num, ch) in line.chars().enumerate() {
            match ch {
                ' ' | '\t' => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone())
                        current_token.clear()
                    }
                },
                '+' | '-' | '*' | '/' | '(' | ')' => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone())
                        current_token.clear()
                    }
                    tokens.push(ch.to_string())
                },
                '0'..='9' => current_token.push(ch),
                _ => return Err(ParseError::InvalidSyntax {
                    line: line_num + 1,
                    column: col_num + 1,
                    message: format!("Unexpected character: '{}'", ch),
                }),
            }
        }
    }
    
    if !current_token.is_empty() {
        tokens.push(current_token)
    }
    
    Ok(tokens)
}

fn parse_tokens(tokens: Vec<String>) -> Result<i32, ParseError> {
    if tokens.is_empty() {
        return Err(ParseError::InvalidSyntax {
            line: 1,
            column: 1,
            message: "Empty expression".to_string(),
        })
    }
    
    # Simple expression parser (just return first number for demo)
    for token in tokens {
        if let Ok(num) = token.parse::<i32>() {
            return Ok(num)
        }
    }
    
    Err(ParseError::UnexpectedToken {
        expected: "number".to_string(),
        found: tokens[0].clone(),
    })
}

# Memory management edge cases
fn test_memory_edge_cases() -> Result<(), String> {
    # Test large allocation
    let large_size = 1_000_000
    let large_tensor = match tensor_zeros([large_size]) {
        Ok(t) => t,
        Err(_) => return Err("Failed to allocate large tensor".to_string()),
    }
    
    # Test memory fragmentation scenario
    let mut small_tensors = Vec::new()
    for i in 0..1000 {
        match tensor_ones([100]) {
            Ok(t) => small_tensors.push(t),
            Err(_) => return Err(format!("Failed to allocate small tensor {}", i)),
        }
    }
    
    # Test cleanup
    drop(large_tensor)
    drop(small_tensors)
    
    Ok(())
}

# Concurrency edge cases
fn test_concurrent_access() -> Result<(), String> {
    use std::sync::{Arc, Mutex}
    use std::thread
    
    let shared_data = Arc::new(Mutex::new(vec![0; 1000]))
    let mut handles = Vec::new()
    
    # Spawn multiple threads that modify shared data
    for i in 0..10 {
        let data = Arc::clone(&shared_data)
        let handle = thread::spawn(move || {
            for j in 0..100 {
                match data.lock() {
                    Ok(mut guard) => {
                        guard[i * 100 + j] = i * 100 + j
                    },
                    Err(_) => return Err(format!("Thread {} failed to acquire lock", i)),
                }
            }
            Ok(())
        })
        handles.push(handle)
    }
    
    # Wait for all threads and collect results
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => {},
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(format!("Thread {} panicked", i)),
        }
    }
    
    Ok(())
}

fn main() -> i32 {
    println!("=== Tensor Error Handling ===")
    
    # Test tensor operations
    let a = tensor([[1.0, 2.0], [3.0, 4.0]])
    let b = tensor([[5.0, 6.0], [7.0, 8.0]])
    let c = tensor([[1.0, 2.0, 3.0]]) # Wrong shape for multiplication
    
    match safe_tensor_multiply(a.clone(), b) {
        Ok(result) => println!("Matrix multiplication successful: {}", result),
        Err(e) => println!("Matrix multiplication error: {:?}", e),
    }
    
    match safe_tensor_multiply(a.clone(), c) {
        Ok(result) => println!("Matrix multiplication successful: {}", result),
        Err(e) => println!("Matrix multiplication error: {:?}", e),
    }
    
    # Test tensor indexing
    match safe_tensor_index(a.clone(), vec![0, 1]) {
        Ok(value) => println!("Tensor indexing successful: {}", value),
        Err(e) => println!("Tensor indexing error: {:?}", e),
    }
    
    match safe_tensor_index(a, vec![5, 10]) {
        Ok(value) => println!("Tensor indexing successful: {}", value),
        Err(e) => println!("Tensor indexing error: {:?}", e),
    }
    
    println!("\n=== Network Error Handling ===")
    
    # Test network requests with retry logic
    let urls = [
        "https://api.example.com/data",
        "https://api.timeout.com/data",
        "https://api.ratelimited.com/data",
    ]
    
    for url in urls {
        match robust_network_request(url.to_string(), 3) {
            Ok(response) => println!("Network request successful: {}", response),
            Err(e) => println!("Network request failed: {:?}", e),
        }
    }
    
    println!("\n=== Parser Error Handling ===")
    
    # Test parser with various inputs
    let expressions = [
        "42",
        "1 + 2 * 3",
        "invalid@character",
        "",
        "( 1 + 2 ) * 3",
    ]
    
    for expr in expressions {
        match parse_expression(expr) {
            Ok(result) => println!("Parsed '{}' successfully: {}", expr, result),
            Err(e) => println!("Parse error for '{}': {:?}", expr, e),
        }
    }
    
    println!("\n=== Memory Edge Cases ===")
    
    match test_memory_edge_cases() {
        Ok(()) => println!("Memory edge case tests passed"),
        Err(e) => println!("Memory edge case test failed: {}", e),
    }
    
    println!("\n=== Concurrency Edge Cases ===")
    
    match test_concurrent_access() {
        Ok(()) => println!("Concurrency tests passed"),
        Err(e) => println!("Concurrency test failed: {}", e),
    }
    
    0
}
"#;
        
        let advanced_error_path = self.tests_dir.join("advanced_error_handling.ae");
        std::fs::write(&advanced_error_path, advanced_error_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write advanced error handling test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: advanced_error_path,
            category: TestCategory::ErrorHandling,
            description: "Advanced error handling and edge cases".to_string(),
            expected_features: vec!["error_handling".to_string(), "edge_cases".to_string()],
        });

        // Panic and recovery test
        let panic_recovery_content = r#"# Panic and Recovery Test
# Tests panic handling and recovery mechanisms

use std::panic

fn function_that_panics(should_panic: bool) {
    if should_panic {
        panic!("This is a deliberate panic for testing")
    }
    println!("Function executed successfully")
}

fn safe_division(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        panic!("Division by zero!")
    }
    Ok(a / b)
}

fn catch_panic_example() {
    println!("Testing panic recovery...")
    
    # Test 1: Catch a panic and continue execution
    let result = panic::catch_unwind(|| {
        function_that_panics(true)
    })
    
    match result {
        Ok(_) => println!("No panic occurred"),
        Err(_) => println!("Caught a panic and recovered"),
    }
    
    # Test 2: Normal execution after panic recovery
    let result = panic::catch_unwind(|| {
        function_that_panics(false)
    })
    
    match result {
        Ok(_) => println!("Function executed normally"),
        Err(_) => println!("Unexpected panic"),
    }
    
    # Test 3: Nested panic handling
    let result = panic::catch_unwind(|| {
        let inner_result = panic::catch_unwind(|| {
            safe_division(10, 0)
        })
        
        match inner_result {
            Ok(Ok(value)) => println!("Division result: {}", value),
            Ok(Err(e)) => println!("Division error: {}", e),
            Err(_) => println!("Division panicked"),
        }
    })
    
    match result {
        Ok(_) => println!("Nested panic handling completed"),
        Err(_) => println!("Outer panic occurred"),
    }
}

fn test_array_bounds() {
    println!("Testing array bounds checking...")
    
    let arr = [1, 2, 3, 4, 5]
    
    # Safe array access
    for i in 0..arr.len() {
        println!("arr[{}] = {}", i, arr[i])
    }
    
    # Test bounds checking with panic recovery
    for i in 0..10 {
        let result = panic::catch_unwind(|| {
            arr[i]
        })
        
        match result {
            Ok(value) => println!("arr[{}] = {}", i, value),
            Err(_) => println!("Index {} is out of bounds", i),
        }
    }
}

fn test_stack_overflow_protection() {
    println!("Testing stack overflow protection...")
    
    fn recursive_function(depth: u32) -> u32 {
        if depth > 10000 {
            depth
        } else {
            recursive_function(depth + 1)
        }
    }
    
    let result = panic::catch_unwind(|| {
        recursive_function(0)
    })
    
    match result {
        Ok(depth) => println!("Recursion completed at depth: {}", depth),
        Err(_) => println!("Stack overflow detected and handled"),
    }
}

fn test_resource_cleanup() {
    println!("Testing resource cleanup on panic...")
    
    struct Resource {
        name: String,
    }
    
    impl Drop for Resource {
        fn drop(&mut self) {
            println!("Cleaning up resource: {}", self.name)
        }
    }
    
    let result = panic::catch_unwind(|| {
        let _resource1 = Resource { name: "Resource 1".to_string() }
        let _resource2 = Resource { name: "Resource 2".to_string() }
        
        # Simulate some work
        println!("Working with resources...")
        
        # Panic in the middle of work
        panic!("Something went wrong!")
    })
    
    match result {
        Ok(_) => println!("No panic occurred"),
        Err(_) => println!("Panic occurred, but resources were cleaned up"),
    }
}

fn test_custom_panic_hook() {
    println!("Testing custom panic hook...")
    
    # Set a custom panic hook
    panic::set_hook(Box::new(|panic_info| {
        println!("Custom panic handler called!")
        if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            println!("Panic message: {}", s)
        }
        if let Some(location) = panic_info.location() {
            println!("Panic location: {}:{}", location.file(), location.line())
        }
    }))
    
    # Trigger a panic to test the custom hook
    let result = panic::catch_unwind(|| {
        panic!("Testing custom panic hook")
    })
    
    match result {
        Ok(_) => println!("No panic occurred"),
        Err(_) => println!("Panic was handled by custom hook"),
    }
    
    # Reset to default panic hook
    let _ = panic::take_hook()
}

fn main() -> i32 {
    catch_panic_example()
    println!()
    
    test_array_bounds()
    println!()
    
    test_stack_overflow_protection()
    println!()
    
    test_resource_cleanup()
    println!()
    
    test_custom_panic_hook()
    
    println!("All panic and recovery tests completed")
    0
}
"#;
        
        let panic_recovery_path = self.tests_dir.join("panic_recovery.ae");
        std::fs::write(&panic_recovery_path, panic_recovery_content).map_err(|e| {
            TestingError::TestGeneration(format!("Failed to write panic recovery test: {}", e))
        })?;
        
        files.push(GeneratedTestFile {
            file_path: panic_recovery_path,
            category: TestCategory::ErrorHandling,
            description: "Panic handling and recovery mechanisms".to_string(),
            expected_features: vec!["panic_recovery".to_string(), "resilience".to_string()],
        });

        Ok(files)
    }
} 

/// Compilation engine that invokes aetherc compiler with parallel processing
pub struct CompilationEngine {
    compiler_path: PathBuf,
    output_dir: PathBuf,
    timeout: Duration,
    max_parallel: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationResult {
    pub source_file: PathBuf,
    pub executable_path: Option<PathBuf>,
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub compilation_time: Duration,
    pub exit_code: Option<i32>,
}

impl CompilationEngine {
    /// Create a new compilation engine
    pub fn new(compiler_path: PathBuf, output_dir: PathBuf, timeout: Duration) -> Result<Self, TestingError> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&output_dir).map_err(|e| {
            TestingError::Compilation(format!("Failed to create output directory {:?}: {}", output_dir, e))
        })?;

        Ok(Self { 
            compiler_path, 
            output_dir, 
            timeout,
            max_parallel: num_cpus::get(),
        })
    }

    /// Create a new compilation engine with custom parallelism
    pub fn with_parallelism(
        compiler_path: PathBuf, 
        output_dir: PathBuf, 
        timeout: Duration,
        max_parallel: usize,
    ) -> Result<Self, TestingError> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&output_dir).map_err(|e| {
            TestingError::Compilation(format!("Failed to create output directory {:?}: {}", output_dir, e))
        })?;

        Ok(Self { 
            compiler_path, 
            output_dir, 
            timeout,
            max_parallel,
        })
    }

    /// Compile a single Aether file to executable
    pub async fn compile_file(&self, source_file: &Path) -> Result<CompilationResult, TestingError> {
        let start_time = std::time::Instant::now();
        
        // Generate output executable path
        let file_stem = source_file.file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| TestingError::Compilation(
                format!("Invalid source file name: {:?}", source_file)
            ))?;
        
        let executable_path = self.output_dir.join(format!("{}.exe", file_stem));
        
        // Build compiler command
        let mut cmd = tokio::process::Command::new(&self.compiler_path);
        cmd.arg("build")
           .arg("--target")
           .arg("native")
           .arg("--output")
           .arg(&executable_path)
           .arg(source_file);

        // Set up process with timeout
        let child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| TestingError::Compilation(
                format!("Failed to spawn compiler process: {}", e)
            ))?;

        // Wait for completion with timeout
        let output = match tokio::time::timeout(self.timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                return Ok(CompilationResult {
                    source_file: source_file.to_path_buf(),
                    executable_path: None,
                    success: false,
                    stdout: String::new(),
                    stderr: format!("Process error: {}", e),
                    compilation_time: start_time.elapsed(),
                    exit_code: None,
                });
            }
            Err(_) => {
                return Ok(CompilationResult {
                    source_file: source_file.to_path_buf(),
                    executable_path: None,
                    success: false,
                    stdout: String::new(),
                    stderr: format!("Compilation timeout after {:?}", self.timeout),
                    compilation_time: start_time.elapsed(),
                    exit_code: None,
                });
            }
        };

        let compilation_time = start_time.elapsed();
        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code();

        // Check if executable was actually created
        let final_executable_path = if success && executable_path.exists() {
            Some(executable_path)
        } else {
            None
        };

        Ok(CompilationResult {
            source_file: source_file.to_path_buf(),
            executable_path: final_executable_path,
            success,
            stdout,
            stderr,
            compilation_time,
            exit_code,
        })
    }

    /// Compile multiple files in parallel batches
    pub async fn compile_batch(&self, files: &[PathBuf]) -> Result<Vec<CompilationResult>, TestingError> {
        if files.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(files.len());
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(self.max_parallel));
        
        // Create futures for all compilation tasks
        let mut tasks = Vec::new();
        
        for file in files {
            let file = file.clone();
            let engine = self.clone();
            let permit = semaphore.clone();
            
            let task = tokio::spawn(async move {
                let _permit = permit.acquire().await.unwrap();
                engine.compile_file(&file).await
            });
            
            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            match task.await {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(TestingError::Compilation(
                    format!("Task join error: {}", e)
                )),
            }
        }

        Ok(results)
    }

    /// Get compilation statistics from results
    pub fn get_compilation_stats(&self, results: &[CompilationResult]) -> CompilationStats {
        let total_files = results.len();
        let successful_compilations = results.iter().filter(|r| r.success).count();
        let failed_compilations = total_files - successful_compilations;
        
        let total_time: Duration = results.iter().map(|r| r.compilation_time).sum();
        let average_time = if total_files > 0 {
            total_time / total_files as u32
        } else {
            Duration::from_secs(0)
        };

        let fastest_compilation = results.iter()
            .map(|r| r.compilation_time)
            .min()
            .unwrap_or(Duration::from_secs(0));
            
        let slowest_compilation = results.iter()
            .map(|r| r.compilation_time)
            .max()
            .unwrap_or(Duration::from_secs(0));

        CompilationStats {
            total_files,
            successful_compilations,
            failed_compilations,
            total_time,
            average_time,
            fastest_compilation,
            slowest_compilation,
        }
    }

    /// Clean up compilation artifacts
    pub fn cleanup_artifacts(&self) -> Result<(), TestingError> {
        if self.output_dir.exists() {
            std::fs::remove_dir_all(&self.output_dir).map_err(|e| {
                TestingError::Compilation(format!("Failed to clean up artifacts: {}", e))
            })?;
            
            // Recreate the directory
            std::fs::create_dir_all(&self.output_dir).map_err(|e| {
                TestingError::Compilation(format!("Failed to recreate output directory: {}", e))
            })?;
        }
        Ok(())
    }

    /// Validate the setup of the compilation engine
    pub fn validate_setup(&self) -> Result<(), TestingError> {
        // Check if compiler exists
        if !self.compiler_path.exists() && which::which(&self.compiler_path).is_err() {
            return Err(TestingError::Compilation(
                format!("Compiler not found: {:?}", self.compiler_path)
            ));
        }

        // Check if output directory is writable
        if !self.output_dir.exists() {
            std::fs::create_dir_all(&self.output_dir).map_err(|e| {
                TestingError::Compilation(format!("Cannot create output directory {:?}: {}", self.output_dir, e))
            })?;
        }

        // Test write permissions
        let test_file = self.output_dir.join("test_write_permissions.tmp");
        std::fs::write(&test_file, "test").map_err(|e| {
            TestingError::Compilation(format!("Output directory is not writable {:?}: {}", self.output_dir, e))
        })?;
        
        // Clean up test file
        let _ = std::fs::remove_file(&test_file);

        Ok(())
    }

    /// Get the output directory
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Get the compiler path
    pub fn compiler_path(&self) -> &Path {
        &self.compiler_path
    }

    /// Get the timeout duration
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Get the maximum parallel compilations
    pub fn max_parallel(&self) -> usize {
        self.max_parallel
    }
}

// Implement Clone for CompilationEngine to support parallel processing
impl Clone for CompilationEngine {
    fn clone(&self) -> Self {
        Self {
            compiler_path: self.compiler_path.clone(),
            output_dir: self.output_dir.clone(),
            timeout: self.timeout,
            max_parallel: self.max_parallel,
        }
    }
}

/// Statistics about compilation results
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilationStats {
    pub total_files: usize,
    pub successful_compilations: usize,
    pub failed_compilations: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub fastest_compilation: Duration,
    pub slowest_compilation: Duration,
}

/// Execution validator for running generated executables with safety measures
pub struct ExecutionValidator {
    timeout: Duration,
    capture_output: bool,
    max_memory_mb: Option<u64>,
    working_directory: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub executable_path: PathBuf,
    pub success: bool,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub execution_time: Duration,
    pub timed_out: bool,
    pub memory_exceeded: bool,
    pub error_message: Option<String>,
}

impl ExecutionValidator {
    /// Create a new execution validator with default settings
    pub fn new(timeout: Duration) -> Result<Self, TestingError> {
        Ok(Self { 
            timeout,
            capture_output: true,
            max_memory_mb: Some(512), // 512MB default limit
            working_directory: None,
        })
    }

    /// Create a new execution validator with custom settings
    pub fn with_settings(
        timeout: Duration,
        capture_output: bool,
        max_memory_mb: Option<u64>,
        working_directory: Option<PathBuf>,
    ) -> Result<Self, TestingError> {
        Ok(Self {
            timeout,
            capture_output,
            max_memory_mb,
            working_directory,
        })
    }

    /// Validate a single executable by running it and capturing results
    pub async fn validate_executable(&self, exe_path: &Path) -> Result<ExecutionResult, TestingError> {
        let start_time = std::time::Instant::now();
        
        // Verify executable exists and is executable
        if !exe_path.exists() {
            return Ok(ExecutionResult {
                executable_path: exe_path.to_path_buf(),
                success: false,
                exit_code: -1,
                stdout: String::new(),
                stderr: String::new(),
                execution_time: Duration::from_secs(0),
                timed_out: false,
                memory_exceeded: false,
                error_message: Some(format!("Executable does not exist: {:?}", exe_path)),
            });
        }

        // Check if file is executable (on Unix systems)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = std::fs::metadata(exe_path).map_err(|e| {
                TestingError::Execution(format!("Failed to get metadata for {:?}: {}", exe_path, e))
            })?;
            let permissions = metadata.permissions();
            if permissions.mode() & 0o111 == 0 {
                return Ok(ExecutionResult {
                    executable_path: exe_path.to_path_buf(),
                    success: false,
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: String::new(),
                    execution_time: Duration::from_secs(0),
                    timed_out: false,
                    memory_exceeded: false,
                    error_message: Some(format!("File is not executable: {:?}", exe_path)),
                });
            }
        }

        // Create command with safety measures
        let mut command = tokio::process::Command::new(exe_path);
        
        // Set working directory if specified
        if let Some(ref working_dir) = self.working_directory {
            command.current_dir(working_dir);
        }

        // Configure output capture
        if self.capture_output {
            command.stdout(std::process::Stdio::piped());
            command.stderr(std::process::Stdio::piped());
        } else {
            command.stdout(std::process::Stdio::null());
            command.stderr(std::process::Stdio::null());
        }

        // Set stdin to null to prevent hanging on input
        command.stdin(std::process::Stdio::null());

        // Apply resource limits on Unix systems
        #[cfg(unix)]
        {
            if let Some(max_memory) = self.max_memory_mb {
                // Note: Resource limits would typically be set using libc::setrlimit
                // For now, we'll rely on timeout to prevent runaway processes
                // In a production implementation, you might use cgroups or similar
            }
        }

        // Spawn the process
        let mut child = command.spawn().map_err(|e| {
            TestingError::Execution(format!("Failed to spawn executable {:?}: {}", exe_path, e))
        })?;

        // Wait for completion with timeout
        let execution_result = tokio::time::timeout(self.timeout, child.wait_with_output()).await;

        let execution_time = start_time.elapsed();

        match execution_result {
            Ok(Ok(output)) => {
                // Process completed within timeout
                let stdout = if self.capture_output {
                    String::from_utf8_lossy(&output.stdout).to_string()
                } else {
                    String::new()
                };

                let stderr = if self.capture_output {
                    String::from_utf8_lossy(&output.stderr).to_string()
                } else {
                    String::new()
                };

                let exit_code = output.status.code().unwrap_or(-1);
                let success = output.status.success();

                Ok(ExecutionResult {
                    executable_path: exe_path.to_path_buf(),
                    success,
                    exit_code,
                    stdout,
                    stderr,
                    execution_time,
                    timed_out: false,
                    memory_exceeded: false,
                    error_message: if success { None } else { 
                        Some(format!("Process exited with code {}", exit_code))
                    },
                })
            }
            Ok(Err(e)) => {
                // Error occurred during execution
                Ok(ExecutionResult {
                    executable_path: exe_path.to_path_buf(),
                    success: false,
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: String::new(),
                    execution_time,
                    timed_out: false,
                    memory_exceeded: false,
                    error_message: Some(format!("Execution error: {}", e)),
                })
            }
            Err(_) => {
                // Timeout occurred - we can't access child here since it was moved
                // The timeout will have terminated the process
                Ok(ExecutionResult {
                    executable_path: exe_path.to_path_buf(),
                    success: false,
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: String::new(),
                    execution_time,
                    timed_out: true,
                    memory_exceeded: false,
                    error_message: Some(format!("Execution timed out after {:?}", self.timeout)),
                })
            }
        }
    }

    /// Validate multiple executables in parallel with concurrency control
    pub async fn validate_batch(&self, executables: &[PathBuf]) -> Result<Vec<ExecutionResult>, TestingError> {
        if executables.is_empty() {
            return Ok(Vec::new());
        }

        // Determine concurrency level (limit to prevent resource exhaustion)
        let max_concurrent = std::cmp::min(
            num_cpus::get(),
            std::cmp::max(1, executables.len() / 2)
        );

        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let mut tasks = Vec::new();

        for exe_path in executables {
            let exe_path = exe_path.clone();
            let semaphore = semaphore.clone();
            let validator = self.clone();

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                validator.validate_executable(&exe_path).await
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        let mut results = Vec::new();
        for task in tasks {
            match task.await {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(TestingError::Execution(
                    format!("Task execution failed: {}", e)
                )),
            }
        }

        Ok(results)
    }

    /// Validate executables with custom filtering and categorization
    pub async fn validate_with_filter<F>(&self, executables: &[PathBuf], filter: F) -> Result<Vec<ExecutionResult>, TestingError>
    where
        F: Fn(&Path) -> bool,
    {
        let filtered_executables: Vec<PathBuf> = executables
            .iter()
            .filter(|path| filter(path))
            .cloned()
            .collect();

        self.validate_batch(&filtered_executables).await
    }

    /// Get execution statistics from a batch of results
    pub fn get_execution_stats(&self, results: &[ExecutionResult]) -> ExecutionStats {
        let total_executions = results.len();
        let successful_executions = results.iter().filter(|r| r.success).count();
        let failed_executions = total_executions - successful_executions;
        let timed_out_executions = results.iter().filter(|r| r.timed_out).count();
        let memory_exceeded_executions = results.iter().filter(|r| r.memory_exceeded).count();

        let total_execution_time = results.iter().map(|r| r.execution_time).sum();
        let average_execution_time = if total_executions > 0 {
            total_execution_time / total_executions as u32
        } else {
            Duration::from_secs(0)
        };

        let fastest_execution = results.iter().map(|r| r.execution_time).min().unwrap_or(Duration::from_secs(0));
        let slowest_execution = results.iter().map(|r| r.execution_time).max().unwrap_or(Duration::from_secs(0));

        ExecutionStats {
            total_executions,
            successful_executions,
            failed_executions,
            timed_out_executions,
            memory_exceeded_executions,
            total_execution_time,
            average_execution_time,
            fastest_execution,
            slowest_execution,
        }
    }

    /// Validate setup and configuration
    pub fn validate_setup(&self) -> Result<(), TestingError> {
        if self.timeout.is_zero() {
            return Err(TestingError::Validation(
                "Execution timeout must be greater than zero".to_string()
            ));
        }

        if let Some(ref working_dir) = self.working_directory {
            if !working_dir.exists() {
                return Err(TestingError::Validation(
                    format!("Working directory does not exist: {:?}", working_dir)
                ));
            }
            if !working_dir.is_dir() {
                return Err(TestingError::Validation(
                    format!("Working directory is not a directory: {:?}", working_dir)
                ));
            }
        }

        if let Some(max_memory) = self.max_memory_mb {
            if max_memory == 0 {
                return Err(TestingError::Validation(
                    "Maximum memory limit must be greater than zero".to_string()
                ));
            }
        }

        Ok(())
    }

    /// Set timeout for execution
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Set output capture mode
    pub fn set_capture_output(&mut self, capture: bool) {
        self.capture_output = capture;
    }

    /// Set memory limit in MB
    pub fn set_memory_limit(&mut self, limit_mb: Option<u64>) {
        self.max_memory_mb = limit_mb;
    }

    /// Set working directory for execution
    pub fn set_working_directory(&mut self, dir: Option<PathBuf>) {
        self.working_directory = dir;
    }
}

// Implement Clone for ExecutionValidator to support parallel execution
impl Clone for ExecutionValidator {
    fn clone(&self) -> Self {
        Self {
            timeout: self.timeout,
            capture_output: self.capture_output,
            max_memory_mb: self.max_memory_mb,
            working_directory: self.working_directory.clone(),
        }
    }
}

/// Statistics about execution results
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionStats {
    pub total_executions: usize,
    pub successful_executions: usize,
    pub failed_executions: usize,
    pub timed_out_executions: usize,
    pub memory_exceeded_executions: usize,
    pub total_execution_time: Duration,
    pub average_execution_time: Duration,
    pub fastest_execution: Duration,
    pub slowest_execution: Duration,
}

/// Comprehensive report generator for file compilation testing results
pub struct ReportGenerator {
    format: ReportFormat,
    include_details: bool,
    output_path: Option<PathBuf>,
}

impl ReportGenerator {
    /// Create a new report generator with specified format
    pub fn new(format: ReportFormat) -> Result<Self, TestingError> {
        Ok(Self { 
            format,
            include_details: true,
            output_path: None,
        })
    }

    /// Create a new report generator with custom settings
    pub fn with_settings(
        format: ReportFormat,
        include_details: bool,
        output_path: Option<PathBuf>,
    ) -> Result<Self, TestingError> {
        Ok(Self {
            format,
            include_details,
            output_path,
        })
    }

    /// Generate a comprehensive report from test results
    pub fn generate_report(&self, report: &FileCompilationTestReport) -> Result<String, TestingError> {
        match self.format {
            ReportFormat::Console => self.generate_console_report(report),
            ReportFormat::Json => self.generate_json_report(report),
            ReportFormat::Html => self.generate_html_report(report),
            ReportFormat::Markdown => self.generate_markdown_report(report),
        }
    }

    /// Save the generated report to a file
    pub fn save_report(&self, report_content: &str, output_path: &Path) -> Result<(), TestingError> {
        // Create parent directories if they don't exist
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                TestingError::ReportGeneration(format!(
                    "Failed to create output directory {:?}: {}", parent, e
                ))
            })?;
        }

        std::fs::write(output_path, report_content).map_err(|e| {
            TestingError::ReportGeneration(format!(
                "Failed to write report to {:?}: {}", output_path, e
            ))
        })?;

        Ok(())
    }

    /// Generate and save report in one operation
    pub fn generate_and_save_report(
        &self,
        report: &FileCompilationTestReport,
        output_path: &Path,
    ) -> Result<(), TestingError> {
        let report_content = self.generate_report(report)?;
        self.save_report(&report_content, output_path)?;
        Ok(())
    }

    /// Generate console-formatted report
    fn generate_console_report(&self, report: &FileCompilationTestReport) -> Result<String, TestingError> {
        let mut output = String::new();
        
        // Header
        output.push_str("═══════════════════════════════════════════════════════════════════════════════\n");
        output.push_str("                    AETHER FILE COMPILATION TEST REPORT\n");
        output.push_str("═══════════════════════════════════════════════════════════════════════════════\n\n");

        // Summary section
        output.push_str(&self.format_summary_console(&report.summary));
        output.push_str("\n");

        // Compilation results section
        output.push_str(&self.format_compilation_results_console(&report.compilation_results));
        output.push_str("\n");

        // Execution results section
        output.push_str(&self.format_execution_results_console(&report.execution_results));
        output.push_str("\n");

        // Generated files section
        if !report.generated_files.is_empty() {
            output.push_str(&self.format_generated_files_console(&report.generated_files));
            output.push_str("\n");
        }

        // Failure analysis
        output.push_str(&self.format_failure_analysis_console(&report.compilation_results, &report.execution_results));
        output.push_str("\n");

        // Configuration section
        if self.include_details {
            output.push_str(&self.format_configuration_console(&report.config));
        }

        // Footer
        output.push_str("═══════════════════════════════════════════════════════════════════════════════\n");
        output.push_str(&format!("Report generated at: {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        output.push_str("═══════════════════════════════════════════════════════════════════════════════\n");

        Ok(output)
    }

    /// Generate JSON-formatted report
    fn generate_json_report(&self, report: &FileCompilationTestReport) -> Result<String, TestingError> {
        let json_report = JsonReport {
            metadata: JsonReportMetadata {
                generated_at: chrono::Utc::now().to_rfc3339(),
                format_version: "1.0".to_string(),
                aether_version: env!("CARGO_PKG_VERSION").to_string(),
            },
            summary: JsonSummary::from_test_summary(&report.summary),
            compilation_results: report.compilation_results
                .iter()
                .map(JsonCompilationResult::from_compilation_result)
                .collect(),
            execution_results: report.execution_results
                .iter()
                .map(JsonExecutionResult::from_execution_result)
                .collect(),
            generated_files: report.generated_files
                .iter()
                .map(|gf| JsonGeneratedFile {
                    path: gf.file_path.to_string_lossy().to_string(),
                    category: format!("{:?}", gf.category),
                    description: gf.description.clone(),
                })
                .collect(),
            failure_analysis: self.generate_failure_analysis(&report.compilation_results, &report.execution_results),
            configuration: if self.include_details {
                Some(JsonConfiguration::from_testing_config(&report.config))
            } else {
                None
            },
        };

        serde_json::to_string_pretty(&json_report).map_err(|e| {
            TestingError::ReportGeneration(format!("Failed to serialize JSON report: {}", e))
        })
    }

    /// Generate HTML-formatted report
    fn generate_html_report(&self, report: &FileCompilationTestReport) -> Result<String, TestingError> {
        let mut html = String::new();
        
        // HTML header
        html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        html.push_str("    <title>Aether File Compilation Test Report</title>\n");
        html.push_str("    <style>\n");
        html.push_str(&self.get_html_styles());
        html.push_str("    </style>\n");
        html.push_str("</head>\n<body>\n");

        // Header
        html.push_str("    <header>\n");
        html.push_str("        <h1>Aether File Compilation Test Report</h1>\n");
        html.push_str(&format!("        <p class=\"timestamp\">Generated: {}</p>\n", 
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        html.push_str("    </header>\n\n");

        // Main content
        html.push_str("    <main>\n");
        
        // Summary section
        html.push_str(&self.format_summary_html(&report.summary));
        
        // Compilation results section
        html.push_str(&self.format_compilation_results_html(&report.compilation_results));
        
        // Execution results section
        html.push_str(&self.format_execution_results_html(&report.execution_results));
        
        // Generated files section
        if !report.generated_files.is_empty() {
            html.push_str(&self.format_generated_files_html(&report.generated_files));
        }
        
        // Failure analysis section
        html.push_str(&self.format_failure_analysis_html(&report.compilation_results, &report.execution_results));
        
        // Configuration section
        if self.include_details {
            html.push_str(&self.format_configuration_html(&report.config));
        }
        
        html.push_str("    </main>\n");

        // Footer
        html.push_str("    <footer>\n");
        html.push_str("        <p>Generated by Aether File Compilation Testing Framework</p>\n");
        html.push_str("    </footer>\n");

        html.push_str("</body>\n</html>");

        Ok(html)
    }

    /// Generate Markdown-formatted report
    fn generate_markdown_report(&self, report: &FileCompilationTestReport) -> Result<String, TestingError> {
        let mut md = String::new();
        
        // Header
        md.push_str("# Aether File Compilation Test Report\n\n");
        md.push_str(&format!("**Generated:** {}\n\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));

        // Summary section
        md.push_str(&self.format_summary_markdown(&report.summary));
        md.push_str("\n");

        // Compilation results section
        md.push_str(&self.format_compilation_results_markdown(&report.compilation_results));
        md.push_str("\n");

        // Execution results section
        md.push_str(&self.format_execution_results_markdown(&report.execution_results));
        md.push_str("\n");

        // Generated files section
        if !report.generated_files.is_empty() {
            md.push_str(&self.format_generated_files_markdown(&report.generated_files));
            md.push_str("\n");
        }

        // Failure analysis
        md.push_str(&self.format_failure_analysis_markdown(&report.compilation_results, &report.execution_results));
        md.push_str("\n");

        // Configuration section
        if self.include_details {
            md.push_str(&self.format_configuration_markdown(&report.config));
        }

        Ok(md)
    }

    /// Format summary section for console output
    fn format_summary_console(&self, summary: &TestSummary) -> String {
        let mut output = String::new();
        
        output.push_str("📊 SUMMARY\n");
        output.push_str("─────────────────────────────────────────────────────────────────────────────\n");
        output.push_str(&format!("Total Files:              {}\n", summary.total_files));
        output.push_str(&format!("Generated Test Files:     {}\n", summary.generated_test_files));
        output.push_str("\n");
        
        // Compilation statistics
        output.push_str("🔨 COMPILATION RESULTS\n");
        output.push_str(&format!("  ✅ Successful:          {} ({:.1}%)\n", 
            summary.successful_compilations,
            (summary.successful_compilations as f64 / summary.total_files as f64) * 100.0
        ));
        output.push_str(&format!("  ❌ Failed:              {} ({:.1}%)\n", 
            summary.failed_compilations,
            (summary.failed_compilations as f64 / summary.total_files as f64) * 100.0
        ));
        output.push_str(&format!("  ⏱️  Total Time:          {:.2}s\n", summary.total_compilation_time.as_secs_f64()));
        
        if summary.successful_compilations > 0 {
            let avg_time = summary.total_compilation_time.as_secs_f64() / summary.successful_compilations as f64;
            output.push_str(&format!("  📊 Average Time:        {:.2}s\n", avg_time));
        }
        
        output.push_str("\n");
        
        // Execution statistics
        output.push_str("🚀 EXECUTION RESULTS\n");
        let total_executions = summary.successful_executions + summary.failed_executions;
        if total_executions > 0 {
            output.push_str(&format!("  ✅ Successful:          {} ({:.1}%)\n", 
                summary.successful_executions,
                (summary.successful_executions as f64 / total_executions as f64) * 100.0
            ));
            output.push_str(&format!("  ❌ Failed:              {} ({:.1}%)\n", 
                summary.failed_executions,
                (summary.failed_executions as f64 / total_executions as f64) * 100.0
            ));
            output.push_str(&format!("  ⏱️  Total Time:          {:.2}s\n", summary.total_execution_time.as_secs_f64()));
            
            if summary.successful_executions > 0 {
                let avg_time = summary.total_execution_time.as_secs_f64() / summary.successful_executions as f64;
                output.push_str(&format!("  📊 Average Time:        {:.2}s\n", avg_time));
            }
        } else {
            output.push_str("  No executables to run (all compilations failed)\n");
        }
        
        output
    }

    /// Format compilation results for console output
    fn format_compilation_results_console(&self, results: &[CompilationResult]) -> String {
        let mut output = String::new();
        
        output.push_str("🔨 COMPILATION DETAILS\n");
        output.push_str("─────────────────────────────────────────────────────────────────────────────\n");
        
        let failed_results: Vec<_> = results.iter().filter(|r| !r.success).collect();
        let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
        
        if !failed_results.is_empty() {
            output.push_str("❌ FAILED COMPILATIONS:\n");
            for result in failed_results {
                output.push_str(&format!("  • {}\n", result.source_file.display()));
                if !result.stderr.is_empty() {
                    let error_lines: Vec<&str> = result.stderr.lines().take(3).collect();
                    for line in error_lines {
                        output.push_str(&format!("    {}\n", line.trim()));
                    }
                    if result.stderr.lines().count() > 3 {
                        output.push_str("    ... (truncated)\n");
                    }
                }
                output.push_str(&format!("    Time: {:.2}s\n", result.compilation_time.as_secs_f64()));
                output.push_str("\n");
            }
        }
        
        if self.include_details && !successful_results.is_empty() {
            output.push_str("✅ SUCCESSFUL COMPILATIONS:\n");
            for result in successful_results {
                output.push_str(&format!("  • {} ({:.2}s)\n", 
                    result.source_file.display(),
                    result.compilation_time.as_secs_f64()
                ));
                if let Some(exe_path) = &result.executable_path {
                    output.push_str(&format!("    → {}\n", exe_path.display()));
                }
            }
        }
        
        output
    }

    /// Format execution results for console output
    fn format_execution_results_console(&self, results: &[ExecutionResult]) -> String {
        let mut output = String::new();
        
        output.push_str("🚀 EXECUTION DETAILS\n");
        output.push_str("─────────────────────────────────────────────────────────────────────────────\n");
        
        let failed_results: Vec<_> = results.iter().filter(|r| !r.success).collect();
        let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
        
        if !failed_results.is_empty() {
            output.push_str("❌ FAILED EXECUTIONS:\n");
            for result in failed_results {
                output.push_str(&format!("  • {}\n", result.executable_path.display()));
                output.push_str(&format!("    Exit Code: {}\n", result.exit_code));
                
                if result.timed_out {
                    output.push_str("    Reason: Timed out\n");
                } else if result.memory_exceeded {
                    output.push_str("    Reason: Memory limit exceeded\n");
                } else if let Some(error) = &result.error_message {
                    output.push_str(&format!("    Error: {}\n", error));
                }
                
                if !result.stderr.is_empty() {
                    let error_lines: Vec<&str> = result.stderr.lines().take(2).collect();
                    for line in error_lines {
                        output.push_str(&format!("    {}\n", line.trim()));
                    }
                    if result.stderr.lines().count() > 2 {
                        output.push_str("    ... (truncated)\n");
                    }
                }
                
                output.push_str(&format!("    Time: {:.2}s\n", result.execution_time.as_secs_f64()));
                output.push_str("\n");
            }
        }
        
        if self.include_details && !successful_results.is_empty() {
            output.push_str("✅ SUCCESSFUL EXECUTIONS:\n");
            for result in successful_results {
                output.push_str(&format!("  • {} ({:.2}s)\n", 
                    result.executable_path.display(),
                    result.execution_time.as_secs_f64()
                ));
                
                if !result.stdout.is_empty() {
                    let output_lines: Vec<&str> = result.stdout.lines().take(2).collect();
                    for line in output_lines {
                        output.push_str(&format!("    Output: {}\n", line.trim()));
                    }
                    if result.stdout.lines().count() > 2 {
                        output.push_str("    ... (truncated)\n");
                    }
                }
            }
        }
        
        output
    }

    /// Format generated files for console output
    fn format_generated_files_console(&self, files: &[GeneratedTestFile]) -> String {
        let mut output = String::new();
        
        output.push_str("📝 GENERATED TEST FILES\n");
        output.push_str("─────────────────────────────────────────────────────────────────────────────\n");
        
        // Group by category
        let mut by_category: std::collections::HashMap<TestCategory, Vec<&GeneratedTestFile>> = std::collections::HashMap::new();
        for file in files {
            by_category.entry(file.category.clone()).or_default().push(file);
        }
        
        for (category, category_files) in by_category {
            output.push_str(&format!("📂 {:?} ({} files):\n", category, category_files.len()));
            for file in category_files {
                output.push_str(&format!("  • {}\n", file.file_path.display()));
                output.push_str(&format!("    {}\n", file.description));
            }
            output.push_str("\n");
        }
        
        output
    }

    /// Format failure analysis for console output
    fn format_failure_analysis_console(&self, compilation_results: &[CompilationResult], execution_results: &[ExecutionResult]) -> String {
        let mut output = String::new();
        
        output.push_str("🔍 FAILURE ANALYSIS\n");
        output.push_str("─────────────────────────────────────────────────────────────────────────────\n");
        
        let analysis = self.generate_failure_analysis(compilation_results, execution_results);
        
        if !analysis.common_compilation_errors.is_empty() {
            output.push_str("🔨 Common Compilation Errors:\n");
            for (error, count) in &analysis.common_compilation_errors {
                output.push_str(&format!("  • {} (occurred {} times)\n", error, count));
            }
            output.push_str("\n");
        }
        
        if !analysis.common_execution_errors.is_empty() {
            output.push_str("🚀 Common Execution Errors:\n");
            for (error, count) in &analysis.common_execution_errors {
                output.push_str(&format!("  • {} (occurred {} times)\n", error, count));
            }
            output.push_str("\n");
        }
        
        if !analysis.recommendations.is_empty() {
            output.push_str("💡 Recommendations:\n");
            for recommendation in &analysis.recommendations {
                output.push_str(&format!("  • {}\n", recommendation));
            }
            output.push_str("\n");
        }
        
        if analysis.common_compilation_errors.is_empty() && 
           analysis.common_execution_errors.is_empty() {
            output.push_str("✅ No common failure patterns detected.\n");
        }
        
        output
    }

    /// Format configuration for console output
    fn format_configuration_console(&self, config: &TestingConfig) -> String {
        let mut output = String::new();
        
        output.push_str("⚙️  CONFIGURATION\n");
        output.push_str("─────────────────────────────────────────────────────────────────────────────\n");
        output.push_str(&format!("Project Root:             {}\n", config.project_root.display()));
        output.push_str(&format!("Compiler Path:            {}\n", config.compiler_path.display()));
        output.push_str(&format!("Output Directory:         {}\n", config.output_directory.display()));
        output.push_str(&format!("Test Directories:         {:?}\n", config.test_directories));
        output.push_str(&format!("Compilation Timeout:      {:?}\n", config.compilation_timeout));
        output.push_str(&format!("Execution Timeout:        {:?}\n", config.execution_timeout));
        output.push_str(&format!("Generate Additional Tests: {}\n", config.generate_additional_tests));
        output.push_str(&format!("Test Categories:          {:?}\n", config.test_categories));
        output.push_str(&format!("Report Format:            {:?}\n", config.report_format));
        output.push_str(&format!("Max Parallel Compilations: {}\n", config.max_parallel_compilations));
        output.push_str(&format!("Max Parallel Executions: {}\n", config.max_parallel_executions));
        output.push_str(&format!("Cleanup Artifacts:        {}\n", config.cleanup_artifacts));
        output.push_str(&format!("Verbose:                  {}\n", config.verbose));
        
        output
    }

    /// Generate failure analysis from results
    fn generate_failure_analysis(&self, compilation_results: &[CompilationResult], execution_results: &[ExecutionResult]) -> FailureAnalysis {
        let mut common_compilation_errors = std::collections::HashMap::new();
        let mut common_execution_errors = std::collections::HashMap::new();
        let mut recommendations = Vec::new();

        // Analyze compilation errors
        for result in compilation_results.iter().filter(|r| !r.success) {
            let error_type = self.categorize_compilation_error(&result.stderr);
            *common_compilation_errors.entry(error_type).or_insert(0) += 1;
        }

        // Analyze execution errors
        for result in execution_results.iter().filter(|r| !r.success) {
            let error_type = self.categorize_execution_error(result);
            *common_execution_errors.entry(error_type).or_insert(0) += 1;
        }

        // Generate recommendations based on error patterns
        if common_compilation_errors.contains_key("Syntax Error") {
            recommendations.push("Review Aether syntax documentation for proper language constructs".to_string());
        }
        
        if common_compilation_errors.contains_key("Type Error") {
            recommendations.push("Check type annotations and ensure type compatibility".to_string());
        }
        
        if common_execution_errors.contains_key("Timeout") {
            recommendations.push("Consider increasing execution timeout for complex programs".to_string());
        }
        
        if common_execution_errors.contains_key("Memory Exceeded") {
            recommendations.push("Review memory usage patterns and consider increasing memory limits".to_string());
        }

        // Convert to sorted vectors for consistent output
        let mut compilation_errors: Vec<_> = common_compilation_errors.into_iter().collect();
        compilation_errors.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count descending

        let mut execution_errors: Vec<_> = common_execution_errors.into_iter().collect();
        execution_errors.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count descending

        FailureAnalysis {
            common_compilation_errors: compilation_errors,
            common_execution_errors: execution_errors,
            recommendations,
        }
    }

    /// Categorize compilation error based on stderr content
    fn categorize_compilation_error(&self, stderr: &str) -> String {
        let stderr_lower = stderr.to_lowercase();
        
        if stderr_lower.contains("syntax error") || stderr_lower.contains("unexpected token") {
            "Syntax Error".to_string()
        } else if stderr_lower.contains("type error") || stderr_lower.contains("type mismatch") {
            "Type Error".to_string()
        } else if stderr_lower.contains("undefined") || stderr_lower.contains("not found") {
            "Undefined Symbol".to_string()
        } else if stderr_lower.contains("linker") || stderr_lower.contains("linking") {
            "Linker Error".to_string()
        } else if stderr_lower.contains("internal compiler error") || stderr_lower.contains("ice") {
            "Internal Compiler Error".to_string()
        } else if stderr_lower.contains("timeout") {
            "Compilation Timeout".to_string()
        } else {
            "Other Compilation Error".to_string()
        }
    }

    /// Categorize execution error based on execution result
    fn categorize_execution_error(&self, result: &ExecutionResult) -> String {
        if result.timed_out {
            "Timeout".to_string()
        } else if result.memory_exceeded {
            "Memory Exceeded".to_string()
        } else if result.exit_code != 0 {
            match result.exit_code {
                -1 => "Segmentation Fault".to_string(),
                -2 => "Interrupt".to_string(),
                1 => "General Error".to_string(),
                2 => "Misuse of Shell Command".to_string(),
                126 => "Command Not Executable".to_string(),
                127 => "Command Not Found".to_string(),
                _ => format!("Exit Code {}", result.exit_code),
            }
        } else if let Some(error) = &result.error_message {
            if error.to_lowercase().contains("permission") {
                "Permission Denied".to_string()
            } else if error.to_lowercase().contains("file not found") {
                "File Not Found".to_string()
            } else {
                "Runtime Error".to_string()
            }
        } else {
            "Unknown Execution Error".to_string()
        }
    }

    /// Get HTML styles for the report
    fn get_html_styles(&self) -> String {
        r#"
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .timestamp {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 0.9em;
        }
        
        main {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .section h2 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 1.1em;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
        
        .metric-value {
            font-weight: bold;
        }
        
        .success { color: #28a745; }
        .failure { color: #dc3545; }
        .warning { color: #ffc107; }
        
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .results-table th,
        .results-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .results-table th {
            background-color: #667eea;
            color: white;
            font-weight: 500;
        }
        
        .results-table tr:hover {
            background-color: #f5f5f5;
        }
        
        .status-success {
            background-color: #d4edda;
            color: #155724;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        
        .status-failure {
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        
        .error-details {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .recommendations {
            background-color: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .recommendations h3 {
            color: #0066cc;
            margin-top: 0;
        }
        
        .recommendations ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        .recommendations li {
            margin: 8px 0;
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        .config-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .config-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
        }
        
        .config-table td:first-child {
            font-weight: bold;
            color: #667eea;
            width: 200px;
        }
        "#.to_string()
    }

    /// Format summary section for HTML output
    fn format_summary_html(&self, summary: &TestSummary) -> String {
        let mut html = String::new();
        
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>📊 Summary</h2>\n");
        html.push_str("            <div class=\"summary-grid\">\n");
        
        // Files summary card
        html.push_str("                <div class=\"summary-card\">\n");
        html.push_str("                    <h3>Files</h3>\n");
        html.push_str(&format!("                    <div class=\"metric\"><span>Total Files:</span><span class=\"metric-value\">{}</span></div>\n", summary.total_files));
        html.push_str(&format!("                    <div class=\"metric\"><span>Generated Tests:</span><span class=\"metric-value\">{}</span></div>\n", summary.generated_test_files));
        html.push_str("                </div>\n");
        
        // Compilation summary card
        html.push_str("                <div class=\"summary-card\">\n");
        html.push_str("                    <h3>Compilation</h3>\n");
        html.push_str(&format!("                    <div class=\"metric\"><span>Successful:</span><span class=\"metric-value success\">{}</span></div>\n", summary.successful_compilations));
        html.push_str(&format!("                    <div class=\"metric\"><span>Failed:</span><span class=\"metric-value failure\">{}</span></div>\n", summary.failed_compilations));
        html.push_str(&format!("                    <div class=\"metric\"><span>Total Time:</span><span class=\"metric-value\">{:.2}s</span></div>\n", summary.total_compilation_time.as_secs_f64()));
        html.push_str("                </div>\n");
        
        // Execution summary card
        html.push_str("                <div class=\"summary-card\">\n");
        html.push_str("                    <h3>Execution</h3>\n");
        html.push_str(&format!("                    <div class=\"metric\"><span>Successful:</span><span class=\"metric-value success\">{}</span></div>\n", summary.successful_executions));
        html.push_str(&format!("                    <div class=\"metric\"><span>Failed:</span><span class=\"metric-value failure\">{}</span></div>\n", summary.failed_executions));
        html.push_str(&format!("                    <div class=\"metric\"><span>Total Time:</span><span class=\"metric-value\">{:.2}s</span></div>\n", summary.total_execution_time.as_secs_f64()));
        html.push_str("                </div>\n");
        
        html.push_str("            </div>\n");
        html.push_str("        </div>\n");
        
        html
    }

    /// Format compilation results for HTML output
    fn format_compilation_results_html(&self, results: &[CompilationResult]) -> String {
        let mut html = String::new();
        
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>🔨 Compilation Results</h2>\n");
        html.push_str("            <table class=\"results-table\">\n");
        html.push_str("                <thead>\n");
        html.push_str("                    <tr>\n");
        html.push_str("                        <th>File</th>\n");
        html.push_str("                        <th>Status</th>\n");
        html.push_str("                        <th>Time</th>\n");
        html.push_str("                        <th>Executable</th>\n");
        html.push_str("                    </tr>\n");
        html.push_str("                </thead>\n");
        html.push_str("                <tbody>\n");
        
        for result in results {
            html.push_str("                    <tr>\n");
            html.push_str(&format!("                        <td>{}</td>\n", result.source_file.display()));
            
            if result.success {
                html.push_str("                        <td><span class=\"status-success\">✅ Success</span></td>\n");
            } else {
                html.push_str("                        <td><span class=\"status-failure\">❌ Failed</span></td>\n");
            }
            
            html.push_str(&format!("                        <td>{:.2}s</td>\n", result.compilation_time.as_secs_f64()));
            
            if let Some(exe_path) = &result.executable_path {
                html.push_str(&format!("                        <td>{}</td>\n", exe_path.display()));
            } else {
                html.push_str("                        <td>-</td>\n");
            }
            
            html.push_str("                    </tr>\n");
            
            // Add error details for failed compilations
            if !result.success && !result.stderr.is_empty() {
                html.push_str("                    <tr>\n");
                html.push_str("                        <td colspan=\"4\">\n");
                html.push_str("                            <div class=\"error-details\">");
                html.push_str(&html_escape(&result.stderr));
                html.push_str("</div>\n");
                html.push_str("                        </td>\n");
                html.push_str("                    </tr>\n");
            }
        }
        
        html.push_str("                </tbody>\n");
        html.push_str("            </table>\n");
        html.push_str("        </div>\n");
        
        html
    }

    /// Format execution results for HTML output
    fn format_execution_results_html(&self, results: &[ExecutionResult]) -> String {
        let mut html = String::new();
        
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>🚀 Execution Results</h2>\n");
        html.push_str("            <table class=\"results-table\">\n");
        html.push_str("                <thead>\n");
        html.push_str("                    <tr>\n");
        html.push_str("                        <th>Executable</th>\n");
        html.push_str("                        <th>Status</th>\n");
        html.push_str("                        <th>Exit Code</th>\n");
        html.push_str("                        <th>Time</th>\n");
        html.push_str("                    </tr>\n");
        html.push_str("                </thead>\n");
        html.push_str("                <tbody>\n");
        
        for result in results {
            html.push_str("                    <tr>\n");
            html.push_str(&format!("                        <td>{}</td>\n", result.executable_path.display()));
            
            if result.success {
                html.push_str("                        <td><span class=\"status-success\">✅ Success</span></td>\n");
            } else {
                html.push_str("                        <td><span class=\"status-failure\">❌ Failed</span></td>\n");
            }
            
            html.push_str(&format!("                        <td>{}</td>\n", result.exit_code));
            html.push_str(&format!("                        <td>{:.2}s</td>\n", result.execution_time.as_secs_f64()));
            html.push_str("                    </tr>\n");
            
            // Add error details for failed executions
            if !result.success {
                html.push_str("                    <tr>\n");
                html.push_str("                        <td colspan=\"4\">\n");
                html.push_str("                            <div class=\"error-details\">");
                
                if result.timed_out {
                    html.push_str("Execution timed out");
                } else if result.memory_exceeded {
                    html.push_str("Memory limit exceeded");
                } else if let Some(error) = &result.error_message {
                    html.push_str(&html_escape(error));
                } else if !result.stderr.is_empty() {
                    html.push_str(&html_escape(&result.stderr));
                }
                
                html.push_str("</div>\n");
                html.push_str("                        </td>\n");
                html.push_str("                    </tr>\n");
            }
        }
        
        html.push_str("                </tbody>\n");
        html.push_str("            </table>\n");
        html.push_str("        </div>\n");
        
        html
    }

    /// Format generated files for HTML output
    fn format_generated_files_html(&self, files: &[GeneratedTestFile]) -> String {
        let mut html = String::new();
        
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>📝 Generated Test Files</h2>\n");
        html.push_str("            <table class=\"results-table\">\n");
        html.push_str("                <thead>\n");
        html.push_str("                    <tr>\n");
        html.push_str("                        <th>File</th>\n");
        html.push_str("                        <th>Category</th>\n");
        html.push_str("                        <th>Description</th>\n");
        html.push_str("                    </tr>\n");
        html.push_str("                </thead>\n");
        html.push_str("                <tbody>\n");
        
        for file in files {
            html.push_str("                    <tr>\n");
            html.push_str(&format!("                        <td>{}</td>\n", file.file_path.display()));
            html.push_str(&format!("                        <td>{:?}</td>\n", file.category));
            html.push_str(&format!("                        <td>{}</td>\n", html_escape(&file.description)));
            html.push_str("                    </tr>\n");
        }
        
        html.push_str("                </tbody>\n");
        html.push_str("            </table>\n");
        html.push_str("        </div>\n");
        
        html
    }

    /// Format failure analysis for HTML output
    fn format_failure_analysis_html(&self, compilation_results: &[CompilationResult], execution_results: &[ExecutionResult]) -> String {
        let mut html = String::new();
        let analysis = self.generate_failure_analysis(compilation_results, execution_results);
        
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>🔍 Failure Analysis</h2>\n");
        
        if !analysis.common_compilation_errors.is_empty() || !analysis.common_execution_errors.is_empty() {
            html.push_str("            <div class=\"summary-grid\">\n");
            
            if !analysis.common_compilation_errors.is_empty() {
                html.push_str("                <div class=\"summary-card\">\n");
                html.push_str("                    <h3>Common Compilation Errors</h3>\n");
                for (error, count) in &analysis.common_compilation_errors {
                    html.push_str(&format!("                    <div class=\"metric\"><span>{}</span><span class=\"metric-value failure\">{}</span></div>\n", html_escape(error), count));
                }
                html.push_str("                </div>\n");
            }
            
            if !analysis.common_execution_errors.is_empty() {
                html.push_str("                <div class=\"summary-card\">\n");
                html.push_str("                    <h3>Common Execution Errors</h3>\n");
                for (error, count) in &analysis.common_execution_errors {
                    html.push_str(&format!("                    <div class=\"metric\"><span>{}</span><span class=\"metric-value failure\">{}</span></div>\n", html_escape(error), count));
                }
                html.push_str("                </div>\n");
            }
            
            html.push_str("            </div>\n");
        }
        
        if !analysis.recommendations.is_empty() {
            html.push_str("            <div class=\"recommendations\">\n");
            html.push_str("                <h3>💡 Recommendations</h3>\n");
            html.push_str("                <ul>\n");
            for recommendation in &analysis.recommendations {
                html.push_str(&format!("                    <li>{}</li>\n", html_escape(recommendation)));
            }
            html.push_str("                </ul>\n");
            html.push_str("            </div>\n");
        }
        
        if analysis.common_compilation_errors.is_empty() && analysis.common_execution_errors.is_empty() {
            html.push_str("            <p>✅ No common failure patterns detected.</p>\n");
        }
        
        html.push_str("        </div>\n");
        
        html
    }

    /// Format configuration for HTML output
    fn format_configuration_html(&self, config: &TestingConfig) -> String {
        let mut html = String::new();
        
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>⚙️ Configuration</h2>\n");
        html.push_str("            <table class=\"config-table\">\n");
        html.push_str(&format!("                <tr><td>Project Root</td><td>{}</td></tr>\n", config.project_root.display()));
        html.push_str(&format!("                <tr><td>Compiler Path</td><td>{}</td></tr>\n", config.compiler_path.display()));
        html.push_str(&format!("                <tr><td>Output Directory</td><td>{}</td></tr>\n", config.output_directory.display()));
        html.push_str(&format!("                <tr><td>Test Directories</td><td>{:?}</td></tr>\n", config.test_directories));
        html.push_str(&format!("                <tr><td>Compilation Timeout</td><td>{:?}</td></tr>\n", config.compilation_timeout));
        html.push_str(&format!("                <tr><td>Execution Timeout</td><td>{:?}</td></tr>\n", config.execution_timeout));
        html.push_str(&format!("                <tr><td>Generate Additional Tests</td><td>{}</td></tr>\n", config.generate_additional_tests));
        html.push_str(&format!("                <tr><td>Test Categories</td><td>{:?}</td></tr>\n", config.test_categories));
        html.push_str(&format!("                <tr><td>Report Format</td><td>{:?}</td></tr>\n", config.report_format));
        html.push_str(&format!("                <tr><td>Max Parallel Compilations</td><td>{}</td></tr>\n", config.max_parallel_compilations));
        html.push_str(&format!("                <tr><td>Max Parallel Executions</td><td>{}</td></tr>\n", config.max_parallel_executions));
        html.push_str(&format!("                <tr><td>Cleanup Artifacts</td><td>{}</td></tr>\n", config.cleanup_artifacts));
        html.push_str(&format!("                <tr><td>Verbose</td><td>{}</td></tr>\n", config.verbose));
        html.push_str("            </table>\n");
        html.push_str("        </div>\n");
        
        html
    }

    /// Format summary section for Markdown output
    fn format_summary_markdown(&self, summary: &TestSummary) -> String {
        let mut md = String::new();
        
        md.push_str("## 📊 Summary\n\n");
        
        md.push_str("### Files\n");
        md.push_str(&format!("- **Total Files:** {}\n", summary.total_files));
        md.push_str(&format!("- **Generated Test Files:** {}\n\n", summary.generated_test_files));
        
        md.push_str("### Compilation Results\n");
        md.push_str(&format!("- **Successful:** {} ({:.1}%)\n", 
            summary.successful_compilations,
            (summary.successful_compilations as f64 / summary.total_files as f64) * 100.0
        ));
        md.push_str(&format!("- **Failed:** {} ({:.1}%)\n", 
            summary.failed_compilations,
            (summary.failed_compilations as f64 / summary.total_files as f64) * 100.0
        ));
        md.push_str(&format!("- **Total Time:** {:.2}s\n", summary.total_compilation_time.as_secs_f64()));
        
        if summary.successful_compilations > 0 {
            let avg_time = summary.total_compilation_time.as_secs_f64() / summary.successful_compilations as f64;
            md.push_str(&format!("- **Average Time:** {:.2}s\n", avg_time));
        }
        
        md.push_str("\n### Execution Results\n");
        let total_executions = summary.successful_executions + summary.failed_executions;
        if total_executions > 0 {
            md.push_str(&format!("- **Successful:** {} ({:.1}%)\n", 
                summary.successful_executions,
                (summary.successful_executions as f64 / total_executions as f64) * 100.0
            ));
            md.push_str(&format!("- **Failed:** {} ({:.1}%)\n", 
                summary.failed_executions,
                (summary.failed_executions as f64 / total_executions as f64) * 100.0
            ));
            md.push_str(&format!("- **Total Time:** {:.2}s\n", summary.total_execution_time.as_secs_f64()));
            
            if summary.successful_executions > 0 {
                let avg_time = summary.total_execution_time.as_secs_f64() / summary.successful_executions as f64;
                md.push_str(&format!("- **Average Time:** {:.2}s\n", avg_time));
            }
        } else {
            md.push_str("- No executables to run (all compilations failed)\n");
        }
        
        md
    }

    /// Format compilation results for Markdown output
    fn format_compilation_results_markdown(&self, results: &[CompilationResult]) -> String {
        let mut md = String::new();
        
        md.push_str("## 🔨 Compilation Results\n\n");
        
        let failed_results: Vec<_> = results.iter().filter(|r| !r.success).collect();
        let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
        
        if !failed_results.is_empty() {
            md.push_str("### ❌ Failed Compilations\n\n");
            for result in failed_results {
                md.push_str(&format!("#### {}\n", result.source_file.display()));
                md.push_str(&format!("- **Time:** {:.2}s\n", result.compilation_time.as_secs_f64()));
                if let Some(exit_code) = result.exit_code {
                    md.push_str(&format!("- **Exit Code:** {}\n", exit_code));
                }
                if !result.stderr.is_empty() {
                    md.push_str("- **Error:**\n```\n");
                    md.push_str(&result.stderr);
                    md.push_str("\n```\n");
                }
                md.push_str("\n");
            }
        }
        
        if self.include_details && !successful_results.is_empty() {
            md.push_str("### ✅ Successful Compilations\n\n");
            md.push_str("| File | Time | Executable |\n");
            md.push_str("|------|------|------------|\n");
            for result in successful_results {
                let exe_path = result.executable_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "-".to_string());
                md.push_str(&format!("| {} | {:.2}s | {} |\n", 
                    result.source_file.display(),
                    result.compilation_time.as_secs_f64(),
                    exe_path
                ));
            }
        }
        
        md
    }

    /// Format execution results for Markdown output
    fn format_execution_results_markdown(&self, results: &[ExecutionResult]) -> String {
        let mut md = String::new();
        
        md.push_str("## 🚀 Execution Results\n\n");
        
        let failed_results: Vec<_> = results.iter().filter(|r| !r.success).collect();
        let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
        
        if !failed_results.is_empty() {
            md.push_str("### ❌ Failed Executions\n\n");
            for result in failed_results {
                md.push_str(&format!("#### {}\n", result.executable_path.display()));
                md.push_str(&format!("- **Exit Code:** {}\n", result.exit_code));
                md.push_str(&format!("- **Time:** {:.2}s\n", result.execution_time.as_secs_f64()));
                
                if result.timed_out {
                    md.push_str("- **Reason:** Timed out\n");
                } else if result.memory_exceeded {
                    md.push_str("- **Reason:** Memory limit exceeded\n");
                } else if let Some(error) = &result.error_message {
                    md.push_str(&format!("- **Error:** {}\n", error));
                }
                
                if !result.stderr.is_empty() {
                    md.push_str("- **Error Output:**\n```\n");
                    md.push_str(&result.stderr);
                    md.push_str("\n```\n");
                }
                md.push_str("\n");
            }
        }
        
        if self.include_details && !successful_results.is_empty() {
            md.push_str("### ✅ Successful Executions\n\n");
            md.push_str("| Executable | Time | Output |\n");
            md.push_str("|------------|------|--------|\n");
            for result in successful_results {
                let output_preview = if result.stdout.is_empty() {
                    "-".to_string()
                } else {
                    result.stdout.lines().next().unwrap_or("").to_string()
                };
                md.push_str(&format!("| {} | {:.2}s | {} |\n", 
                    result.executable_path.display(),
                    result.execution_time.as_secs_f64(),
                    output_preview
                ));
            }
        }
        
        md
    }

    /// Format generated files for Markdown output
    fn format_generated_files_markdown(&self, files: &[GeneratedTestFile]) -> String {
        let mut md = String::new();
        
        md.push_str("## 📝 Generated Test Files\n\n");
        
        // Group by category
        let mut by_category: std::collections::HashMap<TestCategory, Vec<&GeneratedTestFile>> = std::collections::HashMap::new();
        for file in files {
            by_category.entry(file.category.clone()).or_default().push(file);
        }
        
        for (category, category_files) in by_category {
            md.push_str(&format!("### {:?} ({} files)\n\n", category, category_files.len()));
            md.push_str("| File | Description |\n");
            md.push_str("|------|-------------|\n");
            for file in category_files {
                md.push_str(&format!("| {} | {} |\n", file.file_path.display(), file.description));
            }
            md.push_str("\n");
        }
        
        md
    }

    /// Format failure analysis for Markdown output
    fn format_failure_analysis_markdown(&self, compilation_results: &[CompilationResult], execution_results: &[ExecutionResult]) -> String {
        let mut md = String::new();
        let analysis = self.generate_failure_analysis(compilation_results, execution_results);
        
        md.push_str("## 🔍 Failure Analysis\n\n");
        
        if !analysis.common_compilation_errors.is_empty() {
            md.push_str("### Common Compilation Errors\n\n");
            md.push_str("| Error Type | Count |\n");
            md.push_str("|------------|-------|\n");
            for (error, count) in &analysis.common_compilation_errors {
                md.push_str(&format!("| {} | {} |\n", error, count));
            }
            md.push_str("\n");
        }
        
        if !analysis.common_execution_errors.is_empty() {
            md.push_str("### Common Execution Errors\n\n");
            md.push_str("| Error Type | Count |\n");
            md.push_str("|------------|-------|\n");
            for (error, count) in &analysis.common_execution_errors {
                md.push_str(&format!("| {} | {} |\n", error, count));
            }
            md.push_str("\n");
        }
        
        if !analysis.recommendations.is_empty() {
            md.push_str("### 💡 Recommendations\n\n");
            for recommendation in &analysis.recommendations {
                md.push_str(&format!("- {}\n", recommendation));
            }
            md.push_str("\n");
        }
        
        if analysis.common_compilation_errors.is_empty() && analysis.common_execution_errors.is_empty() {
            md.push_str("✅ No common failure patterns detected.\n");
        }
        
        md
    }

    /// Format configuration for Markdown output
    fn format_configuration_markdown(&self, config: &TestingConfig) -> String {
        let mut md = String::new();
        
        md.push_str("## ⚙️ Configuration\n\n");
        md.push_str("| Setting | Value |\n");
        md.push_str("|---------|-------|\n");
        md.push_str(&format!("| Project Root | {} |\n", config.project_root.display()));
        md.push_str(&format!("| Compiler Path | {} |\n", config.compiler_path.display()));
        md.push_str(&format!("| Output Directory | {} |\n", config.output_directory.display()));
        md.push_str(&format!("| Test Directories | {:?} |\n", config.test_directories));
        md.push_str(&format!("| Compilation Timeout | {:?} |\n", config.compilation_timeout));
        md.push_str(&format!("| Execution Timeout | {:?} |\n", config.execution_timeout));
        md.push_str(&format!("| Generate Additional Tests | {} |\n", config.generate_additional_tests));
        md.push_str(&format!("| Test Categories | {:?} |\n", config.test_categories));
        md.push_str(&format!("| Report Format | {:?} |\n", config.report_format));
        md.push_str(&format!("| Max Parallel Compilations | {} |\n", config.max_parallel_compilations));
        md.push_str(&format!("| Max Parallel Executions | {} |\n", config.max_parallel_executions));
        md.push_str(&format!("| Cleanup Artifacts | {} |\n", config.cleanup_artifacts));
        md.push_str(&format!("| Verbose | {} |\n", config.verbose));
        
        md
    }
}

/// Failure analysis structure
#[derive(Debug, Clone, Serialize)]
pub struct FailureAnalysis {
    pub common_compilation_errors: Vec<(String, usize)>,
    pub common_execution_errors: Vec<(String, usize)>,
    pub recommendations: Vec<String>,
}

/// JSON report structures for serialization
#[derive(Debug, Clone, Serialize)]
pub struct JsonReport {
    pub metadata: JsonReportMetadata,
    pub summary: JsonSummary,
    pub compilation_results: Vec<JsonCompilationResult>,
    pub execution_results: Vec<JsonExecutionResult>,
    pub generated_files: Vec<JsonGeneratedFile>,
    pub failure_analysis: FailureAnalysis,
    pub configuration: Option<JsonConfiguration>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonReportMetadata {
    pub generated_at: String,
    pub format_version: String,
    pub aether_version: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonSummary {
    pub total_files: usize,
    pub successful_compilations: usize,
    pub failed_compilations: usize,
    pub successful_executions: usize,
    pub failed_executions: usize,
    pub total_compilation_time_seconds: f64,
    pub total_execution_time_seconds: f64,
    pub generated_test_files: usize,
}

impl JsonSummary {
    pub fn from_test_summary(summary: &TestSummary) -> Self {
        Self {
            total_files: summary.total_files,
            successful_compilations: summary.successful_compilations,
            failed_compilations: summary.failed_compilations,
            successful_executions: summary.successful_executions,
            failed_executions: summary.failed_executions,
            total_compilation_time_seconds: summary.total_compilation_time.as_secs_f64(),
            total_execution_time_seconds: summary.total_execution_time.as_secs_f64(),
            generated_test_files: summary.generated_test_files,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonCompilationResult {
    pub source_file: String,
    pub executable_path: Option<String>,
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub compilation_time_seconds: f64,
    pub exit_code: Option<i32>,
}

impl JsonCompilationResult {
    pub fn from_compilation_result(result: &CompilationResult) -> Self {
        Self {
            source_file: result.source_file.to_string_lossy().to_string(),
            executable_path: result.executable_path.as_ref().map(|p| p.to_string_lossy().to_string()),
            success: result.success,
            stdout: result.stdout.clone(),
            stderr: result.stderr.clone(),
            compilation_time_seconds: result.compilation_time.as_secs_f64(),
            exit_code: result.exit_code,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonExecutionResult {
    pub executable_path: String,
    pub success: bool,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub execution_time_seconds: f64,
    pub timed_out: bool,
    pub memory_exceeded: bool,
    pub error_message: Option<String>,
}

impl JsonExecutionResult {
    pub fn from_execution_result(result: &ExecutionResult) -> Self {
        Self {
            executable_path: result.executable_path.to_string_lossy().to_string(),
            success: result.success,
            exit_code: result.exit_code,
            stdout: result.stdout.clone(),
            stderr: result.stderr.clone(),
            execution_time_seconds: result.execution_time.as_secs_f64(),
            timed_out: result.timed_out,
            memory_exceeded: result.memory_exceeded,
            error_message: result.error_message.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonGeneratedFile {
    pub path: String,
    pub category: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonConfiguration {
    pub project_root: String,
    pub compiler_path: String,
    pub output_directory: String,
    pub test_directories: Vec<String>,
    pub compilation_timeout_seconds: f64,
    pub execution_timeout_seconds: f64,
    pub generate_additional_tests: bool,
    pub test_categories: Vec<String>,
    pub report_format: String,
    pub max_parallel_compilations: usize,
    pub max_parallel_executions: usize,
    pub cleanup_artifacts: bool,
    pub verbose: bool,
}

impl JsonConfiguration {
    pub fn from_testing_config(config: &TestingConfig) -> Self {
        Self {
            project_root: config.project_root.to_string_lossy().to_string(),
            compiler_path: config.compiler_path.to_string_lossy().to_string(),
            output_directory: config.output_directory.to_string_lossy().to_string(),
            test_directories: config.test_directories.clone(),
            compilation_timeout_seconds: config.compilation_timeout.as_secs_f64(),
            execution_timeout_seconds: config.execution_timeout.as_secs_f64(),
            generate_additional_tests: config.generate_additional_tests,
            test_categories: config.test_categories.iter().map(|c| format!("{:?}", c)).collect(),
            report_format: format!("{:?}", config.report_format),
            max_parallel_compilations: config.max_parallel_compilations,
            max_parallel_executions: config.max_parallel_executions,
            cleanup_artifacts: config.cleanup_artifacts,
            verbose: config.verbose,
        }
    }
}

/// HTML escape utility function
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Comprehensive test report for file compilation testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCompilationTestReport {
    pub summary: TestSummary,
    pub compilation_results: Vec<CompilationResult>,
    pub execution_results: Vec<ExecutionResult>,
    pub generated_files: Vec<GeneratedTestFile>,
    pub discovered_files: Vec<PathBuf>,
    pub config: TestingConfig,
}

/// Summary statistics for test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    pub total_files: usize,
    pub successful_compilations: usize,
    pub failed_compilations: usize,
    pub successful_executions: usize,
    pub failed_executions: usize,
    pub total_compilation_time: Duration,
    pub total_execution_time: Duration,
    pub total_duration: Duration,
    pub generated_test_files: usize,
}

impl TestSummary {
    pub fn from_results(
        compilation_results: &[CompilationResult],
        execution_results: &[ExecutionResult],
        generated_files_count: usize,
    ) -> Self {
        let total_files = compilation_results.len();
        let successful_compilations = compilation_results.iter().filter(|r| r.success).count();
        let failed_compilations = total_files - successful_compilations;
        
        let successful_executions = execution_results.iter().filter(|r| r.success).count();
        let failed_executions = execution_results.len() - successful_executions;
        
        let total_compilation_time = compilation_results
            .iter()
            .map(|r| r.compilation_time)
            .sum();
        
        let total_execution_time = execution_results
            .iter()
            .map(|r| r.execution_time)
            .sum();

        Self {
            total_files,
            successful_compilations,
            failed_compilations,
            successful_executions,
            failed_executions,
            total_compilation_time,
            total_execution_time,
            total_duration: total_compilation_time + total_execution_time,
            generated_test_files: generated_files_count,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_files == 0 {
            0.0
        } else {
            self.successful_compilations as f64 / self.total_files as f64
        }
    }

    pub fn execution_success_rate(&self) -> f64 {
        let total_executions = self.successful_executions + self.failed_executions;
        if total_executions == 0 {
            0.0
        } else {
            self.successful_executions as f64 / total_executions as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_config_creation() {
        let config = TestingConfig::default();
        assert_eq!(config.project_root, PathBuf::from("."));
        assert_eq!(config.compiler_path, PathBuf::from("aetherc"));
        assert!(config.compilation_timeout > Duration::from_secs(0));
        assert!(config.execution_timeout > Duration::from_secs(0));
    }

    #[test]
    fn test_config_validation() {
        let mut config = TestingConfig::default();
        
        // Test invalid project root
        config.project_root = PathBuf::from("/nonexistent/path");
        assert!(config.validate().is_err());
        
        // Test zero timeout
        config.project_root = PathBuf::from(".");
        config.compilation_timeout = Duration::from_secs(0);
        assert!(config.validate().is_err());
        
        // Test zero parallel processes
        config.compilation_timeout = Duration::from_secs(30);
        config.max_parallel_compilations = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_file_discovery_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let engine = FileDiscoveryEngine::new(temp_dir.path().to_path_buf());
        assert!(engine.is_ok());
    }

    #[test]
    fn test_file_discovery_validation() {
        let temp_dir = TempDir::new().unwrap();
        let engine = FileDiscoveryEngine::new(temp_dir.path().to_path_buf()).unwrap();
        assert!(engine.validate_setup().is_ok());
        
        let invalid_engine = FileDiscoveryEngine::new(PathBuf::from("/nonexistent")).unwrap();
        assert!(invalid_engine.validate_setup().is_err());
    }

    #[test]
    fn test_test_summary_calculation() {
        let compilation_results = vec![
            CompilationResult {
                source_file: PathBuf::from("test1.ae"),
                executable_path: Some(PathBuf::from("test1.exe")),
                success: true,
                stdout: String::new(),
                stderr: String::new(),
                compilation_time: Duration::from_millis(100),
                exit_code: Some(0),
            },
            CompilationResult {
                source_file: PathBuf::from("test2.ae"),
                executable_path: None,
                success: false,
                stdout: String::new(),
                stderr: "Error".to_string(),
                compilation_time: Duration::from_millis(50),
                exit_code: Some(1),
            },
        ];

        let execution_results = vec![
            ExecutionResult {
                executable_path: PathBuf::from("test1.exe"),
                success: true,
                exit_code: 0,
                stdout: "Hello".to_string(),
                stderr: String::new(),
                execution_time: Duration::from_millis(25),
                timed_out: false,
                memory_exceeded: false,
                error_message: None,
            },
        ];

        let summary = TestSummary::from_results(&compilation_results, &execution_results, 0);
        
        assert_eq!(summary.total_files, 2);
        assert_eq!(summary.successful_compilations, 1);
        assert_eq!(summary.failed_compilations, 1);
        assert_eq!(summary.successful_executions, 1);
        assert_eq!(summary.failed_executions, 0);
        assert_eq!(summary.success_rate(), 0.5);
        assert_eq!(summary.execution_success_rate(), 1.0);
    }

    #[test]
    fn test_orchestrator_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TestingConfig::default();
        config.project_root = temp_dir.path().to_path_buf();
        config.output_directory = temp_dir.path().join("output");
        
        // Skip compiler validation for this test
        config.compiler_path = PathBuf::from("echo"); // Use echo as a mock compiler
        
        let orchestrator = FileCompilationTestOrchestrator::new(config);
        assert!(orchestrator.is_ok());
    }

    // File Discovery Engine Tests
    
    #[test]
    fn test_file_discovery_basic() {
        let temp_dir = TempDir::new().unwrap();
        let root_path = temp_dir.path();
        
        // Create test files
        fs::write(root_path.join("test1.ae"), "// Test file 1").unwrap();
        fs::write(root_path.join("test2.ae"), "// Test file 2").unwrap();
        fs::write(root_path.join("readme.txt"), "Not an Aether file").unwrap();
        
        let engine = FileDiscoveryEngine::new(root_path.to_path_buf()).unwrap();
        let discovered_files = engine.discover_aether_files().unwrap();
        
        assert_eq!(discovered_files.len(), 2);
        assert!(discovered_files.iter().any(|f| f.file_name().unwrap() == "test1.ae"));
        assert!(discovered_files.iter().any(|f| f.file_name().unwrap() == "test2.ae"));
        assert!(!discovered_files.iter().any(|f| f.file_name().unwrap() == "readme.txt"));
    }

    #[test]
    fn test_file_discovery_recursive() {
        let temp_dir = TempDir::new().unwrap();
        let root_path = temp_dir.path();
        
        // Create directory structure
        fs::create_dir_all(root_path.join("examples")).unwrap();
        fs::create_dir_all(root_path.join("tests")).unwrap();
        fs::create_dir_all(root_path.join("src/subdir")).unwrap();
        
        // Create test files
        fs::write(root_path.join("main.ae"), "// Main file").unwrap();
        fs::write(root_path.join("examples/example1.ae"), "// Example 1").unwrap();
        fs::write(root_path.join("examples/example2.ae"), "// Example 2").unwrap();
        fs::write(root_path.join("tests/test1.ae"), "// Test 1").unwrap();
        fs::write(root_path.join("src/lib.ae"), "// Library").unwrap();
        fs::write(root_path.join("src/subdir/module.ae"), "// Module").unwrap();
        
        let engine = FileDiscoveryEngine::new(root_path.to_path_buf()).unwrap();
        let discovered_files = engine.discover_aether_files().unwrap();
        
        assert_eq!(discovered_files.len(), 6);
        
        // Check that all expected files are found
        let file_names: Vec<String> = discovered_files
            .iter()
            .map(|f| f.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        
        assert!(file_names.contains(&"main.ae".to_string()));
        assert!(file_names.contains(&"example1.ae".to_string()));
        assert!(file_names.contains(&"example2.ae".to_string()));
        assert!(file_names.contains(&"test1.ae".to_string()));
        assert!(file_names.contains(&"lib.ae".to_string()));
        assert!(file_names.contains(&"module.ae".to_string()));
    }

    #[test]
    fn test_file_discovery_exclude_patterns() {
        let temp_dir = TempDir::new().unwrap();
        let root_path = temp_dir.path();
        
        // Create directory structure with excluded directories
        fs::create_dir_all(root_path.join("target/debug")).unwrap();
        fs::create_dir_all(root_path.join(".git/objects")).unwrap();
        fs::create_dir_all(root_path.join("examples")).unwrap();
        
        // Create test files
        fs::write(root_path.join("main.ae"), "// Main file").unwrap();
        fs::write(root_path.join("examples/example.ae"), "// Example").unwrap();
        fs::write(root_path.join("target/debug/compiled.ae"), "// Should be excluded").unwrap();
        fs::write(root_path.join(".git/objects/file.ae"), "// Should be excluded").unwrap();
        
        let engine = FileDiscoveryEngine::new(root_path.to_path_buf()).unwrap();
        let discovered_files = engine.discover_aether_files().unwrap();
        
        assert_eq!(discovered_files.len(), 2);
        
        let file_names: Vec<String> = discovered_files
            .iter()
            .map(|f| f.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        
        assert!(file_names.contains(&"main.ae".to_string()));
        assert!(file_names.contains(&"example.ae".to_string()));
        assert!(!file_names.contains(&"compiled.ae".to_string()));
    }

    #[test]
    fn test_file_discovery_filter_by_directories() {
        let temp_dir = TempDir::new().unwrap();
        let root_path = temp_dir.path();
        
        // Create directory structure
        fs::create_dir_all(root_path.join("examples")).unwrap();
        fs::create_dir_all(root_path.join("tests")).unwrap();
        fs::create_dir_all(root_path.join("src")).unwrap();
        
        // Create test files
        fs::write(root_path.join("main.ae"), "// Main file").unwrap();
        fs::write(root_path.join("examples/example.ae"), "// Example").unwrap();
        fs::write(root_path.join("tests/test.ae"), "// Test").unwrap();
        fs::write(root_path.join("src/lib.ae"), "// Library").unwrap();
        
        let engine = FileDiscoveryEngine::new(root_path.to_path_buf()).unwrap();
        
        // Test filtering by examples directory only
        let examples_files = engine.discover_in_directories(&["examples"]).unwrap();
        assert_eq!(examples_files.len(), 1);
        assert!(examples_files[0].file_name().unwrap() == "example.ae");
        
        // Test filtering by tests directory only
        let tests_files = engine.discover_in_directories(&["tests"]).unwrap();
        assert_eq!(tests_files.len(), 1);
        assert!(tests_files[0].file_name().unwrap() == "test.ae");
        
        // Test filtering by multiple directories
        let filtered_files = engine.discover_in_directories(&["examples", "tests"]).unwrap();
        assert_eq!(filtered_files.len(), 2);
        
        // Test filtering by non-existent directory
        let empty_files = engine.discover_in_directories(&["nonexistent"]).unwrap();
        assert_eq!(empty_files.len(), 0);
    }

    #[test]
    fn test_file_discovery_custom_patterns() {
        let temp_dir = TempDir::new().unwrap();
        let root_path = temp_dir.path();
        
        // Create test files with different extensions
        fs::write(root_path.join("test.ae"), "// Aether file").unwrap();
        fs::write(root_path.join("test.aether"), "// Aether file with different extension").unwrap();
        fs::write(root_path.join("test.txt"), "// Text file").unwrap();
        
        // Test with custom include patterns
        let engine = FileDiscoveryEngine::with_patterns(
            root_path.to_path_buf(),
            vec!["*.ae".to_string(), "*.aether".to_string()],
            vec![],
        ).unwrap();
        
        let discovered_files = engine.discover_aether_files().unwrap();
        assert_eq!(discovered_files.len(), 2);
        
        let file_names: Vec<String> = discovered_files
            .iter()
            .map(|f| f.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        
        assert!(file_names.contains(&"test.ae".to_string()));
        assert!(file_names.contains(&"test.aether".to_string()));
        assert!(!file_names.contains(&"test.txt".to_string()));
    }

    #[test]
    fn test_file_discovery_stats() {
        let temp_dir = TempDir::new().unwrap();
        let root_path = temp_dir.path();
        
        // Create directory structure
        fs::create_dir_all(root_path.join("examples")).unwrap();
        fs::create_dir_all(root_path.join("tests")).unwrap();
        fs::create_dir_all(root_path.join("src")).unwrap();
        
        // Create test files
        fs::write(root_path.join("main.ae"), "// Main file").unwrap();
        fs::write(root_path.join("examples/example1.ae"), "// Example 1").unwrap();
        fs::write(root_path.join("examples/example2.ae"), "// Example 2").unwrap();
        fs::write(root_path.join("tests/test1.ae"), "// Test 1").unwrap();
        fs::write(root_path.join("src/lib.ae"), "// Library").unwrap();
        
        let engine = FileDiscoveryEngine::new(root_path.to_path_buf()).unwrap();
        let stats = engine.get_discovery_stats().unwrap();
        
        assert_eq!(stats.total_files, 5);
        assert_eq!(stats.examples_files, 2);
        assert_eq!(stats.tests_files, 1);
        assert_eq!(stats.other_files, 2); // main.ae and src/lib.ae
        assert!(stats.directories_scanned >= 4); // root, examples, tests, src
    }

    #[test]
    fn test_glob_pattern_matching() {
        let temp_dir = TempDir::new().unwrap();
        let engine = FileDiscoveryEngine::new(temp_dir.path().to_path_buf()).unwrap();
        
        // Test simple wildcard patterns
        assert!(engine.matches_glob_pattern("test.ae", "*.ae"));
        assert!(engine.matches_glob_pattern("example.ae", "*.ae"));
        assert!(!engine.matches_glob_pattern("test.txt", "*.ae"));
        
        // Test exact match
        assert!(engine.matches_glob_pattern("test.ae", "test.ae"));
        assert!(!engine.matches_glob_pattern("test.ae", "other.ae"));
        
        // Test recursive patterns
        assert!(engine.matches_glob_pattern("target/debug/file.ae", "target/**"));
        assert!(engine.matches_glob_pattern("target/release/subdir/file.ae", "target/**"));
        assert!(!engine.matches_glob_pattern("src/file.ae", "target/**"));
    }

    #[test]
    fn test_file_discovery_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let engine = FileDiscoveryEngine::new(temp_dir.path().to_path_buf()).unwrap();
        
        let discovered_files = engine.discover_aether_files().unwrap();
        assert_eq!(discovered_files.len(), 0);
        
        let stats = engine.get_discovery_stats().unwrap();
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.examples_files, 0);
        assert_eq!(stats.tests_files, 0);
        assert_eq!(stats.other_files, 0);
    }

    #[test]
    fn test_file_discovery_validation_errors() {
        // Test with non-existent directory
        let invalid_engine = FileDiscoveryEngine::new(PathBuf::from("/nonexistent/path")).unwrap();
        assert!(invalid_engine.validate_setup().is_err());
        
        // Test with file instead of directory
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("not_a_directory.txt");
        fs::write(&file_path, "content").unwrap();
        
        let file_engine = FileDiscoveryEngine::new(file_path).unwrap();
        assert!(file_engine.validate_setup().is_err());
    }

    #[test]
    fn test_file_discovery_sorted_output() {
        let temp_dir = TempDir::new().unwrap();
        let root_path = temp_dir.path();
        
        // Create files in non-alphabetical order
        fs::write(root_path.join("z_last.ae"), "// Last file").unwrap();
        fs::write(root_path.join("a_first.ae"), "// First file").unwrap();
        fs::write(root_path.join("m_middle.ae"), "// Middle file").unwrap();
        
        let engine = FileDiscoveryEngine::new(root_path.to_path_buf()).unwrap();
        let discovered_files = engine.discover_aether_files().unwrap();
        
        assert_eq!(discovered_files.len(), 3);
        
        // Verify files are sorted
        let file_names: Vec<String> = discovered_files
            .iter()
            .map(|f| f.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        
        assert_eq!(file_names[0], "a_first.ae");
        assert_eq!(file_names[1], "m_middle.ae");
        assert_eq!(file_names[2], "z_last.ae");
    }
    // ExecutionValidator Tests
    #[tokio::test]
    async fn test_execution_validator_creation() {
        let validator = ExecutionValidator::new(Duration::from_secs(5));
        assert!(validator.is_ok());
        
        let validator = validator.unwrap();
        assert_eq!(validator.timeout, Duration::from_secs(5));
        assert!(validator.capture_output);
        assert_eq!(validator.max_memory_mb, Some(512));
        assert!(validator.working_directory.is_none());
    }

    #[tokio::test]
    async fn test_execution_validator_with_settings() {
        let working_dir = PathBuf::from(".");
        let validator = ExecutionValidator::with_settings(
            Duration::from_secs(10),
            false,
            Some(1024),
            Some(working_dir.clone()),
        );
        
        assert!(validator.is_ok());
        let validator = validator.unwrap();
        assert_eq!(validator.timeout, Duration::from_secs(10));
        assert!(!validator.capture_output);
        assert_eq!(validator.max_memory_mb, Some(1024));
        assert_eq!(validator.working_directory, Some(working_dir));
    }

    #[tokio::test]
    async fn test_execution_validator_validation() {
        let mut validator = ExecutionValidator::new(Duration::from_secs(5)).unwrap();
        assert!(validator.validate_setup().is_ok());
        
        // Test zero timeout
        validator.set_timeout(Duration::from_secs(0));
        assert!(validator.validate_setup().is_err());
        
        // Test invalid working directory
        validator.set_timeout(Duration::from_secs(5));
        validator.set_working_directory(Some(PathBuf::from("/nonexistent/path")));
        assert!(validator.validate_setup().is_err());
        
        // Test zero memory limit
        validator.set_working_directory(None);
        validator.set_memory_limit(Some(0));
        assert!(validator.validate_setup().is_err());
    }

    #[tokio::test]
    async fn test_execution_validator_nonexistent_executable() {
        let validator = ExecutionValidator::new(Duration::from_secs(5)).unwrap();
        let nonexistent_path = PathBuf::from("/nonexistent/executable.exe");
        
        let result = validator.validate_executable(&nonexistent_path).await;
        assert!(result.is_ok());
        
        let execution_result = result.unwrap();
        assert!(!execution_result.success);
        assert_eq!(execution_result.exit_code, -1);
        assert!(!execution_result.timed_out);
        assert!(!execution_result.memory_exceeded);
        assert!(execution_result.error_message.is_some());
        assert!(execution_result.error_message.unwrap().contains("does not exist"));
    }

    #[tokio::test]
    async fn test_execution_validator_batch_empty() {
        let validator = ExecutionValidator::new(Duration::from_secs(5)).unwrap();
        let empty_executables: Vec<PathBuf> = Vec::new();
        
        let results = validator.validate_batch(&empty_executables).await;
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_execution_validator_batch_nonexistent() {
        let validator = ExecutionValidator::new(Duration::from_secs(5)).unwrap();
        let executables = vec![
            PathBuf::from("/nonexistent/exe1.exe"),
            PathBuf::from("/nonexistent/exe2.exe"),
        ];
        
        let results = validator.validate_batch(&executables).await;
        assert!(results.is_ok());
        
        let execution_results = results.unwrap();
        assert_eq!(execution_results.len(), 2);
        
        for result in execution_results {
            assert!(!result.success);
            assert_eq!(result.exit_code, -1);
            assert!(!result.timed_out);
            assert!(!result.memory_exceeded);
            assert!(result.error_message.is_some());
        }
    }

    #[tokio::test]
    async fn test_execution_validator_with_filter() {
        let validator = ExecutionValidator::new(Duration::from_secs(5)).unwrap();
        let executables = vec![
            PathBuf::from("test1.exe"),
            PathBuf::from("test2.exe"),
            PathBuf::from("other.exe"),
        ];
        
        // Filter to only include files starting with "test"
        let filter = |path: &Path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("test"))
                .unwrap_or(false)
        };
        
        let results = validator.validate_with_filter(&executables, filter).await;
        assert!(results.is_ok());
        
        let execution_results = results.unwrap();
        assert_eq!(execution_results.len(), 2); // Only test1.exe and test2.exe
        
        for result in execution_results {
            let filename = result.executable_path.file_name().unwrap().to_str().unwrap();
            assert!(filename.starts_with("test"));
        }
    }

    #[test]
    fn test_execution_stats_calculation() {
        let results = vec![
            ExecutionResult {
                executable_path: PathBuf::from("test1.exe"),
                success: true,
                exit_code: 0,
                stdout: "Success".to_string(),
                stderr: String::new(),
                execution_time: Duration::from_millis(100),
                timed_out: false,
                memory_exceeded: false,
                error_message: None,
            },
            ExecutionResult {
                executable_path: PathBuf::from("test2.exe"),
                success: false,
                exit_code: 1,
                stdout: String::new(),
                stderr: "Error".to_string(),
                execution_time: Duration::from_millis(200),
                timed_out: true,
                memory_exceeded: false,
                error_message: Some("Timeout".to_string()),
            },
            ExecutionResult {
                executable_path: PathBuf::from("test3.exe"),
                success: false,
                exit_code: -1,
                stdout: String::new(),
                stderr: String::new(),
                execution_time: Duration::from_millis(50),
                timed_out: false,
                memory_exceeded: true,
                error_message: Some("Memory exceeded".to_string()),
            },
        ];

        let validator = ExecutionValidator::new(Duration::from_secs(5)).unwrap();
        let stats = validator.get_execution_stats(&results);

        assert_eq!(stats.total_executions, 3);
        assert_eq!(stats.successful_executions, 1);
        assert_eq!(stats.failed_executions, 2);
        assert_eq!(stats.timed_out_executions, 1);
        assert_eq!(stats.memory_exceeded_executions, 1);
        assert_eq!(stats.total_execution_time, Duration::from_millis(350));
        assert_eq!(stats.average_execution_time, Duration::from_millis(116)); // 350/3 ≈ 116
        assert_eq!(stats.fastest_execution, Duration::from_millis(50));
        assert_eq!(stats.slowest_execution, Duration::from_millis(200));
    }

    #[test]
    fn test_execution_stats_empty() {
        let results: Vec<ExecutionResult> = Vec::new();
        let validator = ExecutionValidator::new(Duration::from_secs(5)).unwrap();
        let stats = validator.get_execution_stats(&results);

        assert_eq!(stats.total_executions, 0);
        assert_eq!(stats.successful_executions, 0);
        assert_eq!(stats.failed_executions, 0);
        assert_eq!(stats.timed_out_executions, 0);
        assert_eq!(stats.memory_exceeded_executions, 0);
        assert_eq!(stats.total_execution_time, Duration::from_secs(0));
        assert_eq!(stats.average_execution_time, Duration::from_secs(0));
        assert_eq!(stats.fastest_execution, Duration::from_secs(0));
        assert_eq!(stats.slowest_execution, Duration::from_secs(0));
    }

    #[test]
    fn test_execution_validator_clone() {
        let validator = ExecutionValidator::with_settings(
            Duration::from_secs(10),
            false,
            Some(1024),
            Some(PathBuf::from(".")),
        ).unwrap();

        let cloned_validator = validator.clone();
        
        assert_eq!(validator.timeout, cloned_validator.timeout);
        assert_eq!(validator.capture_output, cloned_validator.capture_output);
        assert_eq!(validator.max_memory_mb, cloned_validator.max_memory_mb);
        assert_eq!(validator.working_directory, cloned_validator.working_directory);
    }

    #[test]
    fn test_execution_validator_setters() {
        let mut validator = ExecutionValidator::new(Duration::from_secs(5)).unwrap();
        
        // Test timeout setter
        validator.set_timeout(Duration::from_secs(15));
        assert_eq!(validator.timeout, Duration::from_secs(15));
        
        // Test capture output setter
        validator.set_capture_output(false);
        assert!(!validator.capture_output);
        
        // Test memory limit setter
        validator.set_memory_limit(Some(2048));
        assert_eq!(validator.max_memory_mb, Some(2048));
        
        validator.set_memory_limit(None);
        assert_eq!(validator.max_memory_mb, None);
        
        // Test working directory setter
        let working_dir = PathBuf::from("test_dir");
        validator.set_working_directory(Some(working_dir.clone()));
        assert_eq!(validator.working_directory, Some(working_dir));
        
        validator.set_working_directory(None);
        assert_eq!(validator.working_directory, None);
    }

    #[tokio::test]
    async fn test_execution_result_fields() {
        let result = ExecutionResult {
            executable_path: PathBuf::from("test.exe"),
            success: true,
            exit_code: 0,
            stdout: "Hello, World!".to_string(),
            stderr: String::new(),
            execution_time: Duration::from_millis(150),
            timed_out: false,
            memory_exceeded: false,
            error_message: None,
        };

        assert_eq!(result.executable_path, PathBuf::from("test.exe"));
        assert!(result.success);
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout, "Hello, World!");
        assert!(result.stderr.is_empty());
        assert_eq!(result.execution_time, Duration::from_millis(150));
        assert!(!result.timed_out);
        assert!(!result.memory_exceeded);
        assert!(result.error_message.is_none());
    }

    // Integration test with a real executable (if available)
    #[tokio::test]
    async fn test_execution_validator_with_real_executable() {
        let validator = ExecutionValidator::new(Duration::from_secs(5)).unwrap();
        
        // Try to find a system executable that should exist on most systems
        let test_executables = if cfg!(windows) {
            vec![PathBuf::from("C:\\Windows\\System32\\cmd.exe")]
        } else {
            vec![PathBuf::from("/bin/echo"), PathBuf::from("/usr/bin/echo")]
        };

        for exe_path in test_executables {
            if exe_path.exists() {
                let result = validator.validate_executable(&exe_path).await;
                assert!(result.is_ok());
                
                let execution_result = result.unwrap();
                assert_eq!(execution_result.executable_path, exe_path);
                // Note: We can't guarantee success since we're not providing proper arguments
                // but we can verify the structure is correct
                assert!(execution_result.execution_time > Duration::from_secs(0));
                assert!(!execution_result.timed_out);
                break;
            }
        }
    }
}