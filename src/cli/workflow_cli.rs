// Command-line interface for Aether build workflow orchestration
// Provides easy access to workflow execution with configuration options

use crate::build_system::{
    BuildWorkflowOrchestrator, WorkflowConfig,
    BuildSystemManager, ErrorHandler, AutoFixEngine, EnvironmentValidator,
    BuildError, AetherCompiler, AetherCompilationConfig,
    CompilationTarget, WindowsExecutableConfig,
    WindowsExecutableTester, WindowsTestConfig, WindowsArchitecture, WindowsSubsystem,
    WindowsVersion, WindowsRuntimeDetector, WindowsEnvironmentHandler,
    aether_compiler_helpers, ErrorType, ErrorSeverity
};
use std::path::{Path, PathBuf};
use std::time::Duration;
use std::io::{self, Write};
use clap::{Parser, Subcommand};
use serde_json;
use colored::*;

/// Aether Build Workflow CLI
#[derive(Parser)]
#[command(name = "aether-workflow")]
#[command(about = "Aether build workflow orchestration tool")]
#[command(version = "0.1.0")]
pub struct WorkflowCli {
    /// Enable verbose output
    #[arg(long, short, global = true)]
    pub verbose: bool,
    
    /// Enable debug output
    #[arg(long, global = true)]
    pub debug: bool,
    
    /// Enable interactive mode
    #[arg(long, short, global = true)]
    pub interactive: bool,
    
    #[command(subcommand)]
    pub command: WorkflowCommand,
}

#[derive(Subcommand)]
pub enum WorkflowCommand {
    /// Execute complete build workflow
    Build {
        /// Source files to compile
        #[arg(required = true)]
        source_files: Vec<PathBuf>,
        
        /// Enable progress reporting
        #[arg(long, default_value = "true")]
        progress: bool,
        
        /// Enable rollback on failure
        #[arg(long, default_value = "true")]
        rollback: bool,
        
        /// Maximum retry attempts
        #[arg(long, default_value = "3")]
        max_retries: usize,
        
        /// Timeout in seconds
        #[arg(long, default_value = "300")]
        timeout: u64,
        
        /// Enable verification tests
        #[arg(long, default_value = "true")]
        verify: bool,
        
        /// Cleanup on failure
        #[arg(long, default_value = "true")]
        cleanup: bool,
        
        /// State file path for persistence
        #[arg(long)]
        state_file: Option<PathBuf>,
        
        /// Configuration file path
        #[arg(long, short)]
        config: Option<PathBuf>,
        
        /// Skip environment validation
        #[arg(long)]
        skip_env_check: bool,
        
        /// Force rebuild even if up-to-date
        #[arg(long)]
        force: bool,
    },
    
    /// Run tests on compiled executables
    Test {
        /// Executable files to test
        #[arg(required = true)]
        executables: Vec<PathBuf>,
        
        /// Run only basic tests
        #[arg(long)]
        basic_only: bool,
        
        /// Run performance benchmarks
        #[arg(long)]
        benchmark: bool,
        
        /// Test timeout in seconds
        #[arg(long, default_value = "60")]
        timeout: u64,
        
        /// Configuration file path
        #[arg(long, short)]
        config: Option<PathBuf>,
    },
    
    /// Automatically fix common build issues
    Fix {
        /// Specific error patterns to fix (optional)
        #[arg(long)]
        pattern: Option<String>,
        
        /// Apply fixes without confirmation
        #[arg(long)]
        auto_apply: bool,
        
        /// Dry run - show fixes without applying
        #[arg(long)]
        dry_run: bool,
        
        /// Configuration file path
        #[arg(long, short)]
        config: Option<PathBuf>,
    },
    
    /// Validate build environment
    Env {
        /// Check specific component (rust, mlir, system)
        #[arg(long)]
        component: Option<String>,
        
        /// Attempt to fix environment issues
        #[arg(long)]
        fix: bool,
        
        /// Show detailed environment information
        #[arg(long)]
        detailed: bool,
    },
    
    /// Resume workflow from saved state
    Resume {
        /// Source files to compile
        #[arg(required = true)]
        source_files: Vec<PathBuf>,
        
        /// State file path
        #[arg(long)]
        state_file: Option<PathBuf>,
    },
    
    /// Show workflow status
    Status {
        /// State file path
        #[arg(long)]
        state_file: Option<PathBuf>,
        
        /// Show detailed status information
        #[arg(long)]
        detailed: bool,
    },
    
    /// Clean workflow state and build artifacts
    Clean {
        /// State file path
        #[arg(long)]
        state_file: Option<PathBuf>,
        
        /// Clean all build artifacts
        #[arg(long)]
        all: bool,
        
        /// Clean target directory
        #[arg(long)]
        target: bool,
    },
    
    /// Generate or manage configuration files
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    
    /// Windows-specific executable generation and testing
    Windows {
        #[command(subcommand)]
        action: WindowsAction,
    },
}

#[derive(Subcommand)]
pub enum ConfigAction {
    /// Generate default configuration file
    Init {
        /// Output configuration file path
        #[arg(long, default_value = "aether_workflow.json")]
        output: PathBuf,
        
        /// Overwrite existing configuration
        #[arg(long)]
        force: bool,
    },
    
    /// Validate configuration file
    Validate {
        /// Configuration file path
        #[arg(required = true)]
        config: PathBuf,
    },
    
    /// Show current configuration
    Show {
        /// Configuration file path
        #[arg(long, default_value = "aether_workflow.json")]
        config: PathBuf,
    },
    
    /// Edit configuration interactively
    Edit {
        /// Configuration file path
        #[arg(long, default_value = "aether_workflow.json")]
        config: PathBuf,
    },
}

#[derive(Subcommand)]
pub enum WindowsAction {
    /// Generate Windows PE executable from Aether source
    Generate {
        /// Aether source file
        #[arg(required = true)]
        source: PathBuf,
        
        /// Output executable path
        #[arg(long, short)]
        output: Option<PathBuf>,
        
        /// Target architecture (x86, x64, arm64)
        #[arg(long, default_value = "x64")]
        arch: String,
        
        /// Subsystem type (console, windows, native)
        #[arg(long, default_value = "console")]
        subsystem: String,
        
        /// Bundle runtime dependencies
        #[arg(long, default_value = "true")]
        bundle_deps: bool,
        
        /// Include debug information
        #[arg(long, default_value = "true")]
        debug_info: bool,
        
        /// Manifest file path
        #[arg(long)]
        manifest: Option<PathBuf>,
        
        /// Icon file path
        #[arg(long)]
        icon: Option<PathBuf>,
        
        /// Additional linker flags
        #[arg(long)]
        linker_flags: Vec<String>,
        
        /// Skip testing after generation
        #[arg(long)]
        no_test: bool,
    },
    
    /// Test Windows executable compatibility and functionality
    Test {
        /// Windows executable to test
        #[arg(required = true)]
        executable: PathBuf,
        
        /// Test timeout in seconds
        #[arg(long, default_value = "60")]
        timeout: u64,
        
        /// Test compatibility with different Windows versions
        #[arg(long, default_value = "true")]
        test_compatibility: bool,
        
        /// Validate runtime dependencies
        #[arg(long, default_value = "true")]
        validate_deps: bool,
        
        /// Target Windows versions to test (win10, win11, server2019, server2022)
        #[arg(long)]
        test_versions: Vec<String>,
        
        /// Target architectures to test (x86, x64, arm64)
        #[arg(long)]
        test_archs: Vec<String>,
    },
    
    /// Detect and analyze Windows runtime dependencies
    Deps {
        /// Windows executable to analyze
        #[arg(required = true)]
        executable: PathBuf,
        
        /// Show detailed dependency information
        #[arg(long)]
        detailed: bool,
        
        /// Check for missing dependencies
        #[arg(long)]
        check_missing: bool,
        
        /// Suggest dependency fixes
        #[arg(long)]
        suggest_fixes: bool,
    },
    
    /// Check Windows environment and system compatibility
    Env {
        /// Show detailed environment information
        #[arg(long)]
        detailed: bool,
        
        /// Check for specific runtime libraries
        #[arg(long)]
        check_runtime: bool,
        
        /// Validate Windows development environment
        #[arg(long)]
        validate_dev_env: bool,
    },
    
    /// Configure Windows-specific build settings
    Configure {
        /// Configuration file to create or update
        #[arg(long, default_value = "windows_config.json")]
        config: PathBuf,
        
        /// Target architecture
        #[arg(long)]
        arch: Option<String>,
        
        /// Subsystem type
        #[arg(long)]
        subsystem: Option<String>,
        
        /// Enable dependency bundling
        #[arg(long)]
        bundle_deps: Option<bool>,
        
        /// Enable debug information
        #[arg(long)]
        debug_info: Option<bool>,
        
        /// Show current configuration
        #[arg(long)]
        show: bool,
    },
}

impl WorkflowCli {
    /// Execute the CLI command
    pub fn execute(self) -> Result<(), Box<dyn std::error::Error>> {
        // Set up logging based on verbosity
        self.setup_logging();
        
        match self.command {
            WorkflowCommand::Build { 
                ref source_files, 
                progress, 
                rollback, 
                max_retries, 
                timeout, 
                verify, 
                cleanup, 
                ref state_file, 
                ref config,
                skip_env_check,
                force
            } => {
                self.execute_build(
                    source_files.clone(), progress, rollback, max_retries, 
                    timeout, verify, cleanup, state_file.clone(), config.clone(),
                    skip_env_check, force
                )
            }
            WorkflowCommand::Test { 
                ref executables, 
                basic_only, 
                benchmark, 
                timeout, 
                ref config 
            } => {
                self.execute_test(executables.clone(), basic_only, benchmark, timeout, config.clone())
            }
            WorkflowCommand::Fix { 
                ref pattern, 
                auto_apply, 
                dry_run, 
                ref config 
            } => {
                self.execute_fix(pattern.clone(), auto_apply, dry_run, config.clone())
            }
            WorkflowCommand::Env { 
                ref component, 
                fix, 
                detailed 
            } => {
                self.execute_env(component.clone(), fix, detailed)
            }
            WorkflowCommand::Resume { ref source_files, ref state_file } => {
                self.execute_resume(source_files.clone(), state_file.clone())
            }
            WorkflowCommand::Status { ref state_file, detailed } => {
                self.execute_status(state_file.clone(), detailed)
            }
            WorkflowCommand::Clean { ref state_file, all, target } => {
                self.execute_clean(state_file.clone(), all, target)
            }
            WorkflowCommand::Config { ref action } => {
                self.execute_config(action)
            }
            WorkflowCommand::Windows { ref action } => {
                self.execute_windows(action)
            }
        }
    }
    
    /// Set up logging based on verbosity flags
    fn setup_logging(&self) {
        if self.debug {
            println!("{}", "Debug mode enabled".blue());
        } else if self.verbose {
            println!("{}", "Verbose mode enabled".blue());
        }
    }
    
    /// Print verbose message if verbose mode is enabled
    fn verbose_println(&self, message: &str) {
        if self.verbose || self.debug {
            println!("{} {}", "[VERBOSE]".blue(), message);
        }
    }
    
    /// Print debug message if debug mode is enabled
    fn debug_println(&self, message: &str) {
        if self.debug {
            println!("{} {}", "[DEBUG]".yellow(), message);
        }
    }
    
    /// Prompt user for confirmation in interactive mode
    fn prompt_confirmation(&self, message: &str) -> Result<bool, Box<dyn std::error::Error>> {
        if !self.interactive {
            return Ok(true);
        }
        
        print!("{} {} (y/N): ", "?".blue(), message);
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        Ok(input.trim().to_lowercase() == "y" || input.trim().to_lowercase() == "yes")
    }
    
    /// Prompt user for input in interactive mode
    fn prompt_input(&self, message: &str) -> Result<String, Box<dyn std::error::Error>> {
        if !self.interactive {
            return Ok(String::new());
        }
        
        print!("{} {}: ", "?".blue(), message);
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        Ok(input.trim().to_string())
    }

    /// Execute build workflow
    fn execute_build(
        &self,
        source_files: Vec<PathBuf>,
        progress: bool,
        rollback: bool,
        max_retries: usize,
        timeout: u64,
        verify: bool,
        cleanup: bool,
        state_file: Option<PathBuf>,
        config_file: Option<PathBuf>,
        skip_env_check: bool,
        _force: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Starting Aether build workflow...", "🚀".green());
        
        if self.interactive && !self.prompt_confirmation("Proceed with build workflow")? {
            println!("Build cancelled by user.");
            return Ok(());
        }
        
        // Load configuration
        let mut workflow_config = if let Some(config_path) = config_file {
            self.verbose_println(&format!("Loading configuration from: {}", config_path.display()));
            self.load_config_from_file(&config_path)?
        } else {
            self.verbose_println("Using default configuration");
            WorkflowConfig::default()
        };
        
        // Environment validation
        if !skip_env_check {
            self.verbose_println("Validating build environment...");
            let env_validator = EnvironmentValidator::new();
            match env_validator.validate_environment() {
                Ok(status) => {
                    if self.verbose {
                        println!("Environment validation: {}", "✅ PASSED".green());
                        self.debug_println(&format!("Environment status: {:?}", status));
                    }
                }
                Err(e) => {
                    println!("{} Environment validation failed: {}", "❌".red(), e);
                    if self.interactive && self.prompt_confirmation("Continue anyway")? {
                        println!("Continuing with potentially invalid environment...");
                    } else {
                        return Err(e.into());
                    }
                }
            }
        }
        
        // Apply CLI overrides
        workflow_config.enable_progress_reporting = progress;
        workflow_config.enable_rollback = rollback;
        workflow_config.max_retry_attempts = max_retries;
        workflow_config.timeout_duration = Duration::from_secs(timeout);
        workflow_config.verification_enabled = verify;
        workflow_config.cleanup_on_failure = cleanup;
        
        if let Some(state_path) = state_file {
            workflow_config.state_file_path = state_path;
            workflow_config.persist_state = true;
        }
        
        // Create and execute workflow
        let mut orchestrator = BuildWorkflowOrchestrator::new(workflow_config);
        let result = orchestrator.execute_workflow(&source_files);
        
        // Report results
        if result.success {
            println!("✅ Build workflow completed successfully!");
            println!("   Duration: {:?}", result.total_duration);
            println!("   Stages completed: {}", result.stages_completed.len());
            println!("   Executables created: {}", result.compiled_executables.len());
            
            if let Some(test_results) = &result.test_results {
                if let Some(basic_tests) = &test_results.basic_tests {
                    println!("   Test success rate: {:.1}%", basic_tests.success_rate * 100.0);
                }
            }
        } else {
            println!("❌ Build workflow failed!");
            println!("   Duration: {:?}", result.total_duration);
            println!("   Stages completed: {}", result.stages_completed.len());
            println!("   Stages failed: {}", result.stages_failed.len());
            
            if !result.error_summary.is_empty() {
                println!("   Errors:");
                for error in &result.error_summary {
                    println!("     - {}", error);
                }
            }
            
            if !result.recovery_actions.is_empty() {
                println!("   Recovery actions taken: {}", result.recovery_actions.len());
            }
            
            return Err("Build workflow failed".into());
        }
        
        Ok(())
    }
    
    /// Execute resume workflow
    fn execute_resume(
        &self,
        source_files: Vec<PathBuf>,
        state_file: Option<PathBuf>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔄 Resuming Aether build workflow...");
        
        let mut workflow_config = WorkflowConfig::default();
        
        if let Some(state_path) = state_file {
            workflow_config.state_file_path = state_path;
            workflow_config.persist_state = true;
        }
        
        let mut orchestrator = BuildWorkflowOrchestrator::new(workflow_config);
        
        match orchestrator.resume_workflow(&source_files) {
            Ok(result) => {
                if result.success {
                    println!("✅ Resumed workflow completed successfully!");
                    println!("   Duration: {:?}", result.total_duration);
                } else {
                    println!("❌ Resumed workflow failed!");
                    for error in &result.error_summary {
                        println!("     - {}", error);
                    }
                    return Err("Resumed workflow failed".into());
                }
            }
            Err(e) => {
                println!("❌ Failed to resume workflow: {}", e);
                return Err(e.into());
            }
        }
        
        Ok(())
    }
    
    /// Execute status command
    fn execute_status(&self, state_file: Option<PathBuf>, detailed: bool) -> Result<(), Box<dyn std::error::Error>> {
        let state_path = state_file.unwrap_or_else(|| PathBuf::from(".aether_build_state.json"));
        
        if !state_path.exists() {
            println!("{} No workflow state found at {}", "ℹ️".blue(), state_path.display());
            return Ok(());
        }
        
        match crate::build_system::workflow::WorkflowStateManager::load_state(&state_path) {
            Ok(state) => {
                println!("{} Workflow Status:", "📊".blue());
                println!("   State file: {}", state_path.display());
                
                if let Some(current_stage) = &state.current_stage {
                    println!("   Current stage: {:?}", current_stage);
                } else {
                    println!("   Current stage: None (workflow not running)");
                }
                
                println!("   Completed stages: {}", state.completed_stages.len());
                if detailed || self.verbose {
                    for stage in &state.completed_stages {
                        println!("     {} {:?}", "✅".green(), stage);
                    }
                } else if !state.completed_stages.is_empty() {
                    println!("     (use --detailed to see all stages)");
                }
                
                if !state.failed_stages.is_empty() {
                    println!("   Failed stages: {}", state.failed_stages.len());
                    if detailed || self.verbose {
                        for stage in &state.failed_stages {
                            println!("     {} {:?}", "❌".red(), stage);
                        }
                    }
                }
                
                if let Some(compiler_path) = &state.compiler_binary_path {
                    println!("   Compiler binary: {}", compiler_path.display());
                }
                
                println!("   Compiled files: {}", state.compiled_files.len());
                if detailed && !state.compiled_files.is_empty() {
                    for file in &state.compiled_files {
                        println!("     - {}", file.display());
                    }
                }
                
                if let Some(error) = &state.last_error {
                    println!("   Last error: {}", error);
                }
                
                let timestamp = std::time::UNIX_EPOCH + std::time::Duration::from_secs(state.timestamp);
                if let Ok(datetime) = timestamp.duration_since(std::time::UNIX_EPOCH) {
                    println!("   Last updated: {} seconds ago", datetime.as_secs());
                }
            }
            Err(e) => {
                println!("{} Failed to read workflow state: {}", "❌".red(), e);
                return Err(e.into());
            }
        }
        
        Ok(())
    }
    
    /// Execute clean command
    fn execute_clean(&self, state_file: Option<PathBuf>, all: bool, target: bool) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Cleaning build artifacts...", "🧹".blue());
        
        if self.interactive && !self.prompt_confirmation("Proceed with cleanup")? {
            println!("Cleanup cancelled by user.");
            return Ok(());
        }
        
        let mut cleaned_items = 0;
        
        // Clean state file
        let state_path = state_file.unwrap_or_else(|| PathBuf::from(".aether_build_state.json"));
        if state_path.exists() {
            std::fs::remove_file(&state_path)?;
            println!("{} Cleaned workflow state file: {}", "✅".green(), state_path.display());
            cleaned_items += 1;
        } else {
            self.verbose_println(&format!("No workflow state file found at {}", state_path.display()));
        }
        
        // Clean target directory if requested
        if target || all {
            let target_dir = PathBuf::from("target");
            if target_dir.exists() {
                self.verbose_println("Removing target directory...");
                std::fs::remove_dir_all(&target_dir)?;
                println!("{} Cleaned target directory", "✅".green());
                cleaned_items += 1;
            }
        }
        
        // Clean all build artifacts if requested
        if all {
            self.verbose_println("Cleaning all build artifacts...");
            
            // Clean compiled executables
            let patterns = ["*.exe", "*.o", "*.ll", "*.wasm", "*.d.ts", "*.js", "*.html"];
            for pattern in &patterns {
                // Simple pattern matching for common build artifacts
                if let Ok(entries) = std::fs::read_dir(".") {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if let Some(extension) = path.extension() {
                            let ext_str = format!(".{}", extension.to_string_lossy());
                            if pattern.ends_with(&ext_str) {
                                if let Some(stem) = path.file_stem() {
                                    let stem_str = stem.to_string_lossy();
                                    // Only remove files that look like build artifacts
                                    if stem_str.contains("test") || stem_str.contains("demo") || 
                                       stem_str.contains("hello") || stem_str.contains("game") ||
                                       stem_str.contains("advanced") {
                                        std::fs::remove_file(&path)?;
                                        self.verbose_println(&format!("Removed: {}", path.display()));
                                        cleaned_items += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if cleaned_items > 0 {
            println!("{} Cleaned {} item(s)", "🎉".green(), cleaned_items);
        } else {
            println!("{} No items to clean", "ℹ️".blue());
        }
        
        Ok(())
    }
    
    /// Execute init-config command
    fn execute_init_config(&self, output: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let default_config = WorkflowConfig::default();
        let config_json = serde_json::to_string_pretty(&default_config)?;
        
        std::fs::write(&output, config_json)?;
        println!("📝 Generated default configuration file: {}", output.display());
        println!("   Edit this file to customize your workflow settings.");
        
        Ok(())
    }
    
    /// Execute test command
    fn execute_test(
        &self,
        executables: Vec<PathBuf>,
        basic_only: bool,
        benchmark: bool,
        _timeout: u64,
        _config_file: Option<PathBuf>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Running tests on executables...", "🧪".green());
        
        if self.interactive && !self.prompt_confirmation("Proceed with testing")? {
            println!("Testing cancelled by user.");
            return Ok(());
        }
        
        let build_manager = BuildSystemManager::new();
        let mut all_passed = true;
        
        for executable in &executables {
            if !executable.exists() {
                println!("{} Executable not found: {}", "❌".red(), executable.display());
                all_passed = false;
                continue;
            }
            
            self.verbose_println(&format!("Testing executable: {}", executable.display()));
            
            // Run basic tests using build manager
            match build_manager.run_verification_tests(executable) {
                Ok(results) => {
                    if results.overall_success {
                        println!("{} Tests passed for: {}", "✅".green(), executable.display());
                        if self.verbose {
                            println!("   Execution success: {}", results.basic_test.execution_success);
                            println!("   Exit code: {}", results.basic_test.exit_code);
                        }
                    } else {
                        println!("{} Tests failed for: {}", "❌".red(), executable.display());
                        println!("   Exit code: {}", results.basic_test.exit_code);
                        all_passed = false;
                    }
                    
                    // Show advanced test results if available
                    if !basic_only {
                        if let Some(advanced) = &results.advanced_tests {
                            self.verbose_println("Advanced test results available");
                            if self.verbose {
                                println!("   Advanced tests: {:?}", advanced);
                            }
                        } else {
                            self.verbose_println("No advanced tests available");
                        }
                    }
                }
                Err(e) => {
                    println!("{} Test error for {}: {}", "❌".red(), executable.display(), e);
                    all_passed = false;
                }
            }
            
            // Run benchmarks if requested
            if benchmark {
                self.verbose_println("Performance benchmarking not yet implemented");
                println!("{} Benchmark skipped for: {} (not implemented)", "⏭️".yellow(), executable.display());
            }
        }
        
        if all_passed {
            println!("{} All tests passed!", "🎉".green());
        } else {
            println!("{} Some tests failed!", "⚠️".yellow());
            return Err("Test failures detected".into());
        }
        
        Ok(())
    }
    
    /// Execute fix command
    fn execute_fix(
        &self,
        pattern: Option<String>,
        auto_apply: bool,
        dry_run: bool,
        _config_file: Option<PathBuf>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Analyzing and fixing build issues...", "🔧".green());
        
        let mut error_handler = ErrorHandler::new();
        let mut fix_engine = AutoFixEngine::new();
        
        // Try to compile to detect errors
        self.verbose_println("Attempting compilation to detect errors...");
        
        let mut build_manager = BuildSystemManager::new();
        let detected_errors = match build_manager.compile_aether_compiler() {
            Ok(_) => {
                println!("{} No build errors detected - compilation successful!", "✅".green());
                return Ok(());
            }
            Err(e) => {
                self.verbose_println(&format!("Compilation failed, analyzing error: {}", e));
                
                // Try to detect error from the error message
                if let Some(detected_error) = error_handler.detect_error(&e.to_string()) {
                    vec![detected_error]
                } else {
                    // Create a generic error if detection fails
                    vec![BuildError {
                        error_type: ErrorType::CompilationError,
                        message: e.to_string(),
                        location: None,
                        suggested_fixes: vec![],
                        severity: ErrorSeverity::Error,
                        context: crate::build_system::ErrorContext::empty(),
                        timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                    }]
                }
            }
        };
        
        println!("Found {} build error(s):", detected_errors.len());
        
        let mut fixes_applied = 0;
        
        for (i, error) in detected_errors.iter().enumerate() {
            println!("  {}. {:?}: {}", i + 1, error.error_type, error.message);
            
            // Filter by pattern if specified
            if let Some(ref filter_pattern) = pattern {
                if !error.message.contains(filter_pattern) {
                    self.verbose_println(&format!("Skipping error (doesn't match pattern): {}", error.message));
                    continue;
                }
            }
            
            // Analyze and suggest fixes
            if let Some(fix_strategy) = fix_engine.analyze_error(error) {
                println!("    Suggested fix: {:?}", fix_strategy);
                
                if dry_run {
                    println!("    {} (dry run - not applied)", "[DRY RUN]".yellow());
                    continue;
                }
                
                let should_apply = if auto_apply {
                    true
                } else if self.interactive {
                    self.prompt_confirmation(&format!("Apply fix for error {}", i + 1))?
                } else {
                    false
                };
                
                if should_apply {
                    let fix_result = fix_engine.apply_fix(&fix_strategy);
                    match fix_result.application_result {
                        Ok(()) => {
                            println!("    {} Fix applied successfully", "✅".green());
                            fixes_applied += 1;
                        }
                        Err(e) => {
                            println!("    {} Failed to apply fix: {}", "❌".red(), e);
                        }
                    }
                } else {
                    println!("    {} Fix skipped", "⏭️".blue());
                }
            } else {
                println!("    {} No automatic fix available", "⚠️".yellow());
            }
        }
        
        if fixes_applied > 0 {
            println!("{} Applied {} fix(es) successfully!", "🎉".green(), fixes_applied);
        } else {
            println!("{} No fixes were applied.", "ℹ️".blue());
        }
        
        Ok(())
    }
    
    /// Execute environment validation command
    fn execute_env(
        &self,
        component: Option<String>,
        fix: bool,
        detailed: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Validating build environment...", "🔍".green());
        
        let env_validator = EnvironmentValidator::new();
        
        match env_validator.validate_environment() {
            Ok(status) => {
                println!("{} Environment validation passed!", "✅".green());
                
                if detailed || self.verbose {
                    println!("\nEnvironment Details:");
                    println!("  Rust toolchain: {:?}", status.rust_toolchain);
                    println!("  Dependencies: {:?}", status.dependencies);
                    println!("  System: {:?}", status.system);
                    println!("  Overall status: {:?}", status.overall_status);
                }
                
                // Component-specific checks
                if let Some(comp) = component {
                    match comp.as_str() {
                        "rust" => {
                            self.verbose_println("Checking Rust toolchain specifically...");
                            match env_validator.validate_rust_toolchain() {
                                Ok(rust_status) => {
                                    println!("  Rust toolchain: {:?}", rust_status);
                                }
                                Err(e) => {
                                    println!("  Rust toolchain error: {}", e);
                                }
                            }
                        }
                        "mlir" => {
                            self.verbose_println("Checking MLIR integration specifically...");
                            println!("  MLIR validation not yet implemented");
                        }
                        "system" => {
                            self.verbose_println("Checking system dependencies specifically...");
                            match env_validator.validate_system_requirements() {
                                Ok(system_status) => {
                                    println!("  System requirements: {:?}", system_status);
                                }
                                Err(e) => {
                                    println!("  System requirements error: {}", e);
                                }
                            }
                        }
                        _ => {
                            println!("{} Unknown component: {}", "⚠️".yellow(), comp);
                        }
                    }
                }
            }
            Err(e) => {
                println!("{} Environment validation failed: {}", "❌".red(), e);
                
                if fix {
                    println!("{} Automatic environment fixes not yet implemented", "⚠️".yellow());
                    println!("Please fix the environment issues manually:");
                    println!("  - Ensure Rust toolchain is installed");
                    println!("  - Check that all dependencies are available");
                    println!("  - Verify system requirements are met");
                } else {
                    return Err(e.into());
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute configuration management command
    fn execute_config(&self, action: &ConfigAction) -> Result<(), Box<dyn std::error::Error>> {
        match action {
            ConfigAction::Init { output, force } => {
                self.execute_config_init(output.clone(), *force)
            }
            ConfigAction::Validate { config } => {
                self.execute_config_validate(config.clone())
            }
            ConfigAction::Show { config } => {
                self.execute_config_show(config.clone())
            }
            ConfigAction::Edit { config } => {
                self.execute_config_edit(config.clone())
            }
        }
    }
    
    /// Execute config init command
    fn execute_config_init(&self, output: PathBuf, force: bool) -> Result<(), Box<dyn std::error::Error>> {
        if output.exists() && !force {
            if self.interactive {
                if !self.prompt_confirmation(&format!("Configuration file {} already exists. Overwrite", output.display()))? {
                    println!("Configuration initialization cancelled.");
                    return Ok(());
                }
            } else {
                return Err(format!("Configuration file {} already exists. Use --force to overwrite.", output.display()).into());
            }
        }
        
        let default_config = WorkflowConfig::default();
        let config_json = serde_json::to_string_pretty(&default_config)?;
        
        std::fs::write(&output, config_json)?;
        println!("{} Generated configuration file: {}", "📝".green(), output.display());
        
        if self.verbose {
            println!("Edit this file to customize your workflow settings.");
            println!("Use 'aether-workflow config validate' to check your configuration.");
        }
        
        Ok(())
    }
    
    /// Execute config validate command
    fn execute_config_validate(&self, config: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Validating configuration: {}", "🔍".blue(), config.display());
        
        if !config.exists() {
            return Err(format!("Configuration file not found: {}", config.display()).into());
        }
        
        match self.load_config_from_file(&config) {
            Ok(_) => {
                println!("{} Configuration is valid!", "✅".green());
            }
            Err(e) => {
                println!("{} Configuration validation failed: {}", "❌".red(), e);
                return Err(e);
            }
        }
        
        Ok(())
    }
    
    /// Execute config show command
    fn execute_config_show(&self, config: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        if !config.exists() {
            return Err(format!("Configuration file not found: {}", config.display()).into());
        }
        
        let config_content = std::fs::read_to_string(&config)?;
        println!("Configuration from {}:", config.display());
        println!("{}", config_content);
        
        Ok(())
    }
    
    /// Execute config edit command
    fn execute_config_edit(&self, config: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        if !self.interactive {
            return Err("Config edit requires interactive mode. Use --interactive flag.".into());
        }
        
        println!("{} Interactive configuration editor", "✏️".blue());
        println!("Configuration file: {}", config.display());
        
        // Load existing config or create default
        let mut workflow_config = if config.exists() {
            self.load_config_from_file(&config)?
        } else {
            println!("Configuration file not found. Creating new configuration.");
            WorkflowConfig::default()
        };
        
        // Interactive editing
        if self.prompt_confirmation("Edit progress reporting settings")? {
            let input = self.prompt_input("Enable progress reporting (true/false)")?;
            if !input.is_empty() {
                workflow_config.enable_progress_reporting = input.parse().unwrap_or(workflow_config.enable_progress_reporting);
            }
        }
        
        if self.prompt_confirmation("Edit rollback settings")? {
            let input = self.prompt_input("Enable rollback on failure (true/false)")?;
            if !input.is_empty() {
                workflow_config.enable_rollback = input.parse().unwrap_or(workflow_config.enable_rollback);
            }
        }
        
        if self.prompt_confirmation("Edit retry settings")? {
            let input = self.prompt_input("Maximum retry attempts")?;
            if !input.is_empty() {
                workflow_config.max_retry_attempts = input.parse().unwrap_or(workflow_config.max_retry_attempts);
            }
        }
        
        // Save updated configuration
        let config_json = serde_json::to_string_pretty(&workflow_config)?;
        std::fs::write(&config, config_json)?;
        
        println!("{} Configuration saved to: {}", "💾".green(), config.display());
        
        Ok(())
    }
    
    /// Execute Windows-specific commands
    fn execute_windows(&self, action: &WindowsAction) -> Result<(), Box<dyn std::error::Error>> {
        match action {
            WindowsAction::Generate { 
                source, output, arch, subsystem, bundle_deps, debug_info, 
                manifest, icon, linker_flags, no_test
            } => {
                self.execute_windows_generate(
                    source, output.as_ref(), arch, subsystem, *bundle_deps, 
                    *debug_info, manifest.as_ref(), icon.as_ref(), linker_flags, *no_test
                )
            }
            WindowsAction::Test { 
                executable, timeout, test_compatibility, validate_deps, 
                test_versions, test_archs 
            } => {
                self.execute_windows_test(
                    executable, *timeout, *test_compatibility, *validate_deps,
                    test_versions, test_archs
                )
            }
            WindowsAction::Deps { 
                executable, detailed, check_missing, suggest_fixes 
            } => {
                self.execute_windows_deps(executable, *detailed, *check_missing, *suggest_fixes)
            }
            WindowsAction::Env { 
                detailed, check_runtime, validate_dev_env 
            } => {
                self.execute_windows_env(*detailed, *check_runtime, *validate_dev_env)
            }
            WindowsAction::Configure { 
                config, arch, subsystem, bundle_deps, debug_info, show 
            } => {
                self.execute_windows_configure(
                    config, arch.as_ref(), subsystem.as_ref(), 
                    *bundle_deps, *debug_info, *show
                )
            }
        }
    }
    
    /// Execute Windows executable generation
    fn execute_windows_generate(
        &self,
        source: &PathBuf,
        output: Option<&PathBuf>,
        arch: &str,
        _subsystem: &str,
        bundle_deps: bool,
        debug_info: bool,
        _manifest: Option<&PathBuf>,
        _icon: Option<&PathBuf>,
        _linker_flags: &[String],
        no_test: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Generating Windows executable from Aether source...", "🏗️".green());
        
        if !source.exists() {
            return Err(format!("Source file not found: {}", source.display()).into());
        }
        
        self.verbose_println(&format!("Source file: {}", source.display()));
        
        // Determine output path
        let output_path = if let Some(out) = output {
            out.clone()
        } else {
            let mut path = source.clone();
            path.set_extension("exe");
            path
        };
        
        self.verbose_println(&format!("Output path: {}", output_path.display()));
        
        // Parse architecture
        let target_arch = match arch.to_lowercase().as_str() {
            "x86" => WindowsArchitecture::X86,
            "x64" | "amd64" => WindowsArchitecture::X64,
            "arm64" => WindowsArchitecture::ARM64,
            _ => return Err(format!("Unsupported architecture: {}", arch).into()),
        };
        
        // Create Windows executable configuration
        let mut windows_config = WindowsExecutableConfig::default();
        windows_config.target_arch = target_arch;
        windows_config.bundle_dependencies = bundle_deps;
        windows_config.include_debug_info = debug_info;
        
        // Create build system manager
        let mut manager = BuildSystemManager::new();
        
        // Compile Aether compiler first
        let compiler_binary = manager.compile_aether_compiler()?;
        self.verbose_println(&format!("Aether compiler ready: {}", compiler_binary.path.display()));
        
        // Create Aether compilation config
        let aether_config = AetherCompilationConfig {
            target: CompilationTarget::WindowsNative,
            optimization_level: if debug_info { 0 } else { 2 },
            debug_info: debug_info,
            compiler_flags: Vec::new(),
            output_directory: output_path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            verbose: self.verbose,
        };
        
        // Create Aether compiler with Windows support
        let mut aether_compiler = AetherCompiler::with_config(compiler_binary.path, aether_config)?;
        
        // Configure Windows-specific settings
        aether_compiler.configure_windows_executable(windows_config)?;
        
        // Compile to Windows executable
        let result = aether_compiler.compile_to_executable(&source, &output_path)?;
        
        if result.success {
            println!("{} Windows executable generated successfully!", "✅".green());
            println!("Output: {}", output_path.display());
            
            if let Some(exe_path) = result.executable_path {
                // Test the executable if requested
                if !no_test {
                    println!("{} Testing Windows executable...", "🧪".yellow());
                    let test_result = aether_compiler.test_windows_executable(&exe_path)?;
                    
                    if test_result.success {
                        println!("{} Windows executable tests passed!", "✅".green());
                    } else {
                        println!("{} Windows executable tests failed!", "❌".red());
                        for exec_result in &test_result.execution_results {
                            if !exec_result.success {
                                println!("  Execution failed with exit code: {}", exec_result.exit_code);
                                if !exec_result.stderr.is_empty() {
                                    println!("  Error: {}", exec_result.stderr);
                                }
                            }
                        }
                    }
                }
            }
        } else {
            println!("{} Windows executable generation failed!", "❌".red());
            if !result.stderr.is_empty() {
                println!("Error: {}", result.stderr);
            }
            return Err("Windows executable generation failed".into());
        }
        
        Ok(())
    }

    /// Execute Windows executable testing
    fn execute_windows_test(
        &self,
        executable: &PathBuf,
        timeout: u64,
        test_compatibility: bool,
        validate_deps: bool,
        test_versions: &[String],
        test_archs: &[String],
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Testing Windows executable...", "🧪".yellow());
        
        if !executable.exists() {
            return Err(format!("Executable not found: {}", executable.display()).into());
        }
        
        // Parse test versions
        let versions: Result<Vec<WindowsVersion>, _> = test_versions.iter()
            .map(|v| match v.to_lowercase().as_str() {
                "windows10" | "win10" => Ok(WindowsVersion::Windows10),
                "windows11" | "win11" => Ok(WindowsVersion::Windows11),
                "server2019" => Ok(WindowsVersion::WindowsServer2019),
                "server2022" => Ok(WindowsVersion::WindowsServer2022),
                _ => Err(format!("Unknown Windows version: {}", v)),
            })
            .collect();
        let versions = versions?;
        
        // Parse test architectures
        let architectures: Result<Vec<WindowsArchitecture>, _> = test_archs.iter()
            .map(|a| match a.to_lowercase().as_str() {
                "x86" => Ok(WindowsArchitecture::X86),
                "x64" | "amd64" => Ok(WindowsArchitecture::X64),
                "arm64" => Ok(WindowsArchitecture::ARM64),
                _ => Err(format!("Unknown architecture: {}", a)),
            })
            .collect();
        let architectures = architectures?;
        
        // Create test configuration
        let test_config = WindowsTestConfig {
            timeout: Duration::from_secs(timeout),
            test_compatibility,
            test_versions: versions,
            test_architectures: architectures,
            validate_dependencies: validate_deps,
        };
        
        // Create tester and run tests
        let mut tester = WindowsExecutableTester::with_config(test_config);
        let test_result = tester.test_executable(executable)?;
        
        // Display results
        if test_result.success {
            println!("{} All Windows executable tests passed!", "✅".green());
        } else {
            println!("{} Some Windows executable tests failed!", "❌".red());
        }
        
        // Show execution results
        for (i, exec_result) in test_result.execution_results.iter().enumerate() {
            println!("  Execution Test {}: {}", i + 1, 
                if exec_result.success { "✅ PASS" } else { "❌ FAIL" });
            if !exec_result.success {
                println!("    Exit Code: {}", exec_result.exit_code);
                if !exec_result.stderr.is_empty() {
                    println!("    Error: {}", exec_result.stderr);
                }
            }
        }
        
        // Show compatibility results
        for compat_result in &test_result.compatibility_results {
            println!("  Compatibility {:?} {:?}: {}", 
                compat_result.windows_version, compat_result.architecture,
                if compat_result.compatible { "✅ COMPATIBLE" } else { "❌ INCOMPATIBLE" });
            for issue in &compat_result.issues {
                println!("    Issue: {}", issue);
            }
        }
        
        // Show dependency results
        for dep_result in &test_result.dependency_results {
            println!("  Dependency {}: {}", dep_result.dependency_name,
                if dep_result.valid { "✅ VALID" } else { "❌ INVALID" });
            if !dep_result.valid {
                for suggestion in &dep_result.suggestions {
                    println!("    Suggestion: {}", suggestion);
                }
            }
        }
        
        Ok(())
    }

    /// Execute Windows dependency analysis
    fn execute_windows_deps(
        &self,
        executable: &PathBuf,
        detailed: bool,
        check_missing: bool,
        suggest_fixes: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Analyzing Windows dependencies...", "🔍".blue());
        
        if !executable.exists() {
            return Err(format!("Executable not found: {}", executable.display()).into());
        }
        
        // Create dependency detector
        let mut detector = WindowsRuntimeDetector::new();
        let dependencies = detector.detect_dependencies(executable)?;
        
        println!("Found {} dependencies:", dependencies.len());
        
        for dep in &dependencies {
            let status = if dep.path.exists() { "✅" } else { "❌" };
            let dep_type = if dep.is_system_dll { "System" } else { "Runtime" };
            
            if detailed {
                println!("  {} {} [{}] - {}", status, dep.name, dep_type, dep.path.display());
                if let Some(version) = &dep.version {
                    println!("    Version: {}", version);
                }
                println!("    Architecture: {:?}", dep.architecture);
            } else {
                println!("  {} {} [{}]", status, dep.name, dep_type);
            }
            
            if check_missing && !dep.path.exists() {
                println!("    ❌ Missing dependency!");
                if suggest_fixes {
                    if dep.is_system_dll {
                        println!("    💡 This is a system DLL - check Windows installation");
                    } else {
                        println!("    💡 Install Visual C++ Redistributable or bundle this DLL");
                    }
                }
            }
        }
        
        let missing_count = dependencies.iter().filter(|d| !d.path.exists()).count();
        if missing_count > 0 {
            println!("\n{} {} missing dependencies found!", "⚠️".yellow(), missing_count);
        } else {
            println!("\n{} All dependencies are available!", "✅".green());
        }
        
        Ok(())
    }

    /// Execute Windows environment analysis
    fn execute_windows_env(
        &self,
        detailed: bool,
        check_runtime: bool,
        validate_dev_env: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Analyzing Windows environment...", "🔍".blue());
        
        // Create environment handler
        let mut env_handler = WindowsEnvironmentHandler::new();
        let runtime_env = env_handler.detect_runtime_environment()?;
        
        // Display system information
        println!("Windows System Information:");
        println!("  Version: {:?}", runtime_env.version);
        println!("  Architecture: {:?}", runtime_env.architecture);
        
        if detailed {
            println!("  Runtime Libraries: {}", runtime_env.runtime_libraries.len());
            for lib in &runtime_env.runtime_libraries {
                println!("    - {}", lib);
            }
            
            println!("  .NET Versions: {}", runtime_env.dotnet_versions.len());
            for version in &runtime_env.dotnet_versions {
                println!("    - .NET {}", version);
            }
            
            println!("  VC++ Redistributables: {}", runtime_env.vcredist_versions.len());
            for version in &runtime_env.vcredist_versions {
                println!("    - VC++ {}", version);
            }
        }
        
        if check_runtime {
            println!("\nRuntime Environment Check:");
            let required_libs = vec!["msvcr140.dll", "vcruntime140.dll", "msvcp140.dll"];
            for lib in required_libs {
                let available = runtime_env.runtime_libraries.contains(&lib.to_string());
                println!("  {} {}", if available { "✅" } else { "❌" }, lib);
            }
        }
        
        if validate_dev_env {
            println!("\nDevelopment Environment Check:");
            
            // Check for common development tools
            let dev_tools = vec![
                ("rustc", "Rust compiler"),
                ("cargo", "Rust package manager"),
                ("git", "Version control"),
                ("cmake", "Build system"),
            ];
            
            for (tool, description) in dev_tools {
                match std::process::Command::new(tool).arg("--version").output() {
                    Ok(output) if output.status.success() => {
                        println!("  ✅ {} ({})", tool, description);
                        if detailed {
                            let version = String::from_utf8_lossy(&output.stdout);
                            println!("    Version: {}", version.lines().next().unwrap_or("Unknown"));
                        }
                    }
                    _ => {
                        println!("  ❌ {} ({}) - Not found", tool, description);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Execute Windows configuration management
    fn execute_windows_configure(
        &self,
        config: &PathBuf,
        arch: Option<&String>,
        subsystem: Option<&String>,
        bundle_deps: Option<bool>,
        debug_info: Option<bool>,
        show: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("{} Managing Windows configuration...", "⚙️".blue());
        
        if show {
            // Show current configuration
            let current_config = WindowsExecutableConfig::default();
            println!("Current Windows Configuration:");
            println!("  Architecture: {:?}", current_config.target_arch);
            println!("  Subsystem: {:?}", current_config.subsystem);
            println!("  Bundle Dependencies: {}", current_config.bundle_dependencies);
            println!("  Include Debug Info: {}", current_config.include_debug_info);
            println!("  Version: {}", current_config.version_info.file_version);
            return Ok(());
        }
        
        // Create new configuration
        let mut windows_config = if config.exists() {
            // Load from file
            let config_content = std::fs::read_to_string(config)?;
            serde_json::from_str(&config_content)?
        } else {
            WindowsExecutableConfig::default()
        };
        
        // Apply command line overrides
        if let Some(arch_str) = arch {
            windows_config.target_arch = match arch_str.to_lowercase().as_str() {
                "x86" => WindowsArchitecture::X86,
                "x64" | "amd64" => WindowsArchitecture::X64,
                "arm64" => WindowsArchitecture::ARM64,
                _ => return Err(format!("Unknown architecture: {}", arch_str).into()),
            };
        }
        
        if let Some(subsystem_str) = subsystem {
            windows_config.subsystem = match subsystem_str.to_lowercase().as_str() {
                "console" => WindowsSubsystem::Console,
                "windows" => WindowsSubsystem::Windows,
                "native" => WindowsSubsystem::Native,
                _ => return Err(format!("Unknown subsystem: {}", subsystem_str).into()),
            };
        }
        
        if let Some(bundle) = bundle_deps {
            windows_config.bundle_dependencies = bundle;
        }
        
        if let Some(debug_info_val) = debug_info {
            windows_config.include_debug_info = debug_info_val;
        }
        
        // Save configuration
        let config_json = serde_json::to_string_pretty(&windows_config)?;
        std::fs::write(config, config_json)?;
        println!("Configuration saved to: {}", config.display());
        
        println!("Windows Configuration Updated:");
        println!("  Architecture: {:?}", windows_config.target_arch);
        println!("  Subsystem: {:?}", windows_config.subsystem);
        println!("  Bundle Dependencies: {}", windows_config.bundle_dependencies);
        println!("  Include Debug Info: {}", windows_config.include_debug_info);
        
        Ok(())
    }
  
    fn load_config_from_file(&self, config_path: &PathBuf) -> Result<WorkflowConfig, Box<dyn std::error::Error>> {
        let config_content = std::fs::read_to_string(config_path)?;
        let config: WorkflowConfig = serde_json::from_str(&config_content)?;
        self.verbose_println(&format!("Loaded configuration from: {}", config_path.display()));
        Ok(config)
    }
}

/// Main entry point for workflow CLI
pub fn run_workflow_cli() -> Result<(), Box<dyn std::error::Error>> {
    let cli = WorkflowCli::parse();
    cli.execute()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    fn create_test_cli(command: WorkflowCommand) -> WorkflowCli {
        WorkflowCli {
            verbose: false,
            debug: false,
            interactive: false,
            command,
        }
    }

    #[test]
    fn test_config_init_command() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.json");
        
        let cli = create_test_cli(WorkflowCommand::Config {
            action: ConfigAction::Init {
                output: config_path.clone(),
                force: false,
            },
        });
        
        let result = cli.execute();
        assert!(result.is_ok());
        assert!(config_path.exists());
        
        // Verify the config file contains valid JSON
        let config_content = fs::read_to_string(&config_path).unwrap();
        let _: WorkflowConfig = serde_json::from_str(&config_content).unwrap();
    }
    
    #[test]
    fn test_config_init_force_overwrite() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.json");
        
        // Create existing file
        fs::write(&config_path, "existing content").unwrap();
        assert!(config_path.exists());
        
        let cli = create_test_cli(WorkflowCommand::Config {
            action: ConfigAction::Init {
                output: config_path.clone(),
                force: true,
            },
        });
        
        let result = cli.execute();
        assert!(result.is_ok());
        assert!(config_path.exists());
        
        // Verify the file was overwritten with valid JSON
        let config_content = fs::read_to_string(&config_path).unwrap();
        assert_ne!(config_content, "existing content");
        let _: WorkflowConfig = serde_json::from_str(&config_content).unwrap();
    }
    
    #[test]
    fn test_config_validate_valid_file() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("valid_config.json");
        
        // Create valid config file
        let default_config = WorkflowConfig::default();
        let config_json = serde_json::to_string_pretty(&default_config).unwrap();
        fs::write(&config_path, config_json).unwrap();
        
        let cli = create_test_cli(WorkflowCommand::Config {
            action: ConfigAction::Validate {
                config: config_path,
            },
        });
        
        let result = cli.execute();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_config_validate_invalid_file() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("invalid_config.json");
        
        // Create invalid JSON file
        fs::write(&config_path, "{ invalid json }").unwrap();
        
        let cli = create_test_cli(WorkflowCommand::Config {
            action: ConfigAction::Validate {
                config: config_path,
            },
        });
        
        let result = cli.execute();
        assert!(result.is_err());
    }
    
    #[test]
    fn test_config_show_existing_file() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("show_config.json");
        
        // Create config file
        let test_content = r#"{"test": "value"}"#;
        fs::write(&config_path, test_content).unwrap();
        
        let cli = create_test_cli(WorkflowCommand::Config {
            action: ConfigAction::Show {
                config: config_path,
            },
        });
        
        let result = cli.execute();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_clean_command_basic() {
        let temp_dir = TempDir::new().unwrap();
        let state_path = temp_dir.path().join("test_state.json");
        
        // Create a dummy state file
        fs::write(&state_path, "{}").unwrap();
        assert!(state_path.exists());
        
        let cli = create_test_cli(WorkflowCommand::Clean {
            state_file: Some(state_path.clone()),
            all: false,
            target: false,
        });
        
        let result = cli.execute();
        assert!(result.is_ok());
        assert!(!state_path.exists());
    }
    
    #[test]
    fn test_clean_command_with_target() {
        let temp_dir = TempDir::new().unwrap();
        let current_dir = std::env::current_dir().unwrap();
        
        // Change to temp directory for this test
        std::env::set_current_dir(&temp_dir).unwrap();
        
        // Create target directory
        let target_dir = temp_dir.path().join("target");
        fs::create_dir(&target_dir).unwrap();
        assert!(target_dir.exists());
        
        let cli = create_test_cli(WorkflowCommand::Clean {
            state_file: None,
            all: false,
            target: true,
        });
        
        let result = cli.execute();
        assert!(result.is_ok());
        assert!(!target_dir.exists());
        
        // Restore original directory
        std::env::set_current_dir(current_dir).unwrap();
    }
    
    #[test]
    fn test_status_command_no_state() {
        let temp_dir = TempDir::new().unwrap();
        let state_path = temp_dir.path().join("nonexistent_state.json");
        
        let cli = create_test_cli(WorkflowCommand::Status {
            state_file: Some(state_path),
            detailed: false,
        });
        
        let result = cli.execute();
        assert!(result.is_ok()); // Should not fail when no state file exists
    }
    
    #[test]
    fn test_env_command_basic() {
        let cli = create_test_cli(WorkflowCommand::Env {
            component: None,
            fix: false,
            detailed: false,
        });
        
        // This test may fail if the environment is not properly set up,
        // but it should not panic
        let _result = cli.execute();
    }
    
    #[test]
    fn test_env_command_with_component() {
        let cli = create_test_cli(WorkflowCommand::Env {
            component: Some("rust".to_string()),
            fix: false,
            detailed: true,
        });
        
        // This test may fail if Rust is not installed,
        // but it should not panic
        let _result = cli.execute();
    }
    
    #[test]
    fn test_test_command_nonexistent_executable() {
        let temp_dir = TempDir::new().unwrap();
        let nonexistent_exe = temp_dir.path().join("nonexistent.exe");
        
        let cli = create_test_cli(WorkflowCommand::Test {
            executables: vec![nonexistent_exe],
            basic_only: true,
            benchmark: false,
            timeout: 60,
            config: None,
        });
        
        let result = cli.execute();
        assert!(result.is_err()); // Should fail when executable doesn't exist
    }
    
    #[test]
    fn test_fix_command_dry_run() {
        let cli = create_test_cli(WorkflowCommand::Fix {
            pattern: None,
            auto_apply: false,
            dry_run: true,
            config: None,
        });
        
        // Dry run should not fail even if there are no errors to fix
        let result = cli.execute();
        // This may succeed or fail depending on the build state, but should not panic
        let _result = result;
    }
    
    #[test]
    fn test_verbose_and_debug_flags() {
        let cli = WorkflowCli {
            verbose: true,
            debug: true,
            interactive: false,
            command: WorkflowCommand::Env {
                component: None,
                fix: false,
                detailed: false,
            },
        };
        
        // Test that verbose and debug flags don't cause panics
        let _result = cli.execute();
    }
    
    #[test]
    fn test_build_command_with_flags() {
        let temp_dir = TempDir::new().unwrap();
        let source_file = temp_dir.path().join("test.ae");
        fs::write(&source_file, "// test aether file").unwrap();
        
        let cli = create_test_cli(WorkflowCommand::Build {
            source_files: vec![source_file],
            progress: true,
            rollback: true,
            max_retries: 1,
            timeout: 30,
            verify: false,
            cleanup: true,
            state_file: None,
            config: None,
            skip_env_check: true,
            force: false,
        });
        
        // This test will likely fail due to missing compiler setup,
        // but it should not panic
        let _result = cli.execute();
    }
}