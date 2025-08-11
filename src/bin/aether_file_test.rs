// Aether File Compilation Testing CLI
// Command-line interface for the comprehensive file compilation testing system

use std::path::PathBuf;
use std::time::Duration;
use clap::{Parser, Subcommand, ValueEnum};
use aether_language::testing::file_compilation_testing::{
    FileCompilationTestOrchestrator, TestingConfig, TestCategory, ReportFormat, TestingError,
    FileCompilationTestReport
};

#[derive(Parser)]
#[command(name = "aether-file-test")]
#[command(about = "Comprehensive testing system for compiling all Aether source files")]
#[command(version = "0.1.0")]
#[command(author = "Aether Language Team")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Project root directory to scan for Aether files
    #[arg(short, long, default_value = ".")]
    project_root: PathBuf,

    /// Path to the aetherc compiler executable
    #[arg(short, long, default_value = "aetherc")]
    compiler: PathBuf,

    /// Output directory for test artifacts
    #[arg(short, long, default_value = "target/file_compilation_tests")]
    output: PathBuf,

    /// Directories to include in testing (comma-separated)
    #[arg(short, long, value_delimiter = ',')]
    directories: Option<Vec<String>>,

    /// Report output format
    #[arg(short = 'f', long, default_value = "console")]
    format: ReportFormatArg,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Configuration file path
    #[arg(long, default_value = "aether-file-test.toml")]
    config: Option<PathBuf>,

    /// Compilation timeout in seconds
    #[arg(long, default_value = "30")]
    compile_timeout: u64,

    /// Execution timeout in seconds
    #[arg(long, default_value = "10")]
    exec_timeout: u64,

    /// Maximum parallel compilation processes
    #[arg(long)]
    max_parallel_compile: Option<usize>,

    /// Maximum parallel execution processes
    #[arg(long)]
    max_parallel_exec: Option<usize>,

    /// Disable generation of additional test files
    #[arg(long)]
    no_generate: bool,

    /// Test categories to generate (comma-separated)
    #[arg(long, value_delimiter = ',')]
    test_categories: Option<Vec<TestCategoryArg>>,

    /// Keep temporary artifacts after testing
    #[arg(long)]
    keep_artifacts: bool,

    /// Save configuration to file and exit
    #[arg(long)]
    save_config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the complete file compilation test suite
    Run {
        /// Only compile files, don't execute them
        #[arg(long)]
        compile_only: bool,

        /// Only execute already compiled files
        #[arg(long)]
        execute_only: bool,

        /// Generate report from existing results
        #[arg(long)]
        report_only: bool,
    },

    /// Generate additional test files only
    Generate {
        /// Specific categories to generate
        #[arg(short, long, value_delimiter = ',')]
        categories: Option<Vec<TestCategoryArg>>,

        /// Output directory for generated files
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Discover and list Aether files
    Discover {
        /// Show detailed file information
        #[arg(short, long)]
        detailed: bool,

        /// Filter by specific directories
        #[arg(short = 'f', long, value_delimiter = ',')]
        filter_dirs: Option<Vec<String>>,
    },

    /// Validate configuration and setup
    Validate,

    /// Show statistics about the project
    Stats,

    /// Clean up test artifacts
    Clean {
        /// Also remove generated test files
        #[arg(long)]
        include_generated: bool,
    },
}

#[derive(Clone, ValueEnum)]
enum ReportFormatArg {
    Console,
    Json,
    Html,
    Markdown,
}

impl From<ReportFormatArg> for ReportFormat {
    fn from(arg: ReportFormatArg) -> Self {
        match arg {
            ReportFormatArg::Console => ReportFormat::Console,
            ReportFormatArg::Json => ReportFormat::Json,
            ReportFormatArg::Html => ReportFormat::Html,
            ReportFormatArg::Markdown => ReportFormat::Markdown,
        }
    }
}

#[derive(Clone, ValueEnum)]
enum TestCategoryArg {
    Core,
    Types,
    Ai,
    Errors,
    Performance,
}

impl From<TestCategoryArg> for TestCategory {
    fn from(arg: TestCategoryArg) -> Self {
        match arg {
            TestCategoryArg::Core => TestCategory::CoreLanguage,
            TestCategoryArg::Types => TestCategory::TypeSystem,
            TestCategoryArg::Ai => TestCategory::AIFeatures,
            TestCategoryArg::Errors => TestCategory::ErrorHandling,
            TestCategoryArg::Performance => TestCategory::Performance,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Handle configuration saving first
    if let Some(ref config_path) = cli.save_config {
        let config = build_config_from_args(&cli)?;
        config.save_to_file(&config_path)?;
        println!("Configuration saved to: {}", config_path.display());
        return Ok(());
    }

    // Load configuration from file if specified, otherwise build from args
    let config = if let Some(config_path) = &cli.config {
        let mut config = TestingConfig::load_from_file(config_path)?;
        // Override with command line arguments
        override_config_with_args(&mut config, &cli);
        config
    } else {
        build_config_from_args(&cli)?
    };

    // Set up logging based on verbosity
    if config.verbose {
        println!("Aether File Compilation Testing System");
        println!("Project root: {}", config.project_root.display());
        println!("Compiler: {}", config.compiler_path.display());
        println!("Output directory: {}", config.output_directory.display());
        println!("Report format: {:?}", config.report_format);
        println!();
    }

    // Execute the appropriate command
    match cli.command.unwrap_or(Commands::Run { 
        compile_only: false, 
        execute_only: false, 
        report_only: false 
    }) {
        Commands::Run { compile_only, execute_only, report_only } => {
            run_test_suite(config, compile_only, execute_only, report_only).await?;
        }
        Commands::Generate { categories, output } => {
            generate_test_files(config, categories, output).await?;
        }
        Commands::Discover { detailed, filter_dirs } => {
            discover_files(config, detailed, filter_dirs).await?;
        }
        Commands::Validate => {
            validate_setup(config).await?;
        }
        Commands::Stats => {
            show_statistics(config).await?;
        }
        Commands::Clean { include_generated } => {
            clean_artifacts(config, include_generated).await?;
        }
    }

    Ok(())
}

/// Build configuration from command line arguments
fn build_config_from_args(cli: &Cli) -> Result<TestingConfig, TestingError> {
    let test_directories = cli.directories.clone().unwrap_or_else(|| {
        vec!["examples".to_string(), "tests".to_string()]
    });

    let test_categories = cli.test_categories.clone()
        .map(|cats| cats.into_iter().map(|c| c.into()).collect())
        .unwrap_or_else(|| vec![
            TestCategory::CoreLanguage,
            TestCategory::TypeSystem,
            TestCategory::AIFeatures,
            TestCategory::ErrorHandling,
        ]);

    Ok(TestingConfig {
        project_root: cli.project_root.clone(),
        compiler_path: cli.compiler.clone(),
        output_directory: cli.output.clone(),
        test_directories,
        compilation_timeout: Duration::from_secs(cli.compile_timeout),
        execution_timeout: Duration::from_secs(cli.exec_timeout),
        generate_additional_tests: !cli.no_generate,
        test_categories,
        report_format: cli.format.clone().into(),
        cleanup_artifacts: !cli.keep_artifacts,
        max_parallel_compilations: cli.max_parallel_compile.unwrap_or_else(num_cpus::get),
        max_parallel_executions: cli.max_parallel_exec.unwrap_or_else(|| num_cpus::get() / 2),
        verbose: cli.verbose,
    })
}

/// Override configuration with command line arguments
fn override_config_with_args(config: &mut TestingConfig, cli: &Cli) {
    config.project_root = cli.project_root.clone();
    config.compiler_path = cli.compiler.clone();
    config.output_directory = cli.output.clone();
    config.verbose = cli.verbose;
    config.report_format = cli.format.clone().into();
    config.compilation_timeout = Duration::from_secs(cli.compile_timeout);
    config.execution_timeout = Duration::from_secs(cli.exec_timeout);
    config.generate_additional_tests = !cli.no_generate;
    config.cleanup_artifacts = !cli.keep_artifacts;

    if let Some(dirs) = &cli.directories {
        config.test_directories = dirs.clone();
    }

    if let Some(categories) = &cli.test_categories {
        config.test_categories = categories.iter().map(|c| c.clone().into()).collect();
    }

    if let Some(max_compile) = cli.max_parallel_compile {
        config.max_parallel_compilations = max_compile;
    }

    if let Some(max_exec) = cli.max_parallel_exec {
        config.max_parallel_executions = max_exec;
    }
}

/// Run the complete test suite
async fn run_test_suite(
    config: TestingConfig,
    compile_only: bool,
    execute_only: bool,
    report_only: bool,
) -> Result<(), TestingError> {
    if config.verbose {
        println!("Starting file compilation test suite...");
    }

    let mut orchestrator = FileCompilationTestOrchestrator::new(config)?;
    
    // Validate setup before running
    orchestrator.validate_setup()?;

    if report_only {
        // TODO: Implement report-only mode by loading existing results
        println!("Report-only mode not yet implemented");
        return Ok(());
    }

    if execute_only {
        // TODO: Implement execute-only mode
        println!("Execute-only mode not yet implemented");
        return Ok(());
    }

    if compile_only {
        // TODO: Implement compile-only mode
        println!("Compile-only mode not yet implemented");
        return Ok(());
    }

    // Run the complete test suite
    let report = orchestrator.run_complete_test_suite().await?;

    // Display results based on format
    match report.config.report_format {
        ReportFormat::Console => {
            print_console_report(&report);
        }
        ReportFormat::Json => {
            let json_output = serde_json::to_string_pretty(&report)
                .map_err(|e| TestingError::ReportGeneration(format!("JSON serialization failed: {}", e)))?;
            println!("{}", json_output);
        }
        ReportFormat::Html => {
            // TODO: Implement HTML report generation
            println!("HTML report generation not yet implemented");
        }
        ReportFormat::Markdown => {
            // TODO: Implement Markdown report generation
            println!("Markdown report generation not yet implemented");
        }
    }

    Ok(())
}

/// Generate additional test files
async fn generate_test_files(
    config: TestingConfig,
    _categories: Option<Vec<TestCategoryArg>>,
    _output: Option<PathBuf>,
) -> Result<(), TestingError> {
    if config.verbose {
        println!("Generating additional test files...");
    }

    let _orchestrator = FileCompilationTestOrchestrator::new(config)?;
    
    // TODO: Implement test file generation with specific categories and output
    println!("Test file generation not yet fully implemented");
    
    Ok(())
}

/// Discover and list Aether files
async fn discover_files(
    config: TestingConfig,
    _detailed: bool,
    _filter_dirs: Option<Vec<String>>,
) -> Result<(), TestingError> {
    if config.verbose {
        println!("Discovering Aether files...");
    }

    let _orchestrator = FileCompilationTestOrchestrator::new(config)?;
    
    // TODO: Implement file discovery with filtering and detailed output
    println!("File discovery not yet fully implemented");
    
    Ok(())
}

/// Validate configuration and setup
async fn validate_setup(config: TestingConfig) -> Result<(), TestingError> {
    println!("Validating configuration and setup...");

    let orchestrator = FileCompilationTestOrchestrator::new(config)?;
    orchestrator.validate_setup()?;

    println!("✓ Configuration is valid");
    println!("✓ Compiler is accessible");
    println!("✓ Project root exists and is readable");
    println!("✓ Output directory can be created");
    
    Ok(())
}

/// Show project statistics
async fn show_statistics(config: TestingConfig) -> Result<(), TestingError> {
    if config.verbose {
        println!("Gathering project statistics...");
    }

    let _orchestrator = FileCompilationTestOrchestrator::new(config)?;
    
    // TODO: Implement statistics gathering and display
    println!("Statistics gathering not yet fully implemented");
    
    Ok(())
}

/// Clean up test artifacts
async fn clean_artifacts(config: TestingConfig, include_generated: bool) -> Result<(), TestingError> {
    if config.verbose {
        println!("Cleaning up test artifacts...");
    }

    // Remove output directory
    if config.output_directory.exists() {
        std::fs::remove_dir_all(&config.output_directory)?;
        println!("Removed output directory: {}", config.output_directory.display());
    }

    if include_generated {
        // TODO: Remove generated test files
        println!("Generated file cleanup not yet implemented");
    }

    println!("Cleanup completed");
    Ok(())
}

/// Print console report for test results
fn print_console_report(report: &FileCompilationTestReport) {
    println!("\n=== Aether File Compilation Test Report ===");
    println!();
    
    // Summary
    println!("Summary:");
    println!("  Total files discovered: {}", report.discovered_files.len());
    println!("  Generated test files: {}", report.generated_files.len());
    println!("  Compilation attempts: {}", report.compilation_results.len());
    println!("  Successful compilations: {}", 
        report.compilation_results.iter().filter(|r| r.success).count());
    println!("  Execution attempts: {}", report.execution_results.len());
    println!("  Successful executions: {}", 
        report.execution_results.iter().filter(|r| r.success).count());
    println!();

    // Compilation failures
    let failed_compilations: Vec<_> = report.compilation_results
        .iter()
        .filter(|r| !r.success)
        .collect();

    if !failed_compilations.is_empty() {
        println!("Compilation Failures:");
        for failure in failed_compilations {
            println!("  ❌ {}", failure.source_file.display());
            if !failure.stderr.is_empty() {
                println!("     Error: {}", failure.stderr.lines().next().unwrap_or("Unknown error"));
            }
        }
        println!();
    }

    // Execution failures
    let failed_executions: Vec<_> = report.execution_results
        .iter()
        .filter(|r| !r.success)
        .collect();

    if !failed_executions.is_empty() {
        println!("Execution Failures:");
        for failure in failed_executions {
            println!("  ❌ {}", failure.executable_path.display());
            println!("     Exit code: {}", failure.exit_code);
            if !failure.stderr.is_empty() {
                println!("     Error: {}", failure.stderr.lines().next().unwrap_or("Unknown error"));
            }
        }
        println!();
    }

    // Success summary
    let total_success = report.compilation_results.iter().filter(|r| r.success).count() == report.compilation_results.len()
        && report.execution_results.iter().filter(|r| r.success).count() == report.execution_results.len();

    if total_success {
        println!("🎉 All tests passed successfully!");
    } else {
        println!("⚠️  Some tests failed. See details above.");
    }
}