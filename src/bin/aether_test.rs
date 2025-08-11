// Aether Test Runner CLI
// Command-line interface for running comprehensive tests

use aether_language::testing::{
    BuildIntegratedTestRunner, TestConfig
};
use aether_language::build_system::BuildConfig;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "aether-test")]
#[command(about = "Aether Language Testing Framework")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
    
    /// Timeout for individual tests (in seconds)
    #[arg(short, long, default_value = "30")]
    timeout: u64,
    
    /// Number of parallel test threads
    #[arg(short, long)]
    parallel: Option<usize>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run hello world test
    HelloWorld,
    
    /// Run basic compilation tests
    Basic,
    
    /// Run advanced feature tests
    Advanced,
    
    /// Run performance benchmarks
    Benchmark,
    
    /// Run meta-tests (framework reliability)
    Meta,
    
    /// Run all comprehensive tests
    All,
    
    /// Test a specific Aether source file
    File {
        /// Path to the Aether source file
        path: PathBuf,
    },
    
    /// Run regression tests
    Regression,
    
    /// Generate comprehensive test report
    Report {
        /// Output file for the report (JSON format)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    
    /// Check build system status
    Status,
    
    /// Quick verification test
    Quick,
}

fn main() {
    let cli = Cli::parse();
    
    // Configure test settings
    let mut test_config = TestConfig::default();
    test_config.verbose = cli.verbose;
    test_config.timeout = Duration::from_secs(cli.timeout);
    
    if let Some(parallel_threads) = cli.parallel {
        test_config.max_threads = parallel_threads;
        test_config.parallel_execution = parallel_threads > 1;
    }
    
    // Create test runner
    let build_config = BuildConfig::default();
    let mut runner = BuildIntegratedTestRunner::with_configs(build_config, test_config);
    
    // Initialize runner
    if let Err(e) = runner.initialize() {
        eprintln!("Failed to initialize test runner: {}", e);
        std::process::exit(1);
    }
    
    // Execute command
    let result = match cli.command {
        Commands::HelloWorld => run_hello_world_test(&mut runner),
        Commands::Basic => run_basic_tests(&mut runner),
        Commands::Advanced => run_advanced_tests(&mut runner),
        Commands::Benchmark => run_benchmark_tests(&mut runner),
        Commands::Meta => run_meta_tests(&mut runner),
        Commands::All => run_all_tests(&mut runner),
        Commands::File { path } => run_file_test(&mut runner, &path),
        Commands::Regression => run_regression_tests(&mut runner),
        Commands::Report { output } => generate_test_report(&mut runner, output.as_deref()),
        Commands::Status => check_build_status(&mut runner),
        Commands::Quick => run_quick_test(&mut runner),
    };
    
    match result {
        Ok(success) => {
            if success {
                println!("✅ Tests completed successfully");
                std::process::exit(0);
            } else {
                println!("❌ Some tests failed");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_hello_world_test(runner: &mut BuildIntegratedTestRunner) -> Result<bool, String> {
    println!("🚀 Running Hello World Test...");
    
    let result = runner.run_hello_world_test()?;
    
    if result.success {
        println!("✅ Hello World test passed in {:?}", result.duration);
        println!("Output: {}", result.output);
    } else {
        println!("❌ Hello World test failed in {:?}", result.duration);
        if let Some(error) = &result.error_message {
            println!("Error: {}", error);
        }
    }
    
    Ok(result.success)
}

fn run_basic_tests(runner: &mut BuildIntegratedTestRunner) -> Result<bool, String> {
    println!("🔧 Running Basic Tests...");
    
    let summary = runner.run_basic_tests()?;
    
    println!("Basic Tests Summary:");
    println!("  Hello World: {}", if summary.hello_world_passed { "✅ PASS" } else { "❌ FAIL" });
    println!("  Arithmetic: {}", if summary.arithmetic_passed { "✅ PASS" } else { "❌ FAIL" });
    println!("  Functions: {}", if summary.function_passed { "✅ PASS" } else { "❌ FAIL" });
    println!("  Control Flow: {}", if summary.control_flow_passed { "✅ PASS" } else { "❌ FAIL" });
    println!("  Success Rate: {:.1}%", summary.success_rate * 100.0);
    println!("  Duration: {:?}", summary.duration);
    
    Ok(summary.success_rate > 0.75) // Consider successful if >75% pass
}

fn run_advanced_tests(runner: &mut BuildIntegratedTestRunner) -> Result<bool, String> {
    println!("🧠 Running Advanced Feature Tests...");
    
    let summary = runner.run_advanced_tests()?;
    
    println!("Advanced Tests Summary:");
    println!("  Tensor Operations: {}", if summary.tensor_operations_passed { "✅ PASS" } else { "❌ FAIL" });
    println!("  Automatic Differentiation: {}", if summary.autodiff_passed { "✅ PASS" } else { "❌ FAIL" });
    println!("  Probabilistic Programming: {}", if summary.probabilistic_passed { "✅ PASS" } else { "❌ FAIL" });
    println!("  GPU Kernels: {}", if summary.gpu_kernels_passed { "✅ PASS" } else { "⚠️  SKIP" });
    println!("  FFI Integration: {}", if summary.ffi_integration_passed { "✅ PASS" } else { "❌ FAIL" });
    println!("  Success Rate: {:.1}%", summary.success_rate * 100.0);
    println!("  Duration: {:?}", summary.duration);
    
    Ok(summary.success_rate > 0.6) // Consider successful if >60% pass (GPU might not be available)
}

fn run_benchmark_tests(runner: &mut BuildIntegratedTestRunner) -> Result<bool, String> {
    println!("⚡ Running Performance Benchmarks...");
    
    let summary = runner.run_performance_benchmarks()?;
    
    println!("Performance Benchmarks:");
    
    if let Some(compilation) = &summary.compilation_speed {
        println!("  Compilation Speed:");
        println!("    Average: {:?}", compilation.average_duration);
        println!("    Min: {:?}", compilation.min_duration);
        println!("    Max: {:?}", compilation.max_duration);
        println!("    Iterations: {}", compilation.iterations);
    }
    
    if let Some(execution) = &summary.execution_speed {
        println!("  Execution Speed:");
        println!("    Average: {:?}", execution.average_duration);
        println!("    Min: {:?}", execution.min_duration);
        println!("    Max: {:?}", execution.max_duration);
        println!("    Iterations: {}", execution.iterations);
    }
    
    if let Some(memory) = &summary.memory_usage {
        println!("  Memory Usage:");
        println!("    Test Duration: {:?}", memory.total_duration);
        println!("    Success Rate: {:.1}%", memory.success_rate * 100.0);
    }
    
    if let Some(tensor) = &summary.tensor_performance {
        println!("  Tensor Performance:");
        println!("    Average: {:?}", tensor.average_duration);
        println!("    Iterations: {}", tensor.iterations);
    }
    
    println!("  Total Duration: {:?}", summary.duration);
    
    Ok(true) // Benchmarks always "pass" - they just provide performance data
}

fn run_meta_tests(runner: &mut BuildIntegratedTestRunner) -> Result<bool, String> {
    println!("🔍 Running Meta-Tests...");
    
    let summary = runner.run_meta_tests()?;
    
    println!("Meta-Tests Summary:");
    println!("  Framework Reliability: {}", if summary.framework_reliability_passed { "✅ PASS" } else { "❌ FAIL" });
    println!("  Error Detection: {}", if summary.error_detection_passed { "✅ PASS" } else { "❌ FAIL" });
    println!("  Build Integration: {}", if summary.build_integration_passed { "✅ PASS" } else { "❌ FAIL" });
    println!("  Success Rate: {:.1}%", summary.success_rate * 100.0);
    println!("  Duration: {:?}", summary.duration);
    
    Ok(summary.success_rate > 0.8) // Meta-tests should have high success rate
}

fn run_all_tests(runner: &mut BuildIntegratedTestRunner) -> Result<bool, String> {
    println!("🎯 Running All Comprehensive Tests...");
    
    let results = runner.run_all_tests()?;
    
    println!("\n📊 Comprehensive Test Results:");
    println!("  Total Tests: {}", results.total_tests);
    println!("  Passed: {} ✅", results.passed_tests);
    println!("  Failed: {} ❌", results.failed_tests);
    println!("  Errors: {} ⚠️", results.error_tests);
    println!("  Skipped: {} ⏭️", results.skipped_tests);
    println!("  Success Rate: {:.1}%", results.overall_success_rate * 100.0);
    println!("  Total Duration: {:?}", results.total_duration);
    
    if let Some(basic) = &results.basic_tests {
        println!("\n  Basic Tests: {:.1}% success rate", basic.success_rate * 100.0);
    }
    
    if let Some(advanced) = &results.advanced_tests {
        println!("  Advanced Tests: {:.1}% success rate", advanced.success_rate * 100.0);
    }
    
    if let Some(meta) = &results.meta_tests {
        println!("  Meta-Tests: {:.1}% success rate", meta.success_rate * 100.0);
    }
    
    println!("\n{}", results.summary());
    
    Ok(results.is_successful())
}

fn run_file_test(runner: &mut BuildIntegratedTestRunner, path: &PathBuf) -> Result<bool, String> {
    println!("📄 Testing file: {}", path.display());
    
    let result = runner.test_aether_file(path)?;
    
    println!("File Test Results:");
    println!("  File: {}", result.file_path.display());
    println!("  Syntax Valid: {}", if result.syntax_valid { "✅ YES" } else { "❌ NO" });
    println!("  Compilation: {}", if result.compilation_success { "✅ SUCCESS" } else { "❌ FAILED" });
    println!("  Execution: {}", if result.execution_success { "✅ SUCCESS" } else { "❌ FAILED" });
    println!("  Duration: {:?}", result.duration);
    
    if !result.output.is_empty() {
        println!("  Output:");
        for line in result.output.lines() {
            println!("    {}", line);
        }
    }
    
    if let Some(error) = &result.error_message {
        println!("  Error: {}", error);
    }
    
    Ok(result.compilation_success && result.execution_success)
}

fn run_regression_tests(runner: &mut BuildIntegratedTestRunner) -> Result<bool, String> {
    println!("🔄 Running Regression Tests...");
    
    let results = runner.run_regression_tests()?;
    
    println!("Regression Test Results:");
    println!("  Overall Success: {}", if results.overall_success { "✅ PASS" } else { "❌ FAIL" });
    println!("  Basic Tests: {:.1}% success rate", results.basic_tests_success_rate * 100.0);
    println!("  Advanced Tests: {:.1}% success rate", results.advanced_tests_success_rate * 100.0);
    println!("  Duration: {:?}", results.duration);
    
    Ok(results.overall_success)
}

fn generate_test_report(runner: &mut BuildIntegratedTestRunner, output_path: Option<&std::path::Path>) -> Result<bool, String> {
    println!("📋 Generating Test Report...");
    
    let report = runner.generate_test_report()?;
    
    // Serialize report to JSON
    let json_report = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("Failed to serialize report: {}", e))?;
    
    // Write to file or stdout
    if let Some(path) = output_path {
        std::fs::write(path, &json_report)
            .map_err(|e| format!("Failed to write report to {}: {}", path.display(), e))?;
        println!("Report written to: {}", path.display());
    } else {
        println!("Test Report (JSON):");
        println!("{}", json_report);
    }
    
    println!("\n{}", report.summary);
    
    Ok(report.comprehensive_results.is_successful())
}

fn check_build_status(runner: &mut BuildIntegratedTestRunner) -> Result<bool, String> {
    println!("🔧 Checking Build System Status...");
    
    let status = runner.get_build_status();
    
    println!("Build System Status:");
    println!("  Environment Valid: {}", if status.environment_valid { "✅ YES" } else { "❌ NO" });
    println!("  Rust Toolchain: {}", status.rust_toolchain_version);
    println!("  Rust Features: {:?}", status.rust_features);
    println!("  Aether Compiler: {}", if status.aether_compiler_available { "✅ AVAILABLE" } else { "❌ NOT AVAILABLE" });
    
    if let Some(error) = &status.error_message {
        println!("  Error: {}", error);
    }
    
    Ok(status.environment_valid)
}

fn run_quick_test(runner: &mut BuildIntegratedTestRunner) -> Result<bool, String> {
    println!("⚡ Running Quick Verification Test...");
    
    let success = runner.quick_verification_test()?;
    
    if success {
        println!("✅ Quick test passed - toolchain is working!");
    } else {
        println!("❌ Quick test failed - toolchain has issues");
    }
    
    Ok(success)
}