// Build-Integrated Test Runner
// Main interface for running comprehensive tests with build system integration

use super::{TestConfig, comprehensive_test_runner::*};
use crate::build_system::{BuildSystemManager, BuildConfig};
use std::path::Path;
use std::time::Instant;
use serde::{Serialize, Deserialize};

/// Main test runner that integrates with the build system
pub struct BuildIntegratedTestRunner {
    build_manager: BuildSystemManager,
    test_config: TestConfig,
    comprehensive_runner: Option<ComprehensiveTestRunner>,
}

impl BuildIntegratedTestRunner {
    /// Create new build-integrated test runner
    pub fn new() -> Self {
        let build_manager = BuildSystemManager::new();
        let test_config = TestConfig::default();
        
        Self {
            build_manager,
            test_config,
            comprehensive_runner: None,
        }
    }

    /// Create with custom build and test configurations
    pub fn with_configs(build_config: BuildConfig, test_config: TestConfig) -> Self {
        let build_manager = BuildSystemManager::with_config(build_config);
        
        Self {
            build_manager,
            test_config,
            comprehensive_runner: None,
        }
    }

    /// Initialize the comprehensive test runner
    pub fn initialize(&mut self) -> Result<(), String> {
        // Validate build environment first
        match self.build_manager.validate_environment() {
            Ok(env_status) => {
                if !matches!(env_status.overall_status, crate::build_system::ValidationStatus::Valid) {
                    return Err("Build environment validation failed. Please fix environment issues first.".to_string());
                }
            }
            Err(e) => {
                return Err(format!("Environment validation error: {}", e));
            }
        }

        // Create comprehensive test runner
        let comprehensive_runner = ComprehensiveTestRunner::new(
            self.test_config.clone(),
            BuildSystemManager::new() // Create a new instance since BuildSystemManager doesn't implement Clone
        );
        
        self.comprehensive_runner = Some(comprehensive_runner);
        Ok(())
    }

    /// Run hello world compilation and execution test
    pub fn run_hello_world_test(&mut self) -> Result<HelloWorldTestResult, String> {
        self.ensure_initialized()?;
        
        println!("Running Hello World Test...");
        let start_time = Instant::now();
        
        let hello_world_source = r#"
(func main ()
  (call print "Hello, World from Aether!")
  (return 0))
"#;

        let result = self.comprehensive_runner.as_mut().unwrap()
            .compile_and_run_source("hello_world_test.ae", hello_world_source);

        let duration = start_time.elapsed();

        match result {
            Ok(output) => {
                let success = output.contains("Hello, World from Aether!");
                Ok(HelloWorldTestResult {
                    success,
                    output,
                    error_message: None,
                    duration,
                })
            }
            Err(error) => {
                Ok(HelloWorldTestResult {
                    success: false,
                    output: String::new(),
                    error_message: Some(error),
                    duration,
                })
            }
        }
    }

    /// Run basic compilation tests
    pub fn run_basic_tests(&mut self) -> Result<BasicTestSummary, String> {
        self.ensure_initialized()?;
        
        println!("Running Basic Tests...");
        let start_time = Instant::now();
        
        let runner = self.comprehensive_runner.as_mut().unwrap();
        let basic_results = runner.run_basic_tests();
        
        Ok(BasicTestSummary {
            hello_world_passed: basic_results.hello_world_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            arithmetic_passed: basic_results.arithmetic_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            function_passed: basic_results.function_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            control_flow_passed: basic_results.control_flow_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            success_rate: basic_results.success_rate,
            duration: start_time.elapsed(),
            details: basic_results,
        })
    }

    /// Run advanced feature tests
    pub fn run_advanced_tests(&mut self) -> Result<AdvancedTestSummary, String> {
        self.ensure_initialized()?;
        
        println!("Running Advanced Feature Tests...");
        let start_time = Instant::now();
        
        let runner = self.comprehensive_runner.as_mut().unwrap();
        let advanced_results = runner.run_advanced_tests();
        
        Ok(AdvancedTestSummary {
            tensor_operations_passed: advanced_results.tensor_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            autodiff_passed: advanced_results.autodiff_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            probabilistic_passed: advanced_results.probabilistic_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            gpu_kernels_passed: advanced_results.gpu_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            ffi_integration_passed: advanced_results.ffi_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            success_rate: advanced_results.success_rate,
            duration: start_time.elapsed(),
            details: advanced_results,
        })
    }

    /// Run performance benchmarks
    pub fn run_performance_benchmarks(&mut self) -> Result<PerformanceBenchmarkSummary, String> {
        self.ensure_initialized()?;
        
        println!("Running Performance Benchmarks...");
        let start_time = Instant::now();
        
        let runner = self.comprehensive_runner.as_mut().unwrap();
        let benchmark_results = runner.run_performance_benchmarks();
        
        Ok(PerformanceBenchmarkSummary {
            compilation_speed: benchmark_results.compilation_benchmark.clone(),
            execution_speed: benchmark_results.execution_benchmark.clone(),
            memory_usage: benchmark_results.memory_benchmark.clone(),
            tensor_performance: benchmark_results.tensor_benchmark.clone(),
            duration: start_time.elapsed(),
            details: benchmark_results,
        })
    }

    /// Run meta-tests to verify testing framework reliability
    pub fn run_meta_tests(&mut self) -> Result<MetaTestSummary, String> {
        self.ensure_initialized()?;
        
        println!("Running Meta-Tests...");
        let start_time = Instant::now();
        
        let runner = self.comprehensive_runner.as_mut().unwrap();
        let meta_results = runner.run_meta_tests();
        
        Ok(MetaTestSummary {
            framework_reliability_passed: meta_results.framework_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            error_detection_passed: meta_results.error_detection_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            build_integration_passed: meta_results.build_integration_test.as_ref()
                .map_or(false, |t| t.status == crate::testing::TestStatus::Passed),
            success_rate: meta_results.success_rate,
            duration: start_time.elapsed(),
            details: meta_results,
        })
    }

    /// Run all comprehensive tests
    pub fn run_all_tests(&mut self) -> Result<ComprehensiveTestResults, String> {
        self.ensure_initialized()?;
        
        println!("Running All Comprehensive Tests...");
        let runner = self.comprehensive_runner.as_mut().unwrap();
        Ok(runner.run_all_tests())
    }

    /// Test specific Aether source file
    pub fn test_aether_file(&mut self, source_path: &Path) -> Result<FileTestResult, String> {
        self.ensure_initialized()?;
        
        if !source_path.exists() {
            return Err(format!("Source file not found: {}", source_path.display()));
        }

        println!("Testing Aether file: {}", source_path.display());
        let start_time = Instant::now();

        // First, validate syntax
        let syntax_validation = match self.build_manager.validate_aether_syntax(source_path) {
            Ok(validation) => validation,
            Err(e) => {
                return Ok(FileTestResult {
                    file_path: source_path.to_path_buf(),
                    compilation_success: false,
                    execution_success: false,
                    syntax_valid: false,
                    output: String::new(),
                    error_message: Some(format!("Syntax validation failed: {}", e)),
                    duration: start_time.elapsed(),
                });
            }
        };

        if !syntax_validation.is_valid {
            return Ok(FileTestResult {
                file_path: source_path.to_path_buf(),
                compilation_success: false,
                execution_success: false,
                syntax_valid: false,
                output: String::new(),
                error_message: Some("Syntax validation failed".to_string()),
                duration: start_time.elapsed(),
            });
        }

        // Compile the file
        let executable = match self.build_manager.compile_aether_source(source_path) {
            Ok(exe) => exe,
            Err(e) => {
                return Ok(FileTestResult {
                    file_path: source_path.to_path_buf(),
                    compilation_success: false,
                    execution_success: false,
                    syntax_valid: true,
                    output: String::new(),
                    error_message: Some(format!("Compilation failed: {}", e)),
                    duration: start_time.elapsed(),
                });
            }
        };

        // Run the executable
        let execution_result = self.build_manager.run_verification_tests(&executable.path);
        
        match execution_result {
            Ok(test_results) => {
                Ok(FileTestResult {
                    file_path: source_path.to_path_buf(),
                    compilation_success: true,
                    execution_success: test_results.overall_success,
                    syntax_valid: true,
                    output: test_results.basic_test.stdout,
                    error_message: if test_results.overall_success { 
                        None 
                    } else { 
                        Some(test_results.basic_test.stderr) 
                    },
                    duration: start_time.elapsed(),
                })
            }
            Err(e) => {
                Ok(FileTestResult {
                    file_path: source_path.to_path_buf(),
                    compilation_success: true,
                    execution_success: false,
                    syntax_valid: true,
                    output: String::new(),
                    error_message: Some(format!("Execution failed: {}", e)),
                    duration: start_time.elapsed(),
                })
            }
        }
    }

    /// Get build system status
    pub fn get_build_status(&self) -> BuildStatus {
        let env_status = self.build_manager.validate_environment();
        let toolchain_info = self.build_manager.get_toolchain_info();
        
        BuildStatus {
            environment_valid: env_status.is_ok(),
            rust_toolchain_version: toolchain_info.version,
            rust_features: vec![], // ToolchainInfo doesn't have features field
            aether_compiler_available: self.build_manager.get_aether_config().is_some(),
            error_message: env_status.err().map(|e| e.to_string()),
        }
    }

    /// Update test configuration
    pub fn update_test_config(&mut self, config: TestConfig) {
        self.test_config = config;
        // Reinitialize if already initialized
        if self.comprehensive_runner.is_some() {
            let _ = self.initialize();
        }
    }

    /// Update build configuration
    pub fn update_build_config(&mut self, config: BuildConfig) {
        self.build_manager.update_config(config);
        // Reinitialize if already initialized
        if self.comprehensive_runner.is_some() {
            let _ = self.initialize();
        }
    }

    /// Get current test configuration
    pub fn get_test_config(&self) -> &TestConfig {
        &self.test_config
    }

    /// Get current build configuration
    pub fn get_build_config(&self) -> &BuildConfig {
        self.build_manager.config()
    }

    /// Ensure the runner is initialized
    fn ensure_initialized(&mut self) -> Result<(), String> {
        if self.comprehensive_runner.is_none() {
            self.initialize()?;
        }
        Ok(())
    }
}

impl Default for BuildIntegratedTestRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of hello world test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloWorldTestResult {
    pub success: bool,
    pub output: String,
    pub error_message: Option<String>,
    pub duration: std::time::Duration,
}

/// Summary of basic tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicTestSummary {
    pub hello_world_passed: bool,
    pub arithmetic_passed: bool,
    pub function_passed: bool,
    pub control_flow_passed: bool,
    pub success_rate: f64,
    pub duration: std::time::Duration,
    pub details: BasicTestResults,
}

/// Summary of advanced tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedTestSummary {
    pub tensor_operations_passed: bool,
    pub autodiff_passed: bool,
    pub probabilistic_passed: bool,
    pub gpu_kernels_passed: bool,
    pub ffi_integration_passed: bool,
    pub success_rate: f64,
    pub duration: std::time::Duration,
    pub details: AdvancedTestResults,
}

/// Summary of performance benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmarkSummary {
    pub compilation_speed: Option<BenchmarkResult>,
    pub execution_speed: Option<BenchmarkResult>,
    pub memory_usage: Option<BenchmarkResult>,
    pub tensor_performance: Option<BenchmarkResult>,
    pub duration: std::time::Duration,
    pub details: BenchmarkResults,
}

/// Summary of meta-tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaTestSummary {
    pub framework_reliability_passed: bool,
    pub error_detection_passed: bool,
    pub build_integration_passed: bool,
    pub success_rate: f64,
    pub duration: std::time::Duration,
    pub details: MetaTestResults,
}

/// Result of testing a specific file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileTestResult {
    pub file_path: std::path::PathBuf,
    pub compilation_success: bool,
    pub execution_success: bool,
    pub syntax_valid: bool,
    pub output: String,
    pub error_message: Option<String>,
    pub duration: std::time::Duration,
}

/// Build system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildStatus {
    pub environment_valid: bool,
    pub rust_toolchain_version: String,
    pub rust_features: Vec<String>,
    pub aether_compiler_available: bool,
    pub error_message: Option<String>,
}

/// Convenience functions for running tests
impl BuildIntegratedTestRunner {
    /// Quick test to verify the toolchain is working
    pub fn quick_verification_test(&mut self) -> Result<bool, String> {
        let hello_world_result = self.run_hello_world_test()?;
        Ok(hello_world_result.success)
    }

    /// Run regression tests to ensure no functionality has broken
    pub fn run_regression_tests(&mut self) -> Result<RegressionTestResults, String> {
        let start_time = Instant::now();
        
        let basic_summary = self.run_basic_tests()?;
        let advanced_summary = self.run_advanced_tests()?;
        
        let overall_success = basic_summary.success_rate > 0.8 && advanced_summary.success_rate > 0.6;
        
        Ok(RegressionTestResults {
            overall_success,
            basic_tests_success_rate: basic_summary.success_rate,
            advanced_tests_success_rate: advanced_summary.success_rate,
            duration: start_time.elapsed(),
            basic_details: basic_summary,
            advanced_details: advanced_summary,
        })
    }

    /// Generate comprehensive test report
    pub fn generate_test_report(&mut self) -> Result<TestReport, String> {
        let start_time = Instant::now();
        
        let build_status = self.get_build_status();
        let comprehensive_results = self.run_all_tests()?;
        
        let summary = format!(
            "Test Report: {} total tests, {:.1}% success rate, completed in {:?}",
            comprehensive_results.total_tests,
            comprehensive_results.overall_success_rate * 100.0,
            comprehensive_results.total_duration
        );
        
        Ok(TestReport {
            timestamp: chrono::Utc::now(),
            build_status,
            comprehensive_results,
            total_duration: start_time.elapsed(),
            summary,
        })
    }
}

/// Regression test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestResults {
    pub overall_success: bool,
    pub basic_tests_success_rate: f64,
    pub advanced_tests_success_rate: f64,
    pub duration: std::time::Duration,
    pub basic_details: BasicTestSummary,
    pub advanced_details: AdvancedTestSummary,
}

/// Comprehensive test report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub build_status: BuildStatus,
    pub comprehensive_results: ComprehensiveTestResults,
    pub total_duration: std::time::Duration,
    pub summary: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_integrated_runner_creation() {
        let runner = BuildIntegratedTestRunner::new();
        assert!(runner.comprehensive_runner.is_none());
    }

    #[test]
    fn test_build_status_creation() {
        let status = BuildStatus {
            environment_valid: true,
            rust_toolchain_version: "1.70.0".to_string(),
            rust_features: vec!["mlir".to_string()],
            aether_compiler_available: false,
            error_message: None,
        };
        
        assert!(status.environment_valid);
        assert_eq!(status.rust_toolchain_version, "1.70.0");
        assert!(status.rust_features.contains(&"mlir".to_string()));
    }

    #[test]
    fn test_hello_world_test_result() {
        let result = HelloWorldTestResult {
            success: true,
            output: "Hello, World from Aether!".to_string(),
            error_message: None,
            duration: std::time::Duration::from_millis(100),
        };
        
        assert!(result.success);
        assert!(result.output.contains("Hello, World"));
        assert!(result.error_message.is_none());
    }

    #[test]
    fn test_file_test_result() {
        let result = FileTestResult {
            file_path: std::path::PathBuf::from("test.ae"),
            compilation_success: true,
            execution_success: true,
            syntax_valid: true,
            output: "Test output".to_string(),
            error_message: None,
            duration: std::time::Duration::from_millis(200),
        };
        
        assert!(result.compilation_success);
        assert!(result.execution_success);
        assert!(result.syntax_valid);
        assert_eq!(result.file_path, std::path::PathBuf::from("test.ae"));
    }
}