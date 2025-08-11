// Aether Comprehensive Testing and Validation Framework
// Core testing infrastructure for unit tests, integration tests, and benchmarks

pub mod unit_testing;
pub mod property_testing;
pub mod integration_testing;
pub mod benchmark_testing;
pub mod correctness_validation;
pub mod meta_testing;
pub mod test_runner;
pub mod test_reporting;
pub mod comprehensive_test_runner;
pub mod build_integrated_test_runner;
pub mod mlir_test_runner;
pub mod file_compilation_testing;
pub mod cargo_integration;
pub mod test_cache;

use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[cfg(test)]
use tempfile;

// Re-export main testing interfaces
pub use comprehensive_test_runner::{
    ComprehensiveTestRunner, ComprehensiveTestResults, BasicTestResults, 
    AdvancedTestResults, BenchmarkResults, BenchmarkResult, MetaTestResults
};
pub use build_integrated_test_runner::{
    BuildIntegratedTestRunner, HelloWorldTestResult, BasicTestSummary,
    AdvancedTestSummary, PerformanceBenchmarkSummary, MetaTestSummary,
    FileTestResult, BuildStatus, RegressionTestResults, TestReport
};
pub use mlir_test_runner::{
    MLIRTestRunner, MLIRBasicTestSuite, MLIRCompilationTest, MLIRNativeCompilationTest,
    MLIRWasmCompilationTest, MLIRPerformanceTest, MLIRErrorHandlingTest
};
pub use file_compilation_testing::{
    FileCompilationTestOrchestrator, TestingConfig, TestingError, TestingResult,
    TestCategory, ReportFormat, FileCompilationTestReport, TestSummary, CompilationResult, ExecutionResult
};
pub use cargo_integration::{
    CargoIntegration, CargoIntegrationConfig, CargoIntegrationError,
    run_cargo_test_integration, run_build_integration
};
pub use test_cache::{TestCache, CacheConfig, CacheError, CacheStats};

/// Test result status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
    Timeout,
    Error,
}

/// Test execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,
    pub status: TestStatus,
    pub duration: Duration,
    pub message: Option<String>,
    pub error_details: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl TestResult {
    pub fn passed(name: String, duration: Duration) -> Self {
        TestResult {
            name,
            status: TestStatus::Passed,
            duration,
            message: None,
            error_details: None,
            metadata: HashMap::new(),
        }
    }

    pub fn failed(name: String, duration: Duration, message: String) -> Self {
        TestResult {
            name,
            status: TestStatus::Failed,
            duration,
            message: Some(message),
            error_details: None,
            metadata: HashMap::new(),
        }
    }

    pub fn error(name: String, duration: Duration, error: String) -> Self {
        TestResult {
            name,
            status: TestStatus::Error,
            duration,
            message: None,
            error_details: Some(error),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Test suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    pub timeout: Duration,
    pub parallel_execution: bool,
    pub max_threads: usize,
    pub verbose: bool,
    pub fail_fast: bool,
    pub filter: Option<String>,
    pub benchmark_iterations: usize,
    pub property_test_cases: usize,
}

impl Default for TestConfig {
    fn default() -> Self {
        TestConfig {
            timeout: Duration::from_secs(30),
            parallel_execution: true,
            max_threads: num_cpus::get(),
            verbose: false,
            fail_fast: false,
            filter: None,
            benchmark_iterations: 100,
            property_test_cases: 1000,
        }
    }
}

/// Test execution context
pub struct TestContext {
    pub config: TestConfig,
    pub start_time: Instant,
    pub results: Vec<TestResult>,
    pub current_suite: Option<String>,
}

impl TestContext {
    pub fn new(config: TestConfig) -> Self {
        TestContext {
            config,
            start_time: Instant::now(),
            results: Vec::new(),
            current_suite: None,
        }
    }

    pub fn add_result(&mut self, result: TestResult) {
        self.results.push(result);
    }

    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.status == TestStatus::Passed).count()
    }

    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| r.status == TestStatus::Failed).count()
    }

    pub fn error_count(&self) -> usize {
        self.results.iter().filter(|r| r.status == TestStatus::Error).count()
    }

    pub fn total_duration(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Trait for test cases
pub trait TestCase: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, context: &mut TestContext) -> TestResult;
    fn setup(&self) -> Result<(), String> { Ok(()) }
    fn teardown(&self) -> Result<(), String> { Ok(()) }
    fn timeout(&self) -> Option<Duration> { None }
}

/// Trait for test suites
pub trait TestSuite {
    fn name(&self) -> &str;
    fn tests(&self) -> Vec<Box<dyn TestCase>>;
    fn setup_suite(&self) -> Result<(), String> { Ok(()) }
    fn teardown_suite(&self) -> Result<(), String> { Ok(()) }
}

/// Test assertion macros
#[macro_export]
macro_rules! assert_test {
    ($condition:expr, $message:expr) => {
        if !$condition {
            return TestResult::failed(
                "assertion_failed".to_string(),
                Duration::from_nanos(0),
                format!("Assertion failed: {}", $message)
            );
        }
    };
}

#[macro_export]
macro_rules! assert_eq_test {
    ($left:expr, $right:expr) => {
        if $left != $right {
            return TestResult::failed(
                "equality_assertion_failed".to_string(),
                Duration::from_nanos(0),
                format!("Assertion failed: {:?} != {:?}", $left, $right)
            );
        }
    };
}

#[macro_export]
macro_rules! assert_approx_eq {
    ($left:expr, $right:expr, $epsilon:expr) => {
        if ($left - $right).abs() > $epsilon {
            return TestResult::failed(
                "approximate_equality_failed".to_string(),
                Duration::from_nanos(0),
                format!("Assertion failed: |{:?} - {:?}| > {:?}", $left, $right, $epsilon)
            );
        }
    };
}

/// Test utilities
pub struct TestUtils;

impl TestUtils {
    /// Generate random test data
    pub fn random_f64(min: f64, max: f64) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(min..max)
    }

    /// Generate random integer
    pub fn random_i32(min: i32, max: i32) -> i32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(min..max)
    }

    /// Generate random string
    pub fn random_string(length: usize) -> String {
        use rand::{Rng, distributions::Alphanumeric};
        rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(length)
            .map(char::from)
            .collect()
    }

    /// Create temporary directory for tests
    #[cfg(test)]
    pub fn temp_dir() -> Result<tempfile::TempDir, std::io::Error> {
        tempfile::tempdir()
    }

    /// Measure execution time
    pub fn measure_time<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_creation() {
        let result = TestResult::passed("test".to_string(), Duration::from_millis(100));
        assert_eq!(result.status, TestStatus::Passed);
        assert_eq!(result.name, "test");
        assert_eq!(result.duration, Duration::from_millis(100));
    }

    #[test]
    fn test_context_tracking() {
        let config = TestConfig::default();
        let mut context = TestContext::new(config);
        
        context.add_result(TestResult::passed("test1".to_string(), Duration::from_millis(50)));
        context.add_result(TestResult::failed("test2".to_string(), Duration::from_millis(75), "error".to_string()));
        
        assert_eq!(context.passed_count(), 1);
        assert_eq!(context.failed_count(), 1);
        assert_eq!(context.error_count(), 0);
    }

    #[test]
    fn test_utils_random_generation() {
        let f = TestUtils::random_f64(0.0, 1.0);
        assert!(f >= 0.0 && f <= 1.0);
        
        let i = TestUtils::random_i32(1, 10);
        assert!(i >= 1 && i <= 10);
        
        let s = TestUtils::random_string(10);
        assert_eq!(s.len(), 10);
    }

    #[test]
    fn test_time_measurement() {
        let (result, duration) = TestUtils::measure_time(|| {
            std::thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(10));
    }
}