// Test Runner
// Orchestrates test execution with parallel processing and filtering

use super::{TestCase, TestResult, TestContext, TestStatus, TestConfig};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

/// Test execution strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStrategy {
    Sequential,
    Parallel,
    Adaptive,
}

/// Test filter criteria
#[derive(Debug, Clone)]
pub struct TestFilter {
    pub name_pattern: Option<String>,
    pub tags: Vec<String>,
    pub exclude_tags: Vec<String>,
    pub test_types: Vec<String>,
    pub only_failed: bool,
}

impl Default for TestFilter {
    fn default() -> Self {
        TestFilter {
            name_pattern: None,
            tags: Vec::new(),
            exclude_tags: Vec::new(),
            test_types: Vec::new(),
            only_failed: false,
        }
    }
}

impl TestFilter {
    pub fn matches(&self, test_name: &str, test_tags: &[String], test_type: &str) -> bool {
        // Check name pattern
        if let Some(pattern) = &self.name_pattern {
            if !test_name.contains(pattern) {
                return false;
            }
        }

        // Check required tags
        if !self.tags.is_empty() {
            if !self.tags.iter().any(|tag| test_tags.contains(tag)) {
                return false;
            }
        }

        // Check excluded tags
        if !self.exclude_tags.is_empty() {
            if self.exclude_tags.iter().any(|tag| test_tags.contains(tag)) {
                return false;
            }
        }

        // Check test types
        if !self.test_types.is_empty() {
            if !self.test_types.contains(&test_type.to_string()) {
                return false;
            }
        }

        true
    }
}

/// Test runner for executing test suites
pub struct TestRunner {
    config: TestConfig,
    execution_strategy: ExecutionStrategy,
    filter: TestFilter,
    build_manager: Option<crate::build_system::BuildSystemManager>,
}

impl TestRunner {
    pub fn new(config: TestConfig) -> Self {
        TestRunner {
            config,
            execution_strategy: ExecutionStrategy::Parallel,
            filter: TestFilter::default(),
            build_manager: None,
        }
    }

    pub fn with_build_manager(mut self, build_manager: crate::build_system::BuildSystemManager) -> Self {
        self.build_manager = Some(build_manager);
        self
    }

    pub fn with_execution_strategy(mut self, strategy: ExecutionStrategy) -> Self {
        self.execution_strategy = strategy;
        self
    }

    pub fn with_filter(mut self, filter: TestFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Run a collection of test cases
    pub fn run_tests(&self, tests: Vec<Box<dyn TestCase>>) -> TestRunResult {
        let start_time = Instant::now();
        let mut context = TestContext::new(self.config.clone());

        // Filter tests
        let filtered_tests = self.filter_tests(tests);
        
        if filtered_tests.is_empty() {
            return TestRunResult {
                total_tests: 0,
                passed: 0,
                failed: 0,
                errors: 0,
                skipped: 0,
                duration: start_time.elapsed(),
                results: Vec::new(),
            };
        }

        // Execute tests based on strategy
        let results = match self.execution_strategy {
            ExecutionStrategy::Sequential => self.run_sequential(filtered_tests, &mut context),
            ExecutionStrategy::Parallel => self.run_parallel(filtered_tests, &mut context),
            ExecutionStrategy::Adaptive => self.run_adaptive(filtered_tests, &mut context),
        };

        let duration = start_time.elapsed();

        // Calculate statistics
        let total_tests = results.len();
        let passed = results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let failed = results.iter().filter(|r| r.status == TestStatus::Failed).count();
        let errors = results.iter().filter(|r| r.status == TestStatus::Error).count();
        let skipped = results.iter().filter(|r| r.status == TestStatus::Skipped).count();

        TestRunResult {
            total_tests,
            passed,
            failed,
            errors,
            skipped,
            duration,
            results,
        }
    }

    fn filter_tests(&self, tests: Vec<Box<dyn TestCase>>) -> Vec<Box<dyn TestCase>> {
        tests.into_iter()
            .filter(|test| {
                // For now, we'll use simplified filtering
                // In a full implementation, tests would have metadata for tags and types
                self.filter.matches(test.name(), &[], "unit")
            })
            .collect()
    }

    fn run_sequential(&self, tests: Vec<Box<dyn TestCase>>, context: &mut TestContext) -> Vec<TestResult> {
        let mut results = Vec::new();

        for test in tests {
            if self.config.verbose {
                println!("Running test: {}", test.name());
            }

            let result = self.run_single_test(test, context);
            
            if self.config.verbose {
                println!("  Result: {:?} ({:?})", result.status, result.duration);
            }

            // Check fail-fast
            if self.config.fail_fast && result.status == TestStatus::Failed {
                results.push(result);
                break;
            }

            results.push(result);
        }

        results
    }

    fn run_parallel(&self, tests: Vec<Box<dyn TestCase>>, _context: &mut TestContext) -> Vec<TestResult> {
        let max_threads = self.config.max_threads.min(tests.len());
        let results = Arc::new(Mutex::new(Vec::new()));
        let test_queue = Arc::new(Mutex::new(tests));
        let mut handles = Vec::new();

        for _ in 0..max_threads {
            let queue = Arc::clone(&test_queue);
            let results = Arc::clone(&results);
            let config = self.config.clone();

            let handle = thread::spawn(move || {
                let mut local_context = TestContext::new(config.clone());
                
                loop {
                    let test = {
                        let mut queue = queue.lock().unwrap();
                        if queue.is_empty() {
                            break;
                        }
                        queue.pop()
                    };

                    if let Some(test) = test {
                        if config.verbose {
                            println!("Running test: {} (thread: {:?})", test.name(), thread::current().id());
                        }

                        let result = Self::run_single_test_static(test, &mut local_context);
                        
                        {
                            let mut results = results.lock().unwrap();
                            results.push(result);
                        }
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        let results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
        results
    }

    fn run_adaptive(&self, tests: Vec<Box<dyn TestCase>>, context: &mut TestContext) -> Vec<TestResult> {
        // Adaptive strategy: run fast tests in parallel, slow tests sequentially
        let (fast_tests, slow_tests): (Vec<_>, Vec<_>) = tests.into_iter()
            .partition(|test| {
                // Heuristic: tests with timeout > 5 seconds are considered slow
                test.timeout().map_or(false, |t| t > Duration::from_secs(5))
            });

        let mut results = Vec::new();

        // Run fast tests in parallel
        if !fast_tests.is_empty() {
            let mut fast_results = self.run_parallel(fast_tests, context);
            results.append(&mut fast_results);
        }

        // Run slow tests sequentially
        if !slow_tests.is_empty() {
            let mut slow_results = self.run_sequential(slow_tests, context);
            results.append(&mut slow_results);
        }

        results
    }

    fn run_single_test(&self, test: Box<dyn TestCase>, context: &mut TestContext) -> TestResult {
        Self::run_single_test_static(test, context)
    }

    fn run_single_test_static(test: Box<dyn TestCase>, context: &mut TestContext) -> TestResult {
        let start = Instant::now();
        
        // Check timeout
        let timeout = test.timeout().unwrap_or(Duration::from_secs(30));
        
        // Run the test (simplified timeout handling)
        let result = test.run(context);
        
        // Check if test exceeded timeout
        if start.elapsed() > timeout {
            TestResult {
                name: test.name().to_string(),
                status: TestStatus::Timeout,
                duration: start.elapsed(),
                message: Some(format!("Test timed out after {:?}", timeout)),
                error_details: None,
                metadata: HashMap::new(),
            }
        } else {
            result
        }
    }

    /// Run tests with progress reporting
    pub fn run_with_progress<F>(&self, tests: Vec<Box<dyn TestCase>>, progress_callback: F) -> TestRunResult
    where
        F: Fn(usize, usize, &str) + Send + Sync + 'static,
    {
        let total = tests.len();
        let progress_callback = Arc::new(progress_callback);
        let completed = Arc::new(Mutex::new(0));

        // This is a simplified implementation
        // In a full implementation, we'd integrate progress reporting with parallel execution
        let start_time = Instant::now();
        let mut context = TestContext::new(self.config.clone());
        let mut results = Vec::new();

        for test in tests {
            let test_name = test.name().to_string();
            
            // Report progress
            {
                let completed_count = *completed.lock().unwrap();
                progress_callback(completed_count, total, &test_name);
            }

            let result = self.run_single_test(test, &mut context);
            results.push(result);

            // Update progress
            {
                let mut completed_count = completed.lock().unwrap();
                *completed_count += 1;
                progress_callback(*completed_count, total, "");
            }
        }

        let duration = start_time.elapsed();
        let passed = results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let failed = results.iter().filter(|r| r.status == TestStatus::Failed).count();
        let errors = results.iter().filter(|r| r.status == TestStatus::Error).count();
        let skipped = results.iter().filter(|r| r.status == TestStatus::Skipped).count();

        TestRunResult {
            total_tests: total,
            passed,
            failed,
            errors,
            skipped,
            duration,
            results,
        }
    }
}

/// Result of a test run
#[derive(Debug, Clone)]
pub struct TestRunResult {
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub errors: usize,
    pub skipped: usize,
    pub duration: Duration,
    pub results: Vec<TestResult>,
}

impl TestRunResult {
    pub fn success_rate(&self) -> f64 {
        if self.total_tests == 0 {
            1.0
        } else {
            self.passed as f64 / self.total_tests as f64
        }
    }

    pub fn is_successful(&self) -> bool {
        self.failed == 0 && self.errors == 0
    }

    pub fn summary(&self) -> String {
        format!(
            "Tests: {} passed, {} failed, {} errors, {} skipped ({} total) in {:?}",
            self.passed, self.failed, self.errors, self.skipped, self.total_tests, self.duration
        )
    }

    pub fn failed_tests(&self) -> Vec<&TestResult> {
        self.results.iter()
            .filter(|r| r.status == TestStatus::Failed || r.status == TestStatus::Error)
            .collect()
    }
}

/// Test discovery utilities
pub struct TestDiscovery;

impl TestDiscovery {
    /// Discover tests from a module (simplified implementation)
    pub fn discover_tests(_module_path: &str) -> Vec<Box<dyn TestCase>> {
        // In a full implementation, this would use reflection or macros
        // to automatically discover test functions
        Vec::new()
    }

    /// Group tests by category
    pub fn group_tests_by_category(tests: Vec<Box<dyn TestCase>>) -> HashMap<String, Vec<Box<dyn TestCase>>> {
        let mut groups = HashMap::new();
        
        for test in tests {
            let category = Self::categorize_test(test.name());
            groups.entry(category).or_insert_with(Vec::new).push(test);
        }
        
        groups
    }

    fn categorize_test(test_name: &str) -> String {
        if test_name.contains("unit") || test_name.contains("test_") {
            "unit".to_string()
        } else if test_name.contains("integration") {
            "integration".to_string()
        } else if test_name.contains("benchmark") {
            "benchmark".to_string()
        } else if test_name.contains("property") {
            "property".to_string()
        } else {
            "other".to_string()
        }
    }
}

/// Test execution statistics
#[derive(Debug, Clone)]
pub struct TestStatistics {
    pub total_duration: Duration,
    pub average_duration: Duration,
    pub fastest_test: Option<(String, Duration)>,
    pub slowest_test: Option<(String, Duration)>,
    pub success_rate: f64,
    pub tests_per_second: f64,
}

impl TestStatistics {
    pub fn from_results(results: &[TestResult]) -> Self {
        if results.is_empty() {
            return TestStatistics {
                total_duration: Duration::from_nanos(0),
                average_duration: Duration::from_nanos(0),
                fastest_test: None,
                slowest_test: None,
                success_rate: 0.0,
                tests_per_second: 0.0,
            };
        }

        let total_duration: Duration = results.iter().map(|r| r.duration).sum();
        let average_duration = total_duration / results.len() as u32;
        
        let fastest_test = results.iter()
            .min_by_key(|r| r.duration)
            .map(|r| (r.name.clone(), r.duration));
            
        let slowest_test = results.iter()
            .max_by_key(|r| r.duration)
            .map(|r| (r.name.clone(), r.duration));

        let passed_count = results.iter().filter(|r| r.status == TestStatus::Passed).count();
        let success_rate = passed_count as f64 / results.len() as f64;
        
        let tests_per_second = if total_duration.as_secs_f64() > 0.0 {
            results.len() as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        TestStatistics {
            total_duration,
            average_duration,
            fastest_test,
            slowest_test,
            success_rate,
            tests_per_second,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::unit_testing::UnitTest;

    #[test]
    fn test_filter_creation() {
        let filter = TestFilter::default();
        assert!(filter.name_pattern.is_none());
        assert!(filter.tags.is_empty());
        assert!(!filter.only_failed);
    }

    #[test]
    fn test_filter_matching() {
        let filter = TestFilter {
            name_pattern: Some("test".to_string()),
            tags: vec!["unit".to_string()],
            exclude_tags: vec!["slow".to_string()],
            test_types: vec!["unit".to_string()],
            only_failed: false,
        };

        assert!(filter.matches("test_example", &["unit".to_string()], "unit"));
        assert!(!filter.matches("example", &["unit".to_string()], "unit")); // No "test" in name
        assert!(!filter.matches("test_example", &["slow".to_string()], "unit")); // Has excluded tag
        assert!(!filter.matches("test_example", &["unit".to_string()], "integration")); // Wrong type
    }

    #[test]
    fn test_runner_creation() {
        let config = TestConfig::default();
        let runner = TestRunner::new(config)
            .with_execution_strategy(ExecutionStrategy::Sequential)
            .with_filter(TestFilter::default());

        assert_eq!(runner.execution_strategy, ExecutionStrategy::Sequential);
    }

    #[test]
    fn test_sequential_execution() {
        let config = TestConfig::default();
        let runner = TestRunner::new(config)
            .with_execution_strategy(ExecutionStrategy::Sequential);

        let tests: Vec<Box<dyn TestCase>> = vec![
            Box::new(UnitTest::new("test1", || Ok(()))),
            Box::new(UnitTest::new("test2", || Ok(()))),
            Box::new(UnitTest::new("test3", || Err("fail".to_string()))),
        ];

        let result = runner.run_tests(tests);
        
        assert_eq!(result.total_tests, 3);
        assert_eq!(result.passed, 2);
        assert_eq!(result.failed, 1);
        assert_eq!(result.errors, 0);
    }

    #[test]
    fn test_run_result_statistics() {
        let results = vec![
            TestResult::passed("test1".to_string(), Duration::from_millis(10)),
            TestResult::passed("test2".to_string(), Duration::from_millis(20)),
            TestResult::failed("test3".to_string(), Duration::from_millis(15), "error".to_string()),
        ];

        let run_result = TestRunResult {
            total_tests: 3,
            passed: 2,
            failed: 1,
            errors: 0,
            skipped: 0,
            duration: Duration::from_millis(45),
            results: results.clone(),
        };

        assert_eq!(run_result.success_rate(), 2.0 / 3.0);
        assert!(!run_result.is_successful());
        assert_eq!(run_result.failed_tests().len(), 1);

        let stats = TestStatistics::from_results(&results);
        assert_eq!(stats.total_duration, Duration::from_millis(45));
        assert_eq!(stats.success_rate, 2.0 / 3.0);
        assert!(stats.fastest_test.is_some());
        assert!(stats.slowest_test.is_some());
    }

    #[test]
    fn test_test_discovery() {
        let tests = TestDiscovery::discover_tests("test_module");
        assert!(tests.is_empty()); // Simplified implementation returns empty

        let test_cases: Vec<Box<dyn TestCase>> = vec![
            Box::new(UnitTest::new("unit_test_1", || Ok(()))),
            Box::new(UnitTest::new("integration_test_1", || Ok(()))),
            Box::new(UnitTest::new("benchmark_test_1", || Ok(()))),
        ];

        let groups = TestDiscovery::group_tests_by_category(test_cases);
        assert!(groups.contains_key("unit"));
        assert!(groups.contains_key("integration"));
        assert!(groups.contains_key("benchmark"));
    }
}