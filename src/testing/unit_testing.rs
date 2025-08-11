// Unit Testing Framework
// Provides structured unit testing capabilities for Aether language components

use super::{TestCase, TestResult, TestContext};
use std::time::{Duration, Instant};

/// Unit test case
pub struct UnitTest {
    pub name: String,
    pub test_fn: Box<dyn Fn() -> Result<(), String> + Send + Sync>,
    pub timeout: Option<Duration>,
    pub tags: Vec<String>,
    pub setup: Option<Box<dyn Fn() -> Result<(), String> + Send + Sync>>,
    pub teardown: Option<Box<dyn Fn() -> Result<(), String> + Send + Sync>>,
}

impl UnitTest {
    pub fn new<F>(name: &str, test_fn: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        UnitTest {
            name: name.to_string(),
            test_fn: Box::new(test_fn),
            timeout: None,
            tags: Vec::new(),
            setup: None,
            teardown: None,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_setup<F>(mut self, setup: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        self.setup = Some(Box::new(setup));
        self
    }

    pub fn with_teardown<F>(mut self, teardown: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        self.teardown = Some(Box::new(teardown));
        self
    }
}

impl TestCase for UnitTest {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _context: &mut TestContext) -> TestResult {
        let start = Instant::now();

        // Setup
        if let Some(setup) = &self.setup {
            if let Err(e) = setup() {
                return TestResult::error(
                    self.name.clone(),
                    start.elapsed(),
                    format!("Setup failed: {}", e)
                );
            }
        }

        // Run test with timeout handling
        let result = if let Some(timeout) = self.timeout {
            run_with_timeout(&self.test_fn, timeout)
        } else {
            (self.test_fn)()
        };

        let duration = start.elapsed();

        // Teardown
        if let Some(teardown) = &self.teardown {
            if let Err(e) = teardown() {
                // Log teardown error but don't fail the test
                eprintln!("Warning: Teardown failed for {}: {}", self.name, e);
            }
        }

        match result {
            Ok(()) => TestResult::passed(self.name.clone(), duration),
            Err(e) => TestResult::failed(self.name.clone(), duration, e),
        }
    }

    fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
}

/// Unit test suite
pub struct UnitTestSuite {
    pub name: String,
    pub tests: Vec<UnitTest>,
    pub setup_suite: Option<Box<dyn Fn() -> Result<(), String> + Send + Sync>>,
    pub teardown_suite: Option<Box<dyn Fn() -> Result<(), String> + Send + Sync>>,
}

impl UnitTestSuite {
    pub fn new(name: &str) -> Self {
        UnitTestSuite {
            name: name.to_string(),
            tests: Vec::new(),
            setup_suite: None,
            teardown_suite: None,
        }
    }

    pub fn add_test(mut self, test: UnitTest) -> Self {
        self.tests.push(test);
        self
    }

    pub fn with_suite_setup<F>(mut self, setup: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        self.setup_suite = Some(Box::new(setup));
        self
    }

    pub fn with_suite_teardown<F>(mut self, teardown: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        self.teardown_suite = Some(Box::new(teardown));
        self
    }
}

/// Unit test builder for fluent API
pub struct UnitTestBuilder {
    name: String,
    timeout: Option<Duration>,
    tags: Vec<String>,
    setup: Option<Box<dyn Fn() -> Result<(), String> + Send + Sync>>,
    teardown: Option<Box<dyn Fn() -> Result<(), String> + Send + Sync>>,
}

impl UnitTestBuilder {
    pub fn new(name: &str) -> Self {
        UnitTestBuilder {
            name: name.to_string(),
            timeout: None,
            tags: Vec::new(),
            setup: None,
            teardown: None,
        }
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn setup<F>(mut self, setup: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        self.setup = Some(Box::new(setup));
        self
    }

    pub fn teardown<F>(mut self, teardown: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        self.teardown = Some(Box::new(teardown));
        self
    }

    pub fn test<F>(self, test_fn: F) -> UnitTest
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        let mut test = UnitTest::new(&self.name, test_fn);
        if let Some(timeout) = self.timeout {
            test = test.with_timeout(timeout);
        }
        if !self.tags.is_empty() {
            test = test.with_tags(self.tags);
        }
        if let Some(setup) = self.setup {
            test.setup = Some(setup);
        }
        if let Some(teardown) = self.teardown {
            test.teardown = Some(teardown);
        }
        test
    }
}

/// Run function with timeout (simplified implementation)
fn run_with_timeout<F>(f: F, timeout: Duration) -> Result<(), String>
where
    F: Fn() -> Result<(), String>,
{
    // For now, just run the function directly
    // In a full implementation, this would use threads or async to enforce timeout
    let start = Instant::now();
    let result = f();
    if start.elapsed() > timeout {
        Err(format!("Test timed out after {:?}", timeout))
    } else {
        result
    }
}

/// Assertion helpers for unit tests
pub struct UnitAssert;

impl UnitAssert {
    pub fn equals<T: PartialEq + std::fmt::Debug>(actual: T, expected: T) -> Result<(), String> {
        if actual == expected {
            Ok(())
        } else {
            Err(format!("Expected {:?}, got {:?}", expected, actual))
        }
    }

    pub fn not_equals<T: PartialEq + std::fmt::Debug>(actual: T, expected: T) -> Result<(), String> {
        if actual != expected {
            Ok(())
        } else {
            Err(format!("Expected values to be different, but both were {:?}", actual))
        }
    }

    pub fn is_true(condition: bool) -> Result<(), String> {
        if condition {
            Ok(())
        } else {
            Err("Expected condition to be true".to_string())
        }
    }

    pub fn is_false(condition: bool) -> Result<(), String> {
        if !condition {
            Ok(())
        } else {
            Err("Expected condition to be false".to_string())
        }
    }

    pub fn is_some<T>(option: Option<T>) -> Result<(), String> {
        if option.is_some() {
            Ok(())
        } else {
            Err("Expected Some, got None".to_string())
        }
    }

    pub fn is_none<T>(option: Option<T>) -> Result<(), String> {
        if option.is_none() {
            Ok(())
        } else {
            Err("Expected None, got Some".to_string())
        }
    }

    pub fn is_ok<T, E: std::fmt::Debug>(result: Result<T, E>) -> Result<(), String> {
        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Expected Ok, got Err({:?})", e)),
        }
    }

    pub fn is_err<T: std::fmt::Debug, E>(result: Result<T, E>) -> Result<(), String> {
        match result {
            Ok(v) => Err(format!("Expected Err, got Ok({:?})", v)),
            Err(_) => Ok(()),
        }
    }

    pub fn approx_equals(actual: f64, expected: f64, epsilon: f64) -> Result<(), String> {
        if (actual - expected).abs() <= epsilon {
            Ok(())
        } else {
            Err(format!("Expected {:.6} ± {:.6}, got {:.6}", expected, epsilon, actual))
        }
    }

    pub fn contains<T: PartialEq + std::fmt::Debug>(collection: &[T], item: &T) -> Result<(), String> {
        if collection.contains(item) {
            Ok(())
        } else {
            Err(format!("Collection {:?} does not contain {:?}", collection, item))
        }
    }

    pub fn not_contains<T: PartialEq + std::fmt::Debug>(collection: &[T], item: &T) -> Result<(), String> {
        if !collection.contains(item) {
            Ok(())
        } else {
            Err(format!("Collection {:?} should not contain {:?}", collection, item))
        }
    }

    pub fn length_equals<T>(collection: &[T], expected_length: usize) -> Result<(), String> {
        if collection.len() == expected_length {
            Ok(())
        } else {
            Err(format!("Expected length {}, got {}", expected_length, collection.len()))
        }
    }

    pub fn is_empty<T>(collection: &[T]) -> Result<(), String> {
        if collection.is_empty() {
            Ok(())
        } else {
            Err(format!("Expected empty collection, got {} items", collection.len()))
        }
    }

    pub fn not_empty<T>(collection: &[T]) -> Result<(), String> {
        if !collection.is_empty() {
            Ok(())
        } else {
            Err("Expected non-empty collection".to_string())
        }
    }
}

/// Macro for creating unit tests
#[macro_export]
macro_rules! unit_test {
    ($name:expr, $body:expr) => {
        UnitTest::new($name, || $body)
    };
    ($name:expr, timeout = $timeout:expr, $body:expr) => {
        UnitTest::new($name, || $body).with_timeout($timeout)
    };
    ($name:expr, tags = $tags:expr, $body:expr) => {
        UnitTest::new($name, || $body).with_tags($tags)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_test_creation() {
        let test = UnitTest::new("sample_test", || Ok(()));
        assert_eq!(test.name, "sample_test");
        assert!(test.timeout.is_none());
        assert!(test.tags.is_empty());
    }

    #[test]
    fn test_unit_test_with_timeout() {
        let test = UnitTest::new("timeout_test", || Ok(()))
            .with_timeout(Duration::from_secs(5));
        assert_eq!(test.timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn test_unit_test_with_tags() {
        let tags = vec!["integration".to_string(), "slow".to_string()];
        let test = UnitTest::new("tagged_test", || Ok(()))
            .with_tags(tags.clone());
        assert_eq!(test.tags, tags);
    }

    #[test]
    fn test_unit_test_builder() {
        let test = UnitTestBuilder::new("builder_test")
            .timeout(Duration::from_secs(10))
            .tags(vec!["unit".to_string()])
            .test(|| Ok(()));
        
        assert_eq!(test.name, "builder_test");
        assert_eq!(test.timeout, Some(Duration::from_secs(10)));
        assert_eq!(test.tags, vec!["unit".to_string()]);
    }

    #[test]
    fn test_unit_assert_equals() {
        assert!(UnitAssert::equals(42, 42).is_ok());
        assert!(UnitAssert::equals(42, 43).is_err());
    }

    #[test]
    fn test_unit_assert_approx_equals() {
        assert!(UnitAssert::approx_equals(3.14159, 3.14160, 0.001).is_ok());
        assert!(UnitAssert::approx_equals(3.14159, 3.15000, 0.001).is_err());
    }

    #[test]
    fn test_unit_assert_contains() {
        let vec = vec![1, 2, 3, 4, 5];
        assert!(UnitAssert::contains(&vec, &3).is_ok());
        assert!(UnitAssert::contains(&vec, &6).is_err());
        assert!(UnitAssert::not_contains(&vec, &6).is_ok());
        assert!(UnitAssert::not_contains(&vec, &3).is_err());
    }

    #[test]
    fn test_unit_assert_length() {
        let vec = vec![1, 2, 3];
        assert!(UnitAssert::length_equals(&vec, 3).is_ok());
        assert!(UnitAssert::length_equals(&vec, 4).is_err());
        assert!(UnitAssert::not_empty(&vec).is_ok());
        
        let empty_vec: Vec<i32> = vec![];
        assert!(UnitAssert::is_empty(&empty_vec).is_ok());
        assert!(UnitAssert::not_empty(&empty_vec).is_err());
    }

    #[test]
    fn test_unit_assert_option() {
        assert!(UnitAssert::is_some(Some(42)).is_ok());
        assert!(UnitAssert::is_some(None::<i32>).is_err());
        assert!(UnitAssert::is_none(None::<i32>).is_ok());
        assert!(UnitAssert::is_none(Some(42)).is_err());
    }

    #[test]
    fn test_unit_assert_result() {
        let ok_result: Result<i32, String> = Ok(42);
        let err_result: Result<i32, String> = Err("error".to_string());
        
        assert!(UnitAssert::is_ok(ok_result).is_ok());
        assert!(UnitAssert::is_err(err_result).is_ok());
    }
}