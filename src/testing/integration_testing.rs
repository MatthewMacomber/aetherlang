// Integration Testing Framework
// End-to-end compilation validation and system integration tests

use super::{TestCase, TestResult, TestContext};
use std::time::{Duration, Instant};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::fs;
use std::env;

/// Integration test configuration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    pub timeout: Duration,
    pub temp_dir: Option<PathBuf>,
    pub preserve_artifacts: bool,
    pub verbose_output: bool,
    pub compiler_flags: Vec<String>,
    pub target_platforms: Vec<CompilationTarget>,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        IntegrationConfig {
            timeout: Duration::from_secs(60),
            temp_dir: None,
            preserve_artifacts: false,
            verbose_output: false,
            compiler_flags: Vec::new(),
            target_platforms: vec![CompilationTarget::Native],
        }
    }
}

/// Compilation targets for integration testing
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationTarget {
    Native,
    WebAssembly,
    GPU,
    Mobile,
}

impl CompilationTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            CompilationTarget::Native => "native",
            CompilationTarget::WebAssembly => "wasm32",
            CompilationTarget::GPU => "gpu",
            CompilationTarget::Mobile => "mobile",
        }
    }
}

/// Integration test case
pub struct IntegrationTest {
    pub name: String,
    pub source_code: String,
    pub expected_output: Option<String>,
    pub expected_exit_code: i32,
    pub config: IntegrationConfig,
    pub setup: Option<Box<dyn Fn(&Path) -> Result<(), String> + Send + Sync>>,
    pub teardown: Option<Box<dyn Fn(&Path) -> Result<(), String> + Send + Sync>>,
    pub validation: Option<Box<dyn Fn(&IntegrationTestResult) -> Result<(), String> + Send + Sync>>,
}

impl IntegrationTest {
    pub fn new(name: &str, source_code: &str) -> Self {
        IntegrationTest {
            name: name.to_string(),
            source_code: source_code.to_string(),
            expected_output: None,
            expected_exit_code: 0,
            config: IntegrationConfig::default(),
            setup: None,
            teardown: None,
            validation: None,
        }
    }

    pub fn expect_output(mut self, output: &str) -> Self {
        self.expected_output = Some(output.to_string());
        self
    }

    pub fn expect_exit_code(mut self, code: i32) -> Self {
        self.expected_exit_code = code;
        self
    }

    pub fn with_config(mut self, config: IntegrationConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_setup<F>(mut self, setup: F) -> Self
    where
        F: Fn(&Path) -> Result<(), String> + Send + Sync + 'static,
    {
        self.setup = Some(Box::new(setup));
        self
    }

    pub fn with_teardown<F>(mut self, teardown: F) -> Self
    where
        F: Fn(&Path) -> Result<(), String> + Send + Sync + 'static,
    {
        self.teardown = Some(Box::new(teardown));
        self
    }

    pub fn with_validation<F>(mut self, validation: F) -> Self
    where
        F: Fn(&IntegrationTestResult) -> Result<(), String> + Send + Sync + 'static,
    {
        self.validation = Some(Box::new(validation));
        self
    }

    pub fn run_integration_test(&self) -> Result<IntegrationTestResult, String> {
        let temp_dir = if let Some(dir) = &self.config.temp_dir {
            dir.clone()
        } else {
            let temp_path = env::temp_dir().join(format!("aether_test_{}", std::process::id()));
            std::fs::create_dir_all(&temp_path).map_err(|e| format!("Failed to create temp directory: {}", e))?;
            temp_path
        };

        // Setup
        if let Some(setup) = &self.setup {
            setup(&temp_dir)?;
        }

        let mut results = Vec::new();

        // Test each target platform
        for target in &self.config.target_platforms {
            let result = self.test_compilation_target(&temp_dir, target)?;
            results.push(result);
        }

        // Teardown
        if let Some(teardown) = &self.teardown {
            teardown(&temp_dir)?;
        }

        // Clean up temp directory if not preserving artifacts
        if !self.config.preserve_artifacts {
            let _ = fs::remove_dir_all(&temp_dir);
        }

        let integration_result = IntegrationTestResult {
            test_name: self.name.clone(),
            target_results: results,
            temp_dir: temp_dir.clone(),
        };

        // Run custom validation if provided
        if let Some(validation) = &self.validation {
            validation(&integration_result)?;
        }

        Ok(integration_result)
    }

    fn test_compilation_target(&self, temp_dir: &Path, target: &CompilationTarget) -> Result<TargetTestResult, String> {
        let source_file = temp_dir.join(format!("{}.ae", self.name));
        let output_file = temp_dir.join(format!("{}_{}", self.name, target.as_str()));

        // Write source code to file
        fs::write(&source_file, &self.source_code)
            .map_err(|e| format!("Failed to write source file: {}", e))?;

        let start_time = Instant::now();

        // Compile
        let compile_result = self.compile_source(&source_file, &output_file, target)?;
        let compile_time = start_time.elapsed();

        if !compile_result.success {
            return Ok(TargetTestResult {
                target: target.clone(),
                compilation: compile_result,
                execution: None,
                total_time: compile_time,
            });
        }

        // Execute (if compilation succeeded and target supports execution)
        let execution_result = if target == &CompilationTarget::Native {
            Some(self.execute_binary(&output_file)?)
        } else {
            None
        };

        let total_time = start_time.elapsed();

        Ok(TargetTestResult {
            target: target.clone(),
            compilation: compile_result,
            execution: execution_result,
            total_time,
        })
    }

    fn compile_source(&self, source_file: &Path, output_file: &Path, target: &CompilationTarget) -> Result<CompilationResult, String> {
        let mut cmd = Command::new("aetherc");
        cmd.arg("build")
           .arg("--target")
           .arg(target.as_str())
           .arg("--output")
           .arg(output_file)
           .arg(source_file);

        // Add compiler flags
        for flag in &self.config.compiler_flags {
            cmd.arg(flag);
        }

        cmd.stdout(Stdio::piped())
           .stderr(Stdio::piped());

        let start = Instant::now();
        let output = cmd.output()
            .map_err(|e| format!("Failed to execute compiler: {}", e))?;
        let duration = start.elapsed();

        Ok(CompilationResult {
            success: output.status.success(),
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            duration,
        })
    }

    fn execute_binary(&self, binary_path: &Path) -> Result<ExecutionResult, String> {
        let mut cmd = Command::new(binary_path);
        cmd.stdout(Stdio::piped())
           .stderr(Stdio::piped());

        let start = Instant::now();
        let output = cmd.output()
            .map_err(|e| format!("Failed to execute binary: {}", e))?;
        let duration = start.elapsed();

        Ok(ExecutionResult {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            duration,
        })
    }
}

impl TestCase for IntegrationTest {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _context: &mut TestContext) -> TestResult {
        let start = Instant::now();
        
        match self.run_integration_test() {
            Ok(result) => {
                let duration = start.elapsed();
                
                // Check if all targets passed
                let all_passed = result.target_results.iter().all(|r| {
                    r.compilation.success && 
                    r.execution.as_ref().map_or(true, |e| e.exit_code == self.expected_exit_code)
                });

                if all_passed {
                    // Check expected output if specified
                    if let Some(expected) = &self.expected_output {
                        for target_result in &result.target_results {
                            if let Some(execution) = &target_result.execution {
                                if execution.stdout.trim() != expected.trim() {
                                    return TestResult::failed(
                                        self.name.clone(),
                                        duration,
                                        format!("Output mismatch for target {:?}. Expected: '{}', Got: '{}'",
                                               target_result.target, expected, execution.stdout.trim())
                                    );
                                }
                            }
                        }
                    }
                    
                    TestResult::passed(self.name.clone(), duration)
                        .with_metadata("targets_tested".to_string(), result.target_results.len().to_string())
                } else {
                    let failed_targets: Vec<String> = result.target_results.iter()
                        .filter(|r| !r.compilation.success || 
                                   r.execution.as_ref().map_or(false, |e| e.exit_code != self.expected_exit_code))
                        .map(|r| format!("{:?}", r.target))
                        .collect();
                    
                    TestResult::failed(
                        self.name.clone(),
                        duration,
                        format!("Failed targets: {}", failed_targets.join(", "))
                    )
                }
            }
            Err(error) => {
                TestResult::error(self.name.clone(), start.elapsed(), error)
            }
        }
    }

    fn timeout(&self) -> Option<Duration> {
        Some(self.config.timeout)
    }
}

/// Result of an integration test
#[derive(Debug, Clone)]
pub struct IntegrationTestResult {
    pub test_name: String,
    pub target_results: Vec<TargetTestResult>,
    pub temp_dir: PathBuf,
}

/// Result for a specific compilation target
#[derive(Debug, Clone)]
pub struct TargetTestResult {
    pub target: CompilationTarget,
    pub compilation: CompilationResult,
    pub execution: Option<ExecutionResult>,
    pub total_time: Duration,
}

/// Compilation result
#[derive(Debug, Clone)]
pub struct CompilationResult {
    pub success: bool,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration: Duration,
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration: Duration,
}

/// Integration test suite builder
pub struct IntegrationTestSuite {
    pub name: String,
    pub tests: Vec<IntegrationTest>,
    pub common_config: IntegrationConfig,
}

impl IntegrationTestSuite {
    pub fn new(name: &str) -> Self {
        IntegrationTestSuite {
            name: name.to_string(),
            tests: Vec::new(),
            common_config: IntegrationConfig::default(),
        }
    }

    pub fn with_config(mut self, config: IntegrationConfig) -> Self {
        self.common_config = config;
        self
    }

    pub fn add_test(mut self, mut test: IntegrationTest) -> Self {
        // Apply common config if test doesn't have custom config
        if test.config.target_platforms == vec![CompilationTarget::Native] && 
           self.common_config.target_platforms != vec![CompilationTarget::Native] {
            test.config.target_platforms = self.common_config.target_platforms.clone();
        }
        
        self.tests.push(test);
        self
    }

    pub fn add_simple_test(self, name: &str, source: &str, expected_output: &str) -> Self {
        let test = IntegrationTest::new(name, source)
            .expect_output(expected_output);
        self.add_test(test)
    }

    pub fn add_compilation_test(self, name: &str, source: &str) -> Self {
        let test = IntegrationTest::new(name, source);
        self.add_test(test)
    }
}

/// Predefined integration test scenarios
pub struct IntegrationScenarios;

impl IntegrationScenarios {
    /// Test basic arithmetic compilation and execution
    pub fn arithmetic_test() -> IntegrationTest {
        IntegrationTest::new(
            "arithmetic_basic",
            r#"
            fn main() {
                let a = 5
                let b = 3
                let result = a + b * 2
                println!("{}", result)
            }
            "#
        ).expect_output("11")
    }

    /// Test tensor operations
    pub fn tensor_test() -> IntegrationTest {
        IntegrationTest::new(
            "tensor_basic",
            r#"
            fn main() {
                let a = tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                let b = tensor([2, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                let result = a + b
                println!("{}", result.sum())
            }
            "#
        ).expect_output("27")
    }

    /// Test automatic differentiation
    pub fn autodiff_test() -> IntegrationTest {
        IntegrationTest::new(
            "autodiff_basic",
            r#"
            @differentiable
            fn square(x: f64) -> f64 {
                x * x
            }

            fn main() {
                let x = 3.0
                let grad = gradient(square, x)
                println!("{}", grad)
            }
            "#
        ).expect_output("6")
    }

    /// Test probabilistic programming
    pub fn probabilistic_test() -> IntegrationTest {
        IntegrationTest::new(
            "probabilistic_basic",
            r#"
            fn main() {
                let weight ~ Normal(0.0, 1.0)
                observe(data, 1.5)
                let posterior = infer(weight)
                println!("Posterior mean: {:.2}", posterior.mean())
            }
            "#
        )
    }

    /// Test FFI integration
    pub fn ffi_test() -> IntegrationTest {
        IntegrationTest::new(
            "ffi_basic",
            r#"
            extern "C" {
                fn sqrt(x: f64) -> f64;
            }

            fn main() {
                let result = sqrt(16.0)
                println!("{}", result)
            }
            "#
        ).expect_output("4")
    }

    /// Test concurrency
    pub fn concurrency_test() -> IntegrationTest {
        IntegrationTest::new(
            "concurrency_basic",
            r#"
            fn main() {
                let data = [1, 2, 3, 4, 5]
                let result = @parallel for x in data {
                    x * x
                }
                println!("{}", result.sum())
            }
            "#
        ).expect_output("55")
    }

    /// Test error handling
    pub fn error_handling_test() -> IntegrationTest {
        IntegrationTest::new(
            "error_handling",
            r#"
            fn divide(a: f64, b: f64) -> Result<f64, String> {
                if b == 0.0 {
                    Err("Division by zero")
                } else {
                    Ok(a / b)
                }
            }

            fn main() {
                match divide(10.0, 2.0) {
                    Ok(result) => println!("{}", result),
                    Err(e) => println!("Error: {}", e),
                }
            }
            "#
        ).expect_output("5")
    }

    /// Create a comprehensive test suite
    pub fn comprehensive_suite() -> IntegrationTestSuite {
        IntegrationTestSuite::new("comprehensive_integration")
            .with_config(IntegrationConfig {
                target_platforms: vec![
                    CompilationTarget::Native,
                    CompilationTarget::WebAssembly,
                ],
                timeout: Duration::from_secs(120),
                ..IntegrationConfig::default()
            })
            .add_test(Self::arithmetic_test())
            .add_test(Self::tensor_test())
            .add_test(Self::autodiff_test())
            .add_test(Self::probabilistic_test())
            .add_test(Self::ffi_test())
            .add_test(Self::concurrency_test())
            .add_test(Self::error_handling_test())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_config_default() {
        let config = IntegrationConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(!config.preserve_artifacts);
        assert!(!config.verbose_output);
        assert_eq!(config.target_platforms, vec![CompilationTarget::Native]);
    }

    #[test]
    fn test_compilation_target_as_str() {
        assert_eq!(CompilationTarget::Native.as_str(), "native");
        assert_eq!(CompilationTarget::WebAssembly.as_str(), "wasm32");
        assert_eq!(CompilationTarget::GPU.as_str(), "gpu");
        assert_eq!(CompilationTarget::Mobile.as_str(), "mobile");
    }

    #[test]
    fn test_integration_test_creation() {
        let test = IntegrationTest::new("test", "fn main() {}")
            .expect_output("hello")
            .expect_exit_code(0);
        
        assert_eq!(test.name, "test");
        assert_eq!(test.source_code, "fn main() {}");
        assert_eq!(test.expected_output, Some("hello".to_string()));
        assert_eq!(test.expected_exit_code, 0);
    }

    #[test]
    fn test_integration_test_suite() {
        let suite = IntegrationTestSuite::new("test_suite")
            .add_simple_test("test1", "code1", "output1")
            .add_compilation_test("test2", "code2");
        
        assert_eq!(suite.name, "test_suite");
        assert_eq!(suite.tests.len(), 2);
        assert_eq!(suite.tests[0].name, "test1");
        assert_eq!(suite.tests[1].name, "test2");
    }

    #[test]
    fn test_predefined_scenarios() {
        let arithmetic = IntegrationScenarios::arithmetic_test();
        assert_eq!(arithmetic.name, "arithmetic_basic");
        assert!(arithmetic.source_code.contains("let a = 5"));
        assert_eq!(arithmetic.expected_output, Some("11".to_string()));

        let tensor = IntegrationScenarios::tensor_test();
        assert_eq!(tensor.name, "tensor_basic");
        assert!(tensor.source_code.contains("tensor"));

        let suite = IntegrationScenarios::comprehensive_suite();
        assert_eq!(suite.name, "comprehensive_integration");
        assert_eq!(suite.tests.len(), 7);
    }
}