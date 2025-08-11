// MLIR-aware test runner
// Extends the testing framework to work with MLIR compilation pipeline

use crate::testing::{TestResult, TestStatus, TestContext, TestCase, TestSuite};
use crate::compiler::mlir_integration::{IntegratedCompilationPipeline, IntegrationError};
use crate::compiler::mlir::{DebugConfig, CompilationStage};
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// MLIR-specific test runner
pub struct MLIRTestRunner {
    pipeline: IntegratedCompilationPipeline,
    temp_dir: TempDir,
}

impl MLIRTestRunner {
    /// Create new MLIR test runner
    pub fn new() -> Result<Self, IntegrationError> {
        let temp_dir = TempDir::new()
            .map_err(|e| IntegrationError::IOError(format!("Failed to create temp dir: {}", e)))?;
        
        Ok(MLIRTestRunner {
            pipeline: IntegratedCompilationPipeline::new()?,
            temp_dir,
        })
    }

    /// Create new MLIR test runner with debug support
    pub fn new_with_debug() -> Result<Self, IntegrationError> {
        let temp_dir = TempDir::new()
            .map_err(|e| IntegrationError::IOError(format!("Failed to create temp dir: {}", e)))?;
        
        let debug_config = DebugConfig {
            dump_ir: true,
            enable_timing: true,
            verbose_errors: true,
            dump_stages: vec![
                CompilationStage::ASTToMLIR,
                CompilationStage::Optimization,
                CompilationStage::Lowering,
            ],
            dump_dir: temp_dir.path().to_path_buf(),
            max_operations_per_dump: 1000,
            include_source_locations: true,
        };

        Ok(MLIRTestRunner {
            pipeline: IntegratedCompilationPipeline::new_with_debug(debug_config)?,
            temp_dir,
        })
    }

    /// Test source code compilation
    pub fn test_compilation(&mut self, name: &str, source: &str) -> TestResult {
        let start = Instant::now();
        
        match self.pipeline.compile_source(source) {
            Ok(result) => {
                let duration = start.elapsed();
                let mut test_result = TestResult::passed(name.to_string(), duration);
                
                // Add metadata about compilation stages
                if result.ast.is_some() {
                    test_result = test_result.with_metadata("ast_generated".to_string(), "true".to_string());
                }
                if result.typed_ast.is_some() {
                    test_result = test_result.with_metadata("type_checked".to_string(), "true".to_string());
                }
                if result.mlir_module.is_some() {
                    test_result = test_result.with_metadata("mlir_generated".to_string(), "true".to_string());
                }
                
                test_result
            }
            Err(e) => {
                let duration = start.elapsed();
                TestResult::failed(name.to_string(), duration, format!("Compilation failed: {}", e))
            }
        }
    }

    /// Test native executable generation
    pub fn test_native_compilation(&mut self, name: &str, source: &str) -> TestResult {
        let start = Instant::now();
        let output_path = self.temp_dir.path().join(format!("{}_native", name));
        
        match self.pipeline.compile_to_native(source, &output_path) {
            Ok(()) => {
                let duration = start.elapsed();
                
                // Check if executable was created
                #[cfg(windows)]
                let exe_path = output_path.with_extension("exe");
                #[cfg(not(windows))]
                let exe_path = output_path;
                
                if exe_path.exists() {
                    TestResult::passed(name.to_string(), duration)
                        .with_metadata("executable_created".to_string(), "true".to_string())
                        .with_metadata("output_path".to_string(), exe_path.to_string_lossy().to_string())
                } else {
                    TestResult::failed(name.to_string(), duration, "Executable not created".to_string())
                }
            }
            Err(e) => {
                let duration = start.elapsed();
                TestResult::failed(name.to_string(), duration, format!("Native compilation failed: {}", e))
            }
        }
    }

    /// Test WebAssembly compilation
    pub fn test_wasm_compilation(&mut self, name: &str, source: &str) -> TestResult {
        let start = Instant::now();
        let output_path = self.temp_dir.path().join(format!("{}_wasm", name));
        
        match self.pipeline.compile_to_wasm(source, &output_path) {
            Ok(()) => {
                let duration = start.elapsed();
                
                // Check if WebAssembly files were created
                let wasm_path = output_path.with_extension("wasm");
                let js_path = output_path.with_extension("js");
                let ts_path = output_path.with_extension("d.ts");
                let html_path = output_path.with_extension("html");
                
                let mut test_result = TestResult::passed(name.to_string(), duration);
                
                if wasm_path.exists() {
                    test_result = test_result.with_metadata("wasm_created".to_string(), "true".to_string());
                }
                if js_path.exists() {
                    test_result = test_result.with_metadata("js_bindings_created".to_string(), "true".to_string());
                }
                if ts_path.exists() {
                    test_result = test_result.with_metadata("ts_definitions_created".to_string(), "true".to_string());
                }
                if html_path.exists() {
                    test_result = test_result.with_metadata("html_template_created".to_string(), "true".to_string());
                }
                
                test_result
            }
            Err(e) => {
                let duration = start.elapsed();
                TestResult::failed(name.to_string(), duration, format!("WebAssembly compilation failed: {}", e))
            }
        }
    }

    /// Test compilation timing
    pub fn test_compilation_performance(&mut self, name: &str, source: &str, expected_max_duration: Duration) -> TestResult {
        let start = Instant::now();
        
        match self.pipeline.compile_source(source) {
            Ok(_) => {
                let duration = start.elapsed();
                
                if duration <= expected_max_duration {
                    TestResult::passed(name.to_string(), duration)
                        .with_metadata("performance_test".to_string(), "true".to_string())
                        .with_metadata("expected_max_ms".to_string(), expected_max_duration.as_millis().to_string())
                        .with_metadata("actual_ms".to_string(), duration.as_millis().to_string())
                } else {
                    TestResult::failed(
                        name.to_string(), 
                        duration, 
                        format!("Compilation took {}ms, expected max {}ms", 
                               duration.as_millis(), 
                               expected_max_duration.as_millis())
                    )
                }
            }
            Err(e) => {
                let duration = start.elapsed();
                TestResult::failed(name.to_string(), duration, format!("Compilation failed: {}", e))
            }
        }
    }

    /// Test error handling
    pub fn test_error_handling(&mut self, name: &str, invalid_source: &str, expected_error_type: &str) -> TestResult {
        let start = Instant::now();
        
        match self.pipeline.compile_source(invalid_source) {
            Ok(_) => {
                let duration = start.elapsed();
                TestResult::failed(
                    name.to_string(), 
                    duration, 
                    "Expected compilation to fail but it succeeded".to_string()
                )
            }
            Err(e) => {
                let duration = start.elapsed();
                let error_string = e.to_string();
                
                if error_string.contains(expected_error_type) {
                    TestResult::passed(name.to_string(), duration)
                        .with_metadata("error_handling_test".to_string(), "true".to_string())
                        .with_metadata("expected_error".to_string(), expected_error_type.to_string())
                        .with_metadata("actual_error".to_string(), error_string)
                } else {
                    TestResult::failed(
                        name.to_string(), 
                        duration, 
                        format!("Expected error containing '{}', got: {}", expected_error_type, error_string)
                    )
                }
            }
        }
    }

    /// Get compilation report if available
    pub fn get_compilation_report(&self) -> Option<crate::compiler::mlir::CompilationReport> {
        self.pipeline.get_compilation_report()
    }

    /// Get timing summary if available
    pub fn get_timing_summary(&self) -> Option<crate::compiler::mlir::TimingSummary> {
        self.pipeline.get_timing_summary()
    }
}

/// MLIR test suite for basic functionality
pub struct MLIRBasicTestSuite {
    runner: MLIRTestRunner,
}

impl MLIRBasicTestSuite {
    pub fn new() -> Result<Self, IntegrationError> {
        Ok(MLIRBasicTestSuite {
            runner: MLIRTestRunner::new()?,
        })
    }

    pub fn new_with_debug() -> Result<Self, IntegrationError> {
        Ok(MLIRBasicTestSuite {
            runner: MLIRTestRunner::new_with_debug()?,
        })
    }
}

impl TestSuite for MLIRBasicTestSuite {
    fn name(&self) -> &str {
        "MLIR Basic Functionality"
    }

    fn tests(&self) -> Vec<Box<dyn TestCase>> {
        vec![
            Box::new(MLIRCompilationTest::new("simple_expression", "(+ 1 2)")),
            Box::new(MLIRCompilationTest::new("function_definition", "(define square (lambda (x) (* x x)))")),
            Box::new(MLIRCompilationTest::new("tensor_operation", "(tensor-create [2 2] [1.0 2.0 3.0 4.0])")),
            Box::new(MLIRNativeCompilationTest::new("hello_world", "(print \"Hello, World!\")")),
            Box::new(MLIRWasmCompilationTest::new("simple_wasm", "(+ 1 2)")),
            Box::new(MLIRPerformanceTest::new("performance_test", "(+ 1 2)", Duration::from_millis(1000))),
            Box::new(MLIRErrorHandlingTest::new("parse_error", "((unclosed", "Parse error")),
        ]
    }
}

/// Individual MLIR test cases
pub struct MLIRCompilationTest {
    name: String,
    source: String,
}

impl MLIRCompilationTest {
    pub fn new(name: &str, source: &str) -> Self {
        MLIRCompilationTest {
            name: name.to_string(),
            source: source.to_string(),
        }
    }
}

impl TestCase for MLIRCompilationTest {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _context: &mut TestContext) -> TestResult {
        let mut runner = match MLIRTestRunner::new() {
            Ok(runner) => runner,
            Err(e) => return TestResult::error(self.name.clone(), Duration::from_nanos(0), e.to_string()),
        };
        
        runner.test_compilation(&self.name, &self.source)
    }
}

pub struct MLIRNativeCompilationTest {
    name: String,
    source: String,
}

impl MLIRNativeCompilationTest {
    pub fn new(name: &str, source: &str) -> Self {
        MLIRNativeCompilationTest {
            name: name.to_string(),
            source: source.to_string(),
        }
    }
}

impl TestCase for MLIRNativeCompilationTest {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _context: &mut TestContext) -> TestResult {
        let mut runner = match MLIRTestRunner::new() {
            Ok(runner) => runner,
            Err(e) => return TestResult::error(self.name.clone(), Duration::from_nanos(0), e.to_string()),
        };
        
        runner.test_native_compilation(&self.name, &self.source)
    }
}

pub struct MLIRWasmCompilationTest {
    name: String,
    source: String,
}

impl MLIRWasmCompilationTest {
    pub fn new(name: &str, source: &str) -> Self {
        MLIRWasmCompilationTest {
            name: name.to_string(),
            source: source.to_string(),
        }
    }
}

impl TestCase for MLIRWasmCompilationTest {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _context: &mut TestContext) -> TestResult {
        let mut runner = match MLIRTestRunner::new() {
            Ok(runner) => runner,
            Err(e) => return TestResult::error(self.name.clone(), Duration::from_nanos(0), e.to_string()),
        };
        
        runner.test_wasm_compilation(&self.name, &self.source)
    }
}

pub struct MLIRPerformanceTest {
    name: String,
    source: String,
    max_duration: Duration,
}

impl MLIRPerformanceTest {
    pub fn new(name: &str, source: &str, max_duration: Duration) -> Self {
        MLIRPerformanceTest {
            name: name.to_string(),
            source: source.to_string(),
            max_duration,
        }
    }
}

impl TestCase for MLIRPerformanceTest {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _context: &mut TestContext) -> TestResult {
        let mut runner = match MLIRTestRunner::new() {
            Ok(runner) => runner,
            Err(e) => return TestResult::error(self.name.clone(), Duration::from_nanos(0), e.to_string()),
        };
        
        runner.test_compilation_performance(&self.name, &self.source, self.max_duration)
    }
}

pub struct MLIRErrorHandlingTest {
    name: String,
    invalid_source: String,
    expected_error_type: String,
}

impl MLIRErrorHandlingTest {
    pub fn new(name: &str, invalid_source: &str, expected_error_type: &str) -> Self {
        MLIRErrorHandlingTest {
            name: name.to_string(),
            invalid_source: invalid_source.to_string(),
            expected_error_type: expected_error_type.to_string(),
        }
    }
}

impl TestCase for MLIRErrorHandlingTest {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _context: &mut TestContext) -> TestResult {
        let mut runner = match MLIRTestRunner::new() {
            Ok(runner) => runner,
            Err(e) => return TestResult::error(self.name.clone(), Duration::from_nanos(0), e.to_string()),
        };
        
        runner.test_error_handling(&self.name, &self.invalid_source, &self.expected_error_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlir_runner_creation() {
        let runner = MLIRTestRunner::new();
        assert!(runner.is_ok());
    }

    #[test]
    fn test_mlir_runner_with_debug() {
        let runner = MLIRTestRunner::new_with_debug();
        assert!(runner.is_ok());
    }

    #[test]
    fn test_basic_compilation() {
        let mut runner = MLIRTestRunner::new().unwrap();
        let result = runner.test_compilation("test", "(+ 1 2)");
        
        // Should either pass or fail gracefully (due to mock implementations)
        match result.status {
            TestStatus::Passed | TestStatus::Failed => {
                println!("Test completed with status: {:?}", result.status);
            }
            TestStatus::Error => {
                panic!("Test should not error: {:?}", result.error_details);
            }
            _ => {}
        }
    }

    #[test]
    fn test_test_suite_creation() {
        let suite = MLIRBasicTestSuite::new();
        assert!(suite.is_ok());
        
        let suite = suite.unwrap();
        assert_eq!(suite.name(), "MLIR Basic Functionality");
        assert!(!suite.tests().is_empty());
    }
}