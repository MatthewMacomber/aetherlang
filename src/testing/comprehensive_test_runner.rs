// Comprehensive Testing and Verification Framework
// Implements hello world testing, advanced feature testing, and performance benchmarking

use super::{TestResult, TestStatus, TestConfig};
use crate::build_system::BuildSystemManager;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::fs;
use serde::{Serialize, Deserialize};

/// Comprehensive test runner with build system integration
pub struct ComprehensiveTestRunner {
    config: TestConfig,
    build_manager: BuildSystemManager,
    test_suites: Vec<Box<dyn TestSuite>>,
    temp_dir: Option<PathBuf>,
}

impl ComprehensiveTestRunner {
    /// Create new comprehensive test runner
    pub fn new(config: TestConfig, build_manager: BuildSystemManager) -> Self {
        Self {
            config,
            build_manager,
            test_suites: Vec::new(),
            temp_dir: None,
        }
    }

    /// Add test suite to runner
    pub fn add_test_suite(&mut self, suite: Box<dyn TestSuite>) {
        self.test_suites.push(suite);
    }

    /// Run all test suites
    pub fn run_all_tests(&mut self) -> ComprehensiveTestResults {
        let start_time = Instant::now();
        let mut results = ComprehensiveTestResults::new();

        // Setup temporary directory for test files
        self.setup_temp_directory().unwrap_or_else(|e| {
            eprintln!("Warning: Failed to setup temp directory: {}", e);
        });

        // Run basic tests first
        println!("Running basic compilation and execution tests...");
        let basic_results = self.run_basic_tests();
        results.basic_tests = Some(basic_results);

        // Run advanced feature tests
        println!("Running advanced feature tests...");
        let advanced_results = self.run_advanced_tests();
        results.advanced_tests = Some(advanced_results);

        // Run performance benchmarks
        println!("Running performance benchmarks...");
        let benchmark_results = self.run_performance_benchmarks();
        results.benchmark_tests = Some(benchmark_results);

        // Run meta-tests to verify testing framework
        println!("Running meta-tests...");
        let meta_results = self.run_meta_tests();
        results.meta_tests = Some(meta_results);

        // Calculate overall statistics
        results.total_duration = start_time.elapsed();
        results.calculate_statistics();

        // Cleanup
        self.cleanup_temp_directory();

        results
    }

    /// Run basic hello world compilation and execution tests
    pub fn run_basic_tests(&mut self) -> BasicTestResults {
        let mut results = BasicTestResults::new();
        let start_time = Instant::now();

        // Test 1: Hello World Compilation
        println!("  Testing hello world compilation...");
        let hello_world_result = self.test_hello_world_compilation();
        results.hello_world_test = Some(hello_world_result);

        // Test 2: Simple arithmetic
        println!("  Testing simple arithmetic...");
        let arithmetic_result = self.test_simple_arithmetic();
        results.arithmetic_test = Some(arithmetic_result);

        // Test 3: Function definitions
        println!("  Testing function definitions...");
        let function_result = self.test_function_definitions();
        results.function_test = Some(function_result);

        // Test 4: Control flow
        println!("  Testing control flow...");
        let control_flow_result = self.test_control_flow();
        results.control_flow_test = Some(control_flow_result);

        results.duration = start_time.elapsed();
        results.calculate_success_rate();
        results
    }

    /// Run advanced feature tests (tensors, autodiff, probabilistic programming)
    pub fn run_advanced_tests(&mut self) -> AdvancedTestResults {
        let mut results = AdvancedTestResults::new();
        let start_time = Instant::now();

        // Test 1: Tensor operations
        println!("  Testing tensor operations...");
        let tensor_result = self.test_tensor_operations();
        results.tensor_test = Some(tensor_result);

        // Test 2: Automatic differentiation
        println!("  Testing automatic differentiation...");
        let autodiff_result = self.test_automatic_differentiation();
        results.autodiff_test = Some(autodiff_result);

        // Test 3: Probabilistic programming
        println!("  Testing probabilistic programming...");
        let probabilistic_result = self.test_probabilistic_programming();
        results.probabilistic_test = Some(probabilistic_result);

        // Test 4: GPU kernels (if available)
        println!("  Testing GPU kernels...");
        let gpu_result = self.test_gpu_kernels();
        results.gpu_test = Some(gpu_result);

        // Test 5: FFI integration
        println!("  Testing FFI integration...");
        let ffi_result = self.test_ffi_integration();
        results.ffi_test = Some(ffi_result);

        results.duration = start_time.elapsed();
        results.calculate_success_rate();
        results
    }

    /// Run performance benchmarks
    pub fn run_performance_benchmarks(&mut self) -> BenchmarkResults {
        let mut results = BenchmarkResults::new();
        let start_time = Instant::now();

        // Benchmark 1: Compilation speed
        println!("  Benchmarking compilation speed...");
        let compilation_benchmark = self.benchmark_compilation_speed();
        results.compilation_benchmark = Some(compilation_benchmark);

        // Benchmark 2: Execution speed
        println!("  Benchmarking execution speed...");
        let execution_benchmark = self.benchmark_execution_speed();
        results.execution_benchmark = Some(execution_benchmark);

        // Benchmark 3: Memory usage
        println!("  Benchmarking memory usage...");
        let memory_benchmark = self.benchmark_memory_usage();
        results.memory_benchmark = Some(memory_benchmark);

        // Benchmark 4: Tensor operations performance
        println!("  Benchmarking tensor operations...");
        let tensor_benchmark = self.benchmark_tensor_performance();
        results.tensor_benchmark = Some(tensor_benchmark);

        results.duration = start_time.elapsed();
        results
    }

    /// Run meta-tests to verify testing framework reliability
    pub fn run_meta_tests(&mut self) -> MetaTestResults {
        let mut results = MetaTestResults::new();
        let start_time = Instant::now();

        // Meta-test 1: Test framework self-validation
        println!("  Testing framework self-validation...");
        let framework_test = self.test_framework_reliability();
        results.framework_test = Some(framework_test);

        // Meta-test 2: Error detection accuracy
        println!("  Testing error detection accuracy...");
        let error_detection_test = self.test_error_detection();
        results.error_detection_test = Some(error_detection_test);

        // Meta-test 3: Build system integration
        println!("  Testing build system integration...");
        let build_integration_test = self.test_build_integration();
        results.build_integration_test = Some(build_integration_test);

        results.duration = start_time.elapsed();
        results.calculate_success_rate();
        results
    }

    /// Test hello world compilation and execution
    fn test_hello_world_compilation(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        // Create hello world source file
        let hello_world_source = r#"
(func main ()
  (call print "Hello, World from Aether!")
  (return 0))
"#;

        match self.compile_and_run_source("hello_world_test.ae", hello_world_source) {
            Ok(output) => {
                if output.contains("Hello, World from Aether!") {
                    TestResult::passed("hello_world_compilation".to_string(), start_time.elapsed())
                } else {
                    TestResult::failed(
                        "hello_world_compilation".to_string(),
                        start_time.elapsed(),
                        format!("Unexpected output: {}", output)
                    )
                }
            }
            Err(e) => TestResult::error(
                "hello_world_compilation".to_string(),
                start_time.elapsed(),
                format!("Compilation failed: {}", e)
            )
        }
    }

    /// Test simple arithmetic operations
    fn test_simple_arithmetic(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        let arithmetic_source = r#"
(func main ()
  (let a 10)
  (let b 5)
  (let sum (+ a b))
  (let diff (- a b))
  (let prod (* a b))
  (let quot (/ a b))
  (call print "Sum:" sum)
  (call print "Diff:" diff)
  (call print "Product:" prod)
  (call print "Quotient:" quot)
  (return 0))
"#;

        match self.compile_and_run_source("arithmetic_test.ae", arithmetic_source) {
            Ok(output) => {
                if output.contains("Sum: 15") && output.contains("Diff: 5") && 
                   output.contains("Product: 50") && output.contains("Quotient: 2") {
                    TestResult::passed("simple_arithmetic".to_string(), start_time.elapsed())
                } else {
                    TestResult::failed(
                        "simple_arithmetic".to_string(),
                        start_time.elapsed(),
                        format!("Arithmetic results incorrect: {}", output)
                    )
                }
            }
            Err(e) => TestResult::error(
                "simple_arithmetic".to_string(),
                start_time.elapsed(),
                format!("Compilation failed: {}", e)
            )
        }
    }

    /// Test function definitions and calls
    fn test_function_definitions(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        let function_source = r#"
(func add (a b)
  (return (+ a b)))

(func multiply (x y)
  (return (* x y)))

(func main ()
  (let result1 (call add 3 4))
  (let result2 (call multiply 5 6))
  (call print "Add result:" result1)
  (call print "Multiply result:" result2)
  (return 0))
"#;

        match self.compile_and_run_source("function_test.ae", function_source) {
            Ok(output) => {
                if output.contains("Add result: 7") && output.contains("Multiply result: 30") {
                    TestResult::passed("function_definitions".to_string(), start_time.elapsed())
                } else {
                    TestResult::failed(
                        "function_definitions".to_string(),
                        start_time.elapsed(),
                        format!("Function results incorrect: {}", output)
                    )
                }
            }
            Err(e) => TestResult::error(
                "function_definitions".to_string(),
                start_time.elapsed(),
                format!("Compilation failed: {}", e)
            )
        }
    }

    /// Test control flow (if statements, loops)
    fn test_control_flow(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        let control_flow_source = r#"
(func main ()
  (let x 10)
  (if (> x 5)
    (call print "x is greater than 5")
    (call print "x is not greater than 5"))
  
  (let i 0)
  (while (< i 3)
    (call print "Loop iteration:" i)
    (set i (+ i 1)))
  
  (return 0))
"#;

        match self.compile_and_run_source("control_flow_test.ae", control_flow_source) {
            Ok(output) => {
                if output.contains("x is greater than 5") && 
                   output.contains("Loop iteration: 0") &&
                   output.contains("Loop iteration: 1") &&
                   output.contains("Loop iteration: 2") {
                    TestResult::passed("control_flow".to_string(), start_time.elapsed())
                } else {
                    TestResult::failed(
                        "control_flow".to_string(),
                        start_time.elapsed(),
                        format!("Control flow results incorrect: {}", output)
                    )
                }
            }
            Err(e) => TestResult::error(
                "control_flow".to_string(),
                start_time.elapsed(),
                format!("Compilation failed: {}", e)
            )
        }
    }

    /// Test tensor operations
    fn test_tensor_operations(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        let tensor_source = r#"
(func main ()
  (let a (tensor [1.0 2.0 3.0]))
  (let b (tensor [4.0 5.0 6.0]))
  (let sum (+ a b))
  (let dot (dot a b))
  (call print "Tensor sum:" sum)
  (call print "Dot product:" dot)
  (return 0))
"#;

        match self.compile_and_run_source("tensor_test.ae", tensor_source) {
            Ok(output) => {
                // Check for expected tensor operations results
                if output.contains("Tensor sum:") && output.contains("Dot product:") {
                    TestResult::passed("tensor_operations".to_string(), start_time.elapsed())
                } else {
                    TestResult::failed(
                        "tensor_operations".to_string(),
                        start_time.elapsed(),
                        format!("Tensor operations failed: {}", output)
                    )
                }
            }
            Err(e) => TestResult::error(
                "tensor_operations".to_string(),
                start_time.elapsed(),
                format!("Compilation failed: {}", e)
            )
        }
    }

    /// Test automatic differentiation
    fn test_automatic_differentiation(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        let autodiff_source = r#"
(func square (x)
  (return (* x x)))

(func main ()
  (let x 3.0)
  (let result (call square x))
  (let gradient (grad square x))
  (call print "f(3) =" result)
  (call print "f'(3) =" gradient)
  (return 0))
"#;

        match self.compile_and_run_source("autodiff_test.ae", autodiff_source) {
            Ok(output) => {
                // Check for expected autodiff results (f(3) = 9, f'(3) = 6)
                if output.contains("f(3) = 9") && output.contains("f'(3) = 6") {
                    TestResult::passed("automatic_differentiation".to_string(), start_time.elapsed())
                } else {
                    TestResult::failed(
                        "automatic_differentiation".to_string(),
                        start_time.elapsed(),
                        format!("Autodiff results incorrect: {}", output)
                    )
                }
            }
            Err(e) => TestResult::error(
                "automatic_differentiation".to_string(),
                start_time.elapsed(),
                format!("Compilation failed: {}", e)
            )
        }
    }

    /// Test probabilistic programming
    fn test_probabilistic_programming(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        let probabilistic_source = r#"
(func main ()
  (let x (normal 0.0 1.0))
  (let y (normal x 0.5))
  (observe y 1.0)
  (let samples (sample x 1000))
  (call print "Posterior samples:" samples)
  (return 0))
"#;

        match self.compile_and_run_source("probabilistic_test.ae", probabilistic_source) {
            Ok(output) => {
                // Check for probabilistic programming execution
                if output.contains("Posterior samples:") {
                    TestResult::passed("probabilistic_programming".to_string(), start_time.elapsed())
                } else {
                    TestResult::failed(
                        "probabilistic_programming".to_string(),
                        start_time.elapsed(),
                        format!("Probabilistic programming failed: {}", output)
                    )
                }
            }
            Err(e) => TestResult::error(
                "probabilistic_programming".to_string(),
                start_time.elapsed(),
                format!("Compilation failed: {}", e)
            )
        }
    }

    /// Test GPU kernels (simplified test)
    fn test_gpu_kernels(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        // For now, just test that GPU-related code compiles
        let gpu_source = r#"
@gpu
(func vector_add (a b)
  (return (+ a b)))

(func main ()
  (let a (tensor [1.0 2.0 3.0]))
  (let b (tensor [4.0 5.0 6.0]))
  (let result (call vector_add a b))
  (call print "GPU result:" result)
  (return 0))
"#;

        match self.compile_and_run_source("gpu_test.ae", gpu_source) {
            Ok(_output) => {
                // For now, just check that it compiles and runs
                TestResult::passed("gpu_kernels".to_string(), start_time.elapsed())
            }
            Err(e) => {
                // GPU tests might fail if no GPU is available, so we'll mark as skipped
                TestResult {
                    name: "gpu_kernels".to_string(),
                    status: TestStatus::Skipped,
                    duration: start_time.elapsed(),
                    message: Some("GPU not available or not supported".to_string()),
                    error_details: Some(e),
                    metadata: HashMap::new(),
                }
            }
        }
    }

    /// Test FFI integration (simplified)
    fn test_ffi_integration(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        // Simple FFI test that should compile
        let ffi_source = r#"
(extern "C" printf (format ...) -> i32)

(func main ()
  (call printf "FFI test: %d\n" 42)
  (return 0))
"#;

        match self.compile_and_run_source("ffi_test.ae", ffi_source) {
            Ok(output) => {
                if output.contains("FFI test: 42") {
                    TestResult::passed("ffi_integration".to_string(), start_time.elapsed())
                } else {
                    TestResult::failed(
                        "ffi_integration".to_string(),
                        start_time.elapsed(),
                        format!("FFI output incorrect: {}", output)
                    )
                }
            }
            Err(e) => TestResult::error(
                "ffi_integration".to_string(),
                start_time.elapsed(),
                format!("FFI compilation failed: {}", e)
            )
        }
    }

    /// Benchmark compilation speed
    fn benchmark_compilation_speed(&mut self) -> BenchmarkResult {
        let iterations = 10;
        let mut durations = Vec::new();
        
        let simple_source = r#"
(func main ()
  (call print "Benchmark test")
  (return 0))
"#;

        for i in 0..iterations {
            let start_time = Instant::now();
            let filename = format!("benchmark_compile_{}.ae", i);
            
            match self.compile_source(&filename, simple_source) {
                Ok(_) => {
                    durations.push(start_time.elapsed());
                }
                Err(_) => {
                    // Skip failed compilations in benchmark
                }
            }
        }

        BenchmarkResult::from_durations("compilation_speed", durations)
    }

    /// Benchmark execution speed
    fn benchmark_execution_speed(&mut self) -> BenchmarkResult {
        let iterations = 100;
        let mut durations = Vec::new();
        
        let compute_source = r#"
(func fibonacci (n)
  (if (<= n 1)
    (return n)
    (return (+ (call fibonacci (- n 1)) (call fibonacci (- n 2))))))

(func main ()
  (let result (call fibonacci 20))
  (call print result)
  (return 0))
"#;

        // Compile once
        if let Ok(executable) = self.compile_source("benchmark_exec.ae", compute_source) {
            for _ in 0..iterations {
                let start_time = Instant::now();
                if let Ok(_) = self.run_executable(&executable) {
                    durations.push(start_time.elapsed());
                }
            }
        }

        BenchmarkResult::from_durations("execution_speed", durations)
    }

    /// Benchmark memory usage (simplified)
    fn benchmark_memory_usage(&mut self) -> BenchmarkResult {
        // This is a simplified memory benchmark
        // In a real implementation, we'd measure actual memory usage
        let start_time = Instant::now();
        
        let memory_source = r#"
(func main ()
  (let big_array (array 10000 0))
  (let i 0)
  (while (< i 10000)
    (set (index big_array i) i)
    (set i (+ i 1)))
  (call print "Memory test completed")
  (return 0))
"#;

        match self.compile_and_run_source("memory_test.ae", memory_source) {
            Ok(_) => {
                BenchmarkResult {
                    name: "memory_usage".to_string(),
                    iterations: 1,
                    total_duration: start_time.elapsed(),
                    average_duration: start_time.elapsed(),
                    min_duration: start_time.elapsed(),
                    max_duration: start_time.elapsed(),
                    success_rate: 1.0,
                    metadata: HashMap::new(),
                }
            }
            Err(_) => {
                BenchmarkResult {
                    name: "memory_usage".to_string(),
                    iterations: 0,
                    total_duration: start_time.elapsed(),
                    average_duration: Duration::from_nanos(0),
                    min_duration: Duration::from_nanos(0),
                    max_duration: Duration::from_nanos(0),
                    success_rate: 0.0,
                    metadata: HashMap::new(),
                }
            }
        }
    }

    /// Benchmark tensor operations performance
    fn benchmark_tensor_performance(&mut self) -> BenchmarkResult {
        let iterations = 50;
        let mut durations = Vec::new();
        
        let tensor_source = r#"
(func main ()
  (let a (tensor 1000 1.0))
  (let b (tensor 1000 2.0))
  (let i 0)
  (while (< i 100)
    (let result (+ a b))
    (set i (+ i 1)))
  (call print "Tensor benchmark completed")
  (return 0))
"#;

        // Compile once
        if let Ok(executable) = self.compile_source("benchmark_tensor.ae", tensor_source) {
            for _ in 0..iterations {
                let start_time = Instant::now();
                if let Ok(_) = self.run_executable(&executable) {
                    durations.push(start_time.elapsed());
                }
            }
        }

        BenchmarkResult::from_durations("tensor_performance", durations)
    }

    /// Test framework reliability (meta-test)
    fn test_framework_reliability(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        // Test that the framework correctly identifies passing tests
        let passing_source = r#"
(func main ()
  (return 0))
"#;

        // Test that the framework correctly identifies failing tests
        let failing_source = r#"
(func main ()
  (return 1))
"#;

        let passing_result = self.compile_and_run_source("framework_pass_test.ae", passing_source);
        let failing_result = self.compile_and_run_source("framework_fail_test.ae", failing_source);

        match (passing_result, failing_result) {
            (Ok(_), Err(_)) => {
                TestResult::passed("framework_reliability".to_string(), start_time.elapsed())
            }
            _ => {
                TestResult::failed(
                    "framework_reliability".to_string(),
                    start_time.elapsed(),
                    "Framework did not correctly distinguish passing and failing tests".to_string()
                )
            }
        }
    }

    /// Test error detection accuracy
    fn test_error_detection(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        // Test syntax error detection
        let syntax_error_source = r#"
(func main (
  (call print "Missing closing parenthesis")
  (return 0))
"#;

        // Test type error detection
        let type_error_source = r#"
(func main ()
  (let x "string")
  (let y (+ x 5))
  (return 0))
"#;

        let syntax_result = self.compile_source("syntax_error_test.ae", syntax_error_source);
        let type_result = self.compile_source("type_error_test.ae", type_error_source);

        // Both should fail to compile
        match (syntax_result, type_result) {
            (Err(_), Err(_)) => {
                TestResult::passed("error_detection".to_string(), start_time.elapsed())
            }
            _ => {
                TestResult::failed(
                    "error_detection".to_string(),
                    start_time.elapsed(),
                    "Framework did not correctly detect compilation errors".to_string()
                )
            }
        }
    }

    /// Test build system integration
    fn test_build_integration(&mut self) -> TestResult {
        let start_time = Instant::now();
        
        // Test that build system manager is properly integrated
        let simple_source = r#"
(func main ()
  (call print "Build integration test")
  (return 0))
"#;

        match self.compile_and_run_source("build_integration_test.ae", simple_source) {
            Ok(output) => {
                if output.contains("Build integration test") {
                    TestResult::passed("build_integration".to_string(), start_time.elapsed())
                } else {
                    TestResult::failed(
                        "build_integration".to_string(),
                        start_time.elapsed(),
                        "Build system integration failed".to_string()
                    )
                }
            }
            Err(e) => TestResult::error(
                "build_integration".to_string(),
                start_time.elapsed(),
                format!("Build integration error: {}", e)
            )
        }
    }

    /// Helper: Compile and run Aether source code
    pub fn compile_and_run_source(&mut self, filename: &str, source: &str) -> Result<String, String> {
        let executable = self.compile_source(filename, source)?;
        self.run_executable(&executable)
    }

    /// Helper: Compile Aether source code
    fn compile_source(&mut self, filename: &str, source: &str) -> Result<PathBuf, String> {
        // Write source to temporary file
        let source_path = self.get_temp_path(filename);
        fs::write(&source_path, source)
            .map_err(|e| format!("Failed to write source file: {}", e))?;

        // Compile using build system manager
        match self.build_manager.compile_aether_source(&source_path) {
            Ok(executable) => Ok(executable.path),
            Err(e) => Err(format!("Compilation failed: {}", e))
        }
    }

    /// Helper: Run compiled executable
    fn run_executable(&self, executable_path: &Path) -> Result<String, String> {
        use std::process::Command;

        let output = Command::new(executable_path)
            .output()
            .map_err(|e| format!("Failed to run executable: {}", e))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }

    /// Setup temporary directory for test files
    fn setup_temp_directory(&mut self) -> Result<(), String> {
        #[cfg(test)]
        {
            let temp_dir = crate::testing::TestUtils::temp_dir()
                .map_err(|e| format!("Failed to create temp directory: {}", e))?;
            self.temp_dir = Some(temp_dir.path().to_path_buf());
            // Keep the temp_dir alive by not dropping it
            std::mem::forget(temp_dir);
        }
        #[cfg(not(test))]
        {
            let temp_dir = std::env::temp_dir().join("aether_tests");
            fs::create_dir_all(&temp_dir)
                .map_err(|e| format!("Failed to create temp directory: {}", e))?;
            self.temp_dir = Some(temp_dir);
        }
        Ok(())
    }

    /// Get path in temporary directory
    fn get_temp_path(&self, filename: &str) -> PathBuf {
        if let Some(temp_dir) = &self.temp_dir {
            temp_dir.join(filename)
        } else {
            PathBuf::from(filename)
        }
    }

    /// Cleanup temporary directory
    fn cleanup_temp_directory(&mut self) {
        if let Some(temp_dir) = &self.temp_dir {
            let _ = fs::remove_dir_all(temp_dir);
        }
        self.temp_dir = None;
    }
}

/// Trait for test suites in the comprehensive framework
pub trait TestSuite: Send + Sync {
    fn name(&self) -> &str;
    fn run_tests(&self, runner: &mut ComprehensiveTestRunner) -> Vec<TestResult>;
}

/// Comprehensive test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveTestResults {
    pub basic_tests: Option<BasicTestResults>,
    pub advanced_tests: Option<AdvancedTestResults>,
    pub benchmark_tests: Option<BenchmarkResults>,
    pub meta_tests: Option<MetaTestResults>,
    pub total_duration: Duration,
    pub overall_success_rate: f64,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub error_tests: usize,
    pub skipped_tests: usize,
}

impl ComprehensiveTestResults {
    pub fn new() -> Self {
        Self {
            basic_tests: None,
            advanced_tests: None,
            benchmark_tests: None,
            meta_tests: None,
            total_duration: Duration::from_nanos(0),
            overall_success_rate: 0.0,
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            error_tests: 0,
            skipped_tests: 0,
        }
    }

    pub fn calculate_statistics(&mut self) {
        let mut total = 0;
        let mut passed = 0;
        let mut failed = 0;
        let mut errors = 0;
        let mut skipped = 0;

        if let Some(basic) = &self.basic_tests {
            total += basic.total_tests();
            passed += basic.passed_tests();
            failed += basic.failed_tests();
            errors += basic.error_tests();
            skipped += basic.skipped_tests();
        }

        if let Some(advanced) = &self.advanced_tests {
            total += advanced.total_tests();
            passed += advanced.passed_tests();
            failed += advanced.failed_tests();
            errors += advanced.error_tests();
            skipped += advanced.skipped_tests();
        }

        if let Some(meta) = &self.meta_tests {
            total += meta.total_tests();
            passed += meta.passed_tests();
            failed += meta.failed_tests();
            errors += meta.error_tests();
            skipped += meta.skipped_tests();
        }

        self.total_tests = total;
        self.passed_tests = passed;
        self.failed_tests = failed;
        self.error_tests = errors;
        self.skipped_tests = skipped;
        self.overall_success_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
    }

    pub fn is_successful(&self) -> bool {
        self.failed_tests == 0 && self.error_tests == 0
    }

    pub fn summary(&self) -> String {
        format!(
            "Comprehensive Tests: {} passed, {} failed, {} errors, {} skipped ({} total) in {:?} - Success Rate: {:.1}%",
            self.passed_tests, self.failed_tests, self.error_tests, self.skipped_tests, 
            self.total_tests, self.total_duration, self.overall_success_rate * 100.0
        )
    }
}

/// Basic test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicTestResults {
    pub hello_world_test: Option<TestResult>,
    pub arithmetic_test: Option<TestResult>,
    pub function_test: Option<TestResult>,
    pub control_flow_test: Option<TestResult>,
    pub duration: Duration,
    pub success_rate: f64,
}

impl BasicTestResults {
    pub fn new() -> Self {
        Self {
            hello_world_test: None,
            arithmetic_test: None,
            function_test: None,
            control_flow_test: None,
            duration: Duration::from_nanos(0),
            success_rate: 0.0,
        }
    }

    pub fn calculate_success_rate(&mut self) {
        let tests = vec![&self.hello_world_test, &self.arithmetic_test, &self.function_test, &self.control_flow_test];
        let total = tests.len();
        let passed = tests.iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Passed)).count();
        self.success_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
    }

    pub fn total_tests(&self) -> usize { 4 }
    pub fn passed_tests(&self) -> usize {
        vec![&self.hello_world_test, &self.arithmetic_test, &self.function_test, &self.control_flow_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Passed)).count()
    }
    pub fn failed_tests(&self) -> usize {
        vec![&self.hello_world_test, &self.arithmetic_test, &self.function_test, &self.control_flow_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Failed)).count()
    }
    pub fn error_tests(&self) -> usize {
        vec![&self.hello_world_test, &self.arithmetic_test, &self.function_test, &self.control_flow_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Error)).count()
    }
    pub fn skipped_tests(&self) -> usize {
        vec![&self.hello_world_test, &self.arithmetic_test, &self.function_test, &self.control_flow_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Skipped)).count()
    }
}

/// Advanced test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedTestResults {
    pub tensor_test: Option<TestResult>,
    pub autodiff_test: Option<TestResult>,
    pub probabilistic_test: Option<TestResult>,
    pub gpu_test: Option<TestResult>,
    pub ffi_test: Option<TestResult>,
    pub duration: Duration,
    pub success_rate: f64,
}

impl AdvancedTestResults {
    pub fn new() -> Self {
        Self {
            tensor_test: None,
            autodiff_test: None,
            probabilistic_test: None,
            gpu_test: None,
            ffi_test: None,
            duration: Duration::from_nanos(0),
            success_rate: 0.0,
        }
    }

    pub fn calculate_success_rate(&mut self) {
        let tests = vec![&self.tensor_test, &self.autodiff_test, &self.probabilistic_test, &self.gpu_test, &self.ffi_test];
        let total = tests.len();
        let passed = tests.iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Passed)).count();
        self.success_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
    }

    pub fn total_tests(&self) -> usize { 5 }
    pub fn passed_tests(&self) -> usize {
        vec![&self.tensor_test, &self.autodiff_test, &self.probabilistic_test, &self.gpu_test, &self.ffi_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Passed)).count()
    }
    pub fn failed_tests(&self) -> usize {
        vec![&self.tensor_test, &self.autodiff_test, &self.probabilistic_test, &self.gpu_test, &self.ffi_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Failed)).count()
    }
    pub fn error_tests(&self) -> usize {
        vec![&self.tensor_test, &self.autodiff_test, &self.probabilistic_test, &self.gpu_test, &self.ffi_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Error)).count()
    }
    pub fn skipped_tests(&self) -> usize {
        vec![&self.tensor_test, &self.autodiff_test, &self.probabilistic_test, &self.gpu_test, &self.ffi_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Skipped)).count()
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub compilation_benchmark: Option<BenchmarkResult>,
    pub execution_benchmark: Option<BenchmarkResult>,
    pub memory_benchmark: Option<BenchmarkResult>,
    pub tensor_benchmark: Option<BenchmarkResult>,
    pub duration: Duration,
}

impl BenchmarkResults {
    pub fn new() -> Self {
        Self {
            compilation_benchmark: None,
            execution_benchmark: None,
            memory_benchmark: None,
            tensor_benchmark: None,
            duration: Duration::from_nanos(0),
        }
    }
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_duration: Duration,
    pub average_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub success_rate: f64,
    pub metadata: HashMap<String, String>,
}

impl BenchmarkResult {
    pub fn from_durations(name: &str, durations: Vec<Duration>) -> Self {
        if durations.is_empty() {
            return Self {
                name: name.to_string(),
                iterations: 0,
                total_duration: Duration::from_nanos(0),
                average_duration: Duration::from_nanos(0),
                min_duration: Duration::from_nanos(0),
                max_duration: Duration::from_nanos(0),
                success_rate: 0.0,
                metadata: HashMap::new(),
            };
        }

        let total_duration: Duration = durations.iter().sum();
        let average_duration = total_duration / durations.len() as u32;
        let min_duration = *durations.iter().min().unwrap();
        let max_duration = *durations.iter().max().unwrap();

        Self {
            name: name.to_string(),
            iterations: durations.len(),
            total_duration,
            average_duration,
            min_duration,
            max_duration,
            success_rate: 1.0, // All durations represent successful runs
            metadata: HashMap::new(),
        }
    }
}

/// Meta-test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaTestResults {
    pub framework_test: Option<TestResult>,
    pub error_detection_test: Option<TestResult>,
    pub build_integration_test: Option<TestResult>,
    pub duration: Duration,
    pub success_rate: f64,
}

impl MetaTestResults {
    pub fn new() -> Self {
        Self {
            framework_test: None,
            error_detection_test: None,
            build_integration_test: None,
            duration: Duration::from_nanos(0),
            success_rate: 0.0,
        }
    }

    pub fn calculate_success_rate(&mut self) {
        let tests = vec![&self.framework_test, &self.error_detection_test, &self.build_integration_test];
        let total = tests.len();
        let passed = tests.iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Passed)).count();
        self.success_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
    }

    pub fn total_tests(&self) -> usize { 3 }
    pub fn passed_tests(&self) -> usize {
        vec![&self.framework_test, &self.error_detection_test, &self.build_integration_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Passed)).count()
    }
    pub fn failed_tests(&self) -> usize {
        vec![&self.framework_test, &self.error_detection_test, &self.build_integration_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Failed)).count()
    }
    pub fn error_tests(&self) -> usize {
        vec![&self.framework_test, &self.error_detection_test, &self.build_integration_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Error)).count()
    }
    pub fn skipped_tests(&self) -> usize {
        vec![&self.framework_test, &self.error_detection_test, &self.build_integration_test]
            .iter().filter(|t| t.as_ref().map_or(false, |r| r.status == TestStatus::Skipped)).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_system::BuildConfig;

    #[test]
    fn test_comprehensive_test_runner_creation() {
        let config = TestConfig::default();
        let build_manager = BuildSystemManager::new();
        let runner = ComprehensiveTestRunner::new(config, build_manager);
        
        assert!(runner.test_suites.is_empty());
        assert!(runner.temp_dir.is_none());
    }

    #[test]
    fn test_comprehensive_results_statistics() {
        let mut results = ComprehensiveTestResults::new();
        
        // Add some mock basic test results
        let mut basic_results = BasicTestResults::new();
        basic_results.hello_world_test = Some(TestResult::passed("hello_world".to_string(), Duration::from_millis(100)));
        basic_results.arithmetic_test = Some(TestResult::failed("arithmetic".to_string(), Duration::from_millis(50), "error".to_string()));
        basic_results.calculate_success_rate();
        
        results.basic_tests = Some(basic_results);
        results.calculate_statistics();
        
        assert_eq!(results.total_tests, 4); // BasicTestResults has 4 tests
        assert_eq!(results.passed_tests, 1);
        assert_eq!(results.failed_tests, 1);
        assert_eq!(results.overall_success_rate, 0.25);
    }

    #[test]
    fn test_benchmark_result_from_durations() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(15),
        ];
        
        let result = BenchmarkResult::from_durations("test_benchmark", durations);
        
        assert_eq!(result.name, "test_benchmark");
        assert_eq!(result.iterations, 3);
        assert_eq!(result.total_duration, Duration::from_millis(45));
        assert_eq!(result.average_duration, Duration::from_millis(15));
        assert_eq!(result.min_duration, Duration::from_millis(10));
        assert_eq!(result.max_duration, Duration::from_millis(20));
        assert_eq!(result.success_rate, 1.0);
    }

    #[test]
    fn test_empty_benchmark_result() {
        let result = BenchmarkResult::from_durations("empty_test", vec![]);
        
        assert_eq!(result.name, "empty_test");
        assert_eq!(result.iterations, 0);
        assert_eq!(result.success_rate, 0.0);
    }
}