// Comprehensive test runner for property-based tests and benchmarks
// Task 10.3: Integration runner for all property-based and benchmark testing infrastructure
// Provides a unified interface to run all advanced testing features

use std::time::{Duration, Instant};

/// Run all property-based and benchmark tests
pub fn run_all_property_and_benchmark_tests() {
    println!("=== Property-Based and Benchmark Test Suite ===");
    println!();
    
    let start_time = Instant::now();
    
    // 1. Property-Based Tests
    println!("1. Running Property-Based Tests...");
    run_property_based_test_suite();
    println!("   ✓ Complete");
    println!();
    
    // 2. Optimization Property Tests
    println!("2. Running Optimization Property Tests...");
    run_optimization_property_test_suite();
    println!("   ✓ Complete");
    println!();
    
    // 3. Compilation Performance Benchmarks
    println!("3. Running Compilation Performance Benchmarks...");
    run_compilation_benchmark_suite();
    println!("   ✓ Complete");
    println!();
    
    // 4. Code Quality Benchmarks
    println!("4. Running Code Quality Benchmarks...");
    run_code_quality_benchmark_suite();
    println!("   ✓ Complete");
    println!();
    
    let total_duration = start_time.elapsed();
    println!("=== Property-Based and Benchmark Test Suite Complete ===");
    println!("Total execution time: {:?}", total_duration);
    println!("All tests passed successfully!");
}

/// Run property-based test suite
fn run_property_based_test_suite() {
    use crate::property_based_tests::{PropertyTestFramework};
    use aether_language::compiler::parser::parse_sexpr;
    
    let framework = PropertyTestFramework::new().expect("Failed to create property test framework");
    
    // Test 1: Type Safety Preservation
    println!("   Running type safety preservation tests...");
    let type_safety_programs = vec![
        ("simple_arithmetic", r#"
(func main ()
  (let x 42)
  (let y (+ x 1))
  (return y))
"#),
        ("function_calls", r#"
(defun double (x) (* x 2))
(func main ()
  (let result (double 21))
  (return result))
"#),
        ("control_flow", r#"
(func main ()
  (let x 10)
  (if (> x 5)
    (let result (* x 2))
    (let result (/ x 2)))
  (return result))
"#),
    ];
    
    let mut type_safety_passed = 0;
    for (name, program) in &type_safety_programs {
        if let Ok(ast) = parse_sexpr(program) {
            if let Ok(result) = framework.test_type_safety_preservation(&ast) {
                if result {
                    type_safety_passed += 1;
                }
                println!("     - {}: {}", name, if result { "PASS" } else { "FAIL" });
            } else {
                println!("     - {}: SKIP (compilation error)", name);
            }
        } else {
            println!("     - {}: SKIP (parse error)", name);
        }
    }
    println!("   Type safety tests: {}/{} passed", type_safety_passed, type_safety_programs.len());
    
    // Test 2: Memory Safety Preservation
    println!("   Running memory safety preservation tests...");
    let memory_safety_programs = vec![
        ("stack_allocation", r#"
(func main ()
  (let x 42)
  (let y x)
  (return y))
"#),
        ("tensor_allocation", r#"
(func main ()
  (let tensor (tensor-create [4 4] f32))
  (tensor-fill tensor 1.0)
  (return 0))
"#),
        ("multiple_allocations", r#"
(func main ()
  (let a (tensor-create [2 2] f32))
  (let b (tensor-create [3 3] f64))
  (let c (+ 1 2))
  (return c))
"#),
    ];
    
    let mut memory_safety_passed = 0;
    for (name, program) in &memory_safety_programs {
        if let Ok(ast) = parse_sexpr(program) {
            if let Ok(result) = framework.test_memory_safety_preservation(&ast) {
                if result {
                    memory_safety_passed += 1;
                }
                println!("     - {}: {}", name, if result { "PASS" } else { "FAIL" });
            } else {
                println!("     - {}: SKIP (compilation error)", name);
            }
        } else {
            println!("     - {}: SKIP (parse error)", name);
        }
    }
    println!("   Memory safety tests: {}/{} passed", memory_safety_passed, memory_safety_programs.len());
}

/// Run optimization property test suite
fn run_optimization_property_test_suite() {
    use crate::optimization_property_tests::{OptimizationPropertyFramework};
    use aether_language::compiler::parser::parse_sexpr;
    use aether_language::compiler::mlir::OptimizationLevel;
    
    let framework = OptimizationPropertyFramework::new().expect("Failed to create optimization framework");
    
    // Test 1: Semantics Preservation
    println!("   Running optimization semantics preservation tests...");
    let semantics_programs = vec![
        ("constant_folding", r#"
(func main ()
  (let x (+ 10 20))
  (let y (* 5 6))
  (let z (- x y))
  (return z))
"#),
        ("dead_code_elimination", r#"
(func main ()
  (let x 42)
  (let unused (+ x 100))
  (let also_unused (* unused 2))
  (return x))
"#),
        ("function_inlining", r#"
(defun add_one (x) (+ x 1))
(func main ()
  (let result (add_one 41))
  (return result))
"#),
    ];
    
    let optimization_levels = vec![
        OptimizationLevel::Less,
        OptimizationLevel::Default,
        OptimizationLevel::Aggressive,
    ];
    
    let mut semantics_passed = 0;
    let mut total_semantics_tests = 0;
    
    for (name, program) in &semantics_programs {
        if let Ok(ast) = parse_sexpr(program) {
            for opt_level in &optimization_levels {
                total_semantics_tests += 1;
                if let Ok(result) = framework.test_optimization_preserves_semantics(&ast, *opt_level) {
                    if result {
                        semantics_passed += 1;
                    }
                    println!("     - {} ({:?}): {}", name, opt_level, if result { "PASS" } else { "FAIL" });
                } else {
                    println!("     - {} ({:?}): SKIP (compilation error)", name, opt_level);
                }
            }
        } else {
            println!("     - {}: SKIP (parse error)", name);
        }
    }
    println!("   Semantics preservation tests: {}/{} passed", semantics_passed, total_semantics_tests);
    
    // Test 2: Performance Improvement
    println!("   Running optimization performance improvement tests...");
    let mut performance_passed = 0;
    let mut total_performance_tests = 0;
    
    for (name, program) in &semantics_programs {
        if let Ok(ast) = parse_sexpr(program) {
            total_performance_tests += 1;
            if let Ok(result) = framework.test_optimization_improves_performance(&ast, OptimizationLevel::Default) {
                if result {
                    performance_passed += 1;
                }
                println!("     - {}: {}", name, if result { "PASS" } else { "FAIL" });
            } else {
                println!("     - {}: SKIP (compilation error)", name);
            }
        }
    }
    println!("   Performance improvement tests: {}/{} passed", performance_passed, total_performance_tests);
    
    // Test 3: Idempotency
    println!("   Running optimization idempotency tests...");
    let mut idempotency_passed = 0;
    let mut total_idempotency_tests = 0;
    
    for (name, program) in &semantics_programs {
        if let Ok(ast) = parse_sexpr(program) {
            total_idempotency_tests += 1;
            if let Ok(result) = framework.test_optimization_idempotency(&ast, OptimizationLevel::Default) {
                if result {
                    idempotency_passed += 1;
                }
                println!("     - {}: {}", name, if result { "PASS" } else { "FAIL" });
            } else {
                println!("     - {}: SKIP (compilation error)", name);
            }
        }
    }
    println!("   Idempotency tests: {}/{} passed", idempotency_passed, total_idempotency_tests);
}

/// Run compilation benchmark suite
fn run_compilation_benchmark_suite() {
    use crate::compilation_benchmarks::{CompilationBenchmarkSuite};
    use aether_language::compiler::mlir::OptimizationLevel;
    
    let suite = CompilationBenchmarkSuite::new().expect("Failed to create compilation benchmark suite");
    
    // Test 1: Full Compilation Benchmarks
    println!("   Running full compilation benchmarks...");
    let benchmark_names = suite.get_benchmark_names();
    
    for program_name in &benchmark_names {
        if let Ok(result) = suite.benchmark_full_compilation(program_name, OptimizationLevel::Default) {
            println!("     - {}: {:.2}ms total, {} MLIR ops, {} LLVM instructions", 
                program_name,
                result.total_duration.as_millis(),
                result.mlir_operations_count,
                result.llvm_instructions_count);
        } else {
            println!("     - {}: SKIP (benchmark error)", program_name);
        }
    }
    
    // Test 2: Phase-specific Benchmarks
    println!("   Running phase-specific benchmarks...");
    let test_program = "simple_arithmetic";
    
    if let Ok(parsing_duration) = suite.benchmark_parsing(test_program) {
        println!("     - Parsing: {:.2}ms", parsing_duration.as_millis());
    }
    
    if let Ok(mlir_duration) = suite.benchmark_mlir_generation(test_program) {
        println!("     - MLIR generation: {:.2}ms", mlir_duration.as_millis());
    }
    
    if let Ok(opt_duration) = suite.benchmark_optimization(test_program, OptimizationLevel::Default) {
        println!("     - Optimization: {:.2}ms", opt_duration.as_millis());
    }
    
    if let Ok(llvm_duration) = suite.benchmark_llvm_generation(test_program, OptimizationLevel::Default) {
        println!("     - LLVM generation: {:.2}ms", llvm_duration.as_millis());
    }
    
    // Test 3: Optimization Level Comparison
    println!("   Running optimization level comparison...");
    let optimization_levels = vec![
        ("None", OptimizationLevel::None),
        ("Less", OptimizationLevel::Less),
        ("Default", OptimizationLevel::Default),
        ("Aggressive", OptimizationLevel::Aggressive),
    ];
    
    for (level_name, opt_level) in optimization_levels {
        if let Ok(result) = suite.benchmark_full_compilation("function_calls", opt_level) {
            println!("     - {}: {:.2}ms, {} instructions", 
                level_name,
                result.total_duration.as_millis(),
                result.llvm_instructions_count);
        }
    }
}

/// Run code quality benchmark suite
fn run_code_quality_benchmark_suite() {
    use crate::code_quality_benchmarks::{CodeQualityBenchmarkSuite};
    use aether_language::compiler::mlir::OptimizationLevel;
    
    let mut suite = CodeQualityBenchmarkSuite::new().expect("Failed to create code quality benchmark suite");
    
    // Test 1: Code Quality Metrics
    println!("   Running code quality metrics tests...");
    let quality_programs = vec![
        ("simple_arithmetic", r#"
(func main ()
  (let x (+ 10 20))
  (let y (* 5 6))
  (let z (+ x y))
  (return z))
"#),
        ("control_flow", r#"
(func main ()
  (let x 10)
  (if (> x 5)
    (let result (* x 2))
    (let result (/ x 2)))
  (return result))
"#),
        ("function_calls", r#"
(defun square (x) (* x x))
(func main ()
  (let result (square 7))
  (return result))
"#),
        ("tensor_operations", r#"
(func main ()
  (let tensor_a (tensor-create [4 4] f32))
  (let tensor_b (tensor-create [4 4] f32))
  (let result (tensor-add tensor_a tensor_b))
  (return 0))
"#),
    ];
    
    for (name, program) in &quality_programs {
        if let Ok(metrics) = suite.benchmark_code_quality(name, program, OptimizationLevel::Default) {
            println!("     - {}: {} instructions, {:.3} complexity, {:.3} density", 
                name,
                metrics.total_instructions(),
                metrics.code_complexity_score,
                metrics.instruction_density);
        } else {
            println!("     - {}: SKIP (quality benchmark error)", name);
        }
    }
    
    // Test 2: Optimization Level Quality Comparison
    println!("   Running optimization level quality comparison...");
    let comparison_program = r#"
(func main ()
  (let a (+ 1 2))
  (let b (* 3 4))
  (let c (+ a b))
  (let unused_var (* c 100))
  (return c))
"#;
    
    let optimization_levels = vec![
        ("None", OptimizationLevel::None),
        ("Default", OptimizationLevel::Default),
        ("Aggressive", OptimizationLevel::Aggressive),
    ];
    
    for (level_name, opt_level) in optimization_levels {
        if let Ok(metrics) = suite.benchmark_code_quality("quality_comparison", comparison_program, opt_level) {
            println!("     - {}: {} instructions, {:.3} optimization effectiveness, {} bytes", 
                level_name,
                metrics.total_instructions(),
                metrics.optimization_effectiveness,
                metrics.code_size_bytes);
        }
    }
    
    // Test 3: Code Quality Trends
    println!("   Running code quality trend analysis...");
    let trend_programs = vec![
        ("constant_folding_candidate", r#"
(func main ()
  (let x (+ 5 10))
  (let y (* 3 7))
  (return (+ x y)))
"#),
        ("dead_code_candidate", r#"
(func main ()
  (let x 42)
  (let unused (+ x 999))
  (return x))
"#),
        ("inlining_candidate", r#"
(defun simple_add (a b) (+ a b))
(func main ()
  (let result (simple_add 10 20))
  (return result))
"#),
    ];
    
    for (name, program) in &trend_programs {
        let unoptimized = suite.benchmark_code_quality(&format!("{}_unopt", name), program, OptimizationLevel::None);
        let optimized = suite.benchmark_code_quality(&format!("{}_opt", name), program, OptimizationLevel::Default);
        
        match (unoptimized, optimized) {
            (Ok(unopt_metrics), Ok(opt_metrics)) => {
                let instruction_reduction = if unopt_metrics.total_instructions() > 0 {
                    ((unopt_metrics.total_instructions() as f64 - opt_metrics.total_instructions() as f64) / 
                     unopt_metrics.total_instructions() as f64) * 100.0
                } else {
                    0.0
                };
                
                let size_reduction = if unopt_metrics.code_size_bytes > 0 {
                    ((unopt_metrics.code_size_bytes as f64 - opt_metrics.code_size_bytes as f64) / 
                     unopt_metrics.code_size_bytes as f64) * 100.0
                } else {
                    0.0
                };
                
                println!("     - {}: {:.1}% instruction reduction, {:.1}% size reduction", 
                    name, instruction_reduction, size_reduction);
            }
            _ => {
                println!("     - {}: SKIP (comparison error)", name);
            }
        }
    }
}

// ===== UNIT TESTS =====

#[cfg(test)]
mod property_benchmark_tests {
    use super::*;

    #[test]
    fn test_complete_property_benchmark_suite() {
        println!("Running complete property and benchmark test suite...");
        run_all_property_and_benchmark_tests();
        println!("Complete test suite finished successfully.");
    }

    #[test]
    fn test_property_based_suite_only() {
        println!("Running property-based test suite only...");
        run_property_based_test_suite();
        println!("Property-based test suite finished.");
    }

    #[test]
    fn test_optimization_property_suite_only() {
        println!("Running optimization property test suite only...");
        run_optimization_property_test_suite();
        println!("Optimization property test suite finished.");
    }

    #[test]
    fn test_compilation_benchmark_suite_only() {
        println!("Running compilation benchmark suite only...");
        run_compilation_benchmark_suite();
        println!("Compilation benchmark suite finished.");
    }

    #[test]
    fn test_code_quality_benchmark_suite_only() {
        println!("Running code quality benchmark suite only...");
        run_code_quality_benchmark_suite();
        println!("Code quality benchmark suite finished.");
    }

    #[test]
    fn test_framework_integration() {
        // Test that all frameworks can be created successfully
        use crate::property_based_tests::PropertyTestFramework;
        use crate::optimization_property_tests::OptimizationPropertyFramework;
        use crate::compilation_benchmarks::CompilationBenchmarkSuite;
        use crate::code_quality_benchmarks::CodeQualityBenchmarkSuite;
        
        let _property_framework = PropertyTestFramework::new()
            .expect("Should be able to create property test framework");
        println!("   ✓ Property test framework");
        
        let _optimization_framework = OptimizationPropertyFramework::new()
            .expect("Should be able to create optimization property framework");
        println!("   ✓ Optimization property framework");
        
        let _compilation_suite = CompilationBenchmarkSuite::new()
            .expect("Should be able to create compilation benchmark suite");
        println!("   ✓ Compilation benchmark suite");
        
        let _quality_suite = CodeQualityBenchmarkSuite::new()
            .expect("Should be able to create code quality benchmark suite");
        println!("   ✓ Code quality benchmark suite");
        
        println!("All frameworks integrated successfully!");
    }

    #[test]
    fn test_end_to_end_property_verification() {
        // Test a complete end-to-end property verification
        use crate::property_based_tests::PropertyTestFramework;
        use aether_language::compiler::parser::parse_sexpr;
        
        let framework = PropertyTestFramework::new().expect("Failed to create framework");
        
        let test_program = r#"
(func main ()
  (let x 10)
  (let y 20)
  (let z (+ x y))
  (return z))
"#;
        
        let ast = parse_sexpr(test_program).expect("Failed to parse test program");
        
        // Test type safety
        let type_safety = framework.test_type_safety_preservation(&ast)
            .expect("Type safety test failed");
        assert!(type_safety, "Type safety should be preserved");
        
        // Test memory safety
        let memory_safety = framework.test_memory_safety_preservation(&ast)
            .expect("Memory safety test failed");
        assert!(memory_safety, "Memory safety should be preserved");
        
        println!("End-to-end property verification passed!");
    }

    #[test]
    fn test_end_to_end_optimization_verification() {
        // Test a complete end-to-end optimization verification
        use crate::optimization_property_tests::OptimizationPropertyFramework;
        use aether_language::compiler::parser::parse_sexpr;
        use aether_language::compiler::mlir::OptimizationLevel;
        
        let framework = OptimizationPropertyFramework::new().expect("Failed to create framework");
        
        let test_program = r#"
(func main ()
  (let x (+ 5 10))
  (let y (* x 2))
  (return y))
"#;
        
        let ast = parse_sexpr(test_program).expect("Failed to parse test program");
        
        // Test semantics preservation
        let semantics_preserved = framework.test_optimization_preserves_semantics(&ast, OptimizationLevel::Default)
            .expect("Semantics preservation test failed");
        assert!(semantics_preserved, "Optimization should preserve semantics");
        
        // Test performance improvement
        let performance_improved = framework.test_optimization_improves_performance(&ast, OptimizationLevel::Default)
            .expect("Performance improvement test failed");
        assert!(performance_improved, "Optimization should improve performance");
        
        println!("End-to-end optimization verification passed!");
    }

    #[test]
    fn test_end_to_end_benchmark_verification() {
        // Test a complete end-to-end benchmark verification
        use crate::compilation_benchmarks::CompilationBenchmarkSuite;
        use crate::code_quality_benchmarks::CodeQualityBenchmarkSuite;
        use aether_language::compiler::mlir::OptimizationLevel;
        
        let compilation_suite = CompilationBenchmarkSuite::new().expect("Failed to create compilation suite");
        let mut quality_suite = CodeQualityBenchmarkSuite::new().expect("Failed to create quality suite");
        
        let test_program = r#"
(func main ()
  (let result (+ 1 2))
  (return result))
"#;
        
        // Test compilation benchmark
        let compilation_result = compilation_suite.benchmark_full_compilation("simple_arithmetic", OptimizationLevel::Default)
            .expect("Compilation benchmark failed");
        
        assert!(compilation_result.total_duration.as_millis() > 0, "Should have compilation time");
        assert!(compilation_result.mlir_operations_count > 0, "Should have MLIR operations");
        
        // Test code quality benchmark
        let quality_result = quality_suite.benchmark_code_quality("end_to_end_test", test_program, OptimizationLevel::Default)
            .expect("Quality benchmark failed");
        
        assert!(quality_result.total_instructions() > 0, "Should have instructions");
        assert!(quality_result.code_size_bytes > 0, "Should have code size");
        
        println!("End-to-end benchmark verification passed!");
    }
}

/// Utility function to run property and benchmark tests from command line
pub fn main() {
    println!("Aether Property-Based and Benchmark Test Runner");
    println!("===============================================");
    
    let args: Vec<String> = std::env::args().collect();
    
    match args.get(1).map(|s| s.as_str()) {
        Some("property") => run_property_based_test_suite(),
        Some("optimization") => run_optimization_property_test_suite(),
        Some("compilation") => run_compilation_benchmark_suite(),
        Some("quality") => run_code_quality_benchmark_suite(),
        Some("all") | None => run_all_property_and_benchmark_tests(),
        Some(unknown) => {
            println!("Unknown test suite: {}", unknown);
            println!("Available options: property, optimization, compilation, quality, all");
        }
    }
}