// Compilation performance benchmarking infrastructure
// Task 10.3: Implement compilation performance benchmarking
// Provides comprehensive benchmarking for compilation speed and resource usage

use aether_language::compiler::mlir::{
    MLIRContext, MLIRModule, AetherMLIRFrontend, AetherOptimizer,
    LLVMCodeGenerator, TargetConfig, OptimizationLevel
};
use aether_language::compiler::ast::{AST, ASTNode, ASTNodeRef};
use aether_language::compiler::parser::{parse_sexpr};
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tempfile::TempDir;

/// Compilation benchmark suite
pub struct CompilationBenchmarkSuite {
    temp_dir: TempDir,
    benchmark_programs: HashMap<String, BenchmarkProgram>,
}

impl CompilationBenchmarkSuite {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let mut benchmark_programs = HashMap::new();
        
        // Add standard benchmark programs
        benchmark_programs.insert("simple_arithmetic".to_string(), BenchmarkProgram {
            name: "simple_arithmetic".to_string(),
            source_code: r#"
(func main ()
  (let x 10)
  (let y 20)
  (let z (+ x y))
  (return z))
"#.to_string(),
            expected_complexity: ProgramComplexity::Simple,
            source_lines: 6,
        });
        
        benchmark_programs.insert("control_flow".to_string(), BenchmarkProgram {
            name: "control_flow".to_string(),
            source_code: r#"
(func main ()
  (let x 10)
  (if (> x 5)
    (let y (* x 2))
    (let y (/ x 2)))
  (let i 0)
  (while (< i x)
    (set i (+ i 1)))
  (return i))
"#.to_string(),
            expected_complexity: ProgramComplexity::ControlFlow,
            source_lines: 10,
        });
        
        benchmark_programs.insert("function_calls".to_string(), BenchmarkProgram {
            name: "function_calls".to_string(),
            source_code: r#"
(defun factorial (n)
  (if (<= n 1)
    1
    (* n (factorial (- n 1)))))

(defun fibonacci (n)
  (if (<= n 1)
    n
    (+ (fibonacci (- n 1)) (fibonacci (- n 2)))))

(func main ()
  (let fact_result (factorial 5))
  (let fib_result (fibonacci 8))
  (return (+ fact_result fib_result)))
"#.to_string(),
            expected_complexity: ProgramComplexity::Recursive,
            source_lines: 14,
        });
        
        benchmark_programs.insert("tensor_operations".to_string(), BenchmarkProgram {
            name: "tensor_operations".to_string(),
            source_code: r#"
(func main ()
  (let tensor_a (tensor-create [10 10] f32))
  (let tensor_b (tensor-create [10 10] f32))
  (tensor-fill tensor_a 1.0)
  (tensor-fill tensor_b 2.0)
  (let result (tensor-add tensor_a tensor_b))
  (let scaled (tensor-scale result 0.5))
  (let transposed (tensor-transpose scaled))
  (return 0))
"#.to_string(),
            expected_complexity: ProgramComplexity::TensorOps,
            source_lines: 9,
        });
        
        Ok(CompilationBenchmarkSuite {
            temp_dir,
            benchmark_programs,
        })
    }

    /// Add a custom benchmark program
    pub fn add_benchmark_program(&mut self, program: BenchmarkProgram) {
        self.benchmark_programs.insert(program.name.clone(), program);
    }

    /// Benchmark full compilation pipeline
    pub fn benchmark_full_compilation(&self, program_name: &str, opt_level: OptimizationLevel) -> Result<CompilationBenchmarkResult, Box<dyn std::error::Error>> {
        let program = self.benchmark_programs.get(program_name)
            .ok_or_else(|| format!("Benchmark program '{}' not found", program_name))?;
        
        let mut result = CompilationBenchmarkResult::new(program_name.to_string(), opt_level);
        
        // Parse AST
        let parse_start = Instant::now();
        let ast = parse_sexpr(&program.source_code)?;
        result.parsing_duration = parse_start.elapsed();
        
        // Generate MLIR
        let mlir_start = Instant::now();
        let context = MLIRContext::new()?;
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module(&format!("benchmark_{}", program_name))?;
        frontend.convert_ast_to_module(&ast, &mut module)?;
        module.verify()?;
        result.mlir_generation_duration = mlir_start.elapsed();
        result.mlir_operations_count = module.operations().len();
        
        // Apply optimizations
        let opt_start = Instant::now();
        if opt_level != OptimizationLevel::None {
            let optimizer = AetherOptimizer::new(&context);
            optimizer.optimize_with_level(&mut module, opt_level)?;
            module.verify()?;
        }
        result.optimization_duration = opt_start.elapsed();
        
        // Generate LLVM IR
        let llvm_start = Instant::now();
        let target_config = TargetConfig {
            triple: "x86_64-unknown-linux-gnu".to_string(),
            cpu: "generic".to_string(),
            features: "".to_string(),
            optimization_level: opt_level,
            relocation_model: aether_language::compiler::mlir::RelocModel::Default,
            code_model: aether_language::compiler::mlir::CodeModel::Default,
        };
        let mut codegen = LLVMCodeGenerator::new(target_config)?;
        codegen.generate_from_mlir(&module)?;
        let llvm_ir = codegen.get_llvm_ir_string()?;
        result.llvm_generation_duration = llvm_start.elapsed();
        result.llvm_instructions_count = self.count_llvm_instructions(&llvm_ir);
        result.generated_code_size = llvm_ir.len();
        
        result.total_duration = result.parsing_duration + result.mlir_generation_duration + 
                               result.optimization_duration + result.llvm_generation_duration;
        
        // Estimate memory usage
        result.peak_memory_usage_mb = self.estimate_memory_usage(&result);
        
        Ok(result)
    }

    /// Benchmark parsing phase only
    pub fn benchmark_parsing(&self, program_name: &str) -> Result<Duration, Box<dyn std::error::Error>> {
        let program = self.benchmark_programs.get(program_name)
            .ok_or_else(|| format!("Benchmark program '{}' not found", program_name))?;
        
        let start = Instant::now();
        let _ast = parse_sexpr(&program.source_code)?;
        Ok(start.elapsed())
    }

    /// Benchmark MLIR generation phase only
    pub fn benchmark_mlir_generation(&self, program_name: &str) -> Result<Duration, Box<dyn std::error::Error>> {
        let program = self.benchmark_programs.get(program_name)
            .ok_or_else(|| format!("Benchmark program '{}' not found", program_name))?;
        
        let ast = parse_sexpr(&program.source_code)?;
        
        let start = Instant::now();
        let context = MLIRContext::new()?;
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module(&format!("benchmark_{}", program_name))?;
        frontend.convert_ast_to_module(&ast, &mut module)?;
        module.verify()?;
        Ok(start.elapsed())
    }

    /// Benchmark optimization phase only
    pub fn benchmark_optimization(&self, program_name: &str, opt_level: OptimizationLevel) -> Result<Duration, Box<dyn std::error::Error>> {
        let program = self.benchmark_programs.get(program_name)
            .ok_or_else(|| format!("Benchmark program '{}' not found", program_name))?;
        
        let ast = parse_sexpr(&program.source_code)?;
        let context = MLIRContext::new()?;
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module(&format!("benchmark_{}", program_name))?;
        frontend.convert_ast_to_module(&ast, &mut module)?;
        module.verify()?;
        
        let start = Instant::now();
        if opt_level != OptimizationLevel::None {
            let optimizer = AetherOptimizer::new(&context);
            optimizer.optimize_with_level(&mut module, opt_level)?;
            module.verify()?;
        }
        Ok(start.elapsed())
    }

    /// Benchmark LLVM code generation phase only
    pub fn benchmark_llvm_generation(&self, program_name: &str, opt_level: OptimizationLevel) -> Result<Duration, Box<dyn std::error::Error>> {
        let program = self.benchmark_programs.get(program_name)
            .ok_or_else(|| format!("Benchmark program '{}' not found", program_name))?;
        
        let ast = parse_sexpr(&program.source_code)?;
        let context = MLIRContext::new()?;
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module(&format!("benchmark_{}", program_name))?;
        frontend.convert_ast_to_module(&ast, &mut module)?;
        module.verify()?;
        
        if opt_level != OptimizationLevel::None {
            let optimizer = AetherOptimizer::new(&context);
            optimizer.optimize_with_level(&mut module, opt_level)?;
            module.verify()?;
        }
        
        let start = Instant::now();
        let target_config = TargetConfig {
            triple: "x86_64-unknown-linux-gnu".to_string(),
            cpu: "generic".to_string(),
            features: "".to_string(),
            optimization_level: opt_level,
            relocation_model: aether_language::compiler::mlir::RelocModel::Default,
            code_model: aether_language::compiler::mlir::CodeModel::Default,
        };
        let mut codegen = LLVMCodeGenerator::new(target_config)?;
        codegen.generate_from_mlir(&module)?;
        let _llvm_ir = codegen.get_llvm_ir_string()?;
        Ok(start.elapsed())
    }

    /// Generate scalability test programs
    pub fn generate_scalability_programs(&mut self, base_size: usize, scale_factors: &[usize]) {
        for &scale_factor in scale_factors {
            let program_name = format!("scalability_{}x", scale_factor);
            let program = self.generate_scaled_program(base_size * scale_factor);
            self.add_benchmark_program(program);
        }
    }

    /// Generate a program with specified number of operations
    fn generate_scaled_program(&self, operation_count: usize) -> BenchmarkProgram {
        let mut source_code = String::from("(func main ()\n");
        
        for i in 0..operation_count {
            let op = match i % 4 {
                0 => "+",
                1 => "-",
                2 => "*",
                _ => "/",
            };
            source_code.push_str(&format!("  (let var_{} ({} {} {}))\n", i, op, i % 100, (i + 1) % 100));
        }
        
        source_code.push_str("  (return 0))\n");
        
        BenchmarkProgram {
            name: format!("scalability_{}ops", operation_count),
            source_code,
            expected_complexity: ProgramComplexity::Arithmetic,
            source_lines: operation_count + 2,
        }
    }

    /// Count LLVM instructions
    fn count_llvm_instructions(&self, llvm_ir: &str) -> usize {
        llvm_ir.lines()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && 
                !trimmed.starts_with(';') && 
                !trimmed.starts_with("define") &&
                !trimmed.starts_with("declare") &&
                !trimmed.ends_with(':') &&
                !trimmed.starts_with('}') &&
                !trimmed.starts_with("target")
            })
            .count()
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self, result: &CompilationBenchmarkResult) -> usize {
        // Simple heuristic based on operations and code size
        let base_memory = 10; // MB
        let operations_factor = result.mlir_operations_count / 100;
        let instructions_factor = result.llvm_instructions_count / 1000;
        let code_size_factor = result.generated_code_size / 10000;
        
        base_memory + operations_factor + instructions_factor + code_size_factor
    }

    /// Get all benchmark program names
    pub fn get_benchmark_names(&self) -> Vec<String> {
        self.benchmark_programs.keys().cloned().collect()
    }
}

/// Benchmark program definition
#[derive(Debug, Clone)]
pub struct BenchmarkProgram {
    pub name: String,
    pub source_code: String,
    pub expected_complexity: ProgramComplexity,
    pub source_lines: usize,
}

/// Program complexity levels
#[derive(Debug, Clone, Copy)]
pub enum ProgramComplexity {
    Simple,
    Arithmetic,
    ControlFlow,
    TensorOps,
    Recursive,
}

/// Compilation benchmark result
#[derive(Debug, Clone)]
pub struct CompilationBenchmarkResult {
    pub program_name: String,
    pub optimization_level: OptimizationLevel,
    pub parsing_duration: Duration,
    pub mlir_generation_duration: Duration,
    pub optimization_duration: Duration,
    pub llvm_generation_duration: Duration,
    pub total_duration: Duration,
    pub mlir_operations_count: usize,
    pub llvm_instructions_count: usize,
    pub generated_code_size: usize,
    pub peak_memory_usage_mb: usize,
}

impl CompilationBenchmarkResult {
    fn new(program_name: String, optimization_level: OptimizationLevel) -> Self {
        CompilationBenchmarkResult {
            program_name,
            optimization_level,
            parsing_duration: Duration::from_millis(0),
            mlir_generation_duration: Duration::from_millis(0),
            optimization_duration: Duration::from_millis(0),
            llvm_generation_duration: Duration::from_millis(0),
            total_duration: Duration::from_millis(0),
            mlir_operations_count: 0,
            llvm_instructions_count: 0,
            generated_code_size: 0,
            peak_memory_usage_mb: 0,
        }
    }

    /// Calculate compilation throughput (lines per second)
    pub fn calculate_throughput(&self, source_lines: usize) -> f64 {
        if self.total_duration.as_secs_f64() > 0.0 {
            source_lines as f64 / self.total_duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Calculate compilation efficiency (operations per millisecond)
    pub fn calculate_efficiency(&self) -> f64 {
        let total_ms = self.total_duration.as_millis() as f64;
        if total_ms > 0.0 {
            (self.mlir_operations_count + self.llvm_instructions_count) as f64 / total_ms
        } else {
            0.0
        }
    }
}

// ===== CRITERION BENCHMARKS =====

/// Benchmark full compilation pipeline for different programs
fn benchmark_full_compilation(c: &mut Criterion) {
    let suite = CompilationBenchmarkSuite::new().expect("Failed to create benchmark suite");
    
    let mut group = c.benchmark_group("full_compilation");
    
    for program_name in suite.get_benchmark_names() {
        let program = suite.benchmark_programs.get(&program_name).unwrap();
        
        group.throughput(Throughput::Elements(program.source_lines as u64));
        group.bench_with_input(
            BenchmarkId::new("unoptimized", &program_name),
            &program_name,
            |b, name| {
                b.iter(|| {
                    suite.benchmark_full_compilation(name, OptimizationLevel::None)
                        .expect("Benchmark failed")
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("optimized", &program_name),
            &program_name,
            |b, name| {
                b.iter(|| {
                    suite.benchmark_full_compilation(name, OptimizationLevel::Default)
                        .expect("Benchmark failed")
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark individual compilation phases
fn benchmark_compilation_phases(c: &mut Criterion) {
    let suite = CompilationBenchmarkSuite::new().expect("Failed to create benchmark suite");
    
    let program_name = "control_flow";
    let program = suite.benchmark_programs.get(program_name).unwrap();
    
    let mut group = c.benchmark_group("compilation_phases");
    group.throughput(Throughput::Elements(program.source_lines as u64));
    
    group.bench_function("parsing", |b| {
        b.iter(|| suite.benchmark_parsing(program_name).expect("Parsing benchmark failed"))
    });
    
    group.bench_function("mlir_generation", |b| {
        b.iter(|| suite.benchmark_mlir_generation(program_name).expect("MLIR benchmark failed"))
    });
    
    group.bench_function("optimization", |b| {
        b.iter(|| suite.benchmark_optimization(program_name, OptimizationLevel::Default)
            .expect("Optimization benchmark failed"))
    });
    
    group.bench_function("llvm_generation", |b| {
        b.iter(|| suite.benchmark_llvm_generation(program_name, OptimizationLevel::Default)
            .expect("LLVM benchmark failed"))
    });
    
    group.finish();
}

/// Benchmark optimization levels
fn benchmark_optimization_levels(c: &mut Criterion) {
    let suite = CompilationBenchmarkSuite::new().expect("Failed to create benchmark suite");
    
    let program_name = "function_calls";
    let program = suite.benchmark_programs.get(program_name).unwrap();
    
    let mut group = c.benchmark_group("optimization_levels");
    group.throughput(Throughput::Elements(program.source_lines as u64));
    
    let optimization_levels = vec![
        ("none", OptimizationLevel::None),
        ("less", OptimizationLevel::Less),
        ("default", OptimizationLevel::Default),
        ("aggressive", OptimizationLevel::Aggressive),
    ];
    
    for (level_name, opt_level) in optimization_levels {
        group.bench_with_input(
            BenchmarkId::new("optimization", level_name),
            &opt_level,
            |b, &level| {
                b.iter(|| {
                    suite.benchmark_full_compilation(program_name, level)
                        .expect("Optimization level benchmark failed")
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark compilation scalability
fn benchmark_scalability(c: &mut Criterion) {
    let mut suite = CompilationBenchmarkSuite::new().expect("Failed to create benchmark suite");
    
    // Generate scalability test programs
    let scale_factors = vec![1, 2, 4, 8, 16];
    suite.generate_scalability_programs(10, &scale_factors);
    
    let mut group = c.benchmark_group("scalability");
    
    for &scale_factor in &scale_factors {
        let program_name = format!("scalability_{}x", scale_factor);
        let program = suite.benchmark_programs.get(&program_name).unwrap();
        
        group.throughput(Throughput::Elements(program.source_lines as u64));
        group.bench_with_input(
            BenchmarkId::new("scale", scale_factor),
            &program_name,
            |b, name| {
                b.iter(|| {
                    suite.benchmark_full_compilation(name, OptimizationLevel::Default)
                        .expect("Scalability benchmark failed")
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark tensor operations compilation
fn benchmark_tensor_compilation(c: &mut Criterion) {
    let suite = CompilationBenchmarkSuite::new().expect("Failed to create benchmark suite");
    
    let program_name = "tensor_operations";
    let program = suite.benchmark_programs.get(program_name).unwrap();
    
    let mut group = c.benchmark_group("tensor_compilation");
    group.throughput(Throughput::Elements(program.source_lines as u64));
    
    group.bench_function("tensor_ops_unoptimized", |b| {
        b.iter(|| {
            suite.benchmark_full_compilation(program_name, OptimizationLevel::None)
                .expect("Tensor benchmark failed")
        })
    });
    
    group.bench_function("tensor_ops_optimized", |b| {
        b.iter(|| {
            suite.benchmark_full_compilation(program_name, OptimizationLevel::Aggressive)
                .expect("Tensor benchmark failed")
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_full_compilation,
    benchmark_compilation_phases,
    benchmark_optimization_levels,
    benchmark_scalability,
    benchmark_tensor_compilation
);
criterion_main!(benches);

// ===== UNIT TESTS FOR BENCHMARK INFRASTRUCTURE =====

#[cfg(test)]
mod benchmark_tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = CompilationBenchmarkSuite::new().expect("Failed to create suite");
        let benchmark_names = suite.get_benchmark_names();
        
        assert!(!benchmark_names.is_empty(), "Should have benchmark programs");
        assert!(benchmark_names.contains(&"simple_arithmetic".to_string()));
        assert!(benchmark_names.contains(&"control_flow".to_string()));
        assert!(benchmark_names.contains(&"function_calls".to_string()));
        assert!(benchmark_names.contains(&"tensor_operations".to_string()));
    }

    #[test]
    fn test_full_compilation_benchmark() {
        let suite = CompilationBenchmarkSuite::new().expect("Failed to create suite");
        
        let result = suite.benchmark_full_compilation("simple_arithmetic", OptimizationLevel::Default)
            .expect("Benchmark failed");
        
        assert_eq!(result.program_name, "simple_arithmetic");
        assert!(result.total_duration > Duration::from_millis(0));
        assert!(result.parsing_duration > Duration::from_millis(0));
        assert!(result.mlir_generation_duration > Duration::from_millis(0));
        assert!(result.mlir_operations_count > 0);
        assert!(result.generated_code_size > 0);
        
        println!("Benchmark result: {:?}", result);
    }

    #[test]
    fn test_individual_phase_benchmarks() {
        let suite = CompilationBenchmarkSuite::new().expect("Failed to create suite");
        
        let parsing_duration = suite.benchmark_parsing("simple_arithmetic")
            .expect("Parsing benchmark failed");
        assert!(parsing_duration > Duration::from_millis(0));
        
        let mlir_duration = suite.benchmark_mlir_generation("simple_arithmetic")
            .expect("MLIR benchmark failed");
        assert!(mlir_duration > Duration::from_millis(0));
        
        let opt_duration = suite.benchmark_optimization("simple_arithmetic", OptimizationLevel::Default)
            .expect("Optimization benchmark failed");
        assert!(opt_duration >= Duration::from_millis(0)); // Can be 0 for simple programs
        
        let llvm_duration = suite.benchmark_llvm_generation("simple_arithmetic", OptimizationLevel::Default)
            .expect("LLVM benchmark failed");
        assert!(llvm_duration > Duration::from_millis(0));
        
        println!("Phase durations - Parsing: {:?}, MLIR: {:?}, Opt: {:?}, LLVM: {:?}",
            parsing_duration, mlir_duration, opt_duration, llvm_duration);
    }

    #[test]
    fn test_optimization_level_comparison() {
        let suite = CompilationBenchmarkSuite::new().expect("Failed to create suite");
        
        let unoptimized = suite.benchmark_full_compilation("function_calls", OptimizationLevel::None)
            .expect("Unoptimized benchmark failed");
        
        let optimized = suite.benchmark_full_compilation("function_calls", OptimizationLevel::Default)
            .expect("Optimized benchmark failed");
        
        // Optimized version should have same or fewer LLVM instructions
        assert!(optimized.llvm_instructions_count <= unoptimized.llvm_instructions_count,
            "Optimized version should have fewer or equal instructions");
        
        println!("Unoptimized: {} instructions, Optimized: {} instructions",
            unoptimized.llvm_instructions_count, optimized.llvm_instructions_count);
    }

    #[test]
    fn test_scalability_program_generation() {
        let mut suite = CompilationBenchmarkSuite::new().expect("Failed to create suite");
        
        let initial_count = suite.get_benchmark_names().len();
        
        suite.generate_scalability_programs(5, &[2, 4]);
        
        let final_count = suite.get_benchmark_names().len();
        assert_eq!(final_count, initial_count + 2, "Should add 2 scalability programs");
        
        let names = suite.get_benchmark_names();
        assert!(names.contains(&"scalability_2x".to_string()));
        assert!(names.contains(&"scalability_4x".to_string()));
    }

    #[test]
    fn test_benchmark_result_calculations() {
        let suite = CompilationBenchmarkSuite::new().expect("Failed to create suite");
        
        let result = suite.benchmark_full_compilation("simple_arithmetic", OptimizationLevel::Default)
            .expect("Benchmark failed");
        
        let program = suite.benchmark_programs.get("simple_arithmetic").unwrap();
        let throughput = result.calculate_throughput(program.source_lines);
        let efficiency = result.calculate_efficiency();
        
        assert!(throughput > 0.0, "Throughput should be positive");
        assert!(efficiency > 0.0, "Efficiency should be positive");
        
        println!("Throughput: {:.2} lines/sec, Efficiency: {:.2} ops/ms", throughput, efficiency);
    }

    #[test]
    fn test_tensor_operations_benchmark() {
        let suite = CompilationBenchmarkSuite::new().expect("Failed to create suite");
        
        let result = suite.benchmark_full_compilation("tensor_operations", OptimizationLevel::Default)
            .expect("Tensor benchmark failed");
        
        assert_eq!(result.program_name, "tensor_operations");
        assert!(result.mlir_operations_count > 0, "Should have MLIR operations for tensor ops");
        assert!(result.generated_code_size > 0, "Should generate code for tensor ops");
        
        println!("Tensor operations benchmark: {:?}", result);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let suite = CompilationBenchmarkSuite::new().expect("Failed to create suite");
        
        let simple_result = suite.benchmark_full_compilation("simple_arithmetic", OptimizationLevel::Default)
            .expect("Simple benchmark failed");
        
        let complex_result = suite.benchmark_full_compilation("function_calls", OptimizationLevel::Default)
            .expect("Complex benchmark failed");
        
        // More complex program should use more memory
        assert!(complex_result.peak_memory_usage_mb >= simple_result.peak_memory_usage_mb,
            "Complex program should use more or equal memory");
        
        println!("Memory usage - Simple: {}MB, Complex: {}MB",
            simple_result.peak_memory_usage_mb, complex_result.peak_memory_usage_mb);
    }

    #[test]
    fn test_custom_benchmark_program() {
        let mut suite = CompilationBenchmarkSuite::new().expect("Failed to create suite");
        
        let custom_program = BenchmarkProgram {
            name: "custom_test".to_string(),
            source_code: r#"
(func main ()
  (let custom_var 123)
  (return custom_var))
"#.to_string(),
            expected_complexity: ProgramComplexity::Simple,
            source_lines: 4,
        };
        
        suite.add_benchmark_program(custom_program);
        
        let names = suite.get_benchmark_names();
        assert!(names.contains(&"custom_test".to_string()));
        
        let result = suite.benchmark_full_compilation("custom_test", OptimizationLevel::Default)
            .expect("Custom benchmark failed");
        
        assert_eq!(result.program_name, "custom_test");
        assert!(result.total_duration > Duration::from_millis(0));
    }
}