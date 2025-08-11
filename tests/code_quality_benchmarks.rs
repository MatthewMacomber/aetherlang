// Generated code quality benchmarking infrastructure
// Task 10.3: Write generated code quality benchmarks
// Measures and compares the quality of generated LLVM IR and native code

use aether_language::compiler::mlir::{
    MLIRContext, MLIRModule, AetherMLIRFrontend, AetherOptimizer,
    LLVMCodeGenerator, TargetConfig, OptimizationLevel
};
use aether_language::compiler::ast::{AST, ASTNode, ASTNodeRef};
use aether_language::compiler::parser::{parse_sexpr};
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::process::Command;
use tempfile::TempDir;
use std::fs;
use std::path::Path;

/// Code quality benchmark suite
pub struct CodeQualityBenchmarkSuite {
    temp_dir: TempDir,
    quality_metrics: HashMap<String, CodeQualityMetrics>,
}

impl CodeQualityBenchmarkSuite {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let quality_metrics = HashMap::new();
        
        Ok(CodeQualityBenchmarkSuite {
            temp_dir,
            quality_metrics,
        })
    }

    /// Benchmark code quality for a given program
    pub fn benchmark_code_quality(&mut self, program_name: &str, source_code: &str, opt_level: OptimizationLevel) -> Result<CodeQualityMetrics, Box<dyn std::error::Error>> {
        let mut metrics = CodeQualityMetrics::new(program_name.to_string(), opt_level);
        
        // Compile the program
        let llvm_ir = self.compile_to_llvm_ir(source_code, opt_level)?;
        
        // Analyze LLVM IR quality
        self.analyze_llvm_ir_quality(&llvm_ir, &mut metrics)?;
        
        // Try to compile to native code and analyze
        if let Ok(native_metrics) = self.analyze_native_code_quality(&llvm_ir, program_name) {
            metrics.native_code_metrics = Some(native_metrics);
        }
        
        // Store metrics for comparison
        self.quality_metrics.insert(format!("{}_{:?}", program_name, opt_level), metrics.clone());
        
        Ok(metrics)
    }

    /// Compile source code to LLVM IR
    fn compile_to_llvm_ir(&self, source_code: &str, opt_level: OptimizationLevel) -> Result<String, Box<dyn std::error::Error>> {
        let ast = parse_sexpr(source_code)?;
        
        let context = MLIRContext::new()?;
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("quality_test")?;
        
        frontend.convert_ast_to_module(&ast, &mut module)?;
        module.verify()?;
        
        // Apply optimizations
        if opt_level != OptimizationLevel::None {
            let optimizer = AetherOptimizer::new(&context);
            optimizer.optimize_with_level(&mut module, opt_level)?;
            module.verify()?;
        }
        
        // Generate LLVM IR
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
        
        codegen.get_llvm_ir_string()
    }

    /// Analyze LLVM IR quality metrics
    fn analyze_llvm_ir_quality(&self, llvm_ir: &str, metrics: &mut CodeQualityMetrics) -> Result<(), Box<dyn std::error::Error>> {
        let lines: Vec<&str> = llvm_ir.lines().collect();
        
        // Basic metrics
        metrics.total_lines = lines.len();
        metrics.code_size_bytes = llvm_ir.len();
        
        // Instruction analysis
        for line in &lines {
            let trimmed = line.trim();
            
            if trimmed.is_empty() || trimmed.starts_with(';') {
                metrics.comment_lines += 1;
                continue;
            }
            
            // Count different types of instructions
            if self.is_arithmetic_instruction(trimmed) {
                metrics.arithmetic_instructions += 1;
            } else if self.is_memory_instruction(trimmed) {
                metrics.memory_instructions += 1;
            } else if self.is_control_flow_instruction(trimmed) {
                metrics.control_flow_instructions += 1;
            } else if self.is_function_instruction(trimmed) {
                metrics.function_instructions += 1;
            }
            
            // Count optimizations
            if self.indicates_constant_folding(trimmed) {
                metrics.constant_folding_instances += 1;
            }
            if self.indicates_dead_code_elimination(trimmed) {
                metrics.dead_code_eliminations += 1;
            }
            if self.indicates_loop_optimization(trimmed) {
                metrics.loop_optimizations += 1;
            }
            if self.indicates_inlining(trimmed) {
                metrics.function_inlinings += 1;
            }
        }
        
        // Calculate derived metrics
        metrics.instruction_density = self.calculate_instruction_density(&lines);
        metrics.optimization_effectiveness = self.calculate_optimization_effectiveness(metrics);
        metrics.code_complexity_score = self.calculate_complexity_score(metrics);
        
        Ok(())
    }

    /// Check if instruction is arithmetic
    fn is_arithmetic_instruction(&self, line: &str) -> bool {
        line.contains(" add ") || line.contains(" sub ") || 
        line.contains(" mul ") || line.contains(" div ") ||
        line.contains(" fadd ") || line.contains(" fsub ") ||
        line.contains(" fmul ") || line.contains(" fdiv ") ||
        line.contains(" and ") || line.contains(" or ") ||
        line.contains(" xor ") || line.contains(" shl ") ||
        line.contains(" shr ")
    }

    /// Check if instruction is memory-related
    fn is_memory_instruction(&self, line: &str) -> bool {
        line.contains(" load ") || line.contains(" store ") ||
        line.contains(" alloca ") || line.contains(" malloc ") ||
        line.contains(" free ") || line.contains(" getelementptr ")
    }

    /// Check if instruction is control flow
    fn is_control_flow_instruction(&self, line: &str) -> bool {
        line.contains(" br ") || line.contains(" switch ") ||
        line.contains(" ret ") || line.contains(" call ") ||
        line.contains(" invoke ") || line.contains(" resume ")
    }

    /// Check if instruction is function-related
    fn is_function_instruction(&self, line: &str) -> bool {
        line.starts_with("define ") || line.starts_with("declare ") ||
        line.contains(" call ")
    }

    /// Check if line indicates constant folding optimization
    fn indicates_constant_folding(&self, line: &str) -> bool {
        // Look for immediate values in arithmetic operations
        if self.is_arithmetic_instruction(line) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            parts.iter().any(|part| part.parse::<i64>().is_ok() || part.parse::<f64>().is_ok())
        } else {
            false
        }
    }

    /// Check if optimization eliminated dead code
    fn indicates_dead_code_elimination(&self, _line: &str) -> bool {
        // This is difficult to detect directly in LLVM IR
        // In a real implementation, we'd compare before/after optimization
        false
    }

    /// Check if line indicates loop optimization
    fn indicates_loop_optimization(&self, line: &str) -> bool {
        line.contains("loop") || line.contains("vector") || line.contains("unroll")
    }

    /// Check if line indicates function inlining
    fn indicates_inlining(&self, line: &str) -> bool {
        // Inlined functions often have mangled names or inline attributes
        line.contains("inline") || line.contains("alwaysinline")
    }

    /// Calculate instruction density (instructions per line)
    fn calculate_instruction_density(&self, lines: &[&str]) -> f64 {
        let instruction_lines = lines.iter()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && !trimmed.starts_with(';') && 
                !trimmed.starts_with("define") && !trimmed.starts_with("declare") &&
                !trimmed.ends_with(':') && !trimmed.starts_with('}')
            })
            .count();
        
        if lines.len() > 0 {
            instruction_lines as f64 / lines.len() as f64
        } else {
            0.0
        }
    }

    /// Calculate optimization effectiveness score
    fn calculate_optimization_effectiveness(&self, metrics: &CodeQualityMetrics) -> f64 {
        let total_optimizations = metrics.constant_folding_instances + 
                                 metrics.dead_code_eliminations +
                                 metrics.loop_optimizations +
                                 metrics.function_inlinings;
        
        let total_instructions = metrics.arithmetic_instructions + 
                               metrics.memory_instructions +
                               metrics.control_flow_instructions +
                               metrics.function_instructions;
        
        if total_instructions > 0 {
            total_optimizations as f64 / total_instructions as f64
        } else {
            0.0
        }
    }

    /// Calculate code complexity score
    fn calculate_complexity_score(&self, metrics: &CodeQualityMetrics) -> f64 {
        // Higher control flow and memory operations increase complexity
        let complexity_weight = (metrics.control_flow_instructions as f64 * 2.0) +
                               (metrics.memory_instructions as f64 * 1.5) +
                               (metrics.function_instructions as f64 * 1.2) +
                               (metrics.arithmetic_instructions as f64 * 1.0);
        
        let total_instructions = metrics.arithmetic_instructions + 
                               metrics.memory_instructions +
                               metrics.control_flow_instructions +
                               metrics.function_instructions;
        
        if total_instructions > 0 {
            complexity_weight / total_instructions as f64
        } else {
            0.0
        }
    }

    /// Analyze native code quality (if compilation tools are available)
    fn analyze_native_code_quality(&self, llvm_ir: &str, program_name: &str) -> Result<NativeCodeMetrics, Box<dyn std::error::Error>> {
        let ir_file = self.temp_dir.path().join(format!("{}.ll", program_name));
        let obj_file = self.temp_dir.path().join(format!("{}.o", program_name));
        let asm_file = self.temp_dir.path().join(format!("{}.s", program_name));
        
        // Write LLVM IR to file
        fs::write(&ir_file, llvm_ir)?;
        
        let mut metrics = NativeCodeMetrics::new();
        
        // Try to compile to object file
        if let Ok(output) = Command::new("llc")
            .arg("-filetype=obj")
            .arg("-o")
            .arg(&obj_file)
            .arg(&ir_file)
            .output()
        {
            if output.status.success() && obj_file.exists() {
                metrics.object_file_size = fs::metadata(&obj_file)?.len() as usize;
                metrics.compilation_successful = true;
            }
        }
        
        // Try to generate assembly for analysis
        if let Ok(output) = Command::new("llc")
            .arg("-filetype=asm")
            .arg("-o")
            .arg(&asm_file)
            .arg(&ir_file)
            .output()
        {
            if output.status.success() && asm_file.exists() {
                let asm_content = fs::read_to_string(&asm_file)?;
                self.analyze_assembly_quality(&asm_content, &mut metrics)?;
            }
        }
        
        Ok(metrics)
    }

    /// Analyze assembly code quality
    fn analyze_assembly_quality(&self, assembly: &str, metrics: &mut NativeCodeMetrics) -> Result<(), Box<dyn std::error::Error>> {
        let lines: Vec<&str> = assembly.lines().collect();
        metrics.assembly_lines = lines.len();
        
        for line in &lines {
            let trimmed = line.trim();
            
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
                continue;
            }
            
            // Count different types of assembly instructions
            if self.is_x86_arithmetic_instruction(trimmed) {
                metrics.native_arithmetic_instructions += 1;
            } else if self.is_x86_memory_instruction(trimmed) {
                metrics.native_memory_instructions += 1;
            } else if self.is_x86_control_flow_instruction(trimmed) {
                metrics.native_control_flow_instructions += 1;
            }
            
            // Count optimizations in assembly
            if self.indicates_vectorization(trimmed) {
                metrics.vectorized_instructions += 1;
            }
            if self.indicates_register_optimization(trimmed) {
                metrics.register_optimizations += 1;
            }
        }
        
        // Calculate assembly-specific metrics
        metrics.instruction_efficiency = self.calculate_assembly_efficiency(metrics);
        
        Ok(())
    }

    /// Check if assembly instruction is arithmetic
    fn is_x86_arithmetic_instruction(&self, line: &str) -> bool {
        let arithmetic_ops = ["add", "sub", "mul", "div", "imul", "idiv", 
                             "fadd", "fsub", "fmul", "fdiv", "and", "or", "xor"];
        arithmetic_ops.iter().any(|op| line.contains(op))
    }

    /// Check if assembly instruction is memory-related
    fn is_x86_memory_instruction(&self, line: &str) -> bool {
        line.contains("mov") || line.contains("lea") || 
        line.contains("push") || line.contains("pop") ||
        line.contains("load") || line.contains("store")
    }

    /// Check if assembly instruction is control flow
    fn is_x86_control_flow_instruction(&self, line: &str) -> bool {
        let control_ops = ["jmp", "je", "jne", "jl", "jg", "jle", "jge", 
                          "call", "ret", "cmp", "test"];
        control_ops.iter().any(|op| line.contains(op))
    }

    /// Check if instruction indicates vectorization
    fn indicates_vectorization(&self, line: &str) -> bool {
        line.contains("xmm") || line.contains("ymm") || line.contains("zmm") ||
        line.contains("sse") || line.contains("avx")
    }

    /// Check if instruction indicates register optimization
    fn indicates_register_optimization(&self, line: &str) -> bool {
        // Look for efficient register usage patterns
        line.contains("%rax") || line.contains("%rbx") || line.contains("%rcx") ||
        line.contains("%rdx") || line.contains("%rsi") || line.contains("%rdi")
    }

    /// Calculate assembly instruction efficiency
    fn calculate_assembly_efficiency(&self, metrics: &NativeCodeMetrics) -> f64 {
        let total_instructions = metrics.native_arithmetic_instructions + 
                               metrics.native_memory_instructions +
                               metrics.native_control_flow_instructions;
        
        if total_instructions > 0 {
            let efficiency_score = (metrics.vectorized_instructions as f64 * 2.0) +
                                  (metrics.register_optimizations as f64 * 1.5) +
                                  (metrics.native_arithmetic_instructions as f64 * 1.0);
            efficiency_score / total_instructions as f64
        } else {
            0.0
        }
    }

    /// Compare code quality between different optimization levels
    pub fn compare_optimization_levels(&self, program_name: &str) -> Option<OptimizationComparison> {
        let none_key = format!("{}_None", program_name);
        let default_key = format!("{}_Default", program_name);
        let aggressive_key = format!("{}_Aggressive", program_name);
        
        let none_metrics = self.quality_metrics.get(&none_key)?;
        let default_metrics = self.quality_metrics.get(&default_key)?;
        let aggressive_metrics = self.quality_metrics.get(&aggressive_key)?;
        
        Some(OptimizationComparison {
            program_name: program_name.to_string(),
            none_metrics: none_metrics.clone(),
            default_metrics: default_metrics.clone(),
            aggressive_metrics: aggressive_metrics.clone(),
            code_size_reduction_default: self.calculate_reduction_percentage(
                none_metrics.code_size_bytes, default_metrics.code_size_bytes
            ),
            code_size_reduction_aggressive: self.calculate_reduction_percentage(
                none_metrics.code_size_bytes, aggressive_metrics.code_size_bytes
            ),
            complexity_reduction_default: self.calculate_reduction_percentage_f64(
                none_metrics.code_complexity_score, default_metrics.code_complexity_score
            ),
            complexity_reduction_aggressive: self.calculate_reduction_percentage_f64(
                none_metrics.code_complexity_score, aggressive_metrics.code_complexity_score
            ),
        })
    }

    /// Calculate percentage reduction
    fn calculate_reduction_percentage(&self, original: usize, optimized: usize) -> f64 {
        if original > 0 {
            ((original as f64 - optimized as f64) / original as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate percentage reduction for f64 values
    fn calculate_reduction_percentage_f64(&self, original: f64, optimized: f64) -> f64 {
        if original > 0.0 {
            ((original - optimized) / original) * 100.0
        } else {
            0.0
        }
    }

    /// Get quality metrics for a specific program and optimization level
    pub fn get_quality_metrics(&self, program_name: &str, opt_level: OptimizationLevel) -> Option<&CodeQualityMetrics> {
        let key = format!("{}_{:?}", program_name, opt_level);
        self.quality_metrics.get(&key)
    }

    /// Get all stored quality metrics
    pub fn get_all_metrics(&self) -> &HashMap<String, CodeQualityMetrics> {
        &self.quality_metrics
    }
}

/// Code quality metrics for LLVM IR
#[derive(Debug, Clone)]
pub struct CodeQualityMetrics {
    pub program_name: String,
    pub optimization_level: OptimizationLevel,
    pub total_lines: usize,
    pub comment_lines: usize,
    pub code_size_bytes: usize,
    pub arithmetic_instructions: usize,
    pub memory_instructions: usize,
    pub control_flow_instructions: usize,
    pub function_instructions: usize,
    pub constant_folding_instances: usize,
    pub dead_code_eliminations: usize,
    pub loop_optimizations: usize,
    pub function_inlinings: usize,
    pub instruction_density: f64,
    pub optimization_effectiveness: f64,
    pub code_complexity_score: f64,
    pub native_code_metrics: Option<NativeCodeMetrics>,
}

impl CodeQualityMetrics {
    fn new(program_name: String, optimization_level: OptimizationLevel) -> Self {
        CodeQualityMetrics {
            program_name,
            optimization_level,
            total_lines: 0,
            comment_lines: 0,
            code_size_bytes: 0,
            arithmetic_instructions: 0,
            memory_instructions: 0,
            control_flow_instructions: 0,
            function_instructions: 0,
            constant_folding_instances: 0,
            dead_code_eliminations: 0,
            loop_optimizations: 0,
            function_inlinings: 0,
            instruction_density: 0.0,
            optimization_effectiveness: 0.0,
            code_complexity_score: 0.0,
            native_code_metrics: None,
        }
    }

    /// Calculate total instruction count
    pub fn total_instructions(&self) -> usize {
        self.arithmetic_instructions + self.memory_instructions + 
        self.control_flow_instructions + self.function_instructions
    }

    /// Calculate optimization ratio
    pub fn optimization_ratio(&self) -> f64 {
        let total_optimizations = self.constant_folding_instances + 
                                 self.dead_code_eliminations +
                                 self.loop_optimizations +
                                 self.function_inlinings;
        
        if self.total_instructions() > 0 {
            total_optimizations as f64 / self.total_instructions() as f64
        } else {
            0.0
        }
    }
}

/// Native code quality metrics
#[derive(Debug, Clone)]
pub struct NativeCodeMetrics {
    pub compilation_successful: bool,
    pub object_file_size: usize,
    pub assembly_lines: usize,
    pub native_arithmetic_instructions: usize,
    pub native_memory_instructions: usize,
    pub native_control_flow_instructions: usize,
    pub vectorized_instructions: usize,
    pub register_optimizations: usize,
    pub instruction_efficiency: f64,
}

impl NativeCodeMetrics {
    fn new() -> Self {
        NativeCodeMetrics {
            compilation_successful: false,
            object_file_size: 0,
            assembly_lines: 0,
            native_arithmetic_instructions: 0,
            native_memory_instructions: 0,
            native_control_flow_instructions: 0,
            vectorized_instructions: 0,
            register_optimizations: 0,
            instruction_efficiency: 0.0,
        }
    }

    /// Calculate total native instruction count
    pub fn total_native_instructions(&self) -> usize {
        self.native_arithmetic_instructions + self.native_memory_instructions + 
        self.native_control_flow_instructions
    }
}

/// Comparison of optimization levels
#[derive(Debug, Clone)]
pub struct OptimizationComparison {
    pub program_name: String,
    pub none_metrics: CodeQualityMetrics,
    pub default_metrics: CodeQualityMetrics,
    pub aggressive_metrics: CodeQualityMetrics,
    pub code_size_reduction_default: f64,
    pub code_size_reduction_aggressive: f64,
    pub complexity_reduction_default: f64,
    pub complexity_reduction_aggressive: f64,
}

// ===== CRITERION BENCHMARKS =====

/// Benchmark code quality across different programs
fn benchmark_code_quality_comparison(c: &mut Criterion) {
    let mut suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
    
    let test_programs = vec![
        ("simple_arithmetic", r#"
(func main ()
  (let x (+ 10 20))
  (let y (* 5 6))
  (let z (- x y))
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
    ];
    
    let mut group = c.benchmark_group("code_quality");
    
    for (program_name, source_code) in test_programs {
        group.bench_function(&format!("{}_unoptimized", program_name), |b| {
            b.iter(|| {
                suite.benchmark_code_quality(program_name, source_code, OptimizationLevel::None)
                    .expect("Quality benchmark failed")
            })
        });
        
        group.bench_function(&format!("{}_optimized", program_name), |b| {
            b.iter(|| {
                suite.benchmark_code_quality(program_name, source_code, OptimizationLevel::Default)
                    .expect("Quality benchmark failed")
            })
        });
    }
    
    group.finish();
}

/// Benchmark optimization effectiveness
fn benchmark_optimization_effectiveness(c: &mut Criterion) {
    let mut suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
    
    let optimization_test_program = r#"
(func main ()
  (let a (+ 1 2))
  (let b (* 3 4))
  (let c (+ a b))
  (let unused_var (* c 100))
  (return c))
"#;
    
    let mut group = c.benchmark_group("optimization_effectiveness");
    
    let optimization_levels = vec![
        ("none", OptimizationLevel::None),
        ("less", OptimizationLevel::Less),
        ("default", OptimizationLevel::Default),
        ("aggressive", OptimizationLevel::Aggressive),
    ];
    
    for (level_name, opt_level) in optimization_levels {
        group.bench_function(level_name, |b| {
            b.iter(|| {
                suite.benchmark_code_quality("optimization_test", optimization_test_program, opt_level)
                    .expect("Optimization effectiveness benchmark failed")
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_code_quality_comparison,
    benchmark_optimization_effectiveness
);
criterion_main!(benches);

// ===== UNIT TESTS =====

#[cfg(test)]
mod code_quality_tests {
    use super::*;

    #[test]
    fn test_code_quality_suite_creation() {
        let suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        assert!(suite.quality_metrics.is_empty(), "Should start with no metrics");
    }

    #[test]
    fn test_simple_code_quality_benchmark() {
        let mut suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        let simple_program = r#"
(func main ()
  (let x 42)
  (return x))
"#;
        
        let metrics = suite.benchmark_code_quality("simple_test", simple_program, OptimizationLevel::Default)
            .expect("Quality benchmark failed");
        
        assert_eq!(metrics.program_name, "simple_test");
        assert!(metrics.total_lines > 0, "Should have generated LLVM IR lines");
        assert!(metrics.code_size_bytes > 0, "Should have generated code");
        assert!(metrics.instruction_density > 0.0, "Should have instruction density");
        
        println!("Simple program metrics: {:?}", metrics);
    }

    #[test]
    fn test_arithmetic_instruction_detection() {
        let suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        assert!(suite.is_arithmetic_instruction("  %3 = add i32 %1, %2"));
        assert!(suite.is_arithmetic_instruction("  %4 = mul i32 %3, 5"));
        assert!(suite.is_arithmetic_instruction("  %5 = fadd double %a, %b"));
        assert!(!suite.is_arithmetic_instruction("  %6 = load i32, i32* %ptr"));
        assert!(!suite.is_arithmetic_instruction("  br label %loop"));
    }

    #[test]
    fn test_memory_instruction_detection() {
        let suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        assert!(suite.is_memory_instruction("  %1 = load i32, i32* %ptr"));
        assert!(suite.is_memory_instruction("  store i32 %val, i32* %ptr"));
        assert!(suite.is_memory_instruction("  %2 = alloca i32"));
        assert!(!suite.is_memory_instruction("  %3 = add i32 %1, %2"));
        assert!(!suite.is_memory_instruction("  ret i32 %result"));
    }

    #[test]
    fn test_control_flow_instruction_detection() {
        let suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        assert!(suite.is_control_flow_instruction("  br label %loop"));
        assert!(suite.is_control_flow_instruction("  ret i32 %result"));
        assert!(suite.is_control_flow_instruction("  %1 = call i32 @func()"));
        assert!(!suite.is_control_flow_instruction("  %2 = add i32 %a, %b"));
        assert!(!suite.is_control_flow_instruction("  store i32 %val, i32* %ptr"));
    }

    #[test]
    fn test_optimization_comparison() {
        let mut suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        let test_program = r#"
(func main ()
  (let x (+ 10 20))
  (let y (* 3 4))
  (let z (+ x y))
  (return z))
"#;
        
        // Benchmark different optimization levels
        let none_metrics = suite.benchmark_code_quality("comparison_test", test_program, OptimizationLevel::None)
            .expect("None optimization benchmark failed");
        
        let default_metrics = suite.benchmark_code_quality("comparison_test", test_program, OptimizationLevel::Default)
            .expect("Default optimization benchmark failed");
        
        let aggressive_metrics = suite.benchmark_code_quality("comparison_test", test_program, OptimizationLevel::Aggressive)
            .expect("Aggressive optimization benchmark failed");
        
        // Optimized versions should have same or better metrics
        assert!(default_metrics.optimization_effectiveness >= none_metrics.optimization_effectiveness,
            "Default optimization should be more effective");
        
        assert!(aggressive_metrics.optimization_effectiveness >= default_metrics.optimization_effectiveness,
            "Aggressive optimization should be most effective");
        
        println!("Optimization comparison:");
        println!("  None: {:.3} effectiveness", none_metrics.optimization_effectiveness);
        println!("  Default: {:.3} effectiveness", default_metrics.optimization_effectiveness);
        println!("  Aggressive: {:.3} effectiveness", aggressive_metrics.optimization_effectiveness);
    }

    #[test]
    fn test_constant_folding_detection() {
        let suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        // These should indicate constant folding
        assert!(suite.indicates_constant_folding("  %1 = add i32 10, 20"));
        assert!(suite.indicates_constant_folding("  %2 = mul i32 %x, 5"));
        
        // These should not
        assert!(!suite.indicates_constant_folding("  %3 = add i32 %a, %b"));
        assert!(!suite.indicates_constant_folding("  store i32 %val, i32* %ptr"));
    }

    #[test]
    fn test_complexity_score_calculation() {
        let suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        let mut simple_metrics = CodeQualityMetrics::new("simple".to_string(), OptimizationLevel::Default);
        simple_metrics.arithmetic_instructions = 10;
        simple_metrics.memory_instructions = 2;
        simple_metrics.control_flow_instructions = 1;
        simple_metrics.function_instructions = 1;
        
        let mut complex_metrics = CodeQualityMetrics::new("complex".to_string(), OptimizationLevel::Default);
        complex_metrics.arithmetic_instructions = 5;
        complex_metrics.memory_instructions = 8;
        complex_metrics.control_flow_instructions = 6;
        complex_metrics.function_instructions = 4;
        
        let simple_complexity = suite.calculate_complexity_score(&simple_metrics);
        let complex_complexity = suite.calculate_complexity_score(&complex_metrics);
        
        assert!(complex_complexity > simple_complexity, 
            "Complex program should have higher complexity score");
        
        println!("Complexity scores - Simple: {:.3}, Complex: {:.3}", 
            simple_complexity, complex_complexity);
    }

    #[test]
    fn test_instruction_density_calculation() {
        let suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        let lines = vec![
            "; This is a comment",
            "define i32 @main() {",
            "  %1 = add i32 1, 2",
            "  %2 = mul i32 %1, 3",
            "  ret i32 %2",
            "}",
            "",
        ];
        
        let density = suite.calculate_instruction_density(&lines);
        
        // Should be around 3 instructions / 7 lines = ~0.43
        assert!(density > 0.0 && density < 1.0, "Density should be between 0 and 1");
        
        println!("Instruction density: {:.3}", density);
    }

    #[test]
    fn test_native_code_metrics_creation() {
        let metrics = NativeCodeMetrics::new();
        
        assert!(!metrics.compilation_successful);
        assert_eq!(metrics.object_file_size, 0);
        assert_eq!(metrics.total_native_instructions(), 0);
        assert_eq!(metrics.instruction_efficiency, 0.0);
    }

    #[test]
    fn test_x86_instruction_detection() {
        let suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        // Arithmetic instructions
        assert!(suite.is_x86_arithmetic_instruction("  addq %rax, %rbx"));
        assert!(suite.is_x86_arithmetic_instruction("  imulq $5, %rcx"));
        assert!(suite.is_x86_arithmetic_instruction("  xorq %rdx, %rdx"));
        
        // Memory instructions
        assert!(suite.is_x86_memory_instruction("  movq %rax, -8(%rbp)"));
        assert!(suite.is_x86_memory_instruction("  leaq -16(%rbp), %rax"));
        assert!(suite.is_x86_memory_instruction("  pushq %rbp"));
        
        // Control flow instructions
        assert!(suite.is_x86_control_flow_instruction("  jmp .L1"));
        assert!(suite.is_x86_control_flow_instruction("  callq func"));
        assert!(suite.is_x86_control_flow_instruction("  retq"));
    }

    #[test]
    fn test_vectorization_detection() {
        let suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        assert!(suite.indicates_vectorization("  addps %xmm0, %xmm1"));
        assert!(suite.indicates_vectorization("  vmulpd %ymm0, %ymm1, %ymm2"));
        assert!(suite.indicates_vectorization("  vfmadd213ss %xmm0, %xmm1, %xmm2"));
        assert!(!suite.indicates_vectorization("  addq %rax, %rbx"));
    }

    #[test]
    fn test_quality_metrics_storage_and_retrieval() {
        let mut suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        let test_program = r#"
(func main ()
  (let result (+ 1 2))
  (return result))
"#;
        
        let _metrics = suite.benchmark_code_quality("storage_test", test_program, OptimizationLevel::Default)
            .expect("Quality benchmark failed");
        
        // Check that metrics were stored
        let retrieved_metrics = suite.get_quality_metrics("storage_test", OptimizationLevel::Default);
        assert!(retrieved_metrics.is_some(), "Metrics should be stored and retrievable");
        
        let all_metrics = suite.get_all_metrics();
        assert!(!all_metrics.is_empty(), "Should have stored metrics");
        
        println!("Stored {} metric sets", all_metrics.len());
    }

    #[test]
    fn test_tensor_operations_quality() {
        let mut suite = CodeQualityBenchmarkSuite::new().expect("Failed to create suite");
        
        let tensor_program = r#"
(func main ()
  (let tensor_a (tensor-create [4 4] f32))
  (let tensor_b (tensor-create [4 4] f32))
  (let result (tensor-add tensor_a tensor_b))
  (return 0))
"#;
        
        let metrics = suite.benchmark_code_quality("tensor_quality", tensor_program, OptimizationLevel::Default)
            .expect("Tensor quality benchmark failed");
        
        assert!(metrics.total_instructions() > 0, "Should generate instructions for tensor operations");
        assert!(metrics.code_size_bytes > 0, "Should generate code for tensor operations");
        
        println!("Tensor operations quality metrics: {:?}", metrics);
    }
}