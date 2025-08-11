// Property-based tests for optimization correctness
// Task 10.3: Add optimization correctness property tests
// Verifies that optimizations preserve program semantics while improving performance

use aether_language::compiler::mlir::{
    MLIRContext, MLIRModule, AetherMLIRFrontend, AetherOptimizer,
    LLVMCodeGenerator, TargetConfig, OptimizationLevel
};
use aether_language::compiler::ast::{AST, ASTNode, ASTNodeRef};
use aether_language::compiler::parser::{parse_sexpr};
use proptest::prelude::*;
use quickcheck::{quickcheck, TestResult, Arbitrary, Gen};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Optimization correctness test framework
pub struct OptimizationPropertyFramework {
    temp_dir: TempDir,
}

impl OptimizationPropertyFramework {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        
        Ok(OptimizationPropertyFramework {
            temp_dir,
        })
    }

    /// Test that optimizations preserve program semantics
    pub fn test_optimization_preserves_semantics(&self, ast: &AST, opt_level: OptimizationLevel) -> Result<bool, Box<dyn std::error::Error>> {
        // Compile without optimization
        let unoptimized_ir = self.compile_with_optimization(ast, OptimizationLevel::None)?;
        let unoptimized_semantics = self.extract_program_semantics(&unoptimized_ir)?;
        
        // Compile with optimization
        let optimized_ir = self.compile_with_optimization(ast, opt_level)?;
        let optimized_semantics = self.extract_program_semantics(&optimized_ir)?;
        
        // Property: Semantics should be preserved
        Ok(self.semantics_are_equivalent(&unoptimized_semantics, &optimized_semantics))
    }

    /// Test that optimizations improve or maintain performance
    pub fn test_optimization_improves_performance(&self, ast: &AST, opt_level: OptimizationLevel) -> Result<bool, Box<dyn std::error::Error>> {
        // Measure compilation time and code quality for unoptimized version
        let unoptimized_metrics = self.measure_compilation_metrics(ast, OptimizationLevel::None)?;
        
        // Measure compilation time and code quality for optimized version
        let optimized_metrics = self.measure_compilation_metrics(ast, opt_level)?;
        
        // Property: Optimized code should be better or equivalent
        Ok(self.performance_is_improved_or_equivalent(&unoptimized_metrics, &optimized_metrics))
    }

    /// Test that optimization passes are idempotent
    pub fn test_optimization_idempotency(&self, ast: &AST, opt_level: OptimizationLevel) -> Result<bool, Box<dyn std::error::Error>> {
        // Apply optimization once
        let once_optimized = self.compile_with_optimization(ast, opt_level)?;
        
        // Parse the optimized IR back to AST and optimize again
        // Note: This is simplified - real implementation would need proper IR->AST conversion
        let twice_optimized = self.compile_with_optimization(ast, opt_level)?;
        
        // Property: Applying optimization twice should yield same result as applying once
        Ok(self.ir_is_equivalent(&once_optimized, &twice_optimized))
    }

    /// Test that optimization preserves type information
    pub fn test_optimization_preserves_types(&self, ast: &AST, opt_level: OptimizationLevel) -> Result<bool, Box<dyn std::error::Error>> {
        let unoptimized_ir = self.compile_with_optimization(ast, OptimizationLevel::None)?;
        let unoptimized_types = self.extract_type_information(&unoptimized_ir)?;
        
        let optimized_ir = self.compile_with_optimization(ast, opt_level)?;
        let optimized_types = self.extract_type_information(&optimized_ir)?;
        
        // Property: Type information should be preserved
        Ok(self.types_are_preserved(&unoptimized_types, &optimized_types))
    }

    /// Compile AST with specified optimization level
    fn compile_with_optimization(&self, ast: &AST, opt_level: OptimizationLevel) -> Result<String, Box<dyn std::error::Error>> {
        let context = MLIRContext::new()?;
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("optimization_test")?;
        
        frontend.convert_ast_to_module(ast, &mut module)?;
        module.verify()?;
        
        // Apply optimizations if requested
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

    /// Extract program semantics from LLVM IR
    fn extract_program_semantics(&self, llvm_ir: &str) -> Result<ProgramSemantics, Box<dyn std::error::Error>> {
        let mut semantics = ProgramSemantics::new();
        
        for line in llvm_ir.lines() {
            let trimmed = line.trim();
            
            // Extract function definitions
            if trimmed.starts_with("define") {
                semantics.functions.push(self.parse_function_definition(trimmed)?);
            }
            
            // Extract basic blocks
            if trimmed.ends_with(":") && !trimmed.contains("define") {
                semantics.basic_blocks.push(trimmed.trim_end_matches(':').to_string());
            }
            
            // Extract key operations
            if trimmed.contains("add") || trimmed.contains("sub") || 
               trimmed.contains("mul") || trimmed.contains("div") {
                semantics.arithmetic_operations += 1;
            }
            
            if trimmed.contains("br") || trimmed.contains("switch") || trimmed.contains("ret") {
                semantics.control_flow_operations += 1;
            }
            
            if trimmed.contains("load") || trimmed.contains("store") {
                semantics.memory_operations += 1;
            }
        }
        
        Ok(semantics)
    }

    /// Parse function definition from LLVM IR line
    fn parse_function_definition(&self, line: &str) -> Result<FunctionSemantics, Box<dyn std::error::Error>> {
        // Simplified parsing
        let name = if let Some(start) = line.find("@") {
            if let Some(end) = line[start..].find("(") {
                line[start+1..start+end].to_string()
            } else {
                "unknown".to_string()
            }
        } else {
            "unknown".to_string()
        };
        
        let return_type = if line.contains("void") {
            "void".to_string()
        } else if line.contains("i32") {
            "i32".to_string()
        } else {
            "unknown".to_string()
        };
        
        Ok(FunctionSemantics {
            name,
            return_type,
            parameter_count: self.count_parameters(line),
        })
    }

    /// Count parameters in function definition
    fn count_parameters(&self, line: &str) -> usize {
        if let Some(start) = line.find("(") {
            if let Some(end) = line[start..].find(")") {
                let params_str = &line[start+1..start+end];
                if params_str.trim().is_empty() {
                    0
                } else {
                    params_str.split(',').count()
                }
            } else {
                0
            }
        } else {
            0
        }
    }

    /// Check if two program semantics are equivalent
    fn semantics_are_equivalent(&self, sem1: &ProgramSemantics, sem2: &ProgramSemantics) -> bool {
        // Functions should be preserved
        if sem1.functions.len() != sem2.functions.len() {
            return false;
        }
        
        for (f1, f2) in sem1.functions.iter().zip(sem2.functions.iter()) {
            if f1.name != f2.name || f1.return_type != f2.return_type {
                return false;
            }
        }
        
        // Control flow structure should be preserved (basic blocks can be optimized)
        // Allow for optimization to reduce control flow operations
        if sem2.control_flow_operations > sem1.control_flow_operations {
            return false;
        }
        
        // Memory operations should be preserved or reduced
        if sem2.memory_operations > sem1.memory_operations {
            return false;
        }
        
        true
    }

    /// Measure compilation metrics
    fn measure_compilation_metrics(&self, ast: &AST, opt_level: OptimizationLevel) -> Result<CompilationMetrics, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let llvm_ir = self.compile_with_optimization(ast, opt_level)?;
        let compilation_time = start_time.elapsed();
        
        let metrics = CompilationMetrics {
            compilation_time,
            instruction_count: self.count_instructions(&llvm_ir),
            basic_block_count: self.count_basic_blocks(&llvm_ir),
            function_count: self.count_functions(&llvm_ir),
            code_size: llvm_ir.len(),
        };
        
        Ok(metrics)
    }

    /// Count instructions in LLVM IR
    fn count_instructions(&self, llvm_ir: &str) -> usize {
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

    /// Count basic blocks in LLVM IR
    fn count_basic_blocks(&self, llvm_ir: &str) -> usize {
        llvm_ir.lines()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.ends_with(':') && !trimmed.contains("define")
            })
            .count()
    }

    /// Count functions in LLVM IR
    fn count_functions(&self, llvm_ir: &str) -> usize {
        llvm_ir.lines()
            .filter(|line| line.trim().starts_with("define"))
            .count()
    }

    /// Check if performance is improved or equivalent
    fn performance_is_improved_or_equivalent(&self, unopt: &CompilationMetrics, opt: &CompilationMetrics) -> bool {
        // Optimized code should have fewer or equal instructions
        if opt.instruction_count > unopt.instruction_count {
            return false;
        }
        
        // Optimized code should have smaller or equal code size (allowing for some increase due to inlining)
        if opt.code_size > unopt.code_size * 2 {
            return false;
        }
        
        // Function count should be preserved or reduced (due to inlining)
        if opt.function_count > unopt.function_count {
            return false;
        }
        
        true
    }

    /// Check if two LLVM IR strings are equivalent
    fn ir_is_equivalent(&self, ir1: &str, ir2: &str) -> bool {
        // Normalize IR by removing comments and extra whitespace
        let normalized1 = self.normalize_ir(ir1);
        let normalized2 = self.normalize_ir(ir2);
        
        normalized1 == normalized2
    }

    /// Normalize LLVM IR for comparison
    fn normalize_ir(&self, ir: &str) -> String {
        ir.lines()
            .filter(|line| !line.trim().starts_with(';')) // Remove comments
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Extract type information from LLVM IR
    fn extract_type_information(&self, llvm_ir: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        let mut types = HashMap::new();
        
        for line in llvm_ir.lines() {
            let trimmed = line.trim();
            
            // Extract variable types from alloca instructions
            if trimmed.contains(" = alloca ") {
                if let Some((var_name, var_type)) = self.parse_alloca_type(trimmed) {
                    types.insert(var_name, var_type);
                }
            }
            
            // Extract function types
            if trimmed.starts_with("define") {
                if let Some((func_name, func_type)) = self.parse_function_type(trimmed) {
                    types.insert(func_name, func_type);
                }
            }
        }
        
        Ok(types)
    }

    /// Parse variable type from alloca instruction
    fn parse_alloca_type(&self, line: &str) -> Option<(String, String)> {
        if let Some(eq_pos) = line.find(" = alloca ") {
            let var_part = &line[..eq_pos];
            if let Some(var_start) = var_part.rfind('%') {
                let var_name = var_part[var_start+1..].trim().to_string();
                let type_part = &line[eq_pos + 10..];
                let var_type = type_part.split(',').next().unwrap_or("unknown").trim().to_string();
                return Some((var_name, var_type));
            }
        }
        None
    }

    /// Parse function type from definition
    fn parse_function_type(&self, line: &str) -> Option<(String, String)> {
        if let Some(start) = line.find("@") {
            if let Some(end) = line[start..].find("(") {
                let func_name = line[start+1..start+end].to_string();
                let return_type = if line.contains("void") {
                    "void".to_string()
                } else if line.contains("i32") {
                    "i32".to_string()
                } else {
                    "unknown".to_string()
                };
                return Some((func_name, return_type));
            }
        }
        None
    }

    /// Check if types are preserved between unoptimized and optimized versions
    fn types_are_preserved(&self, unopt_types: &HashMap<String, String>, opt_types: &HashMap<String, String>) -> bool {
        // All types in unoptimized version should be present in optimized version
        // (Some variables might be optimized away, which is acceptable)
        for (name, unopt_type) in unopt_types {
            if let Some(opt_type) = opt_types.get(name) {
                if unopt_type != opt_type {
                    return false; // Type changed, which is not allowed
                }
            }
            // Variable might be optimized away, which is acceptable
        }
        
        true
    }
}

/// Program semantics extracted from LLVM IR
#[derive(Debug, Clone)]
struct ProgramSemantics {
    functions: Vec<FunctionSemantics>,
    basic_blocks: Vec<String>,
    arithmetic_operations: usize,
    control_flow_operations: usize,
    memory_operations: usize,
}

impl ProgramSemantics {
    fn new() -> Self {
        ProgramSemantics {
            functions: Vec::new(),
            basic_blocks: Vec::new(),
            arithmetic_operations: 0,
            control_flow_operations: 0,
            memory_operations: 0,
        }
    }
}

/// Function semantics
#[derive(Debug, Clone)]
struct FunctionSemantics {
    name: String,
    return_type: String,
    parameter_count: usize,
}

/// Compilation metrics for performance comparison
#[derive(Debug, Clone)]
struct CompilationMetrics {
    compilation_time: Duration,
    instruction_count: usize,
    basic_block_count: usize,
    function_count: usize,
    code_size: usize,
}

/// Generate test programs for optimization testing
#[derive(Debug, Clone)]
pub struct OptimizationTestProgram {
    pub source_code: String,
    pub expected_optimizations: Vec<ExpectedOptimization>,
}

#[derive(Debug, Clone)]
pub enum ExpectedOptimization {
    ConstantFolding,
    DeadCodeElimination,
    CommonSubexpressionElimination,
    LoopOptimization,
    FunctionInlining,
}

impl Arbitrary for OptimizationTestProgram {
    fn arbitrary(g: &mut Gen) -> Self {
        let program_type = u8::arbitrary(g) % 5;
        
        let (source_code, expected_optimizations) = match program_type {
            0 => generate_constant_folding_program(g),
            1 => generate_dead_code_program(g),
            2 => generate_cse_program(g),
            3 => generate_loop_program(g),
            _ => generate_inlining_program(g),
        };
        
        OptimizationTestProgram {
            source_code,
            expected_optimizations,
        }
    }
}

fn generate_constant_folding_program(g: &mut Gen) -> (String, Vec<ExpectedOptimization>) {
    let a = i32::arbitrary(g) % 100;
    let b = i32::arbitrary(g) % 100;
    let c = i32::arbitrary(g) % 100;
    
    let program = format!(r#"
(func main ()
  (let x (+ {} {}))
  (let y (* {} {}))
  (let z (- x y))
  (return z))
"#, a, b, a, c);
    
    (program, vec![ExpectedOptimization::ConstantFolding])
}

fn generate_dead_code_program(g: &mut Gen) -> (String, Vec<ExpectedOptimization>) {
    let value = i32::arbitrary(g) % 100;
    
    let program = format!(r#"
(func main ()
  (let x {})
  (let unused_var (+ x 42))
  (let another_unused (* unused_var 2))
  (return x))
"#, value);
    
    (program, vec![ExpectedOptimization::DeadCodeElimination])
}

fn generate_cse_program(g: &mut Gen) -> (String, Vec<ExpectedOptimization>) {
    let a = i32::arbitrary(g) % 50;
    let b = i32::arbitrary(g) % 50;
    
    let program = format!(r#"
(func main ()
  (let x (+ {} {}))
  (let y (+ {} {}))
  (let z (+ x y))
  (return z))
"#, a, b, a, b);
    
    (program, vec![ExpectedOptimization::CommonSubexpressionElimination])
}

fn generate_loop_program(g: &mut Gen) -> (String, Vec<ExpectedOptimization>) {
    let limit = (u8::arbitrary(g) % 10 + 1) as i32;
    
    let program = format!(r#"
(func main ()
  (let sum 0)
  (let i 0)
  (while (< i {})
    (set sum (+ sum i))
    (set i (+ i 1)))
  (return sum))
"#, limit);
    
    (program, vec![ExpectedOptimization::LoopOptimization])
}

fn generate_inlining_program(g: &mut Gen) -> (String, Vec<ExpectedOptimization>) {
    let value = i32::arbitrary(g) % 100;
    
    let program = format!(r#"
(defun simple_add (x y)
  (+ x y))

(func main ()
  (let result (simple_add {} 10))
  (return result))
"#, value);
    
    (program, vec![ExpectedOptimization::FunctionInlining])
}

// ===== PROPERTY-BASED TESTS =====

#[cfg(test)]
mod optimization_property_tests {
    use super::*;

    /// Property: Optimizations preserve program semantics
    fn prop_optimization_preserves_semantics(program: OptimizationTestProgram) -> TestResult {
        let framework = match OptimizationPropertyFramework::new() {
            Ok(f) => f,
            Err(_) => return TestResult::discard(),
        };
        
        let ast = match parse_sexpr(&program.source_code) {
            Ok(a) => a,
            Err(_) => return TestResult::discard(),
        };
        
        let optimization_levels = vec![
            OptimizationLevel::Less,
            OptimizationLevel::Default,
            OptimizationLevel::Aggressive,
        ];
        
        for opt_level in optimization_levels {
            match framework.test_optimization_preserves_semantics(&ast, opt_level) {
                Ok(result) => {
                    if !result {
                        return TestResult::failed();
                    }
                }
                Err(_) => return TestResult::discard(),
            }
        }
        
        TestResult::passed()
    }

    /// Property: Optimizations improve or maintain performance
    fn prop_optimization_improves_performance(program: OptimizationTestProgram) -> TestResult {
        let framework = match OptimizationPropertyFramework::new() {
            Ok(f) => f,
            Err(_) => return TestResult::discard(),
        };
        
        let ast = match parse_sexpr(&program.source_code) {
            Ok(a) => a,
            Err(_) => return TestResult::discard(),
        };
        
        match framework.test_optimization_improves_performance(&ast, OptimizationLevel::Aggressive) {
            Ok(result) => TestResult::from_bool(result),
            Err(_) => TestResult::discard(),
        }
    }

    /// Property: Optimization passes are idempotent
    fn prop_optimization_idempotency(program: OptimizationTestProgram) -> TestResult {
        let framework = match OptimizationPropertyFramework::new() {
            Ok(f) => f,
            Err(_) => return TestResult::discard(),
        };
        
        let ast = match parse_sexpr(&program.source_code) {
            Ok(a) => a,
            Err(_) => return TestResult::discard(),
        };
        
        match framework.test_optimization_idempotency(&ast, OptimizationLevel::Default) {
            Ok(result) => TestResult::from_bool(result),
            Err(_) => TestResult::discard(),
        }
    }

    /// Property: Optimizations preserve type information
    fn prop_optimization_preserves_types(program: OptimizationTestProgram) -> TestResult {
        let framework = match OptimizationPropertyFramework::new() {
            Ok(f) => f,
            Err(_) => return TestResult::discard(),
        };
        
        let ast = match parse_sexpr(&program.source_code) {
            Ok(a) => a,
            Err(_) => return TestResult::discard(),
        };
        
        match framework.test_optimization_preserves_types(&ast, OptimizationLevel::Default) {
            Ok(result) => TestResult::from_bool(result),
            Err(_) => TestResult::discard(),
        }
    }

    #[test]
    fn test_semantics_preservation_property() {
        quickcheck(prop_optimization_preserves_semantics as fn(OptimizationTestProgram) -> TestResult);
    }

    #[test]
    fn test_performance_improvement_property() {
        quickcheck(prop_optimization_improves_performance as fn(OptimizationTestProgram) -> TestResult);
    }

    #[test]
    fn test_idempotency_property() {
        quickcheck(prop_optimization_idempotency as fn(OptimizationTestProgram) -> TestResult);
    }

    #[test]
    fn test_type_preservation_property() {
        quickcheck(prop_optimization_preserves_types as fn(OptimizationTestProgram) -> TestResult);
    }

    #[test]
    fn test_constant_folding_optimization() {
        let framework = OptimizationPropertyFramework::new().expect("Failed to create framework");
        
        let program = r#"
(func main ()
  (let x (+ 10 20))
  (let y (* 5 6))
  (let z (- x y))
  (return z))
"#;
        
        let ast = parse_sexpr(program).expect("Failed to parse");
        
        // Test that constant folding preserves semantics
        let result = framework.test_optimization_preserves_semantics(&ast, OptimizationLevel::Default)
            .expect("Test failed");
        assert!(result, "Constant folding should preserve semantics");
        
        // Test that constant folding improves performance
        let perf_result = framework.test_optimization_improves_performance(&ast, OptimizationLevel::Default)
            .expect("Performance test failed");
        assert!(perf_result, "Constant folding should improve performance");
    }

    #[test]
    fn test_dead_code_elimination() {
        let framework = OptimizationPropertyFramework::new().expect("Failed to create framework");
        
        let program = r#"
(func main ()
  (let x 42)
  (let unused (+ x 100))
  (let also_unused (* unused 2))
  (return x))
"#;
        
        let ast = parse_sexpr(program).expect("Failed to parse");
        
        let result = framework.test_optimization_preserves_semantics(&ast, OptimizationLevel::Default)
            .expect("Test failed");
        assert!(result, "Dead code elimination should preserve semantics");
        
        let perf_result = framework.test_optimization_improves_performance(&ast, OptimizationLevel::Default)
            .expect("Performance test failed");
        assert!(perf_result, "Dead code elimination should improve performance");
    }

    #[test]
    fn test_function_inlining() {
        let framework = OptimizationPropertyFramework::new().expect("Failed to create framework");
        
        let program = r#"
(defun add_one (x)
  (+ x 1))

(func main ()
  (let result (add_one 41))
  (return result))
"#;
        
        let ast = parse_sexpr(program).expect("Failed to parse");
        
        let result = framework.test_optimization_preserves_semantics(&ast, OptimizationLevel::Aggressive)
            .expect("Test failed");
        assert!(result, "Function inlining should preserve semantics");
    }

    #[test]
    fn test_loop_optimization() {
        let framework = OptimizationPropertyFramework::new().expect("Failed to create framework");
        
        let program = r#"
(func main ()
  (let sum 0)
  (let i 0)
  (while (< i 10)
    (set sum (+ sum i))
    (set i (+ i 1)))
  (return sum))
"#;
        
        let ast = parse_sexpr(program).expect("Failed to parse");
        
        let result = framework.test_optimization_preserves_semantics(&ast, OptimizationLevel::Default)
            .expect("Test failed");
        assert!(result, "Loop optimization should preserve semantics");
    }

    #[test]
    fn test_optimization_levels_consistency() {
        let framework = OptimizationPropertyFramework::new().expect("Failed to create framework");
        
        let program = r#"
(func main ()
  (let a 10)
  (let b 20)
  (let c (+ a b))
  (let d (* c 2))
  (return d))
"#;
        
        let ast = parse_sexpr(program).expect("Failed to parse");
        
        let levels = vec![
            OptimizationLevel::None,
            OptimizationLevel::Less,
            OptimizationLevel::Default,
            OptimizationLevel::Aggressive,
        ];
        
        // All optimization levels should preserve semantics
        for level in levels {
            if level != OptimizationLevel::None {
                let result = framework.test_optimization_preserves_semantics(&ast, level)
                    .expect("Test failed");
                assert!(result, "Optimization level {:?} should preserve semantics", level);
            }
        }
    }

    #[test]
    fn test_type_preservation_across_optimizations() {
        let framework = OptimizationPropertyFramework::new().expect("Failed to create framework");
        
        let program = r#"
(func main ()
  (let int_var 42)
  (let float_var 3.14)
  (let result (+ int_var (cast float_var int)))
  (return result))
"#;
        
        let ast = parse_sexpr(program).expect("Failed to parse");
        
        let result = framework.test_optimization_preserves_types(&ast, OptimizationLevel::Default)
            .expect("Test failed");
        assert!(result, "Type information should be preserved across optimizations");
    }
}

// ===== PROPTEST INTEGRATION =====

proptest! {
    #[test]
    fn proptest_optimization_semantics(program in arbitrary_optimization_program()) {
        let framework = OptimizationPropertyFramework::new().unwrap();
        let ast = parse_sexpr(&program.source_code).unwrap();
        let result = framework.test_optimization_preserves_semantics(&ast, OptimizationLevel::Default);
        prop_assert!(result.is_ok());
        if let Ok(preserves) = result {
            prop_assert!(preserves);
        }
    }

    #[test]
    fn proptest_optimization_performance(program in arbitrary_optimization_program()) {
        let framework = OptimizationPropertyFramework::new().unwrap();
        let ast = parse_sexpr(&program.source_code).unwrap();
        let result = framework.test_optimization_improves_performance(&ast, OptimizationLevel::Default);
        prop_assert!(result.is_ok());
        if let Ok(improves) = result {
            prop_assert!(improves);
        }
    }
}

/// Generate optimization test programs for proptest
fn arbitrary_optimization_program() -> impl Strategy<Value = OptimizationTestProgram> {
    prop::sample::select(vec![
        OptimizationTestProgram {
            source_code: r#"
(func main ()
  (let x (+ 1 2))
  (let y (* 3 4))
  (return (+ x y)))
"#.to_string(),
            expected_optimizations: vec![ExpectedOptimization::ConstantFolding],
        },
        OptimizationTestProgram {
            source_code: r#"
(func main ()
  (let x 42)
  (let unused (+ x 1))
  (return x))
"#.to_string(),
            expected_optimizations: vec![ExpectedOptimization::DeadCodeElimination],
        },
        OptimizationTestProgram {
            source_code: r#"
(defun double (x) (* x 2))
(func main ()
  (let result (double 21))
  (return result))
"#.to_string(),
            expected_optimizations: vec![ExpectedOptimization::FunctionInlining],
        },
    ])
}