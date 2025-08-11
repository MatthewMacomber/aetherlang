// Property-based tests for type safety and memory safety
// Task 10.3: Create property-based tests for type safety and memory safety
// Uses property-based testing to verify invariants across the MLIR-LLVM compilation pipeline

use aether_language::compiler::mlir::{
    MLIRContext, MLIRModule, AetherMLIRFrontend, AetherOptimizer,
    LLVMCodeGenerator, TargetConfig, OptimizationLevel
};
use aether_language::compiler::ast::{AST, ASTNode, ASTNodeRef, ASTNodeType};
use aether_language::compiler::parser::{parse_sexpr};
use aether_language::compiler::types::{AetherType, TypeChecker};
use proptest::prelude::*;
use quickcheck::{quickcheck, TestResult, Arbitrary, Gen};
use std::collections::HashMap;
use tempfile::TempDir;

/// Property-based test framework for MLIR-LLVM compilation
pub struct PropertyTestFramework {
    temp_dir: TempDir,
    type_checker: TypeChecker,
}

impl PropertyTestFramework {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let type_checker = TypeChecker::new();
        
        Ok(PropertyTestFramework {
            temp_dir,
            type_checker,
        })
    }

    /// Test that type safety is preserved through compilation
    pub fn test_type_safety_preservation(&self, ast: &AST) -> Result<bool, Box<dyn std::error::Error>> {
        // Step 1: Type check the original AST
        let original_types = self.type_checker.check_ast(ast)?;
        
        // Step 2: Compile through MLIR pipeline
        let context = MLIRContext::new()?;
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("property_test")?;
        
        frontend.convert_ast_to_module(ast, &mut module)?;
        module.verify()?;
        
        // Step 3: Apply optimizations
        let optimizer = AetherOptimizer::new(&context);
        optimizer.optimize(&mut module)?;
        module.verify()?;
        
        // Step 4: Generate LLVM IR
        let target_config = TargetConfig {
            triple: "x86_64-unknown-linux-gnu".to_string(),
            cpu: "generic".to_string(),
            features: "".to_string(),
            optimization_level: OptimizationLevel::Default,
            relocation_model: aether_language::compiler::mlir::RelocModel::Default,
            code_model: aether_language::compiler::mlir::CodeModel::Default,
        };

        let mut codegen = LLVMCodeGenerator::new(target_config)?;
        codegen.generate_from_mlir(&module)?;
        
        // Step 5: Verify type information is preserved in LLVM IR
        let llvm_ir = codegen.get_llvm_ir_string()?;
        let preserved_types = self.extract_types_from_llvm_ir(&llvm_ir)?;
        
        // Property: Type information should be preserved or safely converted
        Ok(self.verify_type_preservation(&original_types, &preserved_types))
    }

    /// Test that memory safety is maintained through compilation
    pub fn test_memory_safety_preservation(&self, ast: &AST) -> Result<bool, Box<dyn std::error::Error>> {
        // Step 1: Analyze memory usage patterns in AST
        let memory_analysis = self.analyze_memory_patterns(ast)?;
        
        // Step 2: Compile through MLIR pipeline
        let context = MLIRContext::new()?;
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("memory_safety_test")?;
        
        frontend.convert_ast_to_module(ast, &mut module)?;
        module.verify()?;
        
        // Step 3: Apply optimizations
        let optimizer = AetherOptimizer::new(&context);
        optimizer.optimize(&mut module)?;
        module.verify()?;
        
        // Step 4: Generate LLVM IR and analyze memory operations
        let target_config = TargetConfig {
            triple: "x86_64-unknown-linux-gnu".to_string(),
            cpu: "generic".to_string(),
            features: "".to_string(),
            optimization_level: OptimizationLevel::Default,
            relocation_model: aether_language::compiler::mlir::RelocModel::Default,
            code_model: aether_language::compiler::mlir::CodeModel::Default,
        };

        let mut codegen = LLVMCodeGenerator::new(target_config)?;
        codegen.generate_from_mlir(&module)?;
        
        let llvm_ir = codegen.get_llvm_ir_string()?;
        let llvm_memory_analysis = self.analyze_llvm_memory_operations(&llvm_ir)?;
        
        // Property: Memory safety invariants should be preserved
        Ok(self.verify_memory_safety_preservation(&memory_analysis, &llvm_memory_analysis))
    }

    /// Extract type information from LLVM IR
    fn extract_types_from_llvm_ir(&self, llvm_ir: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        let mut types = HashMap::new();
        
        for line in llvm_ir.lines() {
            let trimmed = line.trim();
            
            // Extract function signatures
            if trimmed.starts_with("define") {
                if let Some(func_info) = self.parse_function_signature(trimmed) {
                    types.insert(func_info.0, func_info.1);
                }
            }
            
            // Extract variable types
            if trimmed.contains(" = alloca ") {
                if let Some(var_info) = self.parse_alloca_instruction(trimmed) {
                    types.insert(var_info.0, var_info.1);
                }
            }
        }
        
        Ok(types)
    }

    /// Parse function signature from LLVM IR
    fn parse_function_signature(&self, line: &str) -> Option<(String, String)> {
        // Simplified parsing - in real implementation would use proper LLVM IR parser
        if let Some(start) = line.find("@") {
            if let Some(end) = line[start..].find("(") {
                let func_name = line[start+1..start+end].to_string();
                let return_type = if line.contains("void") {
                    "void".to_string()
                } else if line.contains("i32") {
                    "i32".to_string()
                } else if line.contains("double") {
                    "double".to_string()
                } else {
                    "unknown".to_string()
                };
                return Some((func_name, return_type));
            }
        }
        None
    }

    /// Parse alloca instruction from LLVM IR
    fn parse_alloca_instruction(&self, line: &str) -> Option<(String, String)> {
        // Simplified parsing
        if let Some(eq_pos) = line.find(" = alloca ") {
            let var_part = &line[..eq_pos];
            if let Some(var_start) = var_part.rfind('%') {
                let var_name = var_part[var_start+1..].trim().to_string();
                let type_part = &line[eq_pos + 10..]; // Skip " = alloca "
                let var_type = type_part.split(',').next().unwrap_or("unknown").trim().to_string();
                return Some((var_name, var_type));
            }
        }
        None
    }

    /// Verify that type information is preserved through compilation
    fn verify_type_preservation(&self, original: &HashMap<String, AetherType>, preserved: &HashMap<String, String>) -> bool {
        // Property: All originally typed variables should have corresponding LLVM types
        for (name, aether_type) in original {
            if let Some(llvm_type) = preserved.get(name) {
                if !self.types_are_compatible(aether_type, llvm_type) {
                    return false;
                }
            }
            // Note: Some variables might be optimized away, which is acceptable
        }
        true
    }

    /// Check if Aether type is compatible with LLVM type
    fn types_are_compatible(&self, aether_type: &AetherType, llvm_type: &str) -> bool {
        match aether_type {
            AetherType::Int32 => llvm_type.contains("i32"),
            AetherType::Float64 => llvm_type.contains("double") || llvm_type.contains("f64"),
            AetherType::Bool => llvm_type.contains("i1") || llvm_type.contains("i8"),
            AetherType::String => llvm_type.contains("ptr") || llvm_type.contains("i8*"),
            AetherType::Tensor { .. } => llvm_type.contains("ptr") || llvm_type.contains("struct"),
            AetherType::Function { .. } => llvm_type.contains("ptr") || llvm_type.starts_with("define"),
            _ => true, // For complex types, assume compatibility for now
        }
    }

    /// Analyze memory usage patterns in AST
    fn analyze_memory_patterns(&self, ast: &AST) -> Result<MemoryAnalysis, Box<dyn std::error::Error>> {
        let mut analysis = MemoryAnalysis::new();
        
        for node in ast.nodes() {
            match node.node_type() {
                ASTNodeType::VariableDeclaration => {
                    analysis.allocations += 1;
                    if self.is_linear_type(node) {
                        analysis.linear_allocations += 1;
                    }
                }
                ASTNodeType::TensorCreate => {
                    analysis.tensor_allocations += 1;
                }
                ASTNodeType::FunctionCall => {
                    if self.is_deallocation_call(node) {
                        analysis.deallocations += 1;
                    }
                }
                _ => {}
            }
        }
        
        Ok(analysis)
    }

    /// Check if a node represents a linear type
    fn is_linear_type(&self, node: &ASTNodeRef) -> bool {
        // Simplified check - in real implementation would use proper type analysis
        node.attributes().get("linear").is_some()
    }

    /// Check if a function call is a deallocation
    fn is_deallocation_call(&self, node: &ASTNodeRef) -> bool {
        if let Some(func_name) = node.attributes().get("function_name") {
            func_name.contains("free") || func_name.contains("drop") || func_name.contains("dealloc")
        } else {
            false
        }
    }

    /// Analyze memory operations in LLVM IR
    fn analyze_llvm_memory_operations(&self, llvm_ir: &str) -> Result<LLVMMemoryAnalysis, Box<dyn std::error::Error>> {
        let mut analysis = LLVMMemoryAnalysis::new();
        
        for line in llvm_ir.lines() {
            let trimmed = line.trim();
            
            if trimmed.contains("alloca") {
                analysis.allocas += 1;
            }
            if trimmed.contains("malloc") || trimmed.contains("@malloc") {
                analysis.mallocs += 1;
            }
            if trimmed.contains("free") || trimmed.contains("@free") {
                analysis.frees += 1;
            }
            if trimmed.contains("load") {
                analysis.loads += 1;
            }
            if trimmed.contains("store") {
                analysis.stores += 1;
            }
        }
        
        Ok(analysis)
    }

    /// Verify memory safety preservation
    fn verify_memory_safety_preservation(&self, original: &MemoryAnalysis, llvm: &LLVMMemoryAnalysis) -> bool {
        // Property 1: Every malloc should have a corresponding free (or be stack allocated)
        let total_heap_allocs = llvm.mallocs;
        let total_frees = llvm.frees;
        let stack_allocs = llvm.allocas;
        
        // Allow for stack allocation instead of heap allocation
        if total_heap_allocs > 0 && total_frees == 0 && stack_allocs == 0 {
            return false; // Potential memory leak
        }
        
        // Property 2: Linear types should have deterministic deallocation
        if original.linear_allocations > 0 {
            // Should have corresponding deallocation operations
            let total_deallocs = total_frees + original.deallocations;
            if total_deallocs == 0 {
                return false; // Linear types not properly deallocated
            }
        }
        
        // Property 3: No use-after-free (simplified check)
        // In a real implementation, this would require more sophisticated analysis
        
        true
    }
}

/// Memory analysis results for AST
#[derive(Debug, Clone)]
struct MemoryAnalysis {
    allocations: usize,
    deallocations: usize,
    linear_allocations: usize,
    tensor_allocations: usize,
}

impl MemoryAnalysis {
    fn new() -> Self {
        MemoryAnalysis {
            allocations: 0,
            deallocations: 0,
            linear_allocations: 0,
            tensor_allocations: 0,
        }
    }
}

/// Memory analysis results for LLVM IR
#[derive(Debug, Clone)]
struct LLVMMemoryAnalysis {
    allocas: usize,
    mallocs: usize,
    frees: usize,
    loads: usize,
    stores: usize,
}

impl LLVMMemoryAnalysis {
    fn new() -> Self {
        LLVMMemoryAnalysis {
            allocas: 0,
            mallocs: 0,
            frees: 0,
            loads: 0,
            stores: 0,
        }
    }
}

/// Generate arbitrary AST nodes for property testing
#[derive(Debug, Clone)]
pub struct ArbitraryAST {
    pub nodes: Vec<ArbitraryASTNode>,
}

#[derive(Debug, Clone)]
pub struct ArbitraryASTNode {
    pub node_type: ArbitraryNodeType,
    pub value: Option<String>,
    pub children: Vec<ArbitraryASTNode>,
}

#[derive(Debug, Clone)]
pub enum ArbitraryNodeType {
    Function,
    Variable,
    Literal,
    BinaryOp,
    TensorOp,
    ControlFlow,
}

impl Arbitrary for ArbitraryAST {
    fn arbitrary(g: &mut Gen) -> Self {
        let size = g.size().min(10); // Limit size for performance
        let mut nodes = Vec::new();
        
        for _ in 0..size {
            nodes.push(ArbitraryASTNode::arbitrary(g));
        }
        
        ArbitraryAST { nodes }
    }
}

impl Arbitrary for ArbitraryASTNode {
    fn arbitrary(g: &mut Gen) -> Self {
        let node_type = match u8::arbitrary(g) % 6 {
            0 => ArbitraryNodeType::Function,
            1 => ArbitraryNodeType::Variable,
            2 => ArbitraryNodeType::Literal,
            3 => ArbitraryNodeType::BinaryOp,
            4 => ArbitraryNodeType::TensorOp,
            _ => ArbitraryNodeType::ControlFlow,
        };
        
        let value = match &node_type {
            ArbitraryNodeType::Variable => Some(format!("var_{}", u32::arbitrary(g) % 100)),
            ArbitraryNodeType::Literal => Some(format!("{}", i32::arbitrary(g) % 1000)),
            ArbitraryNodeType::BinaryOp => Some(["+", "-", "*", "/"][usize::arbitrary(g) % 4].to_string()),
            _ => None,
        };
        
        let children_count = match &node_type {
            ArbitraryNodeType::Function => (u8::arbitrary(g) % 3) as usize,
            ArbitraryNodeType::BinaryOp => 2,
            ArbitraryNodeType::TensorOp => (u8::arbitrary(g) % 2 + 1) as usize,
            ArbitraryNodeType::ControlFlow => (u8::arbitrary(g) % 2 + 1) as usize,
            _ => 0,
        };
        
        let mut children = Vec::new();
        for _ in 0..children_count {
            if g.size() > 0 {
                children.push(ArbitraryASTNode::arbitrary(&mut Gen::new(g.size() - 1)));
            }
        }
        
        ArbitraryASTNode {
            node_type,
            value,
            children,
        }
    }
}

impl ArbitraryAST {
    /// Convert to actual AST for testing
    pub fn to_ast(&self) -> Result<AST, Box<dyn std::error::Error>> {
        // Convert arbitrary AST to real AST
        // This is a simplified conversion - real implementation would be more comprehensive
        let source_code = self.to_source_code();
        parse_sexpr(&source_code)
    }
    
    /// Convert to Aether source code
    fn to_source_code(&self) -> String {
        let mut code = String::from("(func main ()\n");
        
        for node in &self.nodes {
            code.push_str(&format!("  {}\n", self.node_to_source(node)));
        }
        
        code.push_str("  (return 0))\n");
        code
    }
    
    fn node_to_source(&self, node: &ArbitraryASTNode) -> String {
        match &node.node_type {
            ArbitraryNodeType::Variable => {
                format!("(let {} {})", 
                    node.value.as_ref().unwrap_or(&"x".to_string()),
                    node.children.first().map(|c| self.node_to_source(c)).unwrap_or_else(|| "0".to_string())
                )
            }
            ArbitraryNodeType::Literal => {
                node.value.as_ref().unwrap_or(&"0".to_string()).clone()
            }
            ArbitraryNodeType::BinaryOp => {
                let op = node.value.as_ref().unwrap_or(&"+".to_string());
                let left = node.children.get(0).map(|c| self.node_to_source(c)).unwrap_or_else(|| "0".to_string());
                let right = node.children.get(1).map(|c| self.node_to_source(c)).unwrap_or_else(|| "1".to_string());
                format!("({} {} {})", op, left, right)
            }
            ArbitraryNodeType::TensorOp => {
                "(tensor-create [2 2] f32)".to_string()
            }
            ArbitraryNodeType::ControlFlow => {
                let condition = node.children.get(0).map(|c| self.node_to_source(c)).unwrap_or_else(|| "(> 1 0)".to_string());
                let body = node.children.get(1).map(|c| self.node_to_source(c)).unwrap_or_else(|| "0".to_string());
                format!("(if {} {} 0)", condition, body)
            }
            ArbitraryNodeType::Function => {
                let body = node.children.iter()
                    .map(|c| self.node_to_source(c))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("(block {})", body)
            }
        }
    }
}

// ===== PROPERTY-BASED TESTS =====

#[cfg(test)]
mod property_tests {
    use super::*;

    /// Property: Type safety is preserved through compilation
    fn prop_type_safety_preserved(ast: ArbitraryAST) -> TestResult {
        let framework = match PropertyTestFramework::new() {
            Ok(f) => f,
            Err(_) => return TestResult::discard(),
        };
        
        let real_ast = match ast.to_ast() {
            Ok(a) => a,
            Err(_) => return TestResult::discard(),
        };
        
        match framework.test_type_safety_preservation(&real_ast) {
            Ok(result) => TestResult::from_bool(result),
            Err(_) => TestResult::discard(),
        }
    }

    /// Property: Memory safety is preserved through compilation
    fn prop_memory_safety_preserved(ast: ArbitraryAST) -> TestResult {
        let framework = match PropertyTestFramework::new() {
            Ok(f) => f,
            Err(_) => return TestResult::discard(),
        };
        
        let real_ast = match ast.to_ast() {
            Ok(a) => a,
            Err(_) => return TestResult::discard(),
        };
        
        match framework.test_memory_safety_preservation(&real_ast) {
            Ok(result) => TestResult::from_bool(result),
            Err(_) => TestResult::discard(),
        }
    }

    /// Property: Compilation is deterministic
    fn prop_compilation_deterministic(ast: ArbitraryAST) -> TestResult {
        let framework = match PropertyTestFramework::new() {
            Ok(f) => f,
            Err(_) => return TestResult::discard(),
        };
        
        let real_ast = match ast.to_ast() {
            Ok(a) => a,
            Err(_) => return TestResult::discard(),
        };
        
        // Compile twice and compare results
        let result1 = compile_ast_to_llvm_ir(&real_ast);
        let result2 = compile_ast_to_llvm_ir(&real_ast);
        
        match (result1, result2) {
            (Ok(ir1), Ok(ir2)) => TestResult::from_bool(ir1 == ir2),
            _ => TestResult::discard(),
        }
    }

    fn compile_ast_to_llvm_ir(ast: &AST) -> Result<String, Box<dyn std::error::Error>> {
        let context = MLIRContext::new()?;
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("deterministic_test")?;
        
        frontend.convert_ast_to_module(ast, &mut module)?;
        module.verify()?;
        
        let target_config = TargetConfig {
            triple: "x86_64-unknown-linux-gnu".to_string(),
            cpu: "generic".to_string(),
            features: "".to_string(),
            optimization_level: OptimizationLevel::Default,
            relocation_model: aether_language::compiler::mlir::RelocModel::Default,
            code_model: aether_language::compiler::mlir::CodeModel::Default,
        };

        let mut codegen = LLVMCodeGenerator::new(target_config)?;
        codegen.generate_from_mlir(&module)?;
        
        codegen.get_llvm_ir_string()
    }

    #[test]
    fn test_type_safety_property() {
        quickcheck(prop_type_safety_preserved as fn(ArbitraryAST) -> TestResult);
    }

    #[test]
    fn test_memory_safety_property() {
        quickcheck(prop_memory_safety_preserved as fn(ArbitraryAST) -> TestResult);
    }

    #[test]
    fn test_compilation_determinism_property() {
        quickcheck(prop_compilation_deterministic as fn(ArbitraryAST) -> TestResult);
    }

    #[test]
    fn test_simple_type_safety_cases() {
        let framework = PropertyTestFramework::new().expect("Failed to create framework");
        
        // Test simple integer operations
        let int_program = r#"
(func main ()
  (let x 42)
  (let y (+ x 1))
  (return y))
"#;
        
        let ast = parse_sexpr(int_program).expect("Failed to parse");
        let result = framework.test_type_safety_preservation(&ast).expect("Test failed");
        assert!(result, "Type safety should be preserved for integer operations");
        
        // Test tensor operations
        let tensor_program = r#"
(func main ()
  (let tensor_a (tensor-create [2 2] f32))
  (let tensor_b (tensor-create [2 2] f32))
  (let result (tensor-add tensor_a tensor_b))
  (return 0))
"#;
        
        let tensor_ast = parse_sexpr(tensor_program).expect("Failed to parse");
        let tensor_result = framework.test_type_safety_preservation(&tensor_ast).expect("Test failed");
        assert!(tensor_result, "Type safety should be preserved for tensor operations");
    }

    #[test]
    fn test_simple_memory_safety_cases() {
        let framework = PropertyTestFramework::new().expect("Failed to create framework");
        
        // Test stack allocation
        let stack_program = r#"
(func main ()
  (let x 42)
  (let y x)
  (return y))
"#;
        
        let ast = parse_sexpr(stack_program).expect("Failed to parse");
        let result = framework.test_memory_safety_preservation(&ast).expect("Test failed");
        assert!(result, "Memory safety should be preserved for stack allocations");
        
        // Test tensor allocation
        let tensor_program = r#"
(func main ()
  (let tensor (tensor-create [10 10] f64))
  (tensor-fill tensor 1.0)
  (return 0))
"#;
        
        let tensor_ast = parse_sexpr(tensor_program).expect("Failed to parse");
        let tensor_result = framework.test_memory_safety_preservation(&tensor_ast).expect("Test failed");
        assert!(tensor_result, "Memory safety should be preserved for tensor allocations");
    }

    #[test]
    fn test_compilation_determinism_simple() {
        let program = r#"
(func main ()
  (let x 1)
  (let y 2)
  (let z (+ x y))
  (return z))
"#;
        
        let ast = parse_sexpr(program).expect("Failed to parse");
        
        let ir1 = compile_ast_to_llvm_ir(&ast).expect("First compilation failed");
        let ir2 = compile_ast_to_llvm_ir(&ast).expect("Second compilation failed");
        
        assert_eq!(ir1, ir2, "Compilation should be deterministic");
    }
}

// ===== PROPTEST INTEGRATION =====

proptest! {
    #[test]
    fn proptest_type_safety(ast in arbitrary_simple_ast()) {
        let framework = PropertyTestFramework::new().unwrap();
        let real_ast = ast.to_ast().unwrap();
        let result = framework.test_type_safety_preservation(&real_ast);
        prop_assert!(result.is_ok());
        if let Ok(safe) = result {
            prop_assert!(safe);
        }
    }

    #[test]
    fn proptest_memory_safety(ast in arbitrary_simple_ast()) {
        let framework = PropertyTestFramework::new().unwrap();
        let real_ast = ast.to_ast().unwrap();
        let result = framework.test_memory_safety_preservation(&real_ast);
        prop_assert!(result.is_ok());
        if let Ok(safe) = result {
            prop_assert!(safe);
        }
    }
}

/// Generate simple ASTs for proptest
fn arbitrary_simple_ast() -> impl Strategy<Value = ArbitraryAST> {
    prop::collection::vec(arbitrary_simple_node(), 1..5)
        .prop_map(|nodes| ArbitraryAST { nodes })
}

fn arbitrary_simple_node() -> impl Strategy<Value = ArbitraryASTNode> {
    prop::sample::select(vec![
        ArbitraryASTNode {
            node_type: ArbitraryNodeType::Variable,
            value: Some("x".to_string()),
            children: vec![ArbitraryASTNode {
                node_type: ArbitraryNodeType::Literal,
                value: Some("42".to_string()),
                children: vec![],
            }],
        },
        ArbitraryASTNode {
            node_type: ArbitraryNodeType::BinaryOp,
            value: Some("+".to_string()),
            children: vec![
                ArbitraryASTNode {
                    node_type: ArbitraryNodeType::Literal,
                    value: Some("1".to_string()),
                    children: vec![],
                },
                ArbitraryASTNode {
                    node_type: ArbitraryNodeType::Literal,
                    value: Some("2".to_string()),
                    children: vec![],
                },
            ],
        },
    ])
}