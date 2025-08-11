// Tests for MLIR-based compilation pipeline
// Covers AST to MLIR conversion, lowering, and optimization

use aether_language::compiler::ast::{AST, ASTNode, ASTNodeRef, AtomValue};
use aether_language::compiler::mlir::{MLIRPipeline, AetherFrontend, AetherLowering, AetherOptimizer, MockMLIRContext};

#[test]
fn test_mlir_pipeline_creation() {
    let pipeline = MLIRPipeline::new();
    assert!(pipeline.context().is_registered_operation("builtin.module"));
}

#[test]
fn test_ast_to_mlir_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Create simple AST: (+ 1.0 2.0)
    let ast = AST::new(ASTNode::list(vec![
        ASTNodeRef::direct(ASTNode::symbol("+".to_string())),
        ASTNodeRef::direct(ASTNode::number(1.0)),
        ASTNodeRef::direct(ASTNode::number(2.0)),
    ]));

    let result = pipeline.compile_ast(&ast);
    assert!(result.is_ok(), "AST to MLIR conversion should succeed");
}

#[test]
fn test_function_definition_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Create function definition AST: (defun add (x y) (+ x y))
    let ast = AST::new(ASTNode::list(vec![
        ASTNodeRef::direct(ASTNode::symbol("defun".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("add".to_string())),
        ASTNodeRef::direct(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("y".to_string())),
        ])),
        ASTNodeRef::direct(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("+".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("y".to_string())),
        ])),
    ]));

    let result = pipeline.compile_ast(&ast);
    assert!(result.is_ok(), "Function definition conversion should succeed");
}

#[test]
fn test_variable_declaration_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Create variable declaration AST: (let x 42.0)
    let ast = AST::new(ASTNode::list(vec![
        ASTNodeRef::direct(ASTNode::symbol("let".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
        ASTNodeRef::direct(ASTNode::number(42.0)),
    ]));

    let result = pipeline.compile_ast(&ast);
    assert!(result.is_ok(), "Variable declaration conversion should succeed");
}

#[test]
fn test_tensor_operation_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Create tensor operation AST: (tensor matmul A B)
    let ast = AST::new(ASTNode::list(vec![
        ASTNodeRef::direct(ASTNode::symbol("tensor".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("matmul".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("A".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("B".to_string())),
    ]));

    let result = pipeline.compile_ast(&ast);
    assert!(result.is_ok(), "Tensor operation conversion should succeed");
}

#[test]
fn test_autodiff_marker_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Create autodiff marker AST: (autodiff my_function)
    let ast = AST::new(ASTNode::list(vec![
        ASTNodeRef::direct(ASTNode::symbol("autodiff".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("my_function".to_string())),
    ]));

    let result = pipeline.compile_ast(&ast);
    assert!(result.is_ok(), "Autodiff marker conversion should succeed");
}

#[test]
fn test_probabilistic_variable_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Create probabilistic variable AST: (prob-var weight Normal 0.0 1.0)
    let ast = AST::new(ASTNode::list(vec![
        ASTNodeRef::direct(ASTNode::symbol("prob-var".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("weight".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("Normal".to_string())),
        ASTNodeRef::direct(ASTNode::number(0.0)),
        ASTNodeRef::direct(ASTNode::number(1.0)),
    ]));

    let result = pipeline.compile_ast(&ast);
    assert!(result.is_ok(), "Probabilistic variable conversion should succeed");
}

#[test]
fn test_linear_type_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Create linear type AST: (linear Float64)
    let ast = AST::new(ASTNode::list(vec![
        ASTNodeRef::direct(ASTNode::symbol("linear".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("Float64".to_string())),
    ]));

    let result = pipeline.compile_ast(&ast);
    assert!(result.is_ok(), "Linear type conversion should succeed");
}

#[test]
fn test_arithmetic_operations_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Test all arithmetic operations
    let operations = vec!["+", "-", "*", "/"];
    
    for op in operations {
        let ast = AST::new(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol(op.to_string())),
            ASTNodeRef::direct(ASTNode::number(3.14)),
            ASTNodeRef::direct(ASTNode::number(2.71)),
        ]));

        let result = pipeline.compile_ast(&ast);
        assert!(result.is_ok(), "Arithmetic operation {} conversion should succeed", op);
    }
}

#[test]
fn test_nested_expressions_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Create nested expression AST: (+ (* 2.0 3.0) (/ 8.0 4.0))
    let ast = AST::new(ASTNode::list(vec![
        ASTNodeRef::direct(ASTNode::symbol("+".to_string())),
        ASTNodeRef::direct(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("*".to_string())),
            ASTNodeRef::direct(ASTNode::number(2.0)),
            ASTNodeRef::direct(ASTNode::number(3.0)),
        ])),
        ASTNodeRef::direct(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("/".to_string())),
            ASTNodeRef::direct(ASTNode::number(8.0)),
            ASTNodeRef::direct(ASTNode::number(4.0)),
        ])),
    ]));

    let result = pipeline.compile_ast(&ast);
    assert!(result.is_ok(), "Nested expressions conversion should succeed");
}

#[test]
fn test_lowering_to_standard_dialects() {
    let pipeline = MLIRPipeline::new();
    
    // Create simple AST and convert to MLIR
    let ast = AST::new(ASTNode::list(vec![
        ASTNodeRef::direct(ASTNode::symbol("tensor".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("add".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("A".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("B".to_string())),
    ]));

    let mut module = pipeline.compile_ast(&ast).expect("AST conversion should succeed");
    
    // Test lowering to standard dialects
    let result = pipeline.lower_to_standard(&mut module);
    assert!(result.is_ok(), "Lowering to standard dialects should succeed");
}

#[test]
fn test_frontend_error_handling() {
    let context = MockMLIRContext::new();
    let mut frontend = AetherFrontend::new(&context);
    
    // Test error handling for invalid function definition
    let invalid_ast = AST::new(ASTNode::list(vec![
        ASTNodeRef::direct(ASTNode::symbol("defun".to_string())),
        // Missing required arguments
    ]));

    let result = frontend.convert_ast(&invalid_ast);
    assert!(result.is_err(), "Invalid AST should produce error");
}

#[test]
fn test_lowering_error_handling() {
    let context = MockMLIRContext::new();
    let lowering = AetherLowering::new(&context);
    
    // Test with empty module
    let mut module = aether_language::compiler::mlir::MockMLIRModule::new();
    
    let result = lowering.lower_to_standard(&mut module);
    assert!(result.is_ok(), "Lowering empty module should succeed");
}

#[test]
fn test_optimization_passes() {
    let context = MockMLIRContext::new();
    let optimizer = AetherOptimizer::new(&context);
    
    // Test with empty module
    let mut module = aether_language::compiler::mlir::MockMLIRModule::new();
    
    let result = optimizer.optimize(&mut module);
    assert!(result.is_ok(), "Optimization passes should succeed on empty module");
}

#[test]
fn test_complex_program_compilation() {
    let pipeline = MLIRPipeline::new();
    
    // Create complex program with multiple constructs
    let ast = AST::new(ASTNode::list(vec![
        ASTNodeRef::direct(ASTNode::symbol("defun".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("neural_network".to_string())),
        ASTNodeRef::direct(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("input".to_string())),
        ])),
        ASTNodeRef::direct(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("let".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("weights".to_string())),
            ASTNodeRef::direct(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol("prob-var".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("w".to_string())),
                ASTNodeRef::direct(ASTNode::symbol("Normal".to_string())),
                ASTNodeRef::direct(ASTNode::number(0.0)),
                ASTNodeRef::direct(ASTNode::number(1.0)),
            ])),
        ])),
    ]));

    let result = pipeline.compile_ast(&ast);
    assert!(result.is_ok(), "Complex program compilation should succeed");
}

#[test]
fn test_graph_structure_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Create AST with graph structure
    let mut ast = AST::new(ASTNode::Graph {
        nodes: vec![
            ASTNodeRef::direct(ASTNode::symbol("node1".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("node2".to_string())),
        ],
        edges: vec![],
        labels: std::collections::HashMap::new(),
    });

    let result = pipeline.compile_ast(&ast);
    assert!(result.is_ok(), "Graph structure conversion should succeed");
}

#[test]
fn test_boolean_and_nil_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Test boolean conversion
    let bool_ast = AST::new(ASTNode::boolean(true));
    let result = pipeline.compile_ast(&bool_ast);
    assert!(result.is_ok(), "Boolean conversion should succeed");
    
    // Test nil conversion
    let nil_ast = AST::new(ASTNode::nil());
    let result = pipeline.compile_ast(&nil_ast);
    assert!(result.is_ok(), "Nil conversion should succeed");
}

#[test]
fn test_string_literal_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Test string literal conversion
    let string_ast = AST::new(ASTNode::string("hello world".to_string()));
    let result = pipeline.compile_ast(&string_ast);
    assert!(result.is_ok(), "String literal conversion should succeed");
}

#[test]
fn test_empty_list_conversion() {
    let pipeline = MLIRPipeline::new();
    
    // Test empty list conversion
    let empty_list_ast = AST::new(ASTNode::empty_list());
    let result = pipeline.compile_ast(&empty_list_ast);
    assert!(result.is_ok(), "Empty list conversion should succeed");
}