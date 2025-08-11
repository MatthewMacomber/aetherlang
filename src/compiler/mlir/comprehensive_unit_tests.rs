// Comprehensive unit tests for all MLIR components
// Task 10.1: Implement unit tests for all components

use super::*;
use super::mlir_context::{MLIRContext, MLIRModule, MLIROperation, MLIRValue, MLIRType, MLIRAttribute};
use super::dialect::{AetherOps, AetherOperationBuilder, DistributionType};
use super::frontend::AetherMLIRFrontend;
use super::lowering::{AetherToStandardLowering, StandardToLLVMLowering, DialectLowering};
use super::optimization::{OptimizationPass, AetherOptimizer};
use super::llvm_codegen::{LLVMCodeGenerator, TargetConfig, OptimizationLevel};
use crate::compiler::ast::{AST, ASTNode, ASTNodeRef, AtomValue};
use std::collections::HashMap;

// ===== MLIR DIALECT OPERATIONS AND TYPES TESTS =====

#[cfg(test)]
mod dialect_tests {
    use super::*;

    fn create_test_context() -> MLIRContext {
        MLIRContext::new_mock()
    }

    #[test]
    fn test_tensor_create_operation() {
        let context = create_test_context();
        let element_type = MLIRType::Float { width: 32 };
        let shape = vec![2, 3, 4];
        let device = "cpu";
        let is_differentiable = true;

        let result = AetherOps::tensor_create(&context, element_type.clone(), &shape, device, is_differentiable);
        
        assert!(result.is_ok());
        let op = result.unwrap();
        assert_eq!(op.name, "aether.tensor_create");
        
        // Check shape attribute
        if let Some(MLIRAttribute::Array(shape_attr)) = op.attributes.get("shape") {
            assert_eq!(shape_attr.len(), 3);
            for (i, attr) in shape_attr.iter().enumerate() {
                if let MLIRAttribute::Integer(dim) = attr {
                    assert_eq!(*dim, shape[i]);
                }
            }
        } else {
            panic!("Expected shape attribute");
        }
        
        // Check device attribute
        assert_eq!(op.attributes.get("device"), Some(&MLIRAttribute::String("cpu".to_string())));
        
        // Check differentiability attribute
        assert_eq!(op.attributes.get("is_differentiable"), Some(&MLIRAttribute::Boolean(true)));
        
        // Check result type
        assert_eq!(op.results.len(), 1);
        match &op.results[0].value_type {
            MLIRType::AetherTensor { element_type: elem, shape: s, device: d } => {
                // Check element type is float with width 32
                match elem.as_ref() {
                    MLIRType::Float { width } => assert_eq!(*width, 32),
                    _ => panic!("Expected Float element type"),
                }
                assert_eq!(*s, vec![2, 3, 4]);
                assert_eq!(*d, "cpu");
            }
            _ => panic!("Expected AetherTensor type"),
        }
    }

    #[test]
    fn test_matrix_multiplication_operation() {
        let context = create_test_context();
        let lhs = MLIRValue::new("lhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 4],
            device: "cpu".to_string(),
        });
        let rhs = MLIRValue::new("rhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![4, 5],
            device: "cpu".to_string(),
        });
        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 5],
            device: "cpu".to_string(),
        };

        let result = AetherOps::matmul(&context, lhs, rhs, result_type, false, false);
        
        assert!(result.is_ok());
        let op = result.unwrap();
        assert_eq!(op.name, "aether.matmul");
        assert_eq!(op.operands.len(), 2);
        assert_eq!(op.results.len(), 1);
        
        // Check transpose attributes
        assert_eq!(op.attributes.get("transpose_a"), Some(&MLIRAttribute::Boolean(false)));
        assert_eq!(op.attributes.get("transpose_b"), Some(&MLIRAttribute::Boolean(false)));
    }

    #[test]
    fn test_automatic_differentiation_operations() {
        let context = create_test_context();
        let function = MLIRValue::new("func".to_string(), MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 32 }],
            outputs: vec![MLIRType::Float { width: 32 }],
        });
        let input = MLIRValue::new("input".to_string(), MLIRType::Float { width: 32 });
        let tangent = MLIRValue::new("tangent".to_string(), MLIRType::Float { width: 32 });
        let result_type = MLIRType::Float { width: 32 };

        // Test forward mode
        let forward_result = AetherOps::autodiff_forward(&context, function.clone(), input.clone(), tangent, result_type.clone());
        assert!(forward_result.is_ok());
        let forward_op = forward_result.unwrap();
        assert_eq!(forward_op.name, "aether.autodiff_forward");
        assert_eq!(forward_op.operands.len(), 3);
        assert_eq!(forward_op.attributes.get("mode"), Some(&MLIRAttribute::String("forward".to_string())));

        // Test reverse mode
        let reverse_result = AetherOps::autodiff_reverse(&context, function, input, result_type);
        assert!(reverse_result.is_ok());
        let reverse_op = reverse_result.unwrap();
        assert_eq!(reverse_op.name, "aether.autodiff_reverse");
        assert_eq!(reverse_op.operands.len(), 2);
        assert_eq!(reverse_op.attributes.get("mode"), Some(&MLIRAttribute::String("reverse".to_string())));
    }

    #[test]
    fn test_probabilistic_programming_operations() {
        let context = create_test_context();
        let distribution = DistributionType::Normal { mean: 0.0, std: 1.0 };
        let parameters = vec![
            MLIRValue::new("mean".to_string(), MLIRType::Float { width: 64 }),
            MLIRValue::new("std".to_string(), MLIRType::Float { width: 64 }),
        ];
        let result_type = MLIRType::AetherProbabilistic {
            distribution: "normal".to_string(),
            inner_type: Box::new(MLIRType::Float { width: 64 }),
        };

        let result = AetherOps::prob_var(&context, "x", distribution, parameters, result_type);
        
        assert!(result.is_ok());
        let op = result.unwrap();
        assert_eq!(op.name, "aether.prob_var");
        assert_eq!(op.attributes.get("name"), Some(&MLIRAttribute::String("x".to_string())));
        
        // Check distribution attribute
        if let Some(MLIRAttribute::Dictionary(dist_dict)) = op.attributes.get("distribution") {
            assert_eq!(dist_dict.get("type"), Some(&MLIRAttribute::String("normal".to_string())));
            assert_eq!(dist_dict.get("mean"), Some(&MLIRAttribute::Float(0.0)));
            assert_eq!(dist_dict.get("std"), Some(&MLIRAttribute::Float(1.0)));
        } else {
            panic!("Expected distribution dictionary attribute");
        }
    }

    #[test]
    fn test_linear_type_operations() {
        let context = create_test_context();
        let inner_type = MLIRType::Float { width: 32 };
        let allocation_site = "test_site";

        // Test linear allocation
        let alloc_result = AetherOps::linear_alloc(&context, inner_type.clone(), None, allocation_site);
        assert!(alloc_result.is_ok());
        let alloc_op = alloc_result.unwrap();
        assert_eq!(alloc_op.name, "aether.linear_alloc");
        assert_eq!(alloc_op.attributes.get("allocation_site"), 
            Some(&MLIRAttribute::String("test_site".to_string())));

        // Test linear move
        let source = MLIRValue::new("source".to_string(), MLIRType::AetherLinear {
            inner_type: Box::new(inner_type.clone()),
        });
        let result_type = MLIRType::AetherLinear {
            inner_type: Box::new(inner_type.clone()),
        };
        
        let move_result = AetherOps::linear_move(&context, source, result_type);
        assert!(move_result.is_ok());
        let move_op = move_result.unwrap();
        assert_eq!(move_op.name, "aether.linear_move");
        assert_eq!(move_op.operands.len(), 1);

        // Test linear drop
        let value = MLIRValue::new("value".to_string(), MLIRType::AetherLinear {
            inner_type: Box::new(inner_type),
        });
        
        let drop_result = AetherOps::linear_drop(&context, value);
        assert!(drop_result.is_ok());
        let drop_op = drop_result.unwrap();
        assert_eq!(drop_op.name, "aether.linear_drop");
        assert_eq!(drop_op.operands.len(), 1);
        assert_eq!(drop_op.results.len(), 0); // Drop has no results
    }

    #[test]
    fn test_concurrency_operations() {
        let context = create_test_context();
        let actor_type = MLIRType::Function {
            inputs: vec![MLIRType::Integer { width: 32, signed: true }],
            outputs: vec![MLIRType::Integer { width: 32, signed: true }],
        };
        let result_type = MLIRType::Pointer;

        // Test actor spawn
        let spawn_result = AetherOps::spawn_actor(&context, actor_type, None, result_type);
        assert!(spawn_result.is_ok());
        let spawn_op = spawn_result.unwrap();
        assert_eq!(spawn_op.name, "aether.spawn_actor");
        assert_eq!(spawn_op.results.len(), 1);

        // Test message send
        let actor_ref = MLIRValue::new("actor".to_string(), MLIRType::Pointer);
        let message = MLIRValue::new("msg".to_string(), MLIRType::Integer { width: 32, signed: true });
        
        let send_result = AetherOps::send_message(&context, actor_ref, message);
        assert!(send_result.is_ok());
        let send_op = send_result.unwrap();
        assert_eq!(send_op.name, "aether.send_message");
        assert_eq!(send_op.operands.len(), 2);
        assert_eq!(send_op.results.len(), 0); // Send has no results

        // Test parallel for
        let lower = MLIRValue::new("lower".to_string(), MLIRType::Index);
        let upper = MLIRValue::new("upper".to_string(), MLIRType::Index);
        let body = MLIRValue::new("body".to_string(), MLIRType::Function {
            inputs: vec![MLIRType::Index],
            outputs: vec![],
        });
        
        let parallel_result = AetherOps::parallel_for(&context, lower, upper, None, body);
        assert!(parallel_result.is_ok());
        let parallel_op = parallel_result.unwrap();
        assert_eq!(parallel_op.name, "aether.parallel_for");
        assert_eq!(parallel_op.operands.len(), 3); // lower, upper, body
    }

    #[test]
    fn test_operation_builder_validation() {
        let context = create_test_context();
        let builder = AetherOperationBuilder::new(&context);

        // Test valid tensor creation
        let valid_result = builder.build_tensor_create(
            MLIRType::Float { width: 32 },
            &[2, 3, 4],
            "cpu",
            false
        );
        assert!(valid_result.is_ok());

        // Test invalid tensor creation (empty shape)
        let invalid_result = builder.build_tensor_create(
            MLIRType::Float { width: 32 },
            &[],
            "cpu",
            false
        );
        assert!(invalid_result.is_err());

        // Test invalid device
        let invalid_device_result = builder.build_tensor_create(
            MLIRType::Float { width: 32 },
            &[2, 3],
            "invalid_device",
            false
        );
        assert!(invalid_device_result.is_err());
    }
}

// ===== FRONTEND CONVERSION TESTS FOR ALL AST NODE TYPES =====

#[cfg(test)]
mod frontend_conversion_tests {
    use super::*;

    fn create_test_context() -> MLIRContext {
        MLIRContext::new_mock()
    }

    fn create_simple_ast(root: ASTNode) -> AST {
        AST::new(root)
    }

    #[test]
    fn test_atom_conversions() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").unwrap();

        // Test number atom conversion through AST
        let number_ast = AST::new(ASTNode::number(42.5));
        let result = frontend.convert_ast_to_module(&number_ast, &mut module);
        assert!(result.is_ok());

        // Test string atom conversion through AST
        let string_ast = AST::new(ASTNode::string("hello world".to_string()));
        let result = frontend.convert_ast_to_module(&string_ast, &mut module);
        assert!(result.is_ok());

        // Test boolean atom conversion through AST
        let bool_ast = AST::new(ASTNode::boolean(true));
        let result = frontend.convert_ast_to_module(&bool_ast, &mut module);
        assert!(result.is_ok());

        // Test symbol atom conversion through AST
        let symbol_ast = AST::new(ASTNode::symbol("variable_name".to_string()));
        let result = frontend.convert_ast_to_module(&symbol_ast, &mut module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_function_definition_conversion() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").unwrap();

        // Create function definition: (defun add (x y) (+ x y))
        let args = vec![
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
        ];

        // Convert through AST instead of calling private method
        let func_ast = AST::new(ASTNode::list(vec![
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
        
        let result = frontend.convert_ast_to_module(&func_ast, &mut module);
        assert!(result.is_ok());

        // Verify function operation was added
        let operations = module.operations();
        let func_ops: Vec<_> = operations.iter()
            .filter(|op| op.name == "func.func")
            .collect();
        assert!(!func_ops.is_empty());
    }

    #[test]
    fn test_variable_declaration_conversion() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").unwrap();

        // Create variable declaration: (let x 42)
        let args = vec![
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ASTNodeRef::direct(ASTNode::number(42.0)),
        ];

        // Convert through AST instead of calling private method
        let var_ast = AST::new(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("let".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("x".to_string())),
            ASTNodeRef::direct(ASTNode::number(42.0)),
        ]));
        
        let result = frontend.convert_ast_to_module(&var_ast, &mut module);
        assert!(result.is_ok());

        // Verify variable operation was added
        let operations = module.operations();
        let var_ops: Vec<_> = operations.iter()
            .filter(|op| op.name == "aether.var")
            .collect();
        assert!(!var_ops.is_empty());
    }

    #[test]
    fn test_arithmetic_operation_conversions() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").unwrap();

        let arithmetic_ops = vec!["+", "-", "*", "/", "%"];
        
        for op in arithmetic_ops {
            let arith_ast = AST::new(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol(op.to_string())),
                ASTNodeRef::direct(ASTNode::number(1.0)),
                ASTNodeRef::direct(ASTNode::number(2.0)),
            ]));
            
            let result = frontend.convert_ast_to_module(&arith_ast, &mut module);
            assert!(result.is_ok(), "Failed to convert arithmetic operation: {}", op);
        }
    }

    #[test]
    fn test_comparison_operation_conversions() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").unwrap();

        let comparison_ops = vec!["=", "==", "!=", "<", ">", "<=", ">="];
        
        for op in comparison_ops {
            let comp_ast = AST::new(ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol(op.to_string())),
                ASTNodeRef::direct(ASTNode::number(1.0)),
                ASTNodeRef::direct(ASTNode::number(2.0)),
            ]));
            
            let result = frontend.convert_ast_to_module(&comp_ast, &mut module);
            assert!(result.is_ok(), "Failed to convert comparison operation: {}", op);
        }
    }

    #[test]
    fn test_complex_nested_expressions() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").unwrap();

        // Test nested expression: (let result (+ (* 2 3) (/ 8 4)))
        let nested_expr = ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("let".to_string())),
            ASTNodeRef::direct(ASTNode::symbol("result".to_string())),
            ASTNodeRef::direct(ASTNode::list(vec![
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
            ])),
        ]);

        let result = frontend.traverse_node(&nested_expr, &mut module);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());

        // Verify multiple operations were generated
        assert!(module.operations().len() >= 3); // At least multiplication, division, and addition
    }
}

// ===== OPTIMIZATION PASS CORRECTNESS TESTS =====

#[cfg(test)]
mod optimization_tests {
    use super::*;

    fn create_test_context() -> MLIRContext {
        MLIRContext::new_mock()
    }

    #[test]
    fn test_aether_optimizer_creation() {
        let context = create_test_context();
        let optimizer = AetherOptimizer::new(&context);
        
        // Test that optimizer was created successfully
        // (Internal state is private, so we can't test much here)
    }

    #[test]
    fn test_optimization_on_simple_module() {
        let context = create_test_context();
        let mut module = context.create_module("test").unwrap();

        // Add a simple operation
        let mut op = MLIROperation::new("aether.tensor_op".to_string());
        op.add_attribute("op_name".to_string(), MLIRAttribute::String("add".to_string()));
        module.add_operation(op).unwrap();

        let initial_op_count = module.operations().len();
        assert_eq!(initial_op_count, 1);

        // Apply optimization
        let optimizer = AetherOptimizer::new(&context);
        let result = optimizer.optimize(&mut module);
        assert!(result.is_ok());

        // Module should still have operations (may be modified)
        assert!(!module.operations().is_empty());
    }
}

// ===== LOWERING AND CODE GENERATION UNIT TESTS =====

#[cfg(test)]
mod lowering_and_codegen_tests {
    use super::*;

    fn create_test_context() -> MLIRContext {
        MLIRContext::new_mock()
    }

    #[test]
    fn test_aether_to_standard_lowering_creation() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        assert_eq!(lowering.get_source_dialect(), "aether");
        assert_eq!(lowering.get_target_dialect(), "linalg");
    }

    #[test]
    fn test_standard_to_llvm_lowering_creation() {
        let context = create_test_context();
        let lowering = StandardToLLVMLowering::new(&context);
        
        assert_eq!(lowering.get_source_dialect(), "standard");
        assert_eq!(lowering.get_target_dialect(), "llvm");
    }

    #[test]
    fn test_tensor_create_lowering() {
        let context = create_test_context();
        let lowering = AetherToStandardLowering::new(&context);
        
        let mut op = MLIROperation::new("aether.tensor_create".to_string());
        op.add_attribute("shape".to_string(), MLIRAttribute::Array(vec![
            MLIRAttribute::Integer(3),
            MLIRAttribute::Integer(4),
        ]));
        op.add_result(MLIRValue::new("tensor".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 4],
            device: "cpu".to_string(),
        }));
        
        let result = lowering.lower_operation(&op);
        assert!(result.is_ok());
        let lowered_ops = result.unwrap();
        assert!(!lowered_ops.is_empty());
    }

    #[test]
    fn test_llvm_code_generation_creation() {
        let target_config = TargetConfig::default();
        let codegen = LLVMCodeGenerator::new(target_config);
        assert!(codegen.is_ok());
    }

    #[test]
    fn test_optimization_levels() {
        let target_config = TargetConfig::default();
        let mut codegen = LLVMCodeGenerator::new(target_config).unwrap();

        // Test different optimization levels
        let opt_levels = vec![
            OptimizationLevel::None,
            OptimizationLevel::Less,
            OptimizationLevel::Default,
            OptimizationLevel::Aggressive,
        ];

        for level in opt_levels {
            let result = codegen.optimize(level);
            assert!(result.is_ok(), "Failed to optimize at level {:?}", level);
        }
    }
}

// ===== INTEGRATION TESTS =====

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn create_test_context() -> MLIRContext {
        MLIRContext::new_mock()
    }

    #[test]
    fn test_end_to_end_ast_to_mlir() {
        let context = create_test_context();
        let mut frontend = AetherMLIRFrontend::new(&context);
        let mut module = context.create_module("test").unwrap();

        // Create a simple AST: (+ 1 2)
        let ast = AST::new(ASTNode::list(vec![
            ASTNodeRef::direct(ASTNode::symbol("+".to_string())),
            ASTNodeRef::direct(ASTNode::number(1.0)),
            ASTNodeRef::direct(ASTNode::number(2.0)),
        ]));

        // Convert AST to MLIR
        let result = frontend.convert_ast_to_module(&ast, &mut module);
        assert!(result.is_ok());
        assert!(!module.operations().is_empty());
    }

    #[test]
    fn test_optimization_pipeline() {
        let context = create_test_context();
        let mut module = context.create_module("test").unwrap();

        // Add some operations
        let mut op1 = MLIROperation::new("aether.tensor_op".to_string());
        op1.add_attribute("op_name".to_string(), MLIRAttribute::String("add".to_string()));
        module.add_operation(op1).unwrap();

        let mut op2 = MLIROperation::new("aether.tensor_op".to_string());
        op2.add_attribute("op_name".to_string(), MLIRAttribute::String("mul".to_string()));
        module.add_operation(op2).unwrap();

        // Apply optimization
        let optimizer = AetherOptimizer::new(&context);
        let result = optimizer.optimize(&mut module);
        assert!(result.is_ok());

        // Module should still have operations
        assert!(!module.operations().is_empty());
    }

    #[test]
    fn test_lowering_pipeline() {
        let context = create_test_context();
        let mut module = context.create_module("test").unwrap();

        // Add Aether operation
        let mut tensor_create = MLIROperation::new("aether.tensor_create".to_string());
        tensor_create.add_attribute("shape".to_string(), MLIRAttribute::Array(vec![
            MLIRAttribute::Integer(2),
            MLIRAttribute::Integer(2),
        ]));
        tensor_create.add_result(MLIRValue::new("tensor".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 2],
            device: "cpu".to_string(),
        }));
        module.add_operation(tensor_create).unwrap();

        // Lower Aether to Standard
        let aether_lowering = AetherToStandardLowering::new(&context);
        let operations = module.operations();
        
        for op in operations {
            if op.name.starts_with("aether.") {
                let result = aether_lowering.lower_operation(op);
                assert!(result.is_ok(), "Failed to lower operation: {}", op.name);
            }
        }
    }
}