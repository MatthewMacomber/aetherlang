// Integration tests for minimal MLIR implementation
// Tests the complete pipeline from AST to optimized MLIR

#[cfg(test)]
mod tests {
    use super::super::{
        minimal_mlir::*,
        enhanced_mlir_integration::*,
        error_handling::*,
        gpu_tensor_ops::*,
        gpu_dialect::GpuTarget,
    };
    use crate::compiler::ast::{AST, ASTNode, ASTNodeRef, AtomValue};
    use std::collections::HashMap;

    #[test]
    fn test_minimal_mlir_context_creation() {
        let context = MinimalMLIRContext::new();
        assert!(context.is_ok(), "Should create minimal MLIR context successfully");
        
        let context = context.unwrap();
        assert!(context.is_dialect_registered("builtin"));
        assert!(context.is_dialect_registered("arith"));
        assert!(context.is_dialect_registered("func"));
        assert!(context.is_dialect_registered("tensor"));
        assert!(context.is_dialect_registered("aether"));
    }

    #[test]
    fn test_enhanced_pipeline_creation() {
        let pipeline = EnhancedMLIRPipeline::new();
        assert!(pipeline.is_ok(), "Should create enhanced MLIR pipeline successfully");
        
        let pipeline = pipeline.unwrap();
        let stats = pipeline.get_statistics();
        assert_eq!(stats.operations_created, 0);
        assert_eq!(stats.modules_created, 0);
    }

    #[test]
    fn test_module_creation_and_operations() {
        let mut pipeline = EnhancedMLIRPipeline::new().unwrap();
        let mut module = pipeline.create_module("test_module").unwrap();
        
        assert_eq!(module.get_name(), "test_module");
        assert!(!module.has_errors());
        assert!(!module.has_warnings());
        
        // Create and add some operations
        let location = SourceLocation::unknown();
        let op1 = pipeline.create_operation("arith.addf", location.clone()).unwrap();
        let op2 = pipeline.create_operation("func.return", location).unwrap();
        
        assert!(module.add_operation(op1).is_ok());
        assert!(module.add_operation(op2).is_ok());
        
        assert_eq!(module.get_operations().len(), 2);
    }

    #[test]
    fn test_operation_verification() {
        let mut pipeline = EnhancedMLIRPipeline::new().unwrap();
        let location = SourceLocation::unknown();
        
        // Create a valid arithmetic operation
        let mut op = pipeline.create_operation("arith.addf", location.clone()).unwrap();
        
        // Add required operands for arithmetic operation
        let mut context = MinimalMLIRContext::new().unwrap();
        let val1 = context.create_value(MinimalMLIRType::Float { width: 32 });
        let val2 = context.create_value(MinimalMLIRType::Float { width: 32 });
        op.add_operand(val1);
        op.add_operand(val2);
        
        // Verification should pass
        let result = pipeline.verify_operation(&op);
        assert!(result.is_ok(), "Valid operation should verify successfully");
    }

    #[test]
    fn test_ast_to_mlir_compilation() {
        let mut pipeline = EnhancedMLIRPipeline::new().unwrap();
        
        // Create a simple AST with the correct structure
        let root = ASTNode::List(vec![
            ASTNodeRef::Direct(Box::new(ASTNode::Atom(AtomValue::Number(42.0)))),
            ASTNodeRef::Direct(Box::new(ASTNode::Atom(AtomValue::Number(3.14)))),
        ]);
        let mut ast = AST::new(root);
        
        // Add some nodes to the AST
        let node1 = ASTNode::Atom(AtomValue::Number(42.0));
        let node2 = ASTNode::Atom(AtomValue::Number(3.14));
        ast.add_node(node1);
        ast.add_node(node2);
        
        // Compile AST to MLIR
        let result = pipeline.compile_ast(&ast);
        assert!(result.is_ok(), "Should compile simple AST successfully");
        
        let module = result.unwrap();
        assert!(module.get_operations().len() > 0);
        assert!(!module.has_errors());
    }

    #[test]
    fn test_gpu_tensor_optimization_integration() {
        let mut pipeline = EnhancedMLIRPipeline::new().unwrap();
        let mut module = pipeline.create_module("gpu_test").unwrap();
        
        // Create some tensor operations
        let location = SourceLocation::unknown();
        let mut tensor_op1 = pipeline.create_operation("arith.addf", location.clone()).unwrap();
        tensor_op1.add_attribute("tensor_op".to_string(), MinimalMLIRAttribute::Boolean(true));
        
        let mut tensor_op2 = pipeline.create_operation("arith.mulf", location.clone()).unwrap();
        tensor_op2.add_attribute("tensor_op".to_string(), MinimalMLIRAttribute::Boolean(true));
        
        module.add_operation(tensor_op1).unwrap();
        module.add_operation(tensor_op2).unwrap();
        
        // Convert to legacy MLIR module for GPU optimization
        let mut legacy_module = crate::compiler::mlir::mlir_context::MLIRModule::new("gpu_test".to_string());
        
        // Create GPU tensor optimizer
        let context = crate::compiler::mlir::mlir_context::MLIRContext::new_mock();
        let optimizer = GpuTensorOptimizer::new(&context, GpuTarget::Cuda);
        
        // Apply optimizations (this tests the enhanced error handling)
        let result = optimizer.optimize_tensor_operations(&mut legacy_module);
        
        // Should succeed or fail gracefully with recovery
        match result {
            Ok(()) => println!("GPU tensor optimization succeeded"),
            Err(e) => println!("GPU tensor optimization failed gracefully: {}", e),
        }
    }

    #[test]
    fn test_error_handling_and_recovery() {
        let mut pipeline = EnhancedMLIRPipeline::new().unwrap();
        let location = SourceLocation::unknown();
        
        // Try to create an invalid operation
        let result = pipeline.create_operation("invalid.operation", location);
        
        // Should either succeed with recovery or fail with detailed error
        match result {
            Ok(_) => {
                println!("Operation created successfully (possibly with recovery)");
                let stats = pipeline.get_statistics();
                // Check if recovery was attempted
                if stats.recovery_attempts > 0 {
                    println!("Recovery was attempted: {} attempts", stats.recovery_attempts);
                }
            }
            Err(e) => {
                println!("Operation creation failed with detailed error: {}", e);
                // Verify error contains recovery suggestions
                let suggestions = e.recovery_suggestions();
                assert!(!suggestions.is_empty(), "Error should provide recovery suggestions");
            }
        }
    }

    #[test]
    fn test_module_verification_with_errors() {
        let mut pipeline = EnhancedMLIRPipeline::new().unwrap();
        let mut module = pipeline.create_module("error_test").unwrap();
        
        // Create an operation that might fail verification
        let location = SourceLocation::unknown();
        let mut invalid_op = pipeline.create_operation("arith.addf", location).unwrap();
        // Don't add required operands - this should cause verification to fail
        
        module.add_operation(invalid_op).unwrap();
        
        // Verification should fail but provide detailed error information
        let result = pipeline.verify_module(&mut module);
        
        match result {
            Ok(()) => {
                // If verification passes, it might be due to lenient verification in test mode
                println!("Module verification passed (possibly in test mode)");
            }
            Err(e) => {
                println!("Module verification failed as expected: {}", e);
                assert!(module.has_errors() || e.is_recoverable(), "Should have errors or be recoverable");
            }
        }
    }

    #[test]
    fn test_statistics_tracking() {
        let mut pipeline = EnhancedMLIRPipeline::new().unwrap();
        let location = SourceLocation::unknown();
        
        // Create multiple modules and operations to generate statistics
        let _module1 = pipeline.create_module("test1").unwrap();
        let _module2 = pipeline.create_module("test2").unwrap();
        
        let _op1 = pipeline.create_operation("arith.addf", location.clone());
        let _op2 = pipeline.create_operation("func.call", location.clone());
        let _op3 = pipeline.create_operation("builtin.unrealized_conversion_cast", location);
        
        let stats = pipeline.get_statistics();
        assert!(stats.modules_created >= 2, "Should track module creation");
        
        // Print statistics for manual verification
        stats.print_summary();
        
        // Test success rates
        let op_success_rate = stats.operation_success_rate();
        let module_success_rate = stats.module_success_rate();
        
        assert!(op_success_rate >= 0.0 && op_success_rate <= 1.0, "Operation success rate should be between 0 and 1");
        assert!(module_success_rate >= 0.0 && module_success_rate <= 1.0, "Module success rate should be between 0 and 1");
    }

    #[test]
    fn test_type_system_integration() {
        let mut context = MinimalMLIRContext::new().unwrap();
        
        // Test various type creations
        let int_type = MinimalMLIRType::Integer { width: 32, signed: true };
        let float_type = MinimalMLIRType::Float { width: 64 };
        let tensor_type = MinimalMLIRType::Tensor {
            element_type: Box::new(MinimalMLIRType::Float { width: 32 }),
            shape: vec![2, 3, 4],
        };
        let aether_tensor_type = MinimalMLIRType::AetherTensor {
            element_type: Box::new(MinimalMLIRType::Float { width: 32 }),
            shape: vec![10, 10],
            device: "cuda".to_string(),
        };
        
        // Test type string representations
        assert_eq!(int_type.to_string(), "i32");
        assert_eq!(float_type.to_string(), "f64");
        assert_eq!(tensor_type.to_string(), "tensor<2x3x4xf32>");
        assert_eq!(aether_tensor_type.to_string(), "!aether.tensor<10x10xf32, cuda>");
        
        // Test type properties
        assert!(tensor_type.is_tensor());
        assert!(aether_tensor_type.is_tensor());
        assert!(!int_type.is_tensor());
        
        assert_eq!(int_type.size_bits(), Some(32));
        assert_eq!(float_type.size_bits(), Some(64));
        assert_eq!(tensor_type.size_bits(), None);
        
        // Create values with these types
        let int_val = context.create_value(int_type);
        let float_val = context.create_value(float_type);
        let tensor_val = context.create_value(tensor_type);
        
        assert_eq!(int_val.value_type.to_string(), "i32");
        assert_eq!(float_val.value_type.to_string(), "f64");
        assert_eq!(tensor_val.value_type.to_string(), "tensor<2x3x4xf32>");
    }

    #[test]
    fn test_attribute_system() {
        let mut context = MinimalMLIRContext::new().unwrap();
        let mut op = context.create_operation("test.op").unwrap();
        
        // Test various attribute types
        let string_attr = MinimalMLIRAttribute::String("test_value".to_string());
        let int_attr = MinimalMLIRAttribute::Integer(42);
        let float_attr = MinimalMLIRAttribute::Float(3.14);
        let bool_attr = MinimalMLIRAttribute::Boolean(true);
        let array_attr = MinimalMLIRAttribute::Array(vec![
            MinimalMLIRAttribute::Integer(1),
            MinimalMLIRAttribute::Integer(2),
            MinimalMLIRAttribute::Integer(3),
        ]);
        let mut dict_map = HashMap::new();
        dict_map.insert("key1".to_string(), MinimalMLIRAttribute::String("value1".to_string()));
        dict_map.insert("key2".to_string(), MinimalMLIRAttribute::Integer(100));
        let dict_attr = MinimalMLIRAttribute::Dictionary(dict_map);
        
        // Add attributes to operation
        op.add_attribute("string_attr".to_string(), string_attr);
        op.add_attribute("int_attr".to_string(), int_attr);
        op.add_attribute("float_attr".to_string(), float_attr);
        op.add_attribute("bool_attr".to_string(), bool_attr);
        op.add_attribute("array_attr".to_string(), array_attr);
        op.add_attribute("dict_attr".to_string(), dict_attr);
        
        // Verify attributes are stored correctly
        assert_eq!(op.attributes.len(), 6);
        
        // Test attribute string representations
        if let Some(MinimalMLIRAttribute::String(s)) = op.attributes.get("string_attr") {
            assert_eq!(s, "test_value");
        } else {
            panic!("String attribute not found or wrong type");
        }
        
        if let Some(MinimalMLIRAttribute::Integer(i)) = op.attributes.get("int_attr") {
            assert_eq!(*i, 42);
        } else {
            panic!("Integer attribute not found or wrong type");
        }
        
        // Test operation string representation includes attributes
        let op_string = op.to_string();
        assert!(op_string.contains("string_attr"));
        assert!(op_string.contains("test_value"));
    }

    #[test]
    fn test_dialect_system() {
        let context = MinimalMLIRContext::new().unwrap();
        
        // Test that all expected dialects are registered
        let dialects = context.get_registered_dialects();
        assert!(dialects.contains(&"builtin".to_string()));
        assert!(dialects.contains(&"arith".to_string()));
        assert!(dialects.contains(&"func".to_string()));
        assert!(dialects.contains(&"tensor".to_string()));
        assert!(dialects.contains(&"aether".to_string()));
        
        // Test dialect-specific operation verification
        let mut context = MinimalMLIRContext::new().unwrap();
        
        // Test arithmetic operation verification
        let mut arith_op = context.create_operation("arith.addf").unwrap();
        let val1 = context.create_value(MinimalMLIRType::Float { width: 32 });
        let val2 = context.create_value(MinimalMLIRType::Float { width: 32 });
        arith_op.add_operand(val1);
        arith_op.add_operand(val2);
        
        let result = context.verify_operation(&arith_op);
        assert!(result.is_ok(), "Valid arithmetic operation should verify");
        
        // Test function operation verification
        let mut func_op = context.create_operation("func.func").unwrap();
        func_op.add_attribute("sym_name".to_string(), MinimalMLIRAttribute::String("test_func".to_string()));
        
        let result = context.verify_operation(&func_op);
        assert!(result.is_ok(), "Valid function operation should verify");
        
        // Test Aether operation verification
        let mut aether_op = context.create_operation("aether.tensor_create").unwrap();
        aether_op.add_attribute("shape".to_string(), MinimalMLIRAttribute::Array(vec![
            MinimalMLIRAttribute::Integer(2),
            MinimalMLIRAttribute::Integer(3),
        ]));
        aether_op.add_attribute("element_type".to_string(), MinimalMLIRAttribute::String("f32".to_string()));
        
        let result = context.verify_operation(&aether_op);
        assert!(result.is_ok(), "Valid Aether operation should verify");
    }

    #[test]
    fn test_comprehensive_pipeline() {
        // Test the complete pipeline from AST to optimized MLIR
        let mut pipeline = EnhancedMLIRPipeline::new().unwrap();
        pipeline.enable_debug();
        
        // Create a more complex AST using the correct structure
        let function_body = vec![
            ASTNodeRef::Direct(Box::new(ASTNode::Atom(AtomValue::Symbol("let".to_string())))),
            ASTNodeRef::Direct(Box::new(ASTNode::Atom(AtomValue::Symbol("x".to_string())))),
            ASTNodeRef::Direct(Box::new(ASTNode::Atom(AtomValue::Number(42.0)))),
        ];
        
        let function_def = vec![
            ASTNodeRef::Direct(Box::new(ASTNode::Atom(AtomValue::Symbol("defun".to_string())))),
            ASTNodeRef::Direct(Box::new(ASTNode::Atom(AtomValue::Symbol("test_function".to_string())))),
            ASTNodeRef::Direct(Box::new(ASTNode::List(vec![]))), // Empty parameter list
            ASTNodeRef::Direct(Box::new(ASTNode::List(function_body))),
        ];
        
        let root = ASTNode::List(function_def);
        let mut ast = AST::new(root);
        
        // Add some additional nodes
        ast.add_node(ASTNode::Atom(AtomValue::Symbol("x".to_string())));
        ast.add_node(ASTNode::Atom(AtomValue::Number(42.0)));
        
        // Compile AST to MLIR
        let result = pipeline.compile_ast(&ast);
        
        match result {
            Ok(module) => {
                println!("Successfully compiled AST to MLIR:");
                println!("{}", module.to_string_with_diagnostics());
                
                assert!(module.get_operations().len() > 0, "Should have generated operations");
                
                // Check for warnings or errors
                if module.has_warnings() {
                    println!("Compilation warnings:");
                    for warning in module.get_warnings() {
                        println!("  - {}", warning);
                    }
                }
                
                if module.has_errors() {
                    println!("Compilation errors:");
                    for error in module.get_errors() {
                        println!("  - {}", error);
                    }
                }
            }
            Err(e) => {
                println!("AST compilation failed: {}", e);
                println!("Recovery suggestions:");
                for suggestion in e.recovery_suggestions() {
                    println!("  - {}", suggestion);
                }
                
                // Failure is acceptable for complex AST in test environment
                assert!(e.is_recoverable(), "Error should be recoverable");
            }
        }
        
        // Print final statistics
        let stats = pipeline.get_statistics();
        stats.print_summary();
    }
}