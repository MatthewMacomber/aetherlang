// Integration tests for Aether MLIR dialect registration and operation builders
// Tests dialect registration, operation creation, and validation

use super::dialect_registration::*;
use super::mlir_context::{AetherMLIRContext, MLIRType, MLIRAttribute, MLIRValue};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create test MLIR context
    fn create_test_context() -> Arc<AetherMLIRContext> {
        Arc::new(AetherMLIRContext::new().expect("Failed to create MLIR context"))
    }

    /// Test dialect registry creation and initialization
    #[test]
    fn test_dialect_registry_creation() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Verify all expected operations are registered
        let operations = registry.list_operations();
        assert!(!operations.is_empty());
        
        // Check for key operations
        assert!(operations.contains(&"aether.tensor_create".to_string()));
        assert!(operations.contains(&"aether.matmul".to_string()));
        assert!(operations.contains(&"aether.autodiff_forward".to_string()));
        assert!(operations.contains(&"aether.prob_var".to_string()));
        assert!(operations.contains(&"aether.linear_alloc".to_string()));
        assert!(operations.contains(&"aether.spawn_actor".to_string()));
    }

    /// Test dialect registration with MLIR context
    #[test]
    fn test_dialect_registration() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Register dialect should succeed
        assert!(registry.register_dialect().is_ok());
        
        // Multiple registrations should be idempotent
        assert!(registry.register_dialect().is_ok());
    }

    /// Test operation definition retrieval
    #[test]
    fn test_operation_definition_retrieval() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Test retrieving existing operation
        let tensor_create_def = registry.get_operation_definition("aether.tensor_create");
        assert!(tensor_create_def.is_some());
        
        let def = tensor_create_def.unwrap();
        assert_eq!(def.name, "aether.tensor_create");
        assert!(def.attributes.contains_key("shape"));
        assert!(def.attributes.contains_key("device"));
        
        // Test retrieving non-existent operation
        let unknown_def = registry.get_operation_definition("aether.unknown_op");
        assert!(unknown_def.is_none());
    }

    /// Test operation builder retrieval
    #[test]
    fn test_operation_builder_retrieval() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Test retrieving existing builder
        let tensor_create_builder = registry.get_operation_builder("aether.tensor_create");
        assert!(tensor_create_builder.is_some());
        
        // Test retrieving non-existent builder
        let unknown_builder = registry.get_operation_builder("aether.unknown_op");
        assert!(unknown_builder.is_none());
    }

    /// Test tensor_create operation creation
    #[test]
    fn test_tensor_create_operation() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create valid tensor_create operation
        let mut attributes = HashMap::new();
        attributes.insert("shape".to_string(), MLIRAttribute::Array(vec![
            MLIRAttribute::Integer(2),
            MLIRAttribute::Integer(3),
            MLIRAttribute::Integer(4),
        ]));
        attributes.insert("device".to_string(), MLIRAttribute::String("cpu".to_string()));
        attributes.insert("is_differentiable".to_string(), MLIRAttribute::Boolean(true));
        
        let operation = registry.create_operation("aether.tensor_create", vec![], attributes);
        assert!(operation.is_ok());
        
        let op = operation.unwrap();
        assert_eq!(op.name, "aether.tensor_create");
        assert_eq!(op.operands.len(), 0);
        assert_eq!(op.results.len(), 1);
        
        // Verify result type
        match &op.results[0].value_type {
            MLIRType::AetherTensor { shape, device, .. } => {
                assert_eq!(shape, &vec![2, 3, 4]);
                assert_eq!(device, "cpu");
            }
            _ => panic!("Expected AetherTensor result type"),
        }
    }

    /// Test tensor_create operation with invalid attributes
    #[test]
    fn test_tensor_create_invalid_attributes() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Missing shape attribute
        let mut attributes = HashMap::new();
        attributes.insert("device".to_string(), MLIRAttribute::String("cpu".to_string()));
        
        let operation = registry.create_operation("aether.tensor_create", vec![], attributes);
        assert!(operation.is_err());
        
        // Invalid shape (negative dimension)
        let mut attributes = HashMap::new();
        attributes.insert("shape".to_string(), MLIRAttribute::Array(vec![
            MLIRAttribute::Integer(2),
            MLIRAttribute::Integer(-3), // Invalid negative dimension
        ]));
        attributes.insert("device".to_string(), MLIRAttribute::String("cpu".to_string()));
        
        let operation = registry.create_operation("aether.tensor_create", vec![], attributes);
        assert!(operation.is_err());
    }

    /// Test matmul operation creation
    #[test]
    fn test_matmul_operation() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create operands
        let lhs = MLIRValue::new("lhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
            device: "cpu".to_string(),
        });
        
        let rhs = MLIRValue::new("rhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 4],
            device: "cpu".to_string(),
        });
        
        // Create attributes
        let mut attributes = HashMap::new();
        attributes.insert("transpose_a".to_string(), MLIRAttribute::Boolean(false));
        attributes.insert("transpose_b".to_string(), MLIRAttribute::Boolean(false));
        
        let operation = registry.create_operation("aether.matmul", vec![lhs, rhs], attributes);
        assert!(operation.is_ok());
        
        let op = operation.unwrap();
        assert_eq!(op.name, "aether.matmul");
        assert_eq!(op.operands.len(), 2);
        assert_eq!(op.results.len(), 1);
        
        // Verify result shape is [2, 4]
        match &op.results[0].value_type {
            MLIRType::AetherTensor { shape, .. } => {
                assert_eq!(shape, &vec![2, 4]);
            }
            _ => panic!("Expected AetherTensor result type"),
        }
    }

    /// Test matmul operation with incompatible shapes
    #[test]
    fn test_matmul_incompatible_shapes() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create operands with incompatible shapes
        let lhs = MLIRValue::new("lhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
            device: "cpu".to_string(),
        });
        
        let rhs = MLIRValue::new("rhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![5, 4], // Inner dimension 5 doesn't match 3
            device: "cpu".to_string(),
        });
        
        let operation = registry.create_operation("aether.matmul", vec![lhs, rhs], HashMap::new());
        assert!(operation.is_err());
    }

    /// Test matmul operation with transpose
    #[test]
    fn test_matmul_with_transpose() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create operands
        let lhs = MLIRValue::new("lhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 2], // Will be transposed to [2, 3]
            device: "cpu".to_string(),
        });
        
        let rhs = MLIRValue::new("rhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 4],
            device: "cpu".to_string(),
        });
        
        // Create attributes with transpose_a = true
        let mut attributes = HashMap::new();
        attributes.insert("transpose_a".to_string(), MLIRAttribute::Boolean(true));
        attributes.insert("transpose_b".to_string(), MLIRAttribute::Boolean(false));
        
        let operation = registry.create_operation("aether.matmul", vec![lhs, rhs], attributes);
        assert!(operation.is_ok());
        
        let op = operation.unwrap();
        // Result should be [2, 4] (transposed lhs [2, 3] x rhs [3, 4])
        match &op.results[0].value_type {
            MLIRType::AetherTensor { shape, .. } => {
                assert_eq!(shape, &vec![2, 4]);
            }
            _ => panic!("Expected AetherTensor result type"),
        }
    }

    /// Test autodiff_forward operation creation
    #[test]
    fn test_autodiff_forward_operation() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create operands
        let function = MLIRValue::new("func".to_string(), MLIRType::Function {
            inputs: vec![MLIRType::Float { width: 32 }],
            outputs: vec![MLIRType::Float { width: 32 }],
        });
        
        let input = MLIRValue::new("input".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![1],
            device: "cpu".to_string(),
        });
        
        let tangent = MLIRValue::new("tangent".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![1],
            device: "cpu".to_string(),
        });
        
        let operation = registry.create_operation(
            "aether.autodiff_forward", 
            vec![function, input, tangent], 
            HashMap::new()
        );
        assert!(operation.is_ok());
        
        let op = operation.unwrap();
        assert_eq!(op.name, "aether.autodiff_forward");
        assert_eq!(op.operands.len(), 3);
        assert_eq!(op.results.len(), 1);
    }

    /// Test prob_var operation creation
    #[test]
    fn test_prob_var_operation() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create attributes
        let mut attributes = HashMap::new();
        attributes.insert("name".to_string(), MLIRAttribute::String("x".to_string()));
        
        let mut dist_dict = HashMap::new();
        dist_dict.insert("type".to_string(), MLIRAttribute::String("normal".to_string()));
        dist_dict.insert("mean".to_string(), MLIRAttribute::Float(0.0));
        dist_dict.insert("std".to_string(), MLIRAttribute::Float(1.0));
        attributes.insert("distribution".to_string(), MLIRAttribute::Dictionary(dist_dict));
        
        let operation = registry.create_operation("aether.prob_var", vec![], attributes);
        assert!(operation.is_ok());
        
        let op = operation.unwrap();
        assert_eq!(op.name, "aether.prob_var");
        assert_eq!(op.operands.len(), 0);
        assert_eq!(op.results.len(), 1);
        
        // Verify result type is probabilistic
        match &op.results[0].value_type {
            MLIRType::AetherProbabilistic { .. } => {}
            _ => panic!("Expected AetherProbabilistic result type"),
        }
    }

    /// Test linear_alloc operation creation
    #[test]
    fn test_linear_alloc_operation() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create attributes
        let mut attributes = HashMap::new();
        attributes.insert("allocation_site".to_string(), MLIRAttribute::String("test_site".to_string()));
        attributes.insert("inner_type".to_string(), MLIRAttribute::String("f32".to_string()));
        
        let operation = registry.create_operation("aether.linear_alloc", vec![], attributes);
        assert!(operation.is_ok());
        
        let op = operation.unwrap();
        assert_eq!(op.name, "aether.linear_alloc");
        assert_eq!(op.operands.len(), 0);
        assert_eq!(op.results.len(), 1);
        
        // Verify result type is linear
        match &op.results[0].value_type {
            MLIRType::AetherLinear { .. } => {}
            _ => panic!("Expected AetherLinear result type"),
        }
    }

    /// Test spawn_actor operation creation
    #[test]
    fn test_spawn_actor_operation() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create attributes
        let mut attributes = HashMap::new();
        attributes.insert("actor_type".to_string(), MLIRAttribute::String("TestActor".to_string()));
        
        let operation = registry.create_operation("aether.spawn_actor", vec![], attributes);
        assert!(operation.is_ok());
        
        let op = operation.unwrap();
        assert_eq!(op.name, "aether.spawn_actor");
        assert_eq!(op.operands.len(), 0);
        assert_eq!(op.results.len(), 1);
    }

    /// Test operation verification
    #[test]
    fn test_operation_verification() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create valid tensor_create operation
        let mut attributes = HashMap::new();
        attributes.insert("shape".to_string(), MLIRAttribute::Array(vec![
            MLIRAttribute::Integer(2),
            MLIRAttribute::Integer(3),
        ]));
        attributes.insert("device".to_string(), MLIRAttribute::String("cpu".to_string()));
        
        let operation = registry.create_operation("aether.tensor_create", vec![], attributes)
            .expect("Failed to create operation");
        
        // Verification should pass
        assert!(registry.verify_operation(&operation).is_ok());
        
        // Test verification of operation with missing required attribute
        let mut invalid_op = operation.clone();
        invalid_op.attributes.remove("shape");
        
        assert!(registry.verify_operation(&invalid_op).is_err());
    }

    /// Test unknown operation creation
    #[test]
    fn test_unknown_operation_creation() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        let operation = registry.create_operation("aether.unknown_op", vec![], HashMap::new());
        assert!(operation.is_err());
        
        match operation.unwrap_err() {
            crate::compiler::mlir::MLIRError::OperationError(msg) => {
                assert!(msg.contains("Unknown operation"));
            }
            _ => panic!("Expected OperationError"),
        }
    }

    /// Test batch matrix multiplication with broadcasting
    #[test]
    fn test_batch_matmul() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create batch operands
        let lhs = MLIRValue::new("lhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![5, 2, 3], // Batch size 5
            device: "cpu".to_string(),
        });
        
        let rhs = MLIRValue::new("rhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![5, 3, 4], // Batch size 5
            device: "cpu".to_string(),
        });
        
        let operation = registry.create_operation("aether.matmul", vec![lhs, rhs], HashMap::new());
        assert!(operation.is_ok());
        
        let op = operation.unwrap();
        // Result should be [5, 2, 4]
        match &op.results[0].value_type {
            MLIRType::AetherTensor { shape, .. } => {
                assert_eq!(shape, &vec![5, 2, 4]);
            }
            _ => panic!("Expected AetherTensor result type"),
        }
    }

    /// Test broadcasting in batch matmul
    #[test]
    fn test_broadcast_matmul() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create operands with different batch dimensions
        let lhs = MLIRValue::new("lhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![1, 2, 3], // Batch size 1 (will broadcast)
            device: "cpu".to_string(),
        });
        
        let rhs = MLIRValue::new("rhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![5, 3, 4], // Batch size 5
            device: "cpu".to_string(),
        });
        
        let operation = registry.create_operation("aether.matmul", vec![lhs, rhs], HashMap::new());
        assert!(operation.is_ok());
        
        let op = operation.unwrap();
        // Result should be [5, 2, 4] (broadcasted)
        match &op.results[0].value_type {
            MLIRType::AetherTensor { shape, .. } => {
                assert_eq!(shape, &vec![5, 2, 4]);
            }
            _ => panic!("Expected AetherTensor result type"),
        }
    }

    /// Test incompatible device error
    #[test]
    fn test_incompatible_device_error() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Create operands on different devices
        let lhs = MLIRValue::new("lhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
            device: "cpu".to_string(),
        });
        
        let rhs = MLIRValue::new("rhs".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![3, 4],
            device: "gpu".to_string(), // Different device
        });
        
        let operation = registry.create_operation("aether.matmul", vec![lhs, rhs], HashMap::new());
        assert!(operation.is_err());
    }

    /// Test operation trait verification
    #[test]
    fn test_operation_trait_verification() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Get operation definition to check traits
        let tensor_create_def = registry.get_operation_definition("aether.tensor_create").unwrap();
        assert!(tensor_create_def.traits.contains(&OperationTrait::Pure));
        assert!(tensor_create_def.traits.contains(&OperationTrait::MemorySafe));
        
        let matmul_def = registry.get_operation_definition("aether.matmul").unwrap();
        assert!(matmul_def.traits.contains(&OperationTrait::Pure));
        assert!(matmul_def.traits.contains(&OperationTrait::Differentiable));
    }

    /// Test attribute type validation
    #[test]
    fn test_attribute_type_validation() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Test with wrong attribute type (string instead of array for shape)
        let mut attributes = HashMap::new();
        attributes.insert("shape".to_string(), MLIRAttribute::String("invalid".to_string()));
        attributes.insert("device".to_string(), MLIRAttribute::String("cpu".to_string()));
        
        let operation = registry.create_operation("aether.tensor_create", vec![], attributes);
        assert!(operation.is_err());
    }

    /// Test comprehensive operation list
    #[test]
    fn test_comprehensive_operation_list() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        let operations = registry.list_operations();
        
        // Verify we have operations from all categories
        let tensor_ops = operations.iter().filter(|op| op.starts_with("aether.tensor")).count();
        let autodiff_ops = operations.iter().filter(|op| op.starts_with("aether.autodiff") || op.contains("gradient")).count();
        let prob_ops = operations.iter().filter(|op| op.contains("prob") || op.contains("sample") || op.contains("observe")).count();
        let linear_ops = operations.iter().filter(|op| op.contains("linear")).count();
        let concurrency_ops = operations.iter().filter(|op| op.contains("actor") || op.contains("message") || op.contains("parallel")).count();
        
        assert!(tensor_ops >= 3); // tensor_create, tensor_op, matmul
        assert!(autodiff_ops >= 3); // autodiff_forward, autodiff_reverse, gradient
        assert!(prob_ops >= 3); // prob_var, sample, observe
        assert!(linear_ops >= 3); // linear_alloc, linear_move, linear_drop
        assert!(concurrency_ops >= 3); // spawn_actor, send_message, parallel_for
        
        println!("Total operations registered: {}", operations.len());
        println!("Tensor operations: {}", tensor_ops);
        println!("Autodiff operations: {}", autodiff_ops);
        println!("Probabilistic operations: {}", prob_ops);
        println!("Linear operations: {}", linear_ops);
        println!("Concurrency operations: {}", concurrency_ops);
    }

    /// Test operation builder interface consistency
    #[test]
    fn test_operation_builder_interface() {
        let context = create_test_context();
        let registry = AetherDialectRegistry::new(context).expect("Failed to create dialect registry");
        
        // Test that all registered operations have builders
        let operations = registry.list_operations();
        for op_name in operations {
            let builder = registry.get_operation_builder(&op_name);
            assert!(builder.is_some(), "Missing builder for operation: {}", op_name);
        }
    }
}