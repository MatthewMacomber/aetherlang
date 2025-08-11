// Unit tests for MLIR module creation and verification
// Tests module manipulation utilities and verification workflows

use aether_language::compiler::mlir::{AetherMLIRContext, MLIRModule, MLIROperation, MLIRValue, MLIRType, MLIRAttribute};
use aether_language::compiler::mlir::mlir_context::MLIRError as ContextMLIRError;
use std::collections::HashMap;

#[test]
fn test_module_creation_with_name() {
    let context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    let result = context.create_module("test_module");
    
    match result {
        Ok(module) => {
            println!("Successfully created MLIR module with name");
            
            // Check that module name was added as attribute
            let attributes = module.attributes();
            assert!(attributes.contains_key("module_name"), "Module should have module_name attribute");
            assert_eq!(attributes.get("module_name").unwrap(), "test_module", "Module name should match");
            
            // Module should start empty
            assert_eq!(module.operations().len(), 0, "New module should have no operations");
        }
        Err(e) => {
            if cfg!(not(feature = "mlir")) {
                println!("Module creation failed as expected in stub mode: {}", e);
            } else {
                panic!("Module creation should succeed when MLIR feature is enabled: {}", e);
            }
        }
    }
}

#[test]
fn test_module_creation_from_operations() {
    let context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    // Create some test operations
    let mut operations = Vec::new();
    
    let mut op1 = MLIROperation::new("arith.constant".to_string());
    op1.add_attribute("value".to_string(), MLIRAttribute::Integer(42));
    operations.push(op1);
    
    let mut op2 = MLIROperation::new("func.return".to_string());
    op2.add_operand(MLIRValue::new("val1".to_string(), MLIRType::Integer { width: 32, signed: true }));
    operations.push(op2);

    let result = context.create_module_from_operations("test_module_with_ops", operations);
    
    match result {
        Ok(module) => {
            println!("Successfully created MLIR module from operations");
            
            // Check operations were added
            assert_eq!(module.operations().len(), 2, "Module should have 2 operations");
            assert_eq!(module.operations()[0].name, "arith.constant", "First operation should be arith.constant");
            assert_eq!(module.operations()[1].name, "func.return", "Second operation should be func.return");
        }
        Err(e) => {
            if cfg!(not(feature = "mlir")) {
                println!("Module creation from operations failed as expected in stub mode: {}", e);
            } else {
                panic!("Module creation from operations should succeed: {}", e);
            }
        }
    }
}

#[test]
fn test_module_creation_with_attributes() {
    let context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    let mut attributes = HashMap::new();
    attributes.insert("target".to_string(), "x86_64".to_string());
    attributes.insert("optimization_level".to_string(), "O2".to_string());

    let result = context.create_module_with_attributes("test_module_with_attrs", attributes);
    
    match result {
        Ok(module) => {
            println!("Successfully created MLIR module with attributes");
            
            let module_attrs = module.attributes();
            assert!(module_attrs.contains_key("target"), "Module should have target attribute");
            assert!(module_attrs.contains_key("optimization_level"), "Module should have optimization_level attribute");
            assert_eq!(module_attrs.get("target").unwrap(), "x86_64", "Target should be x86_64");
            assert_eq!(module_attrs.get("optimization_level").unwrap(), "O2", "Optimization level should be O2");
        }
        Err(e) => {
            if cfg!(not(feature = "mlir")) {
                println!("Module creation with attributes failed as expected in stub mode: {}", e);
            } else {
                panic!("Module creation with attributes should succeed: {}", e);
            }
        }
    }
}

#[test]
fn test_module_operation_manipulation() {
    let context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    let mut module = match context.create_module("test_manipulation") {
        Ok(m) => m,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create module");
            }
        }
    };

    // Test adding operations
    let mut op1 = MLIROperation::new("arith.constant".to_string());
    op1.add_attribute("value".to_string(), MLIRAttribute::Integer(10));
    
    let result = module.add_operation(op1);
    assert!(result.is_ok(), "Adding valid operation should succeed");
    assert_eq!(module.operations().len(), 1, "Module should have 1 operation");

    // Test adding function
    let inputs = vec![MLIRType::Integer { width: 32, signed: true }];
    let outputs = vec![MLIRType::Integer { width: 32, signed: true }];
    let result = module.add_function("test_func".to_string(), inputs, outputs);
    assert!(result.is_ok(), "Adding function should succeed");
    assert_eq!(module.operations().len(), 2, "Module should have 2 operations");

    // Test finding operations by name
    let constants = module.find_operations_by_name("arith.constant");
    assert_eq!(constants.len(), 1, "Should find 1 arith.constant operation");

    let functions = module.find_operations_by_name("func.func");
    assert_eq!(functions.len(), 1, "Should find 1 func.func operation");

    // Test removing operation
    let removed = module.remove_operation(0);
    assert!(removed.is_ok(), "Removing operation should succeed");
    assert_eq!(module.operations().len(), 1, "Module should have 1 operation after removal");

    // Test clearing operations
    module.clear_operations();
    assert_eq!(module.operations().len(), 0, "Module should be empty after clearing");

    println!("Module operation manipulation tests completed successfully");
}

#[test]
fn test_module_verification() {
    let context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    let mut module = match context.create_module("test_verification") {
        Ok(m) => m,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create module");
            }
        }
    };

    // Test verification of empty module
    let result = module.verify();
    match result {
        Ok(_) => println!("Empty module verification succeeded"),
        Err(e) => {
            if cfg!(feature = "mlir") {
                panic!("Empty module verification should succeed: {}", e);
            } else {
                println!("Module verification failed in stub mode: {}", e);
            }
        }
    }

    // Test verification with valid operations
    let mut valid_op = MLIROperation::new("arith.constant".to_string());
    valid_op.add_attribute("value".to_string(), MLIRAttribute::Integer(42));
    valid_op.add_result(MLIRValue::new("result1".to_string(), MLIRType::Integer { width: 32, signed: true }));
    
    let result = module.add_operation(valid_op);
    assert!(result.is_ok(), "Adding valid operation should succeed");

    let result = module.verify();
    match result {
        Ok(_) => println!("Module with valid operations verification succeeded"),
        Err(e) => {
            if cfg!(feature = "mlir") {
                panic!("Module with valid operations verification should succeed: {}", e);
            } else {
                println!("Module verification failed in stub mode: {}", e);
            }
        }
    }

    // Test verification with invalid operation (empty name)
    let invalid_op = MLIROperation::new("".to_string());
    let result = module.add_operation(invalid_op);
    assert!(result.is_err(), "Adding invalid operation should fail");

    // Test verification with invalid operation name (no dialect)
    let invalid_op2 = MLIROperation::new("invalid_name".to_string());
    let result = module.add_operation(invalid_op2);
    assert!(result.is_err(), "Adding operation without dialect should fail");

    println!("Module verification tests completed successfully");
}

#[test]
fn test_module_statistics() {
    let context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    let mut module = match context.create_module("test_statistics") {
        Ok(m) => m,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create module");
            }
        }
    };

    // Add some operations from different dialects
    let mut arith_op = MLIROperation::new("arith.constant".to_string());
    arith_op.add_attribute("value".to_string(), MLIRAttribute::Integer(42));
    arith_op.add_result(MLIRValue::new("val1".to_string(), MLIRType::Integer { width: 32, signed: true }));
    module.add_operation(arith_op).unwrap();

    let mut func_op = MLIROperation::new("func.return".to_string());
    func_op.add_operand(MLIRValue::new("val1".to_string(), MLIRType::Integer { width: 32, signed: true }));
    module.add_operation(func_op).unwrap();

    let mut tensor_op = MLIROperation::new("tensor.empty".to_string());
    tensor_op.add_attribute("shape".to_string(), MLIRAttribute::Array(vec![
        MLIRAttribute::Integer(2),
        MLIRAttribute::Integer(3),
    ]));
    module.add_operation(tensor_op).unwrap();

    // Add module attributes
    module.add_attribute("target".to_string(), "cpu".to_string());
    module.add_attribute("opt_level".to_string(), "2".to_string());

    let stats = module.get_statistics();
    
    assert_eq!(stats.operation_count, 3, "Should have 3 operations");
    assert_eq!(stats.module_attributes, 3, "Should have 3 module attributes (including module_name)"); // module_name + target + opt_level
    
    // Check dialect counts
    assert_eq!(*stats.operations_by_dialect.get("arith").unwrap_or(&0), 1, "Should have 1 arith operation");
    assert_eq!(*stats.operations_by_dialect.get("func").unwrap_or(&0), 1, "Should have 1 func operation");
    assert_eq!(*stats.operations_by_dialect.get("tensor").unwrap_or(&0), 1, "Should have 1 tensor operation");

    println!("Module statistics: {:?}", stats);
    println!("Module statistics tests completed successfully");
}

#[test]
fn test_module_string_representation() {
    let context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    let mut module = match context.create_module("test_string_repr") {
        Ok(m) => m,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create module");
            }
        }
    };

    // Add some operations
    let mut op1 = MLIROperation::new("arith.constant".to_string());
    op1.add_attribute("value".to_string(), MLIRAttribute::Integer(42));
    op1.add_result(MLIRValue::new("c0".to_string(), MLIRType::Integer { width: 32, signed: true }));
    module.add_operation(op1).unwrap();

    let mut op2 = MLIROperation::new("func.return".to_string());
    op2.add_operand(MLIRValue::new("c0".to_string(), MLIRType::Integer { width: 32, signed: true }));
    module.add_operation(op2).unwrap();

    let result = module.to_string();
    match result {
        Ok(module_str) => {
            println!("Module string representation:\n{}", module_str);
            
            // Check that the string contains expected elements
            assert!(module_str.contains("module {"), "Should contain module header");
            assert!(module_str.contains("arith.constant"), "Should contain arith.constant operation");
            assert!(module_str.contains("func.return"), "Should contain func.return operation");
            assert!(module_str.contains("%c0"), "Should contain result reference");
            assert!(module_str.contains("value=42"), "Should contain attribute");
        }
        Err(e) => {
            panic!("Module string representation should succeed: {}", e);
        }
    }

    println!("Module string representation tests completed successfully");
}

#[test]
fn test_module_attribute_manipulation() {
    let context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    let mut module = match context.create_module("test_attributes") {
        Ok(m) => m,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create module");
            }
        }
    };

    // Test adding attributes
    module.add_attribute("target".to_string(), "x86_64".to_string());
    module.add_attribute("opt_level".to_string(), "2".to_string());
    
    let attrs = module.attributes();
    assert_eq!(attrs.len(), 3, "Should have 3 attributes (including module_name)"); // module_name + target + opt_level
    assert_eq!(attrs.get("target").unwrap(), "x86_64", "Target should be x86_64");
    assert_eq!(attrs.get("opt_level").unwrap(), "2", "Opt level should be 2");

    // Test removing attributes
    let removed = module.remove_attribute("opt_level");
    assert!(removed.is_some(), "Should remove opt_level attribute");
    assert_eq!(removed.unwrap(), "2", "Removed value should be 2");
    
    let attrs = module.attributes();
    assert_eq!(attrs.len(), 2, "Should have 2 attributes after removal");
    assert!(!attrs.contains_key("opt_level"), "Should not contain opt_level");

    // Test removing non-existent attribute
    let removed = module.remove_attribute("non_existent");
    assert!(removed.is_none(), "Should return None for non-existent attribute");

    println!("Module attribute manipulation tests completed successfully");
}

#[test]
fn test_module_operations_cloning() {
    let context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    let mut module = match context.create_module("test_cloning") {
        Ok(m) => m,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create module");
            }
        }
    };

    // Add some operations and attributes
    let mut op = MLIROperation::new("arith.constant".to_string());
    op.add_attribute("value".to_string(), MLIRAttribute::Integer(42));
    module.add_operation(op).unwrap();
    
    module.add_attribute("target".to_string(), "cpu".to_string());

    // Clone operations and attributes
    let (cloned_ops, cloned_attrs) = module.clone_operations_and_attributes();
    
    assert_eq!(cloned_ops.len(), 1, "Should have 1 cloned operation");
    assert_eq!(cloned_ops[0].name, "arith.constant", "Cloned operation should have correct name");
    
    assert_eq!(cloned_attrs.len(), 2, "Should have 2 cloned attributes"); // module_name + target
    assert!(cloned_attrs.contains_key("target"), "Should contain target attribute");

    println!("Module operations cloning tests completed successfully");
}

#[test]
fn test_error_handling() {
    let context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    let mut module = match context.create_module("test_errors") {
        Ok(m) => m,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create module");
            }
        }
    };

    // Test removing operation from empty module
    let result = module.remove_operation(0);
    assert!(result.is_err(), "Removing operation from empty module should fail");
    assert!(matches!(result.unwrap_err(), ContextMLIRError::OperationError(_)), "Should be OperationError");

    // Test removing operation with invalid index
    let mut op = MLIROperation::new("arith.constant".to_string());
    op.add_result(MLIRValue::new("val1".to_string(), MLIRType::Integer { width: 32, signed: true }));
    module.add_operation(op).unwrap();
    
    let result = module.remove_operation(5);
    assert!(result.is_err(), "Removing operation with invalid index should fail");

    // Test adding operation with empty name
    let invalid_op = MLIROperation::new("".to_string());
    let result = module.add_operation(invalid_op);
    assert!(result.is_err(), "Adding operation with empty name should fail");

    // Test adding operation with invalid name (no dialect)
    let invalid_op2 = MLIROperation::new("invalid_name".to_string());
    let result = module.add_operation(invalid_op2);
    assert!(result.is_err(), "Adding operation without dialect should fail");

    // Test adding operation with empty operand ID
    let mut invalid_op3 = MLIROperation::new("func.call".to_string());
    invalid_op3.add_operand(MLIRValue::new("".to_string(), MLIRType::Integer { width: 32, signed: true }));
    let result = module.add_operation(invalid_op3);
    assert!(result.is_err(), "Adding operation with empty operand ID should fail");

    // Test adding operation with empty result ID
    let mut invalid_op4 = MLIROperation::new("arith.constant".to_string());
    invalid_op4.add_result(MLIRValue::new("".to_string(), MLIRType::Integer { width: 32, signed: true }));
    let result = module.add_operation(invalid_op4);
    assert!(result.is_err(), "Adding operation with empty result ID should fail");

    println!("Error handling tests completed successfully");
}