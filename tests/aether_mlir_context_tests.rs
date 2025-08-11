// Unit tests for AetherMLIRContext with proper resource management
// Tests context creation, dialect registration, and resource cleanup

use aether_language::compiler::mlir::{AetherMLIRContext, MLIRError};
use aether_language::compiler::mlir::mlir_context::MLIRError as ContextMLIRError;

#[test]
fn test_aether_mlir_context_creation() {
    let result = AetherMLIRContext::new();
    
    match result {
        Ok(context) => {
            println!("Successfully created AetherMLIRContext");
            
            // Verify that standard dialects are registered
            assert!(context.is_dialect_registered("builtin"), "builtin dialect should be registered");
            assert!(context.is_dialect_registered("func"), "func dialect should be registered");
            assert!(context.is_dialect_registered("arith"), "arith dialect should be registered");
            assert!(context.is_dialect_registered("tensor"), "tensor dialect should be registered");
            assert!(context.is_dialect_registered("linalg"), "linalg dialect should be registered");
            assert!(context.is_dialect_registered("memref"), "memref dialect should be registered");
            assert!(context.is_dialect_registered("scf"), "scf dialect should be registered");
            assert!(context.is_dialect_registered("gpu"), "gpu dialect should be registered");
            assert!(context.is_dialect_registered("spirv"), "spirv dialect should be registered");
            assert!(context.is_dialect_registered("llvm"), "llvm dialect should be registered");
            
            println!("All standard dialects are registered");
        }
        Err(e) => {
            if cfg!(not(feature = "mlir")) {
                println!("Context creation failed as expected in stub mode: {}", e);
                return;
            } else {
                panic!("AetherMLIRContext creation should succeed when MLIR feature is enabled: {}", e);
            }
        }
    }
}

#[test]
fn test_aether_mlir_context_dialect_registration() {
    let mut context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    // Test registering a new dialect (this might fail if the dialect doesn't exist)
    let result = context.register_dialect("test_dialect");
    
    // In real MLIR mode, this should fail because "test_dialect" doesn't exist
    // In stub mode, it might succeed
    match result {
        Ok(_) => {
            if cfg!(feature = "mlir") {
                // This is unexpected in real MLIR mode unless the dialect actually exists
                println!("Warning: test_dialect registration succeeded unexpectedly");
            } else {
                println!("test_dialect registration succeeded in stub mode");
            }
        }
        Err(e) => {
            println!("test_dialect registration failed as expected: {}", e);
            assert!(matches!(e, ContextMLIRError::DialectError(_)));
        }
    }

    // Test duplicate registration (should succeed without error)
    let result = context.register_dialect("builtin");
    assert!(result.is_ok(), "Duplicate dialect registration should succeed");
    
    // Verify the dialect is still registered
    assert!(context.is_dialect_registered("builtin"), "builtin dialect should still be registered");
}

#[test]
fn test_aether_mlir_context_get_registered_dialects() {
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

    let dialects = context.get_registered_dialects();
    
    // Should have at least the standard dialects
    assert!(!dialects.is_empty(), "Should have registered dialects");
    assert!(dialects.contains(&"builtin".to_string()), "Should contain builtin dialect");
    assert!(dialects.contains(&"func".to_string()), "Should contain func dialect");
    assert!(dialects.contains(&"arith".to_string()), "Should contain arith dialect");
    
    println!("Registered dialects: {:?}", dialects);
}

#[test]
fn test_aether_mlir_context_module_creation() {
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
            println!("Successfully created MLIR module");
            
            // Test module verification
            let verify_result = context.verify_module(&module);
            match verify_result {
                Ok(_) => println!("Module verification succeeded"),
                Err(e) => {
                    if cfg!(feature = "mlir") {
                        panic!("Module verification should succeed: {}", e);
                    } else {
                        println!("Module verification failed in stub mode: {}", e);
                    }
                }
            }
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
fn test_aether_mlir_context_type_converter() {
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

    let type_converter = context.get_type_converter();
    
    // Test type conversion
    use aether_language::compiler::mlir::MLIRType;
    
    let int_type = MLIRType::Integer { width: 32, signed: true };
    let result = type_converter.convert_type(&int_type);
    
    match result {
        Ok(_) => println!("Integer type conversion succeeded"),
        Err(e) => {
            if cfg!(feature = "mlir") {
                panic!("Integer type conversion should succeed: {}", e);
            } else {
                println!("Type conversion failed in stub mode: {}", e);
            }
        }
    }

    let float_type = MLIRType::Float { width: 32 };
    let result = type_converter.convert_type(&float_type);
    
    match result {
        Ok(_) => println!("Float type conversion succeeded"),
        Err(e) => {
            if cfg!(feature = "mlir") {
                panic!("Float type conversion should succeed: {}", e);
            } else {
                println!("Type conversion failed in stub mode: {}", e);
            }
        }
    }
}

#[test]
fn test_aether_mlir_context_pass_manager() {
    let mut context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create AetherMLIRContext");
            }
        }
    };

    let pass_manager = context.get_pass_manager_mut();
    
    // Test adding passes
    pass_manager.add_pass("test-pass".to_string());
    pass_manager.add_pass("another-pass".to_string());
    
    let passes = pass_manager.get_passes();
    assert_eq!(passes.len(), 2, "Should have 2 passes");
    assert!(passes.contains(&"test-pass".to_string()), "Should contain test-pass");
    assert!(passes.contains(&"another-pass".to_string()), "Should contain another-pass");
    
    // Test removing a pass
    pass_manager.remove_pass("test-pass");
    let passes = pass_manager.get_passes();
    assert_eq!(passes.len(), 1, "Should have 1 pass after removal");
    assert!(!passes.contains(&"test-pass".to_string()), "Should not contain test-pass");
    assert!(passes.contains(&"another-pass".to_string()), "Should still contain another-pass");
    
    // Test enabling/disabling
    assert!(pass_manager.is_enabled(), "Pass manager should be enabled by default");
    pass_manager.set_enabled(false);
    assert!(!pass_manager.is_enabled(), "Pass manager should be disabled");
    pass_manager.set_enabled(true);
    assert!(pass_manager.is_enabled(), "Pass manager should be enabled again");
    
    // Test clearing passes
    pass_manager.clear_passes();
    let passes = pass_manager.get_passes();
    assert_eq!(passes.len(), 0, "Should have no passes after clearing");
    
    println!("Pass manager tests completed successfully");
}

#[test]
fn test_aether_mlir_context_resource_cleanup() {
    // Test that contexts can be created and dropped without issues
    for i in 0..5 {
        let context = match AetherMLIRContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                if cfg!(not(feature = "mlir")) {
                    return; // Skip test in stub mode
                } else {
                    panic!("Failed to create AetherMLIRContext {}", i);
                }
            }
        };
        
        // Create a module to test resource management
        let _module = context.create_module(&format!("test_module_{}", i));
        
        // Context and module will be dropped at the end of this iteration
    }
    
    println!("Resource cleanup test completed successfully");
}

#[test]
fn test_legacy_mlir_context_compatibility() {
    use aether_language::compiler::mlir::MLIRContext;
    
    let context = match MLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return; // Skip test in stub mode
            } else {
                panic!("Failed to create legacy MLIRContext");
            }
        }
    };

    // Test that legacy interface still works
    assert!(context.is_dialect_registered("builtin"), "builtin dialect should be registered");
    assert!(context.is_dialect_registered("func"), "func dialect should be registered");
    
    let result = context.create_module("legacy_test_module");
    match result {
        Ok(_) => println!("Legacy module creation succeeded"),
        Err(e) => {
            if cfg!(feature = "mlir") {
                panic!("Legacy module creation should succeed: {}", e);
            } else {
                println!("Legacy module creation failed in stub mode: {}", e);
            }
        }
    }
    
    println!("Legacy compatibility test completed successfully");
}

#[test]
fn test_error_types() {
    // Test that all error types can be created and displayed
    let errors = vec![
        ContextMLIRError::ContextCreationError("test context error".to_string()),
        ContextMLIRError::DialectError("test dialect error".to_string()),
        ContextMLIRError::ModuleError("test module error".to_string()),
        ContextMLIRError::OperationError("test operation error".to_string()),
        ContextMLIRError::TypeError("test type error".to_string()),
        ContextMLIRError::VerificationError("test verification error".to_string()),
        ContextMLIRError::LoweringError("test lowering error".to_string()),
        ContextMLIRError::OptimizationError("test optimization error".to_string()),
    ];

    for error in errors {
        let error_string = format!("{}", error);
        assert!(!error_string.is_empty(), "Error should have non-empty string representation");
        println!("Error: {}", error_string);
    }
}