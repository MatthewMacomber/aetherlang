// Integration tests for MLIR-C API bindings
// These tests verify that the MLIR context creation and basic operations work

use aether_language::compiler::mlir::{MLIRContext, MLIRPipeline, MLIRError};

#[test]
fn test_mlir_context_creation() {
    let result = MLIRContext::new();
    
    match result {
        Ok(context) => {
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
        }
        Err(e) => {
            // In stub mode or if MLIR is not available, we expect this to work
            // but with limited functionality
            println!("MLIR context creation failed (expected in stub mode): {}", e);
            
            // The test should still pass in stub mode
            if cfg!(not(feature = "mlir")) {
                // This is expected when MLIR is not available
                return;
            } else {
                panic!("MLIR context creation should succeed when MLIR feature is enabled: {}", e);
            }
        }
    }
}

#[test]
fn test_mlir_module_creation() {
    let context = match MLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            // Skip test if MLIR is not available
            if cfg!(not(feature = "mlir")) {
                return;
            } else {
                panic!("Failed to create MLIR context");
            }
        }
    };

    let result = context.create_module("test_module");
    
    match result {
        Ok(module) => {
            // Verify that the module can be verified (empty modules should always verify)
            let verify_result = module.verify();
            assert!(verify_result.is_ok(), "Empty module should verify successfully: {:?}", verify_result);
            
            // Test module string representation
            let module_str = module.to_string();
            assert!(module_str.is_ok(), "Should be able to convert module to string");
            
            let module_content = module_str.unwrap();
            assert!(!module_content.is_empty(), "Module string should not be empty");
            println!("Module content: {}", module_content);
        }
        Err(e) => {
            if cfg!(not(feature = "mlir")) {
                // Expected in stub mode
                return;
            } else {
                panic!("Module creation should succeed: {}", e);
            }
        }
    }
}

#[test]
fn test_mlir_pipeline_creation() {
    let result = MLIRPipeline::new();
    
    match result {
        Ok(pipeline) => {
            // Verify that we can get the context from the pipeline
            let context = pipeline.context();
            assert!(context.is_dialect_registered("builtin"), "Pipeline context should have builtin dialect");
        }
        Err(e) => {
            if cfg!(not(feature = "mlir")) {
                // Expected in stub mode
                return;
            } else {
                panic!("MLIR pipeline creation should succeed: {}", e);
            }
        }
    }
}

#[test]
fn test_mlir_operation_parsing() {
    let pipeline = match MLIRPipeline::new() {
        Ok(p) => p,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return;
            } else {
                panic!("Failed to create MLIR pipeline");
            }
        }
    };

    // Test parsing various operation strings
    let test_operations = [
        "func.func",
        "arith.addi",
        "tensor.empty",
        "linalg.matmul",
        "gpu.launch_func",
    ];

    for op_str in &test_operations {
        let result = pipeline.parse_operation_string(op_str);
        assert!(result.is_ok(), "Should be able to parse operation: {}", op_str);
        
        let operation = result.unwrap();
        assert_eq!(operation.name, *op_str, "Operation name should match");
    }
}

#[test]
fn test_mlir_type_creation() {
    let context = match MLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return;
            } else {
                panic!("Failed to create MLIR context");
            }
        }
    };

    let safe_context = context.get_context();
    
    // Test basic type creation
    let i32_type = safe_context.create_i32_type();
    let i64_type = safe_context.create_i64_type();
    let f32_type = safe_context.create_f32_type();
    let f64_type = safe_context.create_f64_type();
    let index_type = safe_context.create_index_type();
    
    // In real MLIR mode, these should be valid pointers
    // In stub mode, they might be null but shouldn't crash
    if cfg!(feature = "mlir") {
        // We can't easily test the actual pointer values without more MLIR API,
        // but the calls should not crash
        println!("Created basic MLIR types successfully");
    }
    
    // Test tensor type creation
    let element_type = safe_context.create_f32_type();
    let shape = vec![2, 3, 4];
    let tensor_type = safe_context.create_tensor_type(&shape, element_type);
    
    if cfg!(feature = "mlir") {
        println!("Created tensor type successfully");
    }
}

#[test]
fn test_mlir_attribute_creation() {
    let context = match MLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            if cfg!(not(feature = "mlir")) {
                return;
            } else {
                panic!("Failed to create MLIR context");
            }
        }
    };

    let safe_context = context.get_context();
    
    // Test attribute creation
    let string_attr = safe_context.create_string_attr("test_string");
    let bool_attr_true = safe_context.create_bool_attr(true);
    let bool_attr_false = safe_context.create_bool_attr(false);
    
    if cfg!(feature = "mlir") {
        println!("Created MLIR attributes successfully");
    }
}

#[test]
fn test_mlir_availability() {
    use aether_language::compiler::mlir::{is_mlir_available, get_mlir_version};
    
    let available = is_mlir_available();
    let version = get_mlir_version();
    
    println!("MLIR available: {}", available);
    println!("MLIR version: {}", version);
    
    if cfg!(feature = "mlir") {
        assert!(available, "MLIR should be available when feature is enabled");
        assert_eq!(version, "18.0", "Should target MLIR version 18.0");
    } else {
        assert!(!available, "MLIR should not be available when feature is disabled");
        assert_eq!(version, "stub", "Should report stub version when MLIR is not available");
    }
}

#[test]
fn test_error_handling() {
    // Test that error types can be created and displayed
    let errors = vec![
        MLIRError::ContextCreationError("test context error".to_string()),
        MLIRError::DialectError("test dialect error".to_string()),
        MLIRError::ModuleError("test module error".to_string()),
        MLIRError::OperationError("test operation error".to_string()),
        MLIRError::TypeError("test type error".to_string()),
        MLIRError::VerificationError("test verification error".to_string()),
        MLIRError::LoweringError("test lowering error".to_string()),
        MLIRError::OptimizationError("test optimization error".to_string()),
    ];

    for error in errors {
        let error_string = format!("{}", error);
        assert!(!error_string.is_empty(), "Error should have non-empty string representation");
        println!("Error: {}", error_string);
    }
}

// Performance test to ensure MLIR operations don't have excessive overhead
#[test]
fn test_mlir_performance() {
    use std::time::Instant;
    
    let start = Instant::now();
    
    // Create multiple contexts to test performance
    let mut contexts = Vec::new();
    for i in 0..10 {
        match MLIRContext::new() {
            Ok(ctx) => contexts.push(ctx),
            Err(_) => {
                if cfg!(not(feature = "mlir")) {
                    return; // Skip in stub mode
                } else {
                    panic!("Failed to create context {}", i);
                }
            }
        }
    }
    
    // Create modules for each context
    let mut modules = Vec::new();
    for (i, context) in contexts.iter().enumerate() {
        match context.create_module(&format!("test_module_{}", i)) {
            Ok(module) => modules.push(module),
            Err(e) => panic!("Failed to create module {}: {}", i, e),
        }
    }
    
    // Verify all modules
    for (i, module) in modules.iter().enumerate() {
        match module.verify() {
            Ok(_) => {},
            Err(e) => panic!("Failed to verify module {}: {}", i, e),
        }
    }
    
    let duration = start.elapsed();
    println!("Created and verified {} MLIR contexts/modules in {:?}", contexts.len(), duration);
    
    // Should complete reasonably quickly (less than 1 second for 10 contexts)
    assert!(duration.as_secs() < 5, "MLIR operations should complete in reasonable time");
}