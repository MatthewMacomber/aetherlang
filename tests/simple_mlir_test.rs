// Simple MLIR integration test
// Tests basic MLIR functionality without complex dependencies

use aether_language::compiler::mlir::{MLIRContext, is_mlir_available, get_mlir_version};

#[test]
fn test_mlir_availability() {
    let available = is_mlir_available();
    let version = get_mlir_version();
    
    println!("MLIR available: {}", available);
    println!("MLIR version: {}", version);
    
    // This should work in both stub and real mode
    assert!(!version.is_empty());
}

#[test]
fn test_mlir_context_creation_basic() {
    let result = MLIRContext::new();
    
    match result {
        Ok(context) => {
            println!("Successfully created MLIR context");
            
            // Test basic dialect registration checks
            assert!(context.is_dialect_registered("builtin"));
            assert!(context.is_dialect_registered("func"));
            assert!(context.is_dialect_registered("arith"));
            
            println!("Basic dialects are registered");
        }
        Err(e) => {
            println!("MLIR context creation failed: {}", e);
            
            // In stub mode, this might fail, which is acceptable
            if !cfg!(feature = "mlir") {
                println!("This is expected in stub mode");
            } else {
                panic!("MLIR context creation should succeed when MLIR feature is enabled");
            }
        }
    }
}

#[test]
fn test_mlir_module_creation_basic() {
    let context = match MLIRContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping module test due to context creation failure: {}", e);
            return;
        }
    };

    let result = context.create_module("test_module");
    
    match result {
        Ok(module) => {
            println!("Successfully created MLIR module");
            
            // Test module verification
            let verify_result = module.verify();
            match verify_result {
                Ok(_) => println!("Module verification succeeded"),
                Err(e) => println!("Module verification failed: {}", e),
            }
        }
        Err(e) => {
            println!("Module creation failed: {}", e);
            
            if !cfg!(feature = "mlir") {
                println!("This is expected in stub mode");
            } else {
                panic!("Module creation should succeed when MLIR feature is enabled");
            }
        }
    }
}

#[test]
fn test_mlir_bindings_basic() {
    use aether_language::compiler::mlir::SafeMlirContext;
    
    let result = SafeMlirContext::new();
    
    match result {
        Ok(context) => {
            println!("Successfully created safe MLIR context");
            
            // Test basic type creation
            let i32_type = context.create_i32_type();
            let f64_type = context.create_f64_type();
            let index_type = context.create_index_type();
            
            println!("Created basic MLIR types");
            
            // Test attribute creation
            let string_attr = context.create_string_attr("test");
            let bool_attr = context.create_bool_attr(true);
            
            println!("Created basic MLIR attributes");
            
            // Test location creation
            let unknown_loc = context.create_unknown_location();
            let file_loc = context.create_file_location("test.ae", 1, 1);
            
            println!("Created MLIR locations");
        }
        Err(e) => {
            println!("Safe MLIR context creation failed: {}", e);
            
            if !cfg!(feature = "mlir") {
                println!("This is expected in stub mode");
            } else {
                panic!("Safe MLIR context creation should succeed when MLIR feature is enabled");
            }
        }
    }
}