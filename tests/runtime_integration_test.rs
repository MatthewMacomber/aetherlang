// Integration test for Aether runtime library
// Tests the complete runtime integration with MLIR/LLVM pipeline

use aether_language::runtime::{
    RuntimeLibrary, LLVMRuntimeDeclarations, RuntimeLinkingConfig,
    get_runtime_library, AetherRuntime
};
use aether_language::compiler::mlir::llvm_codegen::{LLVMCodeGenerator, TargetConfig};
use std::path::PathBuf;

#[test]
fn test_runtime_library_initialization() {
    // Test that the runtime library can be initialized
    let library_result = RuntimeLibrary::new();
    assert!(library_result.is_ok(), "Failed to create runtime library: {:?}", library_result.err());
    
    let library = library_result.unwrap();
    let function_names = library.get_function_names();
    
    // Verify essential functions are present
    let essential_functions = vec![
        "aether_alloc",
        "aether_dealloc", 
        "aether_tensor_create",
        "aether_tensor_add",
        "aether_autodiff_forward",
        "aether_prob_sample",
    ];
    
    for function in essential_functions {
        assert!(function_names.contains(&function.to_string()),
            "Missing essential runtime function: {}", function);
    }
}

#[test]
fn test_llvm_runtime_declarations() {
    let declarations = LLVMRuntimeDeclarations::new();
    
    // Test that we have declarations for all essential functions
    let essential_functions = vec![
        "aether_runtime_init",
        "aether_runtime_shutdown",
        "aether_alloc",
        "aether_dealloc",
        "aether_tensor_create",
        "aether_tensor_add",
        "aether_autodiff_forward",
        "aether_prob_sample",
    ];
    
    for function in essential_functions {
        assert!(declarations.has_declaration(function),
            "Missing LLVM declaration for function: {}", function);
        
        let declaration = declarations.get_declaration(function).unwrap();
        assert!(declaration.contains("declare"), 
            "Invalid declaration format for {}: {}", function, declaration);
        assert!(declaration.contains(&format!("@{}", function)),
            "Declaration doesn't contain function name for {}: {}", function, declaration);
    }
}

#[test]
fn test_runtime_module_generation() {
    let declarations = LLVMRuntimeDeclarations::new();
    let module = declarations.generate_runtime_module();
    
    // Verify the module has proper LLVM IR structure
    assert!(module.contains("target datalayout"), "Missing target datalayout");
    assert!(module.contains("target triple"), "Missing target triple");
    assert!(module.contains("define i32 @main()"), "Missing main function");
    
    // Verify runtime initialization is included
    assert!(module.contains("@aether_runtime_init"), "Missing runtime init call");
    assert!(module.contains("@aether_runtime_shutdown"), "Missing runtime shutdown call");
    
    // Verify proper control flow
    assert!(module.contains("br i1"), "Missing conditional branch");
    assert!(module.contains("ret i32"), "Missing return statement");
}

#[test]
fn test_linking_configuration() {
    let targets = vec![
        "x86_64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc", 
        "x86_64-apple-darwin",
        "wasm32-unknown-unknown",
    ];
    
    for target in targets {
        let config = RuntimeLinkingConfig::new(target.to_string());
        
        // All targets should include the runtime library
        assert!(config.static_libraries.contains(&"aether_runtime".to_string()),
            "Target {} missing runtime library", target);
        
        // Test linker argument generation
        let object_files = vec![PathBuf::from("test.o")];
        let output_path = std::path::Path::new("test_output");
        let args = config.generate_linker_args(&object_files, output_path);
        
        assert!(args.contains(&"-o".to_string()), "Missing output flag for {}", target);
        assert!(args.contains(&"test_output".to_string()), "Missing output path for {}", target);
        assert!(args.contains(&"test.o".to_string()), "Missing object file for {}", target);
        
        // Test CMake config generation
        let cmake = config.generate_cmake_config();
        assert!(cmake.contains("AETHER_TARGET_TRIPLE"), "Missing target triple in CMake for {}", target);
        assert!(cmake.contains("aether_target_link_libraries"), "Missing link function in CMake for {}", target);
    }
}

#[test]
fn test_llvm_codegen_integration() {
    let target_config = TargetConfig::default();
    let codegen_result = LLVMCodeGenerator::new(target_config);
    
    assert!(codegen_result.is_ok(), "Failed to create LLVM code generator: {:?}", codegen_result.err());
    
    let codegen = codegen_result.unwrap();
    
    // Test that runtime declarations are available
    let runtime_declarations = codegen.get_runtime_declarations();
    assert!(runtime_declarations.declaration_count() > 0, "No runtime declarations in code generator");
    
    // Test that linking configuration is available
    let linking_config = codegen.get_linking_config();
    assert!(linking_config.static_libraries.contains(&"aether_runtime".to_string()),
        "Runtime library not in linking configuration");
}

#[test]
fn test_global_runtime_access() {
    // Test that the global runtime can be accessed
    let runtime_result = get_runtime_library();
    assert!(runtime_result.is_ok(), "Failed to get global runtime library: {:?}", runtime_result.err());
    
    let runtime_library = runtime_result.unwrap();
    let function_names = runtime_library.get_function_names();
    assert!(!function_names.is_empty(), "Global runtime library has no functions");
    
    // Test that we can get the underlying runtime
    let runtime = runtime_library.runtime();
    let memory_manager = runtime.memory_manager();
    
    // Test basic memory operations
    {
        let mut mm = memory_manager.lock().unwrap();
        let stats = mm.get_stats();
        assert_eq!(stats.active_allocations, 0, "Should start with no active allocations");
    }
}

#[test]
fn test_c_api_functions() {
    // Test the C-compatible API functions
    use aether_language::runtime::runtime_library::{
        aether_runtime_init, aether_runtime_shutdown,
        aether_alloc, aether_dealloc, aether_realloc,
        aether_tensor_create, aether_tensor_destroy,
    };
    
    // Test runtime initialization
    let init_result = aether_runtime_init();
    assert_eq!(init_result, 0, "Runtime initialization failed");
    
    // Test memory allocation
    let ptr = aether_alloc(1024, 8);
    assert!(!ptr.is_null(), "Memory allocation failed");
    
    // Test memory reallocation
    let new_ptr = aether_realloc(ptr, 1024, 2048, 8);
    assert!(!new_ptr.is_null(), "Memory reallocation failed");
    
    // Test memory deallocation
    aether_dealloc(new_ptr);
    
    // Test tensor operations
    let shape = vec![2, 3];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    
    let tensor_ptr = aether_tensor_create(
        data.as_ptr() as *const std::ffi::c_void,
        shape.as_ptr(),
        shape.len(),
        0 // Float32
    );
    assert!(!tensor_ptr.is_null(), "Tensor creation failed");
    
    aether_tensor_destroy(tensor_ptr);
    
    // Test runtime shutdown
    let shutdown_result = aether_runtime_shutdown();
    assert_eq!(shutdown_result, 0, "Runtime shutdown failed");
}

#[test]
fn test_runtime_function_categories() {
    let declarations = LLVMRuntimeDeclarations::new();
    let categories = declarations.get_functions_by_category();
    
    // Verify all expected categories are present
    let expected_categories = vec![
        "Runtime Management",
        "Memory Management", 
        "Tensor Operations",
        "Automatic Differentiation",
        "Probabilistic Programming",
        "Linear Types",
        "Concurrency",
        "Math Library",
    ];
    
    for category in expected_categories {
        assert!(categories.contains_key(category),
            "Missing function category: {}", category);
        
        let functions = categories.get(category).unwrap();
        assert!(!functions.is_empty(),
            "Category {} has no functions", category);
    }
}

#[test]
fn test_cross_platform_linking() {
    let platforms = vec![
        ("linux", "x86_64-unknown-linux-gnu"),
        ("windows", "x86_64-pc-windows-msvc"),
        ("macos", "x86_64-apple-darwin"),
        ("wasm", "wasm32-unknown-unknown"),
    ];
    
    for (platform_name, target_triple) in platforms {
        let config = RuntimeLinkingConfig::new(target_triple.to_string());
        
        // Generate CMake configuration
        let cmake = config.generate_cmake_config();
        assert!(cmake.contains(&format!("\"{}\"", target_triple)),
            "CMake config for {} doesn't contain target triple", platform_name);
        
        // Generate pkg-config
        let pkgconfig = config.generate_pkgconfig(
            &format!("aether-runtime-{}", platform_name),
            "1.0.0",
            &format!("Aether Runtime Library for {}", platform_name)
        );
        assert!(pkgconfig.contains(&format!("aether-runtime-{}", platform_name)),
            "pkg-config for {} doesn't contain correct name", platform_name);
    }
}

#[test]
fn test_runtime_error_handling() {
    use aether_language::runtime::runtime_library::aether_handle_error;
    use std::ffi::CString;
    
    // Test error handling function
    let error_message = CString::new("Test error message").unwrap();
    aether_handle_error(42, error_message.as_ptr());
    
    // If we get here without crashing, error handling worked
    // In a real implementation, we might capture stderr to verify the message
}

#[test]
fn test_performance_characteristics() {
    use std::time::Instant;
    
    // Test runtime library creation performance
    let start = Instant::now();
    for _ in 0..10 {
        let _library = RuntimeLibrary::new().unwrap();
    }
    let creation_time = start.elapsed();
    assert!(creation_time.as_millis() < 100, 
        "Runtime library creation too slow: {}ms", creation_time.as_millis());
    
    // Test function lookup performance
    let library = RuntimeLibrary::new().unwrap();
    let start = Instant::now();
    for _ in 0..1000 {
        let _function = library.get_function("aether_alloc");
    }
    let lookup_time = start.elapsed();
    assert!(lookup_time.as_millis() < 10,
        "Function lookup too slow: {}ms", lookup_time.as_millis());
    
    // Test LLVM declarations generation performance
    let start = Instant::now();
    for _ in 0..10 {
        let declarations = LLVMRuntimeDeclarations::new();
        let _module = declarations.generate_runtime_module();
    }
    let generation_time = start.elapsed();
    assert!(generation_time.as_millis() < 100,
        "LLVM module generation too slow: {}ms", generation_time.as_millis());
}

#[test]
fn test_memory_safety() {
    use aether_language::runtime::runtime_library::{aether_alloc, aether_dealloc};
    
    // Test that we can allocate and deallocate many blocks without issues
    let mut ptrs = Vec::new();
    
    // Allocate many blocks
    for i in 0..100 {
        let ptr = aether_alloc(64 + i, 8);
        if !ptr.is_null() {
            ptrs.push(ptr);
        }
    }
    
    assert!(ptrs.len() > 90, "Too many allocation failures");
    
    // Deallocate all blocks
    for ptr in ptrs {
        aether_dealloc(ptr);
    }
    
    // Test null pointer handling
    aether_dealloc(std::ptr::null_mut()); // Should not crash
}

#[test]
fn test_complete_integration() {
    // This test verifies that all components work together
    
    // 1. Create runtime library
    let library = RuntimeLibrary::new().unwrap();
    
    // 2. Create LLVM declarations
    let declarations = LLVMRuntimeDeclarations::new();
    
    // 3. Create linking configuration
    let linking_config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
    
    // 4. Verify all runtime functions have LLVM declarations
    let function_names = library.get_function_names();
    let mut missing_declarations = Vec::new();
    
    for function_name in &function_names {
        if !declarations.has_declaration(function_name) {
            missing_declarations.push(function_name.clone());
        }
    }
    
    assert!(missing_declarations.is_empty(),
        "Missing LLVM declarations for functions: {:?}", missing_declarations);
    
    // 5. Generate complete runtime module
    let runtime_module = declarations.generate_runtime_module();
    assert!(runtime_module.len() > 1000, "Runtime module too small");
    
    // 6. Generate linking configuration
    let cmake_config = linking_config.generate_cmake_config();
    assert!(cmake_config.contains("aether_target_link_libraries"), 
        "CMake config missing link function");
    
    // 7. Test that LLVM code generator can use the runtime
    let target_config = TargetConfig::default();
    let codegen = LLVMCodeGenerator::new(target_config).unwrap();
    
    let codegen_declarations = codegen.get_runtime_declarations();
    assert_eq!(codegen_declarations.declaration_count(), declarations.declaration_count(),
        "Code generator has different number of declarations");
    
    println!("✓ Complete runtime integration test passed");
}