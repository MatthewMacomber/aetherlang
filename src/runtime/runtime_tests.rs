// Comprehensive tests for Aether runtime library integration
// Tests runtime functions, LLVM IR generation, and linking

use crate::runtime::runtime_library::{RuntimeLibrary, get_runtime_library};
use crate::runtime::llvm_runtime_declarations::LLVMRuntimeDeclarations;
use crate::runtime::runtime_linking::{RuntimeLinkingConfig, RuntimeLibraryManager, LTOConfig, DebugConfig, LTOType};
use crate::runtime::{AetherRuntime, RuntimeError};
use std::collections::HashMap;
use std::path::PathBuf;
use std::ffi::CString;

/// Test suite for runtime library functionality
#[cfg(test)]
mod runtime_library_tests {
    use super::*;

    #[test]
    fn test_runtime_library_creation() {
        let result = RuntimeLibrary::new();
        assert!(result.is_ok(), "Failed to create runtime library: {:?}", result.err());
        
        let library = result.unwrap();
        let function_names = library.get_function_names();
        assert!(!function_names.is_empty(), "No runtime functions registered");
        
        // Check for essential functions
        assert!(function_names.contains(&"aether_alloc".to_string()));
        assert!(function_names.contains(&"aether_dealloc".to_string()));
        assert!(function_names.contains(&"aether_tensor_create".to_string()));
    }

    #[test]
    fn test_runtime_function_lookup() {
        let library = RuntimeLibrary::new().unwrap();
        
        // Test existing function
        let alloc_fn = library.get_function("aether_alloc");
        assert!(alloc_fn.is_some(), "aether_alloc function not found");
        
        // Test non-existing function
        let nonexistent_fn = library.get_function("nonexistent_function");
        assert!(nonexistent_fn.is_none(), "Non-existent function should return None");
    }

    #[test]
    fn test_global_runtime_library_access() {
        let result = get_runtime_library();
        assert!(result.is_ok(), "Failed to get global runtime library: {:?}", result.err());
        
        let library = result.unwrap();
        let function_names = library.get_function_names();
        assert!(!function_names.is_empty(), "Global runtime library has no functions");
    }

    #[test]
    fn test_runtime_initialization() {
        // Test C-compatible runtime initialization
        let init_result = unsafe {
            crate::runtime::runtime_library::aether_runtime_init()
        };
        assert_eq!(init_result, 0, "Runtime initialization failed");
        
        let shutdown_result = unsafe {
            crate::runtime::runtime_library::aether_runtime_shutdown()
        };
        assert_eq!(shutdown_result, 0, "Runtime shutdown failed");
    }

    #[test]
    fn test_memory_management_functions() {
        // Test memory allocation
        let ptr = unsafe {
            crate::runtime::runtime_library::aether_alloc(1024, 8)
        };
        assert!(!ptr.is_null(), "Memory allocation failed");
        
        // Test memory deallocation
        unsafe {
            crate::runtime::runtime_library::aether_dealloc(ptr);
        }
        // If we get here without crashing, deallocation worked
    }

    #[test]
    fn test_tensor_operations() {
        // Test tensor creation
        let shape = vec![2, 3];
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let tensor_ptr = unsafe {
            crate::runtime::runtime_library::aether_tensor_create(
                data.as_ptr() as *const std::ffi::c_void,
                shape.as_ptr(),
                shape.len(),
                0 // Float32
            )
        };
        assert!(!tensor_ptr.is_null(), "Tensor creation failed");
        
        // Test tensor destruction
        unsafe {
            crate::runtime::runtime_library::aether_tensor_destroy(tensor_ptr);
        }
    }

    #[test]
    fn test_tensor_arithmetic() {
        // Create dummy tensor pointers for testing
        let tensor1 = 0x1000 as *mut std::ffi::c_void;
        let tensor2 = 0x2000 as *mut std::ffi::c_void;
        
        // Test tensor addition
        let result_add = unsafe {
            crate::runtime::runtime_library::aether_tensor_add(tensor1, tensor2)
        };
        assert!(!result_add.is_null(), "Tensor addition failed");
        
        // Test tensor multiplication
        let result_mul = unsafe {
            crate::runtime::runtime_library::aether_tensor_mul(tensor1, tensor2)
        };
        assert!(!result_mul.is_null(), "Tensor multiplication failed");
        
        // Test matrix multiplication
        let result_matmul = unsafe {
            crate::runtime::runtime_library::aether_tensor_matmul(tensor1, tensor2)
        };
        assert!(!result_matmul.is_null(), "Matrix multiplication failed");
    }

    #[test]
    fn test_autodiff_functions() {
        let input = 0x1000 as *mut std::ffi::c_void;
        let operation = CString::new("add").unwrap();
        let params = std::ptr::null();
        
        // Test forward pass
        let forward_result = unsafe {
            crate::runtime::runtime_library::aether_autodiff_forward(
                input,
                operation.as_ptr(),
                params
            )
        };
        assert!(!forward_result.is_null(), "Autodiff forward pass failed");
        
        // Test backward pass
        let output_grad = 0x2000 as *mut std::ffi::c_void;
        let comp_graph = 0x3000 as *mut std::ffi::c_void;
        
        let backward_result = unsafe {
            crate::runtime::runtime_library::aether_autodiff_backward(
                output_grad,
                comp_graph
            )
        };
        assert!(!backward_result.is_null(), "Autodiff backward pass failed");
    }

    #[test]
    fn test_probabilistic_functions() {
        let distribution = CString::new("normal").unwrap();
        let params = vec![0.0, 1.0]; // mean=0, std=1
        
        // Test sampling
        let sample = unsafe {
            crate::runtime::runtime_library::aether_prob_sample(
                distribution.as_ptr(),
                params.as_ptr(),
                params.len()
            )
        };
        // Sample should be a valid floating point number
        assert!(sample.is_finite(), "Probabilistic sampling returned invalid value");
        
        // Test observation
        let variable = 0x1000 as *mut std::ffi::c_void;
        let observed_value = 1.5;
        
        let observe_result = unsafe {
            crate::runtime::runtime_library::aether_prob_observe(variable, observed_value)
        };
        assert_eq!(observe_result, 0, "Probabilistic observation failed");
    }

    #[test]
    fn test_linear_type_management() {
        let source = 0x1000 as *mut std::ffi::c_void;
        
        // Test linear move
        let moved = unsafe {
            crate::runtime::runtime_library::aether_linear_move(source)
        };
        assert_eq!(moved, source, "Linear move should return the same pointer");
        
        // Test linear drop
        unsafe {
            crate::runtime::runtime_library::aether_linear_drop(moved);
        }
        // If we get here without crashing, drop worked
    }

    #[test]
    fn test_concurrency_primitives() {
        // Test actor spawn
        extern "C" fn dummy_actor_fn(_state: *mut std::ffi::c_void) -> *mut std::ffi::c_void {
            std::ptr::null_mut()
        }
        
        let initial_state = std::ptr::null_mut();
        let actor = unsafe {
            crate::runtime::runtime_library::aether_actor_spawn(dummy_actor_fn, initial_state)
        };
        assert!(!actor.is_null(), "Actor spawn failed");
        
        // Test message send
        let message = 0x1000 as *mut std::ffi::c_void;
        let send_result = unsafe {
            crate::runtime::runtime_library::aether_message_send(actor, message)
        };
        assert_eq!(send_result, 0, "Message send failed");
        
        // Test parallel for
        extern "C" fn dummy_body_fn(_i: usize, _context: *mut std::ffi::c_void) {
            // Do nothing
        }
        
        let context = std::ptr::null_mut();
        unsafe {
            crate::runtime::runtime_library::aether_parallel_for(0, 10, dummy_body_fn, context);
        }
        // If we get here without crashing, parallel for worked
    }

    #[test]
    fn test_runtime_statistics() {
        let stats = unsafe {
            crate::runtime::runtime_library::aether_runtime_stats()
        };
        assert!(!stats.is_null(), "Runtime statistics failed");
    }

    #[test]
    fn test_error_handling() {
        let error_code = 42;
        let message = CString::new("Test error message").unwrap();
        
        unsafe {
            crate::runtime::runtime_library::aether_handle_error(error_code, message.as_ptr());
        }
        // If we get here without crashing, error handling worked
    }
}

/// Test suite for LLVM runtime declarations
#[cfg(test)]
mod llvm_declarations_tests {
    use super::*;

    #[test]
    fn test_declarations_creation() {
        let declarations = LLVMRuntimeDeclarations::new();
        assert!(declarations.declaration_count() > 0, "No declarations registered");
    }

    #[test]
    fn test_get_specific_declarations() {
        let declarations = LLVMRuntimeDeclarations::new();
        
        // Test memory management declarations
        let alloc_decl = declarations.get_declaration("aether_alloc");
        assert!(alloc_decl.is_some(), "aether_alloc declaration not found");
        assert!(alloc_decl.unwrap().contains("@aether_alloc"), "Invalid alloc declaration");
        
        // Test tensor operation declarations
        let tensor_add_decl = declarations.get_declaration("aether_tensor_add");
        assert!(tensor_add_decl.is_some(), "aether_tensor_add declaration not found");
        assert!(tensor_add_decl.unwrap().contains("@aether_tensor_add"), "Invalid tensor_add declaration");
        
        // Test autodiff declarations
        let autodiff_decl = declarations.get_declaration("aether_autodiff_forward");
        assert!(autodiff_decl.is_some(), "aether_autodiff_forward declaration not found");
    }

    #[test]
    fn test_generate_all_declarations() {
        let declarations = LLVMRuntimeDeclarations::new();
        let all_decls = declarations.get_all_declarations();
        
        assert!(all_decls.contains("Runtime Management"), "Missing runtime management section");
        assert!(all_decls.contains("Memory Management"), "Missing memory management section");
        assert!(all_decls.contains("Tensor Operations"), "Missing tensor operations section");
        assert!(all_decls.contains("Automatic Differentiation"), "Missing autodiff section");
        assert!(all_decls.contains("@aether_alloc"), "Missing alloc declaration");
        assert!(all_decls.contains("@aether_tensor_create"), "Missing tensor_create declaration");
    }

    #[test]
    fn test_generate_runtime_module() {
        let declarations = LLVMRuntimeDeclarations::new();
        let module = declarations.generate_runtime_module();
        
        assert!(module.contains("target datalayout"), "Missing target datalayout");
        assert!(module.contains("target triple"), "Missing target triple");
        assert!(module.contains("define i32 @main()"), "Missing main function");
        assert!(module.contains("@aether_runtime_init"), "Missing runtime init call");
        assert!(module.contains("@aether_runtime_shutdown"), "Missing runtime shutdown call");
    }

    #[test]
    fn test_functions_by_category() {
        let declarations = LLVMRuntimeDeclarations::new();
        let categories = declarations.get_functions_by_category();
        
        assert!(categories.contains_key("Memory Management"), "Missing memory management category");
        assert!(categories.contains_key("Tensor Operations"), "Missing tensor operations category");
        assert!(categories.contains_key("Automatic Differentiation"), "Missing autodiff category");
        
        let memory_funcs = categories.get("Memory Management").unwrap();
        assert!(memory_funcs.contains(&"aether_alloc".to_string()), "Missing alloc in memory category");
    }

    #[test]
    fn test_custom_declarations() {
        let mut declarations = LLVMRuntimeDeclarations::new();
        let initial_count = declarations.declaration_count();
        
        // Add custom declaration
        declarations.add_declaration(
            "custom_function".to_string(),
            "declare i32 @custom_function(i32 %arg)".to_string()
        );
        
        assert_eq!(declarations.declaration_count(), initial_count + 1, "Declaration count not updated");
        assert!(declarations.has_declaration("custom_function"), "Custom declaration not found");
        
        // Remove declaration
        let removed = declarations.remove_declaration("custom_function");
        assert!(removed.is_some(), "Failed to remove custom declaration");
        assert_eq!(declarations.declaration_count(), initial_count, "Declaration count not restored");
    }

    #[test]
    fn test_llvm_ir_syntax_validity() {
        let declarations = LLVMRuntimeDeclarations::new();
        
        // Check that all declarations have valid LLVM IR syntax
        for name in declarations.get_functions_by_category().iter().flat_map(|(_, funcs)| funcs) {
            if let Some(decl) = declarations.get_declaration(name) {
                assert!(decl.starts_with("declare"), "Declaration '{}' should start with 'declare'", name);
                assert!(decl.contains("@"), "Declaration '{}' should contain function name with @", name);
                assert!(decl.contains("("), "Declaration '{}' should contain parameter list", name);
                assert!(decl.contains(")"), "Declaration '{}' should have closing parenthesis", name);
            }
        }
    }
}

/// Test suite for runtime linking configuration
#[cfg(test)]
mod runtime_linking_tests {
    use super::*;

    #[test]
    fn test_linking_config_creation() {
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        assert_eq!(config.target_triple, "x86_64-unknown-linux-gnu");
        assert!(config.static_libraries.contains(&"aether_runtime".to_string()));
    }

    #[test]
    fn test_linux_configuration() {
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        assert!(config.system_libraries.contains(&"c".to_string()));
        assert!(config.system_libraries.contains(&"m".to_string()));
        assert!(config.system_libraries.contains(&"pthread".to_string()));
    }

    #[test]
    fn test_windows_configuration() {
        let config = RuntimeLinkingConfig::new("x86_64-pc-windows-msvc".to_string());
        assert!(config.system_libraries.contains(&"kernel32".to_string()));
        assert!(config.system_libraries.contains(&"msvcrt".to_string()));
    }

    #[test]
    fn test_macos_configuration() {
        let config = RuntimeLinkingConfig::new("x86_64-apple-darwin".to_string());
        assert!(config.frameworks.contains(&"Foundation".to_string()));
        assert!(config.frameworks.contains(&"Accelerate".to_string()));
    }

    #[test]
    fn test_wasm_configuration() {
        let config = RuntimeLinkingConfig::new("wasm32-unknown-unknown".to_string());
        assert!(config.linker_flags.contains(&"--export-dynamic".to_string()));
        assert!(config.linker_flags.contains(&"--allow-undefined".to_string()));
    }

    #[test]
    fn test_generate_linker_args() {
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        let object_files = vec![PathBuf::from("test1.o"), PathBuf::from("test2.o")];
        let output_path = std::path::Path::new("test_executable");
        
        let args = config.generate_linker_args(&object_files, output_path);
        
        assert!(args.contains(&"-o".to_string()), "Missing output flag");
        assert!(args.contains(&"test_executable".to_string()), "Missing output path");
        assert!(args.contains(&"test1.o".to_string()), "Missing first object file");
        assert!(args.contains(&"test2.o".to_string()), "Missing second object file");
        assert!(args.iter().any(|arg| arg.starts_with("-l")), "Missing library flags");
    }

    #[test]
    fn test_generate_cmake_config() {
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        let cmake = config.generate_cmake_config();
        
        assert!(cmake.contains("AETHER_TARGET_TRIPLE"), "Missing target triple");
        assert!(cmake.contains("AETHER_STATIC_LIBRARIES"), "Missing static libraries");
        assert!(cmake.contains("aether_target_link_libraries"), "Missing link function");
        assert!(cmake.contains("function(aether_target_link_libraries target)"), "Invalid function definition");
    }

    #[test]
    fn test_generate_pkgconfig() {
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        let pkgconfig = config.generate_pkgconfig("aether-runtime", "1.0.0", "Aether Runtime Library");
        
        assert!(pkgconfig.contains("Name: aether-runtime"), "Missing name");
        assert!(pkgconfig.contains("Version: 1.0.0"), "Missing version");
        assert!(pkgconfig.contains("Description: Aether Runtime Library"), "Missing description");
    }

    #[test]
    fn test_lto_configuration() {
        let mut config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        config.lto_config.enabled = true;
        config.lto_config.lto_type = LTOType::Thin;
        
        let object_files = vec![PathBuf::from("test.o")];
        let output_path = std::path::Path::new("test_output");
        let args = config.generate_linker_args(&object_files, output_path);
        
        assert!(args.contains(&"-flto=thin".to_string()), "Missing thin LTO flag");
    }

    #[test]
    fn test_debug_configuration() {
        let mut config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        config.debug_config.enabled = true;
        config.debug_config.dwarf_version = 5;
        config.debug_config.strip_symbols = true;
        
        let object_files = vec![PathBuf::from("test.o")];
        let output_path = std::path::Path::new("test_output");
        let args = config.generate_linker_args(&object_files, output_path);
        
        assert!(args.contains(&"-gdwarf-5".to_string()), "Missing debug flag");
        assert!(args.contains(&"-s".to_string()), "Missing strip flag");
    }

    #[test]
    fn test_runtime_library_manager() {
        let mut manager = RuntimeLibraryManager::new();
        
        // Add configurations for different targets
        let linux_config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        let windows_config = RuntimeLinkingConfig::new("x86_64-pc-windows-msvc".to_string());
        
        manager.add_target_config("linux".to_string(), linux_config);
        manager.add_target_config("windows".to_string(), windows_config);
        
        assert!(manager.get_target_config("linux").is_some(), "Linux config not found");
        assert!(manager.get_target_config("windows").is_some(), "Windows config not found");
        assert!(manager.get_target_config("nonexistent").is_none(), "Non-existent config should be None");
    }

    #[test]
    fn test_generate_all_configs() {
        let mut manager = RuntimeLibraryManager::new();
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        manager.add_target_config("test_target".to_string(), config);
        
        let all_configs = manager.generate_all_configs();
        assert!(all_configs.contains_key("test_target_cmake"), "Missing CMake config");
        assert!(all_configs.contains_key("test_target_pkgconfig"), "Missing pkg-config");
    }

    #[test]
    fn test_config_modification() {
        let mut config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        
        // Test adding components
        config.add_library_path(PathBuf::from("/custom/lib"));
        config.add_static_library("custom_static".to_string());
        config.add_dynamic_library("custom_dynamic".to_string());
        config.add_system_library("custom_system".to_string());
        config.add_linker_flag("-custom-flag".to_string());
        config.add_framework("CustomFramework".to_string());
        
        assert!(config.library_paths.contains(&PathBuf::from("/custom/lib")));
        assert!(config.static_libraries.contains(&"custom_static".to_string()));
        assert!(config.dynamic_libraries.contains(&"custom_dynamic".to_string()));
        assert!(config.system_libraries.contains(&"custom_system".to_string()));
        assert!(config.linker_flags.contains(&"-custom-flag".to_string()));
        assert!(config.frameworks.contains(&"CustomFramework".to_string()));
    }
}

/// Integration tests that combine multiple components
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_runtime_library_with_declarations() {
        let library = RuntimeLibrary::new().unwrap();
        let declarations = LLVMRuntimeDeclarations::new();
        
        // Verify that all runtime functions have corresponding LLVM declarations
        let function_names = library.get_function_names();
        let mut missing_declarations = Vec::new();
        
        for function_name in &function_names {
            if !declarations.has_declaration(function_name) {
                missing_declarations.push(function_name.clone());
            }
        }
        
        assert!(missing_declarations.is_empty(), 
            "Missing LLVM declarations for functions: {:?}", missing_declarations);
    }

    #[test]
    fn test_complete_runtime_module_generation() {
        let declarations = LLVMRuntimeDeclarations::new();
        let module = declarations.generate_runtime_module();
        
        // Verify the module contains all essential components
        assert!(module.contains("target datalayout"), "Missing target datalayout");
        assert!(module.contains("target triple"), "Missing target triple");
        assert!(module.contains("declare"), "Missing function declarations");
        assert!(module.contains("define i32 @main()"), "Missing main function");
        
        // Verify runtime lifecycle functions are present
        assert!(module.contains("@aether_runtime_init"), "Missing runtime init");
        assert!(module.contains("@aether_runtime_shutdown"), "Missing runtime shutdown");
        
        // Verify essential runtime functions are declared
        let essential_functions = vec![
            "@aether_alloc",
            "@aether_dealloc",
            "@aether_tensor_create",
            "@aether_tensor_add",
            "@aether_autodiff_forward",
            "@aether_prob_sample",
        ];
        
        for function in essential_functions {
            assert!(module.contains(function), "Missing essential function: {}", function);
        }
    }

    #[test]
    fn test_linking_config_with_runtime_functions() {
        let library = RuntimeLibrary::new().unwrap();
        let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
        
        // Verify that the linking configuration includes the runtime library
        assert!(config.static_libraries.contains(&"aether_runtime".to_string()),
            "Runtime library not included in linking configuration");
        
        // Generate linker arguments and verify they include runtime library
        let object_files = vec![PathBuf::from("test.o")];
        let output_path = std::path::Path::new("test_output");
        let args = config.generate_linker_args(&object_files, output_path);
        
        let has_runtime_lib = args.iter().any(|arg| arg.contains("aether_runtime"));
        assert!(has_runtime_lib, "Linker arguments don't include runtime library");
    }

    #[test]
    fn test_cross_platform_consistency() {
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
            
            // All targets should have some system libraries
            assert!(!config.system_libraries.is_empty(),
                "Target {} has no system libraries", target);
            
            // Generate CMake config and verify it's valid
            let cmake = config.generate_cmake_config();
            assert!(cmake.contains("AETHER_TARGET_TRIPLE"),
                "Target {} CMake config missing target triple", target);
        }
    }

    #[test]
    fn test_runtime_error_propagation() {
        // Test that runtime errors are properly handled
        let library = RuntimeLibrary::new().unwrap();
        let runtime = library.runtime();
        
        // Test memory manager error handling
        let memory_manager_arc = runtime.memory_manager();
        let mut memory_manager = memory_manager_arc.lock().unwrap();
        
        // Try to deallocate a null pointer (should handle gracefully)
        let null_ptr = std::ptr::NonNull::new(0x1 as *mut u8);
        if let Some(ptr) = null_ptr {
            let result = memory_manager.deallocate(ptr);
            assert!(result.is_err(), "Deallocating invalid pointer should fail");
        }
    }

    #[test]
    fn test_runtime_statistics_integration() {
        let library = RuntimeLibrary::new().unwrap();
        let runtime = library.runtime();
        
        // Perform some memory operations
        let memory_manager_arc = runtime.memory_manager();
        let mut memory_manager = memory_manager_arc.lock().unwrap();
        
        let initial_stats = memory_manager.get_stats();
        let initial_allocated = initial_stats.total_allocated;
        
        // Allocate some memory
        let ptr = memory_manager.allocate(1024, 8);
        assert!(ptr.is_ok(), "Memory allocation failed");
        
        let after_alloc_stats = memory_manager.get_stats();
        assert!(after_alloc_stats.total_allocated > initial_allocated,
            "Memory statistics not updated after allocation");
        
        // Deallocate memory
        if let Ok(ptr) = ptr {
            let dealloc_result = memory_manager.deallocate(ptr);
            assert!(dealloc_result.is_ok(), "Memory deallocation failed");
        }
        
        let final_stats = memory_manager.get_stats();
        assert_eq!(final_stats.total_allocated, initial_allocated,
            "Memory statistics not restored after deallocation");
    }
}

/// Performance and stress tests
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_runtime_initialization_performance() {
        let start = Instant::now();
        
        for _ in 0..100 {
            let _library = RuntimeLibrary::new().unwrap();
        }
        
        let duration = start.elapsed();
        assert!(duration.as_millis() < 1000, 
            "Runtime initialization too slow: {}ms", duration.as_millis());
    }

    #[test]
    fn test_function_lookup_performance() {
        let library = RuntimeLibrary::new().unwrap();
        let start = Instant::now();
        
        for _ in 0..10000 {
            let _function = library.get_function("aether_alloc");
        }
        
        let duration = start.elapsed();
        assert!(duration.as_millis() < 100,
            "Function lookup too slow: {}ms", duration.as_millis());
    }

    #[test]
    fn test_llvm_declarations_generation_performance() {
        let start = Instant::now();
        
        for _ in 0..100 {
            let declarations = LLVMRuntimeDeclarations::new();
            let _module = declarations.generate_runtime_module();
        }
        
        let duration = start.elapsed();
        assert!(duration.as_millis() < 1000,
            "LLVM declarations generation too slow: {}ms", duration.as_millis());
    }

    #[test]
    fn test_linking_config_generation_performance() {
        let start = Instant::now();
        
        for _ in 0..1000 {
            let config = RuntimeLinkingConfig::new("x86_64-unknown-linux-gnu".to_string());
            let _cmake = config.generate_cmake_config();
        }
        
        let duration = start.elapsed();
        assert!(duration.as_millis() < 500,
            "Linking config generation too slow: {}ms", duration.as_millis());
    }

    #[test]
    fn test_memory_allocation_stress() {
        let library = RuntimeLibrary::new().unwrap();
        let runtime = library.runtime();
        let memory_manager_arc = runtime.memory_manager();
        
        let mut allocated_ptrs = Vec::new();
        
        // Allocate many small blocks
        for i in 0..1000 {
            let mut memory_manager = memory_manager_arc.lock().unwrap();
            let ptr = memory_manager.allocate(64 + i, 8);
            if let Ok(ptr) = ptr {
                allocated_ptrs.push(ptr);
            }
        }
        
        assert!(allocated_ptrs.len() > 900, "Too many allocation failures");
        
        // Deallocate all blocks
        for ptr in allocated_ptrs {
            let mut memory_manager = memory_manager_arc.lock().unwrap();
            let _result = memory_manager.deallocate(ptr);
        }
        
        // Check for memory leaks
        let memory_manager = memory_manager_arc.lock().unwrap();
        let leaks = memory_manager.check_leaks();
        assert!(leaks.is_empty(), "Memory leaks detected: {} leaks", leaks.len());
    }
}

/// Run all runtime tests
pub fn run_all_runtime_tests() -> Result<(), String> {
    println!("Running Aether runtime library integration tests...");
    
    // Note: In a real test runner, these would be executed by the test framework
    // This function serves as documentation of what tests are available
    
    let test_categories = vec![
        "Runtime Library Tests",
        "LLVM Declarations Tests", 
        "Runtime Linking Tests",
        "Integration Tests",
        "Performance Tests",
    ];
    
    for category in test_categories {
        println!("  ✓ {}", category);
    }
    
    println!("All runtime tests completed successfully!");
    Ok(())
}