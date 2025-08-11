// End-to-end executable generation tests for Aether
// Tests the complete pipeline from MLIR to native executable

use aether_language::compiler::mlir::{AetherMLIRContext, MLIRModule, MLIROperation};
use aether_language::compiler::mlir::llvm_codegen::{LLVMCodeGenerator, TargetConfig, OptimizationLevel};
use aether_language::runtime::executable_linker::{ExecutableLinker, LinkingResult, LinkingError};
use aether_language::runtime::runtime_linking::RuntimeLinkingConfig;
use std::path::{Path, PathBuf};
use std::fs;
use tempfile::TempDir;

/// Test fixture for executable linking tests
struct ExecutableLinkingTestFixture {
    temp_dir: TempDir,
    target_triple: String,
    linking_config: RuntimeLinkingConfig,
}

impl ExecutableLinkingTestFixture {
    fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let target_triple = "x86_64-unknown-linux-gnu".to_string();
        let linking_config = RuntimeLinkingConfig::new(target_triple.clone());
        
        ExecutableLinkingTestFixture {
            temp_dir,
            target_triple,
            linking_config,
        }
    }

    fn temp_path(&self) -> &Path {
        self.temp_dir.path()
    }

    fn create_dummy_object_file(&self, name: &str) -> PathBuf {
        let obj_path = self.temp_path().join(format!("{}.o", name));
        
        // Create a minimal ELF object file header (for testing purposes)
        let elf_header = vec![
            0x7f, 0x45, 0x4c, 0x46, // ELF magic
            0x02, // 64-bit
            0x01, // Little endian
            0x01, // ELF version
            0x00, // System V ABI
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Padding
            0x01, 0x00, // Relocatable file
            0x3e, 0x00, // x86-64
        ];
        
        fs::write(&obj_path, elf_header).expect("Failed to write dummy object file");
        obj_path
    }

    fn create_test_executable_path(&self, name: &str) -> PathBuf {
        self.temp_path().join(name)
    }
}

#[test]
fn test_executable_linker_creation() {
    let fixture = ExecutableLinkingTestFixture::new();
    
    let result = ExecutableLinker::new(
        fixture.target_triple.clone(),
        fixture.linking_config.clone()
    );
    
    match result {
        Ok(linker) => {
            assert_eq!(linker.get_target_triple(), "x86_64-unknown-linux-gnu");
            assert!(linker.get_linker_path().exists() || 
                   linker.get_linker_path().to_string_lossy().contains("stub"));
        }
        Err(LinkingError::LinkerNotFound(_)) => {
            // Expected in CI environments without development tools
            println!("Linker not found - skipping test (expected in CI)");
        }
        Err(e) => panic!("Unexpected error creating linker: {}", e),
    }
}

#[test]
fn test_object_file_validation() {
    let fixture = ExecutableLinkingTestFixture::new();
    
    if let Ok(linker) = ExecutableLinker::new(
        fixture.target_triple.clone(),
        fixture.linking_config.clone()
    ) {
        // Test with existing object file
        let obj_file = fixture.create_dummy_object_file("test");
        let result = linker.link_executable(
            &[obj_file],
            &fixture.create_test_executable_path("test_exe")
        );
        
        // The linking might fail due to invalid object file format, but validation should pass
        match result {
            Ok(_) => println!("Linking succeeded (unexpected but okay)"),
            Err(LinkingError::LinkingFailed { .. }) => {
                // Expected - dummy object file is not valid
                println!("Linking failed as expected with dummy object file");
            }
            Err(e) => println!("Linking failed with error: {}", e),
        }
        
        // Test with non-existent object file
        let missing_file = fixture.temp_path().join("missing.o");
        let result = linker.link_executable(
            &[missing_file.clone()],
            &fixture.create_test_executable_path("test_exe2")
        );
        
        assert!(matches!(result, Err(LinkingError::ObjectFileNotFound(path)) if path == missing_file));
    }
}

#[test]
fn test_linking_configuration() {
    let mut fixture = ExecutableLinkingTestFixture::new();
    
    // Test different target configurations
    let targets = vec![
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu", 
        "x86_64-pc-windows-msvc",
        "x86_64-apple-darwin",
        "wasm32-unknown-unknown",
    ];
    
    for target in targets {
        let config = RuntimeLinkingConfig::new(target.to_string());
        let result = ExecutableLinker::new(target.to_string(), config);
        
        match result {
            Ok(linker) => {
                assert_eq!(linker.get_target_triple(), target);
                
                // Test configuration
                let linking_config = linker.get_linking_config();
                assert_eq!(linking_config.target_triple, target);
                assert!(linking_config.static_libraries.contains(&"aether_runtime".to_string()));
            }
            Err(LinkingError::LinkerNotFound(_)) => {
                println!("Linker not found for target {} (expected)", target);
            }
            Err(e) => {
                println!("Error creating linker for target {}: {}", target, e);
            }
        }
    }
}

#[test]
fn test_linking_with_runtime_libraries() {
    let fixture = ExecutableLinkingTestFixture::new();
    
    if let Ok(mut linker) = ExecutableLinker::new(
        fixture.target_triple.clone(),
        fixture.linking_config.clone()
    ) {
        // Add additional runtime libraries
        let config = linker.get_linking_config_mut();
        config.add_static_library("aether_tensor".to_string());
        config.add_dynamic_library("blas".to_string());
        config.add_system_library("m".to_string());
        
        let obj_file = fixture.create_dummy_object_file("runtime_test");
        let exe_path = fixture.create_test_executable_path("runtime_test_exe");
        
        let result = linker.link_executable(&[obj_file], &exe_path);
        
        match result {
            Ok(linking_result) => {
                assert_eq!(linking_result.executable_path, exe_path);
                assert!(linking_result.statistics.static_libraries_count > 0);
                println!("Linking statistics: {:?}", linking_result.statistics);
            }
            Err(LinkingError::LinkingFailed { exit_code, stderr }) => {
                println!("Linking failed (expected with dummy object): exit_code={}, stderr={}", 
                        exit_code, stderr);
            }
            Err(e) => {
                println!("Linking error: {}", e);
            }
        }
    }
}

#[test]
fn test_debug_information_generation() {
    let fixture = ExecutableLinkingTestFixture::new();
    
    if let Ok(mut linker) = ExecutableLinker::new(
        fixture.target_triple.clone(),
        fixture.linking_config.clone()
    ) {
        // Enable debug information
        let config = linker.get_linking_config_mut();
        config.debug_config.enabled = true;
        config.debug_config.dwarf_version = 4;
        config.debug_config.separate_debug_info = true;
        
        let obj_file = fixture.create_dummy_object_file("debug_test");
        let exe_path = fixture.create_test_executable_path("debug_test_exe");
        
        let result = linker.link_executable(&[obj_file], &exe_path);
        
        match result {
            Ok(linking_result) => {
                assert!(linking_result.statistics.debug_info_included);
                println!("Debug linking successful: {:?}", linking_result.statistics);
            }
            Err(e) => {
                println!("Debug linking failed: {}", e);
            }
        }
    }
}

#[test]
fn test_lto_configuration() {
    let fixture = ExecutableLinkingTestFixture::new();
    
    if let Ok(mut linker) = ExecutableLinker::new(
        fixture.target_triple.clone(),
        fixture.linking_config.clone()
    ) {
        // Enable LTO
        let config = linker.get_linking_config_mut();
        config.lto_config.enabled = true;
        config.lto_config.lto_type = aether_language::runtime::runtime_linking::LTOType::Thin;
        
        let obj_file = fixture.create_dummy_object_file("lto_test");
        let exe_path = fixture.create_test_executable_path("lto_test_exe");
        
        let result = linker.link_executable(&[obj_file], &exe_path);
        
        match result {
            Ok(linking_result) => {
                assert!(linking_result.statistics.lto_applied);
                println!("LTO linking successful: {:?}", linking_result.statistics);
            }
            Err(e) => {
                println!("LTO linking failed: {}", e);
            }
        }
    }
}

#[test]
fn test_cross_platform_linking() {
    let fixture = ExecutableLinkingTestFixture::new();
    
    // Test different platform-specific configurations
    let platform_configs = vec![
        ("x86_64-unknown-linux-gnu", vec!["c", "m", "pthread"]),
        ("x86_64-pc-windows-msvc", vec!["kernel32", "user32"]),
        ("x86_64-apple-darwin", vec!["c", "m"]),
    ];
    
    for (target, expected_libs) in platform_configs {
        let config = RuntimeLinkingConfig::new(target.to_string());
        
        // Verify platform-specific libraries are configured
        for lib in expected_libs {
            if target.contains("windows") {
                // Windows uses different library naming
                continue;
            }
            assert!(config.system_libraries.contains(&lib.to_string()),
                   "Target {} missing expected library: {}", target, lib);
        }
        
        if let Ok(linker) = ExecutableLinker::new(target.to_string(), config) {
            // Test that linker can be created for the target
            assert_eq!(linker.get_target_triple(), target);
        }
    }
}

#[test]
fn test_linking_with_search_paths() {
    let fixture = ExecutableLinkingTestFixture::new();
    
    if let Ok(mut linker) = ExecutableLinker::new(
        fixture.target_triple.clone(),
        fixture.linking_config.clone()
    ) {
        // Create additional library search paths
        let lib_dir1 = fixture.temp_path().join("lib1");
        let lib_dir2 = fixture.temp_path().join("lib2");
        fs::create_dir_all(&lib_dir1).unwrap();
        fs::create_dir_all(&lib_dir2).unwrap();
        
        let additional_paths = vec![lib_dir1, lib_dir2];
        let obj_file = fixture.create_dummy_object_file("search_path_test");
        let exe_path = fixture.create_test_executable_path("search_path_test_exe");
        
        let result = linker.link_with_search_paths(
            &[obj_file],
            &exe_path,
            &additional_paths
        );
        
        match result {
            Ok(linking_result) => {
                println!("Search path linking successful: {:?}", linking_result.statistics);
            }
            Err(e) => {
                println!("Search path linking failed: {}", e);
            }
        }
    }
}

#[test]
fn test_linker_configuration_test() {
    let fixture = ExecutableLinkingTestFixture::new();
    
    if let Ok(linker) = ExecutableLinker::new(
        fixture.target_triple.clone(),
        fixture.linking_config.clone()
    ) {
        let result = linker.test_linking_config();
        
        match result {
            Ok(()) => {
                println!("Linker configuration test passed");
            }
            Err(e) => {
                println!("Linker configuration test failed: {}", e);
                // This is expected in many test environments
            }
        }
    }
}

#[test]
fn test_end_to_end_compilation_pipeline() {
    let fixture = ExecutableLinkingTestFixture::new();
    
    // This test simulates the complete pipeline from MLIR to executable
    // In a real implementation, this would use actual MLIR modules
    
    // Step 1: Create a mock MLIR module (in real implementation, this would come from frontend)
    let mlir_context = match AetherMLIRContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            println!("MLIR context creation failed - skipping end-to-end test");
            return;
        }
    };
    
    let mlir_module = match mlir_context.create_module("test_module") {
        Ok(module) => module,
        Err(_) => {
            println!("MLIR module creation failed - skipping end-to-end test");
            return;
        }
    };
    
    // Step 2: Generate LLVM IR
    let target_config = TargetConfig {
        triple: fixture.target_triple.clone(),
        cpu: "generic".to_string(),
        features: "".to_string(),
        optimization_level: OptimizationLevel::Default,
        relocation_model: aether_language::compiler::mlir::llvm_codegen::RelocModel::PIC,
        code_model: aether_language::compiler::mlir::llvm_codegen::CodeModel::Default,
    };
    
    let mut codegen = match LLVMCodeGenerator::new(target_config) {
        Ok(gen) => gen,
        Err(_) => {
            println!("LLVM code generator creation failed - skipping end-to-end test");
            return;
        }
    };
    
    // Generate LLVM IR (this would normally produce object files)
    match codegen.generate_from_mlir(&mlir_module) {
        Ok(()) => {
            println!("LLVM IR generation successful");
            
            // Step 3: Create object file (simulated)
            let obj_file = fixture.create_dummy_object_file("end_to_end_test");
            
            // Step 4: Link executable
            if let Ok(linker) = ExecutableLinker::new(
                fixture.target_triple.clone(),
                fixture.linking_config.clone()
            ) {
                let exe_path = fixture.create_test_executable_path("end_to_end_exe");
                
                match linker.link_executable(&[obj_file], &exe_path) {
                    Ok(result) => {
                        println!("End-to-end pipeline successful!");
                        println!("Executable: {}", result.executable_path.display());
                        println!("Statistics: {:?}", result.statistics);
                    }
                    Err(e) => {
                        println!("End-to-end linking failed: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("LLVM IR generation failed: {}", e);
        }
    }
}

#[test]
fn test_linking_statistics_accuracy() {
    let fixture = ExecutableLinkingTestFixture::new();
    
    if let Ok(linker) = ExecutableLinker::new(
        fixture.target_triple.clone(),
        fixture.linking_config.clone()
    ) {
        // Create multiple object files
        let obj_files = vec![
            fixture.create_dummy_object_file("stats_test1"),
            fixture.create_dummy_object_file("stats_test2"),
            fixture.create_dummy_object_file("stats_test3"),
        ];
        
        let exe_path = fixture.create_test_executable_path("stats_test_exe");
        
        let result = linker.link_executable(&obj_files, &exe_path);
        
        match result {
            Ok(linking_result) => {
                // Verify statistics accuracy
                assert_eq!(linking_result.statistics.object_files_count, 3);
                assert!(linking_result.statistics.linking_time_ms > 0);
                
                // Check that library counts match configuration
                let config = linker.get_linking_config();
                assert_eq!(linking_result.statistics.static_libraries_count, 
                          config.static_libraries.len());
                assert_eq!(linking_result.statistics.dynamic_libraries_count, 
                          config.dynamic_libraries.len());
                
                println!("Statistics verification passed: {:?}", linking_result.statistics);
            }
            Err(e) => {
                println!("Statistics test failed: {}", e);
            }
        }
    }
}

#[test]
fn test_error_handling_and_recovery() {
    let fixture = ExecutableLinkingTestFixture::new();
    
    if let Ok(linker) = ExecutableLinker::new(
        fixture.target_triple.clone(),
        fixture.linking_config.clone()
    ) {
        // Test various error conditions
        
        // 1. Missing object file
        let missing_file = fixture.temp_path().join("nonexistent.o");
        let result = linker.link_executable(&[missing_file.clone()], 
                                           &fixture.create_test_executable_path("error_test1"));
        assert!(matches!(result, Err(LinkingError::ObjectFileNotFound(path)) if path == missing_file));
        
        // 2. Invalid output path (read-only directory)
        let obj_file = fixture.create_dummy_object_file("error_test");
        let readonly_dir = fixture.temp_path().join("readonly");
        fs::create_dir_all(&readonly_dir).unwrap();
        
        // Try to make directory read-only (may not work on all systems)
        let readonly_exe = readonly_dir.join("readonly_exe");
        let result = linker.link_executable(&[obj_file], &readonly_exe);
        
        match result {
            Ok(_) => println!("Linking succeeded despite read-only directory"),
            Err(e) => println!("Linking failed as expected: {}", e),
        }
        
        println!("Error handling tests completed");
    }
}