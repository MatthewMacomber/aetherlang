// Tests for LLVM code generation from MLIR

use super::llvm_codegen::*;
use super::mlir_context::{MLIRContext, MLIRModule, MLIROperation, MLIRValue, MLIRType, MLIRAttribute};
use std::path::Path;

#[test]
fn test_llvm_codegen_creation() {
    let config = TargetConfig::default();
    let codegen = LLVMCodeGenerator::new(config);
    assert!(codegen.is_ok(), "Should be able to create LLVM code generator");
}

#[test]
fn test_llvm_codegen_default() {
    let codegen = LLVMCodeGenerator::new_default();
    assert!(codegen.is_ok(), "Should be able to create default LLVM code generator");
}

#[test]
fn test_target_config_default() {
    let config = TargetConfig::default();
    assert_eq!(config.triple, "x86_64-unknown-linux-gnu");
    assert_eq!(config.cpu, "generic");
    assert_eq!(config.features, "");
    assert_eq!(config.optimization_level, OptimizationLevel::Default);
    assert_eq!(config.relocation_model, RelocModel::Default);
    assert_eq!(config.code_model, CodeModel::Default);
}

#[test]
fn test_optimization_level_conversion() {
    use super::llvm_codegen::llvm_bindings::LLVMCodeGenOptLevel;
    
    assert_eq!(LLVMCodeGenOptLevel::from(OptimizationLevel::None), LLVMCodeGenOptLevel::LLVMCodeGenLevelNone);
    assert_eq!(LLVMCodeGenOptLevel::from(OptimizationLevel::Less), LLVMCodeGenOptLevel::LLVMCodeGenLevelLess);
    assert_eq!(LLVMCodeGenOptLevel::from(OptimizationLevel::Default), LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault);
    assert_eq!(LLVMCodeGenOptLevel::from(OptimizationLevel::Aggressive), LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive);
}

#[test]
fn test_relocation_model_conversion() {
    use super::llvm_codegen::llvm_bindings::LLVMRelocMode;
    
    assert_eq!(LLVMRelocMode::from(RelocModel::Default), LLVMRelocMode::LLVMRelocDefault);
    assert_eq!(LLVMRelocMode::from(RelocModel::Static), LLVMRelocMode::LLVMRelocStatic);
    assert_eq!(LLVMRelocMode::from(RelocModel::PIC), LLVMRelocMode::LLVMRelocPIC);
}

#[test]
fn test_code_model_conversion() {
    use super::llvm_codegen::llvm_bindings::LLVMCodeModel;
    
    assert_eq!(LLVMCodeModel::from(CodeModel::Default), LLVMCodeModel::LLVMCodeModelDefault);
    assert_eq!(LLVMCodeModel::from(CodeModel::Small), LLVMCodeModel::LLVMCodeModelSmall);
    assert_eq!(LLVMCodeModel::from(CodeModel::Large), LLVMCodeModel::LLVMCodeModelLarge);
}

#[test]
fn test_target_machine_creation() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    let result = codegen.create_target_machine();
    assert!(result.is_ok(), "Should be able to create target machine");
}

#[test]
fn test_mlir_to_llvm_translation_empty_module() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create an empty MLIR module for testing
    let context = MLIRContext::new_mock();
    let module = context.create_module("test_module").expect("Failed to create module");
    
    let result = codegen.generate_from_mlir(&module);
    assert!(result.is_ok(), "Should be able to translate empty MLIR module to LLVM IR");
    
    let stats = codegen.get_statistics();
    assert_eq!(stats.total_operations, 0);
    assert_eq!(stats.successful_translations, 0);
    assert!(stats.translation_successful);
}

#[test]
fn test_mlir_to_llvm_translation_with_operations() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a MLIR module with some operations
    let context = MLIRContext::new_mock();
    let mut module = context.create_module("test_module").expect("Failed to create module");
    
    // Add a function operation
    let mut func_op = MLIROperation::new("func.func".to_string());
    func_op.add_attribute("sym_name".to_string(), MLIRAttribute::String("test_func".to_string()));
    func_op.add_attribute("function_type".to_string(), MLIRAttribute::String("() -> ()".to_string()));
    module.add_operation(func_op).expect("Failed to add function operation");
    
    // Add a return operation
    let return_op = MLIROperation::new("func.return".to_string());
    module.add_operation(return_op).expect("Failed to add return operation");
    
    let result = codegen.generate_from_mlir(&module);
    assert!(result.is_ok(), "Should be able to translate MLIR module with operations to LLVM IR");
    
    let stats = codegen.get_statistics();
    assert_eq!(stats.total_operations, 2);
    assert_eq!(stats.successful_translations, 2);
    assert!(stats.translation_successful);
}

#[test]
fn test_llvm_dialect_operations() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a MLIR module with LLVM dialect operations
    let context = MLIRContext::new_mock();
    let mut module = context.create_module("test_module").expect("Failed to create module");
    
    // Add LLVM dialect operations
    let llvm_func_op = MLIROperation::new("llvm.func".to_string());
    module.add_operation(llvm_func_op).expect("Failed to add LLVM func operation");
    
    let llvm_add_op = MLIROperation::new("llvm.add".to_string());
    module.add_operation(llvm_add_op).expect("Failed to add LLVM add operation");
    
    let llvm_return_op = MLIROperation::new("llvm.return".to_string());
    module.add_operation(llvm_return_op).expect("Failed to add LLVM return operation");
    
    let result = codegen.generate_from_mlir(&module);
    assert!(result.is_ok(), "Should be able to translate LLVM dialect operations");
    
    let stats = codegen.get_statistics();
    assert_eq!(stats.total_operations, 3);
    assert_eq!(stats.successful_translations, 3);
}

#[test]
fn test_arithmetic_dialect_operations() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a MLIR module with arithmetic dialect operations
    let context = MLIRContext::new_mock();
    let mut module = context.create_module("test_module").expect("Failed to create module");
    
    // Add arithmetic operations
    let addi_op = MLIROperation::new("arith.addi".to_string());
    module.add_operation(addi_op).expect("Failed to add arith.addi operation");
    
    let mulf_op = MLIROperation::new("arith.mulf".to_string());
    module.add_operation(mulf_op).expect("Failed to add arith.mulf operation");
    
    let cmpi_op = MLIROperation::new("arith.cmpi".to_string());
    module.add_operation(cmpi_op).expect("Failed to add arith.cmpi operation");
    
    let result = codegen.generate_from_mlir(&module);
    assert!(result.is_ok(), "Should be able to translate arithmetic dialect operations");
    
    let stats = codegen.get_statistics();
    assert_eq!(stats.total_operations, 3);
    assert_eq!(stats.successful_translations, 3);
}

#[test]
fn test_memref_dialect_operations() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a MLIR module with memref dialect operations
    let context = MLIRContext::new_mock();
    let mut module = context.create_module("test_module").expect("Failed to create module");
    
    // Add memref operations
    let alloc_op = MLIROperation::new("memref.alloc".to_string());
    module.add_operation(alloc_op).expect("Failed to add memref.alloc operation");
    
    let load_op = MLIROperation::new("memref.load".to_string());
    module.add_operation(load_op).expect("Failed to add memref.load operation");
    
    let store_op = MLIROperation::new("memref.store".to_string());
    module.add_operation(store_op).expect("Failed to add memref.store operation");
    
    let dealloc_op = MLIROperation::new("memref.dealloc".to_string());
    module.add_operation(dealloc_op).expect("Failed to add memref.dealloc operation");
    
    let result = codegen.generate_from_mlir(&module);
    assert!(result.is_ok(), "Should be able to translate memref dialect operations");
    
    let stats = codegen.get_statistics();
    assert_eq!(stats.total_operations, 4);
    assert_eq!(stats.successful_translations, 4);
}

#[test]
fn test_unsupported_operation_handling() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a MLIR module with an unsupported operation
    let context = MLIRContext::new_mock();
    let mut module = context.create_module("test_module").expect("Failed to create module");
    
    // Add an unsupported operation
    let unsupported_op = MLIROperation::new("unsupported.operation".to_string());
    module.add_operation(unsupported_op).expect("Failed to add unsupported operation");
    
    let result = codegen.generate_from_mlir(&module);
    assert!(result.is_err(), "Should fail to translate unsupported operations");
    
    match result {
        Err(CodegenError::TranslationError(msg)) => {
            assert!(msg.contains("unsupported.operation"), "Error message should mention the unsupported operation");
        }
        _ => panic!("Expected TranslationError"),
    }
}

#[test]
fn test_optimization_levels() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a simple MLIR module
    let context = MLIRContext::new_mock();
    let module = context.create_module("test_module").expect("Failed to create module");
    
    // Generate LLVM IR
    codegen.generate_from_mlir(&module).expect("Failed to generate LLVM IR");
    
    // Test different optimization levels
    for opt_level in [OptimizationLevel::None, OptimizationLevel::Less, OptimizationLevel::Default, OptimizationLevel::Aggressive] {
        let result = codegen.optimize(opt_level);
        assert!(result.is_ok(), "Should be able to optimize with level {:?}", opt_level);
        
        let stats = codegen.get_statistics();
        assert!(stats.translation_successful, "Optimization should be marked as successful");
    }
}

#[test]
fn test_optimization_without_module() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Try to optimize without generating LLVM IR first
    let result = codegen.optimize(OptimizationLevel::Default);
    assert!(result.is_err(), "Should fail to optimize without LLVM module");
    
    match result {
        Err(CodegenError::OptimizationError(msg)) => {
            assert!(msg.contains("No LLVM module"), "Error message should mention missing LLVM module");
        }
        _ => panic!("Expected OptimizationError"),
    }
}

#[test]
fn test_llvm_ir_output() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a simple MLIR module
    let context = MLIRContext::new_mock();
    let module = context.create_module("test_module").expect("Failed to create module");
    
    // Generate LLVM IR
    codegen.generate_from_mlir(&module).expect("Failed to generate LLVM IR");
    
    // Get LLVM IR as string
    let ir = codegen.get_llvm_ir();
    assert!(ir.is_ok(), "Should be able to get LLVM IR as string");
    
    let ir_string = ir.unwrap();
    assert!(!ir_string.is_empty(), "LLVM IR string should not be empty");
    
    // In stub mode, we should get the stub IR
    if !cfg!(feature = "llvm") {
        assert!(ir_string.contains("Stub LLVM IR"), "Should contain stub IR marker");
        assert!(ir_string.contains("define i32 @main()"), "Should contain main function");
    }
}

#[test]
fn test_llvm_ir_output_without_module() {
    let codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Try to get LLVM IR without generating it first
    let result = codegen.get_llvm_ir();
    assert!(result.is_err(), "Should fail to get LLVM IR without module");
    
    match result {
        Err(CodegenError::GeneralError(msg)) => {
            assert!(msg.contains("No LLVM module"), "Error message should mention missing LLVM module");
        }
        _ => panic!("Expected GeneralError"),
    }
}

#[test]
fn test_target_config_update() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    let mut new_config = TargetConfig::default();
    new_config.optimization_level = OptimizationLevel::Aggressive;
    new_config.cpu = "native".to_string();
    new_config.features = "+avx2,+fma".to_string();
    new_config.relocation_model = RelocModel::PIC;
    new_config.code_model = CodeModel::Small;
    
    let result = codegen.set_target_config(new_config);
    assert!(result.is_ok(), "Should be able to update target configuration");
    
    let config = codegen.get_target_config();
    assert_eq!(config.optimization_level, OptimizationLevel::Aggressive);
    assert_eq!(config.cpu, "native");
    assert_eq!(config.features, "+avx2,+fma");
    assert_eq!(config.relocation_model, RelocModel::PIC);
    assert_eq!(config.code_model, CodeModel::Small);
}

#[test]
fn test_statistics() {
    let codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    let stats = codegen.get_statistics();
    
    assert_eq!(stats.total_operations, 0);
    assert_eq!(stats.successful_translations, 0);
    assert_eq!(stats.failed_translations, 0);
    assert!(!stats.translation_successful);
    assert!(!stats.translation_successful);
    assert_eq!(stats.success_rate(), 0.0);
}

#[test]
fn test_statistics_after_translation() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a MLIR module with operations
    let context = MLIRContext::new_mock();
    let mut module = context.create_module("test_module").expect("Failed to create module");
    
    // Add operations
    let func_op = MLIROperation::new("func.func".to_string());
    module.add_operation(func_op).expect("Failed to add operation");
    
    let return_op = MLIROperation::new("func.return".to_string());
    module.add_operation(return_op).expect("Failed to add operation");
    
    // Generate LLVM IR
    codegen.generate_from_mlir(&module).expect("Failed to generate LLVM IR");
    
    let stats = codegen.get_statistics();
    assert_eq!(stats.total_operations, 2);
    assert_eq!(stats.successful_translations, 2);
    assert_eq!(stats.failed_translations, 0);
    assert!(stats.translation_successful);
    assert_eq!(stats.success_rate(), 1.0);
}

#[test]
fn test_codegen_error_display() {
    let errors = vec![
        CodegenError::ContextCreationError("test".to_string()),
        CodegenError::ModuleCreationError("test".to_string()),
        CodegenError::TranslationError("test".to_string()),
        CodegenError::VerificationError("test".to_string()),
        CodegenError::TargetMachineError("test".to_string()),
        CodegenError::OptimizationError("test".to_string()),
        CodegenError::CodeGenerationError("test".to_string()),
        CodegenError::IOError("test".to_string()),
        CodegenError::GeneralError("test".to_string()),
    ];
    
    for error in errors {
        let error_string = format!("{}", error);
        assert!(!error_string.is_empty(), "Error should have non-empty display string");
        assert!(error_string.contains("test"), "Error should contain the error message");
    }
}

#[test]
fn test_statistics_reset() {
    let mut stats = CodegenStatistics::new();
    
    // Modify statistics
    stats.total_operations = 10;
    stats.successful_translations = 8;
    stats.failed_translations = 2;
    stats.translation_successful = true;
    
    // Reset statistics
    stats.reset();
    
    assert_eq!(stats.total_operations, 0);
    assert_eq!(stats.successful_translations, 0);
    assert_eq!(stats.failed_translations, 0);
    assert!(!stats.translation_successful);
}

#[test]
fn test_success_rate_calculation() {
    let mut stats = CodegenStatistics::new();
    
    // Test with no operations
    assert_eq!(stats.success_rate(), 0.0);
    
    // Test with all successful
    stats.total_operations = 10;
    stats.successful_translations = 10;
    assert_eq!(stats.success_rate(), 1.0);
    
    // Test with partial success
    stats.successful_translations = 7;
    assert_eq!(stats.success_rate(), 0.7);
    
    // Test with no success
    stats.successful_translations = 0;
    assert_eq!(stats.success_rate(), 0.0);
}

#[test]
fn test_custom_optimization_config() {
    let mut config = OptimizationConfig::default();
    assert!(config.enable_inlining);
    assert!(!config.aggressive_inlining);
    assert!(config.enable_vectorization);
    assert!(config.enable_loop_optimizations);
    
    // Test custom configuration
    config.aggressive_inlining = true;
    config.enable_vectorization = false;
    
    assert!(config.aggressive_inlining);
    assert!(!config.enable_vectorization);
}

#[test]
fn test_optimize_with_config() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a simple MLIR module
    let context = MLIRContext::new_mock();
    let module = context.create_module("test_module").expect("Failed to create module");
    
    // Generate LLVM IR
    codegen.generate_from_mlir(&module).expect("Failed to generate LLVM IR");
    
    // Test optimization with custom config
    let config = OptimizationConfig {
        enable_inlining: true,
        aggressive_inlining: true,
        enable_vectorization: false,
        enable_loop_optimizations: true,
        enable_memory_optimizations: true,
        enable_global_optimizations: false,
        enable_control_flow_optimizations: true,
        target_specific_optimizations: false,
    };
    
    let result = codegen.optimize_with_config(config);
    assert!(result.is_ok(), "Should be able to optimize with custom config");
    
    let stats = codegen.get_statistics();
    assert!(stats.translation_successful);
}

#[test]
fn test_lto_configuration() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Test different LTO types
    let result = codegen.enable_lto(LTOType::None);
    assert!(result.is_ok(), "Should be able to disable LTO");
    
    let result = codegen.enable_lto(LTOType::Thin);
    assert!(result.is_ok(), "Should be able to enable thin LTO");
    
    let result = codegen.enable_lto(LTOType::Full);
    assert!(result.is_ok(), "Should be able to enable full LTO");
}

#[test]
fn test_pgo_configuration() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Test with non-existent profile data
    let result = codegen.enable_pgo("non_existent_profile.profdata");
    assert!(result.is_err(), "Should fail with non-existent profile data");
    
    match result {
        Err(CodegenError::OptimizationError(msg)) => {
            assert!(msg.contains("Profile data file not found"), "Error should mention missing profile data");
        }
        _ => panic!("Expected OptimizationError"),
    }
}

#[test]
fn test_target_architecture_detection() {
    let mut config = TargetConfig::default();
    
    // Test x86_64
    config.triple = "x86_64-unknown-linux-gnu".to_string();
    let codegen = LLVMCodeGenerator::new(config.clone()).expect("Failed to create codegen");
    assert_eq!(codegen.get_target_architecture(), "x86_64");
    
    // Test aarch64
    config.triple = "aarch64-unknown-linux-gnu".to_string();
    let codegen = LLVMCodeGenerator::new(config.clone()).expect("Failed to create codegen");
    assert_eq!(codegen.get_target_architecture(), "aarch64");
    
    // Test unknown
    config.triple = "unknown-unknown-unknown".to_string();
    let codegen = LLVMCodeGenerator::new(config).expect("Failed to create codegen");
    assert_eq!(codegen.get_target_architecture(), "unknown");
}

#[test]
fn test_lto_type_values() {
    assert_eq!(LTOType::None, LTOType::None);
    assert_eq!(LTOType::Thin, LTOType::Thin);
    assert_eq!(LTOType::Full, LTOType::Full);
    
    assert_ne!(LTOType::None, LTOType::Thin);
    assert_ne!(LTOType::Thin, LTOType::Full);
}

#[test]
fn test_cross_compilation_configuration() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Test cross-compilation to different targets
    let result = codegen.configure_cross_compilation(
        "aarch64-unknown-linux-gnu",
        "generic",
        "+neon"
    );
    assert!(result.is_ok(), "Should be able to configure cross-compilation");
    
    let config = codegen.get_target_config();
    assert_eq!(config.triple, "aarch64-unknown-linux-gnu");
    assert_eq!(config.cpu, "generic");
    assert_eq!(config.features, "+neon");
}

#[test]
fn test_supported_targets() {
    let targets = LLVMCodeGenerator::get_supported_targets();
    assert!(!targets.is_empty(), "Should have supported targets");
    
    let target_triples: Vec<String> = targets.iter().map(|t| t.triple.clone()).collect();
    assert!(target_triples.contains(&"x86_64-unknown-linux-gnu".to_string()), "Should support x86_64 Linux");
    assert!(target_triples.contains(&"aarch64-apple-darwin".to_string()), "Should support ARM64 macOS");
    assert!(target_triples.contains(&"wasm32-unknown-unknown".to_string()), "Should support WebAssembly");
}

#[test]
fn test_target_support_check() {
    assert!(LLVMCodeGenerator::is_target_supported("x86_64-unknown-linux-gnu"));
    assert!(LLVMCodeGenerator::is_target_supported("aarch64-apple-darwin"));
    assert!(!LLVMCodeGenerator::is_target_supported("unknown-unknown-unknown"));
}

#[test]
fn test_recommended_cpu_selection() {
    assert_eq!(LLVMCodeGenerator::get_recommended_cpu("x86_64-unknown-linux-gnu"), "x86-64");
    assert_eq!(LLVMCodeGenerator::get_recommended_cpu("aarch64-apple-darwin"), "generic");
    assert_eq!(LLVMCodeGenerator::get_recommended_cpu("i686-pc-windows-msvc"), "i686");
    assert_eq!(LLVMCodeGenerator::get_recommended_cpu("armv7-unknown-linux-gnueabihf"), "cortex-a9");
    assert_eq!(LLVMCodeGenerator::get_recommended_cpu("wasm32-unknown-unknown"), "generic");
    assert_eq!(LLVMCodeGenerator::get_recommended_cpu("unknown-target"), "generic");
}

#[test]
fn test_recommended_features_selection() {
    assert_eq!(LLVMCodeGenerator::get_recommended_features("x86_64-unknown-linux-gnu"), "+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2");
    assert_eq!(LLVMCodeGenerator::get_recommended_features("aarch64-apple-darwin"), "+neon");
    assert_eq!(LLVMCodeGenerator::get_recommended_features("armv7-unknown-linux-gnueabihf"), "+neon,+vfp3");
    assert_eq!(LLVMCodeGenerator::get_recommended_features("unknown-target"), "");
}

#[test]
fn test_cross_config_creation() {
    let config = LLVMCodeGenerator::create_cross_config("x86_64-unknown-linux-gnu", OptimizationLevel::Default).unwrap();
    assert_eq!(config.triple, "x86_64-unknown-linux-gnu");
    assert_eq!(config.cpu, "x86-64");
    assert_eq!(config.features, "+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2");
    assert_eq!(config.optimization_level, OptimizationLevel::Default);
    assert_eq!(config.relocation_model, RelocModel::PIC);
    
    // Test Windows target (should use default relocation model)
    let windows_config = LLVMCodeGenerator::create_cross_config("x86_64-pc-windows-msvc", OptimizationLevel::Aggressive).unwrap();
    assert_eq!(windows_config.relocation_model, RelocModel::Default);
}

#[test]
fn test_object_file_generation_cross_platform() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a simple MLIR module
    let context = MLIRContext::new_mock();
    let module = context.create_module("test_module").expect("Failed to create module");
    
    // Generate LLVM IR
    codegen.generate_from_mlir(&module).expect("Failed to generate LLVM IR");
    
    // Create target machine
    codegen.create_target_machine().expect("Failed to create target machine");
    
    // Test object file generation (would create actual file in real implementation)
    let temp_path = std::path::Path::new("test_output.o");
    let result = codegen.emit_object_file(temp_path);
    assert!(result.is_ok(), "Should be able to generate object file");
}

#[test]
fn test_assembly_generation_cross_platform() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a simple MLIR module
    let context = MLIRContext::new_mock();
    let module = context.create_module("test_module").expect("Failed to create module");
    
    // Generate LLVM IR
    codegen.generate_from_mlir(&module).expect("Failed to generate LLVM IR");
    
    // Create target machine
    codegen.create_target_machine().expect("Failed to create target machine");
    
    // Test assembly generation (would create actual file in real implementation)
    let temp_path = std::path::Path::new("test_output.s");
    let result = codegen.emit_assembly(temp_path);
    assert!(result.is_ok(), "Should be able to generate assembly file");
}

// Target-specific code generation tests

#[test]
fn test_target_info_structure() {
    let targets = LLVMCodeGenerator::get_supported_targets();
    
    for target in targets {
        assert!(!target.triple.is_empty(), "Target triple should not be empty");
        assert!(!target.arch.is_empty(), "Target architecture should not be empty");
        assert!(!target.os.is_empty(), "Target OS should not be empty");
        assert!(!target.default_cpu.is_empty(), "Default CPU should not be empty");
        
        // Verify object format is appropriate for OS
        match target.os.as_str() {
            "linux" => assert_eq!(target.object_format, ObjectFormat::ELF),
            "windows" => assert_eq!(target.object_format, ObjectFormat::COFF),
            "macos" => assert_eq!(target.object_format, ObjectFormat::MachO),
            "unknown" if target.arch == "wasm32" => assert_eq!(target.object_format, ObjectFormat::Wasm),
            _ => {}
        }
    }
}

#[test]
fn test_target_info_lookup() {
    // Test existing targets
    let x86_64_linux = LLVMCodeGenerator::get_target_info("x86_64-unknown-linux-gnu");
    assert!(x86_64_linux.is_some());
    let info = x86_64_linux.unwrap();
    assert_eq!(info.arch, "x86_64");
    assert_eq!(info.os, "linux");
    assert!(info.supports_pic);
    assert_eq!(info.object_format, ObjectFormat::ELF);
    
    // Test non-existent target
    let unknown = LLVMCodeGenerator::get_target_info("unknown-unknown-unknown");
    assert!(unknown.is_none());
}

#[test]
fn test_cross_compilation_config_creation() {
    // Test successful config creation
    let config = LLVMCodeGenerator::create_cross_config("aarch64-apple-darwin", OptimizationLevel::Aggressive);
    assert!(config.is_ok());
    
    let config = config.unwrap();
    assert_eq!(config.triple, "aarch64-apple-darwin");
    assert_eq!(config.cpu, "apple-a14");
    assert!(config.features.contains("+neon"));
    assert_eq!(config.optimization_level, OptimizationLevel::Aggressive);
    assert_eq!(config.relocation_model, RelocModel::PIC);
    
    // Test unsupported target
    let result = LLVMCodeGenerator::create_cross_config("unsupported-target", OptimizationLevel::Default);
    assert!(result.is_err());
    match result {
        Err(CodegenError::TargetMachineError(msg)) => {
            assert!(msg.contains("Unsupported target triple"));
        }
        _ => panic!("Expected TargetMachineError"),
    }
}

#[test]
fn test_target_specific_optimizations() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Test x86_64 optimizations
    let x86_config = LLVMCodeGenerator::create_cross_config("x86_64-unknown-linux-gnu", OptimizationLevel::Default).unwrap();
    codegen.set_target_config(x86_config).expect("Failed to set x86_64 config");
    let result = codegen.configure_target_optimizations();
    assert!(result.is_ok(), "Should configure x86_64 optimizations");
    assert!(codegen.get_target_config().features.contains("+sse"));
    
    // Test AArch64 optimizations
    let arm_config = LLVMCodeGenerator::create_cross_config("aarch64-apple-darwin", OptimizationLevel::Default).unwrap();
    codegen.set_target_config(arm_config).expect("Failed to set AArch64 config");
    let result = codegen.configure_target_optimizations();
    assert!(result.is_ok(), "Should configure AArch64 optimizations");
    assert!(codegen.get_target_config().features.contains("+neon"));
    
    // Test WebAssembly optimizations
    let wasm_config = LLVMCodeGenerator::create_cross_config("wasm32-unknown-unknown", OptimizationLevel::Default).unwrap();
    codegen.set_target_config(wasm_config).expect("Failed to set WebAssembly config");
    let result = codegen.configure_target_optimizations();
    assert!(result.is_ok(), "Should configure WebAssembly optimizations");
    assert_eq!(codegen.get_target_config().relocation_model, RelocModel::Static);
    assert_eq!(codegen.get_target_config().code_model, CodeModel::Small);
}

#[test]
fn test_cross_compilation_validation() {
    let codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Test valid cross-compilation scenarios
    let result = codegen.validate_cross_compilation("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu");
    assert!(result.is_ok(), "Should allow x86_64 to AArch64 cross-compilation");
    
    let result = codegen.validate_cross_compilation("x86_64-unknown-linux-gnu", "wasm32-unknown-unknown");
    assert!(result.is_ok(), "Should allow x86_64 to WebAssembly cross-compilation");
    
    // Test same-architecture compilation
    let result = codegen.validate_cross_compilation("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc");
    assert!(result.is_ok(), "Should allow same-architecture cross-compilation");
    
    // Test unsupported host
    let result = codegen.validate_cross_compilation("unsupported-host", "x86_64-unknown-linux-gnu");
    assert!(result.is_err(), "Should reject unsupported host triple");
    
    // Test unsupported target
    let result = codegen.validate_cross_compilation("x86_64-unknown-linux-gnu", "unsupported-target");
    assert!(result.is_err(), "Should reject unsupported target triple");
}

#[test]
fn test_codegen_settings_creation() {
    // Test default settings
    let settings = CodeGenSettings::default();
    assert!(!settings.debug_info);
    assert!(!settings.emit_debug_symbols);
    assert!(settings.position_independent_code);
    assert_eq!(settings.frame_pointer, FramePointerMode::NonLeaf);
    
    // Test debug settings
    let debug_settings = CodeGenSettings::debug();
    assert!(debug_settings.debug_info);
    assert!(debug_settings.emit_debug_symbols);
    assert!(!debug_settings.strip_symbols);
    assert_eq!(debug_settings.frame_pointer, FramePointerMode::All);
    
    // Test release settings
    let release_settings = CodeGenSettings::release();
    assert!(!release_settings.debug_info);
    assert!(!release_settings.emit_debug_symbols);
    assert!(release_settings.strip_symbols);
    assert_eq!(release_settings.frame_pointer, FramePointerMode::None);
}

#[test]
fn test_target_specific_codegen_settings() {
    let targets = LLVMCodeGenerator::get_supported_targets();
    
    for target in targets {
        let settings = CodeGenSettings::for_target(&target);
        
        match target.os.as_str() {
            "windows" => {
                assert_eq!(settings.exception_handling, ExceptionHandlingMode::WinEH);
                assert!(!settings.position_independent_code);
            }
            "macos" => {
                assert!(settings.position_independent_code);
            }
            "linux" => {
                assert_eq!(settings.position_independent_code, target.supports_pic);
            }
            _ => {}
        }
        
        match target.arch.as_str() {
            "wasm32" => {
                assert!(!settings.position_independent_code);
                assert!(!settings.stack_protection);
                assert!(!settings.thread_local_storage);
                assert_eq!(settings.exception_handling, ExceptionHandlingMode::None);
            }
            "armv7" => {
                assert_eq!(settings.exception_handling, ExceptionHandlingMode::ARM);
            }
            _ => {}
        }
    }
}

#[test]
fn test_object_file_generation_with_settings() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a simple MLIR module
    let context = MLIRContext::new_mock();
    let module = context.create_module("test_module").expect("Failed to create module");
    
    // Generate LLVM IR
    codegen.generate_from_mlir(&module).expect("Failed to generate LLVM IR");
    
    // Create target machine
    codegen.create_target_machine().expect("Failed to create target machine");
    
    // Test object file generation with debug settings
    let debug_settings = CodeGenSettings::debug();
    let temp_path = std::path::Path::new("test_debug.o");
    let result = codegen.emit_object_file_with_settings(temp_path, &debug_settings);
    assert!(result.is_ok(), "Should generate object file with debug settings");
    
    // Test object file generation with release settings
    let release_settings = CodeGenSettings::release();
    let temp_path = std::path::Path::new("test_release.o");
    let result = codegen.emit_object_file_with_settings(temp_path, &release_settings);
    assert!(result.is_ok(), "Should generate object file with release settings");
}

#[test]
fn test_assembly_generation_with_settings() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Create a simple MLIR module
    let context = MLIRContext::new_mock();
    let module = context.create_module("test_module").expect("Failed to create module");
    
    // Generate LLVM IR
    codegen.generate_from_mlir(&module).expect("Failed to generate LLVM IR");
    
    // Create target machine
    codegen.create_target_machine().expect("Failed to create target machine");
    
    // Test assembly generation with debug settings
    let debug_settings = CodeGenSettings::debug();
    let temp_path = std::path::Path::new("test_debug.s");
    let result = codegen.emit_assembly_with_settings(temp_path, &debug_settings);
    assert!(result.is_ok(), "Should generate assembly file with debug settings");
    
    // Test assembly generation with release settings
    let release_settings = CodeGenSettings::release();
    let temp_path = std::path::Path::new("test_release.s");
    let result = codegen.emit_assembly_with_settings(temp_path, &release_settings);
    assert!(result.is_ok(), "Should generate assembly file with release settings");
}

#[test]
fn test_multiple_target_compilation() {
    let targets = vec![
        "x86_64-unknown-linux-gnu",
        "aarch64-apple-darwin",
        "x86_64-pc-windows-msvc",
        "wasm32-unknown-unknown",
    ];
    
    for target_triple in targets {
        let config = LLVMCodeGenerator::create_cross_config(target_triple, OptimizationLevel::Default);
        assert!(config.is_ok(), "Should create config for {}", target_triple);
        
        let mut codegen = LLVMCodeGenerator::new(config.unwrap()).expect("Failed to create codegen");
        
        // Configure target-specific optimizations
        let result = codegen.configure_target_optimizations();
        assert!(result.is_ok(), "Should configure optimizations for {}", target_triple);
        
        // Create target machine
        let result = codegen.create_target_machine();
        assert!(result.is_ok(), "Should create target machine for {}", target_triple);
        
        // Create a simple MLIR module
        let context = MLIRContext::new_mock();
        let module = context.create_module("test_module").expect("Failed to create module");
        
        // Generate LLVM IR
        let result = codegen.generate_from_mlir(&module);
        assert!(result.is_ok(), "Should generate LLVM IR for {}", target_triple);
        
        // Test optimization
        let result = codegen.optimize(OptimizationLevel::Default);
        assert!(result.is_ok(), "Should optimize for {}", target_triple);
    }
}

#[test]
fn test_architecture_specific_features() {
    // Test x86_64 features
    let x86_config = LLVMCodeGenerator::create_cross_config("x86_64-unknown-linux-gnu", OptimizationLevel::Default).unwrap();
    assert!(x86_config.features.contains("+sse"));
    assert!(x86_config.features.contains("+sse2"));
    
    // Test AArch64 features
    let arm_config = LLVMCodeGenerator::create_cross_config("aarch64-unknown-linux-gnu", OptimizationLevel::Default).unwrap();
    assert!(arm_config.features.contains("+neon"));
    
    // Test Apple Silicon features
    let apple_config = LLVMCodeGenerator::create_cross_config("aarch64-apple-darwin", OptimizationLevel::Default).unwrap();
    assert!(apple_config.features.contains("+neon"));
    assert!(apple_config.features.contains("+crypto"));
    
    // Test WebAssembly (minimal features)
    let wasm_config = LLVMCodeGenerator::create_cross_config("wasm32-unknown-unknown", OptimizationLevel::Default).unwrap();
    assert!(wasm_config.features.is_empty());
}

#[test]
fn test_relocation_model_selection() {
    // Test PIC-supporting targets
    let linux_config = LLVMCodeGenerator::create_cross_config("x86_64-unknown-linux-gnu", OptimizationLevel::Default).unwrap();
    assert_eq!(linux_config.relocation_model, RelocModel::PIC);
    
    let macos_config = LLVMCodeGenerator::create_cross_config("x86_64-apple-darwin", OptimizationLevel::Default).unwrap();
    assert_eq!(macos_config.relocation_model, RelocModel::PIC);
    
    // Test non-PIC targets
    let windows_config = LLVMCodeGenerator::create_cross_config("x86_64-pc-windows-msvc", OptimizationLevel::Default).unwrap();
    assert_eq!(windows_config.relocation_model, RelocModel::Default);
    
    let wasm_config = LLVMCodeGenerator::create_cross_config("wasm32-unknown-unknown", OptimizationLevel::Default).unwrap();
    assert_eq!(wasm_config.relocation_model, RelocModel::Default);
}

#[test]
fn test_code_model_selection() {
    // Test default code model
    let x86_config = LLVMCodeGenerator::create_cross_config("x86_64-unknown-linux-gnu", OptimizationLevel::Default).unwrap();
    assert_eq!(x86_config.code_model, CodeModel::Default);
    
    // Test small code model for constrained targets
    let i686_config = LLVMCodeGenerator::create_cross_config("i686-unknown-linux-gnu", OptimizationLevel::Default).unwrap();
    assert_eq!(i686_config.code_model, CodeModel::Small);
    
    let wasm_config = LLVMCodeGenerator::create_cross_config("wasm32-unknown-unknown", OptimizationLevel::Default).unwrap();
    assert_eq!(wasm_config.code_model, CodeModel::Small);
}

#[test]
fn test_cross_compilation_error_handling() {
    let mut codegen = LLVMCodeGenerator::new_default().expect("Failed to create codegen");
    
    // Test invalid target configuration
    let invalid_config = TargetConfig {
        triple: "invalid-target-triple".to_string(),
        cpu: "invalid-cpu".to_string(),
        features: "invalid-features".to_string(),
        optimization_level: OptimizationLevel::Default,
        relocation_model: RelocModel::Default,
        code_model: CodeModel::Default,
    };
    
    codegen.set_target_config(invalid_config).expect("Should set invalid config");
    
    // This should fail when trying to create target machine
    let result = codegen.create_target_machine();
    assert!(result.is_err(), "Should fail to create target machine with invalid config");
    
    match result {
        Err(CodegenError::TargetMachineError(_)) => {
            // Expected error type
        }
        _ => panic!("Expected TargetMachineError"),
    }
}

#[test]
fn test_frame_pointer_modes() {
    assert_eq!(FramePointerMode::None, FramePointerMode::None);
    assert_eq!(FramePointerMode::NonLeaf, FramePointerMode::NonLeaf);
    assert_eq!(FramePointerMode::All, FramePointerMode::All);
    
    assert_ne!(FramePointerMode::None, FramePointerMode::All);
}

#[test]
fn test_exception_handling_modes() {
    assert_eq!(ExceptionHandlingMode::None, ExceptionHandlingMode::None);
    assert_eq!(ExceptionHandlingMode::DWARF, ExceptionHandlingMode::DWARF);
    assert_eq!(ExceptionHandlingMode::WinEH, ExceptionHandlingMode::WinEH);
    assert_eq!(ExceptionHandlingMode::ARM, ExceptionHandlingMode::ARM);
    
    assert_ne!(ExceptionHandlingMode::DWARF, ExceptionHandlingMode::WinEH);
}

#[test]
fn test_object_format_detection() {
    let targets = LLVMCodeGenerator::get_supported_targets();
    
    for target in targets {
        match target.triple.as_str() {
            triple if triple.contains("linux") => {
                assert_eq!(target.object_format, ObjectFormat::ELF);
            }
            triple if triple.contains("windows") => {
                assert_eq!(target.object_format, ObjectFormat::COFF);
            }
            triple if triple.contains("darwin") => {
                assert_eq!(target.object_format, ObjectFormat::MachO);
            }
            triple if triple.contains("wasm32") => {
                assert_eq!(target.object_format, ObjectFormat::Wasm);
            }
            _ => {}
        }
    }
}