// Tests for debugging utilities and IR dumping functionality

#[cfg(test)]
mod tests {
    use crate::compiler::mlir::debug_utils::*;
    use crate::compiler::mlir::{MLIRModule, MLIROperation, MLIRValue, MLIRType, MLIRAttribute};
    use crate::compiler::mlir::error_handling::{MLIRCompilationError, SourceLocation};
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::Duration;
    use tempfile::TempDir;

    fn create_test_module() -> MLIRModule {
        let mut module = MLIRModule::new("test_module".to_string());
        
        // Add some test operations
        let mut op1 = MLIROperation::new("aether.tensor_create".to_string());
        op1.operands = vec![MLIRValue::new("input1".to_string(), MLIRType::Float { width: 32 })];
        op1.results = vec![MLIRValue::new("result1".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 3],
            device: "cpu".to_string(),
        })];
        op1.attributes.insert("shape".to_string(), MLIRAttribute::String("[2, 3]".to_string()));
        op1.source_location = Some(SourceLocation::new(
            Some(PathBuf::from("test.ae")), 10, 5, 100, 20
        ));
        
        let mut op2 = MLIROperation::new("aether.matmul".to_string());
        op2.operands = vec![
            MLIRValue::new("input1".to_string(), MLIRType::AetherTensor {
                element_type: Box::new(MLIRType::Float { width: 32 }),
                shape: vec![2, 3],
                device: "cpu".to_string(),
            }),
            MLIRValue::new("input2".to_string(), MLIRType::AetherTensor {
                element_type: Box::new(MLIRType::Float { width: 32 }),
                shape: vec![3, 4],
                device: "cpu".to_string(),
            })
        ];
        op2.results = vec![MLIRValue::new("result2".to_string(), MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }),
            shape: vec![2, 4],
            device: "cpu".to_string(),
        })];
        op2.attributes.insert("transpose_a".to_string(), MLIRAttribute::Boolean(false));
        
        module.set_operations(vec![op1, op2]);
        module
    }

    #[test]
    fn test_debug_config_default() {
        let config = DebugConfig::default();
        
        assert!(!config.dump_ir);
        assert_eq!(config.dump_dir, PathBuf::from("debug_output"));
        assert!(!config.enable_timing);
        assert!(!config.verbose_errors);
        assert!(config.dump_stages.is_empty());
        assert_eq!(config.max_operations_per_dump, 1000);
        assert!(config.include_source_locations);
    }

    #[test]
    fn test_debug_config_development() {
        let config = DebugConfig::development();
        
        assert!(config.dump_ir);
        assert!(config.enable_timing);
        assert!(config.verbose_errors);
        assert_eq!(config.dump_stages.len(), 4);
        assert!(config.dump_stages.contains(&CompilationStage::ASTToMLIR));
        assert!(config.dump_stages.contains(&CompilationStage::Optimization));
        assert_eq!(config.max_operations_per_dump, 500);
    }

    #[test]
    fn test_debug_config_production() {
        let config = DebugConfig::production();
        
        assert!(!config.dump_ir);
        assert!(config.enable_timing);
        assert!(!config.verbose_errors);
        assert!(config.dump_stages.is_empty());
        assert_eq!(config.max_operations_per_dump, 100);
        assert!(!config.include_source_locations);
    }

    #[test]
    fn test_debug_config_all_features() {
        let config = DebugConfig::all_features();
        
        assert!(config.dump_ir);
        assert!(config.enable_timing);
        assert!(config.verbose_errors);
        assert_eq!(config.dump_stages.len(), 6);
        assert_eq!(config.max_operations_per_dump, 2000);
        assert!(config.include_source_locations);
    }

    #[test]
    fn test_compilation_stage_names() {
        assert_eq!(CompilationStage::ASTToMLIR.name(), "ast_to_mlir");
        assert_eq!(CompilationStage::Optimization.name(), "optimization");
        assert_eq!(CompilationStage::Lowering.name(), "lowering");
        assert_eq!(CompilationStage::LLVMGeneration.name(), "llvm_generation");
        assert_eq!(CompilationStage::LLVMOptimization.name(), "llvm_optimization");
        assert_eq!(CompilationStage::CodeGeneration.name(), "code_generation");
    }

    #[test]
    fn test_compilation_stage_descriptions() {
        assert_eq!(CompilationStage::ASTToMLIR.description(), "AST to MLIR Conversion");
        assert_eq!(CompilationStage::Optimization.description(), "High-level Optimization");
        assert_eq!(CompilationStage::Lowering.description(), "Dialect Lowering");
        assert_eq!(CompilationStage::LLVMGeneration.description(), "LLVM IR Generation");
        assert_eq!(CompilationStage::LLVMOptimization.description(), "LLVM Optimization");
        assert_eq!(CompilationStage::CodeGeneration.description(), "Code Generation");
    }

    #[test]
    fn test_compilation_stage_all_stages() {
        let stages = CompilationStage::all_stages();
        assert_eq!(stages.len(), 6);
        assert!(stages.contains(&CompilationStage::ASTToMLIR));
        assert!(stages.contains(&CompilationStage::Optimization));
        assert!(stages.contains(&CompilationStage::Lowering));
        assert!(stages.contains(&CompilationStage::LLVMGeneration));
        assert!(stages.contains(&CompilationStage::LLVMOptimization));
        assert!(stages.contains(&CompilationStage::CodeGeneration));
    }

    #[test]
    fn test_timing_info_creation() {
        let start = std::time::Instant::now();
        std::thread::sleep(Duration::from_millis(10));
        let end = std::time::Instant::now();
        
        let timing = TimingInfo::new(CompilationStage::ASTToMLIR, start, end);
        
        assert_eq!(timing.stage, CompilationStage::ASTToMLIR);
        assert!(timing.duration_ms() >= 10.0);
        assert!(timing.metadata.is_empty());
    }

    #[test]
    fn test_timing_info_with_metadata() {
        let start = std::time::Instant::now();
        let end = std::time::Instant::now();
        
        let timing = TimingInfo::new(CompilationStage::Optimization, start, end)
            .with_metadata("operations_processed".to_string(), "42".to_string())
            .with_metadata("passes_run".to_string(), "5".to_string());
        
        assert_eq!(timing.metadata.len(), 2);
        assert_eq!(timing.metadata.get("operations_processed"), Some(&"42".to_string()));
        assert_eq!(timing.metadata.get("passes_run"), Some(&"5".to_string()));
    }

    #[test]
    fn test_compilation_debugger_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = DebugConfig::default();
        config.dump_ir = true;
        config.dump_dir = temp_dir.path().to_path_buf();
        
        let debugger = CompilationDebugger::new(config);
        assert!(debugger.is_ok());
        
        let debugger = debugger.unwrap();
        assert_eq!(debugger.timing_info.len(), 0);
        assert!(debugger.current_stage.is_none());
        assert_eq!(debugger.dump_counter, 0);
    }

    #[test]
    fn test_compilation_debugger_timing() {
        let config = DebugConfig::development();
        let mut debugger = CompilationDebugger::new(config).unwrap();
        
        // Start and end a stage
        debugger.start_stage(CompilationStage::ASTToMLIR);
        std::thread::sleep(Duration::from_millis(5));
        let timing = debugger.end_stage();
        
        assert!(timing.is_some());
        let timing = timing.unwrap();
        assert_eq!(timing.stage, CompilationStage::ASTToMLIR);
        assert!(timing.duration_ms() >= 5.0);
        
        assert_eq!(debugger.timing_info.len(), 1);
        assert!(debugger.current_stage.is_none());
    }

    #[test]
    fn test_compilation_debugger_timing_with_metadata() {
        let config = DebugConfig::development();
        let mut debugger = CompilationDebugger::new(config).unwrap();
        
        debugger.start_stage(CompilationStage::Optimization);
        
        let mut metadata = HashMap::new();
        metadata.insert("passes".to_string(), "3".to_string());
        metadata.insert("operations".to_string(), "15".to_string());
        
        let timing = debugger.end_stage_with_metadata(metadata);
        
        assert!(timing.is_some());
        let timing = timing.unwrap();
        assert_eq!(timing.metadata.len(), 2);
        assert_eq!(timing.metadata.get("passes"), Some(&"3".to_string()));
    }

    #[test]
    fn test_mlir_module_dumping() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = DebugConfig::development();
        config.dump_dir = temp_dir.path().to_path_buf();
        
        let mut debugger = CompilationDebugger::new(config).unwrap();
        let module = create_test_module();
        
        let result = debugger.dump_mlir_module(&module, CompilationStage::ASTToMLIR);
        assert!(result.is_ok());
        
        let filepath = result.unwrap();
        assert!(filepath.is_some());
        
        let filepath = filepath.unwrap();
        assert!(filepath.exists());
        assert!(filepath.file_name().unwrap().to_str().unwrap().contains("ast_to_mlir"));
        
        // Check file content
        let content = std::fs::read_to_string(&filepath).unwrap();
        assert!(content.contains("MLIR Dump - Stage: AST to MLIR Conversion"));
        assert!(content.contains("aether.tensor_create"));
        assert!(content.contains("aether.matmul"));
        assert!(content.contains("test.ae:10:5"));
    }

    #[test]
    fn test_mlir_module_dumping_disabled() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = DebugConfig::default(); // dump_ir is false
        config.dump_dir = temp_dir.path().to_path_buf();
        
        let mut debugger = CompilationDebugger::new(config).unwrap();
        let module = create_test_module();
        
        let result = debugger.dump_mlir_module(&module, CompilationStage::ASTToMLIR);
        assert!(result.is_ok());
        
        let filepath = result.unwrap();
        assert!(filepath.is_none());
    }

    #[test]
    fn test_mlir_module_dumping_stage_filter() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = DebugConfig::development();
        config.dump_dir = temp_dir.path().to_path_buf();
        config.dump_stages = vec![CompilationStage::Optimization]; // Only optimization stage
        
        let mut debugger = CompilationDebugger::new(config).unwrap();
        let module = create_test_module();
        
        // Should not dump AST to MLIR stage
        let result = debugger.dump_mlir_module(&module, CompilationStage::ASTToMLIR);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        
        // Should dump optimization stage
        let result = debugger.dump_mlir_module(&module, CompilationStage::Optimization);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_llvm_ir_dumping() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = DebugConfig::development();
        config.dump_dir = temp_dir.path().to_path_buf();
        
        let mut debugger = CompilationDebugger::new(config).unwrap();
        let ir = r#"
define i32 @main() {
entry:
  ret i32 0
}
"#;
        
        let result = debugger.dump_llvm_ir(ir, CompilationStage::LLVMGeneration);
        assert!(result.is_ok());
        
        let filepath = result.unwrap();
        assert!(filepath.is_some());
        
        let filepath = filepath.unwrap();
        assert!(filepath.exists());
        assert!(filepath.extension().unwrap() == "ll");
        
        let content = std::fs::read_to_string(&filepath).unwrap();
        assert!(content.contains("LLVM IR Dump - Stage: LLVM IR Generation"));
        assert!(content.contains("define i32 @main()"));
    }

    #[test]
    fn test_error_context_management() {
        let config = DebugConfig::development();
        let mut debugger = CompilationDebugger::new(config).unwrap();
        
        assert!(debugger.error_context.is_empty());
        
        debugger.add_error_context("Parsing AST node".to_string());
        debugger.add_error_context("Converting to MLIR".to_string());
        
        assert_eq!(debugger.error_context.len(), 2);
        assert_eq!(debugger.error_context[0], "Parsing AST node");
        assert_eq!(debugger.error_context[1], "Converting to MLIR");
        
        debugger.clear_error_context();
        assert!(debugger.error_context.is_empty());
    }

    #[test]
    fn test_verbose_error_report() {
        let config = DebugConfig::development();
        let mut debugger = CompilationDebugger::new(config).unwrap();
        
        // Add some timing and context
        debugger.start_stage(CompilationStage::ASTToMLIR);
        std::thread::sleep(Duration::from_millis(5));
        debugger.end_stage();
        
        debugger.add_error_context("Processing function definition".to_string());
        debugger.add_error_context("Converting tensor operation".to_string());
        
        let error = MLIRCompilationError::ASTConversion {
            node_id: Some(42),
            node_type: "TensorOp".to_string(),
            error: "Unsupported tensor operation".to_string(),
            location: SourceLocation::new(Some(PathBuf::from("test.ae")), 15, 8, 150, 25),
            context: vec!["function_body".to_string()],
        };
        
        let report = debugger.create_verbose_error_report(&error);
        
        assert!(report.contains("=== MLIR Compilation Error Report ==="));
        assert!(report.contains("Error: Failed to convert TensorOp AST node"));
        assert!(report.contains("Severity: error"));
        assert!(report.contains("Location: test.ae:15:8"));
        assert!(report.contains("Context:"));
        assert!(report.contains("Processing function definition"));
        assert!(report.contains("Converting tensor operation"));
        assert!(report.contains("Recovery Suggestions:"));
        assert!(report.contains("Compilation Timing:"));
        assert!(report.contains("AST to MLIR Conversion:"));
        assert!(report.contains("Debug Information:"));
    }

    #[test]
    fn test_verbose_error_report_disabled() {
        let mut config = DebugConfig::default();
        config.verbose_errors = false;
        let debugger = CompilationDebugger::new(config).unwrap();
        
        let error = MLIRCompilationError::ASTConversion {
            node_id: Some(42),
            node_type: "TensorOp".to_string(),
            error: "Unsupported tensor operation".to_string(),
            location: SourceLocation::unknown(),
            context: vec![],
        };
        
        let report = debugger.create_verbose_error_report(&error);
        
        // Should just return the basic error string
        assert!(!report.contains("=== MLIR Compilation Error Report ==="));
        assert!(report.contains("Failed to convert TensorOp AST node"));
    }

    #[test]
    fn test_timing_summary() {
        let config = DebugConfig::development();
        let mut debugger = CompilationDebugger::new(config).unwrap();
        
        // Add multiple stages with different durations
        debugger.start_stage(CompilationStage::ASTToMLIR);
        std::thread::sleep(Duration::from_millis(10));
        debugger.end_stage();
        
        debugger.start_stage(CompilationStage::Optimization);
        std::thread::sleep(Duration::from_millis(5));
        debugger.end_stage();
        
        debugger.start_stage(CompilationStage::Lowering);
        std::thread::sleep(Duration::from_millis(15));
        debugger.end_stage();
        
        let summary = debugger.get_timing_summary();
        
        assert!(summary.total_duration_ms() >= 30.0);
        assert_eq!(summary.stage_timings.len(), 3);
        
        assert!(summary.stage_duration_ms(CompilationStage::ASTToMLIR) >= 10.0);
        assert!(summary.stage_duration_ms(CompilationStage::Optimization) >= 5.0);
        assert!(summary.stage_duration_ms(CompilationStage::Lowering) >= 15.0);
        
        // Lowering should be the slowest (15ms)
        assert_eq!(summary.slowest_stage, Some(CompilationStage::Lowering));
        // Optimization should be the fastest (5ms)
        assert_eq!(summary.fastest_stage, Some(CompilationStage::Optimization));
    }

    #[test]
    fn test_timing_summary_empty() {
        let config = DebugConfig::development();
        let debugger = CompilationDebugger::new(config).unwrap();
        
        let summary = debugger.get_timing_summary();
        
        assert_eq!(summary.total_duration_ms(), 0.0);
        assert!(summary.stage_timings.is_empty());
        assert!(summary.slowest_stage.is_none());
        assert!(summary.fastest_stage.is_none());
    }

    #[test]
    fn test_compilation_report_generation() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = DebugConfig::development();
        config.dump_dir = temp_dir.path().to_path_buf();
        
        let mut debugger = CompilationDebugger::new(config.clone()).unwrap();
        
        // Add some timing and context
        debugger.start_stage(CompilationStage::ASTToMLIR);
        std::thread::sleep(Duration::from_millis(5));
        debugger.end_stage();
        
        debugger.add_error_context("Test context".to_string());
        
        // Create a dump to increment counter
        let module = create_test_module();
        debugger.dump_mlir_module(&module, CompilationStage::ASTToMLIR).unwrap();
        
        let report = debugger.generate_compilation_report();
        
        assert!(report.timing_summary.total_duration_ms() >= 5.0);
        assert_eq!(report.dumps_created, 1);
        assert_eq!(report.dump_directory, temp_dir.path());
        assert_eq!(report.error_context.len(), 1);
        assert_eq!(report.error_context[0], "Test context");
        assert!(report.config.dump_ir);
    }

    #[test]
    fn test_compilation_report_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = DebugConfig::development();
        config.dump_dir = temp_dir.path().to_path_buf();
        
        let mut debugger = CompilationDebugger::new(config).unwrap();
        
        // Add some data
        debugger.start_stage(CompilationStage::Optimization);
        std::thread::sleep(Duration::from_millis(5));
        debugger.end_stage();
        
        let report_path = temp_dir.path().join("report.json");
        let result = debugger.save_compilation_report(&report_path);
        assert!(result.is_ok());
        
        assert!(report_path.exists());
        
        // Verify the file contains valid JSON
        let content = std::fs::read_to_string(&report_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        
        assert!(parsed.get("timing_summary").is_some());
        assert!(parsed.get("dumps_created").is_some());
        assert!(parsed.get("dump_directory").is_some());
        assert!(parsed.get("error_context").is_some());
        assert!(parsed.get("config").is_some());
    }

    #[test]
    fn test_operations_limit_in_dump() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = DebugConfig::development();
        config.dump_dir = temp_dir.path().to_path_buf();
        config.max_operations_per_dump = 1; // Limit to 1 operation
        
        let mut debugger = CompilationDebugger::new(config).unwrap();
        let module = create_test_module(); // Has 2 operations
        
        let result = debugger.dump_mlir_module(&module, CompilationStage::ASTToMLIR);
        assert!(result.is_ok());
        
        let filepath = result.unwrap().unwrap();
        let content = std::fs::read_to_string(&filepath).unwrap();
        
        // Should contain only the first operation
        assert!(content.contains("aether.tensor_create"));
        assert!(!content.contains("aether.matmul"));
        assert!(content.contains("Total operations dumped: 1"));
        assert!(content.contains("Output truncated at 1 operations"));
    }

    #[test]
    fn test_source_locations_in_dump() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = DebugConfig::development();
        config.dump_dir = temp_dir.path().to_path_buf();
        config.include_source_locations = false;
        
        let mut debugger = CompilationDebugger::new(config).unwrap();
        let module = create_test_module();
        
        let result = debugger.dump_mlir_module(&module, CompilationStage::ASTToMLIR);
        assert!(result.is_ok());
        
        let filepath = result.unwrap().unwrap();
        let content = std::fs::read_to_string(&filepath).unwrap();
        
        // Should not contain source location information
        assert!(!content.contains("Source: test.ae:10:5"));
    }
}