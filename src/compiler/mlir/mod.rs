// MLIR-based compilation pipeline for Aether
// Provides frontend stage converting AST to Aether MLIR dialect

pub mod bindings;
pub mod dialect;
pub mod dialect_registration;
pub mod frontend;
pub mod lowering;
pub mod optimization;
pub mod gpu_dialect;
pub mod gpu_lowering;
pub mod gpu_codegen;
pub mod gpu_tensor_ops;
pub mod mlir_context;
pub mod test_utils;
pub mod concurrency_dialect;
pub mod aether_types;
pub mod llvm_codegen;
pub mod error_handling;
pub mod error_recovery;
pub mod debug_utils;
pub mod minimal_mlir;
pub mod enhanced_mlir_integration;

#[cfg(test)]
pub mod aether_types_tests;
#[cfg(test)]
pub mod dialect_registration_tests;
#[cfg(test)]
pub mod frontend_tests;
#[cfg(test)]
pub mod lowering_tests;
#[cfg(test)]
pub mod standard_mlir_tests;
#[cfg(test)]
pub mod llvm_codegen_tests;
#[cfg(test)]
pub mod error_handling_tests;
#[cfg(test)]
pub mod debug_utils_tests;
#[cfg(test)]
pub mod comprehensive_unit_tests;
#[cfg(test)]
pub mod optimization_unit_tests;
#[cfg(test)]
pub mod type_system_unit_tests;
#[cfg(test)]
pub mod test_runner;
#[cfg(test)]
pub mod minimal_mlir_integration_test;

pub use bindings::*;
pub use dialect::*;
pub use dialect_registration::*;
pub use frontend::*;
pub use lowering::*;
pub use optimization::{
    OptimizationPass, PassResult, PassConfig, OptimizationPipeline, PipelineResult, PipelineStatistics,
    AetherOptimizer, AetherPassManager, OperatorFusionPass, MemoryTilingPass, AutomaticParallelizationPass,
    TensorLayoutOptimizationPass, AutodiffOptimizationPass
};
pub use gpu_dialect::*;
pub use gpu_lowering::*;
pub use gpu_codegen::*;
pub use gpu_tensor_ops::*;
pub use mlir_context::*;
pub use concurrency_dialect::*;
pub use aether_types::*;
pub use llvm_codegen::*;
pub use error_handling::*;
pub use error_recovery::*;
pub use debug_utils::*;
pub use minimal_mlir::*;
pub use enhanced_mlir_integration::*;

// Mock implementations removed - using real MLIR implementations

use crate::compiler::ast::AST;
// MLIRAttribute is already re-exported through pub use mlir_context::*;



/// MLIR compilation pipeline for Aether
pub struct MLIRPipeline {
    context: MLIRContext,
    debugger: Option<CompilationDebugger>,
}

impl MLIRPipeline {
    /// Create new MLIR pipeline
    pub fn new() -> Result<Self, MLIRError> {
        let context = match MLIRContext::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                #[cfg(not(feature = "mlir"))]
                {
                    // In stub mode, create a mock context
                    eprintln!("Warning: Using mock MLIR context: {}", e);
                    MLIRContext::new_mock()
                }
                #[cfg(feature = "mlir")]
                return Err(e);
            }
        };
        
        Ok(MLIRPipeline { 
            context,
            debugger: None,
        })
    }

    /// Create new MLIR pipeline with debug configuration
    pub fn new_with_debug(debug_config: DebugConfig) -> Result<Self, MLIRError> {
        let context = MLIRContext::new()?;
        let debugger = CompilationDebugger::new(debug_config)
            .map_err(|e| MLIRError::ContextCreationError(format!("Failed to create debugger: {}", e)))?;
        Ok(MLIRPipeline { 
            context,
            debugger: Some(debugger),
        })
    }

    /// Enable debugging with configuration
    pub fn enable_debug(&mut self, debug_config: DebugConfig) -> Result<(), MLIRError> {
        let debugger = CompilationDebugger::new(debug_config)
            .map_err(|e| MLIRError::ContextCreationError(format!("Failed to create debugger: {}", e)))?;
        self.debugger = Some(debugger);
        Ok(())
    }

    /// Disable debugging
    pub fn disable_debug(&mut self) {
        self.debugger = None;
    }

    /// Get mutable reference to debugger
    pub fn debugger_mut(&mut self) -> Option<&mut CompilationDebugger> {
        self.debugger.as_mut()
    }

    /// Get reference to debugger
    pub fn debugger(&self) -> Option<&CompilationDebugger> {
        self.debugger.as_ref()
    }

    /// Generate LLVM IR from MLIR module with debug support
    pub fn generate_llvm_ir(&mut self, module: &MLIRModule) -> Result<String, MLIRError> {
        use crate::compiler::mlir::llvm_codegen::{LLVMCodeGenerator, TargetConfig, OptimizationLevel};
        
        // Create LLVM code generator
        let mut codegen = LLVMCodeGenerator::new(TargetConfig::default())
            .map_err(|e| MLIRError::OperationError(format!("Failed to create LLVM codegen: {}", e)))?;

        // Generate LLVM IR with debug support
        codegen.generate_from_mlir_with_debug(module, self.debugger.as_mut())
            .map_err(|e| MLIRError::OperationError(format!("LLVM IR generation failed: {}", e)))?;

        // Optimize with debug support
        codegen.optimize_with_debug(OptimizationLevel::Default, self.debugger.as_mut())
            .map_err(|e| MLIRError::OptimizationError(format!("LLVM optimization failed: {}", e)))?;

        // Get IR string for return
        codegen.get_llvm_ir_string()
            .map_err(|e| MLIRError::OperationError(format!("Failed to get LLVM IR string: {}", e)))
    }

    /// Generate object file from MLIR module with debug support
    pub fn generate_object_file(&mut self, module: &MLIRModule, output_path: &std::path::Path) -> Result<(), MLIRError> {
        use crate::compiler::mlir::llvm_codegen::{LLVMCodeGenerator, TargetConfig, OptimizationLevel};
        
        // Create LLVM code generator
        let mut codegen = LLVMCodeGenerator::new(TargetConfig::default())
            .map_err(|e| MLIRError::OperationError(format!("Failed to create LLVM codegen: {}", e)))?;

        // Generate LLVM IR with debug support
        codegen.generate_from_mlir_with_debug(module, self.debugger.as_mut())
            .map_err(|e| MLIRError::OperationError(format!("LLVM IR generation failed: {}", e)))?;

        // Optimize with debug support
        codegen.optimize_with_debug(OptimizationLevel::Default, self.debugger.as_mut())
            .map_err(|e| MLIRError::OptimizationError(format!("LLVM optimization failed: {}", e)))?;

        // Generate object file with debug support
        codegen.emit_object_file_with_debug(output_path, self.debugger.as_mut())
            .map_err(|e| MLIRError::OperationError(format!("Object file generation failed: {}", e)))?;

        Ok(())
    }

    /// Get compilation report if debugging is enabled
    pub fn get_compilation_report(&self) -> Option<CompilationReport> {
        self.debugger.as_ref().map(|d| d.generate_compilation_report())
    }

    /// Save compilation report to file if debugging is enabled
    pub fn save_compilation_report(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        if let Some(debugger) = &self.debugger {
            debugger.save_compilation_report(path)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Debugging is not enabled"
            ))
        }
    }

    /// Get timing summary if debugging is enabled
    pub fn get_timing_summary(&self) -> Option<TimingSummary> {
        self.debugger.as_ref().map(|d| d.get_timing_summary())
    }

    /// Create verbose error report if debugging is enabled
    pub fn create_verbose_error_report(&self, error: &MLIRCompilationError) -> String {
        if let Some(debugger) = &self.debugger {
            debugger.create_verbose_error_report(error)
        } else {
            error.to_string()
        }
    }

    /// Compile AST to MLIR module
    pub fn compile_ast(&mut self, ast: &AST) -> Result<MLIRModule, MLIRError> {
        // Start timing AST to MLIR conversion
        if let Some(debugger) = &mut self.debugger {
            debugger.start_stage(CompilationStage::ASTToMLIR);
            debugger.add_error_context("Starting AST to MLIR conversion".to_string());
        }

        // Create a new module
        let mut module = self.context.create_module("aether_module")?;

        // Convert AST to Aether MLIR dialect
        let mut frontend = AetherFrontend::new(&self.context);
        frontend.convert_ast_to_module(ast, &mut module)?;

        // End timing and dump IR
        if let Some(debugger) = &mut self.debugger {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("ast_nodes".to_string(), ast.nodes.len().to_string());
            metadata.insert("module_name".to_string(), "aether_module".to_string());
            debugger.end_stage_with_metadata(metadata);
            debugger.dump_mlir_module(&module, CompilationStage::ASTToMLIR)?;
            debugger.clear_error_context();
        }

        // Start optimization stage
        if let Some(debugger) = &mut self.debugger {
            debugger.start_stage(CompilationStage::Optimization);
            debugger.add_error_context("Starting optimization passes".to_string());
        }

        // Apply optimization passes
        let optimizer = AetherOptimizer::new(&self.context);
        optimizer.optimize(&mut module)?;

        // End optimization timing and dump IR
        if let Some(debugger) = &mut self.debugger {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("operations_count".to_string(), module.get_operations().len().to_string());
            debugger.end_stage_with_metadata(metadata);
            debugger.dump_mlir_module(&module, CompilationStage::Optimization)?;
            debugger.clear_error_context();
        }

        Ok(module)
    }

    /// Lower Aether MLIR to standard dialects
    pub fn lower_to_standard(&mut self, module: &mut MLIRModule) -> Result<(), MLIRError> {
        // Start timing lowering stage
        if let Some(debugger) = &mut self.debugger {
            debugger.start_stage(CompilationStage::Lowering);
            debugger.add_error_context("Starting dialect lowering to standard dialects".to_string());
        }

        let lowering = AetherLowering::new(&self.context);
        let result = lowering.lower_to_standard(module);

        // End timing and dump IR
        if let Some(debugger) = &mut self.debugger {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("target_dialect".to_string(), "standard".to_string());
            metadata.insert("operations_count".to_string(), module.get_operations().len().to_string());
            debugger.end_stage_with_metadata(metadata);
            if result.is_ok() {
                debugger.dump_mlir_module(module, CompilationStage::Lowering)?;
            }
            debugger.clear_error_context();
        }

        result
    }

    /// Lower Aether MLIR to WebAssembly-compatible dialects
    pub fn lower_to_wasm_dialects(&mut self, module: &mut MLIRModule) -> Result<(), MLIRError> {
        // Start timing lowering stage
        if let Some(debugger) = &mut self.debugger {
            debugger.start_stage(CompilationStage::Lowering);
            debugger.add_error_context("Starting dialect lowering to WebAssembly dialects".to_string());
        }

        let lowering = AetherLowering::new(&self.context);
        let result = lowering.lower_to_wasm_dialects(module);

        // End timing and dump IR
        if let Some(debugger) = &mut self.debugger {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("target_dialect".to_string(), "wasm".to_string());
            metadata.insert("operations_count".to_string(), module.get_operations().len().to_string());
            debugger.end_stage_with_metadata(metadata);
            if result.is_ok() {
                debugger.dump_mlir_module(module, CompilationStage::Lowering)?;
            }
            debugger.clear_error_context();
        }

        result
    }

    /// Lower Aether MLIR to GPU kernels
    pub fn lower_to_gpu_kernels(&mut self, module: &mut MLIRModule, target: GpuTarget) -> Result<(), MLIRError> {
        // Start timing lowering stage
        if let Some(debugger) = &mut self.debugger {
            debugger.start_stage(CompilationStage::Lowering);
            debugger.add_error_context(format!("Starting dialect lowering to GPU kernels for target {:?}", target));
        }

        let gpu_lowering = GpuLowering::new(&self.context, target);
        let result = gpu_lowering.lower_to_gpu_kernels(module);

        // End timing and dump IR
        if let Some(debugger) = &mut self.debugger {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("target_dialect".to_string(), "gpu".to_string());
            metadata.insert("gpu_target".to_string(), format!("{:?}", target));
            metadata.insert("operations_count".to_string(), module.get_operations().len().to_string());
            debugger.end_stage_with_metadata(metadata);
            if result.is_ok() {
                debugger.dump_mlir_module(module, CompilationStage::Lowering)?;
            }
            debugger.clear_error_context();
        }

        result.map_err(|e| e.into())
    }

    /// Generate GPU kernel code
    pub fn generate_gpu_code(&self, module: &MLIRModule, target: GpuTarget) -> Result<String, MLIRError> {
        let codegen = GpuCodeGenerator::new(&self.context, target);
        Ok(codegen.generate_kernel_code(module)?)
    }

    /// Optimize tensor operations for GPU
    pub fn optimize_gpu_tensors(&self, module: &mut MLIRModule, target: GpuTarget) -> Result<(), MLIRError> {
        let optimizer = GpuTensorOptimizer::new(&self.context, target);
        Ok(optimizer.optimize_tensor_operations(module)?)
    }

    /// Parse operation string into MLIR operation
    pub fn parse_operation_string(&self, op_str: &str) -> Result<MLIROperation, MLIRError> {
        // Simple parser for operation strings
        let mut op = MLIROperation::new(self.extract_op_name(op_str));
        
        // Add basic attributes based on the operation string
        if op_str.contains("kernel") {
            op.add_attribute("kernel".to_string(), MLIRAttribute::Boolean(true));
        }
        if op_str.contains("elementwise") {
            op.add_attribute("operation_type".to_string(), MLIRAttribute::String("elementwise".to_string()));
        }
        if op_str.contains("matmul") {
            op.add_attribute("operation_type".to_string(), MLIRAttribute::String("matmul".to_string()));
        }
        if op_str.contains("reduction") {
            op.add_attribute("operation_type".to_string(), MLIRAttribute::String("reduction".to_string()));
        }
        
        Ok(op)
    }

    /// Extract operation name from string
    fn extract_op_name(&self, op_str: &str) -> String {
        if let Some(space_pos) = op_str.find(' ') {
            op_str[..space_pos].to_string()
        } else {
            op_str.to_string()
        }
    }

    /// Get MLIR context
    pub fn context(&self) -> &MLIRContext {
        &self.context
    }
}

/// MLIR compilation errors (legacy compatibility)
#[derive(Debug, Clone)]
pub enum MLIRError {
    /// AST conversion error
    ConversionError(String),
    /// MLIR operation error
    OperationError(String),
    /// Lowering error
    LoweringError(String),
    /// Optimization error
    OptimizationError(String),
    /// Context creation error
    ContextCreationError(String),
    /// Dialect registration error
    DialectError(String),
    /// Module creation/manipulation error
    ModuleError(String),
    /// Type error
    TypeError(String),
    /// Verification error
    VerificationError(String),
    /// Comprehensive error (new system)
    Comprehensive(MLIRCompilationError),
}

impl From<mlir_context::MLIRError> for MLIRError {
    fn from(err: mlir_context::MLIRError) -> Self {
        match err {
            mlir_context::MLIRError::ContextCreationError(msg) => MLIRError::ContextCreationError(msg),
            mlir_context::MLIRError::DialectError(msg) => MLIRError::DialectError(msg),
            mlir_context::MLIRError::ModuleError(msg) => MLIRError::ModuleError(msg),
            mlir_context::MLIRError::OperationError(msg) => MLIRError::OperationError(msg),
            mlir_context::MLIRError::TypeError(msg) => MLIRError::TypeError(msg),
            mlir_context::MLIRError::VerificationError(msg) => MLIRError::VerificationError(msg),
            mlir_context::MLIRError::LoweringError(msg) => MLIRError::LoweringError(msg),
            mlir_context::MLIRError::OptimizationError(msg) => MLIRError::OptimizationError(msg),
        }
    }
}

impl std::fmt::Display for MLIRError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MLIRError::ConversionError(msg) => write!(f, "AST conversion error: {}", msg),
            MLIRError::OperationError(msg) => write!(f, "MLIR operation error: {}", msg),
            MLIRError::LoweringError(msg) => write!(f, "Lowering error: {}", msg),
            MLIRError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
            MLIRError::ContextCreationError(msg) => write!(f, "Context creation error: {}", msg),
            MLIRError::DialectError(msg) => write!(f, "Dialect error: {}", msg),
            MLIRError::ModuleError(msg) => write!(f, "Module error: {}", msg),
            MLIRError::TypeError(msg) => write!(f, "Type error: {}", msg),
            MLIRError::VerificationError(msg) => write!(f, "Verification error: {}", msg),
            MLIRError::Comprehensive(err) => write!(f, "{}", err),
        }
    }
}

impl std::error::Error for MLIRError {}

impl From<MLIRCompilationError> for MLIRError {
    fn from(err: MLIRCompilationError) -> Self {
        MLIRError::Comprehensive(err)
    }
}

impl From<std::io::Error> for MLIRError {
    fn from(err: std::io::Error) -> Self {
        MLIRError::OperationError(format!("IO error: {}", err))
    }
}

impl MLIRError {
    /// Convert to comprehensive error system
    pub fn to_comprehensive(self, location: SourceLocation) -> MLIRCompilationError {
        match self {
            MLIRError::ConversionError(msg) => MLIRCompilationError::ASTConversion {
                node_id: None,
                node_type: "Unknown".to_string(),
                error: msg,
                location,
                context: vec![],
            },
            MLIRError::OperationError(msg) => MLIRCompilationError::OperationCreation {
                operation: "Unknown".to_string(),
                error: msg,
                location,
                operand_types: vec![],
                expected_signature: None,
            },
            MLIRError::LoweringError(msg) => MLIRCompilationError::LoweringFailure {
                from_dialect: "Unknown".to_string(),
                to_dialect: "Unknown".to_string(),
                operation: "Unknown".to_string(),
                error: msg,
                location,
                conversion_patterns: vec![],
            },
            MLIRError::OptimizationError(msg) => MLIRCompilationError::OptimizationFailure {
                pass_name: "Unknown".to_string(),
                error: msg,
                location,
                pass_config: std::collections::HashMap::new(),
                can_skip: true,
            },
            MLIRError::ContextCreationError(msg) => MLIRCompilationError::ContextCreation {
                reason: msg,
                location,
                recovery_suggestion: None,
            },
            MLIRError::DialectError(msg) => MLIRCompilationError::DialectRegistration {
                dialect: "Unknown".to_string(),
                reason: msg,
                location,
                available_dialects: vec![],
            },
            MLIRError::ModuleError(msg) => MLIRCompilationError::ModuleVerification {
                errors: vec![],
                location,
                module_name: msg,
            },
            MLIRError::TypeError(msg) => MLIRCompilationError::TypeConversion {
                aether_type: "Unknown".to_string(),
                target_type: "Unknown".to_string(),
                error: msg,
                location,
                type_constraints: vec![],
            },
            MLIRError::VerificationError(msg) => MLIRCompilationError::ModuleVerification {
                errors: vec![VerificationError {
                    error_type: VerificationErrorType::TypeMismatch,
                    message: msg,
                    location: location.clone(),
                    operation: None,
                    fix_suggestion: None,
                }],
                location,
                module_name: "Unknown".to_string(),
            },
            MLIRError::Comprehensive(err) => err,
        }
    }
}