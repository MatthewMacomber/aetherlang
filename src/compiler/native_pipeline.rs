// Native compilation pipeline for Aether
// Integrates parsing, MLIR lowering, and native code generation

use crate::compiler::{
    ast::AST,
    mlir::{MLIRPipeline, MLIRError},
    codegen::{NativeCodegen, TargetTriple, CodegenError},
};
use std::path::Path;

/// Native compilation pipeline
pub struct NativeCompilationPipeline {
    mlir_pipeline: MLIRPipeline,
    target: TargetTriple,
}

impl NativeCompilationPipeline {
    /// Create new native compilation pipeline
    pub fn new(target: TargetTriple) -> Result<Self, crate::compiler::mlir::MLIRError> {
        Ok(NativeCompilationPipeline {
            mlir_pipeline: MLIRPipeline::new()?,
            target,
        })
    }

    /// Create pipeline for current platform
    pub fn for_current_platform() -> Result<Self, crate::compiler::mlir::MLIRError> {
        Self::new(TargetTriple::current())
    }

    /// Compile AST to native executable
    pub fn compile_to_native(&mut self, ast: &AST, output_path: &Path) -> Result<(), NativeCompilationError> {
        // Step 1: Convert AST to MLIR
        let mut mlir_module = self.mlir_pipeline.compile_ast(ast)?;
        
        // Step 2: Lower to standard MLIR dialects
        self.mlir_pipeline.lower_to_standard(&mut mlir_module)?;
        
        // Step 3: Generate LLVM IR
        let codegen = NativeCodegen::new(self.target.clone());
        let llvm_module = codegen.generate_llvm_ir(&mlir_module)?;
        
        // Step 4: Compile to native executable
        codegen.compile_to_native(&llvm_module, output_path)?;
        
        Ok(())
    }

    /// Compile with optimization level
    pub fn compile_with_optimization(&mut self, ast: &AST, output_path: &Path, opt_level: OptimizationLevel) -> Result<(), NativeCompilationError> {
        // Apply optimization-specific transformations
        let optimized_ast = self.apply_ast_optimizations(ast, opt_level)?;
        
        // Compile optimized AST
        self.compile_to_native(&optimized_ast, output_path)
    }

    /// Apply AST-level optimizations
    fn apply_ast_optimizations(&self, ast: &AST, opt_level: OptimizationLevel) -> Result<AST, NativeCompilationError> {
        let mut optimized_ast = ast.clone();
        
        match opt_level {
            OptimizationLevel::Debug => {
                // No optimizations, preserve debug info
            }
            OptimizationLevel::Release => {
                // Apply standard optimizations
                self.apply_constant_folding(&mut optimized_ast)?;
                self.apply_dead_code_elimination(&mut optimized_ast)?;
            }
            OptimizationLevel::Aggressive => {
                // Apply all optimizations
                self.apply_constant_folding(&mut optimized_ast)?;
                self.apply_dead_code_elimination(&mut optimized_ast)?;
                self.apply_function_inlining(&mut optimized_ast)?;
                self.apply_loop_optimizations(&mut optimized_ast)?;
            }
        }
        
        Ok(optimized_ast)
    }

    /// Apply constant folding optimization
    fn apply_constant_folding(&self, _ast: &mut AST) -> Result<(), NativeCompilationError> {
        // Mock implementation - would fold constant expressions
        Ok(())
    }

    /// Apply dead code elimination
    fn apply_dead_code_elimination(&self, _ast: &mut AST) -> Result<(), NativeCompilationError> {
        // Mock implementation - would remove unused code
        Ok(())
    }

    /// Apply function inlining
    fn apply_function_inlining(&self, _ast: &mut AST) -> Result<(), NativeCompilationError> {
        // Mock implementation - would inline small functions
        Ok(())
    }

    /// Apply loop optimizations
    fn apply_loop_optimizations(&self, _ast: &mut AST) -> Result<(), NativeCompilationError> {
        // Mock implementation - would optimize loops
        Ok(())
    }

    /// Get target information
    pub fn target(&self) -> &TargetTriple {
        &self.target
    }

    /// Check if target is supported
    pub fn is_target_supported(&self) -> bool {
        match (&self.target.arch, &self.target.os) {
            (crate::compiler::codegen::TargetArch::X86_64, _) => true,
            (crate::compiler::codegen::TargetArch::ARM64, _) => true,
            (crate::compiler::codegen::TargetArch::ARM32, crate::compiler::codegen::TargetOS::Linux) => true,
            _ => false,
        }
    }

    /// Link object file to executable
    pub fn link_object_to_executable(&self, object_path: &Path, output_path: &Path) -> Result<(), NativeCompilationError> {
        let codegen = NativeCodegen::new(self.target.clone());
        codegen.link_object_to_executable(object_path, output_path)?;
        Ok(())
    }
    
    /// Link object file to executable with AST for code generation
    pub fn link_object_to_executable_with_ast(&self, object_path: &Path, output_path: &Path, ast: &AST) -> Result<(), NativeCompilationError> {
        let codegen = NativeCodegen::new(self.target.clone());
        codegen.link_object_to_executable_with_ast(object_path, output_path, Some(ast))?;
        Ok(())
    }
}

/// Optimization levels for native compilation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationLevel {
    /// Debug build with no optimizations
    Debug,
    /// Release build with standard optimizations
    Release,
    /// Aggressive optimization for maximum performance
    Aggressive,
}

impl OptimizationLevel {
    /// Get LLVM optimization level
    pub fn to_llvm_opt_level(&self) -> &'static str {
        match self {
            OptimizationLevel::Debug => "O0",
            OptimizationLevel::Release => "O2",
            OptimizationLevel::Aggressive => "O3",
        }
    }
}

/// Native compilation configuration
#[derive(Debug, Clone)]
pub struct NativeCompilationConfig {
    pub target: TargetTriple,
    pub optimization_level: OptimizationLevel,
    pub debug_info: bool,
    pub link_time_optimization: bool,
    pub profile_guided_optimization: bool,
}

impl Default for NativeCompilationConfig {
    fn default() -> Self {
        NativeCompilationConfig {
            target: TargetTriple::current(),
            optimization_level: OptimizationLevel::Release,
            debug_info: false,
            link_time_optimization: false,
            profile_guided_optimization: false,
        }
    }
}

impl NativeCompilationConfig {
    /// Create debug configuration
    pub fn debug() -> Self {
        NativeCompilationConfig {
            optimization_level: OptimizationLevel::Debug,
            debug_info: true,
            ..Default::default()
        }
    }

    /// Create release configuration
    pub fn release() -> Self {
        NativeCompilationConfig {
            optimization_level: OptimizationLevel::Release,
            link_time_optimization: true,
            ..Default::default()
        }
    }

    /// Create aggressive optimization configuration
    pub fn aggressive() -> Self {
        NativeCompilationConfig {
            optimization_level: OptimizationLevel::Aggressive,
            link_time_optimization: true,
            profile_guided_optimization: true,
            ..Default::default()
        }
    }
}

/// Native compilation errors
#[derive(Debug)]
pub enum NativeCompilationError {
    /// MLIR compilation error
    MLIRError(MLIRError),
    /// Code generation error
    CodegenError(CodegenError),
    /// Optimization error
    OptimizationError(String),
    /// Target not supported
    UnsupportedTarget(String),
    /// Configuration error
    ConfigurationError(String),
}

impl std::fmt::Display for NativeCompilationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NativeCompilationError::MLIRError(e) => write!(f, "MLIR error: {}", e),
            NativeCompilationError::CodegenError(e) => write!(f, "Codegen error: {}", e),
            NativeCompilationError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
            NativeCompilationError::UnsupportedTarget(msg) => write!(f, "Unsupported target: {}", msg),
            NativeCompilationError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for NativeCompilationError {}

impl From<MLIRError> for NativeCompilationError {
    fn from(error: MLIRError) -> Self {
        NativeCompilationError::MLIRError(error)
    }
}

impl From<CodegenError> for NativeCompilationError {
    fn from(error: CodegenError) -> Self {
        NativeCompilationError::CodegenError(error)
    }
}

/// Builder for native compilation pipeline
pub struct NativeCompilationBuilder {
    config: NativeCompilationConfig,
}

impl NativeCompilationBuilder {
    /// Create new builder
    pub fn new() -> Self {
        NativeCompilationBuilder {
            config: NativeCompilationConfig::default(),
        }
    }

    /// Set target triple
    pub fn target(mut self, target: TargetTriple) -> Self {
        self.config.target = target;
        self
    }

    /// Set optimization level
    pub fn optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.config.optimization_level = level;
        self
    }

    /// Enable debug info
    pub fn debug_info(mut self, enable: bool) -> Self {
        self.config.debug_info = enable;
        self
    }

    /// Enable link-time optimization
    pub fn link_time_optimization(mut self, enable: bool) -> Self {
        self.config.link_time_optimization = enable;
        self
    }

    /// Enable profile-guided optimization
    pub fn profile_guided_optimization(mut self, enable: bool) -> Self {
        self.config.profile_guided_optimization = enable;
        self
    }

    /// Build the compilation pipeline
    pub fn build(self) -> Result<NativeCompilationPipeline, crate::compiler::mlir::MLIRError> {
        NativeCompilationPipeline::new(self.config.target)
    }
}