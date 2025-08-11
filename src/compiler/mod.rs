// Aether Compiler Module
// Handles parsing, type checking, and code generation

pub mod token;
pub mod symbol_table;
pub mod ast;
pub mod parser;
pub mod sweet_syntax;
pub mod types;
pub mod type_inference;
pub mod type_checker;
pub mod autodiff;
pub mod probabilistic;
pub mod mlir;
pub mod codegen;
pub mod native_pipeline;
pub mod compilation_pipeline;
pub mod wasm_codegen;
pub mod wasm_pipeline;
pub mod ffi;
pub mod ml_integration;
pub mod concurrency;
pub mod diagnostics;
pub mod error_handling;
pub mod comprehensive_errors;
#[cfg(test)]
pub mod comprehensive_errors_integration_test;
pub mod validation;
pub mod mlir_integration;

// Token and lexical analysis
pub use token::{Token, TokenType, TokenSequence};

// Symbol table management
pub use symbol_table::{SymbolTable, Symbol, SymbolType, SymbolId};

// AST types - explicit imports to avoid conflicts
pub use ast::{
    ASTNode, AtomValue, ASTNodeRef, NodeId, AST,
    GraphEdge as AstGraphEdge,  // Renamed to avoid conflict with autodiff::GraphEdge
};

// Parser components
pub use parser::{Parser, ParseError, ParseResult, parse_sexpr, parse_multiple_sexprs};

// Sweet syntax transpiler
pub use sweet_syntax::{SweetSyntaxTranspiler};

// Type system
pub use types::{
    Type, TypeId, TypeVar, TypeConstraint, PrimitiveType, Shape, ShapeConstraint,
    LinearOwnership as TypeLinearOwnership,  // Explicit to avoid confusion
    Distribution, Lifetime, LinearConstraint, AllocationSite, AllocationType
};

// Type inference - explicit imports to avoid conflicts
pub use type_inference::{
    TypeInference, InferenceError, 
    InferenceResult as TypeInferenceResult,  // Renamed to avoid conflict with probabilistic::InferenceResult
    TypeEnvironment, ConstraintSolver
};

// Type checker
pub use type_checker::{TypeChecker, TypeCheckError, TypeCheckResult};

// Automatic differentiation - explicit imports to avoid conflicts
pub use autodiff::{
    AutoDiffEngine, DiffMode, DiffDirection, DiffMarker, GradientFunction,
    GraphEdge as AutodiffGraphEdge,  // Renamed to avoid conflict with ast::GraphEdge
    GraphNode
};

// Probabilistic programming - explicit imports to avoid conflicts
pub use probabilistic::{
    ProbabilisticEngine, RandomVariable, ObservedValue, ProbabilisticConstraint,
    ProbabilisticModel, DependencyGraph, InferenceConfig, MCMCSampler,
    InferenceResult as ProbabilisticInferenceResult,  // Renamed to avoid conflict with type_inference::InferenceResult
    VIOptimizer, MassMatrix
};

// MLIR integration
pub use mlir::{
    MLIRContext, MLIRModule, MLIROperation, MLIRValue, MLIRType, MLIRAttribute,
    MLIRError
};

// Code generation
pub use codegen::{TargetTriple};

// WebAssembly code generation
pub use wasm_codegen::{WasmCodegen, WasmModule, WasmFunction};

// Foreign Function Interface
pub use ffi::{FFIType, FFIFunction, ModelFormat, TensorSpec, LoadedModel};

// ML framework integration - explicit imports to avoid conflicts
pub use ml_integration::{
    MLIntegrationManager, MLFramework, ModelInfo, TensorData, AetherTensor,
    MLError,
    MemoryLayout as MLMemoryLayout,  // Renamed to avoid conflict with gpu_tensor_ops::MemoryLayout
};

// Concurrency system - explicit imports to avoid conflicts
pub use concurrency::{
    ConcurrencySystem, ActorRegistry, ExecutionState, TaskScheduler, ActorId,
    Actor, ActorType, ActorState, MessageHandler, TypedMessage, MessageEvent,
    ParallelTask, PipelineStage, GpuKernel, GpuKernelType, GpuMemoryRequirements,
    KernelLaunchConfig as ConcurrencyKernelLaunchConfig,  // Renamed to avoid conflict with gpu_dialect::KernelLaunchConfig
    StateSnapshot, ConcurrencyError
};

// Diagnostics
pub use diagnostics::{
    Diagnostic, DiagnosticSeverity, SourcePosition, SourceSpan, DiagnosticFix,
    RelatedDiagnostic, DiagnosticEngine, diagnostic_codes, diagnostic_helpers
};

// Error handling
pub use error_handling::{
    AetherError, StackFrame, RecoveryStrategy, ErrorRecovery, PerformanceAnalyzer,
    PerformanceWarning, PerformanceCategory, PerformanceSeverity, PerformanceThresholds,
    AetherErrorHandler
};

// Comprehensive error types and handling
pub use comprehensive_errors::{
    CompilerError, SourceLocation, DiagnosticCollector, CompilerDiagnostic,
    DiagnosticSummary
};

// Validation framework
pub use validation::{
    AetherValidator, ValidationResult, ValidationContext, ValidationReport,
    ValidationMetrics, SyntaxValidator, SemanticValidator, TypeValidator,
    ComprehensiveValidator, FunctionSignature, Parameter
};

// Compilation pipeline
pub use compilation_pipeline::{
    CompilationPipeline, CompilationConfig, CompilationResult, CompilationError,
    CompilationStatistics, IntermediateFiles
};

// MLIR integration
pub use mlir_integration::{
    IntegratedCompilationPipeline, CompilationResult as MLIRCompilationResult,
    IntegrationError, IntegratedPipelineBuilder, OptimizationLevel as MLIROptimizationLevel
};