// Aether Programming Language
// A next-generation language designed for AI-native development

pub mod compiler;
pub mod runtime;
pub mod stdlib;
pub mod testing;
pub mod build_system;
pub mod cli;

// Re-export core types for convenience
pub use compiler::token::{Token, TokenType, TokenSequence};
pub use compiler::symbol_table::{SymbolTable, Symbol, SymbolType, SymbolId};
pub use compiler::ast::{AST, ASTNode, ASTNodeRef, AtomValue, NodeId};
pub use compiler::parser::{Parser, ParseError, parse_sexpr, parse_multiple_sexprs};
pub use compiler::sweet_syntax::{SweetSyntaxTranspiler, SweetSyntaxError, sweet_to_sexpr, sexpr_to_sweet_string};
pub use compiler::types::{Type, PrimitiveType, Shape, LinearOwnership, Distribution, TypeVar, TypeConstraint};
pub use compiler::type_inference::{TypeInference, InferenceError, TypeEnvironment, ConstraintSolver};
pub use compiler::type_checker::{TypeChecker, TypeCheckError, TypeCheckContext};
pub use compiler::autodiff::{AutoDiffEngine, DifferentiableFunction, DiffMarker, DiffMode, DiffDirection, DynamicTape, StaticGraph};
pub use compiler::probabilistic::{
    ProbabilisticEngine, ProbabilisticModel, RandomVariable, ObservedValue, 
    InferenceAlgorithm, InferenceConfig, InferenceResult, MCMCSampler, VIOptimizer,
    UncertaintyPropagation, UncertaintyInfo, UncertaintyOperation, OperationType,
    ConfidenceInterval, IntervalMethod, ConvergenceDiagnostics, ProbabilisticConstraint,
    MassMatrix, parse_random_variable_declaration, parse_observation
};

// Re-export runtime types for tensor operations
pub use runtime::tensor::{Tensor, TensorData, TensorDType, TensorDevice, TensorLayout, TensorResult, TensorError};
pub use runtime::tensor_ops::{TensorOpsRegistry, TensorOperation, OperationParams, OperationParam};
pub use runtime::memory_layout::{MemoryLayoutOptimizer, AccessPattern, AccessType, LayoutStrategy};

// Re-export native compilation types
pub use compiler::codegen::{NativeCodegen, TargetTriple, TargetArch, TargetOS, LLVMModule, CodegenError};
pub use compiler::native_pipeline::{
    NativeCompilationPipeline, OptimizationLevel, NativeCompilationConfig, 
    NativeCompilationBuilder, NativeCompilationError
};

// Re-export WebAssembly compilation types
pub use compiler::wasm_codegen::{
    WasmCodegen, WasmTarget, WasmModule, WasmCodegenError, JsBindings,
    WasmExport, WasmImport, WasmFunction, WasmValueType
};
pub use compiler::wasm_pipeline::{
    WasmCompilationPipeline, WasmCompilationConfig, WasmOptimizationLevel, WasmCompilationError
};

// Re-export MLIR types
pub use compiler::mlir::{MLIRModule, MLIRContext, MLIRPipeline, MLIRError, MLIROperation, MLIRValue, MLIRType, MLIRAttribute};

// Re-export native runtime types
pub use runtime::native_runtime::{
    AetherRuntime, MemoryManager, MemoryStats, RuntimeError, get_runtime
};

// Re-export build system types
pub use build_system::{
    BuildSystemManager, BuildConfig, ToolchainConfig, AetherConfig, TargetConfig, OptimizationConfig,
    Platform, Architecture, ExecutableFormat, Dependency,
    EnvironmentValidator, EnvironmentStatus, ToolchainStatus, DependencyStatus, SystemStatus,
    ErrorHandler, BuildError, ErrorType, ErrorSeverity, FixStrategy,
    AutoFixEngine, FixError, CompilerBinary, Executable, TestResults, BuildSystemError,
    BuildWorkflowOrchestrator, WorkflowConfig, WorkflowResult, WorkflowStage
};

// Re-export CLI types
pub use cli::{WorkflowCli, WorkflowCommand, run_workflow_cli};