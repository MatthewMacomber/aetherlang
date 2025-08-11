// Comprehensive error handling and diagnostics for MLIR compilation pipeline
// Provides detailed error reporting with source location preservation and recovery strategies

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use crate::compiler::ast::{NodeId, ASTNode};
use crate::compiler::token::Token;

/// Source location information for error reporting
#[derive(Debug, Clone, PartialEq)]
pub struct SourceLocation {
    /// Source file path
    pub file: Option<PathBuf>,
    /// Line number (1-based)
    pub line: u32,
    /// Column number (1-based)
    pub column: u32,
    /// Character offset from start of file
    pub offset: u32,
    /// Length of the source span
    pub length: u32,
}

impl SourceLocation {
    /// Create new source location
    pub fn new(file: Option<PathBuf>, line: u32, column: u32, offset: u32, length: u32) -> Self {
        SourceLocation {
            file,
            line,
            column,
            offset,
            length,
        }
    }

    /// Create unknown location
    pub fn unknown() -> Self {
        SourceLocation {
            file: None,
            line: 0,
            column: 0,
            offset: 0,
            length: 0,
        }
    }

    /// Check if location is known
    pub fn is_known(&self) -> bool {
        self.line > 0 && self.column > 0
    }

    /// Get display string for location
    pub fn display(&self) -> String {
        if let Some(file) = &self.file {
            if self.is_known() {
                format!("{}:{}:{}", file.display(), self.line, self.column)
            } else {
                format!("{}", file.display())
            }
        } else if self.is_known() {
            format!("{}:{}", self.line, self.column)
        } else {
            "<unknown>".to_string()
        }
    }
}

/// Compilation error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Informational message
    Info,
    /// Warning that doesn't prevent compilation
    Warning,
    /// Error that prevents successful compilation
    Error,
    /// Critical error that stops compilation immediately
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Info => write!(f, "info"),
            ErrorSeverity::Warning => write!(f, "warning"),
            ErrorSeverity::Error => write!(f, "error"),
            ErrorSeverity::Critical => write!(f, "critical"),
        }
    }
}

/// Comprehensive MLIR compilation error types
#[derive(Debug, Clone)]
pub enum MLIRCompilationError {
    /// Context creation or initialization error
    ContextCreation {
        reason: String,
        location: SourceLocation,
        recovery_suggestion: Option<String>,
    },
    
    /// Dialect registration error
    DialectRegistration {
        dialect: String,
        reason: String,
        location: SourceLocation,
        available_dialects: Vec<String>,
    },
    
    /// AST to MLIR conversion error
    ASTConversion {
        node_id: Option<NodeId>,
        node_type: String,
        error: String,
        location: SourceLocation,
        context: Vec<String>,
    },
    
    /// Type conversion error
    TypeConversion {
        aether_type: String,
        target_type: String,
        error: String,
        location: SourceLocation,
        type_constraints: Vec<String>,
    },
    
    /// MLIR operation creation error
    OperationCreation {
        operation: String,
        error: String,
        location: SourceLocation,
        operand_types: Vec<String>,
        expected_signature: Option<String>,
    },
    
    /// Module verification error
    ModuleVerification {
        errors: Vec<VerificationError>,
        location: SourceLocation,
        module_name: String,
    },
    
    /// Optimization pass failure
    OptimizationFailure {
        pass_name: String,
        error: String,
        location: SourceLocation,
        pass_config: HashMap<String, String>,
        can_skip: bool,
    },
    
    /// Dialect lowering failure
    LoweringFailure {
        from_dialect: String,
        to_dialect: String,
        operation: String,
        error: String,
        location: SourceLocation,
        conversion_patterns: Vec<String>,
    },
    
    /// LLVM IR generation error
    LLVMGeneration {
        error: String,
        location: SourceLocation,
        llvm_context: Option<String>,
        module_state: String,
    },
    
    /// Code generation error
    CodeGeneration {
        target: String,
        error: String,
        location: SourceLocation,
        target_features: Vec<String>,
        optimization_level: String,
    },
    
    /// Resource management error
    ResourceError {
        resource_type: String,
        error: String,
        location: SourceLocation,
        cleanup_actions: Vec<String>,
    },
    
    /// Pipeline configuration error
    PipelineConfiguration {
        stage: String,
        error: String,
        location: SourceLocation,
        valid_configurations: Vec<String>,
    },
    
    /// Minimal MLIR error (for integration with minimal MLIR implementation)
    MinimalMLIR {
        error_type: String,
        message: String,
        location: SourceLocation,
        recovery_actions: Vec<String>,
    },
}

/// Detailed verification error information
#[derive(Debug, Clone)]
pub struct VerificationError {
    pub error_type: VerificationErrorType,
    pub message: String,
    pub location: SourceLocation,
    pub operation: Option<String>,
    pub fix_suggestion: Option<String>,
}

/// Types of verification errors
#[derive(Debug, Clone)]
pub enum VerificationErrorType {
    /// Type mismatch error
    TypeMismatch,
    /// Invalid operation signature
    InvalidSignature,
    /// Undefined symbol reference
    UndefinedSymbol,
    /// Invalid control flow
    InvalidControlFlow,
    /// Memory safety violation
    MemorySafety,
    /// Linear type violation
    LinearTypeViolation,
    /// Tensor shape mismatch
    ShapeMismatch,
}

impl MLIRCompilationError {
    /// Get error severity
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            MLIRCompilationError::ContextCreation { .. } => ErrorSeverity::Critical,
            MLIRCompilationError::DialectRegistration { .. } => ErrorSeverity::Critical,
            MLIRCompilationError::ASTConversion { .. } => ErrorSeverity::Error,
            MLIRCompilationError::TypeConversion { .. } => ErrorSeverity::Error,
            MLIRCompilationError::OperationCreation { .. } => ErrorSeverity::Error,
            MLIRCompilationError::ModuleVerification { .. } => ErrorSeverity::Error,
            MLIRCompilationError::OptimizationFailure { can_skip, .. } => {
                if *can_skip { ErrorSeverity::Warning } else { ErrorSeverity::Error }
            },
            MLIRCompilationError::LoweringFailure { .. } => ErrorSeverity::Error,
            MLIRCompilationError::LLVMGeneration { .. } => ErrorSeverity::Error,
            MLIRCompilationError::CodeGeneration { .. } => ErrorSeverity::Error,
            MLIRCompilationError::ResourceError { .. } => ErrorSeverity::Warning,
            MLIRCompilationError::PipelineConfiguration { .. } => ErrorSeverity::Error,
            MLIRCompilationError::MinimalMLIR { .. } => ErrorSeverity::Error,
        }
    }

    /// Get source location
    pub fn location(&self) -> &SourceLocation {
        match self {
            MLIRCompilationError::ContextCreation { location, .. } => location,
            MLIRCompilationError::DialectRegistration { location, .. } => location,
            MLIRCompilationError::ASTConversion { location, .. } => location,
            MLIRCompilationError::TypeConversion { location, .. } => location,
            MLIRCompilationError::OperationCreation { location, .. } => location,
            MLIRCompilationError::ModuleVerification { location, .. } => location,
            MLIRCompilationError::OptimizationFailure { location, .. } => location,
            MLIRCompilationError::LoweringFailure { location, .. } => location,
            MLIRCompilationError::LLVMGeneration { location, .. } => location,
            MLIRCompilationError::CodeGeneration { location, .. } => location,
            MLIRCompilationError::ResourceError { location, .. } => location,
            MLIRCompilationError::PipelineConfiguration { location, .. } => location,
            MLIRCompilationError::MinimalMLIR { location, .. } => location,
        }
    }

    /// Get primary error message
    pub fn message(&self) -> String {
        match self {
            MLIRCompilationError::ContextCreation { reason, .. } => {
                format!("Failed to create MLIR context: {}", reason)
            },
            MLIRCompilationError::DialectRegistration { dialect, reason, .. } => {
                format!("Failed to register dialect '{}': {}", dialect, reason)
            },
            MLIRCompilationError::ASTConversion { node_type, error, .. } => {
                format!("Failed to convert {} AST node: {}", node_type, error)
            },
            MLIRCompilationError::TypeConversion { aether_type, target_type, error, .. } => {
                format!("Failed to convert type '{}' to '{}': {}", aether_type, target_type, error)
            },
            MLIRCompilationError::OperationCreation { operation, error, .. } => {
                format!("Failed to create operation '{}': {}", operation, error)
            },
            MLIRCompilationError::ModuleVerification { errors, module_name, .. } => {
                format!("Module '{}' verification failed with {} errors", module_name, errors.len())
            },
            MLIRCompilationError::OptimizationFailure { pass_name, error, .. } => {
                format!("Optimization pass '{}' failed: {}", pass_name, error)
            },
            MLIRCompilationError::LoweringFailure { from_dialect, to_dialect, operation, error, .. } => {
                format!("Failed to lower operation '{}' from {} to {}: {}", operation, from_dialect, to_dialect, error)
            },
            MLIRCompilationError::LLVMGeneration { error, .. } => {
                format!("LLVM IR generation failed: {}", error)
            },
            MLIRCompilationError::CodeGeneration { target, error, .. } => {
                format!("Code generation for target '{}' failed: {}", target, error)
            },
            MLIRCompilationError::ResourceError { resource_type, error, .. } => {
                format!("Resource management error for '{}': {}", resource_type, error)
            },
            MLIRCompilationError::PipelineConfiguration { stage, error, .. } => {
                format!("Pipeline configuration error at stage '{}': {}", stage, error)
            },
            MLIRCompilationError::MinimalMLIR { error_type, message, .. } => {
                format!("Minimal MLIR {} error: {}", error_type, message)
            },
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            MLIRCompilationError::ContextCreation { .. } => false,
            MLIRCompilationError::DialectRegistration { .. } => false,
            MLIRCompilationError::ASTConversion { .. } => true,
            MLIRCompilationError::TypeConversion { .. } => true,
            MLIRCompilationError::OperationCreation { .. } => true,
            MLIRCompilationError::ModuleVerification { .. } => true,
            MLIRCompilationError::OptimizationFailure { can_skip, .. } => *can_skip,
            MLIRCompilationError::LoweringFailure { .. } => true,
            MLIRCompilationError::LLVMGeneration { .. } => false,
            MLIRCompilationError::CodeGeneration { .. } => false,
            MLIRCompilationError::ResourceError { .. } => true,
            MLIRCompilationError::PipelineConfiguration { .. } => true,
            MLIRCompilationError::MinimalMLIR { .. } => true,
        }
    }

    /// Get recovery suggestions
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            MLIRCompilationError::ContextCreation { recovery_suggestion, .. } => {
                recovery_suggestion.as_ref().map(|s| vec![s.clone()]).unwrap_or_default()
            },
            MLIRCompilationError::DialectRegistration { available_dialects, .. } => {
                if !available_dialects.is_empty() {
                    vec![format!("Available dialects: {}", available_dialects.join(", "))]
                } else {
                    vec!["Check MLIR installation and dialect availability".to_string()]
                }
            },
            MLIRCompilationError::ASTConversion { context, .. } => {
                let mut suggestions = vec!["Check AST node structure and types".to_string()];
                if !context.is_empty() {
                    suggestions.push(format!("Context: {}", context.join(" -> ")));
                }
                suggestions
            },
            MLIRCompilationError::TypeConversion { type_constraints, .. } => {
                let mut suggestions = vec!["Verify type compatibility".to_string()];
                if !type_constraints.is_empty() {
                    suggestions.push(format!("Type constraints: {}", type_constraints.join(", ")));
                }
                suggestions
            },
            MLIRCompilationError::OperationCreation { expected_signature, .. } => {
                let mut suggestions = vec!["Check operation signature and operand types".to_string()];
                if let Some(signature) = expected_signature {
                    suggestions.push(format!("Expected signature: {}", signature));
                }
                suggestions
            },
            MLIRCompilationError::ModuleVerification { errors, .. } => {
                let mut suggestions = vec!["Fix verification errors:".to_string()];
                for error in errors.iter().take(3) {
                    if let Some(fix) = &error.fix_suggestion {
                        suggestions.push(format!("  - {}", fix));
                    }
                }
                if errors.len() > 3 {
                    suggestions.push(format!("  ... and {} more errors", errors.len() - 3));
                }
                suggestions
            },
            MLIRCompilationError::OptimizationFailure { can_skip, pass_config, .. } => {
                let mut suggestions = vec![];
                if *can_skip {
                    suggestions.push("Consider skipping this optimization pass".to_string());
                }
                if !pass_config.is_empty() {
                    suggestions.push("Check pass configuration parameters".to_string());
                }
                suggestions
            },
            MLIRCompilationError::LoweringFailure { conversion_patterns, .. } => {
                let mut suggestions = vec!["Check lowering patterns and target dialect support".to_string()];
                if !conversion_patterns.is_empty() {
                    suggestions.push(format!("Available patterns: {}", conversion_patterns.join(", ")));
                }
                suggestions
            },
            MLIRCompilationError::LLVMGeneration { llvm_context, module_state, .. } => {
                let mut suggestions = vec!["Check LLVM context and module state".to_string()];
                if let Some(context) = llvm_context {
                    suggestions.push(format!("LLVM context: {}", context));
                }
                suggestions.push(format!("Module state: {}", module_state));
                suggestions
            },
            MLIRCompilationError::CodeGeneration { target_features, optimization_level, .. } => {
                let mut suggestions = vec!["Check target configuration and features".to_string()];
                if !target_features.is_empty() {
                    suggestions.push(format!("Target features: {}", target_features.join(", ")));
                }
                suggestions.push(format!("Optimization level: {}", optimization_level));
                suggestions
            },
            MLIRCompilationError::ResourceError { cleanup_actions, .. } => {
                let mut suggestions = vec!["Check resource management and cleanup".to_string()];
                if !cleanup_actions.is_empty() {
                    suggestions.push(format!("Cleanup actions: {}", cleanup_actions.join(", ")));
                }
                suggestions
            },
            MLIRCompilationError::PipelineConfiguration { valid_configurations, .. } => {
                let mut suggestions = vec!["Check pipeline configuration".to_string()];
                if !valid_configurations.is_empty() {
                    suggestions.push(format!("Valid configurations: {}", valid_configurations.join(", ")));
                }
                suggestions
            },
            MLIRCompilationError::MinimalMLIR { recovery_actions, .. } => {
                if recovery_actions.is_empty() {
                    vec!["Check minimal MLIR implementation and operation validity".to_string()]
                } else {
                    recovery_actions.clone()
                }
            },
        }
    }
}

impl fmt::Display for MLIRCompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} at {}", 
               self.severity(), 
               self.message(), 
               self.location().display())
    }
}

impl std::error::Error for MLIRCompilationError {}

/// Error recovery strategy
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Skip the failing operation and continue
    Skip,
    /// Use a fallback implementation
    Fallback(String),
    /// Retry with different parameters
    Retry(HashMap<String, String>),
    /// Stop compilation immediately
    Abort,
}

/// Error recovery context
#[derive(Debug, Clone)]
pub struct RecoveryContext {
    pub strategy: RecoveryStrategy,
    pub attempted_recoveries: u32,
    pub max_recoveries: u32,
    pub fallback_options: Vec<String>,
}

impl RecoveryContext {
    /// Create new recovery context
    pub fn new(strategy: RecoveryStrategy, max_recoveries: u32) -> Self {
        RecoveryContext {
            strategy,
            attempted_recoveries: 0,
            max_recoveries,
            fallback_options: Vec::new(),
        }
    }

    /// Check if recovery should be attempted
    pub fn should_attempt_recovery(&self) -> bool {
        self.attempted_recoveries < self.max_recoveries
    }

    /// Record recovery attempt
    pub fn record_attempt(&mut self) {
        self.attempted_recoveries += 1;
    }

    /// Add fallback option
    pub fn add_fallback(&mut self, option: String) {
        self.fallback_options.push(option);
    }
    
    /// Attempt recovery from an MLIR error
    pub fn attempt_recovery(&self, error: &MLIRError) -> Result<(), MLIRError> {
        if !self.should_attempt_recovery() {
            return Err(error.clone());
        }
        
        match self.strategy {
            RecoveryStrategy::Skip => {
                // Skip the problematic operation
                Ok(())
            }
            RecoveryStrategy::Fallback => {
                // Try fallback options
                if !self.fallback_options.is_empty() {
                    Ok(())
                } else {
                    Err(error.clone())
                }
            }
            RecoveryStrategy::Retry => {
                // For now, just return the error - retry would need more context
                Err(error.clone())
            }
        }
    }
}

/// Source location tracker for preserving location information through compilation
#[derive(Debug, Clone)]
pub struct SourceLocationTracker {
    /// Map from AST node ID to source location
    node_locations: HashMap<NodeId, SourceLocation>,
    /// Map from token to source location
    token_locations: HashMap<Token, SourceLocation>,
    /// Current source file
    current_file: Option<PathBuf>,
}

impl SourceLocationTracker {
    /// Create new location tracker
    pub fn new() -> Self {
        SourceLocationTracker {
            node_locations: HashMap::new(),
            token_locations: HashMap::new(),
            current_file: None,
        }
    }

    /// Set current source file
    pub fn set_current_file(&mut self, file: PathBuf) {
        self.current_file = Some(file);
    }

    /// Track location for AST node
    pub fn track_node(&mut self, node_id: NodeId, location: SourceLocation) {
        self.node_locations.insert(node_id, location);
    }

    /// Track location for token
    pub fn track_token(&mut self, token: Token, location: SourceLocation) {
        self.token_locations.insert(token, location);
    }

    /// Get location for AST node
    pub fn get_node_location(&self, node_id: NodeId) -> Option<&SourceLocation> {
        self.node_locations.get(&node_id)
    }

    /// Get location for token
    pub fn get_token_location(&self, token: &Token) -> Option<&SourceLocation> {
        self.token_locations.get(token)
    }

    /// Get current file
    pub fn current_file(&self) -> Option<&PathBuf> {
        self.current_file.as_ref()
    }

    /// Create location with current file
    pub fn create_location(&self, line: u32, column: u32, offset: u32, length: u32) -> SourceLocation {
        SourceLocation::new(self.current_file.clone(), line, column, offset, length)
    }
}

impl Default for SourceLocationTracker {
    fn default() -> Self {
        Self::new()
    }
}