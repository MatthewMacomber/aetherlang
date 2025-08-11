// Aether Error Handling System
// Comprehensive compile-time and runtime error management

use std::collections::HashMap;
use std::fmt;
use crate::compiler::diagnostics::{Diagnostic, DiagnosticEngine, SourceSpan, SourcePosition};
use crate::compiler::types::Type;
use crate::compiler::parser::ParseError;
use crate::compiler::type_checker::TypeCheckError;
use crate::runtime::native_runtime::RuntimeError;

/// Unified error type for all Aether compilation and runtime errors
#[derive(Debug, Clone)]
pub enum AetherError {
    /// Parse error with position information
    Parse {
        error: ParseError,
        span: SourceSpan,
    },
    /// Type checking error
    TypeCheck {
        error: TypeCheckError,
        span: SourceSpan,
    },
    /// Code generation error
    CodeGen {
        message: String,
        span: SourceSpan,
    },
    /// Runtime error with stack trace
    Runtime {
        error: RuntimeError,
        stack_trace: Vec<StackFrame>,
    },
    /// I/O error
    IO {
        message: String,
        file: Option<String>,
    },
    /// System error
    System {
        message: String,
        code: Option<i32>,
    },
}

/// Stack frame for error reporting
#[derive(Debug, Clone)]
pub struct StackFrame {
    pub function: String,
    pub file: String,
    pub line: usize,
    pub column: usize,
}

impl StackFrame {
    pub fn new(function: String, file: String, line: usize, column: usize) -> Self {
        StackFrame { function, file, line, column }
    }
}

impl fmt::Display for StackFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "  at {} ({}:{}:{})", self.function, self.file, self.line, self.column)
    }
}

impl fmt::Display for AetherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AetherError::Parse { error, span } => {
                write!(f, "Parse error at {}: {}", span.start, error)
            }
            AetherError::TypeCheck { error, span } => {
                write!(f, "Type error at {}: {}", span.start, error)
            }
            AetherError::CodeGen { message, span } => {
                write!(f, "Code generation error at {}: {}", span.start, message)
            }
            AetherError::Runtime { error, stack_trace } => {
                write!(f, "Runtime error: {}", error)?;
                if !stack_trace.is_empty() {
                    write!(f, "\nStack trace:")?;
                    for frame in stack_trace {
                        write!(f, "\n{}", frame)?;
                    }
                }
                Ok(())
            }
            AetherError::IO { message, file } => {
                if let Some(file) = file {
                    write!(f, "I/O error in {}: {}", file, message)
                } else {
                    write!(f, "I/O error: {}", message)
                }
            }
            AetherError::System { message, code } => {
                if let Some(code) = code {
                    write!(f, "System error ({}): {}", code, message)
                } else {
                    write!(f, "System error: {}", message)
                }
            }
        }
    }
}

impl std::error::Error for AetherError {}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Skip the current token and continue parsing
    SkipToken,
    /// Insert a missing token
    InsertToken(String),
    /// Replace current token with expected token
    ReplaceToken(String),
    /// Synchronize to next statement boundary
    SynchronizeStatement,
    /// Synchronize to next expression boundary
    SynchronizeExpression,
    /// Use default value for missing expression
    DefaultValue(Type),
}

/// Error recovery context
pub struct ErrorRecovery {
    strategies: HashMap<String, RecoveryStrategy>,
    recovery_points: Vec<RecoveryPoint>,
}

#[derive(Debug, Clone)]
struct RecoveryPoint {
    position: SourcePosition,
    context: String,
    expected_tokens: Vec<String>,
}

impl RecoveryPoint {
    /// Get context information for this recovery point
    pub fn get_context(&self) -> &str {
        &self.context
    }
    
    /// Get expected tokens at this recovery point
    pub fn get_expected_tokens(&self) -> &[String] {
        &self.expected_tokens
    }
}

impl ErrorRecovery {
    pub fn new() -> Self {
        let mut recovery = ErrorRecovery {
            strategies: HashMap::new(),
            recovery_points: Vec::new(),
        };
        recovery.init_default_strategies();
        recovery
    }

    fn init_default_strategies(&mut self) {
        // Common recovery strategies
        self.strategies.insert("missing_paren".to_string(), RecoveryStrategy::InsertToken(")".to_string()));
        self.strategies.insert("missing_semicolon".to_string(), RecoveryStrategy::InsertToken(";".to_string()));
        self.strategies.insert("unexpected_token".to_string(), RecoveryStrategy::SkipToken);
        self.strategies.insert("missing_expression".to_string(), RecoveryStrategy::DefaultValue(Type::dynamic()));
    }

    pub fn add_recovery_point(&mut self, position: SourcePosition, context: String, expected: Vec<String>) {
        self.recovery_points.push(RecoveryPoint {
            position,
            context,
            expected_tokens: expected,
        });
    }

    pub fn get_strategy(&self, error_type: &str) -> Option<&RecoveryStrategy> {
        self.strategies.get(error_type)
    }

    pub fn find_recovery_point(&self, position: &SourcePosition) -> Option<&RecoveryPoint> {
        self.recovery_points
            .iter()
            .rev()
            .find(|point| point.position.offset <= position.offset)
    }
}

/// Performance warning system
pub struct PerformanceAnalyzer {
    warnings: Vec<PerformanceWarning>,
    thresholds: PerformanceThresholds,
}

#[derive(Debug, Clone)]
pub struct PerformanceWarning {
    pub category: PerformanceCategory,
    pub message: String,
    pub span: SourceSpan,
    pub severity: PerformanceSeverity,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceCategory {
    Memory,
    Computation,
    GPU,
    Tensor,
    Concurrency,
    IO,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum PerformanceSeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub max_tensor_size: usize,
    pub max_memory_allocation: usize,
    pub max_loop_depth: usize,
    pub max_function_complexity: usize,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        PerformanceThresholds {
            max_tensor_size: 1_000_000,
            max_memory_allocation: 100_000_000, // 100MB
            max_loop_depth: 10,
            max_function_complexity: 50,
        }
    }
}

impl PerformanceAnalyzer {
    pub fn new() -> Self {
        PerformanceAnalyzer {
            warnings: Vec::new(),
            thresholds: PerformanceThresholds::default(),
        }
    }

    pub fn with_thresholds(thresholds: PerformanceThresholds) -> Self {
        PerformanceAnalyzer {
            warnings: Vec::new(),
            thresholds,
        }
    }

    pub fn analyze_tensor_operation(&mut self, shape: &[usize], operation: &str, span: SourceSpan) {
        let total_elements: usize = shape.iter().product();
        
        if total_elements > self.thresholds.max_tensor_size {
            self.warnings.push(PerformanceWarning {
                category: PerformanceCategory::Tensor,
                message: format!(
                    "Large tensor operation ({} elements) in `{}` may cause performance issues",
                    total_elements, operation
                ),
                span: span.clone(),
                severity: if total_elements > self.thresholds.max_tensor_size * 10 {
                    PerformanceSeverity::Critical
                } else {
                    PerformanceSeverity::High
                },
                suggestion: Some("Consider using tensor chunking or streaming operations".to_string()),
            });
        }

        // Check for inefficient tensor shapes
        if shape.len() > 4 {
            self.warnings.push(PerformanceWarning {
                category: PerformanceCategory::Tensor,
                message: format!(
                    "High-dimensional tensor ({} dimensions) may have poor cache locality",
                    shape.len()
                ),
                span,
                severity: PerformanceSeverity::Medium,
                suggestion: Some("Consider reshaping or using lower-dimensional operations".to_string()),
            });
        }
    }

    pub fn analyze_memory_allocation(&mut self, size: usize, allocation_type: &str, span: SourceSpan) {
        if size > self.thresholds.max_memory_allocation {
            self.warnings.push(PerformanceWarning {
                category: PerformanceCategory::Memory,
                message: format!(
                    "Large memory allocation ({} bytes) of type `{}`",
                    size, allocation_type
                ),
                span: span.clone(),
                severity: PerformanceSeverity::High,
                suggestion: Some("Consider using memory pools or streaming allocation".to_string()),
            });
        }

        // Check for potential memory fragmentation
        if allocation_type == "heap" && size < 64 {
            self.warnings.push(PerformanceWarning {
                category: PerformanceCategory::Memory,
                message: "Small heap allocation may cause fragmentation".to_string(),
                span,
                severity: PerformanceSeverity::Low,
                suggestion: Some("Consider using stack allocation or object pooling".to_string()),
            });
        }
    }

    pub fn analyze_loop_complexity(&mut self, depth: usize, span: SourceSpan) {
        if depth > self.thresholds.max_loop_depth {
            self.warnings.push(PerformanceWarning {
                category: PerformanceCategory::Computation,
                message: format!("Deep loop nesting ({} levels) may impact performance", depth),
                span,
                severity: PerformanceSeverity::Medium,
                suggestion: Some("Consider extracting inner loops into functions or using vectorization".to_string()),
            });
        }
    }

    pub fn analyze_gpu_usage(&mut self, operation: &str, data_size: usize, span: SourceSpan) {
        // Check if GPU operation is worth the overhead
        if data_size < 1000 {
            self.warnings.push(PerformanceWarning {
                category: PerformanceCategory::GPU,
                message: format!(
                    "GPU operation `{}` on small data ({} elements) may have high overhead",
                    operation, data_size
                ),
                span,
                severity: PerformanceSeverity::Medium,
                suggestion: Some("Consider using CPU computation for small datasets".to_string()),
            });
        }
    }

    pub fn get_warnings(&self) -> &[PerformanceWarning] {
        &self.warnings
    }

    pub fn clear_warnings(&mut self) {
        self.warnings.clear();
    }

    pub fn warnings_by_severity(&self, min_severity: PerformanceSeverity) -> Vec<&PerformanceWarning> {
        self.warnings
            .iter()
            .filter(|w| w.severity >= min_severity)
            .collect()
    }
}

/// Comprehensive error handler that integrates all error types
pub struct AetherErrorHandler {
    diagnostic_engine: DiagnosticEngine,
    error_recovery: ErrorRecovery,
    performance_analyzer: PerformanceAnalyzer,
    error_callbacks: Vec<Box<dyn Fn(&AetherError) + Send + Sync>>,
}

impl AetherErrorHandler {
    pub fn new() -> Self {
        AetherErrorHandler {
            diagnostic_engine: DiagnosticEngine::new(),
            error_recovery: ErrorRecovery::new(),
            performance_analyzer: PerformanceAnalyzer::new(),
            error_callbacks: Vec::new(),
        }
    }

    pub fn add_source_file(&mut self, filename: String, content: String) {
        self.diagnostic_engine.add_source_file(filename, content);
    }

    pub fn handle_parse_error(&mut self, error: ParseError, span: SourceSpan) -> AetherError {
        let aether_error = AetherError::Parse { error: error.clone(), span: span.clone() };
        
        // Convert to diagnostic
        let diagnostic = self.parse_error_to_diagnostic(&error, &span);
        self.diagnostic_engine.emit(diagnostic);
        
        // Call error callbacks
        for callback in &self.error_callbacks {
            callback(&aether_error);
        }
        
        aether_error
    }

    pub fn handle_type_error(&mut self, error: TypeCheckError, span: SourceSpan) -> AetherError {
        let aether_error = AetherError::TypeCheck { error: error.clone(), span: span.clone() };
        
        // Convert to diagnostic with suggested fixes
        let diagnostic = self.type_error_to_diagnostic(&error, &span);
        self.diagnostic_engine.emit(diagnostic);
        
        // Call error callbacks
        for callback in &self.error_callbacks {
            callback(&aether_error);
        }
        
        aether_error
    }

    pub fn handle_runtime_error(&mut self, error: RuntimeError, stack_trace: Vec<StackFrame>) -> AetherError {
        let aether_error = AetherError::Runtime { error, stack_trace };
        
        // Call error callbacks
        for callback in &self.error_callbacks {
            callback(&aether_error);
        }
        
        aether_error
    }

    pub fn register_error_callback<F>(&mut self, callback: F)
    where
        F: Fn(&AetherError) + Send + Sync + 'static,
    {
        self.error_callbacks.push(Box::new(callback));
    }

    pub fn diagnostic_engine(&self) -> &DiagnosticEngine {
        &self.diagnostic_engine
    }

    pub fn diagnostic_engine_mut(&mut self) -> &mut DiagnosticEngine {
        &mut self.diagnostic_engine
    }

    pub fn performance_analyzer(&mut self) -> &mut PerformanceAnalyzer {
        &mut self.performance_analyzer
    }
    
    /// Attempt error recovery using the error recovery system
    pub fn attempt_recovery(&mut self, position: &SourcePosition) -> Option<String> {
        if let Some(recovery_point) = self.error_recovery.find_recovery_point(position) {
            Some(recovery_point.get_context().to_string())
        } else {
            None
        }
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostic_engine.has_errors()
    }

    pub fn format_all_diagnostics(&self) -> String {
        let mut output = self.diagnostic_engine.format_diagnostics();
        
        // Add performance warnings
        let warnings = self.performance_analyzer.get_warnings();
        if !warnings.is_empty() {
            output.push_str("\nPerformance Warnings:\n");
            for warning in warnings {
                output.push_str(&format!(
                    "{}: {} at {}\n",
                    warning.severity_string(),
                    warning.message,
                    warning.span.start
                ));
                if let Some(suggestion) = &warning.suggestion {
                    output.push_str(&format!("  suggestion: {}\n", suggestion));
                }
            }
        }
        
        output
    }

    fn parse_error_to_diagnostic(&self, error: &ParseError, span: &SourceSpan) -> Diagnostic {
        use crate::compiler::diagnostics::diagnostic_codes;
        
        match error {
            ParseError::UnexpectedEof => {
                Diagnostic::error(
                    diagnostic_codes::PARSE_ERROR.to_string(),
                    "unexpected end of input".to_string(),
                    span.clone(),
                )
            }
            ParseError::UnexpectedChar(ch, _) => {
                Diagnostic::error(
                    diagnostic_codes::PARSE_ERROR.to_string(),
                    format!("unexpected character '{}'", ch),
                    span.clone(),
                )
            }
            ParseError::UnterminatedString(_) => {
                let fix = crate::compiler::diagnostics::DiagnosticFix::new(
                    "add closing quote".to_string(),
                    span.clone(),
                    "\"".to_string(),
                );
                Diagnostic::error(
                    diagnostic_codes::PARSE_ERROR.to_string(),
                    "unterminated string literal".to_string(),
                    span.clone(),
                ).with_fix(fix)
            }
            ParseError::InvalidNumber(num, _) => {
                Diagnostic::error(
                    diagnostic_codes::PARSE_ERROR.to_string(),
                    format!("invalid number format '{}'", num),
                    span.clone(),
                )
            }
            ParseError::UnmatchedParen(_) => {
                let fix = crate::compiler::diagnostics::DiagnosticFix::new(
                    "add closing parenthesis".to_string(),
                    span.clone(),
                    ")".to_string(),
                );
                Diagnostic::error(
                    diagnostic_codes::PARSE_ERROR.to_string(),
                    "unmatched parenthesis".to_string(),
                    span.clone(),
                ).with_fix(fix)
            }
            _ => {
                Diagnostic::error(
                    diagnostic_codes::PARSE_ERROR.to_string(),
                    error.to_string(),
                    span.clone(),
                )
            }
        }
    }

    fn type_error_to_diagnostic(&self, error: &TypeCheckError, span: &SourceSpan) -> Diagnostic {
        use crate::compiler::diagnostics::{diagnostic_codes, diagnostic_helpers};
        
        match error {
            TypeCheckError::TypeMismatch { expected, actual, .. } => {
                diagnostic_helpers::type_mismatch_error(expected, actual, span.clone())
            }
            TypeCheckError::UndefinedSymbol { name, .. } => {
                diagnostic_helpers::undefined_symbol_error(name, span.clone())
            }
            TypeCheckError::ShapeMismatch { expected: _, actual: _, operation, .. } => {
                // Convert Shape to Vec<usize> for display (simplified)
                let expected_dims = vec![]; // Would extract from Shape
                let actual_dims = vec![]; // Would extract from Shape
                diagnostic_helpers::shape_mismatch_error(&expected_dims, &actual_dims, operation, span.clone())
            }
            TypeCheckError::LinearTypeViolation { variable, violation, .. } => {
                diagnostic_helpers::linear_type_violation_error(variable, violation, span.clone())
            }
            TypeCheckError::ArityMismatch { expected, actual, function, .. } => {
                Diagnostic::error(
                    diagnostic_codes::ARITY_MISMATCH.to_string(),
                    format!("function `{}` expects {} arguments, found {}", function, expected, actual),
                    span.clone(),
                )
            }
            _ => {
                Diagnostic::error(
                    diagnostic_codes::TYPE_MISMATCH.to_string(),
                    error.to_string(),
                    span.clone(),
                )
            }
        }
    }
}

impl PerformanceWarning {
    fn severity_string(&self) -> &'static str {
        match self.severity {
            PerformanceSeverity::Critical => "critical",
            PerformanceSeverity::High => "high",
            PerformanceSeverity::Medium => "medium",
            PerformanceSeverity::Low => "low",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::parser::ParseError;

    #[test]
    fn test_error_handler_creation() {
        let handler = AetherErrorHandler::new();
        assert!(!handler.has_errors());
    }

    #[test]
    fn test_parse_error_handling() {
        let mut handler = AetherErrorHandler::new();
        let pos = SourcePosition::new("test.ae".to_string(), 1, 5, 4);
        let span = SourceSpan::single(pos);
        
        let parse_error = ParseError::UnexpectedChar('!', 4);
        let aether_error = handler.handle_parse_error(parse_error, span);
        
        assert!(handler.has_errors());
        match aether_error {
            AetherError::Parse { .. } => {},
            _ => panic!("Expected parse error"),
        }
    }

    #[test]
    fn test_performance_analyzer() {
        let mut analyzer = PerformanceAnalyzer::new();
        let pos = SourcePosition::new("test.ae".to_string(), 1, 5, 4);
        let span = SourceSpan::single(pos);
        
        // Test large tensor warning
        analyzer.analyze_tensor_operation(&[10000, 10000], "matmul", span);
        
        let warnings = analyzer.get_warnings();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].category, PerformanceCategory::Tensor);
    }

    #[test]
    fn test_error_recovery() {
        let recovery = ErrorRecovery::new();
        let strategy = recovery.get_strategy("missing_paren");
        
        assert!(strategy.is_some());
        match strategy.unwrap() {
            RecoveryStrategy::InsertToken(token) => assert_eq!(token, ")"),
            _ => panic!("Expected InsertToken strategy"),
        }
    }
}