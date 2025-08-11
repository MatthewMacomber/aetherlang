// Comprehensive Error Types and Handling for Aether Compiler
// Implements the error handling system as specified in the compiler robustness improvements

use std::fmt;
use std::collections::HashMap;

/// Source location information for precise error reporting
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceLocation {
    /// Source file path
    pub file: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based)
    pub column: usize,
}

impl SourceLocation {
    /// Create a new source location
    pub fn new(file: String, line: usize, column: usize) -> Self {
        SourceLocation { file, line, column }
    }

    /// Create an unknown location
    pub fn unknown() -> Self {
        SourceLocation {
            file: "<unknown>".to_string(),
            line: 0,
            column: 0,
        }
    }

    /// Check if this location is valid/known
    pub fn is_valid(&self) -> bool {
        self.line > 0 && self.column > 0 && self.file != "<unknown>"
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "{}:{}:{}", self.file, self.line, self.column)
        } else {
            write!(f, "<unknown location>")
        }
    }
}

/// Comprehensive compiler error enum with specific variants for different error types
#[derive(Debug, Clone)]
pub enum CompilerError {
    /// Syntax error with location and message
    SyntaxError {
        location: SourceLocation,
        message: String,
        suggestions: Vec<String>,
    },
    
    /// Type mismatch error
    TypeMismatch {
        location: SourceLocation,
        expected: String,
        found: String,
        context: Option<String>,
    },
    
    /// Tensor shape mismatch error
    ShapeMismatch {
        location: SourceLocation,
        expected: Vec<usize>,
        found: Vec<usize>,
        operation: String,
    },
    
    /// Undefined symbol error
    UndefinedSymbol {
        location: SourceLocation,
        symbol: String,
        suggestions: Vec<String>,
    },
    
    /// Undefined variable error
    UndefinedVariable {
        name: String,
        location: SourceLocation,
        suggestions: Vec<String>,
    },
    
    /// Undefined function error
    UndefinedFunction {
        name: String,
        location: SourceLocation,
        suggestions: Vec<String>,
    },
    
    /// Tensor shape mismatch error (simplified version)
    TensorShapeMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
        location: SourceLocation,
    },
    
    /// I/O error (simplified version)
    IoError {
        message: String,
        location: SourceLocation,
    },
    
    /// MLIR operation error
    MLIRError {
        location: SourceLocation,
        operation: String,
        reason: String,
        recovery_hint: Option<String>,
    },
    
    /// MLIR context creation error
    MLIRContextError {
        location: SourceLocation,
        reason: String,
    },
    
    /// MLIR dialect registration error
    MLIRDialectError {
        location: SourceLocation,
        dialect: String,
        reason: String,
        available_dialects: Vec<String>,
    },
    
    /// File I/O error
    IOError {
        location: SourceLocation,
        file_path: String,
        operation: String,
        reason: String,
    },
    
    /// Memory safety violation
    MemorySafetyError {
        location: SourceLocation,
        violation_type: String,
        description: String,
    },
    
    /// Linear type violation
    LinearTypeViolation {
        location: SourceLocation,
        variable: String,
        violation: String,
    },
    
    /// Function arity mismatch
    ArityMismatch {
        location: SourceLocation,
        function: String,
        expected: usize,
        found: usize,
    },
    
    /// Internal compiler error
    InternalError {
        location: SourceLocation,
        message: String,
        debug_info: HashMap<String, String>,
    },
}

impl CompilerError {
    /// Get the source location of this error
    pub fn location(&self) -> &SourceLocation {
        match self {
            CompilerError::SyntaxError { location, .. } => location,
            CompilerError::TypeMismatch { location, .. } => location,
            CompilerError::ShapeMismatch { location, .. } => location,
            CompilerError::UndefinedSymbol { location, .. } => location,
            CompilerError::UndefinedVariable { location, .. } => location,
            CompilerError::UndefinedFunction { location, .. } => location,
            CompilerError::TensorShapeMismatch { location, .. } => location,
            CompilerError::IoError { location, .. } => location,
            CompilerError::MLIRError { location, .. } => location,
            CompilerError::MLIRContextError { location, .. } => location,
            CompilerError::MLIRDialectError { location, .. } => location,
            CompilerError::IOError { location, .. } => location,
            CompilerError::MemorySafetyError { location, .. } => location,
            CompilerError::LinearTypeViolation { location, .. } => location,
            CompilerError::ArityMismatch { location, .. } => location,
            CompilerError::InternalError { location, .. } => location,
        }
    }

    /// Get the primary error message
    pub fn message(&self) -> String {
        match self {
            CompilerError::SyntaxError { message, .. } => message.clone(),
            CompilerError::TypeMismatch { expected, found, context, .. } => {
                let base_msg = format!("type mismatch: expected `{}`, found `{}`", expected, found);
                if let Some(ctx) = context {
                    format!("{} in {}", base_msg, ctx)
                } else {
                    base_msg
                }
            },
            CompilerError::ShapeMismatch { expected, found, operation, .. } => {
                format!("tensor shape mismatch in {}: expected {:?}, found {:?}", operation, expected, found)
            },
            CompilerError::UndefinedSymbol { symbol, .. } => {
                format!("undefined symbol `{}`", symbol)
            },
            CompilerError::UndefinedVariable { name, .. } => {
                format!("undefined variable `{}`", name)
            },
            CompilerError::UndefinedFunction { name, .. } => {
                format!("undefined function `{}`", name)
            },
            CompilerError::TensorShapeMismatch { expected, found, .. } => {
                format!("tensor shape mismatch: expected {:?}, found {:?}", expected, found)
            },
            CompilerError::IoError { message, .. } => {
                format!("I/O error: {}", message)
            },
            CompilerError::MLIRError { operation, reason, .. } => {
                format!("MLIR operation `{}` failed: {}", operation, reason)
            },
            CompilerError::MLIRContextError { reason, .. } => {
                format!("MLIR context error: {}", reason)
            },
            CompilerError::MLIRDialectError { dialect, reason, .. } => {
                format!("MLIR dialect `{}` error: {}", dialect, reason)
            },
            CompilerError::IOError { file_path, operation, reason, .. } => {
                format!("I/O error during {} on `{}`: {}", operation, file_path, reason)
            },
            CompilerError::MemorySafetyError { violation_type, description, .. } => {
                format!("memory safety violation ({}): {}", violation_type, description)
            },
            CompilerError::LinearTypeViolation { variable, violation, .. } => {
                format!("linear type violation for `{}`: {}", variable, violation)
            },
            CompilerError::ArityMismatch { function, expected, found, .. } => {
                format!("function `{}` expects {} arguments, found {}", function, expected, found)
            },
            CompilerError::InternalError { message, .. } => {
                format!("internal compiler error: {}", message)
            },
        }
    }

    /// Get suggestions for fixing this error
    pub fn suggestions(&self) -> Vec<String> {
        match self {
            CompilerError::SyntaxError { suggestions, .. } => suggestions.clone(),
            CompilerError::TypeMismatch { expected, .. } => {
                vec![format!("consider adding type annotation: `{}`", expected)]
            },
            CompilerError::ShapeMismatch { expected, operation, .. } => {
                vec![
                    format!("ensure tensor has shape {:?} before {} operation", expected, operation),
                    "consider using tensor reshaping operations".to_string(),
                ]
            },
            CompilerError::UndefinedSymbol { suggestions, .. } => suggestions.clone(),
            CompilerError::UndefinedVariable { suggestions, .. } => suggestions.clone(),
            CompilerError::UndefinedFunction { suggestions, .. } => suggestions.clone(),
            CompilerError::TensorShapeMismatch { expected, .. } => {
                vec![
                    format!("ensure tensor has shape {:?}", expected),
                    "consider using tensor reshaping operations".to_string(),
                ]
            },
            CompilerError::IoError { .. } => {
                vec!["check file permissions and disk space".to_string()]
            },
            CompilerError::MLIRError { recovery_hint, .. } => {
                recovery_hint.as_ref().map(|hint| vec![hint.clone()]).unwrap_or_default()
            },
            CompilerError::MLIRDialectError { available_dialects, .. } => {
                if !available_dialects.is_empty() {
                    vec![format!("available dialects: {}", available_dialects.join(", "))]
                } else {
                    vec!["check MLIR installation and dialect availability".to_string()]
                }
            },
            CompilerError::IOError { .. } => {
                vec!["check file permissions and disk space".to_string()]
            },
            CompilerError::MemorySafetyError { .. } => {
                vec!["review memory management and ownership".to_string()]
            },
            CompilerError::LinearTypeViolation { .. } => {
                vec!["ensure linear resources are used exactly once".to_string()]
            },
            CompilerError::ArityMismatch { expected, .. } => {
                vec![format!("provide exactly {} arguments", expected)]
            },
            CompilerError::InternalError { .. } => {
                vec!["this is a compiler bug, please report it".to_string()]
            },
            _ => vec![],
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            CompilerError::SyntaxError { .. } => true,
            CompilerError::TypeMismatch { .. } => true,
            CompilerError::ShapeMismatch { .. } => true,
            CompilerError::UndefinedSymbol { .. } => true,
            CompilerError::UndefinedVariable { .. } => true,
            CompilerError::UndefinedFunction { .. } => true,
            CompilerError::TensorShapeMismatch { .. } => true,
            CompilerError::IoError { .. } => true,
            CompilerError::MLIRError { .. } => true,
            CompilerError::MLIRContextError { .. } => false,
            CompilerError::MLIRDialectError { .. } => false,
            CompilerError::IOError { .. } => true,
            CompilerError::MemorySafetyError { .. } => false,
            CompilerError::LinearTypeViolation { .. } => false,
            CompilerError::ArityMismatch { .. } => true,
            CompilerError::InternalError { .. } => false,
        }
    }
}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error at {}: {}", self.location(), self.message())
    }
}

impl std::error::Error for CompilerError {}

/// Diagnostic severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

impl fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticSeverity::Error => write!(f, "error"),
            DiagnosticSeverity::Warning => write!(f, "warning"),
            DiagnosticSeverity::Info => write!(f, "info"),
            DiagnosticSeverity::Hint => write!(f, "hint"),
        }
    }
}

/// Comprehensive diagnostic information
#[derive(Debug, Clone)]
pub struct CompilerDiagnostic {
    pub severity: DiagnosticSeverity,
    pub location: SourceLocation,
    pub message: String,
    pub suggestions: Vec<String>,
    pub related_locations: Vec<SourceLocation>,
    pub error_code: Option<String>,
}

impl CompilerDiagnostic {
    /// Create a new diagnostic from a compiler error
    pub fn from_error(error: CompilerError) -> Self {
        CompilerDiagnostic {
            severity: DiagnosticSeverity::Error,
            location: error.location().clone(),
            message: error.message(),
            suggestions: error.suggestions(),
            related_locations: vec![],
            error_code: None,
        }
    }

    /// Create a warning diagnostic
    pub fn warning(location: SourceLocation, message: String) -> Self {
        CompilerDiagnostic {
            severity: DiagnosticSeverity::Warning,
            location,
            message,
            suggestions: vec![],
            related_locations: vec![],
            error_code: None,
        }
    }

    /// Create an info diagnostic
    pub fn info(location: SourceLocation, message: String) -> Self {
        CompilerDiagnostic {
            severity: DiagnosticSeverity::Info,
            location,
            message,
            suggestions: vec![],
            related_locations: vec![],
            error_code: None,
        }
    }

    /// Add a suggestion to this diagnostic
    pub fn with_suggestion(mut self, suggestion: String) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Add a related location to this diagnostic
    pub fn with_related_location(mut self, location: SourceLocation) -> Self {
        self.related_locations.push(location);
        self
    }

    /// Add an error code to this diagnostic
    pub fn with_error_code(mut self, code: String) -> Self {
        self.error_code = Some(code);
        self
    }
}

/// Diagnostic collector for gathering and reporting multiple errors with context
#[derive(Debug)]
pub struct DiagnosticCollector {
    diagnostics: Vec<CompilerDiagnostic>,
    source_files: HashMap<String, String>,
    max_errors: usize,
    error_count: usize,
    warning_count: usize,
}

impl DiagnosticCollector {
    /// Create a new diagnostic collector
    pub fn new() -> Self {
        DiagnosticCollector {
            diagnostics: Vec::new(),
            source_files: HashMap::new(),
            max_errors: 100, // Default maximum errors before stopping
            error_count: 0,
            warning_count: 0,
        }
    }

    /// Create a diagnostic collector with a custom maximum error count
    pub fn with_max_errors(max_errors: usize) -> Self {
        DiagnosticCollector {
            diagnostics: Vec::new(),
            source_files: HashMap::new(),
            max_errors,
            error_count: 0,
            warning_count: 0,
        }
    }

    /// Add source file content for better error reporting
    pub fn add_source_file(&mut self, file_path: String, content: String) {
        self.source_files.insert(file_path, content);
    }

    /// Report a compiler error
    pub fn report_error(&mut self, error: CompilerError) {
        if self.error_count >= self.max_errors {
            return; // Stop collecting after max errors
        }

        let diagnostic = CompilerDiagnostic::from_error(error);
        self.error_count += 1;
        self.diagnostics.push(diagnostic);
    }

    /// Report a warning
    pub fn report_warning(&mut self, location: SourceLocation, message: String) {
        let diagnostic = CompilerDiagnostic::warning(location, message);
        self.warning_count += 1;
        self.diagnostics.push(diagnostic);
    }

    /// Report an info message
    pub fn report_info(&mut self, location: SourceLocation, message: String) {
        let diagnostic = CompilerDiagnostic::info(location, message);
        self.diagnostics.push(diagnostic);
    }

    /// Add a diagnostic directly
    pub fn add_diagnostic(&mut self, diagnostic: CompilerDiagnostic) {
        match diagnostic.severity {
            DiagnosticSeverity::Error => {
                if self.error_count >= self.max_errors {
                    return;
                }
                self.error_count += 1;
            },
            DiagnosticSeverity::Warning => {
                self.warning_count += 1;
            },
            _ => {},
        }
        self.diagnostics.push(diagnostic);
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Get the number of errors
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Get the number of warnings
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Get all diagnostics
    pub fn diagnostics(&self) -> &[CompilerDiagnostic] {
        &self.diagnostics
    }

    /// Get diagnostics by severity
    pub fn diagnostics_by_severity(&self, severity: DiagnosticSeverity) -> Vec<&CompilerDiagnostic> {
        self.diagnostics.iter()
            .filter(|d| d.severity == severity)
            .collect()
    }

    /// Clear all diagnostics
    pub fn clear(&mut self) {
        self.diagnostics.clear();
        self.error_count = 0;
        self.warning_count = 0;
    }

    /// Format all diagnostics for display
    pub fn format_diagnostics(&self) -> String {
        let mut output = String::new();

        for diagnostic in &self.diagnostics {
            output.push_str(&self.format_diagnostic(diagnostic));
            output.push('\n');
        }

        // Add summary
        if self.error_count > 0 || self.warning_count > 0 {
            output.push_str(&format!(
                "\nCompilation finished with {} error(s) and {} warning(s)\n",
                self.error_count, self.warning_count
            ));
        }

        output
    }

    /// Format a single diagnostic with source context
    pub fn format_diagnostic(&self, diagnostic: &CompilerDiagnostic) -> String {
        let mut output = String::new();

        // Main diagnostic line
        let error_code_str = diagnostic.error_code.as_ref()
            .map(|code| format!(" [{}]", code))
            .unwrap_or_default();
        
        output.push_str(&format!(
            "{}: {}: {}{}\n",
            diagnostic.severity,
            diagnostic.location,
            diagnostic.message,
            error_code_str
        ));

        // Source context if available
        if let Some(context) = self.get_source_context(&diagnostic.location) {
            output.push_str(&context);
        }

        // Suggestions
        for suggestion in &diagnostic.suggestions {
            output.push_str(&format!("  help: {}\n", suggestion));
        }

        // Related locations
        for related_location in &diagnostic.related_locations {
            output.push_str(&format!("  note: related location at {}\n", related_location));
        }

        output
    }

    /// Get source context for a location
    fn get_source_context(&self, location: &SourceLocation) -> Option<String> {
        if !location.is_valid() {
            return None;
        }

        let source = self.source_files.get(&location.file)?;
        let lines: Vec<&str> = source.lines().collect();
        
        if location.line == 0 || location.line > lines.len() {
            return None;
        }

        let mut context = String::new();
        let line_idx = location.line - 1;
        let line = lines[line_idx];

        // Show the line with error
        context.push_str(&format!("   {} | {}\n", location.line, line));

        // Show caret pointing to error location
        let line_num_width = location.line.to_string().len();
        let mut caret_line = format!("   {} | ", " ".repeat(line_num_width));
        
        // Add spaces up to error column
        for _ in 0..location.column.saturating_sub(1) {
            caret_line.push(' ');
        }
        
        // Add caret
        caret_line.push('^');
        
        context.push_str(&caret_line);
        context.push('\n');

        Some(context)
    }

    /// Check if maximum errors reached
    pub fn max_errors_reached(&self) -> bool {
        self.error_count >= self.max_errors
    }

    /// Get a summary of diagnostics
    pub fn summary(&self) -> DiagnosticSummary {
        DiagnosticSummary {
            total_diagnostics: self.diagnostics.len(),
            error_count: self.error_count,
            warning_count: self.warning_count,
            info_count: self.diagnostics.iter().filter(|d| matches!(d.severity, DiagnosticSeverity::Info)).count(),
            hint_count: self.diagnostics.iter().filter(|d| matches!(d.severity, DiagnosticSeverity::Hint)).count(),
            max_errors_reached: self.max_errors_reached(),
        }
    }
}

impl Default for DiagnosticCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of diagnostic information
#[derive(Debug, Clone)]
pub struct DiagnosticSummary {
    pub total_diagnostics: usize,
    pub error_count: usize,
    pub warning_count: usize,
    pub info_count: usize,
    pub hint_count: usize,
    pub max_errors_reached: bool,
}

impl fmt::Display for DiagnosticSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} error(s), {} warning(s), {} info, {} hint(s)", 
               self.error_count, self.warning_count, self.info_count, self.hint_count)?;
        
        if self.max_errors_reached {
            write!(f, " (max errors reached)")?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location_creation() {
        let location = SourceLocation::new("test.ae".to_string(), 10, 5);
        assert_eq!(location.file, "test.ae");
        assert_eq!(location.line, 10);
        assert_eq!(location.column, 5);
        assert!(location.is_valid());
    }

    #[test]
    fn test_source_location_unknown() {
        let location = SourceLocation::unknown();
        assert!(!location.is_valid());
        assert_eq!(location.to_string(), "<unknown location>");
    }

    #[test]
    fn test_compiler_error_creation() {
        let location = SourceLocation::new("test.ae".to_string(), 5, 10);
        let error = CompilerError::SyntaxError {
            location: location.clone(),
            message: "unexpected token".to_string(),
            suggestions: vec!["try adding a semicolon".to_string()],
        };

        assert_eq!(error.location(), &location);
        assert_eq!(error.message(), "unexpected token");
        assert_eq!(error.suggestions(), vec!["try adding a semicolon"]);
        assert!(error.is_recoverable());
    }

    #[test]
    fn test_type_mismatch_error() {
        let location = SourceLocation::new("test.ae".to_string(), 3, 7);
        let error = CompilerError::TypeMismatch {
            location,
            expected: "i32".to_string(),
            found: "f64".to_string(),
            context: Some("function call".to_string()),
        };

        let message = error.message();
        assert!(message.contains("type mismatch"));
        assert!(message.contains("i32"));
        assert!(message.contains("f64"));
        assert!(message.contains("function call"));
    }

    #[test]
    fn test_shape_mismatch_error() {
        let location = SourceLocation::new("test.ae".to_string(), 8, 15);
        let error = CompilerError::ShapeMismatch {
            location,
            expected: vec![2, 3],
            found: vec![3, 2],
            operation: "matrix multiplication".to_string(),
        };

        let message = error.message();
        assert!(message.contains("tensor shape mismatch"));
        assert!(message.contains("matrix multiplication"));
        assert!(message.contains("[2, 3]"));
        assert!(message.contains("[3, 2]"));
    }

    #[test]
    fn test_diagnostic_collector() {
        let mut collector = DiagnosticCollector::new();
        
        let location = SourceLocation::new("test.ae".to_string(), 1, 1);
        let error = CompilerError::SyntaxError {
            location: location.clone(),
            message: "test error".to_string(),
            suggestions: vec![],
        };

        collector.report_error(error);
        collector.report_warning(location, "test warning".to_string());

        assert!(collector.has_errors());
        assert_eq!(collector.error_count(), 1);
        assert_eq!(collector.warning_count(), 1);
        assert_eq!(collector.diagnostics().len(), 2);
    }

    #[test]
    fn test_diagnostic_collector_max_errors() {
        let mut collector = DiagnosticCollector::with_max_errors(2);
        
        let location = SourceLocation::new("test.ae".to_string(), 1, 1);
        
        // Add 3 errors, but only 2 should be collected
        for i in 0..3 {
            let error = CompilerError::SyntaxError {
                location: location.clone(),
                message: format!("error {}", i),
                suggestions: vec![],
            };
            collector.report_error(error);
        }

        assert_eq!(collector.error_count(), 2);
        assert!(collector.max_errors_reached());
    }

    #[test]
    fn test_diagnostic_formatting() {
        let mut collector = DiagnosticCollector::new();
        collector.add_source_file("test.ae".to_string(), "let x = 42\nlet y = x + z\nprint(y)".to_string());
        
        let location = SourceLocation::new("test.ae".to_string(), 2, 13);
        let error = CompilerError::UndefinedSymbol {
            location,
            symbol: "z".to_string(),
            suggestions: vec!["did you mean `x`?".to_string()],
        };

        collector.report_error(error);
        
        let formatted = collector.format_diagnostics();
        assert!(formatted.contains("undefined symbol `z`"));
        assert!(formatted.contains("let y = x + z"));
        assert!(formatted.contains("^"));
        assert!(formatted.contains("did you mean `x`?"));
    }

    #[test]
    fn test_diagnostic_summary() {
        let mut collector = DiagnosticCollector::new();
        
        let location = SourceLocation::new("test.ae".to_string(), 1, 1);
        
        collector.report_error(CompilerError::SyntaxError {
            location: location.clone(),
            message: "error".to_string(),
            suggestions: vec![],
        });
        collector.report_warning(location.clone(), "warning".to_string());
        collector.report_info(location, "info".to_string());

        let summary = collector.summary();
        assert_eq!(summary.error_count, 1);
        assert_eq!(summary.warning_count, 1);
        assert_eq!(summary.info_count, 1);
        assert_eq!(summary.total_diagnostics, 3);
        assert!(!summary.max_errors_reached);
    }
}