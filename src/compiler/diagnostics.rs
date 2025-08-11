// Aether Diagnostics System
// Comprehensive error reporting with position information and suggested fixes

use std::collections::HashMap;
use std::fmt;

/// Source position information
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourcePosition {
    pub file: String,
    pub line: usize,
    pub column: usize,
    pub offset: usize,
}

impl SourcePosition {
    pub fn new(file: String, line: usize, column: usize, offset: usize) -> Self {
        SourcePosition { file, line, column, offset }
    }

    pub fn unknown() -> Self {
        SourcePosition {
            file: "<unknown>".to_string(),
            line: 0,
            column: 0,
            offset: 0,
        }
    }
}

impl fmt::Display for SourcePosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// Source span covering a range of positions
#[derive(Debug, Clone, PartialEq)]
pub struct SourceSpan {
    pub start: SourcePosition,
    pub end: SourcePosition,
}

impl SourceSpan {
    pub fn new(start: SourcePosition, end: SourcePosition) -> Self {
        SourceSpan { start, end }
    }

    pub fn single(pos: SourcePosition) -> Self {
        SourceSpan {
            end: pos.clone(),
            start: pos,
        }
    }
}

/// Diagnostic severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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

/// Suggested fix for a diagnostic
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticFix {
    pub description: String,
    pub span: SourceSpan,
    pub replacement: String,
}

impl DiagnosticFix {
    pub fn new(description: String, span: SourceSpan, replacement: String) -> Self {
        DiagnosticFix { description, span, replacement }
    }
}

/// Diagnostic message with position and suggested fixes
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    pub code: String,
    pub message: String,
    pub span: SourceSpan,
    pub fixes: Vec<DiagnosticFix>,
    pub related: Vec<RelatedDiagnostic>,
}

/// Related diagnostic information
#[derive(Debug, Clone, PartialEq)]
pub struct RelatedDiagnostic {
    pub message: String,
    pub span: SourceSpan,
}

impl Diagnostic {
    pub fn error(code: String, message: String, span: SourceSpan) -> Self {
        Diagnostic {
            severity: DiagnosticSeverity::Error,
            code,
            message,
            span,
            fixes: Vec::new(),
            related: Vec::new(),
        }
    }

    pub fn warning(code: String, message: String, span: SourceSpan) -> Self {
        Diagnostic {
            severity: DiagnosticSeverity::Warning,
            code,
            message,
            span,
            fixes: Vec::new(),
            related: Vec::new(),
        }
    }

    pub fn info(code: String, message: String, span: SourceSpan) -> Self {
        Diagnostic {
            severity: DiagnosticSeverity::Info,
            code,
            message,
            span,
            fixes: Vec::new(),
            related: Vec::new(),
        }
    }

    pub fn with_fix(mut self, fix: DiagnosticFix) -> Self {
        self.fixes.push(fix);
        self
    }

    pub fn with_related(mut self, related: RelatedDiagnostic) -> Self {
        self.related.push(related);
        self
    }
}

/// Diagnostic collector and formatter
pub struct DiagnosticEngine {
    diagnostics: Vec<Diagnostic>,
    source_files: HashMap<String, String>,
    error_count: usize,
    warning_count: usize,
}

impl DiagnosticEngine {
    pub fn new() -> Self {
        DiagnosticEngine {
            diagnostics: Vec::new(),
            source_files: HashMap::new(),
            error_count: 0,
            warning_count: 0,
        }
    }

    /// Add source file content for better error reporting
    pub fn add_source_file(&mut self, filename: String, content: String) {
        self.source_files.insert(filename, content);
    }

    /// Emit a diagnostic
    pub fn emit(&mut self, diagnostic: Diagnostic) {
        match diagnostic.severity {
            DiagnosticSeverity::Error => self.error_count += 1,
            DiagnosticSeverity::Warning => self.warning_count += 1,
            _ => {}
        }
        self.diagnostics.push(diagnostic);
    }

    /// Get all diagnostics
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Clear all diagnostics
    pub fn clear(&mut self) {
        self.diagnostics.clear();
        self.error_count = 0;
        self.warning_count = 0;
    }

    /// Format diagnostics for display
    pub fn format_diagnostics(&self) -> String {
        let mut output = String::new();
        
        for diagnostic in &self.diagnostics {
            output.push_str(&self.format_diagnostic(diagnostic));
            output.push('\n');
        }

        // Summary
        if self.error_count > 0 || self.warning_count > 0 {
            output.push_str(&format!(
                "\nCompilation finished with {} error(s) and {} warning(s)\n",
                self.error_count, self.warning_count
            ));
        }

        output
    }

    /// Format a single diagnostic
    fn format_diagnostic(&self, diagnostic: &Diagnostic) -> String {
        let mut output = String::new();

        // Main diagnostic line
        output.push_str(&format!(
            "{}: {}: {} [{}]\n",
            diagnostic.severity,
            diagnostic.span.start,
            diagnostic.message,
            diagnostic.code
        ));

        // Source context
        if let Some(context) = self.get_source_context(&diagnostic.span) {
            output.push_str(&context);
        }

        // Suggested fixes
        for fix in &diagnostic.fixes {
            output.push_str(&format!("  help: {}\n", fix.description));
        }

        // Related diagnostics
        for related in &diagnostic.related {
            output.push_str(&format!(
                "  note: {} at {}\n",
                related.message,
                related.span.start
            ));
        }

        output
    }

    /// Get source context for a span
    fn get_source_context(&self, span: &SourceSpan) -> Option<String> {
        let source = self.source_files.get(&span.start.file)?;
        let lines: Vec<&str> = source.lines().collect();
        
        if span.start.line == 0 || span.start.line > lines.len() {
            return None;
        }

        let mut context = String::new();
        let line_idx = span.start.line - 1;
        let line = lines[line_idx];

        // Show the line with error
        context.push_str(&format!("   {} | {}\n", span.start.line, line));

        // Show caret pointing to error location
        let mut caret_line = String::new();
        caret_line.push_str(&format!("   {} | ", " ".repeat(span.start.line.to_string().len())));
        
        // Add spaces up to error column
        for _ in 0..span.start.column.saturating_sub(1) {
            caret_line.push(' ');
        }
        
        // Add caret(s)
        let span_length = if span.start.line == span.end.line {
            span.end.column.saturating_sub(span.start.column).max(1)
        } else {
            1
        };
        
        for _ in 0..span_length {
            caret_line.push('^');
        }
        
        context.push_str(&caret_line);
        context.push('\n');

        Some(context)
    }
}

/// Common diagnostic codes
pub mod diagnostic_codes {
    pub const PARSE_ERROR: &str = "E0001";
    pub const TYPE_MISMATCH: &str = "E0002";
    pub const UNDEFINED_SYMBOL: &str = "E0003";
    pub const SHAPE_MISMATCH: &str = "E0004";
    pub const LINEAR_TYPE_VIOLATION: &str = "E0005";
    pub const ARITY_MISMATCH: &str = "E0006";
    pub const INVALID_TYPE_ANNOTATION: &str = "E0007";
    pub const MEMORY_LEAK: &str = "W0001";
    pub const UNUSED_VARIABLE: &str = "W0002";
    pub const PERFORMANCE_WARNING: &str = "W0003";
    pub const DEPRECATED_FEATURE: &str = "W0004";
    pub const STYLE_SUGGESTION: &str = "I0001";
    pub const OPTIMIZATION_HINT: &str = "I0002";
}

/// Helper functions for creating common diagnostics
pub mod diagnostic_helpers {
    use super::*;
    use crate::compiler::types::Type;

    pub fn type_mismatch_error(
        expected: &Type,
        actual: &Type,
        span: SourceSpan,
    ) -> Diagnostic {
        let message = format!(
            "type mismatch: expected `{}`, found `{}`",
            expected, actual
        );
        
        let mut diagnostic = Diagnostic::error(
            diagnostic_codes::TYPE_MISMATCH.to_string(),
            message,
            span.clone(),
        );

        // Add type annotation suggestion if helpful
        if matches!(actual, Type::Dynamic) {
            let fix = DiagnosticFix::new(
                format!("add type annotation: `{}`", expected),
                span,
                format!(": {}", expected),
            );
            diagnostic = diagnostic.with_fix(fix);
        }

        diagnostic
    }

    pub fn undefined_symbol_error(symbol: &str, span: SourceSpan) -> Diagnostic {
        let message = format!("undefined symbol `{}`", symbol);
        
        Diagnostic::error(
            diagnostic_codes::UNDEFINED_SYMBOL.to_string(),
            message,
            span,
        )
    }

    pub fn shape_mismatch_error(
        expected_shape: &[usize],
        actual_shape: &[usize],
        operation: &str,
        span: SourceSpan,
    ) -> Diagnostic {
        let message = format!(
            "tensor shape mismatch in {}: expected {:?}, found {:?}",
            operation, expected_shape, actual_shape
        );
        
        Diagnostic::error(
            diagnostic_codes::SHAPE_MISMATCH.to_string(),
            message,
            span,
        )
    }

    pub fn linear_type_violation_error(
        variable: &str,
        violation: &str,
        span: SourceSpan,
    ) -> Diagnostic {
        let message = format!("linear type violation for `{}`: {}", variable, violation);
        
        Diagnostic::error(
            diagnostic_codes::LINEAR_TYPE_VIOLATION.to_string(),
            message,
            span,
        )
    }

    pub fn unused_variable_warning(variable: &str, span: SourceSpan) -> Diagnostic {
        let message = format!("unused variable `{}`", variable);
        
        let fix = DiagnosticFix::new(
            format!("prefix with underscore to silence warning"),
            span.clone(),
            format!("_{}", variable),
        );
        
        Diagnostic::warning(
            diagnostic_codes::UNUSED_VARIABLE.to_string(),
            message,
            span,
        ).with_fix(fix)
    }

    pub fn performance_warning(
        issue: &str,
        suggestion: &str,
        span: SourceSpan,
    ) -> Diagnostic {
        let message = format!("performance issue: {}", issue);
        
        let fix = DiagnosticFix::new(
            suggestion.to_string(),
            span.clone(),
            "".to_string(), // Would need specific replacement text
        );
        
        Diagnostic::warning(
            diagnostic_codes::PERFORMANCE_WARNING.to_string(),
            message,
            span,
        ).with_fix(fix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_creation() {
        let pos = SourcePosition::new("test.ae".to_string(), 1, 5, 4);
        let span = SourceSpan::single(pos);
        
        let diagnostic = Diagnostic::error(
            "E0001".to_string(),
            "test error".to_string(),
            span,
        );
        
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Error);
        assert_eq!(diagnostic.code, "E0001");
        assert_eq!(diagnostic.message, "test error");
    }

    #[test]
    fn test_diagnostic_engine() {
        let mut engine = DiagnosticEngine::new();
        
        let pos = SourcePosition::new("test.ae".to_string(), 1, 5, 4);
        let span = SourceSpan::single(pos);
        
        let diagnostic = Diagnostic::error(
            "E0001".to_string(),
            "test error".to_string(),
            span,
        );
        
        engine.emit(diagnostic);
        
        assert_eq!(engine.error_count(), 1);
        assert_eq!(engine.warning_count(), 0);
        assert!(engine.has_errors());
    }

    #[test]
    fn test_source_context() {
        let mut engine = DiagnosticEngine::new();
        engine.add_source_file(
            "test.ae".to_string(),
            "let x = 42\nlet y = x + z\nprint(y)".to_string(),
        );
        
        let pos = SourcePosition::new("test.ae".to_string(), 2, 13, 22);
        let span = SourceSpan::single(pos);
        
        let diagnostic = Diagnostic::error(
            "E0003".to_string(),
            "undefined symbol `z`".to_string(),
            span,
        );
        
        engine.emit(diagnostic);
        
        let formatted = engine.format_diagnostics();
        assert!(formatted.contains("let y = x + z"));
        assert!(formatted.contains("^"));
    }
}