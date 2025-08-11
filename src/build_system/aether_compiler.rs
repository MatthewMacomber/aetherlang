// Aether Compiler Interface
// Manages Aether source compilation, syntax validation, and executable generation

use crate::build_system::windows::{
    WindowsExecutableGenerator, WindowsExecutableConfig, WindowsExecutableTester,
    WindowsTestConfig, WindowsExecutableResult, WindowsTestResult
};
use crate::compiler::{
    parse_sexpr, parse_multiple_sexprs, AST, ASTNode, AtomValue,
    DiagnosticEngine, Diagnostic, SourcePosition, SourceSpan, DiagnosticSeverity,
    diagnostic_codes
};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Aether compiler interface for managing source compilation
pub struct AetherCompiler {
    /// Path to the Aether compiler binary
    binary_path: PathBuf,
    /// Compilation configuration
    config: AetherCompilationConfig,
    /// Diagnostic engine for error reporting
    diagnostics: DiagnosticEngine,
    /// Windows executable generator (when targeting Windows)
    windows_generator: Option<WindowsExecutableGenerator>,
    /// Windows executable tester (when targeting Windows)
    windows_tester: Option<WindowsExecutableTester>,
}

/// Configuration for Aether compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AetherCompilationConfig {
    /// Target platform for compilation
    pub target: CompilationTarget,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Enable debug information
    pub debug_info: bool,
    /// Additional compiler flags
    pub compiler_flags: Vec<String>,
    /// Output directory for generated files
    pub output_directory: PathBuf,
    /// Enable verbose compilation output
    pub verbose: bool,
}

/// Compilation target platforms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompilationTarget {
    /// Native Windows executable
    WindowsNative,
    /// WebAssembly module
    WebAssembly,
    /// GPU kernel (CUDA/OpenCL)
    GPU,
    /// Mobile bytecode
    Mobile,
}

/// Result of syntax validation
#[derive(Debug, Clone)]
pub struct SyntaxValidation {
    /// Whether syntax is valid
    pub is_valid: bool,
    /// Parsed AST (if valid)
    pub ast: Option<AST>,
    /// Syntax errors found
    pub errors: Vec<Diagnostic>,
    /// Warnings and suggestions
    pub warnings: Vec<Diagnostic>,
}

/// Result of Aether compilation
#[derive(Debug, Clone)]
pub struct AetherCompilationResult {
    /// Whether compilation succeeded
    pub success: bool,
    /// Path to generated executable
    pub executable_path: Option<PathBuf>,
    /// Compilation time
    pub compilation_time: Duration,
    /// Compiler stdout output
    pub stdout: String,
    /// Compiler stderr output
    pub stderr: String,
    /// Exit code from compiler
    pub exit_code: i32,
    /// Diagnostics from compilation
    pub diagnostics: Vec<Diagnostic>,
}

/// Diagnostic report for compilation errors
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// All diagnostics
    pub diagnostics: Vec<Diagnostic>,
    /// Error count
    pub error_count: usize,
    /// Warning count
    pub warning_count: usize,
    /// Formatted diagnostic output
    pub formatted_output: String,
}

/// Errors specific to Aether compilation
#[derive(Debug)]
pub enum AetherCompilerError {
    /// Compiler binary not found
    CompilerNotFound(PathBuf),
    /// Source file not found
    SourceFileNotFound(PathBuf),
    /// Syntax validation failed
    SyntaxError(Vec<Diagnostic>),
    /// Compilation failed
    CompilationFailed {
        exit_code: i32,
        stderr: String,
        diagnostics: Vec<Diagnostic>,
    },
    /// I/O error during compilation
    IoError(std::io::Error),
    /// Invalid configuration
    InvalidConfiguration(String),
    /// Executable generation failed
    ExecutableGenerationFailed(String),
}

impl std::fmt::Display for AetherCompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AetherCompilerError::CompilerNotFound(path) => {
                write!(f, "Aether compiler not found at: {}", path.display())
            }
            AetherCompilerError::SourceFileNotFound(path) => {
                write!(f, "Source file not found: {}", path.display())
            }
            AetherCompilerError::SyntaxError(diagnostics) => {
                write!(f, "Syntax errors found: {} error(s)", diagnostics.len())
            }
            AetherCompilerError::CompilationFailed { exit_code, stderr, .. } => {
                write!(f, "Compilation failed with exit code {}: {}", exit_code, stderr)
            }
            AetherCompilerError::IoError(err) => {
                write!(f, "I/O error during compilation: {}", err)
            }
            AetherCompilerError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            AetherCompilerError::ExecutableGenerationFailed(msg) => {
                write!(f, "Executable generation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for AetherCompilerError {}

impl Default for AetherCompilationConfig {
    fn default() -> Self {
        AetherCompilationConfig {
            target: CompilationTarget::WindowsNative,
            optimization_level: 2,
            debug_info: true,
            compiler_flags: Vec::new(),
            output_directory: PathBuf::from("target/aether"),
            verbose: false,
        }
    }
}

impl AetherCompiler {
    /// Create new Aether compiler interface
    pub fn new(binary_path: PathBuf) -> Result<Self, AetherCompilerError> {
        if !binary_path.exists() {
            return Err(AetherCompilerError::CompilerNotFound(binary_path));
        }

        let config = AetherCompilationConfig::default();
        let (windows_generator, windows_tester) = if config.target == CompilationTarget::WindowsNative {
            (Some(WindowsExecutableGenerator::new()), Some(WindowsExecutableTester::new()))
        } else {
            (None, None)
        };

        Ok(AetherCompiler {
            binary_path,
            config,
            diagnostics: DiagnosticEngine::new(),
            windows_generator,
            windows_tester,
        })
    }

    /// Create Aether compiler with custom configuration
    pub fn with_config(
        binary_path: PathBuf,
        config: AetherCompilationConfig,
    ) -> Result<Self, AetherCompilerError> {
        if !binary_path.exists() {
            return Err(AetherCompilerError::CompilerNotFound(binary_path));
        }

        let (windows_generator, windows_tester) = if config.target == CompilationTarget::WindowsNative {
            (Some(WindowsExecutableGenerator::new()), Some(WindowsExecutableTester::new()))
        } else {
            (None, None)
        };

        Ok(AetherCompiler {
            binary_path,
            config,
            diagnostics: DiagnosticEngine::new(),
            windows_generator,
            windows_tester,
        })
    }

    /// Validate syntax of Aether source file
    pub fn validate_syntax(&mut self, source_path: &Path) -> Result<SyntaxValidation, AetherCompilerError> {
        if !source_path.exists() {
            return Err(AetherCompilerError::SourceFileNotFound(source_path.to_path_buf()));
        }

        // Read source file
        let source_content = fs::read_to_string(source_path)
            .map_err(AetherCompilerError::IoError)?;

        // Add source to diagnostics engine
        self.diagnostics.add_source_file(
            source_path.to_string_lossy().to_string(),
            source_content.clone(),
        );

        // Parse the source code
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut ast = None;

        match self.parse_aether_source(&source_content, source_path) {
            Ok(parsed_ast) => {
                ast = Some(parsed_ast);
                
                // Perform additional semantic validation
                self.validate_semantics(&ast.as_ref().unwrap(), source_path, &mut warnings);
            }
            Err(parse_errors) => {
                errors.extend(parse_errors);
            }
        }

        let is_valid = errors.is_empty();

        Ok(SyntaxValidation {
            is_valid,
            ast,
            errors,
            warnings,
        })
    }

    /// Parse Aether source code into AST
    fn parse_aether_source(&self, source: &str, file_path: &Path) -> Result<AST, Vec<Diagnostic>> {
        let mut errors = Vec::new();

        // Handle both single expressions and multiple expressions
        let ast = if source.trim().starts_with('(') && source.trim().matches('(').count() == 1 {
            // Single S-expression
            match parse_sexpr(source) {
                Ok(ast) => ast,
                Err(parse_error) => {
                    let diagnostic = self.parse_error_to_diagnostic(parse_error, file_path);
                    errors.push(diagnostic);
                    return Err(errors);
                }
            }
        } else {
            // Multiple expressions - wrap in a program block
            match parse_multiple_sexprs(source) {
                Ok(asts) => {
                    if asts.is_empty() {
                        let pos = SourcePosition::new(
                            file_path.to_string_lossy().to_string(),
                            1, 1, 0
                        );
                        let span = SourceSpan::single(pos);
                        let diagnostic = Diagnostic::error(
                            diagnostic_codes::PARSE_ERROR.to_string(),
                            "Empty source file".to_string(),
                            span,
                        );
                        errors.push(diagnostic);
                        return Err(errors);
                    }

                    // Create a program AST with multiple expressions
                    let program_nodes = asts.into_iter()
                        .map(|ast| crate::compiler::ASTNodeRef::direct(ast.root))
                        .collect();
                    
                    AST::new(ASTNode::list(program_nodes))
                }
                Err(parse_error) => {
                    let diagnostic = self.parse_error_to_diagnostic(parse_error, file_path);
                    errors.push(diagnostic);
                    return Err(errors);
                }
            }
        };

        if errors.is_empty() {
            Ok(ast)
        } else {
            Err(errors)
        }
    }

    /// Convert parse error to diagnostic
    fn parse_error_to_diagnostic(
        &self,
        parse_error: crate::compiler::ParseError,
        file_path: &Path,
    ) -> Diagnostic {
        let (message, position) = match parse_error {
            crate::compiler::ParseError::UnexpectedEof => {
                ("Unexpected end of input".to_string(), 0)
            }
            crate::compiler::ParseError::UnexpectedChar(ch, pos) => {
                (format!("Unexpected character '{}'", ch), pos)
            }
            crate::compiler::ParseError::UnterminatedString(pos) => {
                ("Unterminated string literal".to_string(), pos)
            }
            crate::compiler::ParseError::InvalidNumber(num, pos) => {
                (format!("Invalid number format '{}'", num), pos)
            }
            crate::compiler::ParseError::UnmatchedParen(pos) => {
                ("Unmatched parenthesis".to_string(), pos)
            }
            crate::compiler::ParseError::InvalidDatumLabel(label, pos) => {
                (format!("Invalid datum label '{}'", label), pos)
            }
            crate::compiler::ParseError::UndefinedLabel(label, pos) => {
                (format!("Undefined datum label '{}'", label), pos)
            }
            crate::compiler::ParseError::CyclicLabel(label, pos) => {
                (format!("Cyclic datum label definition '{}'", label), pos)
            }
        };

        let source_pos = SourcePosition::new(
            file_path.to_string_lossy().to_string(),
            1, // Would need proper line/column calculation
            position + 1,
            position,
        );
        let span = SourceSpan::single(source_pos);

        Diagnostic::error(
            diagnostic_codes::PARSE_ERROR.to_string(),
            message,
            span,
        )
    }

    /// Perform semantic validation on AST
    fn validate_semantics(&self, ast: &AST, _file_path: &Path, warnings: &mut Vec<Diagnostic>) {
        // Basic semantic checks
        self.check_function_definitions(ast, warnings);
        self.check_variable_usage(ast, warnings);
        self.check_type_consistency(ast, warnings);
    }

    /// Check function definitions for common issues
    fn check_function_definitions(&self, ast: &AST, warnings: &mut Vec<Diagnostic>) {
        // Look for function definitions and check for common issues
        self.traverse_ast_for_functions(&ast.root, warnings);
    }

    /// Traverse AST looking for function definitions
    fn traverse_ast_for_functions(&self, node: &ASTNode, warnings: &mut Vec<Diagnostic>) {
        match node {
            ASTNode::List(children) => {
                // Check if this is a function definition
                if let Some(_first_child) = children.first() {
                    if let Some(ASTNode::Atom(AtomValue::Symbol(symbol))) = 
                        children.first().and_then(|r| match r {
                            crate::compiler::ASTNodeRef::Direct(node) => Some(node.as_ref()),
                            _ => None,
                        }) {
                        if symbol == "func" || symbol == "fn" {
                            self.validate_function_definition(children, warnings);
                        }
                    }
                }

                // Recursively check children
                for child_ref in children {
                    if let crate::compiler::ASTNodeRef::Direct(child) = child_ref {
                        self.traverse_ast_for_functions(child, warnings);
                    }
                }
            }
            _ => {}
        }
    }

    /// Validate a function definition
    fn validate_function_definition(
        &self,
        _children: &[crate::compiler::ASTNodeRef],
        _warnings: &mut Vec<Diagnostic>,
    ) {
        // Function definition validation logic would go here
        // For now, this is a placeholder
    }

    /// Check variable usage patterns
    fn check_variable_usage(&self, _ast: &AST, _warnings: &mut Vec<Diagnostic>) {
        // Variable usage analysis would go here
        // This could detect unused variables, uninitialized variables, etc.
    }

    /// Check type consistency
    fn check_type_consistency(&self, _ast: &AST, _warnings: &mut Vec<Diagnostic>) {
        // Type consistency checks would go here
        // This could detect potential type mismatches, etc.
    }

    /// Compile Aether source to Windows executable
    pub fn compile_to_executable(
        &mut self,
        source_path: &Path,
        output_path: &Path,
    ) -> Result<AetherCompilationResult, AetherCompilerError> {
        if !source_path.exists() {
            return Err(AetherCompilerError::SourceFileNotFound(source_path.to_path_buf()));
        }

        // First validate syntax
        let validation = self.validate_syntax(source_path)?;
        if !validation.is_valid {
            return Err(AetherCompilerError::SyntaxError(validation.errors));
        }

        // Create output directory if it doesn't exist
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .map_err(AetherCompilerError::IoError)?;
        }

        // Determine final output path with proper extension
        let final_output_path = if self.config.target == CompilationTarget::WindowsNative {
            let mut path = output_path.to_path_buf();
            if path.extension().is_none() {
                path.set_extension("exe");
            }
            path
        } else {
            output_path.to_path_buf()
        };

        // Build compiler command
        let mut cmd = Command::new(&self.binary_path);
        
        // Add source file
        cmd.arg(source_path);
        
        // Add output path
        cmd.arg("-o").arg(&final_output_path);
        
        // Add target specification
        match self.config.target {
            CompilationTarget::WindowsNative => {
                cmd.arg("--target").arg("windows-native");
            }
            CompilationTarget::WebAssembly => {
                cmd.arg("--target").arg("wasm32");
            }
            CompilationTarget::GPU => {
                cmd.arg("--target").arg("gpu");
            }
            CompilationTarget::Mobile => {
                cmd.arg("--target").arg("mobile");
            }
        }
        
        // Add optimization level
        cmd.arg("-O").arg(self.config.optimization_level.to_string());
        
        // Add debug info if enabled
        if self.config.debug_info {
            cmd.arg("--debug");
        }
        
        // Add verbose output if enabled
        if self.config.verbose {
            cmd.arg("--verbose");
        }
        
        // Add additional compiler flags
        for flag in &self.config.compiler_flags {
            cmd.arg(flag);
        }

        // Execute compilation
        let start_time = Instant::now();
        let output = cmd.output()
            .map_err(AetherCompilerError::IoError)?;
        let mut compilation_time = start_time.elapsed();

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);
        let mut success = output.status.success();

        // Parse diagnostics from compiler output
        let diagnostics = self.parse_compiler_diagnostics(&stderr);

        let mut executable_path = if success && final_output_path.exists() {
            Some(final_output_path.clone())
        } else {
            None
        };

        // If targeting Windows and compilation succeeded, apply Windows-specific processing
        if success && self.config.target == CompilationTarget::WindowsNative {
            if let Some(ref mut generator) = self.windows_generator {
                match generator.generate_executable(&final_output_path, &final_output_path) {
                    Ok(windows_result) => {
                        compilation_time += windows_result.generation_time;
                        executable_path = Some(windows_result.executable_path);
                        if !windows_result.success {
                            success = false;
                        }
                    }
                    Err(e) => {
                        success = false;
                        return Err(AetherCompilerError::ExecutableGenerationFailed(
                            format!("Windows executable generation failed: {}", e)
                        ));
                    }
                }
            }
        }

        Ok(AetherCompilationResult {
            success,
            executable_path,
            compilation_time,
            stdout,
            stderr,
            exit_code,
            diagnostics,
        })
    }

    /// Parse diagnostics from compiler stderr output
    fn parse_compiler_diagnostics(&self, stderr: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        
        // Simple diagnostic parsing - in a real implementation this would be more sophisticated
        for line in stderr.lines() {
            if line.contains("error:") {
                let pos = SourcePosition::unknown();
                let span = SourceSpan::single(pos);
                let diagnostic = Diagnostic::error(
                    "COMPILER_ERROR".to_string(),
                    line.to_string(),
                    span,
                );
                diagnostics.push(diagnostic);
            } else if line.contains("warning:") {
                let pos = SourcePosition::unknown();
                let span = SourceSpan::single(pos);
                let diagnostic = Diagnostic::warning(
                    "COMPILER_WARNING".to_string(),
                    line.to_string(),
                    span,
                );
                diagnostics.push(diagnostic);
            }
        }
        
        diagnostics
    }

    /// Generate diagnostics report
    pub fn generate_diagnostics(&self, error: &AetherCompilerError) -> DiagnosticReport {
        let mut diagnostics = Vec::new();
        let mut error_count = 0;
        let mut warning_count = 0;

        match error {
            AetherCompilerError::SyntaxError(syntax_errors) => {
                diagnostics.extend(syntax_errors.clone());
                error_count = syntax_errors.len();
            }
            AetherCompilerError::CompilationFailed { diagnostics: comp_diagnostics, .. } => {
                diagnostics.extend(comp_diagnostics.clone());
                for diag in comp_diagnostics {
                    match diag.severity {
                        DiagnosticSeverity::Error => error_count += 1,
                        DiagnosticSeverity::Warning => warning_count += 1,
                        _ => {}
                    }
                }
            }
            _ => {
                // Create a diagnostic for other error types
                let pos = SourcePosition::unknown();
                let span = SourceSpan::single(pos);
                let diagnostic = Diagnostic::error(
                    "AETHER_COMPILER_ERROR".to_string(),
                    error.to_string(),
                    span,
                );
                diagnostics.push(diagnostic);
                error_count = 1;
            }
        }

        // Format diagnostics
        let mut engine = DiagnosticEngine::new();
        for diagnostic in &diagnostics {
            engine.emit(diagnostic.clone());
        }
        let formatted_output = engine.format_diagnostics();

        DiagnosticReport {
            diagnostics,
            error_count,
            warning_count,
            formatted_output,
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &AetherCompilationConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: AetherCompilationConfig) {
        let old_target = self.config.target.clone();
        self.config = config;

        // Update Windows components if target changed
        if old_target != self.config.target {
            if self.config.target == CompilationTarget::WindowsNative {
                // Initialize Windows components
                if self.windows_generator.is_none() {
                    self.windows_generator = Some(WindowsExecutableGenerator::new());
                }
                if self.windows_tester.is_none() {
                    self.windows_tester = Some(WindowsExecutableTester::new());
                }
            } else {
                // Remove Windows components for non-Windows targets
                self.windows_generator = None;
                self.windows_tester = None;
            }
        }
    }

    /// Get compiler binary path
    pub fn binary_path(&self) -> &Path {
        &self.binary_path
    }

    /// Set compiler binary path
    pub fn set_binary_path(&mut self, path: PathBuf) -> Result<(), AetherCompilerError> {
        if !path.exists() {
            return Err(AetherCompilerError::CompilerNotFound(path));
        }
        self.binary_path = path;
        Ok(())
    }

    /// Check if compiler binary exists and is executable
    pub fn verify_compiler(&self) -> Result<(), AetherCompilerError> {
        if !self.binary_path.exists() {
            return Err(AetherCompilerError::CompilerNotFound(self.binary_path.clone()));
        }

        // Try to run the compiler with --version to verify it works
        let output = Command::new(&self.binary_path)
            .arg("--version")
            .output()
            .map_err(AetherCompilerError::IoError)?;

        if !output.status.success() {
            return Err(AetherCompilerError::ExecutableGenerationFailed(
                "Compiler binary is not executable or corrupted".to_string()
            ));
        }

        Ok(())
    }

    /// Get compiler version information
    pub fn get_version(&self) -> Result<String, AetherCompilerError> {
        let output = Command::new(&self.binary_path)
            .arg("--version")
            .output()
            .map_err(AetherCompilerError::IoError)?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Err(AetherCompilerError::ExecutableGenerationFailed(
                "Failed to get compiler version".to_string()
            ))
        }
    }

    /// List supported compilation targets
    pub fn supported_targets(&self) -> Vec<CompilationTarget> {
        vec![
            CompilationTarget::WindowsNative,
            CompilationTarget::WebAssembly,
            CompilationTarget::GPU,
            CompilationTarget::Mobile,
        ]
    }

    /// Validate compilation configuration
    pub fn validate_config(&self, config: &AetherCompilationConfig) -> Result<(), AetherCompilerError> {
        // Check optimization level
        if config.optimization_level > 3 {
            return Err(AetherCompilerError::InvalidConfiguration(
                "Optimization level must be between 0 and 3".to_string()
            ));
        }

        // Check output directory
        if let Some(parent) = config.output_directory.parent() {
            if !parent.exists() {
                return Err(AetherCompilerError::InvalidConfiguration(
                    format!("Output directory parent does not exist: {}", parent.display())
                ));
            }
        }

        // Validate compiler flags
        for flag in &config.compiler_flags {
            if flag.is_empty() {
                return Err(AetherCompilerError::InvalidConfiguration(
                    "Empty compiler flag is not allowed".to_string()
                ));
            }
        }

        Ok(())
    }

    /// Test Windows executable compatibility and functionality
    pub fn test_windows_executable(
        &mut self,
        executable_path: &Path,
    ) -> Result<WindowsTestResult, AetherCompilerError> {
        if self.config.target != CompilationTarget::WindowsNative {
            return Err(AetherCompilerError::InvalidConfiguration(
                "Windows executable testing is only available for WindowsNative target".to_string()
            ));
        }

        if let Some(ref mut tester) = self.windows_tester {
            tester.test_executable(executable_path)
                .map_err(|e| AetherCompilerError::ExecutableGenerationFailed(
                    format!("Windows executable testing failed: {}", e)
                ))
        } else {
            Err(AetherCompilerError::InvalidConfiguration(
                "Windows tester not initialized".to_string()
            ))
        }
    }

    /// Configure Windows-specific executable settings
    pub fn configure_windows_executable(
        &mut self,
        windows_config: WindowsExecutableConfig,
    ) -> Result<(), AetherCompilerError> {
        if self.config.target != CompilationTarget::WindowsNative {
            return Err(AetherCompilerError::InvalidConfiguration(
                "Windows configuration is only available for WindowsNative target".to_string()
            ));
        }

        if let Some(ref mut generator) = self.windows_generator {
            generator.update_config(windows_config);
            Ok(())
        } else {
            Err(AetherCompilerError::InvalidConfiguration(
                "Windows generator not initialized".to_string()
            ))
        }
    }

    /// Configure Windows-specific testing settings
    pub fn configure_windows_testing(
        &mut self,
        test_config: WindowsTestConfig,
    ) -> Result<(), AetherCompilerError> {
        if self.config.target != CompilationTarget::WindowsNative {
            return Err(AetherCompilerError::InvalidConfiguration(
                "Windows testing configuration is only available for WindowsNative target".to_string()
            ));
        }

        if let Some(ref mut tester) = self.windows_tester {
            tester.update_config(test_config);
            Ok(())
        } else {
            Err(AetherCompilerError::InvalidConfiguration(
                "Windows tester not initialized".to_string()
            ))
        }
    }

    /// Get Windows executable configuration (if available)
    pub fn get_windows_config(&self) -> Option<&WindowsExecutableConfig> {
        self.windows_generator.as_ref().map(|g| g.config())
    }

    /// Get Windows testing configuration (if available)
    pub fn get_windows_test_config(&self) -> Option<&WindowsTestConfig> {
        self.windows_tester.as_ref().map(|t| t.config())
    }

    /// Check if Windows-specific features are available
    pub fn has_windows_support(&self) -> bool {
        self.windows_generator.is_some() && self.windows_tester.is_some()
    }

    /// Generate Windows executable with custom configuration
    pub fn generate_windows_executable(
        &mut self,
        source_path: &Path,
        output_path: &Path,
        windows_config: WindowsExecutableConfig,
    ) -> Result<WindowsExecutableResult, AetherCompilerError> {
        if self.config.target != CompilationTarget::WindowsNative {
            return Err(AetherCompilerError::InvalidConfiguration(
                "Windows executable generation is only available for WindowsNative target".to_string()
            ));
        }

        // First compile the Aether source
        let compilation_result = self.compile_to_executable(source_path, output_path)?;
        
        if !compilation_result.success {
            return Err(AetherCompilerError::CompilationFailed {
                exit_code: compilation_result.exit_code,
                stderr: compilation_result.stderr,
                diagnostics: compilation_result.diagnostics,
            });
        }

        let executable_path = compilation_result.executable_path
            .ok_or_else(|| AetherCompilerError::ExecutableGenerationFailed(
                "Compilation succeeded but no executable was generated".to_string()
            ))?;

        // Apply Windows-specific processing
        if let Some(ref mut generator) = self.windows_generator {
            generator.update_config(windows_config);
            generator.generate_executable(&executable_path, &executable_path)
                .map_err(|e| AetherCompilerError::ExecutableGenerationFailed(
                    format!("Windows executable generation failed: {}", e)
                ))
        } else {
            Err(AetherCompilerError::InvalidConfiguration(
                "Windows generator not initialized".to_string()
            ))
        }
    }
}

/// Helper functions for working with Aether compilation
pub mod aether_compiler_helpers {
    use super::*;

    /// Create default Windows native compilation config
    pub fn windows_native_config(output_dir: PathBuf) -> AetherCompilationConfig {
        AetherCompilationConfig {
            target: CompilationTarget::WindowsNative,
            optimization_level: 2,
            debug_info: true,
            compiler_flags: vec![
                "--enable-ffi".to_string(),
                "--enable-gpu".to_string(),
            ],
            output_directory: output_dir,
            verbose: false,
        }
    }

    /// Create debug compilation config
    pub fn debug_config(output_dir: PathBuf) -> AetherCompilationConfig {
        AetherCompilationConfig {
            target: CompilationTarget::WindowsNative,
            optimization_level: 0,
            debug_info: true,
            compiler_flags: vec![
                "--debug-symbols".to_string(),
                "--enable-assertions".to_string(),
            ],
            output_directory: output_dir,
            verbose: true,
        }
    }

    /// Create release compilation config
    pub fn release_config(output_dir: PathBuf) -> AetherCompilationConfig {
        AetherCompilationConfig {
            target: CompilationTarget::WindowsNative,
            optimization_level: 3,
            debug_info: false,
            compiler_flags: vec![
                "--strip-symbols".to_string(),
                "--enable-lto".to_string(),
            ],
            output_directory: output_dir,
            verbose: false,
        }
    }

    /// Determine output executable name from source file
    pub fn determine_executable_name(source_path: &Path, target: &CompilationTarget) -> PathBuf {
        let stem = source_path.file_stem()
            .unwrap_or_else(|| std::ffi::OsStr::new("output"))
            .to_string_lossy();

        match target {
            CompilationTarget::WindowsNative => PathBuf::from(format!("{}.exe", stem)),
            CompilationTarget::WebAssembly => PathBuf::from(format!("{}.wasm", stem)),
            CompilationTarget::GPU => PathBuf::from(format!("{}.ptx", stem)),
            CompilationTarget::Mobile => PathBuf::from(format!("{}.aether", stem)),
        }
    }

    /// Create Windows-specific compilation configuration
    pub fn windows_compilation_config(output_dir: PathBuf) -> AetherCompilationConfig {
        AetherCompilationConfig {
            target: CompilationTarget::WindowsNative,
            optimization_level: 2,
            debug_info: true,
            compiler_flags: vec![
                "--enable-windows-features".to_string(),
                "--bundle-dependencies".to_string(),
            ],
            output_directory: output_dir,
            verbose: false,
        }
    }

    /// Create Windows debug compilation configuration
    pub fn windows_debug_config(output_dir: PathBuf) -> AetherCompilationConfig {
        AetherCompilationConfig {
            target: CompilationTarget::WindowsNative,
            optimization_level: 0,
            debug_info: true,
            compiler_flags: vec![
                "--enable-windows-features".to_string(),
                "--bundle-dependencies".to_string(),
                "--debug-symbols".to_string(),
                "--enable-assertions".to_string(),
            ],
            output_directory: output_dir,
            verbose: true,
        }
    }

    /// Create Windows release compilation configuration
    pub fn windows_release_config(output_dir: PathBuf) -> AetherCompilationConfig {
        AetherCompilationConfig {
            target: CompilationTarget::WindowsNative,
            optimization_level: 3,
            debug_info: false,
            compiler_flags: vec![
                "--enable-windows-features".to_string(),
                "--bundle-dependencies".to_string(),
                "--strip-symbols".to_string(),
                "--enable-lto".to_string(),
            ],
            output_directory: output_dir,
            verbose: false,
        }
    }

    /// Check if target requires Windows-specific processing
    pub fn requires_windows_processing(target: &CompilationTarget) -> bool {
        matches!(target, CompilationTarget::WindowsNative)
    }

    /// Get recommended Windows executable configuration for Aether
    pub fn recommended_windows_executable_config() -> WindowsExecutableConfig {
        use crate::build_system::windows::{WindowsExecutableConfig, WindowsArchitecture, WindowsSubsystem, WindowsVersionInfo};
        
        WindowsExecutableConfig {
            target_arch: WindowsArchitecture::X64,
            subsystem: WindowsSubsystem::Console,
            bundle_dependencies: true,
            include_debug_info: true,
            manifest_path: None,
            icon_path: None,
            version_info: WindowsVersionInfo {
                file_version: "0.1.0.0".to_string(),
                product_version: "0.1.0".to_string(),
                company_name: "Aether Language Team".to_string(),
                file_description: "Aether Language Executable".to_string(),
                product_name: "Aether Language".to_string(),
                copyright: "Copyright © 2024 Aether Language Team".to_string(),
            },
            linker_flags: vec![
                "/SUBSYSTEM:CONSOLE".to_string(),
                "/DYNAMICBASE".to_string(),
                "/NXCOMPAT".to_string(),
            ],
        }
    }

    /// Get recommended Windows testing configuration for Aether
    pub fn recommended_windows_test_config() -> WindowsTestConfig {
        use crate::build_system::windows::{WindowsTestConfig, WindowsVersion, WindowsArchitecture};
        use std::time::Duration;
        
        WindowsTestConfig {
            timeout: Duration::from_secs(60),
            test_compatibility: true,
            test_versions: vec![WindowsVersion::Windows10, WindowsVersion::Windows11],
            test_architectures: vec![WindowsArchitecture::X64],
            validate_dependencies: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_aether_compilation_config_default() {
        let config = AetherCompilationConfig::default();
        assert_eq!(config.target, CompilationTarget::WindowsNative);
        assert_eq!(config.optimization_level, 2);
        assert!(config.debug_info);
    }

    #[test]
    fn test_syntax_validation_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let source_path = temp_dir.path().join("empty.ae");
        fs::write(&source_path, "").unwrap();

        // Create a mock compiler (this would fail in real usage)
        let compiler_path = temp_dir.path().join("aetherc.exe");
        fs::write(&compiler_path, "mock compiler").unwrap();

        let mut compiler = AetherCompiler::new(compiler_path).unwrap();
        let validation = compiler.validate_syntax(&source_path).unwrap();
        
        assert!(!validation.is_valid);
        assert!(!validation.errors.is_empty());
    }

    #[test]
    fn test_syntax_validation_simple_program() {
        let temp_dir = TempDir::new().unwrap();
        let source_path = temp_dir.path().join("hello.ae");
        fs::write(&source_path, "(func main () (call print \"Hello, World!\") (return 0))").unwrap();

        let compiler_path = temp_dir.path().join("aetherc.exe");
        fs::write(&compiler_path, "mock compiler").unwrap();

        let mut compiler = AetherCompiler::new(compiler_path).unwrap();
        let validation = compiler.validate_syntax(&source_path).unwrap();
        
        assert!(validation.is_valid);
        assert!(validation.ast.is_some());
        assert!(validation.errors.is_empty());
    }

    #[test]
    fn test_config_validation() {
        let temp_dir = TempDir::new().unwrap();
        let compiler_path = temp_dir.path().join("aetherc.exe");
        fs::write(&compiler_path, "mock compiler").unwrap();

        let compiler = AetherCompiler::new(compiler_path).unwrap();
        
        // Valid config
        let valid_config = AetherCompilationConfig::default();
        assert!(compiler.validate_config(&valid_config).is_ok());
        
        // Invalid optimization level
        let mut invalid_config = AetherCompilationConfig::default();
        invalid_config.optimization_level = 5;
        assert!(compiler.validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_determine_executable_name() {
        use aether_compiler_helpers::*;
        
        let source_path = Path::new("test.ae");
        
        let windows_name = determine_executable_name(source_path, &CompilationTarget::WindowsNative);
        assert_eq!(windows_name, PathBuf::from("test.exe"));
        
        let wasm_name = determine_executable_name(source_path, &CompilationTarget::WebAssembly);
        assert_eq!(wasm_name, PathBuf::from("test.wasm"));
    }
}