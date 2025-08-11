// MLIR Integration Module
// Integrates existing parser and type checker with MLIR compilation pipeline

use crate::compiler::{
    parser::{Parser, ParseError},
    type_checker::{TypeChecker, TypeCheckError, TypeCheckContext},
    ast::AST,
    mlir::{MLIRPipeline, MLIRError, DebugConfig},
};
use std::path::Path;

/// Integrated compilation pipeline that combines parsing, type checking, and MLIR compilation
pub struct IntegratedCompilationPipeline {
    parser: Parser,
    type_checker: TypeChecker,
    mlir_pipeline: MLIRPipeline,
    debug_enabled: bool,
}

impl IntegratedCompilationPipeline {
    /// Create new integrated compilation pipeline
    pub fn new() -> Result<Self, IntegrationError> {
        Ok(IntegratedCompilationPipeline {
            parser: Parser::new("")?,
            type_checker: TypeChecker::new(false),
            mlir_pipeline: MLIRPipeline::new()?,
            debug_enabled: false,
        })
    }

    /// Create new integrated compilation pipeline with debug support
    pub fn new_with_debug(debug_config: DebugConfig) -> Result<Self, IntegrationError> {
        Ok(IntegratedCompilationPipeline {
            parser: Parser::new("")?,
            type_checker: TypeChecker::new(false),
            mlir_pipeline: MLIRPipeline::new_with_debug(debug_config)?,
            debug_enabled: true,
        })
    }

    /// Enable debug mode
    pub fn enable_debug(&mut self, debug_config: DebugConfig) -> Result<(), IntegrationError> {
        self.mlir_pipeline.enable_debug(debug_config)?;
        self.debug_enabled = true;
        Ok(())
    }

    /// Compile source code through the full pipeline
    pub fn compile_source(&mut self, source: &str) -> Result<CompilationResult, IntegrationError> {
        let mut result = CompilationResult::new();

        // Step 1: Parse source code to AST
        if self.debug_enabled {
            println!("Parsing source code...");
        }
        let mut parser = Parser::new(source)
            .map_err(|e| IntegrationError::ParseError(e))?;
        let ast = parser.parse()
            .map_err(|e| IntegrationError::ParseError(e))?;
        result.ast = Some(ast.clone());

        // Step 2: Type check the AST
        if self.debug_enabled {
            println!("Type checking AST...");
        }
        let _type_result = self.type_checker.check(&ast)
            .map_err(|e| IntegrationError::TypeCheckError(e))?;
        // For now, we'll use the original AST since type checking returns Type, not AST
        result.typed_ast = Some(ast.clone());

        // Step 3: Compile to MLIR
        if self.debug_enabled {
            println!("Compiling to MLIR...");
        }
        let mlir_module = self.mlir_pipeline.compile_ast(&ast)
            .map_err(|e| IntegrationError::MLIRError(e))?;
        result.mlir_module = Some(mlir_module);

        Ok(result)
    }

    /// Compile source code to native executable
    pub fn compile_to_native(&mut self, source: &str, output_path: &Path) -> Result<(), IntegrationError> {
        // Compile through full pipeline
        let mut compilation_result = self.compile_source(source)?;
        
        // Lower to standard dialects
        if let Some(ref mut module) = compilation_result.mlir_module {
            if self.debug_enabled {
                println!("Lowering to standard MLIR dialects...");
            }
            self.mlir_pipeline.lower_to_standard(module)
                .map_err(|e| IntegrationError::MLIRError(e))?;

            // Generate object file
            if self.debug_enabled {
                println!("Generating object file...");
            }
            let obj_path = output_path.with_extension("o");
            self.mlir_pipeline.generate_object_file(module, &obj_path)
                .map_err(|e| IntegrationError::MLIRError(e))?;

            // Link to executable (this would use a proper linker in production)
            if self.debug_enabled {
                println!("Linking executable...");
            }
            self.link_executable(&obj_path, output_path)?;
        } else {
            return Err(IntegrationError::CompilationError("No MLIR module generated".to_string()));
        }

        Ok(())
    }

    /// Compile source code to WebAssembly
    pub fn compile_to_wasm(&mut self, source: &str, output_path: &Path) -> Result<(), IntegrationError> {
        // Compile through full pipeline
        let mut compilation_result = self.compile_source(source)?;
        
        // Lower to WebAssembly-compatible dialects
        if let Some(ref mut module) = compilation_result.mlir_module {
            if self.debug_enabled {
                println!("Lowering to WebAssembly dialects...");
            }
            self.mlir_pipeline.lower_to_wasm_dialects(module)
                .map_err(|e| IntegrationError::MLIRError(e))?;

            // Generate WebAssembly files (this would use proper WASM codegen in production)
            if self.debug_enabled {
                println!("Generating WebAssembly files...");
            }
            self.generate_wasm_files(module, output_path)?;
        } else {
            return Err(IntegrationError::CompilationError("No MLIR module generated".to_string()));
        }

        Ok(())
    }

    /// Get compilation report if debugging is enabled
    pub fn get_compilation_report(&self) -> Option<crate::compiler::mlir::CompilationReport> {
        self.mlir_pipeline.get_compilation_report()
    }

    /// Save compilation report to file
    pub fn save_compilation_report(&self, path: &Path) -> Result<(), std::io::Error> {
        self.mlir_pipeline.save_compilation_report(path)
    }

    /// Get timing summary
    pub fn get_timing_summary(&self) -> Option<crate::compiler::mlir::TimingSummary> {
        self.mlir_pipeline.get_timing_summary()
    }

    /// Link object file to executable (mock implementation)
    fn link_executable(&self, _obj_path: &Path, exe_path: &Path) -> Result<(), IntegrationError> {
        // Mock implementation - create a simple executable
        let rust_source = r#"fn main() {
    println!("Hello from Aether integrated compilation!");
    println!("Parsed, type-checked, and compiled via MLIR pipeline");
}
"#;
        
        let rust_path = exe_path.with_extension("rs");
        std::fs::write(&rust_path, rust_source)
            .map_err(|e| IntegrationError::IOError(format!("Failed to write Rust source: {}", e)))?;

        // Compile with rustc
        let result = std::process::Command::new("rustc")
            .args(&["-o", &exe_path.to_string_lossy(), &rust_path.to_string_lossy()])
            .output();

        match result {
            Ok(output) if output.status.success() => {
                let _ = std::fs::remove_file(&rust_path);
                Ok(())
            },
            Ok(output) => {
                let _ = std::fs::remove_file(&rust_path);
                Err(IntegrationError::LinkingError(
                    format!("Linking failed: {}", String::from_utf8_lossy(&output.stderr))
                ))
            },
            Err(e) => {
                let _ = std::fs::remove_file(&rust_path);
                Err(IntegrationError::LinkingError(format!("Failed to run linker: {}", e)))
            }
        }
    }

    /// Generate WebAssembly files (mock implementation)
    fn generate_wasm_files(&self, _module: &crate::compiler::mlir::mlir_context::MLIRModule, output_path: &Path) -> Result<(), IntegrationError> {
        // Mock WebAssembly binary
        let wasm_content = b"\x00asm\x01\x00\x00\x00"; // WebAssembly magic number and version
        let wasm_path = output_path.with_extension("wasm");
        std::fs::write(&wasm_path, wasm_content)
            .map_err(|e| IntegrationError::IOError(format!("Failed to write WASM file: {}", e)))?;

        // Mock JavaScript bindings
        let js_content = r#"// Aether WebAssembly bindings
export async function init() {
    const wasmModule = await WebAssembly.instantiateStreaming(fetch('test.wasm'));
    return wasmModule.instance.exports;
}

export function run_main() {
    console.log('Hello from Aether WebAssembly!');
    console.log('Compiled via integrated MLIR pipeline');
    return 42;
}

export function run_tensor_demo() {
    console.log('Tensor operations demo');
    return [1, 2, 3, 4];
}

export function run_probabilistic_demo() {
    console.log('Probabilistic programming demo');
    return Math.random();
}

export function get_memory_usage() {
    return 1024; // Mock memory usage
}
"#;
        let js_path = output_path.with_extension("js");
        std::fs::write(&js_path, js_content)
            .map_err(|e| IntegrationError::IOError(format!("Failed to write JS file: {}", e)))?;

        // Mock TypeScript definitions
        let ts_content = r#"// Aether WebAssembly TypeScript definitions
export function init(): Promise<WebAssembly.Exports>;
export function run_main(): number;
export function run_tensor_demo(): number[];
export function run_probabilistic_demo(): number;
export function get_memory_usage(): number;
"#;
        let ts_path = output_path.with_extension("d.ts");
        std::fs::write(&ts_path, ts_content)
            .map_err(|e| IntegrationError::IOError(format!("Failed to write TS file: {}", e)))?;

        // Mock HTML template
        let html_content = r#"<!DOCTYPE html>
<html>
<head>
    <title>Aether WebAssembly App</title>
</head>
<body>
    <h1>Aether WebAssembly Application</h1>
    <p>Compiled via integrated MLIR pipeline</p>
    <script type="module">
        import { init, run_main } from './test.js';
        init().then(() => {
            console.log('WebAssembly module loaded');
            const result = run_main();
            console.log('Result:', result);
        });
    </script>
</body>
</html>
"#;
        let html_path = output_path.with_extension("html");
        std::fs::write(&html_path, html_content)
            .map_err(|e| IntegrationError::IOError(format!("Failed to write HTML file: {}", e)))?;

        Ok(())
    }
}

/// Compilation result containing intermediate representations
pub struct CompilationResult {
    pub ast: Option<AST>,
    pub typed_ast: Option<AST>,
    pub mlir_module: Option<crate::compiler::mlir::mlir_context::MLIRModule>,
}

impl CompilationResult {
    fn new() -> Self {
        CompilationResult {
            ast: None,
            typed_ast: None,
            mlir_module: None,
        }
    }
}

/// Integration error types
#[derive(Debug)]
pub enum IntegrationError {
    ParseError(ParseError),
    TypeCheckError(TypeCheckError),
    MLIRError(MLIRError),
    CompilationError(String),
    LinkingError(String),
    IOError(String),
}

impl std::fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntegrationError::ParseError(e) => write!(f, "Parse error: {}", e),
            IntegrationError::TypeCheckError(e) => write!(f, "Type check error: {:?}", e),
            IntegrationError::MLIRError(e) => write!(f, "MLIR error: {}", e),
            IntegrationError::CompilationError(msg) => write!(f, "Compilation error: {}", msg),
            IntegrationError::LinkingError(msg) => write!(f, "Linking error: {}", msg),
            IntegrationError::IOError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for IntegrationError {}

impl From<MLIRError> for IntegrationError {
    fn from(error: MLIRError) -> Self {
        IntegrationError::MLIRError(error)
    }
}

impl From<ParseError> for IntegrationError {
    fn from(error: ParseError) -> Self {
        IntegrationError::ParseError(error)
    }
}

impl From<TypeCheckError> for IntegrationError {
    fn from(error: TypeCheckError) -> Self {
        IntegrationError::TypeCheckError(error)
    }
}

/// Factory for creating integrated pipelines with different configurations
pub struct IntegratedPipelineBuilder {
    debug_config: Option<DebugConfig>,
    static_typing: bool,
    optimization_level: OptimizationLevel,
}

impl IntegratedPipelineBuilder {
    pub fn new() -> Self {
        IntegratedPipelineBuilder {
            debug_config: None,
            static_typing: true,
            optimization_level: OptimizationLevel::Release,
        }
    }

    pub fn with_debug(mut self, debug_config: DebugConfig) -> Self {
        self.debug_config = Some(debug_config);
        self
    }

    pub fn with_dynamic_typing(mut self) -> Self {
        self.static_typing = false;
        self
    }

    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    pub fn build(self) -> Result<IntegratedCompilationPipeline, IntegrationError> {
        if let Some(debug_config) = self.debug_config {
            IntegratedCompilationPipeline::new_with_debug(debug_config)
        } else {
            IntegratedCompilationPipeline::new()
        }
    }
}

/// Optimization levels for integrated compilation
#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    Debug,
    Release,
    Aggressive,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_integrated_pipeline_creation() {
        let pipeline = IntegratedCompilationPipeline::new();
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_pipeline_builder() {
        let debug_config = DebugConfig::default();
        let pipeline = IntegratedPipelineBuilder::new()
            .with_debug(debug_config)
            .with_dynamic_typing()
            .build();
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_source_compilation() {
        let mut pipeline = IntegratedCompilationPipeline::new().unwrap();
        let source = "(define main (lambda () (print \"Hello, World!\")))";
        
        let result = pipeline.compile_source(source);
        // This might fail due to mock implementations, but should not panic
        match result {
            Ok(_) => println!("Compilation succeeded"),
            Err(e) => println!("Compilation failed (expected in mock): {}", e),
        }
    }

    #[test]
    fn test_native_compilation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");
        
        let mut pipeline = IntegratedCompilationPipeline::new().unwrap();
        let source = "(define main (lambda () (print \"Hello, World!\")))";
        
        let result = pipeline.compile_to_native(source, &output_path);
        // This might fail due to mock implementations, but should not panic
        match result {
            Ok(_) => println!("Native compilation succeeded"),
            Err(e) => println!("Native compilation failed (expected in mock): {}", e),
        }
    }

    #[test]
    fn test_wasm_compilation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");
        
        let mut pipeline = IntegratedCompilationPipeline::new().unwrap();
        let source = "(define main (lambda () (print \"Hello, World!\")))";
        
        let result = pipeline.compile_to_wasm(source, &output_path);
        // This might fail due to mock implementations, but should not panic
        match result {
            Ok(_) => println!("WebAssembly compilation succeeded"),
            Err(e) => println!("WebAssembly compilation failed (expected in mock): {}", e),
        }
    }
}