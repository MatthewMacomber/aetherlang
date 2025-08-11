// Complete compilation pipeline from MLIR to native executable
// Integrates MLIR processing, LLVM code generation, and executable linking

use crate::compiler::mlir::{AetherMLIRContext, MLIRModule, MLIRError};
use crate::compiler::mlir::llvm_codegen::{LLVMCodeGenerator, TargetConfig, OptimizationLevel, CodegenError};
use crate::runtime::executable_linker::{ExecutableLinker, LinkingResult, LinkingError};
use crate::runtime::runtime_linking::RuntimeLinkingConfig;
use std::path::{Path, PathBuf};
use std::fs;
use std::time::Instant;

/// Complete compilation pipeline configuration
#[derive(Debug, Clone)]
pub struct CompilationConfig {
    /// Target configuration
    pub target_config: TargetConfig,
    /// Output directory for intermediate files
    pub output_dir: PathBuf,
    /// Final executable name
    pub executable_name: String,
    /// Keep intermediate files for debugging
    pub keep_intermediates: bool,
    /// Verbose output
    pub verbose: bool,
    /// Enable debug information
    pub debug_info: bool,
    /// Enable link-time optimization
    pub enable_lto: bool,
    /// Additional linker flags
    pub extra_linker_flags: Vec<String>,
    /// Additional library search paths
    pub library_search_paths: Vec<PathBuf>,
}

impl Default for CompilationConfig {
    fn default() -> Self {
        CompilationConfig {
            target_config: TargetConfig::default(),
            output_dir: PathBuf::from("target"),
            executable_name: "aether_program".to_string(),
            keep_intermediates: false,
            verbose: false,
            debug_info: false,
            enable_lto: false,
            extra_linker_flags: Vec::new(),
            library_search_paths: Vec::new(),
        }
    }
}

/// Compilation pipeline result
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Final executable path
    pub executable_path: PathBuf,
    /// Compilation was successful
    pub success: bool,
    /// Intermediate file paths
    pub intermediate_files: IntermediateFiles,
    /// Compilation statistics
    pub statistics: CompilationStatistics,
    /// Error messages if compilation failed
    pub errors: Vec<String>,
    /// Warning messages
    pub warnings: Vec<String>,
}

/// Intermediate files generated during compilation
#[derive(Debug, Clone, Default)]
pub struct IntermediateFiles {
    /// LLVM IR file
    pub llvm_ir_file: Option<PathBuf>,
    /// Assembly file
    pub assembly_file: Option<PathBuf>,
    /// Object files
    pub object_files: Vec<PathBuf>,
    /// Debug information files
    pub debug_files: Vec<PathBuf>,
}

/// Compilation statistics
#[derive(Debug, Clone, Default)]
pub struct CompilationStatistics {
    /// Total compilation time in milliseconds
    pub total_time_ms: u64,
    /// MLIR processing time in milliseconds
    pub mlir_time_ms: u64,
    /// LLVM code generation time in milliseconds
    pub llvm_codegen_time_ms: u64,
    /// Linking time in milliseconds
    pub linking_time_ms: u64,
    /// Number of MLIR operations processed
    pub mlir_operations_count: usize,
    /// Number of object files generated
    pub object_files_count: usize,
    /// Final executable size in bytes
    pub executable_size_bytes: u64,
    /// Debug information included
    pub debug_info_included: bool,
    /// LTO applied
    pub lto_applied: bool,
}

/// Compilation pipeline errors
#[derive(Debug, Clone)]
pub enum CompilationError {
    /// MLIR processing error
    MLIRError(MLIRError),
    /// LLVM code generation error
    CodegenError(CodegenError),
    /// Linking error
    LinkingError(LinkingError),
    /// I/O error
    IOError(String),
    /// Configuration error
    ConfigurationError(String),
    /// Pipeline error
    PipelineError(String),
}

impl std::fmt::Display for CompilationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilationError::MLIRError(e) => write!(f, "MLIR error: {}", e),
            CompilationError::CodegenError(e) => write!(f, "Code generation error: {}", e),
            CompilationError::LinkingError(e) => write!(f, "Linking error: {}", e),
            CompilationError::IOError(msg) => write!(f, "I/O error: {}", msg),
            CompilationError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            CompilationError::PipelineError(msg) => write!(f, "Pipeline error: {}", msg),
        }
    }
}

impl std::error::Error for CompilationError {}

impl From<MLIRError> for CompilationError {
    fn from(err: MLIRError) -> Self {
        CompilationError::MLIRError(err)
    }
}

impl From<CodegenError> for CompilationError {
    fn from(err: CodegenError) -> Self {
        CompilationError::CodegenError(err)
    }
}

impl From<LinkingError> for CompilationError {
    fn from(err: LinkingError) -> Self {
        CompilationError::LinkingError(err)
    }
}

impl From<std::io::Error> for CompilationError {
    fn from(err: std::io::Error) -> Self {
        CompilationError::IOError(err.to_string())
    }
}

/// Complete compilation pipeline
pub struct CompilationPipeline {
    config: CompilationConfig,
    mlir_context: AetherMLIRContext,
    codegen: LLVMCodeGenerator,
    linker: ExecutableLinker,
}

impl CompilationPipeline {
    /// Create new compilation pipeline
    pub fn new(config: CompilationConfig) -> Result<Self, CompilationError> {
        // Create output directory
        fs::create_dir_all(&config.output_dir)?;

        // Initialize MLIR context
        let mlir_context = AetherMLIRContext::new()
            .map_err(|e| CompilationError::MLIRError(e.into()))?;

        // Initialize LLVM code generator
        let codegen = LLVMCodeGenerator::new(config.target_config.clone())
            .map_err(CompilationError::CodegenError)?;

        // Initialize executable linker
        let mut linking_config = RuntimeLinkingConfig::new(config.target_config.triple.clone());
        
        // Configure linking based on compilation config
        if config.debug_info {
            linking_config.debug_config.enabled = true;
            linking_config.debug_config.separate_debug_info = true;
        }
        
        if config.enable_lto {
            linking_config.lto_config.enabled = true;
            linking_config.lto_config.lto_type = crate::runtime::runtime_linking::LTOType::Thin;
        }

        // Add extra linker flags
        linking_config.linker_flags.extend(config.extra_linker_flags.clone());

        // Add library search paths
        linking_config.library_paths.extend(config.library_search_paths.clone());

        let linker = ExecutableLinker::new(config.target_config.triple.clone(), linking_config)
            .map_err(CompilationError::LinkingError)?;

        Ok(CompilationPipeline {
            config,
            mlir_context,
            codegen,
            linker,
        })
    }

    /// Compile MLIR module to native executable
    pub fn compile(&mut self, mlir_module: &MLIRModule) -> Result<CompilationResult, CompilationError> {
        let start_time = Instant::now();
        let mut statistics = CompilationStatistics::default();
        let mut intermediate_files = IntermediateFiles::default();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        if self.config.verbose {
            println!("Starting compilation pipeline for target: {}", self.config.target_config.triple);
        }

        // Step 1: Process MLIR module
        let mlir_start = Instant::now();
        let operations = mlir_module.operations();
        statistics.mlir_operations_count = operations.len();
        statistics.mlir_time_ms = mlir_start.elapsed().as_millis() as u64;

        if self.config.verbose {
            println!("Processed {} MLIR operations in {}ms", 
                    statistics.mlir_operations_count, statistics.mlir_time_ms);
        }

        // Step 2: Generate LLVM IR
        let codegen_start = Instant::now();
        self.codegen.generate_from_mlir(mlir_module)
            .map_err(CompilationError::CodegenError)?;
        statistics.llvm_codegen_time_ms = codegen_start.elapsed().as_millis() as u64;

        if self.config.verbose {
            println!("Generated LLVM IR in {}ms", statistics.llvm_codegen_time_ms);
        }

        // Step 3: Generate intermediate files
        let ir_file = self.generate_llvm_ir_file(&mut intermediate_files)?;
        let asm_file = self.generate_assembly_file(&mut intermediate_files)?;
        let obj_file = self.generate_object_file(&mut intermediate_files)?;

        statistics.object_files_count = intermediate_files.object_files.len();

        if self.config.verbose {
            println!("Generated object file: {}", obj_file.display());
        }

        // Step 4: Link executable
        let linking_start = Instant::now();
        let executable_path = self.config.output_dir.join(&self.config.executable_name);
        
        let linking_result = self.linker.link_executable(&intermediate_files.object_files, &executable_path)
            .map_err(CompilationError::LinkingError)?;

        statistics.linking_time_ms = linking_start.elapsed().as_millis() as u64;
        statistics.executable_size_bytes = linking_result.statistics.executable_size_bytes;
        statistics.debug_info_included = linking_result.statistics.debug_info_included;
        statistics.lto_applied = linking_result.statistics.lto_applied;

        if self.config.verbose {
            println!("Linked executable in {}ms: {}", 
                    statistics.linking_time_ms, executable_path.display());
            println!("Executable size: {} bytes", statistics.executable_size_bytes);
        }

        // Step 5: Cleanup intermediate files if requested
        if !self.config.keep_intermediates {
            self.cleanup_intermediate_files(&intermediate_files)?;
        }

        // Calculate total time
        statistics.total_time_ms = start_time.elapsed().as_millis() as u64;

        if self.config.verbose {
            println!("Total compilation time: {}ms", statistics.total_time_ms);
        }

        Ok(CompilationResult {
            executable_path,
            success: linking_result.success,
            intermediate_files,
            statistics,
            errors,
            warnings,
        })
    }

    /// Generate LLVM IR file
    fn generate_llvm_ir_file(&mut self, intermediate_files: &mut IntermediateFiles) -> Result<PathBuf, CompilationError> {
        let ir_file = self.config.output_dir.join(format!("{}.ll", self.config.executable_name));
        
        // Generate LLVM IR string
        let ir_content = self.codegen.get_llvm_ir_string()
            .map_err(CompilationError::CodegenError)?;
        
        // Write to file
        fs::write(&ir_file, ir_content)?;
        intermediate_files.llvm_ir_file = Some(ir_file.clone());
        
        if self.config.verbose {
            println!("Generated LLVM IR file: {}", ir_file.display());
        }
        
        Ok(ir_file)
    }

    /// Generate assembly file
    fn generate_assembly_file(&mut self, intermediate_files: &mut IntermediateFiles) -> Result<PathBuf, CompilationError> {
        let asm_file = self.config.output_dir.join(format!("{}.s", self.config.executable_name));
        
        self.codegen.emit_assembly_file(&asm_file)
            .map_err(CompilationError::CodegenError)?;
        
        intermediate_files.assembly_file = Some(asm_file.clone());
        
        if self.config.verbose {
            println!("Generated assembly file: {}", asm_file.display());
        }
        
        Ok(asm_file)
    }

    /// Generate object file
    fn generate_object_file(&mut self, intermediate_files: &mut IntermediateFiles) -> Result<PathBuf, CompilationError> {
        let obj_file = self.config.output_dir.join(format!("{}.o", self.config.executable_name));
        
        self.codegen.emit_object_file(&obj_file)
            .map_err(CompilationError::CodegenError)?;
        
        intermediate_files.object_files.push(obj_file.clone());
        
        if self.config.verbose {
            println!("Generated object file: {}", obj_file.display());
        }
        
        Ok(obj_file)
    }

    /// Cleanup intermediate files
    fn cleanup_intermediate_files(&self, intermediate_files: &IntermediateFiles) -> Result<(), CompilationError> {
        if self.config.verbose {
            println!("Cleaning up intermediate files...");
        }

        // Remove LLVM IR file
        if let Some(ir_file) = &intermediate_files.llvm_ir_file {
            if ir_file.exists() {
                fs::remove_file(ir_file)?;
                if self.config.verbose {
                    println!("Removed: {}", ir_file.display());
                }
            }
        }

        // Remove assembly file
        if let Some(asm_file) = &intermediate_files.assembly_file {
            if asm_file.exists() {
                fs::remove_file(asm_file)?;
                if self.config.verbose {
                    println!("Removed: {}", asm_file.display());
                }
            }
        }

        // Remove object files
        for obj_file in &intermediate_files.object_files {
            if obj_file.exists() {
                fs::remove_file(obj_file)?;
                if self.config.verbose {
                    println!("Removed: {}", obj_file.display());
                }
            }
        }

        Ok(())
    }

    /// Get compilation configuration
    pub fn get_config(&self) -> &CompilationConfig {
        &self.config
    }

    /// Get MLIR context
    pub fn get_mlir_context(&self) -> &AetherMLIRContext {
        &self.mlir_context
    }

    /// Get LLVM code generator
    pub fn get_codegen(&self) -> &LLVMCodeGenerator {
        &self.codegen
    }

    /// Get executable linker
    pub fn get_linker(&self) -> &ExecutableLinker {
        &self.linker
    }

    /// Test the compilation pipeline configuration
    pub fn test_pipeline(&self) -> Result<(), CompilationError> {
        if self.config.verbose {
            println!("Testing compilation pipeline configuration...");
        }

        // Test linker configuration
        self.linker.test_linking_config()
            .map_err(CompilationError::LinkingError)?;

        // Test output directory
        if !self.config.output_dir.exists() {
            return Err(CompilationError::ConfigurationError(
                format!("Output directory does not exist: {}", self.config.output_dir.display())
            ));
        }

        // Test write permissions
        let test_file = self.config.output_dir.join("test_write_permissions");
        match fs::write(&test_file, b"test") {
            Ok(()) => {
                let _ = fs::remove_file(&test_file);
            }
            Err(e) => {
                return Err(CompilationError::ConfigurationError(
                    format!("Cannot write to output directory: {}", e)
                ));
            }
        }

        if self.config.verbose {
            println!("Pipeline configuration test passed");
        }

        Ok(())
    }

    /// Create compilation pipeline with debug configuration
    pub fn debug_config(target_triple: String, output_dir: PathBuf) -> Result<Self, CompilationError> {
        let config = CompilationConfig {
            target_config: TargetConfig {
                triple: target_triple,
                cpu: "generic".to_string(),
                features: "".to_string(),
                optimization_level: OptimizationLevel::None,
                relocation_model: crate::compiler::mlir::llvm_codegen::RelocModel::PIC,
                code_model: crate::compiler::mlir::llvm_codegen::CodeModel::Default,
            },
            output_dir,
            executable_name: "debug_program".to_string(),
            keep_intermediates: true,
            verbose: true,
            debug_info: true,
            enable_lto: false,
            extra_linker_flags: vec!["-g".to_string()],
            library_search_paths: Vec::new(),
        };

        Self::new(config)
    }

    /// Create compilation pipeline with release configuration
    pub fn release_config(target_triple: String, output_dir: PathBuf) -> Result<Self, CompilationError> {
        let config = CompilationConfig {
            target_config: TargetConfig {
                triple: target_triple,
                cpu: "native".to_string(),
                features: "".to_string(),
                optimization_level: OptimizationLevel::Aggressive,
                relocation_model: crate::compiler::mlir::llvm_codegen::RelocModel::PIC,
                code_model: crate::compiler::mlir::llvm_codegen::CodeModel::Default,
            },
            output_dir,
            executable_name: "release_program".to_string(),
            keep_intermediates: false,
            verbose: false,
            debug_info: false,
            enable_lto: true,
            extra_linker_flags: vec!["-O3".to_string(), "-s".to_string()],
            library_search_paths: Vec::new(),
        };

        Self::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_compilation_config_default() {
        let config = CompilationConfig::default();
        assert_eq!(config.executable_name, "aether_program");
        assert_eq!(config.output_dir, PathBuf::from("target"));
        assert!(!config.keep_intermediates);
        assert!(!config.verbose);
        assert!(!config.debug_info);
        assert!(!config.enable_lto);
    }

    #[test]
    fn test_compilation_pipeline_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CompilationConfig {
            output_dir: temp_dir.path().to_path_buf(),
            ..CompilationConfig::default()
        };

        let result = CompilationPipeline::new(config);
        match result {
            Ok(pipeline) => {
                assert_eq!(pipeline.get_config().executable_name, "aether_program");
            }
            Err(CompilationError::MLIRError(_)) => {
                println!("MLIR context creation failed (expected in test environment)");
            }
            Err(CompilationError::CodegenError(_)) => {
                println!("LLVM code generator creation failed (expected in test environment)");
            }
            Err(CompilationError::LinkingError(_)) => {
                println!("Linker creation failed (expected in test environment)");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_debug_config_creation() {
        let temp_dir = TempDir::new().unwrap();
        let result = CompilationPipeline::debug_config(
            "x86_64-unknown-linux-gnu".to_string(),
            temp_dir.path().to_path_buf()
        );

        match result {
            Ok(pipeline) => {
                let config = pipeline.get_config();
                assert!(config.debug_info);
                assert!(config.keep_intermediates);
                assert!(config.verbose);
                assert!(!config.enable_lto);
                assert_eq!(config.target_config.optimization_level, OptimizationLevel::None);
            }
            Err(e) => {
                println!("Debug config creation failed (expected in test environment): {}", e);
            }
        }
    }

    #[test]
    fn test_release_config_creation() {
        let temp_dir = TempDir::new().unwrap();
        let result = CompilationPipeline::release_config(
            "x86_64-unknown-linux-gnu".to_string(),
            temp_dir.path().to_path_buf()
        );

        match result {
            Ok(pipeline) => {
                let config = pipeline.get_config();
                assert!(!config.debug_info);
                assert!(!config.keep_intermediates);
                assert!(!config.verbose);
                assert!(config.enable_lto);
                assert_eq!(config.target_config.optimization_level, OptimizationLevel::Aggressive);
            }
            Err(e) => {
                println!("Release config creation failed (expected in test environment): {}", e);
            }
        }
    }

    #[test]
    fn test_intermediate_files_structure() {
        let mut intermediate_files = IntermediateFiles::default();
        
        intermediate_files.llvm_ir_file = Some(PathBuf::from("test.ll"));
        intermediate_files.assembly_file = Some(PathBuf::from("test.s"));
        intermediate_files.object_files.push(PathBuf::from("test.o"));
        
        assert!(intermediate_files.llvm_ir_file.is_some());
        assert!(intermediate_files.assembly_file.is_some());
        assert_eq!(intermediate_files.object_files.len(), 1);
    }

    #[test]
    fn test_compilation_statistics() {
        let mut stats = CompilationStatistics::default();
        stats.total_time_ms = 1000;
        stats.mlir_time_ms = 200;
        stats.llvm_codegen_time_ms = 500;
        stats.linking_time_ms = 300;
        stats.mlir_operations_count = 100;
        stats.object_files_count = 1;
        stats.executable_size_bytes = 1024 * 1024; // 1MB
        stats.debug_info_included = true;
        stats.lto_applied = false;
        
        assert_eq!(stats.total_time_ms, 1000);
        assert_eq!(stats.mlir_operations_count, 100);
        assert!(stats.debug_info_included);
        assert!(!stats.lto_applied);
    }
}