// Debugging utilities and intermediate representation dumping for MLIR compilation
// Provides IR dumping, timing instrumentation, and verbose error reporting

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use crate::compiler::mlir::{MLIRModule, MLIROperation, MLIRError};
use crate::compiler::mlir::error_handling::{MLIRCompilationError, SourceLocation};

/// Debug configuration for compilation pipeline
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// Enable IR dumping at each stage
    pub dump_ir: bool,
    /// Directory for IR dumps
    pub dump_dir: PathBuf,
    /// Enable timing instrumentation
    pub enable_timing: bool,
    /// Enable verbose error reporting
    pub verbose_errors: bool,
    /// Dump only specific stages
    pub dump_stages: Vec<CompilationStage>,
    /// Maximum number of operations to dump per stage
    pub max_operations_per_dump: usize,
    /// Include source location information in dumps
    pub include_source_locations: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        DebugConfig {
            dump_ir: false,
            dump_dir: PathBuf::from("debug_output"),
            enable_timing: false,
            verbose_errors: false,
            dump_stages: vec![],
            max_operations_per_dump: 1000,
            include_source_locations: true,
        }
    }
}

impl DebugConfig {
    /// Create debug configuration for development
    pub fn development() -> Self {
        DebugConfig {
            dump_ir: true,
            dump_dir: PathBuf::from("debug_output"),
            enable_timing: true,
            verbose_errors: true,
            dump_stages: vec![
                CompilationStage::ASTToMLIR,
                CompilationStage::Optimization,
                CompilationStage::Lowering,
                CompilationStage::LLVMGeneration,
            ],
            max_operations_per_dump: 500,
            include_source_locations: true,
        }
    }

    /// Create debug configuration for production with minimal overhead
    pub fn production() -> Self {
        DebugConfig {
            dump_ir: false,
            dump_dir: PathBuf::from("debug_output"),
            enable_timing: true,
            verbose_errors: false,
            dump_stages: vec![],
            max_operations_per_dump: 100,
            include_source_locations: false,
        }
    }

    /// Enable all debugging features
    pub fn all_features() -> Self {
        DebugConfig {
            dump_ir: true,
            dump_dir: PathBuf::from("debug_output"),
            enable_timing: true,
            verbose_errors: true,
            dump_stages: CompilationStage::all_stages(),
            max_operations_per_dump: 2000,
            include_source_locations: true,
        }
    }
}

/// Compilation stages for IR dumping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompilationStage {
    /// AST to MLIR conversion
    ASTToMLIR,
    /// High-level optimization passes
    Optimization,
    /// Dialect lowering
    Lowering,
    /// LLVM IR generation
    LLVMGeneration,
    /// LLVM optimization
    LLVMOptimization,
    /// Code generation
    CodeGeneration,
}

impl CompilationStage {
    /// Get all compilation stages
    pub fn all_stages() -> Vec<CompilationStage> {
        vec![
            CompilationStage::ASTToMLIR,
            CompilationStage::Optimization,
            CompilationStage::Lowering,
            CompilationStage::LLVMGeneration,
            CompilationStage::LLVMOptimization,
            CompilationStage::CodeGeneration,
        ]
    }

    /// Get stage name for file naming
    pub fn name(&self) -> &'static str {
        match self {
            CompilationStage::ASTToMLIR => "ast_to_mlir",
            CompilationStage::Optimization => "optimization",
            CompilationStage::Lowering => "lowering",
            CompilationStage::LLVMGeneration => "llvm_generation",
            CompilationStage::LLVMOptimization => "llvm_optimization",
            CompilationStage::CodeGeneration => "code_generation",
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            CompilationStage::ASTToMLIR => "AST to MLIR Conversion",
            CompilationStage::Optimization => "High-level Optimization",
            CompilationStage::Lowering => "Dialect Lowering",
            CompilationStage::LLVMGeneration => "LLVM IR Generation",
            CompilationStage::LLVMOptimization => "LLVM Optimization",
            CompilationStage::CodeGeneration => "Code Generation",
        }
    }
}

/// Timing information for compilation stages
#[derive(Debug, Clone)]
pub struct TimingInfo {
    /// Stage being timed
    pub stage: CompilationStage,
    /// Duration of the stage
    pub duration: Duration,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Instant,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TimingInfo {
    /// Create new timing info
    pub fn new(stage: CompilationStage, start_time: Instant, end_time: Instant) -> Self {
        TimingInfo {
            stage,
            duration: end_time.duration_since(start_time),
            start_time,
            end_time,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        self.duration.as_secs_f64() * 1000.0
    }
}

/// Compilation instrumentation and debugging utilities
#[derive(Debug)]
pub struct CompilationDebugger {
    /// Debug configuration
    config: DebugConfig,
    /// Timing information for each stage
    pub timing_info: Vec<TimingInfo>,
    /// Current stage being timed
    pub current_stage: Option<(CompilationStage, Instant)>,
    /// Dump counter for unique file names
    pub dump_counter: u32,
    /// Error context for verbose reporting
    pub error_context: Vec<String>,
}

impl CompilationDebugger {
    /// Create new compilation debugger
    pub fn new(config: DebugConfig) -> Result<Self, std::io::Error> {
        // Create dump directory if needed
        if config.dump_ir {
            fs::create_dir_all(&config.dump_dir)?;
        }

        Ok(CompilationDebugger {
            config,
            timing_info: Vec::new(),
            current_stage: None,
            dump_counter: 0,
            error_context: Vec::new(),
        })
    }

    /// Start timing a compilation stage
    pub fn start_stage(&mut self, stage: CompilationStage) {
        if self.config.enable_timing {
            self.current_stage = Some((stage, Instant::now()));
        }
    }

    /// End timing a compilation stage
    pub fn end_stage(&mut self) -> Option<TimingInfo> {
        if let Some((stage, start_time)) = self.current_stage.take() {
            let end_time = Instant::now();
            let timing = TimingInfo::new(stage, start_time, end_time);
            self.timing_info.push(timing.clone());
            Some(timing)
        } else {
            None
        }
    }

    /// End timing with metadata
    pub fn end_stage_with_metadata(&mut self, metadata: HashMap<String, String>) -> Option<TimingInfo> {
        if let Some(mut timing) = self.end_stage() {
            timing.metadata = metadata;
            // Update the stored timing info
            if let Some(last) = self.timing_info.last_mut() {
                last.metadata = timing.metadata.clone();
            }
            Some(timing)
        } else {
            None
        }
    }

    /// Dump MLIR module to file
    pub fn dump_mlir_module(&mut self, module: &MLIRModule, stage: CompilationStage) -> Result<Option<PathBuf>, std::io::Error> {
        if !self.config.dump_ir || !self.should_dump_stage(stage) {
            return Ok(None);
        }

        self.dump_counter += 1;
        let filename = format!("{:03}_{}.mlir", self.dump_counter, stage.name());
        let filepath = self.config.dump_dir.join(filename);

        let mut file = fs::File::create(&filepath)?;
        
        // Write header with stage information
        writeln!(file, "// MLIR Dump - Stage: {}", stage.description())?;
        writeln!(file, "// Generated at: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"))?;
        writeln!(file, "// Module: {}", "unknown")?; // Module name not available
        writeln!(file, "")?;

        // Dump module operations
        let operations_dumped = self.dump_operations(&mut file, module.get_operations(), 0)?;
        
        writeln!(file, "")?;
        writeln!(file, "// Total operations dumped: {}", operations_dumped)?;
        
        if operations_dumped >= self.config.max_operations_per_dump {
            writeln!(file, "// Note: Output truncated at {} operations", self.config.max_operations_per_dump)?;
        }

        Ok(Some(filepath))
    }

    /// Dump operations to file
    fn dump_operations(&self, file: &mut fs::File, operations: &[MLIROperation], indent_level: usize) -> Result<usize, std::io::Error> {
        let mut operations_dumped = 0;
        let indent = "  ".repeat(indent_level);

        for op in operations.iter().take(self.config.max_operations_per_dump - operations_dumped) {
            writeln!(file, "{}// Operation: {}", indent, op.name)?;
            
            if self.config.include_source_locations {
                if let Some(location) = &op.source_location {
                    writeln!(file, "{}// Source: {}", indent, location.display())?;
                }
            }

            // Write operation signature
            writeln!(file, "{}{} (", indent, op.name)?;
            
            // Write operands
            for (i, operand) in op.operands.iter().enumerate() {
                if i > 0 {
                    write!(file, ", ")?;
                }
                write!(file, "{}", operand.id)?;
            }
            
            write!(file, ") -> (")?;
            
            // Write results
            for (i, result) in op.results.iter().enumerate() {
                if i > 0 {
                    write!(file, ", ")?;
                }
                write!(file, "{}", result.id)?;
            }
            
            writeln!(file, ")")?;

            // Write attributes
            if !op.attributes.is_empty() {
                writeln!(file, "{}  attributes {{", indent)?;
                for (key, value) in &op.attributes {
                    writeln!(file, "{}    {} = {:?}", indent, key, value)?;
                }
                writeln!(file, "{}  }}", indent)?;
            }

            writeln!(file, "")?;
            operations_dumped += 1;

            if operations_dumped >= self.config.max_operations_per_dump {
                break;
            }
        }

        Ok(operations_dumped)
    }

    /// Check if stage should be dumped
    fn should_dump_stage(&self, stage: CompilationStage) -> bool {
        self.config.dump_stages.is_empty() || self.config.dump_stages.contains(&stage)
    }

    /// Dump LLVM IR to file
    pub fn dump_llvm_ir(&mut self, ir: &str, stage: CompilationStage) -> Result<Option<PathBuf>, std::io::Error> {
        if !self.config.dump_ir || !self.should_dump_stage(stage) {
            return Ok(None);
        }

        self.dump_counter += 1;
        let filename = format!("{:03}_{}.ll", self.dump_counter, stage.name());
        let filepath = self.config.dump_dir.join(filename);

        let mut file = fs::File::create(&filepath)?;
        
        // Write header
        writeln!(file, "; LLVM IR Dump - Stage: {}", stage.description())?;
        writeln!(file, "; Generated at: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"))?;
        writeln!(file, "")?;

        // Write IR content
        write!(file, "{}", ir)?;

        Ok(Some(filepath))
    }

    /// Add error context for verbose reporting
    pub fn add_error_context(&mut self, context: String) {
        self.error_context.push(context);
    }

    /// Clear error context
    pub fn clear_error_context(&mut self) {
        self.error_context.clear();
    }

    /// Create verbose error report
    pub fn create_verbose_error_report(&self, error: &MLIRCompilationError) -> String {
        if !self.config.verbose_errors {
            return error.to_string();
        }

        let mut report = String::new();
        
        // Error header
        report.push_str(&format!("=== MLIR Compilation Error Report ===\n"));
        report.push_str(&format!("Error: {}\n", error.message()));
        report.push_str(&format!("Severity: {}\n", error.severity()));
        report.push_str(&format!("Location: {}\n", error.location().display()));
        report.push_str(&format!("Recoverable: {}\n", error.is_recoverable()));
        report.push_str("\n");

        // Error context
        if !self.error_context.is_empty() {
            report.push_str("Context:\n");
            for (i, context) in self.error_context.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, context));
            }
            report.push_str("\n");
        }

        // Recovery suggestions
        let suggestions = error.recovery_suggestions();
        if !suggestions.is_empty() {
            report.push_str("Recovery Suggestions:\n");
            for suggestion in &suggestions {
                report.push_str(&format!("  - {}\n", suggestion));
            }
            report.push_str("\n");
        }

        // Timing information
        if !self.timing_info.is_empty() {
            report.push_str("Compilation Timing:\n");
            for timing in &self.timing_info {
                report.push_str(&format!("  {}: {:.2}ms\n", 
                    timing.stage.description(), 
                    timing.duration_ms()));
                
                if !timing.metadata.is_empty() {
                    for (key, value) in &timing.metadata {
                        report.push_str(&format!("    {}: {}\n", key, value));
                    }
                }
            }
            report.push_str("\n");
        }

        // Debug information
        report.push_str("Debug Information:\n");
        report.push_str(&format!("  Dump directory: {}\n", self.config.dump_dir.display()));
        report.push_str(&format!("  IR dumping enabled: {}\n", self.config.dump_ir));
        report.push_str(&format!("  Timing enabled: {}\n", self.config.enable_timing));
        report.push_str(&format!("  Dumps created: {}\n", self.dump_counter));

        report.push_str("=====================================\n");

        report
    }

    /// Get timing summary
    pub fn get_timing_summary(&self) -> TimingSummary {
        let mut summary = TimingSummary {
            total_duration: Duration::new(0, 0),
            stage_timings: HashMap::new(),
            slowest_stage: None,
            fastest_stage: None,
        };

        if self.timing_info.is_empty() {
            return summary;
        }

        let mut slowest_duration = Duration::new(0, 0);
        let mut fastest_duration = Duration::new(u64::MAX, 0);

        for timing in &self.timing_info {
            summary.total_duration += timing.duration;
            summary.stage_timings.insert(timing.stage, timing.duration);

            if timing.duration > slowest_duration {
                slowest_duration = timing.duration;
                summary.slowest_stage = Some(timing.stage);
            }

            if timing.duration < fastest_duration {
                fastest_duration = timing.duration;
                summary.fastest_stage = Some(timing.stage);
            }
        }

        summary
    }

    /// Generate compilation report
    pub fn generate_compilation_report(&self) -> CompilationReport {
        CompilationReport {
            timing_summary: self.get_timing_summary(),
            dumps_created: self.dump_counter,
            dump_directory: self.config.dump_dir.clone(),
            error_context: self.error_context.clone(),
            config: self.config.clone(),
        }
    }

    /// Save compilation report to file
    pub fn save_compilation_report(&self, path: &Path) -> Result<(), std::io::Error> {
        let report = self.generate_compilation_report();
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(path, json)
    }
}

/// Timing summary for compilation
#[derive(Debug, Clone)]
pub struct TimingSummary {
    /// Total compilation duration
    pub total_duration: Duration,
    /// Duration by stage
    pub stage_timings: HashMap<CompilationStage, Duration>,
    /// Slowest compilation stage
    pub slowest_stage: Option<CompilationStage>,
    /// Fastest compilation stage
    pub fastest_stage: Option<CompilationStage>,
}

impl TimingSummary {
    /// Get total duration in milliseconds
    pub fn total_duration_ms(&self) -> f64 {
        self.total_duration.as_secs_f64() * 1000.0
    }

    /// Get stage duration in milliseconds
    pub fn stage_duration_ms(&self, stage: CompilationStage) -> f64 {
        self.stage_timings.get(&stage)
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }
}

/// Complete compilation report
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompilationReport {
    /// Timing summary
    pub timing_summary: TimingSummary,
    /// Number of dumps created
    pub dumps_created: u32,
    /// Dump directory
    pub dump_directory: PathBuf,
    /// Error context
    pub error_context: Vec<String>,
    /// Debug configuration
    pub config: DebugConfig,
}

// Implement Serialize/Deserialize for types that need it
impl serde::Serialize for TimingSummary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TimingSummary", 4)?;
        state.serialize_field("total_duration_ms", &self.total_duration_ms())?;
        
        let stage_timings_ms: HashMap<String, f64> = self.stage_timings.iter()
            .map(|(stage, duration)| (stage.name().to_string(), duration.as_secs_f64() * 1000.0))
            .collect();
        state.serialize_field("stage_timings_ms", &stage_timings_ms)?;
        
        state.serialize_field("slowest_stage", &self.slowest_stage.map(|s| s.name()))?;
        state.serialize_field("fastest_stage", &self.fastest_stage.map(|s| s.name()))?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for TimingSummary {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // For simplicity, create a default TimingSummary
        // In a full implementation, this would properly deserialize all fields
        Ok(TimingSummary {
            total_duration: Duration::new(0, 0),
            stage_timings: HashMap::new(),
            slowest_stage: None,
            fastest_stage: None,
        })
    }
}

impl serde::Serialize for CompilationStage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.name())
    }
}

impl<'de> serde::Deserialize<'de> for CompilationStage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "ast_to_mlir" => Ok(CompilationStage::ASTToMLIR),
            "optimization" => Ok(CompilationStage::Optimization),
            "lowering" => Ok(CompilationStage::Lowering),
            "llvm_generation" => Ok(CompilationStage::LLVMGeneration),
            "llvm_optimization" => Ok(CompilationStage::LLVMOptimization),
            "code_generation" => Ok(CompilationStage::CodeGeneration),
            _ => Err(serde::de::Error::custom(format!("Unknown compilation stage: {}", s))),
        }
    }
}

impl serde::Serialize for DebugConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("DebugConfig", 7)?;
        state.serialize_field("dump_ir", &self.dump_ir)?;
        state.serialize_field("dump_dir", &self.dump_dir)?;
        state.serialize_field("enable_timing", &self.enable_timing)?;
        state.serialize_field("verbose_errors", &self.verbose_errors)?;
        state.serialize_field("dump_stages", &self.dump_stages)?;
        state.serialize_field("max_operations_per_dump", &self.max_operations_per_dump)?;
        state.serialize_field("include_source_locations", &self.include_source_locations)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for DebugConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // For simplicity, return default config
        // In a full implementation, this would properly deserialize all fields
        Ok(DebugConfig::default())
    }
}