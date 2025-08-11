// Error detection and classification system for Aether build system

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use regex::Regex;

/// Error handler for build system operations
pub struct ErrorHandler {
    error_patterns: HashMap<ErrorPattern, FixStrategy>,
    error_history: Vec<BuildError>,
    pattern_matchers: Vec<PatternMatcher>,
    error_statistics: ErrorStatistics,
}

impl ErrorHandler {
    /// Create new error handler
    pub fn new() -> Self {
        let mut error_patterns = HashMap::new();
        let mut pattern_matchers = Vec::new();
        
        // Initialize common error patterns and their fixes
        error_patterns.insert(
            ErrorPattern::MissingDependency("tempfile".to_string()),
            FixStrategy::InstallDependency(DependencyInfo {
                name: "tempfile".to_string(),
                version: "3.8".to_string(),
                dev_dependency: true,
            })
        );
        
        error_patterns.insert(
            ErrorPattern::MissingDependency("melior".to_string()),
            FixStrategy::EnableFeatureFlag("mlir".to_string())
        );
        
        error_patterns.insert(
            ErrorPattern::TypeMismatch {
                expected: "MLIRContext".to_string(),
                found: "MockMLIRContext".to_string(),
            },
            FixStrategy::ReplaceImplementation(MockImplementation {
                mock_type: "MockMLIRContext".to_string(),
                real_type: "MLIRContext".to_string(),
                replacement_code: "MLIRContext::new()".to_string(),
            })
        );
        
        // Initialize pattern matchers for advanced error detection
        pattern_matchers.extend(Self::create_dependency_matchers());
        pattern_matchers.extend(Self::create_type_matchers());
        pattern_matchers.extend(Self::create_compilation_matchers());
        pattern_matchers.extend(Self::create_mlir_matchers());
        
        Self {
            error_patterns,
            error_history: Vec::new(),
            pattern_matchers,
            error_statistics: ErrorStatistics::new(),
        }
    }
    
    /// Detect and classify error from compilation output
    pub fn detect_error(&mut self, output: &str) -> Option<BuildError> {
        // Try pattern matchers first for more sophisticated detection
        for matcher in &self.pattern_matchers {
            if let Some(error) = matcher.try_match(output) {
                self.record_error(&error);
                return Some(error);
            }
        }
        
        // Fallback to legacy detection methods
        self.detect_legacy_patterns(output)
    }
    
    /// Legacy pattern detection for backward compatibility
    fn detect_legacy_patterns(&mut self, output: &str) -> Option<BuildError> {
        // Check for missing dependency errors
        if output.contains("could not find `tempfile`") {
            let error = BuildError {
                error_type: ErrorType::MissingDependency,
                message: "Missing tempfile dependency".to_string(),
                location: None,
                suggested_fixes: vec![
                    self.error_patterns.get(&ErrorPattern::MissingDependency("tempfile".to_string()))
                        .cloned()
                        .unwrap_or(FixStrategy::InstallDependency(DependencyInfo {
                            name: "tempfile".to_string(),
                            version: "3.8".to_string(),
                            dev_dependency: true,
                        }))
                ],
                severity: ErrorSeverity::Error,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                context: ErrorContext::new(output),
            };
            self.record_error(&error);
            return Some(error);
        }
        
        // Check for MLIR-related errors
        if output.contains("MockMLIR") || output.contains("mock") {
            let error = BuildError {
                error_type: ErrorType::MockImplementation,
                message: "Mock implementation found, needs replacement with real implementation".to_string(),
                location: self.extract_location_from_output(output),
                suggested_fixes: vec![
                    FixStrategy::ReplaceImplementation(MockImplementation {
                        mock_type: "MockMLIR".to_string(),
                        real_type: "MLIR".to_string(),
                        replacement_code: "real MLIR implementation".to_string(),
                    })
                ],
                severity: ErrorSeverity::Error,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                context: ErrorContext::new(output),
            };
            self.record_error(&error);
            return Some(error);
        }
        
        // Check for type mismatch errors
        if output.contains("type mismatch") || (output.contains("expected") && output.contains("found")) {
            let error = BuildError {
                error_type: ErrorType::TypeMismatch,
                message: "Type mismatch detected".to_string(),
                location: self.extract_location_from_output(output),
                suggested_fixes: vec![
                    FixStrategy::ModifyCode(CodeModification {
                        file_path: PathBuf::from("unknown"),
                        line_number: 0,
                        old_code: "unknown".to_string(),
                        new_code: "unknown".to_string(),
                    })
                ],
                severity: ErrorSeverity::Error,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                context: ErrorContext::new(output),
            };
            self.record_error(&error);
            return Some(error);
        }
        
        None
    }
    
    /// Get suggested fixes for an error
    pub fn get_suggested_fixes(&self, error: &BuildError) -> Vec<FixStrategy> {
        error.suggested_fixes.clone()
    }
    
    /// Get error history
    pub fn get_error_history(&self) -> &[BuildError] {
        &self.error_history
    }
    
    /// Clear error history
    pub fn clear_history(&mut self) {
        self.error_history.clear();
        self.error_statistics = ErrorStatistics::new();
    }
    
    /// Record error in history and update statistics
    fn record_error(&mut self, error: &BuildError) {
        self.error_history.push(error.clone());
        self.error_statistics.record_error(error);
    }
    
    /// Get error statistics
    pub fn get_error_statistics(&self) -> &ErrorStatistics {
        &self.error_statistics
    }
    
    /// Analyze error patterns and trends
    pub fn analyze_error_trends(&self) -> ErrorAnalysis {
        ErrorAnalysis::from_history(&self.error_history)
    }
    
    /// Classify error severity based on context and history
    pub fn classify_error_severity(&self, error: &BuildError) -> ErrorSeverity {
        // Check if this is a recurring error
        let similar_errors = self.error_history.iter()
            .filter(|e| e.error_type == error.error_type)
            .count();
        
        match error.error_type {
            ErrorType::MissingDependency => {
                if similar_errors > 3 {
                    ErrorSeverity::Critical
                } else {
                    ErrorSeverity::Error
                }
            }
            ErrorType::TypeMismatch => ErrorSeverity::Error,
            ErrorType::MockImplementation => ErrorSeverity::Warning,
            ErrorType::CompilationError => ErrorSeverity::Error,
            ErrorType::LinkingError => ErrorSeverity::Critical,
            ErrorType::RuntimeError => ErrorSeverity::Critical,
        }
    }
    
    /// Extract source location from compiler output
    fn extract_location_from_output(&self, output: &str) -> Option<SourceLocation> {
        // Enhanced regex-based parsing for location information
        let location_regex = Regex::new(r"([^:]+):(\d+):(\d+)").unwrap();
        
        for line in output.lines() {
            if let Some(captures) = location_regex.captures(line) {
                if let (Some(file), Some(line_str), Some(col_str)) = 
                    (captures.get(1), captures.get(2), captures.get(3)) {
                    if let (Ok(line_num), Ok(col_num)) = 
                        (line_str.as_str().parse::<usize>(), col_str.as_str().parse::<usize>()) {
                        return Some(SourceLocation {
                            file: PathBuf::from(file.as_str()),
                            line: line_num,
                            column: col_num,
                        });
                    }
                }
            }
        }
        
        // Fallback to simple parsing
        for line in output.lines() {
            if let Some(colon_pos) = line.find(':') {
                if let Some(second_colon) = line[colon_pos + 1..].find(':') {
                    let file_part = &line[..colon_pos];
                    let line_part = &line[colon_pos + 1..colon_pos + 1 + second_colon];
                    
                    if let Ok(line_num) = line_part.parse::<usize>() {
                        return Some(SourceLocation {
                            file: PathBuf::from(file_part),
                            line: line_num,
                            column: 0,
                        });
                    }
                }
            }
        }
        None
    }
    
    /// Create dependency-related pattern matchers
    fn create_dependency_matchers() -> Vec<PatternMatcher> {
        vec![
            PatternMatcher::new(
                r"could not find `([^`]+)` in the list of imported crates",
                ErrorType::MissingDependency,
                |captures| {
                    let crate_name = captures.get(1).unwrap().as_str();
                    BuildError {
                        error_type: ErrorType::MissingDependency,
                        message: format!("Missing dependency: {}", crate_name),
                        location: None,
                        suggested_fixes: vec![
                            FixStrategy::InstallDependency(DependencyInfo {
                                name: crate_name.to_string(),
                                version: "latest".to_string(),
                                dev_dependency: false,
                            })
                        ],
                        severity: ErrorSeverity::Error,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        context: ErrorContext::empty(),
                    }
                }
            ),
            PatternMatcher::new(
                r"no external crate `([^`]+)`",
                ErrorType::MissingDependency,
                |captures| {
                    let crate_name = captures.get(1).unwrap().as_str();
                    BuildError {
                        error_type: ErrorType::MissingDependency,
                        message: format!("External crate not found: {}", crate_name),
                        location: None,
                        suggested_fixes: vec![
                            FixStrategy::InstallDependency(DependencyInfo {
                                name: crate_name.to_string(),
                                version: "latest".to_string(),
                                dev_dependency: false,
                            })
                        ],
                        severity: ErrorSeverity::Error,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        context: ErrorContext::empty(),
                    }
                }
            ),
        ]
    }
    
    /// Create type-related pattern matchers
    fn create_type_matchers() -> Vec<PatternMatcher> {
        vec![
            PatternMatcher::new(
                r"expected `([^`]+)`, found `([^`]+)`",
                ErrorType::TypeMismatch,
                |captures| {
                    let expected = captures.get(1).unwrap().as_str();
                    let found = captures.get(2).unwrap().as_str();
                    BuildError {
                        error_type: ErrorType::TypeMismatch,
                        message: format!("Type mismatch: expected `{}`, found `{}`", expected, found),
                        location: None,
                        suggested_fixes: vec![
                            FixStrategy::ModifyCode(CodeModification {
                                file_path: PathBuf::from("unknown"),
                                line_number: 0,
                                old_code: found.to_string(),
                                new_code: expected.to_string(),
                            })
                        ],
                        severity: ErrorSeverity::Error,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        context: ErrorContext::empty(),
                    }
                }
            ),
            PatternMatcher::new(
                r"cannot infer type for `([^`]+)`",
                ErrorType::TypeMismatch,
                |captures| {
                    let var_name = captures.get(1).unwrap().as_str();
                    BuildError {
                        error_type: ErrorType::TypeMismatch,
                        message: format!("Cannot infer type for variable: {}", var_name),
                        location: None,
                        suggested_fixes: vec![
                            FixStrategy::ModifyCode(CodeModification {
                                file_path: PathBuf::from("unknown"),
                                line_number: 0,
                                old_code: format!("let {}", var_name),
                                new_code: format!("let {}: Type", var_name),
                            })
                        ],
                        severity: ErrorSeverity::Error,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        context: ErrorContext::empty(),
                    }
                }
            ),
        ]
    }
    
    /// Create compilation-related pattern matchers
    fn create_compilation_matchers() -> Vec<PatternMatcher> {
        vec![
            PatternMatcher::new(
                r"unresolved import `([^`]+)`",
                ErrorType::CompilationError,
                |captures| {
                    let import_name = captures.get(1).unwrap().as_str();
                    BuildError {
                        error_type: ErrorType::CompilationError,
                        message: format!("Unresolved import: {}", import_name),
                        location: None,
                        suggested_fixes: vec![
                            FixStrategy::ModifyCode(CodeModification {
                                file_path: PathBuf::from("unknown"),
                                line_number: 0,
                                old_code: format!("use {};", import_name),
                                new_code: format!("// use {}; // TODO: Fix import", import_name),
                            })
                        ],
                        severity: ErrorSeverity::Error,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        context: ErrorContext::empty(),
                    }
                }
            ),
            PatternMatcher::new(
                r"cannot find function `([^`]+)` in this scope",
                ErrorType::CompilationError,
                |captures| {
                    let function_name = captures.get(1).unwrap().as_str();
                    BuildError {
                        error_type: ErrorType::CompilationError,
                        message: format!("Function not found: {}", function_name),
                        location: None,
                        suggested_fixes: vec![],
                        severity: ErrorSeverity::Error,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        context: ErrorContext::empty(),
                    }
                }
            ),
        ]
    }
    
    /// Create MLIR-related pattern matchers
    fn create_mlir_matchers() -> Vec<PatternMatcher> {
        vec![
            PatternMatcher::new(
                r"Mock([A-Z][a-zA-Z]*)",
                ErrorType::MockImplementation,
                |captures| {
                    let mock_type = captures.get(0).unwrap().as_str();
                    let real_type = &mock_type[4..]; // Remove "Mock" prefix
                    BuildError {
                        error_type: ErrorType::MockImplementation,
                        message: format!("Mock implementation detected: {}", mock_type),
                        location: None,
                        suggested_fixes: vec![
                            FixStrategy::ReplaceImplementation(MockImplementation {
                                mock_type: mock_type.to_string(),
                                real_type: real_type.to_string(),
                                replacement_code: format!("{}::new()", real_type),
                            })
                        ],
                        severity: ErrorSeverity::Warning,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        context: ErrorContext::empty(),
                    }
                }
            ),
            PatternMatcher::new(
                r"MLIR feature not enabled",
                ErrorType::CompilationError,
                |_| {
                    BuildError {
                        error_type: ErrorType::CompilationError,
                        message: "MLIR feature not enabled".to_string(),
                        location: None,
                        suggested_fixes: vec![
                            FixStrategy::EnableFeatureFlag("mlir".to_string())
                        ],
                        severity: ErrorSeverity::Error,
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        context: ErrorContext::empty(),
                    }
                }
            ),
        ]
    }
}

/// Build error representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildError {
    pub error_type: ErrorType,
    pub message: String,
    pub location: Option<SourceLocation>,
    pub suggested_fixes: Vec<FixStrategy>,
    pub severity: ErrorSeverity,
    pub timestamp: u64,
    pub context: ErrorContext,
}

/// Source location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: PathBuf,
    pub line: usize,
    pub column: usize,
}

/// Error type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorType {
    MissingDependency,
    TypeMismatch,
    MockImplementation,
    CompilationError,
    LinkingError,
    RuntimeError,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Warning,
    Error,
    Critical,
}

/// Error pattern for matching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorPattern {
    MissingDependency(String),
    TypeMismatch { expected: String, found: String },
    MLIRIntegration(MLIRErrorType),
    LinkingError(LinkingErrorType),
    RuntimeFailure(RuntimeErrorType),
}

/// MLIR-specific error types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MLIRErrorType {
    MockImplementation,
    ContextCreation,
    ModuleCreation,
    OperationError,
}

/// Linking error types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LinkingErrorType {
    MissingLibrary,
    SymbolNotFound,
    ArchitectureMismatch,
}

/// Runtime error types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RuntimeErrorType {
    MemoryAccess,
    NullPointer,
    StackOverflow,
    GPUError,
}

/// Fix strategy for errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FixStrategy {
    InstallDependency(DependencyInfo),
    ModifyCode(CodeModification),
    UpdateConfiguration(ConfigChange),
    EnableFeatureFlag(String),
    ReplaceImplementation(MockImplementation),
}

/// Dependency information for installation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    pub name: String,
    pub version: String,
    pub dev_dependency: bool,
}

/// Code modification specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeModification {
    pub file_path: PathBuf,
    pub line_number: usize,
    pub old_code: String,
    pub new_code: String,
}

/// Configuration change specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigChange {
    pub config_file: PathBuf,
    pub section: String,
    pub key: String,
    pub value: String,
}

/// Mock implementation replacement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockImplementation {
    pub mock_type: String,
    pub real_type: String,
    pub replacement_code: String,
}

/// Pattern matcher for error detection
pub struct PatternMatcher {
    regex: Regex,
    error_type: ErrorType,
    error_builder: Box<dyn Fn(&regex::Captures) -> BuildError + Send + Sync>,
}

impl PatternMatcher {
    pub fn new<F>(pattern: &str, error_type: ErrorType, error_builder: F) -> Self 
    where 
        F: Fn(&regex::Captures) -> BuildError + Send + Sync + 'static,
    {
        Self {
            regex: Regex::new(pattern).expect("Invalid regex pattern"),
            error_type,
            error_builder: Box::new(error_builder),
        }
    }
    
    pub fn try_match(&self, input: &str) -> Option<BuildError> {
        if let Some(captures) = self.regex.captures(input) {
            let mut error = (self.error_builder)(&captures);
            error.context = ErrorContext::new(input);
            Some(error)
        } else {
            None
        }
    }
}

/// Error context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub raw_output: String,
    pub surrounding_lines: Vec<String>,
    pub compilation_stage: Option<String>,
}

impl ErrorContext {
    pub fn new(raw_output: &str) -> Self {
        Self {
            raw_output: raw_output.to_string(),
            surrounding_lines: raw_output.lines().take(5).map(|s| s.to_string()).collect(),
            compilation_stage: None,
        }
    }
    
    pub fn empty() -> Self {
        Self {
            raw_output: String::new(),
            surrounding_lines: Vec::new(),
            compilation_stage: None,
        }
    }
}

/// Error statistics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    pub total_errors: usize,
    pub error_counts_by_type: HashMap<ErrorType, usize>,
    pub most_common_errors: Vec<(ErrorType, usize)>,
    pub error_frequency_over_time: Vec<(u64, usize)>,
}

impl ErrorStatistics {
    pub fn new() -> Self {
        Self {
            total_errors: 0,
            error_counts_by_type: HashMap::new(),
            most_common_errors: Vec::new(),
            error_frequency_over_time: Vec::new(),
        }
    }
    
    pub fn record_error(&mut self, error: &BuildError) {
        self.total_errors += 1;
        *self.error_counts_by_type.entry(error.error_type.clone()).or_insert(0) += 1;
        
        // Update most common errors
        self.most_common_errors = self.error_counts_by_type
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        self.most_common_errors.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Record frequency over time (simplified - could be more sophisticated)
        let time_bucket = error.timestamp / 3600; // Hour buckets
        if let Some(last_entry) = self.error_frequency_over_time.last_mut() {
            if last_entry.0 == time_bucket {
                last_entry.1 += 1;
            } else {
                self.error_frequency_over_time.push((time_bucket, 1));
            }
        } else {
            self.error_frequency_over_time.push((time_bucket, 1));
        }
    }
}

/// Error analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    pub total_errors: usize,
    pub unique_error_types: usize,
    pub most_frequent_error: Option<ErrorType>,
    pub error_trends: Vec<ErrorTrend>,
    pub recurring_patterns: Vec<RecurringPattern>,
}

impl ErrorAnalysis {
    pub fn from_history(history: &[BuildError]) -> Self {
        let mut error_counts: HashMap<ErrorType, usize> = HashMap::new();
        let mut location_counts: HashMap<String, usize> = HashMap::new();
        
        for error in history {
            *error_counts.entry(error.error_type.clone()).or_insert(0) += 1;
            
            if let Some(location) = &error.location {
                let location_key = format!("{}:{}", location.file.display(), location.line);
                *location_counts.entry(location_key).or_insert(0) += 1;
            }
        }
        
        let most_frequent_error = error_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(error_type, _)| error_type.clone());
        
        let recurring_patterns = location_counts
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .map(|(location, count)| RecurringPattern {
                location,
                occurrence_count: count,
                error_types: Vec::new(), // Could be enhanced to track specific error types
            })
            .collect();
        
        Self {
            total_errors: history.len(),
            unique_error_types: error_counts.len(),
            most_frequent_error,
            error_trends: Vec::new(), // Could be enhanced with time-based analysis
            recurring_patterns,
        }
    }
}

/// Error trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTrend {
    pub error_type: ErrorType,
    pub trend_direction: TrendDirection,
    pub change_percentage: f64,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Recurring error pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurringPattern {
    pub location: String,
    pub occurrence_count: usize,
    pub error_types: Vec<ErrorType>,
}