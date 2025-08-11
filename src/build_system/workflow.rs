// End-to-end build workflow orchestration for Aether compiler
// Coordinates all compilation stages with progress reporting and recovery mechanisms

use crate::build_system::{
    BuildSystemManager, BuildConfig, CompilerBinary, Executable,
    ValidationStatus
};
use crate::testing::{ComprehensiveTestRunner, ComprehensiveTestResults, TestConfig};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::fs;
use serde::{Serialize, Deserialize};

/// Main build workflow orchestrator
pub struct BuildWorkflowOrchestrator {
    build_manager: BuildSystemManager,
    config: WorkflowConfig,
    progress_reporter: ProgressReporter,
    state_manager: WorkflowStateManager,
    recovery_manager: RecoveryManager,
}

/// Workflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    pub build_config: BuildConfig,
    pub enable_progress_reporting: bool,
    pub enable_rollback: bool,
    pub max_retry_attempts: usize,
    pub timeout_duration: Duration,
    pub parallel_execution: bool,
    pub verification_enabled: bool,
    pub cleanup_on_failure: bool,
    pub persist_state: bool,
    pub state_file_path: PathBuf,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            build_config: BuildConfig::default(),
            enable_progress_reporting: true,
            enable_rollback: true,
            max_retry_attempts: 3,
            timeout_duration: Duration::from_secs(300), // 5 minutes
            parallel_execution: false, // Sequential by default for reliability
            verification_enabled: true,
            cleanup_on_failure: true,
            persist_state: true,
            state_file_path: PathBuf::from(".aether_build_state.json"),
        }
    }
}

/// Build workflow execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub success: bool,
    pub stages_completed: Vec<WorkflowStage>,
    pub stages_failed: Vec<WorkflowStage>,
    pub total_duration: Duration,
    pub compiler_binary: Option<CompilerBinary>,
    pub compiled_executables: Vec<Executable>,
    pub test_results: Option<ComprehensiveTestResults>,
    pub error_summary: Vec<String>,
    pub recovery_actions: Vec<RecoveryAction>,
}

/// Workflow execution stages
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkflowStage {
    EnvironmentValidation,
    DependencyResolution,
    CompilerCompilation,
    SourceCompilation,
    Testing,
    Verification,
    Cleanup,
}

/// Progress reporting system
pub struct ProgressReporter {
    enabled: bool,
    current_stage: Option<WorkflowStage>,
    stage_progress: HashMap<WorkflowStage, f64>,
    start_time: Instant,
    stage_start_time: Option<Instant>,
}

impl ProgressReporter {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            current_stage: None,
            stage_progress: HashMap::new(),
            start_time: Instant::now(),
            stage_start_time: None,
        }
    }

    pub fn start_stage(&mut self, stage: WorkflowStage) {
        if self.enabled {
            self.current_stage = Some(stage.clone());
            self.stage_start_time = Some(Instant::now());
            println!("🔄 Starting stage: {:?}", stage);
        }
    }

    pub fn update_progress(&mut self, stage: WorkflowStage, progress: f64) {
        if self.enabled {
            self.stage_progress.insert(stage.clone(), progress);
            println!("📊 {:?}: {:.1}%", stage, progress * 100.0);
        }
    }

    pub fn complete_stage(&mut self, stage: WorkflowStage, success: bool) {
        if self.enabled {
            let duration = self.stage_start_time
                .map(|start| start.elapsed())
                .unwrap_or_default();
            
            let status = if success { "✅" } else { "❌" };
            println!("{} Completed stage: {:?} (took {:?})", status, stage, duration);
            
            self.stage_progress.insert(stage, if success { 1.0 } else { 0.0 });
            self.current_stage = None;
            self.stage_start_time = None;
        }
    }

    pub fn report_overall_progress(&self) {
        if self.enabled {
            let total_stages = 7.0; // Number of workflow stages
            let completed_progress: f64 = self.stage_progress.values().sum();
            let overall_progress = completed_progress / total_stages;
            
            println!("🎯 Overall progress: {:.1}% (elapsed: {:?})", 
                overall_progress * 100.0, self.start_time.elapsed());
        }
    }
}
// Workflow state management for persistence and recovery
pub struct WorkflowStateManager {
    state_file: PathBuf,
    current_state: WorkflowState,
    persist_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowState {
    pub current_stage: Option<WorkflowStage>,
    pub completed_stages: Vec<WorkflowStage>,
    pub failed_stages: Vec<WorkflowStage>,
    pub compiler_binary_path: Option<PathBuf>,
    pub compiled_files: Vec<PathBuf>,
    pub last_error: Option<String>,
    pub timestamp: u64,
    pub config_hash: u64,
}

impl WorkflowStateManager {
    pub fn new(state_file: PathBuf, persist_enabled: bool) -> Self {
        let current_state = if persist_enabled && state_file.exists() {
            Self::load_state(&state_file).unwrap_or_default()
        } else {
            WorkflowState::default()
        };

        Self {
            state_file,
            current_state,
            persist_enabled,
        }
    }

    pub fn save_state(&mut self) -> Result<(), WorkflowError> {
        if !self.persist_enabled {
            return Ok(());
        }

        self.current_state.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let state_json = serde_json::to_string_pretty(&self.current_state)
            .map_err(|e| WorkflowError::StatePersistence(format!("Failed to serialize state: {}", e)))?;

        fs::write(&self.state_file, state_json)
            .map_err(|e| WorkflowError::StatePersistence(format!("Failed to write state file: {}", e)))?;

        Ok(())
    }

    pub fn load_state(state_file: &Path) -> Result<WorkflowState, WorkflowError> {
        let state_content = fs::read_to_string(state_file)
            .map_err(|e| WorkflowError::StatePersistence(format!("Failed to read state file: {}", e)))?;

        serde_json::from_str(&state_content)
            .map_err(|e| WorkflowError::StatePersistence(format!("Failed to deserialize state: {}", e)))
    }

    pub fn update_stage(&mut self, stage: WorkflowStage) {
        self.current_state.current_stage = Some(stage);
        let _ = self.save_state();
    }

    pub fn complete_stage(&mut self, stage: WorkflowStage, success: bool) {
        self.current_state.current_stage = None;
        
        if success {
            self.current_state.completed_stages.push(stage);
        } else {
            self.current_state.failed_stages.push(stage);
        }
        
        let _ = self.save_state();
    }

    pub fn can_resume(&self, config_hash: u64) -> bool {
        self.current_state.config_hash == config_hash && 
        !self.current_state.completed_stages.is_empty()
    }

    pub fn clear_state(&mut self) {
        self.current_state = WorkflowState::default();
        if self.persist_enabled {
            let _ = fs::remove_file(&self.state_file);
        }
    }
}

impl Default for WorkflowState {
    fn default() -> Self {
        Self {
            current_stage: None,
            completed_stages: Vec::new(),
            failed_stages: Vec::new(),
            compiler_binary_path: None,
            compiled_files: Vec::new(),
            last_error: None,
            timestamp: 0,
            config_hash: 0,
        }
    }
}

/// Recovery and rollback management
pub struct RecoveryManager {
    enabled: bool,
    snapshots: HashMap<WorkflowStage, RecoverySnapshot>,
    max_retry_attempts: usize,
    retry_counts: HashMap<WorkflowStage, usize>,
}

#[derive(Debug, Clone)]
pub struct RecoverySnapshot {
    pub stage: WorkflowStage,
    pub timestamp: Instant,
    pub file_backups: Vec<FileBackup>,
    pub environment_state: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileBackup {
    pub original_path: PathBuf,
    pub backup_path: PathBuf,
    pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    RetryStage(WorkflowStage),
    RollbackToStage(WorkflowStage),
    RestoreFiles(Vec<PathBuf>),
    ResetEnvironment,
    AbortWorkflow,
}

impl RecoveryManager {
    pub fn new(enabled: bool, max_retry_attempts: usize) -> Self {
        Self {
            enabled,
            snapshots: HashMap::new(),
            max_retry_attempts,
            retry_counts: HashMap::new(),
        }
    }

    pub fn create_snapshot(&mut self, stage: WorkflowStage) -> Result<(), WorkflowError> {
        if !self.enabled {
            return Ok(());
        }

        let snapshot = RecoverySnapshot {
            stage: stage.clone(),
            timestamp: Instant::now(),
            file_backups: Vec::new(), // TODO: Implement file backup
            environment_state: std::env::vars().collect(),
        };

        self.snapshots.insert(stage, snapshot);
        Ok(())
    }

    pub fn can_retry(&mut self, stage: &WorkflowStage) -> bool {
        let retry_count = self.retry_counts.get(stage).unwrap_or(&0);
        *retry_count < self.max_retry_attempts
    }

    pub fn increment_retry(&mut self, stage: WorkflowStage) {
        let count = self.retry_counts.entry(stage).or_insert(0);
        *count += 1;
    }

    pub fn suggest_recovery_action(&self, stage: &WorkflowStage, error: &WorkflowError) -> RecoveryAction {
        match error {
            WorkflowError::EnvironmentValidation(_) => RecoveryAction::ResetEnvironment,
            WorkflowError::CompilationFailed(_) => {
                if self.can_retry_stage(stage) {
                    RecoveryAction::RetryStage(stage.clone())
                } else {
                    RecoveryAction::RollbackToStage(WorkflowStage::EnvironmentValidation)
                }
            }
            WorkflowError::TestingFailed(_) => RecoveryAction::RetryStage(stage.clone()),
            _ => RecoveryAction::AbortWorkflow,
        }
    }

    fn can_retry_stage(&self, stage: &WorkflowStage) -> bool {
        let retry_count = self.retry_counts.get(stage).unwrap_or(&0);
        *retry_count < self.max_retry_attempts
    }

    pub fn rollback_to_stage(&mut self, target_stage: WorkflowStage) -> Result<(), WorkflowError> {
        if !self.enabled {
            return Err(WorkflowError::RollbackDisabled);
        }

        if let Some(_snapshot) = self.snapshots.get(&target_stage) {
            println!("🔄 Rolling back to stage: {:?}", target_stage);
            // TODO: Implement actual file restoration
            Ok(())
        } else {
            Err(WorkflowError::RollbackFailed(
                format!("No snapshot found for stage: {:?}", target_stage)
            ))
        }
    }

    pub fn cleanup_snapshots(&mut self) {
        self.snapshots.clear();
        self.retry_counts.clear();
    }
}

impl BuildWorkflowOrchestrator {
    /// Create new workflow orchestrator
    pub fn new(config: WorkflowConfig) -> Self {
        let build_manager = BuildSystemManager::with_config(config.build_config.clone());
        let progress_reporter = ProgressReporter::new(config.enable_progress_reporting);
        let state_manager = WorkflowStateManager::new(
            config.state_file_path.clone(),
            config.persist_state
        );
        let recovery_manager = RecoveryManager::new(
            config.enable_rollback,
            config.max_retry_attempts
        );

        Self {
            build_manager,
            config,
            progress_reporter,
            state_manager,
            recovery_manager,
        }
    }

    /// Execute the complete build workflow
    pub fn execute_workflow(&mut self, source_files: &[PathBuf]) -> WorkflowResult {
        let start_time = Instant::now();
        let mut result = WorkflowResult {
            success: false,
            stages_completed: Vec::new(),
            stages_failed: Vec::new(),
            total_duration: Duration::default(),
            compiler_binary: None,
            compiled_executables: Vec::new(),
            test_results: None,
            error_summary: Vec::new(),
            recovery_actions: Vec::new(),
        };

        println!("🚀 Starting Aether build workflow...");
        self.progress_reporter.report_overall_progress();

        // Stage 1: Environment Validation
        if let Err(e) = self.execute_stage_with_recovery(WorkflowStage::EnvironmentValidation, &mut result) {
            result.error_summary.push(format!("Environment validation failed: {}", e));
            return self.finalize_result(result, start_time);
        }

        // Stage 2: Dependency Resolution
        if let Err(e) = self.execute_stage_with_recovery(WorkflowStage::DependencyResolution, &mut result) {
            result.error_summary.push(format!("Dependency resolution failed: {}", e));
            return self.finalize_result(result, start_time);
        }

        // Stage 3: Compiler Compilation
        if let Err(e) = self.execute_stage_with_recovery(WorkflowStage::CompilerCompilation, &mut result) {
            result.error_summary.push(format!("Compiler compilation failed: {}", e));
            return self.finalize_result(result, start_time);
        }

        // Stage 4: Source Compilation
        if let Err(e) = self.execute_source_compilation_stage(source_files, &mut result) {
            result.error_summary.push(format!("Source compilation failed: {}", e));
            return self.finalize_result(result, start_time);
        }

        // Stage 5: Testing (if enabled)
        if self.config.verification_enabled {
            if let Err(e) = self.execute_stage_with_recovery(WorkflowStage::Testing, &mut result) {
                result.error_summary.push(format!("Testing failed: {}", e));
                // Testing failure is not fatal - continue to verification
            }
        }

        // Stage 6: Verification
        if self.config.verification_enabled {
            if let Err(e) = self.execute_stage_with_recovery(WorkflowStage::Verification, &mut result) {
                result.error_summary.push(format!("Verification failed: {}", e));
                return self.finalize_result(result, start_time);
            }
        }

        // Stage 7: Cleanup
        if let Err(e) = self.execute_stage_with_recovery(WorkflowStage::Cleanup, &mut result) {
            result.error_summary.push(format!("Cleanup failed: {}", e));
            // Cleanup failure is not fatal
        }

        result.success = result.error_summary.is_empty() || 
                        result.error_summary.iter().all(|e| e.contains("Testing failed") || e.contains("Cleanup failed"));

        self.finalize_result(result, start_time)
    }

    /// Execute a workflow stage with recovery support
    fn execute_stage_with_recovery(
        &mut self,
        stage: WorkflowStage,
        result: &mut WorkflowResult,
    ) -> Result<(), WorkflowError> {
        let mut attempts = 0;
        
        loop {
            attempts += 1;
            self.progress_reporter.start_stage(stage.clone());
            self.state_manager.update_stage(stage.clone());
            
            // Create recovery snapshot
            self.recovery_manager.create_snapshot(stage.clone())?;

            let stage_result = match stage {
                WorkflowStage::EnvironmentValidation => self.execute_environment_validation(),
                WorkflowStage::DependencyResolution => self.execute_dependency_resolution(),
                WorkflowStage::CompilerCompilation => self.execute_compiler_compilation(result),
                WorkflowStage::Testing => self.execute_testing_stage(result),
                WorkflowStage::Verification => self.execute_verification_stage(result),
                WorkflowStage::Cleanup => self.execute_cleanup_stage(),
                _ => Err(WorkflowError::UnsupportedStage(format!("{:?}", stage))),
            };

            match stage_result {
                Ok(()) => {
                    self.progress_reporter.complete_stage(stage.clone(), true);
                    self.state_manager.complete_stage(stage.clone(), true);
                    result.stages_completed.push(stage);
                    return Ok(());
                }
                Err(e) => {
                    self.progress_reporter.complete_stage(stage.clone(), false);
                    
                    if self.recovery_manager.can_retry(&stage) && attempts <= self.config.max_retry_attempts {
                        println!("⚠️  Stage failed, attempting retry {} of {}", attempts, self.config.max_retry_attempts);
                        self.recovery_manager.increment_retry(stage.clone());
                        
                        let recovery_action = self.recovery_manager.suggest_recovery_action(&stage, &e);
                        result.recovery_actions.push(recovery_action.clone());
                        
                        match recovery_action {
                            RecoveryAction::RetryStage(_) => continue,
                            RecoveryAction::RollbackToStage(target) => {
                                self.recovery_manager.rollback_to_stage(target)?;
                                continue;
                            }
                            _ => {
                                self.state_manager.complete_stage(stage.clone(), false);
                                result.stages_failed.push(stage);
                                return Err(e);
                            }
                        }
                    } else {
                        self.state_manager.complete_stage(stage.clone(), false);
                        result.stages_failed.push(stage);
                        return Err(e);
                    }
                }
            }
        }
    }    
/// Execute environment validation stage
    fn execute_environment_validation(&mut self) -> Result<(), WorkflowError> {
        self.progress_reporter.update_progress(WorkflowStage::EnvironmentValidation, 0.1);
        
        let env_status = self.build_manager.validate_environment()
            .map_err(|e| WorkflowError::EnvironmentValidation(e.to_string()))?;

        self.progress_reporter.update_progress(WorkflowStage::EnvironmentValidation, 0.5);

        if !matches!(env_status.overall_status, ValidationStatus::Valid) {
            return Err(WorkflowError::EnvironmentValidation(
                "Environment validation failed".to_string()
            ));
        }

        self.progress_reporter.update_progress(WorkflowStage::EnvironmentValidation, 1.0);
        Ok(())
    }

    /// Execute dependency resolution stage
    fn execute_dependency_resolution(&mut self) -> Result<(), WorkflowError> {
        self.progress_reporter.update_progress(WorkflowStage::DependencyResolution, 0.1);

        // Install missing dependencies
        self.build_manager.install_missing_dependencies()
            .map_err(|e| WorkflowError::DependencyResolution(e.to_string()))?;

        self.progress_reporter.update_progress(WorkflowStage::DependencyResolution, 0.6);

        // Manage feature flags
        self.build_manager.manage_feature_flags()
            .map_err(|e| WorkflowError::DependencyResolution(e.to_string()))?;

        self.progress_reporter.update_progress(WorkflowStage::DependencyResolution, 1.0);
        Ok(())
    }

    /// Execute compiler compilation stage
    fn execute_compiler_compilation(&mut self, result: &mut WorkflowResult) -> Result<(), WorkflowError> {
        self.progress_reporter.update_progress(WorkflowStage::CompilerCompilation, 0.1);

        let compiler_binary = self.build_manager.compile_aether_compiler()
            .map_err(|e| WorkflowError::CompilationFailed(e.to_string()))?;

        self.progress_reporter.update_progress(WorkflowStage::CompilerCompilation, 0.8);

        result.compiler_binary = Some(compiler_binary.clone());
        self.state_manager.current_state.compiler_binary_path = Some(compiler_binary.path);

        self.progress_reporter.update_progress(WorkflowStage::CompilerCompilation, 1.0);
        Ok(())
    }

    /// Execute source compilation stage
    fn execute_source_compilation_stage(
        &mut self,
        source_files: &[PathBuf],
        result: &mut WorkflowResult,
    ) -> Result<(), WorkflowError> {
        self.progress_reporter.start_stage(WorkflowStage::SourceCompilation);
        self.state_manager.update_stage(WorkflowStage::SourceCompilation);

        let total_files = source_files.len() as f64;
        let mut compiled_count = 0.0;

        for source_file in source_files {
            self.progress_reporter.update_progress(
                WorkflowStage::SourceCompilation,
                compiled_count / total_files
            );

            let executable = self.build_manager.compile_aether_source(source_file)
                .map_err(|e| WorkflowError::CompilationFailed(
                    format!("Failed to compile {}: {}", source_file.display(), e)
                ))?;

            result.compiled_executables.push(executable.clone());
            self.state_manager.current_state.compiled_files.push(executable.path);
            compiled_count += 1.0;
        }

        self.progress_reporter.complete_stage(WorkflowStage::SourceCompilation, true);
        self.state_manager.complete_stage(WorkflowStage::SourceCompilation, true);
        result.stages_completed.push(WorkflowStage::SourceCompilation);

        Ok(())
    }

    /// Execute testing stage
    fn execute_testing_stage(&mut self, result: &mut WorkflowResult) -> Result<(), WorkflowError> {
        self.progress_reporter.update_progress(WorkflowStage::Testing, 0.1);

        if result.compiled_executables.is_empty() {
            return Err(WorkflowError::TestingFailed(
                "No compiled executables to test".to_string()
            ));
        }

        // Create comprehensive test runner
        let test_config = TestConfig {
            timeout: self.config.timeout_duration,
            parallel_execution: self.config.parallel_execution,
            verbose: true,
            ..TestConfig::default()
        };

        let mut test_runner = ComprehensiveTestRunner::new(
            test_config,
            BuildSystemManager::with_config(self.config.build_config.clone())
        );

        self.progress_reporter.update_progress(WorkflowStage::Testing, 0.3);

        // Run comprehensive tests
        let test_results = test_runner.run_all_tests();
        result.test_results = Some(test_results.clone());

        self.progress_reporter.update_progress(WorkflowStage::Testing, 0.8);

        // Check if tests passed
        let overall_success = test_results.basic_tests
            .as_ref()
            .map(|bt| bt.success_rate > 0.8)
            .unwrap_or(false);

        if !overall_success {
            return Err(WorkflowError::TestingFailed(
                "Test suite failed with low success rate".to_string()
            ));
        }

        self.progress_reporter.update_progress(WorkflowStage::Testing, 1.0);
        Ok(())
    }

    /// Execute verification stage
    fn execute_verification_stage(&mut self, result: &mut WorkflowResult) -> Result<(), WorkflowError> {
        self.progress_reporter.update_progress(WorkflowStage::Verification, 0.1);

        let mut verification_passed = true;
        let mut verification_errors = Vec::new();

        // Verify each compiled executable
        for (i, executable) in result.compiled_executables.iter().enumerate() {
            let progress = (i as f64 + 0.5) / result.compiled_executables.len() as f64;
            self.progress_reporter.update_progress(WorkflowStage::Verification, progress);

            match self.build_manager.run_verification_tests(&executable.path) {
                Ok(test_results) => {
                    if !test_results.overall_success {
                        verification_passed = false;
                        verification_errors.push(format!(
                            "Verification failed for {}: exit code {}",
                            executable.path.display(),
                            test_results.basic_test.exit_code
                        ));
                    }
                }
                Err(e) => {
                    verification_passed = false;
                    verification_errors.push(format!(
                        "Verification error for {}: {}",
                        executable.path.display(),
                        e
                    ));
                }
            }
        }

        if !verification_passed {
            return Err(WorkflowError::VerificationFailed(
                verification_errors.join("; ")
            ));
        }

        self.progress_reporter.update_progress(WorkflowStage::Verification, 1.0);
        Ok(())
    }

    /// Execute cleanup stage
    fn execute_cleanup_stage(&mut self) -> Result<(), WorkflowError> {
        self.progress_reporter.update_progress(WorkflowStage::Cleanup, 0.1);

        // Clean up temporary files
        if self.config.cleanup_on_failure {
            // TODO: Implement temporary file cleanup
        }

        self.progress_reporter.update_progress(WorkflowStage::Cleanup, 0.5);

        // Clean up recovery snapshots
        self.recovery_manager.cleanup_snapshots();

        self.progress_reporter.update_progress(WorkflowStage::Cleanup, 1.0);
        Ok(())
    }

    /// Finalize workflow result
    fn finalize_result(&mut self, mut result: WorkflowResult, start_time: Instant) -> WorkflowResult {
        result.total_duration = start_time.elapsed();
        
        if result.success {
            println!("✅ Build workflow completed successfully in {:?}", result.total_duration);
            self.state_manager.clear_state();
        } else {
            println!("❌ Build workflow failed after {:?}", result.total_duration);
            println!("Errors: {}", result.error_summary.join(", "));
        }

        self.progress_reporter.report_overall_progress();
        result
    }

    /// Get current workflow configuration
    pub fn config(&self) -> &WorkflowConfig {
        &self.config
    }

    /// Update workflow configuration
    pub fn update_config(&mut self, config: WorkflowConfig) {
        self.config = config;
        self.build_manager.update_config(self.config.build_config.clone());
    }

    /// Resume workflow from saved state
    pub fn resume_workflow(&mut self, source_files: &[PathBuf]) -> Result<WorkflowResult, WorkflowError> {
        let config_hash = self.calculate_config_hash();
        
        if self.state_manager.can_resume(config_hash) {
            println!("🔄 Resuming workflow from saved state...");
            // TODO: Implement actual resume logic
            Ok(self.execute_workflow(source_files))
        } else {
            println!("🆕 Starting fresh workflow...");
            self.state_manager.clear_state();
            Ok(self.execute_workflow(source_files))
        }
    }

    /// Calculate configuration hash for resume validation
    fn calculate_config_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        // Hash relevant config fields
        self.config.build_config.rust_toolchain.version.hash(&mut hasher);
        self.config.build_config.rust_toolchain.features.hash(&mut hasher);
        hasher.finish()
    }
}
/// Workflow-specific errors
#[derive(Debug)]
pub enum WorkflowError {
    EnvironmentValidation(String),
    DependencyResolution(String),
    CompilationFailed(String),
    TestingFailed(String),
    VerificationFailed(String),
    StatePersistence(String),
    RollbackFailed(String),
    RollbackDisabled,
    UnsupportedStage(String),
    ConfigurationError(String),
    TimeoutExceeded,
}

impl std::fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkflowError::EnvironmentValidation(msg) => write!(f, "Environment validation error: {}", msg),
            WorkflowError::DependencyResolution(msg) => write!(f, "Dependency resolution error: {}", msg),
            WorkflowError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            WorkflowError::TestingFailed(msg) => write!(f, "Testing failed: {}", msg),
            WorkflowError::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
            WorkflowError::StatePersistence(msg) => write!(f, "State persistence error: {}", msg),
            WorkflowError::RollbackFailed(msg) => write!(f, "Rollback failed: {}", msg),
            WorkflowError::RollbackDisabled => write!(f, "Rollback is disabled"),
            WorkflowError::UnsupportedStage(stage) => write!(f, "Unsupported workflow stage: {}", stage),
            WorkflowError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            WorkflowError::TimeoutExceeded => write!(f, "Workflow timeout exceeded"),
        }
    }
}

impl std::error::Error for WorkflowError {}

/// Convenience function to create a default workflow orchestrator
pub fn create_default_workflow() -> BuildWorkflowOrchestrator {
    BuildWorkflowOrchestrator::new(WorkflowConfig::default())
}

/// Convenience function to create a workflow with custom build config
pub fn create_workflow_with_build_config(build_config: BuildConfig) -> BuildWorkflowOrchestrator {
    let mut workflow_config = WorkflowConfig::default();
    workflow_config.build_config = build_config;
    BuildWorkflowOrchestrator::new(workflow_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_workflow_config_default() {
        let config = WorkflowConfig::default();
        assert!(config.enable_progress_reporting);
        assert!(config.enable_rollback);
        assert_eq!(config.max_retry_attempts, 3);
        assert!(config.verification_enabled);
    }

    #[test]
    fn test_progress_reporter() {
        let mut reporter = ProgressReporter::new(true);
        reporter.start_stage(WorkflowStage::EnvironmentValidation);
        reporter.update_progress(WorkflowStage::EnvironmentValidation, 0.5);
        reporter.complete_stage(WorkflowStage::EnvironmentValidation, true);
        
        assert_eq!(reporter.stage_progress.get(&WorkflowStage::EnvironmentValidation), Some(&1.0));
    }

    #[test]
    fn test_workflow_state_default() {
        let state = WorkflowState::default();
        assert!(state.current_stage.is_none());
        assert!(state.completed_stages.is_empty());
        assert!(state.failed_stages.is_empty());
    }

    #[test]
    fn test_recovery_manager() {
        let mut manager = RecoveryManager::new(true, 3);
        assert!(manager.can_retry(&WorkflowStage::CompilerCompilation));
        
        manager.increment_retry(WorkflowStage::CompilerCompilation);
        manager.increment_retry(WorkflowStage::CompilerCompilation);
        manager.increment_retry(WorkflowStage::CompilerCompilation);
        
        assert!(!manager.can_retry(&WorkflowStage::CompilerCompilation));
    }

    #[test]
    fn test_workflow_orchestrator_creation() {
        let config = WorkflowConfig::default();
        let orchestrator = BuildWorkflowOrchestrator::new(config);
        assert!(orchestrator.config().enable_progress_reporting);
    }
}