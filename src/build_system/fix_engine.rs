// Automatic fix engine for common compilation issues

use crate::build_system::{BuildError, FixStrategy, DependencyInfo, CodeModification, ConfigChange, MockImplementation, ErrorType};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

/// Extension trait for Command to add timeout functionality
trait CommandExt {
    fn timeout(&mut self, duration: Duration) -> &mut Self;
}

impl CommandExt for Command {
    fn timeout(&mut self, _duration: Duration) -> &mut Self {
        // For now, we'll implement a basic version without actual timeout
        // In a real implementation, you'd use a library like `tokio` or `async-std`
        // or implement platform-specific timeout handling
        self
    }
}

/// Automatic fix engine for build errors
pub struct AutoFixEngine {
    fix_database: FixDatabase,
    code_modifier: CodeModifier,
    config_manager: ConfigManager,
    fix_history: Vec<FixApplicationResult>,
    dry_run_mode: bool,
}

impl AutoFixEngine {
    /// Create new auto fix engine
    pub fn new() -> Self {
        Self {
            fix_database: FixDatabase::new(),
            code_modifier: CodeModifier::new(),
            config_manager: ConfigManager::new(),
            fix_history: Vec::new(),
            dry_run_mode: false,
        }
    }
    
    /// Create auto fix engine in dry run mode (won't actually apply fixes)
    pub fn new_dry_run() -> Self {
        let mut engine = Self::new();
        engine.dry_run_mode = true;
        engine
    }
    
    /// Analyze error and suggest fix strategy
    pub fn analyze_error(&self, error: &BuildError) -> Option<FixStrategy> {
        self.fix_database.get_fix_for_error(error)
    }
    
    /// Analyze multiple errors and suggest fix strategies
    pub fn analyze_multiple_errors(&self, errors: &[BuildError]) -> Vec<(BuildError, Option<FixStrategy>)> {
        errors.iter()
            .map(|error| (error.clone(), self.analyze_error(error)))
            .collect()
    }
    
    /// Apply fix strategy with detailed result tracking
    pub fn apply_fix(&mut self, fix: &FixStrategy) -> FixApplicationResult {
        let mut result = FixApplicationResult::new(fix.clone());
        
        if self.dry_run_mode {
            result = result.with_success(vec![]);
            self.fix_history.push(result.clone());
            return result;
        }
        
        let application_result = match fix {
            FixStrategy::InstallDependency(dep_info) => {
                self.install_dependency(dep_info)
                    .map(|_| vec![std::path::PathBuf::from("Cargo.toml")])
            }
            FixStrategy::ModifyCode(code_mod) => {
                self.code_modifier.apply_modification(code_mod)
                    .map(|_| vec![code_mod.file_path.clone()])
            }
            FixStrategy::UpdateConfiguration(config_change) => {
                self.config_manager.apply_change(config_change)
                    .map(|_| vec![config_change.config_file.clone()])
            }
            FixStrategy::EnableFeatureFlag(flag) => {
                self.enable_feature_flag(flag)
                    .map(|_| vec![std::path::PathBuf::from("Cargo.toml")])
            }
            FixStrategy::ReplaceImplementation(mock_impl) => {
                self.replace_mock_implementation(mock_impl)
                    .map(|_| self.get_modified_files_for_mock_replacement(mock_impl))
            }
        };
        
        match application_result {
            Ok(files_modified) => {
                result = result.with_success(files_modified);
            }
            Err(error) => {
                result = result.with_error(error);
            }
        }
        
        self.fix_history.push(result.clone());
        result
    }
    
    /// Apply multiple fixes in sequence
    pub fn apply_multiple_fixes(&mut self, fixes: &[FixStrategy]) -> Vec<FixApplicationResult> {
        fixes.iter()
            .map(|fix| self.apply_fix(fix))
            .collect()
    }
    
    /// Verify that fix resolved the original error
    pub fn verify_fix(&self, original_error: &BuildError) -> Result<bool, FixError> {
        self.verify_fix_with_timeout(original_error, std::time::Duration::from_secs(30))
    }
    
    /// Verify fix with custom timeout
    pub fn verify_fix_with_timeout(&self, original_error: &BuildError, timeout: std::time::Duration) -> Result<bool, FixError> {
        if self.dry_run_mode {
            return Ok(true); // Assume success in dry run mode
        }
        
        // Run a quick compilation check to see if the error is resolved
        let output = Command::new("cargo")
            .args(&["check", "--quiet"])
            .timeout(timeout)
            .output()
            .map_err(|e| FixError::VerificationFailed(format!("Failed to run cargo check: {}", e)))?;
        
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        // Check if the original error message is still present
        let error_resolved = !stderr.contains(&original_error.message);
        
        // Also check for related error patterns
        let related_patterns_resolved = match original_error.error_type {
            ErrorType::MissingDependency => {
                !stderr.contains("could not find") || !stderr.contains("in the list of imported crates")
            }
            ErrorType::TypeMismatch => {
                !stderr.contains("type mismatch") && !stderr.contains("expected")
            }
            ErrorType::MockImplementation => {
                !stderr.contains("Mock") && !stderr.contains("mock")
            }
            _ => true,
        };
        
        Ok(error_resolved && related_patterns_resolved)
    }
    
    /// Get fix application history
    pub fn get_fix_history(&self) -> &[FixApplicationResult] {
        &self.fix_history
    }
    
    /// Clear fix history
    pub fn clear_fix_history(&mut self) {
        self.fix_history.clear();
    }
    
    /// Get fix database statistics
    pub fn get_fix_database_statistics(&self) -> FixDatabaseStatistics {
        self.fix_database.get_statistics()
    }
    
    /// Add custom fix to the database
    pub fn add_custom_dependency_fix(&mut self, name: String, info: DependencyInfo) {
        self.fix_database.add_dependency_fix(name, info);
    }
    
    /// Add custom type mismatch fix to the database
    pub fn add_custom_type_fix(&mut self, pattern: String, modification: CodeModification) {
        self.fix_database.add_type_mismatch_fix(pattern, modification);
    }
    
    /// Check if engine is in dry run mode
    pub fn is_dry_run(&self) -> bool {
        self.dry_run_mode
    }
    
    /// Set dry run mode
    pub fn set_dry_run(&mut self, dry_run: bool) {
        self.dry_run_mode = dry_run;
    }
    
    /// Get files that would be modified by a mock replacement
    fn get_modified_files_for_mock_replacement(&self, mock_impl: &MockImplementation) -> Vec<std::path::PathBuf> {
        let mut files = Vec::new();
        
        // Common files that might contain mock implementations
        let potential_files = [
            "src/compiler/mlir/mod.rs",
            "src/compiler/mlir/context.rs",
            "src/compiler/mlir/module.rs",
            "src/compiler/mod.rs",
        ];
        
        for file_path in &potential_files {
            let path = std::path::PathBuf::from(file_path);
            if path.exists() {
                if let Ok(content) = fs::read_to_string(&path) {
                    if content.contains(&mock_impl.mock_type) {
                        files.push(path);
                    }
                }
            }
        }
        
        files
    }
    
    /// Install missing dependency
    fn install_dependency(&self, dep_info: &DependencyInfo) -> Result<(), FixError> {
        // Read current Cargo.toml
        let cargo_toml_path = Path::new("Cargo.toml");
        let mut cargo_content = fs::read_to_string(cargo_toml_path)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to read Cargo.toml: {}", e)))?;
        
        // Add dependency to appropriate section
        let section = if dep_info.dev_dependency {
            "[dev-dependencies]"
        } else {
            "[dependencies]"
        };
        
        let dependency_line = format!("{} = \"{}\"", dep_info.name, dep_info.version);
        
        if let Some(section_pos) = cargo_content.find(section) {
            // Find the end of the section
            let section_start = section_pos + section.len();
            let mut insertion_point = section_start;
            
            // Find a good place to insert the dependency
            for line in cargo_content[section_start..].lines() {
                if line.trim().is_empty() || line.starts_with('[') {
                    break;
                }
                insertion_point += line.len() + 1; // +1 for newline
            }
            
            cargo_content.insert_str(insertion_point, &format!("\n{}", dependency_line));
        } else {
            // Add the section if it doesn't exist
            cargo_content.push_str(&format!("\n\n{}\n{}", section, dependency_line));
        }
        
        // Write back to Cargo.toml
        fs::write(cargo_toml_path, cargo_content)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to write Cargo.toml: {}", e)))?;
        
        Ok(())
    }
    
    /// Enable feature flag in Cargo.toml
    fn enable_feature_flag(&self, flag: &str) -> Result<(), FixError> {
        let cargo_toml_path = Path::new("Cargo.toml");
        let mut cargo_content = fs::read_to_string(cargo_toml_path)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to read Cargo.toml: {}", e)))?;
        
        // Look for [features] section
        if let Some(features_pos) = cargo_content.find("[features]") {
            // Check if default features line exists
            if let Some(default_pos) = cargo_content[features_pos..].find("default = [") {
                let absolute_default_pos = features_pos + default_pos;
                let closing_bracket = cargo_content[absolute_default_pos..]
                    .find(']')
                    .ok_or_else(|| FixError::ParseError("Could not find closing bracket for default features".to_string()))?;
                
                let insertion_point = absolute_default_pos + closing_bracket;
                
                // Check if the feature is already there
                let default_section = &cargo_content[absolute_default_pos..insertion_point];
                if !default_section.contains(&format!("\"{}\"", flag)) {
                    // Insert the feature flag
                    let insert_text = if default_section.contains("\"") {
                        format!(", \"{}\"", flag)
                    } else {
                        format!("\"{}\"", flag)
                    };
                    cargo_content.insert_str(insertion_point, &insert_text);
                }
            } else {
                // Add default features line
                let features_end = features_pos + "[features]".len();
                cargo_content.insert_str(features_end, &format!("\ndefault = [\"{}\"]", flag));
            }
        } else {
            // Add features section
            cargo_content.push_str(&format!("\n\n[features]\ndefault = [\"{}\"]", flag));
        }
        
        fs::write(cargo_toml_path, cargo_content)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to write Cargo.toml: {}", e)))?;
        
        Ok(())
    }
    
    /// Replace mock implementation with real implementation
    fn replace_mock_implementation(&self, _mock_impl: &MockImplementation) -> Result<(), FixError> {
        // This is a complex operation that would need to scan source files
        // For now, we'll implement a basic version that removes mock exports
        
        // Remove mock exports from MLIR module
        let mlir_mod_path = Path::new("src/compiler/mlir/mod.rs");
        if mlir_mod_path.exists() {
            let mut content = fs::read_to_string(mlir_mod_path)
                .map_err(|e| FixError::FileOperationFailed(format!("Failed to read MLIR mod.rs: {}", e)))?;
            
            // Remove mock exports
            content = content.replace("pub use test_utils::{MockMLIRContext, MockMLIRModule, MockOperation, TestMLIRPipeline};", "");
            
            fs::write(mlir_mod_path, content)
                .map_err(|e| FixError::FileOperationFailed(format!("Failed to write MLIR mod.rs: {}", e)))?;
        }
        
        Ok(())
    }
}

/// Fix database containing known fixes for common errors
pub struct FixDatabase {
    dependency_fixes: std::collections::HashMap<String, DependencyInfo>,
    type_mismatch_fixes: std::collections::HashMap<String, CodeModification>,
    mlir_fixes: std::collections::HashMap<String, FixStrategy>,
    feature_flag_fixes: std::collections::HashMap<String, String>,
    config_fixes: std::collections::HashMap<String, ConfigChange>,
}

impl FixDatabase {
    pub fn new() -> Self {
        let mut dependency_fixes = std::collections::HashMap::new();
        let mut type_mismatch_fixes = std::collections::HashMap::new();
        let mut mlir_fixes = std::collections::HashMap::new();
        let mut feature_flag_fixes = std::collections::HashMap::new();
        let mut config_fixes = std::collections::HashMap::new();
        
        // Common dependency fixes
        dependency_fixes.insert("tempfile".to_string(), DependencyInfo {
            name: "tempfile".to_string(),
            version: "3.8".to_string(),
            dev_dependency: true,
        });
        
        dependency_fixes.insert("melior".to_string(), DependencyInfo {
            name: "melior".to_string(),
            version: "0.19".to_string(),
            dev_dependency: false,
        });
        
        dependency_fixes.insert("mlir-sys".to_string(), DependencyInfo {
            name: "mlir-sys".to_string(),
            version: "0.3".to_string(),
            dev_dependency: false,
        });
        
        dependency_fixes.insert("serde".to_string(), DependencyInfo {
            name: "serde".to_string(),
            version: "1.0".to_string(),
            dev_dependency: false,
        });
        
        dependency_fixes.insert("regex".to_string(), DependencyInfo {
            name: "regex".to_string(),
            version: "1.10".to_string(),
            dev_dependency: false,
        });
        
        // Common type mismatch fixes
        type_mismatch_fixes.insert("MockMLIRContext".to_string(), CodeModification {
            file_path: std::path::PathBuf::from("src/compiler/mlir/mod.rs"),
            line_number: 0,
            old_code: "MockMLIRContext".to_string(),
            new_code: "MLIRContext".to_string(),
        });
        
        type_mismatch_fixes.insert("MockMLIRModule".to_string(), CodeModification {
            file_path: std::path::PathBuf::from("src/compiler/mlir/mod.rs"),
            line_number: 0,
            old_code: "MockMLIRModule".to_string(),
            new_code: "MLIRModule".to_string(),
        });
        
        // MLIR-specific fixes
        mlir_fixes.insert("mlir_feature_disabled".to_string(), 
            FixStrategy::EnableFeatureFlag("mlir".to_string()));
        
        mlir_fixes.insert("mock_mlir_context".to_string(),
            FixStrategy::ReplaceImplementation(MockImplementation {
                mock_type: "MockMLIRContext".to_string(),
                real_type: "MLIRContext".to_string(),
                replacement_code: "MLIRContext::new()".to_string(),
            }));
        
        // Feature flag fixes
        feature_flag_fixes.insert("mlir".to_string(), "mlir".to_string());
        feature_flag_fixes.insert("gpu".to_string(), "gpu".to_string());
        feature_flag_fixes.insert("cuda".to_string(), "cuda".to_string());
        
        // Configuration fixes
        config_fixes.insert("optimization_level".to_string(), ConfigChange {
            config_file: std::path::PathBuf::from("Cargo.toml"),
            section: "profile.release".to_string(),
            key: "opt-level".to_string(),
            value: "3".to_string(),
        });
        
        Self { 
            dependency_fixes,
            type_mismatch_fixes,
            mlir_fixes,
            feature_flag_fixes,
            config_fixes,
        }
    }
    
    pub fn get_fix_for_error(&self, error: &BuildError) -> Option<FixStrategy> {
        match error.error_type {
            ErrorType::MissingDependency => {
                self.get_dependency_fix(&error.message)
            }
            ErrorType::TypeMismatch => {
                self.get_type_mismatch_fix(&error.message)
            }
            ErrorType::MockImplementation => {
                self.get_mlir_fix(&error.message)
            }
            ErrorType::CompilationError => {
                self.get_compilation_fix(&error.message)
            }
            _ => None,
        }
    }
    
    fn get_dependency_fix(&self, message: &str) -> Option<FixStrategy> {
        for (dep_name, dep_info) in &self.dependency_fixes {
            if message.contains(dep_name) {
                return Some(FixStrategy::InstallDependency(dep_info.clone()));
            }
        }
        None
    }
    
    fn get_type_mismatch_fix(&self, message: &str) -> Option<FixStrategy> {
        for (type_name, code_mod) in &self.type_mismatch_fixes {
            if message.contains(type_name) {
                return Some(FixStrategy::ModifyCode(code_mod.clone()));
            }
        }
        None
    }
    
    fn get_mlir_fix(&self, message: &str) -> Option<FixStrategy> {
        for (pattern, fix) in &self.mlir_fixes {
            if message.to_lowercase().contains(pattern) {
                return Some(fix.clone());
            }
        }
        None
    }
    
    fn get_compilation_fix(&self, message: &str) -> Option<FixStrategy> {
        if message.contains("MLIR") && message.contains("feature") {
            return Some(FixStrategy::EnableFeatureFlag("mlir".to_string()));
        }
        
        if message.contains("GPU") && message.contains("feature") {
            return Some(FixStrategy::EnableFeatureFlag("gpu".to_string()));
        }
        
        None
    }
    
    /// Add a new dependency fix to the database
    pub fn add_dependency_fix(&mut self, name: String, info: DependencyInfo) {
        self.dependency_fixes.insert(name, info);
    }
    
    /// Add a new type mismatch fix to the database
    pub fn add_type_mismatch_fix(&mut self, pattern: String, modification: CodeModification) {
        self.type_mismatch_fixes.insert(pattern, modification);
    }
    
    /// Add a new MLIR fix to the database
    pub fn add_mlir_fix(&mut self, pattern: String, fix: FixStrategy) {
        self.mlir_fixes.insert(pattern, fix);
    }
    
    /// Get all available dependency fixes
    pub fn get_all_dependency_fixes(&self) -> &std::collections::HashMap<String, DependencyInfo> {
        &self.dependency_fixes
    }
    
    /// Get statistics about the fix database
    pub fn get_statistics(&self) -> FixDatabaseStatistics {
        FixDatabaseStatistics {
            total_dependency_fixes: self.dependency_fixes.len(),
            total_type_mismatch_fixes: self.type_mismatch_fixes.len(),
            total_mlir_fixes: self.mlir_fixes.len(),
            total_feature_flag_fixes: self.feature_flag_fixes.len(),
            total_config_fixes: self.config_fixes.len(),
        }
    }
}

/// Code modifier for applying code changes
pub struct CodeModifier {
    backup_enabled: bool,
    backup_directory: std::path::PathBuf,
}

impl CodeModifier {
    pub fn new() -> Self {
        Self {
            backup_enabled: true,
            backup_directory: std::path::PathBuf::from(".aether_backups"),
        }
    }
    
    pub fn new_without_backup() -> Self {
        Self {
            backup_enabled: false,
            backup_directory: std::path::PathBuf::from(".aether_backups"),
        }
    }
    
    pub fn apply_modification(&self, modification: &CodeModification) -> Result<(), FixError> {
        let file_path = &modification.file_path;
        
        if !file_path.exists() {
            return Err(FixError::FileNotFound(file_path.to_string_lossy().to_string()));
        }
        
        let content = fs::read_to_string(file_path)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to read file: {}", e)))?;
        
        // Create backup if enabled
        if self.backup_enabled {
            self.create_backup(file_path, &content)?;
        }
        
        // Apply the modification
        let new_content = if modification.line_number > 0 {
            self.apply_line_specific_modification(&content, modification)?
        } else {
            // Global replacement
            content.replace(&modification.old_code, &modification.new_code)
        };
        
        // Verify the change was made
        if new_content == content {
            return Err(FixError::ParseError(format!(
                "No changes were made - old code '{}' not found in file", 
                modification.old_code
            )));
        }
        
        fs::write(file_path, new_content)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to write file: {}", e)))?;
        
        Ok(())
    }
    
    /// Apply modification to a specific line
    fn apply_line_specific_modification(&self, content: &str, modification: &CodeModification) -> Result<String, FixError> {
        let lines: Vec<&str> = content.lines().collect();
        
        if modification.line_number == 0 || modification.line_number > lines.len() {
            return Err(FixError::ParseError(format!(
                "Invalid line number: {} (file has {} lines)", 
                modification.line_number, 
                lines.len()
            )));
        }
        
        let line_index = modification.line_number - 1;
        let current_line = lines[line_index];
        
        if !current_line.contains(&modification.old_code) {
            return Err(FixError::ParseError(format!(
                "Old code '{}' not found on line {}", 
                modification.old_code, 
                modification.line_number
            )));
        }
        
        let new_line = current_line.replace(&modification.old_code, &modification.new_code);
        let mut new_lines = lines;
        new_lines[line_index] = &new_line;
        
        Ok(new_lines.join("\n"))
    }
    
    /// Create backup of file before modification
    fn create_backup(&self, file_path: &std::path::Path, content: &str) -> Result<(), FixError> {
        if !self.backup_directory.exists() {
            fs::create_dir_all(&self.backup_directory)
                .map_err(|e| FixError::FileOperationFailed(format!("Failed to create backup directory: {}", e)))?;
        }
        
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let file_name = file_path.file_name()
            .ok_or_else(|| FixError::FileOperationFailed("Invalid file path".to_string()))?;
        
        let backup_name = format!("{}_{}.backup", file_name.to_string_lossy(), timestamp);
        let backup_path = self.backup_directory.join(backup_name);
        
        fs::write(backup_path, content)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to create backup: {}", e)))?;
        
        Ok(())
    }
    
    /// Apply multiple modifications to the same file
    pub fn apply_multiple_modifications(&self, file_path: &std::path::Path, modifications: &[CodeModification]) -> Result<(), FixError> {
        if !file_path.exists() {
            return Err(FixError::FileNotFound(file_path.to_string_lossy().to_string()));
        }
        
        let mut content = fs::read_to_string(file_path)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to read file: {}", e)))?;
        
        // Create backup if enabled
        if self.backup_enabled {
            self.create_backup(file_path, &content)?;
        }
        
        // Apply modifications in order
        for modification in modifications {
            if modification.file_path != file_path {
                continue; // Skip modifications for other files
            }
            
            content = content.replace(&modification.old_code, &modification.new_code);
        }
        
        fs::write(file_path, content)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to write file: {}", e)))?;
        
        Ok(())
    }
    
    /// Restore file from backup
    pub fn restore_from_backup(&self, file_path: &std::path::Path) -> Result<(), FixError> {
        let file_name = file_path.file_name()
            .ok_or_else(|| FixError::FileOperationFailed("Invalid file path".to_string()))?;
        
        // Find the most recent backup
        let backup_pattern = format!("{}_", file_name.to_string_lossy());
        
        let mut backups = Vec::new();
        if self.backup_directory.exists() {
            for entry in fs::read_dir(&self.backup_directory)
                .map_err(|e| FixError::FileOperationFailed(format!("Failed to read backup directory: {}", e)))? {
                let entry = entry.map_err(|e| FixError::FileOperationFailed(format!("Failed to read directory entry: {}", e)))?;
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&backup_pattern) && name.ends_with(".backup") {
                    backups.push((name, entry.path()));
                }
            }
        }
        
        if backups.is_empty() {
            return Err(FixError::FileNotFound("No backup found".to_string()));
        }
        
        // Sort by timestamp (newest first)
        backups.sort_by(|a, b| b.0.cmp(&a.0));
        
        let backup_content = fs::read_to_string(&backups[0].1)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to read backup: {}", e)))?;
        
        fs::write(file_path, backup_content)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to restore file: {}", e)))?;
        
        Ok(())
    }
    
    /// Set backup enabled/disabled
    pub fn set_backup_enabled(&mut self, enabled: bool) {
        self.backup_enabled = enabled;
    }
    
    /// Set backup directory
    pub fn set_backup_directory(&mut self, directory: std::path::PathBuf) {
        self.backup_directory = directory;
    }
}

/// Configuration manager for updating config files
pub struct ConfigManager {
    backup_enabled: bool,
}

impl ConfigManager {
    pub fn new() -> Self {
        Self {
            backup_enabled: true,
        }
    }
    
    pub fn new_without_backup() -> Self {
        Self {
            backup_enabled: false,
        }
    }
    
    pub fn apply_change(&self, change: &ConfigChange) -> Result<(), FixError> {
        let file_path = &change.config_file;
        
        if !file_path.exists() {
            return Err(FixError::FileNotFound(file_path.to_string_lossy().to_string()));
        }
        
        let content = fs::read_to_string(file_path)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to read config file: {}", e)))?;
        
        // Create backup if enabled
        if self.backup_enabled {
            self.create_config_backup(file_path, &content)?;
        }
        
        // Handle different config file types
        let new_content = match file_path.extension().and_then(|s| s.to_str()) {
            Some("toml") => self.apply_toml_change(&content, change)?,
            Some("json") => self.apply_json_change(&content, change)?,
            Some("yaml") | Some("yml") => self.apply_yaml_change(&content, change)?,
            _ => self.apply_generic_change(&content, change)?,
        };
        
        fs::write(file_path, new_content)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }
    
    /// Apply change to TOML file
    fn apply_toml_change(&self, content: &str, change: &ConfigChange) -> Result<String, FixError> {
        let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
        let section_header = format!("[{}]", change.section);
        let key_value = format!("{} = {}", change.key, change.value);
        
        // Find the section
        let mut section_found = false;
        let mut section_start = 0;
        let mut section_end = lines.len();
        
        for (i, line) in lines.iter().enumerate() {
            if line.trim() == section_header {
                section_found = true;
                section_start = i + 1;
            } else if section_found && line.trim().starts_with('[') && line.trim().ends_with(']') {
                section_end = i;
                break;
            }
        }
        
        if !section_found {
            // Add new section at the end
            lines.push(String::new());
            lines.push(section_header);
            lines.push(key_value);
        } else {
            // Check if key already exists in section
            let mut key_found = false;
            for i in section_start..section_end {
                if i < lines.len() && lines[i].trim().starts_with(&format!("{} =", change.key)) {
                    lines[i] = key_value.clone();
                    key_found = true;
                    break;
                }
            }
            
            if !key_found {
                // Add key to section
                lines.insert(section_end, key_value);
            }
        }
        
        Ok(lines.join("\n"))
    }
    
    /// Apply change to JSON file (simplified)
    fn apply_json_change(&self, content: &str, change: &ConfigChange) -> Result<String, FixError> {
        // This is a simplified implementation
        // In a real implementation, you'd use a proper JSON parser
        let key_value = format!("  \"{}\": {}", change.key, change.value);
        
        if content.trim().ends_with('}') {
            let mut new_content = content.trim_end_matches('}').to_string();
            if !new_content.trim().ends_with(',') && !new_content.trim().ends_with('{') {
                new_content.push(',');
            }
            new_content.push('\n');
            new_content.push_str(&key_value);
            new_content.push('\n');
            new_content.push('}');
            Ok(new_content)
        } else {
            Err(FixError::ParseError("Invalid JSON format".to_string()))
        }
    }
    
    /// Apply change to YAML file (simplified)
    fn apply_yaml_change(&self, content: &str, change: &ConfigChange) -> Result<String, FixError> {
        let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
        let key_value = format!("{}: {}", change.key, change.value);
        
        // Simple YAML handling - just append at the end
        lines.push(key_value);
        Ok(lines.join("\n"))
    }
    
    /// Apply generic change (fallback)
    fn apply_generic_change(&self, content: &str, change: &ConfigChange) -> Result<String, FixError> {
        let key_value = format!("{} = {}", change.key, change.value);
        Ok(format!("{}\n{}", content, key_value))
    }
    
    /// Create backup of config file
    fn create_config_backup(&self, file_path: &std::path::Path, content: &str) -> Result<(), FixError> {
        let backup_dir = std::path::PathBuf::from(".aether_config_backups");
        
        if !backup_dir.exists() {
            fs::create_dir_all(&backup_dir)
                .map_err(|e| FixError::FileOperationFailed(format!("Failed to create backup directory: {}", e)))?;
        }
        
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let file_name = file_path.file_name()
            .ok_or_else(|| FixError::FileOperationFailed("Invalid file path".to_string()))?;
        
        let backup_name = format!("{}_{}.backup", file_name.to_string_lossy(), timestamp);
        let backup_path = backup_dir.join(backup_name);
        
        fs::write(backup_path, content)
            .map_err(|e| FixError::FileOperationFailed(format!("Failed to create config backup: {}", e)))?;
        
        Ok(())
    }
    
    /// Apply multiple configuration changes
    pub fn apply_multiple_changes(&self, changes: &[ConfigChange]) -> Result<(), FixError> {
        // Group changes by file
        let mut changes_by_file: std::collections::HashMap<std::path::PathBuf, Vec<&ConfigChange>> = 
            std::collections::HashMap::new();
        
        for change in changes {
            changes_by_file.entry(change.config_file.clone())
                .or_insert_with(Vec::new)
                .push(change);
        }
        
        // Apply changes file by file
        for (file_path, file_changes) in changes_by_file {
            if !file_path.exists() {
                return Err(FixError::FileNotFound(file_path.to_string_lossy().to_string()));
            }
            
            let mut content = fs::read_to_string(&file_path)
                .map_err(|e| FixError::FileOperationFailed(format!("Failed to read config file: {}", e)))?;
            
            // Create backup if enabled
            if self.backup_enabled {
                self.create_config_backup(&file_path, &content)?;
            }
            
            // Apply all changes to this file
            for change in file_changes {
                content = match file_path.extension().and_then(|s| s.to_str()) {
                    Some("toml") => self.apply_toml_change(&content, change)?,
                    Some("json") => self.apply_json_change(&content, change)?,
                    Some("yaml") | Some("yml") => self.apply_yaml_change(&content, change)?,
                    _ => self.apply_generic_change(&content, change)?,
                };
            }
            
            fs::write(&file_path, content)
                .map_err(|e| FixError::FileOperationFailed(format!("Failed to write config file: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Set backup enabled/disabled
    pub fn set_backup_enabled(&mut self, enabled: bool) {
        self.backup_enabled = enabled;
    }
}

/// Fix application errors
#[derive(Debug, Clone)]
pub enum FixError {
    FileNotFound(String),
    FileOperationFailed(String),
    ParseError(String),
    VerificationFailed(String),
    CommandFailed(String),
}

impl std::fmt::Display for FixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FixError::FileNotFound(path) => write!(f, "File not found: {}", path),
            FixError::FileOperationFailed(msg) => write!(f, "File operation failed: {}", msg),
            FixError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            FixError::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
            FixError::CommandFailed(msg) => write!(f, "Command failed: {}", msg),
        }
    }
}

impl std::error::Error for FixError {}

/// Statistics about the fix database
#[derive(Debug, Clone)]
pub struct FixDatabaseStatistics {
    pub total_dependency_fixes: usize,
    pub total_type_mismatch_fixes: usize,
    pub total_mlir_fixes: usize,
    pub total_feature_flag_fixes: usize,
    pub total_config_fixes: usize,
}

/// Fix application result with detailed information
#[derive(Debug, Clone)]
pub struct FixApplicationResult {
    pub fix_strategy: FixStrategy,
    pub application_result: Result<(), FixError>,
    pub verification_result: Option<bool>,
    pub applied_at: std::time::SystemTime,
    pub files_modified: Vec<std::path::PathBuf>,
}

impl FixApplicationResult {
    pub fn new(fix_strategy: FixStrategy) -> Self {
        Self {
            fix_strategy,
            application_result: Err(FixError::CommandFailed("Not applied yet".to_string())),
            verification_result: None,
            applied_at: std::time::SystemTime::now(),
            files_modified: Vec::new(),
        }
    }
    
    pub fn with_success(mut self, files_modified: Vec<std::path::PathBuf>) -> Self {
        self.application_result = Ok(());
        self.files_modified = files_modified;
        self
    }
    
    pub fn with_error(mut self, error: FixError) -> Self {
        self.application_result = Err(error);
        self
    }
    
    pub fn with_verification(mut self, verified: bool) -> Self {
        self.verification_result = Some(verified);
        self
    }
}