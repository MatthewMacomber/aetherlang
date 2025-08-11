// Cargo Build System Integration for Aether File Compilation Testing
// Provides integration with Cargo build process and CI/CD pipelines

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use serde::{Serialize, Deserialize};
use thiserror::Error;

use super::file_compilation_testing::{
    FileCompilationTestOrchestrator, TestingConfig, TestingError, FileCompilationTestReport
};
use super::test_cache::{TestCache, CacheConfig, CacheError};

/// Integration with Cargo build system
pub struct CargoIntegration {
    config: CargoIntegrationConfig,
    cache: Option<TestCache>,
}

/// Configuration for Cargo integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoIntegrationConfig {
    /// Enable automatic testing during build
    pub auto_test_on_build: bool,
    
    /// Enable testing during cargo test
    pub integrate_with_cargo_test: bool,
    
    /// Cache directory for test results
    pub cache_directory: PathBuf,
    
    /// Enable caching of compilation results
    pub enable_caching: bool,
    
    /// Cache expiration time in hours
    pub cache_expiration_hours: u64,
    
    /// Run tests only on changed files
    pub incremental_testing: bool,
    
    /// Fail build on test failures
    pub fail_build_on_test_failure: bool,
    
    /// Generate CI/CD artifacts
    pub generate_ci_artifacts: bool,
    
    /// CI artifact output directory
    pub ci_artifact_directory: PathBuf,
}

impl Default for CargoIntegrationConfig {
    fn default() -> Self {
        Self {
            auto_test_on_build: false,
            integrate_with_cargo_test: true,
            cache_directory: PathBuf::from("target/aether_test_cache"),
            enable_caching: true,
            cache_expiration_hours: 24,
            incremental_testing: true,
            fail_build_on_test_failure: false,
            generate_ci_artifacts: true,
            ci_artifact_directory: PathBuf::from("target/ci_artifacts"),
        }
    }
}

impl CargoIntegration {
    /// Create a new Cargo integration instance
    pub fn new(config: CargoIntegrationConfig) -> Self {
        let cache = if config.enable_caching {
            let cache_config = CacheConfig {
                max_age_hours: config.cache_expiration_hours,
                ..Default::default()
            };
            
            TestCache::new(config.cache_directory.clone(), cache_config).ok()
        } else {
            None
        };
        
        Self { config, cache }
    }

    /// Run file compilation tests as part of cargo test
    pub async fn run_as_cargo_test(&self, testing_config: TestingConfig) -> Result<(), CargoIntegrationError> {
        println!("Running Aether file compilation tests...");
        
        // Create cache directory if it doesn't exist
        if self.config.enable_caching {
            std::fs::create_dir_all(&self.config.cache_directory)?;
        }
        
        // Check if we should run incremental testing
        let files_to_test = if self.config.incremental_testing {
            self.get_changed_files(&testing_config)?
        } else {
            None
        };
        
        // Create modified testing config for incremental testing
        let mut modified_config = testing_config;
        if let Some(changed_files) = files_to_test {
            if changed_files.is_empty() {
                println!("No changed Aether files detected, skipping tests");
                return Ok(());
            }
            println!("Running incremental tests on {} changed files", changed_files.len());
        }
        
        // Run the test suite
        let mut orchestrator = FileCompilationTestOrchestrator::new(modified_config)?;
        let report = orchestrator.run_complete_test_suite().await?;
        
        // Cache results if enabled
        if self.config.enable_caching {
            self.cache_test_results(&report).await?;
        }
        
        // Generate CI artifacts if enabled
        if self.config.generate_ci_artifacts {
            self.generate_ci_artifacts(&report).await?;
        }
        
        // Check if we should fail the build
        if self.config.fail_build_on_test_failure && !self.all_tests_passed(&report) {
            return Err(CargoIntegrationError::TestFailure(
                "File compilation tests failed".to_string()
            ));
        }
        
        println!("File compilation tests completed successfully");
        Ok(())
    }
    
    /// Run tests during build process
    pub async fn run_during_build(&self, testing_config: TestingConfig) -> Result<(), CargoIntegrationError> {
        if !self.config.auto_test_on_build {
            return Ok(());
        }
        
        println!("cargo:warning=Running Aether file compilation tests during build...");
        
        // Run a lightweight version of tests during build
        let mut lightweight_config = testing_config;
        lightweight_config.generate_additional_tests = false;
        lightweight_config.compilation_timeout = Duration::from_secs(10);
        lightweight_config.execution_timeout = Duration::from_secs(5);
        
        let mut orchestrator = FileCompilationTestOrchestrator::new(lightweight_config)?;
        let report = orchestrator.run_complete_test_suite().await?;
        
        if self.all_tests_passed(&report) {
            println!("cargo:warning=✓ File compilation tests passed");
        } else {
            println!("cargo:warning=⚠ Some file compilation tests failed");
            if self.config.fail_build_on_test_failure {
                return Err(CargoIntegrationError::TestFailure(
                    "Build failed due to file compilation test failures".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    /// Get list of changed Aether files for incremental testing
    fn get_changed_files(&self, config: &TestingConfig) -> Result<Option<Vec<PathBuf>>, CargoIntegrationError> {
        if !self.config.incremental_testing {
            return Ok(None);
        }
        
        // Use git to find changed files
        let output = Command::new("git")
            .args(&["diff", "--name-only", "HEAD~1", "HEAD"])
            .current_dir(&config.project_root)
            .output();
            
        match output {
            Ok(result) if result.status.success() => {
                let stdout = String::from_utf8_lossy(&result.stdout);
                let changed_files: Vec<PathBuf> = stdout
                    .lines()
                    .filter(|line| line.ends_with(".ae"))
                    .map(|line| config.project_root.join(line))
                    .filter(|path| path.exists())
                    .collect();
                    
                Ok(Some(changed_files))
            }
            _ => {
                // Fall back to testing all files if git is not available
                println!("Warning: Could not determine changed files, testing all files");
                Ok(None)
            }
        }
    }
    
    /// Cache test results for future use
    async fn cache_test_results(&self, report: &FileCompilationTestReport) -> Result<(), CargoIntegrationError> {
        let cache_file = self.config.cache_directory.join("test_results.json");
        
        // Create cache entry
        let cache_entry = TestResultCache {
            timestamp: chrono::Utc::now(),
            report: report.clone(),
            config_hash: self.calculate_config_hash(&report.config),
        };
        
        // Serialize and save
        let cache_content = serde_json::to_string_pretty(&cache_entry)?;
        std::fs::write(&cache_file, cache_content)?;
        
        // Clean up old cache entries
        self.cleanup_expired_cache().await?;
        
        Ok(())
    }
    
    /// Generate CI/CD artifacts
    async fn generate_ci_artifacts(&self, report: &FileCompilationTestReport) -> Result<(), CargoIntegrationError> {
        std::fs::create_dir_all(&self.config.ci_artifact_directory)?;
        
        // Generate JUnit XML report for CI systems
        let junit_xml = self.generate_junit_xml(report)?;
        let junit_path = self.config.ci_artifact_directory.join("test_results.xml");
        std::fs::write(&junit_path, junit_xml)?;
        
        // Generate JSON report for programmatic consumption
        let json_report = serde_json::to_string_pretty(report)?;
        let json_path = self.config.ci_artifact_directory.join("test_results.json");
        std::fs::write(&json_path, json_report)?;
        
        // Generate summary report
        let summary = self.generate_test_summary(report);
        let summary_path = self.config.ci_artifact_directory.join("test_summary.txt");
        std::fs::write(&summary_path, summary)?;
        
        println!("Generated CI artifacts in: {}", self.config.ci_artifact_directory.display());
        Ok(())
    }
    
    /// Generate JUnit XML format for CI systems
    fn generate_junit_xml(&self, report: &FileCompilationTestReport) -> Result<String, CargoIntegrationError> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<testsuites>\n");
        
        // Compilation test suite
        xml.push_str(&format!(
            "  <testsuite name=\"compilation\" tests=\"{}\" failures=\"{}\" time=\"0\">\n",
            report.compilation_results.len(),
            report.compilation_results.iter().filter(|r| !r.success).count()
        ));
        
        for result in &report.compilation_results {
            let test_name = result.source_file.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
                
            if result.success {
                xml.push_str(&format!(
                    "    <testcase name=\"{}\" classname=\"compilation\" time=\"{:.3}\"/>\n",
                    test_name,
                    result.compilation_time.as_secs_f64()
                ));
            } else {
                xml.push_str(&format!(
                    "    <testcase name=\"{}\" classname=\"compilation\" time=\"{:.3}\">\n",
                    test_name,
                    result.compilation_time.as_secs_f64()
                ));
                xml.push_str(&format!(
                    "      <failure message=\"Compilation failed\">{}</failure>\n",
                    html_escape::encode_text(&result.stderr)
                ));
                xml.push_str("    </testcase>\n");
            }
        }
        
        xml.push_str("  </testsuite>\n");
        
        // Execution test suite
        xml.push_str(&format!(
            "  <testsuite name=\"execution\" tests=\"{}\" failures=\"{}\" time=\"0\">\n",
            report.execution_results.len(),
            report.execution_results.iter().filter(|r| !r.success).count()
        ));
        
        for result in &report.execution_results {
            let test_name = result.executable_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
                
            if result.success {
                xml.push_str(&format!(
                    "    <testcase name=\"{}\" classname=\"execution\" time=\"{:.3}\"/>\n",
                    test_name,
                    result.execution_time.as_secs_f64()
                ));
            } else {
                xml.push_str(&format!(
                    "    <testcase name=\"{}\" classname=\"execution\" time=\"{:.3}\">\n",
                    test_name,
                    result.execution_time.as_secs_f64()
                ));
                xml.push_str(&format!(
                    "      <failure message=\"Execution failed (exit code: {})\">STDERR:\n{}\nSTDOUT:\n{}</failure>\n",
                    result.exit_code,
                    html_escape::encode_text(&result.stderr),
                    html_escape::encode_text(&result.stdout)
                ));
                xml.push_str("    </testcase>\n");
            }
        }
        
        xml.push_str("  </testsuite>\n");
        xml.push_str("</testsuites>\n");
        
        Ok(xml)
    }
    
    /// Generate test summary for CI
    fn generate_test_summary(&self, report: &FileCompilationTestReport) -> String {
        let mut summary = String::new();
        
        summary.push_str("=== Aether File Compilation Test Summary ===\n\n");
        
        summary.push_str(&format!("Total files discovered: {}\n", report.discovered_files.len()));
        summary.push_str(&format!("Generated test files: {}\n", report.generated_files.len()));
        summary.push_str(&format!("Compilation attempts: {}\n", report.compilation_results.len()));
        summary.push_str(&format!("Successful compilations: {}\n", 
            report.compilation_results.iter().filter(|r| r.success).count()));
        summary.push_str(&format!("Execution attempts: {}\n", report.execution_results.len()));
        summary.push_str(&format!("Successful executions: {}\n", 
            report.execution_results.iter().filter(|r| r.success).count()));
        
        let compilation_failures: Vec<_> = report.compilation_results
            .iter()
            .filter(|r| !r.success)
            .collect();
            
        if !compilation_failures.is_empty() {
            summary.push_str("\nCompilation Failures:\n");
            for failure in compilation_failures {
                summary.push_str(&format!("  - {}\n", failure.source_file.display()));
            }
        }
        
        let execution_failures: Vec<_> = report.execution_results
            .iter()
            .filter(|r| !r.success)
            .collect();
            
        if !execution_failures.is_empty() {
            summary.push_str("\nExecution Failures:\n");
            for failure in execution_failures {
                summary.push_str(&format!("  - {} (exit code: {})\n", 
                    failure.executable_path.display(), failure.exit_code));
            }
        }
        
        if self.all_tests_passed(report) {
            summary.push_str("\n✅ All tests passed successfully!\n");
        } else {
            summary.push_str("\n❌ Some tests failed.\n");
        }
        
        summary
    }
    
    /// Check if all tests passed
    fn all_tests_passed(&self, report: &FileCompilationTestReport) -> bool {
        report.compilation_results.iter().all(|r| r.success) &&
        report.execution_results.iter().all(|r| r.success)
    }
    
    /// Calculate configuration hash for cache invalidation
    fn calculate_config_hash(&self, config: &TestingConfig) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash relevant config fields that affect test results
        config.compiler_path.hash(&mut hasher);
        config.test_directories.hash(&mut hasher);
        config.generate_additional_tests.hash(&mut hasher);
        config.test_categories.hash(&mut hasher);
        
        hasher.finish()
    }
    
    /// Clean up expired cache entries
    async fn cleanup_expired_cache(&self) -> Result<(), CargoIntegrationError> {
        let cache_dir = &self.config.cache_directory;
        if !cache_dir.exists() {
            return Ok(());
        }
        
        let expiration_duration = chrono::Duration::hours(self.config.cache_expiration_hours as i64);
        let cutoff_time = chrono::Utc::now() - expiration_duration;
        
        let entries = std::fs::read_dir(cache_dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                // Check file modification time
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        let modified_time = chrono::DateTime::<chrono::Utc>::from(modified);
                        if modified_time < cutoff_time {
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
}

/// Cache entry for test results
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestResultCache {
    timestamp: chrono::DateTime<chrono::Utc>,
    report: FileCompilationTestReport,
    config_hash: u64,
}

/// Errors that can occur during Cargo integration
#[derive(Debug, Error)]
pub enum CargoIntegrationError {
    #[error("Testing error: {0}")]
    Testing(#[from] TestingError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Test failure: {0}")]
    TestFailure(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("CI artifact generation error: {0}")]
    CiArtifact(String),
}

/// Cargo test integration function
/// This function is called by cargo test to run file compilation tests
pub async fn run_cargo_test_integration() -> Result<(), CargoIntegrationError> {
    let cargo_config = CargoIntegrationConfig::default();
    let testing_config = TestingConfig::default();
    
    let integration = CargoIntegration::new(cargo_config);
    integration.run_as_cargo_test(testing_config).await
}

/// Build script integration function
/// This function is called during cargo build if AETHER_RUN_FILE_TESTS is set
pub async fn run_build_integration() -> Result<(), CargoIntegrationError> {
    let mut cargo_config = CargoIntegrationConfig::default();
    cargo_config.auto_test_on_build = true;
    cargo_config.fail_build_on_test_failure = std::env::var("AETHER_FAIL_BUILD_ON_TEST_FAILURE").is_ok();
    
    let testing_config = TestingConfig::default();
    
    let integration = CargoIntegration::new(cargo_config);
    integration.run_during_build(testing_config).await
}

// Add html_escape dependency simulation (since it's not in Cargo.toml)
mod html_escape {
    pub fn encode_text(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
    }
}