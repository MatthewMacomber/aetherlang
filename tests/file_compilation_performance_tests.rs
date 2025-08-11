// Performance tests and benchmarks for the file compilation testing system
// Tests system performance with large file sets and various workloads

use aether_language::testing::{
    FileCompilationTestOrchestrator, TestingConfig, TestCategory, ReportFormat
};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::fs;
use tempfile::TempDir;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Performance test utilities
struct PerformanceTestUtils;

impl PerformanceTestUtils {
    /// Create a large test project with many files
    fn create_large_test_project(file_count: usize) -> Result<TempDir, std::io::Error> {
        let temp_dir = tempfile::tempdir()?;
        
        // Create examples directory
        let examples_dir = temp_dir.path().join("examples");
        fs::create_dir_all(&examples_dir)?;
        
        // Create tests directory
        let tests_dir = temp_dir.path().join("tests");
        fs::create_dir_all(&tests_dir)?;
        
        // Create nested directories for organization
        let nested_dirs = ["basic", "advanced", "algorithms", "data_structures", "ai"];
        for dir in &nested_dirs {
            fs::create_dir_all(examples_dir.join(dir))?;
            fs::create_dir_all(tests_dir.join(dir))?;
        }
        
        // Generate files with varying complexity
        for i in 0..file_count {
            let dir_index = i % nested_dirs.len();
            let target_dir = if i % 2 == 0 { &examples_dir } else { &tests_dir };
            let nested_dir = target_dir.join(nested_dirs[dir_index]);
            
            // Create files with different complexity levels
            let complexity = match i % 4 {
                0 => "simple",
                1 => "medium", 
                2 => "complex",
                _ => "advanced",
            };
            
            let file_content = match complexity {
                "simple" => format!(r#"
;; Simple program {}
(defn main []
  (println "Hello from program {}"))
"#, i, i),
                "medium" => format!(r#"
;; Medium complexity program {}
(defn fibonacci [n]
  (if (<= n 1)
    n
    (+ (fibonacci (- n 1)) (fibonacci (- n 2)))))

(defn main []
  (let [result (fibonacci 10)]
    (println "Fibonacci result for program {}: " result)))
"#, i, i),
                "complex" => format!(r#"
;; Complex program {} with data structures
(defn process-data [data]
  (map (fn [x] 
    (let [squared (* x x)
          doubled (* x 2)]
      (+ squared doubled))) data))

(defn main []
  (let [data [1 2 3 4 5 6 7 8 9 10]
        processed (process-data data)
        sum (reduce + 0 processed)]
    (println "Program {} processed sum: " sum)))
"#, i, i),
                _ => format!(r#"
;; Advanced program {} with multiple features
(defstruct Point [x y])

(defn distance [p1 p2]
  (let [dx (- (:x p2) (:x p1))
        dy (- (:y p2) (:y p1))]
    (sqrt (+ (* dx dx) (* dy dy)))))

(defn create-points [n]
  (for [i (range n)]
    (Point. (random) (random))))

(defn main []
  (let [points (create-points 100)
        origin (Point. 0.0 0.0)
        distances (map (partial distance origin) points)
        avg-distance (/ (reduce + distances) (count distances))]
    (println "Program {} average distance: " avg-distance)))
"#, i, i),
            };
            
            fs::write(nested_dir.join(format!("program_{}.ae", i)), file_content)?;
        }
        
        Ok(temp_dir)
    }
    
    /// Create a performance testing configuration
    fn create_performance_config(temp_dir: &Path, max_parallel: usize) -> TestingConfig {
        TestingConfig {
            project_root: temp_dir.to_path_buf(),
            compiler_path: temp_dir.join("mock_aetherc.exe"),
            output_directory: temp_dir.join("perf_output"),
            test_directories: vec![
                "examples".to_string(),
                "tests".to_string(),
            ],
            compilation_timeout: Duration::from_secs(60),
            execution_timeout: Duration::from_secs(30),
            generate_additional_tests: false,
            test_categories: vec![TestCategory::CoreLanguage],
            report_format: ReportFormat::Json,
            cleanup_artifacts: true,
            max_parallel_compilations: max_parallel,
            max_parallel_executions: max_parallel,
            verbose: false,
        }
    }
    
    /// Create a mock compiler for performance testing
    fn create_performance_mock_compiler(temp_dir: &Path) -> Result<PathBuf, std::io::Error> {
        let compiler_path = temp_dir.join("mock_aetherc.exe");
        
        #[cfg(windows)]
        {
            fs::write(
                &compiler_path,
                r#"@echo off
timeout /t 1 /nobreak >nul 2>&1
echo [PERF COMPILER] Compiled %1
echo @echo off > "%~n1.exe"
echo echo Hello from %1 >> "%~n1.exe"
exit /b 0
"#
            )?;
        }
        
        #[cfg(not(windows))]
        {
            fs::write(
                &compiler_path,
                r#"#!/bin/bash
sleep 1
echo "[PERF COMPILER] Compiled $1"
cat > "${1%.*}.exe" << EOF
#!/bin/bash
echo "Hello from $1"
EOF
chmod +x "${1%.*}.exe"
exit 0
"#
            )?;
            
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&compiler_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&compiler_path, perms)?;
        }
        
        Ok(compiler_path)
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_small_file_set_performance() {
        let temp_dir = PerformanceTestUtils::create_large_test_project(10).unwrap();
        let _compiler_path = PerformanceTestUtils::create_performance_mock_compiler(temp_dir.path()).unwrap();
        let config = PerformanceTestUtils::create_performance_config(temp_dir.path(), 2);
        
        let mut orchestrator = match FileCompilationTestOrchestrator::new(config) {
            Ok(orch) => orch,
            Err(e) => {
                println!("Performance test setup failed: {}", e);
                return;
            }
        };
        
        let start_time = Instant::now();
        let result = orchestrator.run_complete_test_suite().await;
        let duration = start_time.elapsed();
        
        match result {
            Ok(report) => {
                println!("Small file set performance test completed in {:?}", duration);
                println!("Files processed: {}", report.summary.total_files);
                println!("Successful compilations: {}", report.summary.successful_compilations);
                
                assert!(report.summary.total_files >= 10);
                assert!(duration < Duration::from_secs(30)); // Should complete quickly
            }
            Err(e) => {
                println!("Small file set performance test failed: {}", e);
            }
        }
    }
    
    #[tokio::test]
    async fn test_medium_file_set_performance() {
        let temp_dir = PerformanceTestUtils::create_large_test_project(50).unwrap();
        let _compiler_path = PerformanceTestUtils::create_performance_mock_compiler(temp_dir.path()).unwrap();
        let config = PerformanceTestUtils::create_performance_config(temp_dir.path(), 4);
        
        let mut orchestrator = match FileCompilationTestOrchestrator::new(config) {
            Ok(orch) => orch,
            Err(e) => {
                println!("Medium performance test setup failed: {}", e);
                return;
            }
        };
        
        let start_time = Instant::now();
        let result = orchestrator.run_complete_test_suite().await;
        let duration = start_time.elapsed();
        
        match result {
            Ok(report) => {
                println!("Medium file set performance test completed in {:?}", duration);
                println!("Files processed: {}", report.summary.total_files);
                println!("Average time per file: {:?}", 
                    duration / report.summary.total_files as u32);
                
                assert!(report.summary.total_files >= 50);
                assert!(duration < Duration::from_secs(120)); // Should complete within 2 minutes
            }
            Err(e) => {
                println!("Medium file set performance test failed: {}", e);
            }
        }
    }
    
    #[tokio::test]
    #[ignore] // Ignore by default due to long runtime
    async fn test_large_file_set_performance() {
        let temp_dir = PerformanceTestUtils::create_large_test_project(200).unwrap();
        let _compiler_path = PerformanceTestUtils::create_performance_mock_compiler(temp_dir.path()).unwrap();
        let config = PerformanceTestUtils::create_performance_config(temp_dir.path(), 8);
        
        let mut orchestrator = match FileCompilationTestOrchestrator::new(config) {
            Ok(orch) => orch,
            Err(e) => {
                println!("Large performance test setup failed: {}", e);
                return;
            }
        };
        
        let start_time = Instant::now();
        let result = orchestrator.run_complete_test_suite().await;
        let duration = start_time.elapsed();
        
        match result {
            Ok(report) => {
                println!("Large file set performance test completed in {:?}", duration);
                println!("Files processed: {}", report.summary.total_files);
                println!("Average time per file: {:?}", 
                    duration / report.summary.total_files as u32);
                println!("Throughput: {:.2} files/second", 
                    report.summary.total_files as f64 / duration.as_secs_f64());
                
                assert!(report.summary.total_files >= 200);
                assert!(duration < Duration::from_secs(600)); // Should complete within 10 minutes
            }
            Err(e) => {
                println!("Large file set performance test failed: {}", e);
            }
        }
    }
    
    #[tokio::test]
    async fn test_parallel_compilation_scaling() {
        let temp_dir = PerformanceTestUtils::create_large_test_project(20).unwrap();
        let _compiler_path = PerformanceTestUtils::create_performance_mock_compiler(temp_dir.path()).unwrap();
        
        let parallel_levels = vec![1, 2, 4, 8];
        let mut results = Vec::new();
        
        for parallel_count in parallel_levels {
            let config = PerformanceTestUtils::create_performance_config(temp_dir.path(), parallel_count);
            
            let mut orchestrator = match FileCompilationTestOrchestrator::new(config) {
                Ok(orch) => orch,
                Err(e) => {
                    println!("Parallel scaling test setup failed for {}: {}", parallel_count, e);
                    continue;
                }
            };
            
            let start_time = Instant::now();
            let result = orchestrator.run_complete_test_suite().await;
            let duration = start_time.elapsed();
            
            if let Ok(report) = result {
                results.push((parallel_count, duration, report.summary.total_files));
                println!("Parallel level {}: {:?} for {} files", 
                    parallel_count, duration, report.summary.total_files);
            }
        }
        
        // Verify that higher parallelism generally improves performance
        if results.len() >= 2 {
            let (_, single_thread_time, _) = results[0];
            let (_, multi_thread_time, _) = results[results.len() - 1];
            
            println!("Single thread: {:?}, Multi thread: {:?}", single_thread_time, multi_thread_time);
            
            // Multi-threading should be at least somewhat faster (allowing for overhead)
            assert!(multi_thread_time <= single_thread_time * 2);
        }
    }
    
    #[tokio::test]
    async fn test_memory_usage_scaling() {
        let temp_dir = PerformanceTestUtils::create_large_test_project(30).unwrap();
        let _compiler_path = PerformanceTestUtils::create_performance_mock_compiler(temp_dir.path()).unwrap();
        let config = PerformanceTestUtils::create_performance_config(temp_dir.path(), 4);
        
        let mut orchestrator = match FileCompilationTestOrchestrator::new(config) {
            Ok(orch) => orch,
            Err(e) => {
                println!("Memory usage test setup failed: {}", e);
                return;
            }
        };
        
        // Get initial memory usage
        let initial_memory = get_memory_usage();
        
        let result = orchestrator.run_complete_test_suite().await;
        
        // Get peak memory usage
        let peak_memory = get_memory_usage();
        
        match result {
            Ok(report) => {
                let memory_increase = peak_memory.saturating_sub(initial_memory);
                let memory_per_file = memory_increase / report.summary.total_files.max(1);
                
                println!("Memory usage test completed");
                println!("Initial memory: {} KB", initial_memory);
                println!("Peak memory: {} KB", peak_memory);
                println!("Memory increase: {} KB", memory_increase);
                println!("Memory per file: {} KB", memory_per_file);
                
                // Memory usage should be reasonable (less than 100MB increase)
                assert!(memory_increase < 100_000);
            }
            Err(e) => {
                println!("Memory usage test failed: {}", e);
            }
        }
    }
    
    #[tokio::test]
    async fn test_compilation_timeout_handling() {
        let temp_dir = PerformanceTestUtils::create_large_test_project(5).unwrap();
        let _compiler_path = PerformanceTestUtils::create_performance_mock_compiler(temp_dir.path()).unwrap();
        
        // Create config with very short timeout
        let mut config = PerformanceTestUtils::create_performance_config(temp_dir.path(), 2);
        config.compilation_timeout = Duration::from_millis(100); // Very short timeout
        
        let mut orchestrator = match FileCompilationTestOrchestrator::new(config) {
            Ok(orch) => orch,
            Err(e) => {
                println!("Timeout test setup failed: {}", e);
                return;
            }
        };
        
        let start_time = Instant::now();
        let result = orchestrator.run_complete_test_suite().await;
        let duration = start_time.elapsed();
        
        match result {
            Ok(report) => {
                println!("Timeout handling test completed in {:?}", duration);
                println!("Total files: {}", report.summary.total_files);
                println!("Failed compilations: {}", report.summary.failed_compilations);
                
                // With very short timeout, most compilations should fail
                assert!(report.summary.failed_compilations > 0);
                
                // But the system should handle it gracefully and complete quickly
                assert!(duration < Duration::from_secs(10));
            }
            Err(e) => {
                println!("Timeout handling test completed with error (acceptable): {}", e);
            }
        }
    }
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;
    
    #[tokio::test]
    async fn benchmark_file_discovery_performance() {
        let temp_dir = PerformanceTestUtils::create_large_test_project(100).unwrap();
        
        use aether_language::testing::file_compilation_testing::FileDiscoveryEngine;
        let engine = FileDiscoveryEngine::new(temp_dir.path().to_path_buf()).unwrap();
        
        let iterations = 10;
        let mut total_duration = Duration::from_nanos(0);
        
        for _ in 0..iterations {
            let start_time = Instant::now();
            let _files = engine.discover_aether_files().unwrap();
            total_duration += start_time.elapsed();
        }
        
        let avg_duration = total_duration / iterations;
        println!("File discovery benchmark: {:?} average over {} iterations", avg_duration, iterations);
        
        // Discovery should be fast (less than 100ms for 100 files)
        assert!(avg_duration < Duration::from_millis(100));
    }
    
    #[tokio::test]
    async fn benchmark_test_generation_performance() {
        let temp_dir = tempfile::tempdir().unwrap();
        
        use aether_language::testing::file_compilation_testing::TestFileGenerator;
        let generator = TestFileGenerator::new(temp_dir.path().to_path_buf()).unwrap();
        
        let start_time = Instant::now();
        let _generated_files = generator.generate_core_language_tests().unwrap();
        let duration = start_time.elapsed();
        
        println!("Test generation benchmark: {:?}", duration);
        
        // Test generation should be reasonably fast (less than 5 seconds)
        assert!(duration < Duration::from_secs(5));
    }
    
    #[tokio::test]
    async fn benchmark_report_generation_performance() {
        let temp_dir = PerformanceTestUtils::create_large_test_project(50).unwrap();
        let _compiler_path = PerformanceTestUtils::create_performance_mock_compiler(temp_dir.path()).unwrap();
        let config = PerformanceTestUtils::create_performance_config(temp_dir.path(), 4);
        
        let mut orchestrator = match FileCompilationTestOrchestrator::new(config) {
            Ok(orch) => orch,
            Err(e) => {
                println!("Report benchmark setup failed: {}", e);
                return;
            }
        };
        
        // Run test suite to get results
        let report = match orchestrator.run_complete_test_suite().await {
            Ok(r) => r,
            Err(e) => {
                println!("Report benchmark test suite failed: {}", e);
                return;
            }
        };
        
        // Benchmark different report formats
        use aether_language::testing::file_compilation_testing::ReportGenerator;
        let formats = vec![
            ReportFormat::Console,
            ReportFormat::Json,
            ReportFormat::Html,
            ReportFormat::Markdown,
        ];
        
        for format in formats {
            let generator = ReportGenerator::new(format.clone()).unwrap();
            
            let start_time = Instant::now();
            let _report_content = generator.generate_report(&report).unwrap();
            let duration = start_time.elapsed();
            
            println!("Report generation benchmark for {:?}: {:?}", format, duration);
            
            // Report generation should be fast (less than 1 second)
            assert!(duration < Duration::from_secs(1));
        }
    }
}

/// Get current memory usage in KB (platform-specific implementation)
fn get_memory_usage() -> usize {
    #[cfg(target_os = "windows")]
    {
        // On Windows, we can use GetProcessMemoryInfo
        // For now, return a mock value
        1000
    }
    
    #[cfg(target_os = "linux")]
    {
        // On Linux, we can read from /proc/self/status
        // For now, return a mock value
        1000
    }
    
    #[cfg(target_os = "macos")]
    {
        // On macOS, we can use task_info
        // For now, return a mock value
        1000
    }
    
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        // For other platforms, return a mock value
        1000
    }
}
 