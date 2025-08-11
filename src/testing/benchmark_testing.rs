// Benchmark Testing Framework
// Performance benchmarking and regression testing infrastructure

use super::{TestCase, TestResult, TestContext};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub min_execution_time: Duration,
    pub max_execution_time: Duration,
    pub confidence_level: f64,
    pub significance_threshold: f64,
    pub regression_threshold: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            min_execution_time: Duration::from_millis(1),
            max_execution_time: Duration::from_secs(10),
            confidence_level: 0.95,
            significance_threshold: 0.05,
            regression_threshold: 0.1, // 10% regression threshold
        }
    }
}

/// Benchmark measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    pub name: String,
    pub iterations: usize,
    pub total_time: Duration,
    pub mean_time: Duration,
    pub median_time: Duration,
    pub std_dev: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub throughput: Option<f64>, // Operations per second
    pub memory_usage: Option<usize>, // Bytes
    pub custom_metrics: HashMap<String, f64>,
}

impl BenchmarkMeasurement {
    pub fn from_samples(name: String, samples: Vec<Duration>) -> Self {
        let iterations = samples.len();
        let total_time: Duration = samples.iter().sum();
        let mean_time = total_time / iterations as u32;
        
        let mut sorted_samples = samples.clone();
        sorted_samples.sort();
        let median_time = sorted_samples[iterations / 2];
        
        let min_time = *sorted_samples.first().unwrap();
        let max_time = *sorted_samples.last().unwrap();
        
        // Calculate standard deviation
        let mean_nanos = mean_time.as_nanos() as f64;
        let variance: f64 = samples.iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>() / iterations as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);
        
        BenchmarkMeasurement {
            name,
            iterations,
            total_time,
            mean_time,
            median_time,
            std_dev,
            min_time,
            max_time,
            throughput: None,
            memory_usage: None,
            custom_metrics: HashMap::new(),
        }
    }

    pub fn with_throughput(mut self, ops_per_second: f64) -> Self {
        self.throughput = Some(ops_per_second);
        self
    }

    pub fn with_memory_usage(mut self, bytes: usize) -> Self {
        self.memory_usage = Some(bytes);
        self
    }

    pub fn with_custom_metric(mut self, name: String, value: f64) -> Self {
        self.custom_metrics.insert(name, value);
        self
    }

    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean_time.as_nanos() == 0 {
            0.0
        } else {
            self.std_dev.as_nanos() as f64 / self.mean_time.as_nanos() as f64
        }
    }
}

/// Benchmark test case
pub struct BenchmarkTest {
    pub name: String,
    pub benchmark_fn: Box<dyn Fn() -> Result<(), String> + Send + Sync>,
    pub config: BenchmarkConfig,
    pub setup: Option<Box<dyn Fn() -> Result<(), String> + Send + Sync>>,
    pub teardown: Option<Box<dyn Fn() -> Result<(), String> + Send + Sync>>,
    pub baseline: Option<BenchmarkMeasurement>,
}

impl BenchmarkTest {
    pub fn new<F>(name: &str, benchmark_fn: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        BenchmarkTest {
            name: name.to_string(),
            benchmark_fn: Box::new(benchmark_fn),
            config: BenchmarkConfig::default(),
            setup: None,
            teardown: None,
            baseline: None,
        }
    }

    pub fn with_config(mut self, config: BenchmarkConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_setup<F>(mut self, setup: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        self.setup = Some(Box::new(setup));
        self
    }

    pub fn with_teardown<F>(mut self, teardown: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        self.teardown = Some(Box::new(teardown));
        self
    }

    pub fn with_baseline(mut self, baseline: BenchmarkMeasurement) -> Self {
        self.baseline = Some(baseline);
        self
    }

    pub fn run_benchmark(&self) -> Result<BenchmarkMeasurement, String> {
        // Setup
        if let Some(setup) = &self.setup {
            setup()?;
        }

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            (self.benchmark_fn)()?;
        }

        // Measurement
        let mut samples = Vec::new();
        let measurement_start = Instant::now();

        for _ in 0..self.config.measurement_iterations {
            let start = Instant::now();
            (self.benchmark_fn)()?;
            let duration = start.elapsed();
            
            // Check bounds
            if duration < self.config.min_execution_time {
                return Err(format!("Benchmark execution too fast: {:?} < {:?}", 
                                 duration, self.config.min_execution_time));
            }
            if duration > self.config.max_execution_time {
                return Err(format!("Benchmark execution too slow: {:?} > {:?}", 
                                 duration, self.config.max_execution_time));
            }
            
            samples.push(duration);
            
            // Early termination if we've been running too long
            if measurement_start.elapsed() > Duration::from_secs(60) {
                break;
            }
        }

        // Teardown
        if let Some(teardown) = &self.teardown {
            teardown()?;
        }

        let measurement = BenchmarkMeasurement::from_samples(self.name.clone(), samples);
        Ok(measurement)
    }

    pub fn compare_with_baseline(&self, measurement: &BenchmarkMeasurement) -> Option<BenchmarkComparison> {
        if let Some(baseline) = &self.baseline {
            Some(BenchmarkComparison::new(baseline, measurement))
        } else {
            None
        }
    }
}

impl TestCase for BenchmarkTest {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _context: &mut TestContext) -> TestResult {
        let start = Instant::now();
        
        match self.run_benchmark() {
            Ok(measurement) => {
                let duration = start.elapsed();
                
                // Check for regression if baseline exists
                if let Some(comparison) = self.compare_with_baseline(&measurement) {
                    if comparison.is_regression(self.config.regression_threshold) {
                        return TestResult::failed(
                            self.name.clone(),
                            duration,
                            format!("Performance regression detected: {:.2}% slower than baseline", 
                                   comparison.relative_change * 100.0)
                        )
                        .with_metadata("mean_time".to_string(), format!("{:?}", measurement.mean_time))
                        .with_metadata("baseline_time".to_string(), format!("{:?}", comparison.baseline.mean_time))
                        .with_metadata("regression".to_string(), format!("{:.2}%", comparison.relative_change * 100.0));
                    }
                }
                
                TestResult::passed(self.name.clone(), duration)
                    .with_metadata("mean_time".to_string(), format!("{:?}", measurement.mean_time))
                    .with_metadata("iterations".to_string(), measurement.iterations.to_string())
                    .with_metadata("coefficient_of_variation".to_string(), format!("{:.4}", measurement.coefficient_of_variation()))
            }
            Err(error) => {
                TestResult::error(self.name.clone(), start.elapsed(), error)
            }
        }
    }

    fn timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(120)) // Benchmarks can take longer
    }
}

/// Benchmark comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub baseline: BenchmarkMeasurement,
    pub current: BenchmarkMeasurement,
    pub relative_change: f64, // Positive means slower, negative means faster
    pub absolute_change: Duration,
    pub is_significant: bool,
}

impl BenchmarkComparison {
    pub fn new(baseline: &BenchmarkMeasurement, current: &BenchmarkMeasurement) -> Self {
        let baseline_nanos = baseline.mean_time.as_nanos() as f64;
        let current_nanos = current.mean_time.as_nanos() as f64;
        
        let relative_change = (current_nanos - baseline_nanos) / baseline_nanos;
        let absolute_change = if current.mean_time > baseline.mean_time {
            current.mean_time - baseline.mean_time
        } else {
            baseline.mean_time - current.mean_time
        };
        
        // Simple significance test (could be improved with proper statistical tests)
        let is_significant = relative_change.abs() > 0.05; // 5% threshold
        
        BenchmarkComparison {
            baseline: baseline.clone(),
            current: current.clone(),
            relative_change,
            absolute_change,
            is_significant,
        }
    }

    pub fn is_regression(&self, threshold: f64) -> bool {
        self.relative_change > threshold && self.is_significant
    }

    pub fn is_improvement(&self, threshold: f64) -> bool {
        self.relative_change < -threshold && self.is_significant
    }
}

/// Benchmark suite for organizing related benchmarks
pub struct BenchmarkSuite {
    pub name: String,
    pub benchmarks: Vec<BenchmarkTest>,
    pub common_config: BenchmarkConfig,
}

impl BenchmarkSuite {
    pub fn new(name: &str) -> Self {
        BenchmarkSuite {
            name: name.to_string(),
            benchmarks: Vec::new(),
            common_config: BenchmarkConfig::default(),
        }
    }

    pub fn with_config(mut self, config: BenchmarkConfig) -> Self {
        self.common_config = config;
        self
    }

    pub fn add_benchmark(mut self, mut benchmark: BenchmarkTest) -> Self {
        // Apply common config if benchmark doesn't have custom config
        if benchmark.config.measurement_iterations == BenchmarkConfig::default().measurement_iterations {
            benchmark.config = self.common_config.clone();
        }
        
        self.benchmarks.push(benchmark);
        self
    }

    pub fn run_all(&self) -> Vec<BenchmarkMeasurement> {
        self.benchmarks.iter()
            .filter_map(|b| b.run_benchmark().ok())
            .collect()
    }
}

/// Predefined benchmark scenarios
pub struct BenchmarkScenarios;

impl BenchmarkScenarios {
    /// Benchmark tensor operations
    pub fn tensor_operations() -> BenchmarkSuite {
        BenchmarkSuite::new("tensor_operations")
            .with_config(BenchmarkConfig {
                measurement_iterations: 50,
                warmup_iterations: 5,
                ..BenchmarkConfig::default()
            })
            .add_benchmark(BenchmarkTest::new("tensor_add", || {
                // Simulate tensor addition
                let mut sum = 0.0;
                for i in 0..1000 {
                    sum += i as f64 * 1.5;
                }
                if sum > 0.0 { Ok(()) } else { Err("Invalid result".to_string()) }
            }))
            .add_benchmark(BenchmarkTest::new("tensor_multiply", || {
                // Simulate tensor multiplication
                let mut product = 1.0;
                for i in 1..100 {
                    product *= (i as f64).sqrt();
                }
                if product > 0.0 { Ok(()) } else { Err("Invalid result".to_string()) }
            }))
            .add_benchmark(BenchmarkTest::new("matrix_multiplication", || {
                // Simulate matrix multiplication
                let size = 50;
                let mut result = 0.0;
                for i in 0..size {
                    for j in 0..size {
                        for k in 0..size {
                            result += (i * j * k) as f64;
                        }
                    }
                }
                if result > 0.0 { Ok(()) } else { Err("Invalid result".to_string()) }
            }))
    }

    /// Benchmark compilation performance
    pub fn compilation_performance() -> BenchmarkSuite {
        BenchmarkSuite::new("compilation_performance")
            .add_benchmark(BenchmarkTest::new("parse_simple_expression", || {
                // Simulate parsing
                let _tokens = vec!["let", "x", "=", "5", "+", "3"];
                std::thread::sleep(Duration::from_micros(10));
                Ok(())
            }))
            .add_benchmark(BenchmarkTest::new("type_checking", || {
                // Simulate type checking
                let _types = vec!["i32", "f64", "bool", "string"];
                std::thread::sleep(Duration::from_micros(20));
                Ok(())
            }))
            .add_benchmark(BenchmarkTest::new("code_generation", || {
                // Simulate code generation
                std::thread::sleep(Duration::from_micros(50));
                Ok(())
            }))
    }

    /// Benchmark memory operations
    pub fn memory_operations() -> BenchmarkSuite {
        BenchmarkSuite::new("memory_operations")
            .add_benchmark(BenchmarkTest::new("allocation", || {
                let _vec: Vec<i32> = (0..1000).collect();
                Ok(())
            }))
            .add_benchmark(BenchmarkTest::new("deallocation", || {
                let vec: Vec<i32> = (0..1000).collect();
                drop(vec);
                Ok(())
            }))
            .add_benchmark(BenchmarkTest::new("copy", || {
                let vec: Vec<i32> = (0..1000).collect();
                let _copy = vec.clone();
                Ok(())
            }))
    }

    /// Benchmark automatic differentiation
    pub fn autodiff_performance() -> BenchmarkSuite {
        BenchmarkSuite::new("autodiff_performance")
            .add_benchmark(BenchmarkTest::new("forward_mode", || {
                // Simulate forward mode AD
                let mut result = 1.0;
                for i in 1..100 {
                    result = result * (i as f64) + (i as f64).sin();
                }
                if result.is_finite() { Ok(()) } else { Err("Invalid result".to_string()) }
            }))
            .add_benchmark(BenchmarkTest::new("reverse_mode", || {
                // Simulate reverse mode AD
                let mut gradients = vec![0.0; 100];
                for i in 0..100 {
                    gradients[i] = (i as f64).cos();
                }
                if gradients.iter().all(|g| g.is_finite()) { Ok(()) } else { Err("Invalid gradients".to_string()) }
            }))
    }

    /// Create comprehensive benchmark suite
    pub fn comprehensive_suite() -> Vec<BenchmarkSuite> {
        vec![
            Self::tensor_operations(),
            Self::compilation_performance(),
            Self::memory_operations(),
            Self::autodiff_performance(),
        ]
    }
}

/// Benchmark reporting utilities
pub struct BenchmarkReporter;

impl BenchmarkReporter {
    pub fn generate_report(measurements: &[BenchmarkMeasurement]) -> String {
        let mut report = String::new();
        report.push_str("# Benchmark Report\n\n");
        
        for measurement in measurements {
            report.push_str(&format!("## {}\n", measurement.name));
            report.push_str(&format!("- **Iterations**: {}\n", measurement.iterations));
            report.push_str(&format!("- **Mean Time**: {:?}\n", measurement.mean_time));
            report.push_str(&format!("- **Median Time**: {:?}\n", measurement.median_time));
            report.push_str(&format!("- **Std Dev**: {:?}\n", measurement.std_dev));
            report.push_str(&format!("- **Min Time**: {:?}\n", measurement.min_time));
            report.push_str(&format!("- **Max Time**: {:?}\n", measurement.max_time));
            report.push_str(&format!("- **Coefficient of Variation**: {:.4}\n", measurement.coefficient_of_variation()));
            
            if let Some(throughput) = measurement.throughput {
                report.push_str(&format!("- **Throughput**: {:.2} ops/sec\n", throughput));
            }
            
            if let Some(memory) = measurement.memory_usage {
                report.push_str(&format!("- **Memory Usage**: {} bytes\n", memory));
            }
            
            if !measurement.custom_metrics.is_empty() {
                report.push_str("- **Custom Metrics**:\n");
                for (name, value) in &measurement.custom_metrics {
                    report.push_str(&format!("  - {}: {:.4}\n", name, value));
                }
            }
            
            report.push_str("\n");
        }
        
        report
    }

    pub fn save_baseline(measurement: &BenchmarkMeasurement, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(measurement)
            .map_err(|e| format!("Failed to serialize measurement: {}", e))?;
        
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write baseline file: {}", e))?;
        
        Ok(())
    }

    pub fn load_baseline(path: &str) -> Result<BenchmarkMeasurement, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read baseline file: {}", e))?;
        
        serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize measurement: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.measurement_iterations, 100);
        assert_eq!(config.confidence_level, 0.95);
    }

    #[test]
    fn test_benchmark_measurement_creation() {
        let samples = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(13),
            Duration::from_millis(9),
        ];
        
        let measurement = BenchmarkMeasurement::from_samples("test".to_string(), samples);
        
        assert_eq!(measurement.name, "test");
        assert_eq!(measurement.iterations, 5);
        assert_eq!(measurement.min_time, Duration::from_millis(9));
        assert_eq!(measurement.max_time, Duration::from_millis(13));
        assert!(measurement.coefficient_of_variation() > 0.0);
    }

    #[test]
    fn test_benchmark_comparison() {
        let baseline = BenchmarkMeasurement::from_samples(
            "baseline".to_string(),
            vec![Duration::from_millis(10); 5]
        );
        
        let current = BenchmarkMeasurement::from_samples(
            "current".to_string(),
            vec![Duration::from_millis(12); 5]
        );
        
        let comparison = BenchmarkComparison::new(&baseline, &current);
        
        assert!(comparison.relative_change > 0.0); // Current is slower
        assert!(comparison.is_regression(0.1)); // 20% slower is a regression
        assert!(!comparison.is_improvement(0.1));
    }

    #[test]
    fn test_benchmark_test_creation() {
        let test = BenchmarkTest::new("test_benchmark", || {
            std::thread::sleep(Duration::from_micros(1));
            Ok(())
        });
        
        assert_eq!(test.name, "test_benchmark");
        assert!(test.setup.is_none());
        assert!(test.teardown.is_none());
        assert!(test.baseline.is_none());
    }

    #[test]
    fn test_benchmark_suite() {
        let suite = BenchmarkSuite::new("test_suite")
            .add_benchmark(BenchmarkTest::new("bench1", || Ok(())))
            .add_benchmark(BenchmarkTest::new("bench2", || Ok(())));
        
        assert_eq!(suite.name, "test_suite");
        assert_eq!(suite.benchmarks.len(), 2);
    }

    #[test]
    fn test_predefined_scenarios() {
        let tensor_suite = BenchmarkScenarios::tensor_operations();
        assert_eq!(tensor_suite.name, "tensor_operations");
        assert_eq!(tensor_suite.benchmarks.len(), 3);
        
        let compilation_suite = BenchmarkScenarios::compilation_performance();
        assert_eq!(compilation_suite.name, "compilation_performance");
        assert_eq!(compilation_suite.benchmarks.len(), 3);
        
        let comprehensive = BenchmarkScenarios::comprehensive_suite();
        assert_eq!(comprehensive.len(), 4);
    }
}