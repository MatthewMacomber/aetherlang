// Aether Tensor Performance Tests
// Comprehensive performance testing and benchmarking for tensor operations

use aether_language::runtime::tensor::{Tensor, TensorDType, TensorDevice, TensorResult};
use aether_language::runtime::tensor_ops::{TensorOpsRegistry, OperationParams, OperationParam};
use aether_language::runtime::memory_layout::{MemoryLayoutOptimizer, AccessPattern, AccessType};
use std::time::{Instant, Duration};
use std::collections::HashMap;

/// Performance benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation: String,
    pub tensor_shapes: Vec<Vec<usize>>,
    pub dtype: TensorDType,
    pub duration: Duration,
    pub throughput_gflops: f64,
    pub memory_bandwidth_gbps: f64,
    pub cache_efficiency: f64,
}

/// Performance test suite for tensor operations
pub struct TensorPerformanceTests {
    ops_registry: TensorOpsRegistry,
    layout_optimizer: MemoryLayoutOptimizer,
    benchmark_results: Vec<BenchmarkResult>,
}

impl TensorPerformanceTests {
    /// Create new performance test suite
    pub fn new() -> Self {
        TensorPerformanceTests {
            ops_registry: TensorOpsRegistry::new(),
            layout_optimizer: MemoryLayoutOptimizer::new(),
            benchmark_results: Vec::new(),
        }
    }

    /// Run comprehensive tensor performance benchmarks
    pub fn run_all_benchmarks(&mut self) -> TensorResult<()> {
        println!("Running tensor performance benchmarks...");
        
        // Basic arithmetic operations
        self.benchmark_element_wise_operations()?;
        
        // Matrix operations
        self.benchmark_matrix_operations()?;
        
        // Memory layout optimizations
        self.benchmark_memory_layouts()?;
        
        // Large tensor operations
        self.benchmark_large_tensors()?;
        
        // GPU operations (simulated)
        self.benchmark_gpu_operations()?;
        
        self.print_benchmark_summary();
        Ok(())
    }

    /// Benchmark element-wise operations
    fn benchmark_element_wise_operations(&mut self) -> TensorResult<()> {
        let sizes = vec![
            vec![1000],
            vec![100, 100],
            vec![50, 50, 50],
            vec![10, 10, 10, 10],
        ];

        for shape in sizes {
            let tensor_a = Tensor::ones(shape.clone(), TensorDType::Float32)?;
            let tensor_b = Tensor::ones(shape.clone(), TensorDType::Float32)?;

            // Benchmark addition
            let start = Instant::now();
            let _result = tensor_a.add(&tensor_b)?;
            let duration = start.elapsed();
            
            let ops_count = tensor_a.size() as f64;
            let gflops = ops_count / duration.as_secs_f64() / 1e9;
            
            self.benchmark_results.push(BenchmarkResult {
                operation: "add".to_string(),
                tensor_shapes: vec![shape.clone(), shape.clone()],
                dtype: TensorDType::Float32,
                duration,
                throughput_gflops: gflops,
                memory_bandwidth_gbps: self.estimate_memory_bandwidth(&tensor_a, &duration),
                cache_efficiency: 0.8, // Estimated
            });

            // Benchmark multiplication
            let start = Instant::now();
            let _result = tensor_a.mul(&tensor_b)?;
            let duration = start.elapsed();
            
            let gflops = ops_count / duration.as_secs_f64() / 1e9;
            
            self.benchmark_results.push(BenchmarkResult {
                operation: "mul".to_string(),
                tensor_shapes: vec![shape.clone(), shape.clone()],
                dtype: TensorDType::Float32,
                duration,
                throughput_gflops: gflops,
                memory_bandwidth_gbps: self.estimate_memory_bandwidth(&tensor_a, &duration),
                cache_efficiency: 0.8,
            });
        }

        Ok(())
    }

    /// Benchmark matrix operations
    fn benchmark_matrix_operations(&mut self) -> TensorResult<()> {
        let matrix_sizes = vec![
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
        ];

        for (m, k, n) in matrix_sizes {
            let tensor_a = Tensor::ones(vec![m, k], TensorDType::Float32)?;
            let tensor_b = Tensor::ones(vec![k, n], TensorDType::Float32)?;

            // Benchmark matrix multiplication
            let start = Instant::now();
            let _result = tensor_a.matmul(&tensor_b)?;
            let duration = start.elapsed();
            
            // Matrix multiplication: 2*m*k*n operations
            let ops_count = 2.0 * m as f64 * k as f64 * n as f64;
            let gflops = ops_count / duration.as_secs_f64() / 1e9;
            
            self.benchmark_results.push(BenchmarkResult {
                operation: "matmul".to_string(),
                tensor_shapes: vec![vec![m, k], vec![k, n]],
                dtype: TensorDType::Float32,
                duration,
                throughput_gflops: gflops,
                memory_bandwidth_gbps: self.estimate_memory_bandwidth(&tensor_a, &duration),
                cache_efficiency: 0.6, // Matrix multiplication typically has lower cache efficiency
            });

            // Benchmark transpose
            let start = Instant::now();
            let _result = tensor_a.transpose()?;
            let duration = start.elapsed();
            
            let ops_count = tensor_a.size() as f64;
            let gflops = ops_count / duration.as_secs_f64() / 1e9;
            
            self.benchmark_results.push(BenchmarkResult {
                operation: "transpose".to_string(),
                tensor_shapes: vec![vec![m, k]],
                dtype: TensorDType::Float32,
                duration,
                throughput_gflops: gflops,
                memory_bandwidth_gbps: self.estimate_memory_bandwidth(&tensor_a, &duration),
                cache_efficiency: 0.4, // Transpose has poor cache locality
            });
        }

        Ok(())
    }

    /// Benchmark memory layout optimizations
    fn benchmark_memory_layouts(&mut self) -> TensorResult<()> {
        let tensor = Tensor::ones(vec![1000, 1000], TensorDType::Float32)?;
        
        // Test different access patterns
        let access_patterns = vec![
            AccessPattern {
                access_type: AccessType::Sequential { dimension: 0 },
                access_order: vec![0, 1],
                stride_patterns: vec![],
                access_frequency: HashMap::new(),
                temporal_locality: Default::default(),
            },
            AccessPattern {
                access_type: AccessType::Sequential { dimension: 1 },
                access_order: vec![1, 0],
                stride_patterns: vec![],
                access_frequency: HashMap::new(),
                temporal_locality: Default::default(),
            },
            AccessPattern {
                access_type: AccessType::Random,
                access_order: vec![0, 1],
                stride_patterns: vec![],
                access_frequency: HashMap::new(),
                temporal_locality: Default::default(),
            },
        ];

        for (i, pattern) in access_patterns.iter().enumerate() {
            let start = Instant::now();
            let _optimized = self.layout_optimizer.optimize_tensor_layout(&tensor, pattern)?;
            let duration = start.elapsed();
            
            self.benchmark_results.push(BenchmarkResult {
                operation: format!("layout_optimization_{}", i),
                tensor_shapes: vec![tensor.shape().to_vec()],
                dtype: tensor.dtype(),
                duration,
                throughput_gflops: 0.0, // Layout optimization doesn't have FLOPS
                memory_bandwidth_gbps: 0.0,
                cache_efficiency: 0.9, // Optimized layouts should have high cache efficiency
            });
        }

        Ok(())
    }

    /// Benchmark large tensor operations
    fn benchmark_large_tensors(&mut self) -> TensorResult<()> {
        // Test with tensors that don't fit in cache
        let large_shapes = vec![
            vec![2048, 2048],
            vec![4096, 1024],
            vec![1024, 1024, 4],
        ];

        for shape in large_shapes {
            let tensor_a = Tensor::ones(shape.clone(), TensorDType::Float32)?;
            let tensor_b = Tensor::ones(shape.clone(), TensorDType::Float32)?;

            // Test addition on large tensors
            let start = Instant::now();
            let _result = tensor_a.add(&tensor_b)?;
            let duration = start.elapsed();
            
            let ops_count = tensor_a.size() as f64;
            let gflops = ops_count / duration.as_secs_f64() / 1e9;
            
            self.benchmark_results.push(BenchmarkResult {
                operation: "large_add".to_string(),
                tensor_shapes: vec![shape.clone(), shape.clone()],
                dtype: TensorDType::Float32,
                duration,
                throughput_gflops: gflops,
                memory_bandwidth_gbps: self.estimate_memory_bandwidth(&tensor_a, &duration),
                cache_efficiency: 0.3, // Large tensors have poor cache efficiency
            });
        }

        Ok(())
    }

    /// Benchmark GPU operations (simulated)
    fn benchmark_gpu_operations(&mut self) -> TensorResult<()> {
        let tensor = Tensor::ones(vec![1024, 1024], TensorDType::Float32)?;
        
        // Simulate GPU transfer
        let start = Instant::now();
        let _gpu_tensor = tensor.to_device(TensorDevice::Gpu(0))?;
        let duration = start.elapsed();
        
        self.benchmark_results.push(BenchmarkResult {
            operation: "gpu_transfer".to_string(),
            tensor_shapes: vec![tensor.shape().to_vec()],
            dtype: tensor.dtype(),
            duration,
            throughput_gflops: 0.0,
            memory_bandwidth_gbps: self.estimate_gpu_transfer_bandwidth(&tensor, &duration),
            cache_efficiency: 1.0, // GPU transfers bypass cache
        });

        Ok(())
    }

    /// Estimate memory bandwidth utilization
    fn estimate_memory_bandwidth(&self, tensor: &Tensor, duration: &Duration) -> f64 {
        let bytes_accessed = tensor.size() * tensor.dtype().size_bytes() * 2; // Read + write
        let bandwidth_bps = bytes_accessed as f64 / duration.as_secs_f64();
        bandwidth_bps / 1e9 // Convert to GB/s
    }

    /// Estimate GPU transfer bandwidth
    fn estimate_gpu_transfer_bandwidth(&self, tensor: &Tensor, duration: &Duration) -> f64 {
        let bytes_transferred = tensor.size() * tensor.dtype().size_bytes();
        let bandwidth_bps = bytes_transferred as f64 / duration.as_secs_f64();
        bandwidth_bps / 1e9 // Convert to GB/s
    }

    /// Print benchmark summary
    fn print_benchmark_summary(&self) {
        println!("\n=== Tensor Performance Benchmark Results ===");
        println!("{:<20} {:<15} {:<12} {:<15} {:<15}", 
                 "Operation", "Shape", "Duration(ms)", "GFLOPS", "Bandwidth(GB/s)");
        println!("{}", "-".repeat(80));

        for result in &self.benchmark_results {
            let shape_str = if result.tensor_shapes.len() == 1 {
                format!("{:?}", result.tensor_shapes[0])
            } else {
                format!("{:?}x{:?}", result.tensor_shapes[0], result.tensor_shapes[1])
            };
            
            println!("{:<20} {:<15} {:<12.2} {:<15.2} {:<15.2}",
                     result.operation,
                     shape_str,
                     result.duration.as_secs_f64() * 1000.0,
                     result.throughput_gflops,
                     result.memory_bandwidth_gbps);
        }

        // Print summary statistics
        let total_operations = self.benchmark_results.len();
        let avg_gflops: f64 = self.benchmark_results.iter()
            .map(|r| r.throughput_gflops)
            .sum::<f64>() / total_operations as f64;
        let avg_bandwidth: f64 = self.benchmark_results.iter()
            .map(|r| r.memory_bandwidth_gbps)
            .sum::<f64>() / total_operations as f64;

        println!("\n=== Summary ===");
        println!("Total operations tested: {}", total_operations);
        println!("Average GFLOPS: {:.2}", avg_gflops);
        println!("Average memory bandwidth: {:.2} GB/s", avg_bandwidth);
    }

    /// Compare with reference implementations
    pub fn compare_with_reference(&mut self) -> TensorResult<()> {
        println!("\n=== Reference Implementation Comparison ===");
        
        // Compare matrix multiplication performance
        let sizes = vec![(128, 128, 128), (256, 256, 256), (512, 512, 512)];
        
        for (m, k, n) in sizes {
            let tensor_a = Tensor::ones(vec![m, k], TensorDType::Float32)?;
            let tensor_b = Tensor::ones(vec![k, n], TensorDType::Float32)?;

            // Our implementation
            let start = Instant::now();
            let _result = tensor_a.matmul(&tensor_b)?;
            let our_duration = start.elapsed();

            // Simulated reference (e.g., NumPy/BLAS performance)
            let reference_duration = self.simulate_reference_matmul_performance(m, k, n);
            
            let speedup = reference_duration.as_secs_f64() / our_duration.as_secs_f64();
            
            println!("MatMul {}x{}x{}: Our={:.2}ms, Ref={:.2}ms, Speedup={:.2}x",
                     m, k, n,
                     our_duration.as_secs_f64() * 1000.0,
                     reference_duration.as_secs_f64() * 1000.0,
                     speedup);
        }

        Ok(())
    }

    /// Simulate reference implementation performance
    fn simulate_reference_matmul_performance(&self, m: usize, k: usize, n: usize) -> Duration {
        // Simulate optimized BLAS performance
        // This is a rough estimate based on typical BLAS performance
        let ops = 2.0 * m as f64 * k as f64 * n as f64;
        let estimated_gflops = 50.0; // Assume 50 GFLOPS for reference BLAS
        let estimated_seconds = ops / (estimated_gflops * 1e9);
        Duration::from_secs_f64(estimated_seconds)
    }

    /// Get benchmark results
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.benchmark_results
    }

    /// Clear benchmark results
    pub fn clear_results(&mut self) {
        self.benchmark_results.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation_performance() {
        let mut test_suite = TensorPerformanceTests::new();
        
        // Test tensor creation performance
        let shapes = vec![
            vec![100, 100],
            vec![500, 500],
            vec![1000, 1000],
        ];

        for shape in shapes {
            let start = Instant::now();
            let _tensor = Tensor::zeros(shape.clone(), TensorDType::Float32).unwrap();
            let duration = start.elapsed();
            
            println!("Created tensor {:?} in {:.2}ms", shape, duration.as_secs_f64() * 1000.0);
            assert!(duration.as_millis() < 1000, "Tensor creation took too long");
        }
    }

    #[test]
    fn test_element_wise_operations_performance() {
        let tensor_a = Tensor::ones(vec![1000, 1000], TensorDType::Float32).unwrap();
        let tensor_b = Tensor::ones(vec![1000, 1000], TensorDType::Float32).unwrap();

        // Test addition performance
        let start = Instant::now();
        let _result = tensor_a.add(&tensor_b).unwrap();
        let add_duration = start.elapsed();
        
        // Test multiplication performance
        let start = Instant::now();
        let _result = tensor_a.mul(&tensor_b).unwrap();
        let mul_duration = start.elapsed();

        println!("Addition: {:.2}ms, Multiplication: {:.2}ms", 
                 add_duration.as_secs_f64() * 1000.0,
                 mul_duration.as_secs_f64() * 1000.0);

        // Performance should be reasonable (less than 100ms for 1M elements)
        assert!(add_duration.as_millis() < 100);
        assert!(mul_duration.as_millis() < 100);
    }

    #[test]
    fn test_matrix_multiplication_performance() {
        let sizes = vec![(64, 64, 64), (128, 128, 128), (256, 256, 256)];

        for (m, k, n) in sizes {
            let tensor_a = Tensor::ones(vec![m, k], TensorDType::Float32).unwrap();
            let tensor_b = Tensor::ones(vec![k, n], TensorDType::Float32).unwrap();

            let start = Instant::now();
            let _result = tensor_a.matmul(&tensor_b).unwrap();
            let duration = start.elapsed();

            let ops = 2.0 * m as f64 * k as f64 * n as f64;
            let gflops = ops / duration.as_secs_f64() / 1e9;

            println!("MatMul {}x{}x{}: {:.2}ms, {:.2} GFLOPS", 
                     m, k, n, duration.as_secs_f64() * 1000.0, gflops);

            // Should complete in reasonable time
            assert!(duration.as_secs() < 10, "Matrix multiplication took too long");
        }
    }

    #[test]
    fn test_memory_layout_optimization() {
        let tensor = Tensor::ones(vec![100, 100], TensorDType::Float32).unwrap();
        let optimizer = MemoryLayoutOptimizer::new();
        
        let access_pattern = AccessPattern {
            access_type: AccessType::Sequential { dimension: 0 },
            access_order: vec![0, 1],
            stride_patterns: vec![],
            access_frequency: HashMap::new(),
            temporal_locality: Default::default(),
        };

        let start = Instant::now();
        let _optimized = optimizer.optimize_tensor_layout(&tensor, &access_pattern).unwrap();
        let duration = start.elapsed();

        println!("Layout optimization: {:.2}ms", duration.as_secs_f64() * 1000.0);
        assert!(duration.as_millis() < 100, "Layout optimization took too long");
    }

    #[test]
    fn test_tensor_reshape_performance() {
        let tensor = Tensor::ones(vec![1000, 1000], TensorDType::Float32).unwrap();
        
        let start = Instant::now();
        let _reshaped = tensor.reshape(vec![100, 10000]).unwrap();
        let duration = start.elapsed();

        println!("Reshape: {:.2}ms", duration.as_secs_f64() * 1000.0);
        assert!(duration.as_millis() < 10, "Reshape should be very fast (metadata only)");
    }

    #[test]
    fn test_tensor_transpose_performance() {
        let sizes = vec![
            vec![100, 100],
            vec![500, 200],
            vec![1000, 100],
        ];

        for shape in sizes {
            let tensor = Tensor::ones(shape.clone(), TensorDType::Float32).unwrap();
            
            let start = Instant::now();
            let _transposed = tensor.transpose().unwrap();
            let duration = start.elapsed();

            println!("Transpose {:?}: {:.2}ms", shape, duration.as_secs_f64() * 1000.0);
            assert!(duration.as_millis() < 500, "Transpose took too long");
        }
    }

    #[test]
    fn test_gpu_transfer_simulation() {
        let tensor = Tensor::ones(vec![1000, 1000], TensorDType::Float32).unwrap();
        
        let start = Instant::now();
        let _gpu_tensor = tensor.to_device(TensorDevice::Gpu(0)).unwrap();
        let duration = start.elapsed();

        println!("GPU transfer simulation: {:.2}ms", duration.as_secs_f64() * 1000.0);
        assert!(duration.as_millis() < 100, "GPU transfer simulation took too long");
    }

    #[test]
    fn test_comprehensive_benchmark_suite() {
        let mut test_suite = TensorPerformanceTests::new();
        
        // Run a subset of benchmarks for testing
        test_suite.benchmark_element_wise_operations().unwrap();
        test_suite.benchmark_matrix_operations().unwrap();
        
        let results = test_suite.get_results();
        assert!(!results.is_empty(), "Should have benchmark results");
        
        // Check that we have reasonable performance metrics
        for result in results {
            assert!(result.duration.as_secs() < 10, "Operation took too long");
            if result.throughput_gflops > 0.0 {
                assert!(result.throughput_gflops < 1000.0, "GFLOPS seems unrealistic");
            }
        }
    }

    #[test]
    fn test_different_data_types_performance() {
        let shape = vec![500, 500];
        let dtypes = vec![
            TensorDType::Float32,
            TensorDType::Float64,
            TensorDType::Int32,
        ];

        for dtype in dtypes {
            let tensor_a = Tensor::ones(shape.clone(), dtype).unwrap();
            let tensor_b = Tensor::ones(shape.clone(), dtype).unwrap();

            let start = Instant::now();
            let _result = tensor_a.add(&tensor_b).unwrap();
            let duration = start.elapsed();

            println!("Addition {:?}: {:.2}ms", dtype, duration.as_secs_f64() * 1000.0);
            assert!(duration.as_millis() < 200, "Addition took too long for dtype {:?}", dtype);
        }
    }

    #[test]
    fn test_broadcasting_performance() {
        // Test broadcasting with different shapes
        let test_cases = vec![
            (vec![1000, 1], vec![1000, 1000]),
            (vec![1], vec![1000, 1000]),
            (vec![1000], vec![1000, 1000]),
        ];

        for (shape1, shape2) in test_cases {
            let tensor_a = Tensor::ones(shape1.clone(), TensorDType::Float32).unwrap();
            let tensor_b = Tensor::ones(shape2.clone(), TensorDType::Float32).unwrap();

            let start = Instant::now();
            let _result = tensor_a.add(&tensor_b).unwrap();
            let duration = start.elapsed();

            println!("Broadcasting {:?} + {:?}: {:.2}ms", 
                     shape1, shape2, duration.as_secs_f64() * 1000.0);
            assert!(duration.as_millis() < 500, "Broadcasting took too long");
        }
    }
}