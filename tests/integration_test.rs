use aether_language::runtime::tensor::{Tensor, TensorDType};
use aether_language::runtime::tensor_ops::TensorOpsRegistry;
use aether_language::runtime::memory_layout::MemoryLayoutOptimizer;

#[test]
fn test_tensor_basic_operations() {
    // Test tensor creation
    let tensor_a = Tensor::ones(vec![2, 2], TensorDType::Float32).unwrap();
    let tensor_b = Tensor::ones(vec![2, 2], TensorDType::Float32).unwrap();
    
    // Test addition
    let result = tensor_a.add(&tensor_b).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    
    // Test multiplication
    let result = tensor_a.mul(&tensor_b).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    
    // Test matrix multiplication
    let result = tensor_a.matmul(&tensor_b).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_tensor_ops_registry() {
    let registry = TensorOpsRegistry::new();
    let ops = registry.available_operations();
    
    assert!(ops.contains(&"add"));
    assert!(ops.contains(&"mul"));
    assert!(ops.contains(&"matmul"));
}

#[test]
fn test_memory_layout_optimizer() {
    let optimizer = MemoryLayoutOptimizer::new();
    let tensor = Tensor::ones(vec![10, 10], TensorDType::Float32).unwrap();
    
    // Test that optimizer can be created and used
    assert!(optimizer.cache_info().l1_size > 0);
}

#[test]
fn test_tensor_performance_basic() {
    use std::time::Instant;
    
    let tensor_a = Tensor::ones(vec![100, 100], TensorDType::Float32).unwrap();
    let tensor_b = Tensor::ones(vec![100, 100], TensorDType::Float32).unwrap();
    
    let start = Instant::now();
    let _result = tensor_a.add(&tensor_b).unwrap();
    let duration = start.elapsed();
    
    // Should complete in reasonable time (less than 100ms)
    assert!(duration.as_millis() < 100);
}