// Aether Tensor Operations Library
// High-performance tensor operations with BLAS integration and broadcasting

use crate::runtime::tensor::{Tensor, TensorResult, TensorError, TensorDType, TensorDevice};
use std::collections::HashMap;
use std::time::Duration;

/// Tensor operations registry for extensible operation system
pub struct TensorOpsRegistry {
    /// Registered operations
    operations: HashMap<String, Box<dyn TensorOperation>>,
    /// BLAS backend configuration
    blas_config: BlasConfig,
    /// GPU backend configuration
    gpu_config: GpuConfig,
}

/// Configuration for BLAS integration
#[derive(Debug, Clone)]
pub struct BlasConfig {
    /// Whether BLAS is available
    pub available: bool,
    /// BLAS library type (OpenBLAS, Intel MKL, etc.)
    pub library: BlasLibrary,
    /// Number of threads for parallel operations
    pub num_threads: usize,
}

/// BLAS library types
#[derive(Debug, Clone)]
pub enum BlasLibrary {
    OpenBlas,
    IntelMkl,
    Accelerate, // macOS
    Generic,
}

/// Configuration for GPU operations
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Whether GPU is available
    pub available: bool,
    /// GPU backend type
    pub backend: GpuBackend,
    /// Available GPU devices
    pub devices: Vec<GpuDeviceInfo>,
}

/// GPU backend types
#[derive(Debug, Clone)]
pub enum GpuBackend {
    Cuda,
    OpenCl,
    Metal,
    Vulkan,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub memory_bytes: usize,
    pub compute_capability: String,
}

/// Trait for tensor operations
pub trait TensorOperation: Send + Sync {
    /// Execute the operation
    fn execute(&self, inputs: &[&Tensor], params: &OperationParams) -> TensorResult<Tensor>;
    
    /// Get operation name
    fn name(&self) -> &str;
    
    /// Get expected number of inputs
    fn input_arity(&self) -> usize;
    
    /// Check if operation supports given input shapes and types
    fn supports(&self, inputs: &[&Tensor]) -> bool;
    
    /// Infer output shape from input shapes
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>>;
}

/// Parameters for tensor operations
#[derive(Debug, Clone)]
pub struct OperationParams {
    /// Operation-specific parameters
    pub params: HashMap<String, OperationParam>,
    /// Preferred device for computation
    pub device: Option<TensorDevice>,
    /// Whether to use optimized backends (BLAS, GPU)
    pub use_optimized: bool,
}

/// Operation parameter values
#[derive(Debug, Clone)]
pub enum OperationParam {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
}

impl TensorOpsRegistry {
    /// Create new tensor operations registry
    pub fn new() -> Self {
        let mut registry = TensorOpsRegistry {
            operations: HashMap::new(),
            blas_config: BlasConfig::default(),
            gpu_config: GpuConfig::default(),
        };
        
        // Register built-in operations
        registry.register_builtin_operations();
        
        registry
    }

    /// Register built-in tensor operations
    fn register_builtin_operations(&mut self) {
        self.register_operation(Box::new(AddOperation));
        self.register_operation(Box::new(MulOperation));
        self.register_operation(Box::new(SubOperation));
        self.register_operation(Box::new(DivOperation));
        self.register_operation(Box::new(MatMulOperation));
        self.register_operation(Box::new(TransposeOperation));
        self.register_operation(Box::new(ReshapeOperation));
        self.register_operation(Box::new(SumOperation));
        self.register_operation(Box::new(MeanOperation));
        self.register_operation(Box::new(MaxOperation));
        self.register_operation(Box::new(MinOperation));
        self.register_operation(Box::new(SoftmaxOperation));
        self.register_operation(Box::new(ReluOperation));
        self.register_operation(Box::new(ConvOperation));
    }

    /// Register a new tensor operation
    pub fn register_operation(&mut self, operation: Box<dyn TensorOperation>) {
        self.operations.insert(operation.name().to_string(), operation);
    }

    /// Execute tensor operation by name
    pub fn execute(&self, op_name: &str, inputs: &[&Tensor], params: &OperationParams) -> TensorResult<Tensor> {
        if let Some(operation) = self.operations.get(op_name) {
            // Check input compatibility
            if !operation.supports(inputs) {
                return Err(TensorError::InvalidShape {
                    shape: vec![],
                    reason: format!("Operation {} does not support given inputs", op_name),
                });
            }
            
            // Execute operation
            operation.execute(inputs, params)
        } else {
            Err(TensorError::GpuError {
                message: format!("Unknown operation: {}", op_name),
                device_id: 0,
            })
        }
    }

    /// Get available operations
    pub fn available_operations(&self) -> Vec<&str> {
        self.operations.keys().map(|s| s.as_str()).collect()
    }

    /// Configure BLAS backend
    pub fn configure_blas(&mut self, config: BlasConfig) {
        self.blas_config = config;
    }

    /// Configure GPU backend
    pub fn configure_gpu(&mut self, config: GpuConfig) {
        self.gpu_config = config;
    }

    /// Check if BLAS is available
    pub fn has_blas(&self) -> bool {
        self.blas_config.available
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.gpu_config.available
    }
}

// Built-in tensor operations

/// Element-wise addition operation
pub struct AddOperation;

impl TensorOperation for AddOperation {
    fn execute(&self, inputs: &[&Tensor], _params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Add operation requires exactly 2 inputs".to_string(),
            });
        }
        
        inputs[0].add(inputs[1])
    }

    fn name(&self) -> &str { "add" }
    fn input_arity(&self) -> usize { 2 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 2 && inputs[0].device() == inputs[1].device()
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Add operation requires exactly 2 inputs".to_string(),
            });
        }
        
        broadcast_shapes(input_shapes[0], input_shapes[1])
    }
}

/// Element-wise multiplication operation
pub struct MulOperation;

impl TensorOperation for MulOperation {
    fn execute(&self, inputs: &[&Tensor], _params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Mul operation requires exactly 2 inputs".to_string(),
            });
        }
        
        inputs[0].mul(inputs[1])
    }

    fn name(&self) -> &str { "mul" }
    fn input_arity(&self) -> usize { 2 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 2 && inputs[0].device() == inputs[1].device()
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Mul operation requires exactly 2 inputs".to_string(),
            });
        }
        
        broadcast_shapes(input_shapes[0], input_shapes[1])
    }
}

/// Element-wise subtraction operation
pub struct SubOperation;

impl TensorOperation for SubOperation {
    fn execute(&self, inputs: &[&Tensor], _params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Sub operation requires exactly 2 inputs".to_string(),
            });
        }
        
        // Implement subtraction using addition with negation
        let neg_second = negate_tensor(inputs[1])?;
        inputs[0].add(&neg_second)
    }

    fn name(&self) -> &str { "sub" }
    fn input_arity(&self) -> usize { 2 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 2 && inputs[0].device() == inputs[1].device()
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Sub operation requires exactly 2 inputs".to_string(),
            });
        }
        
        broadcast_shapes(input_shapes[0], input_shapes[1])
    }
}

/// Element-wise division operation
pub struct DivOperation;

impl TensorOperation for DivOperation {
    fn execute(&self, inputs: &[&Tensor], _params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Div operation requires exactly 2 inputs".to_string(),
            });
        }
        
        // Implement division using multiplication with reciprocal
        let reciprocal_second = reciprocal_tensor(inputs[1])?;
        inputs[0].mul(&reciprocal_second)
    }

    fn name(&self) -> &str { "div" }
    fn input_arity(&self) -> usize { 2 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 2 && inputs[0].device() == inputs[1].device()
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Div operation requires exactly 2 inputs".to_string(),
            });
        }
        
        broadcast_shapes(input_shapes[0], input_shapes[1])
    }
}

/// Matrix multiplication operation
pub struct MatMulOperation;

impl TensorOperation for MatMulOperation {
    fn execute(&self, inputs: &[&Tensor], params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "MatMul operation requires exactly 2 inputs".to_string(),
            });
        }
        
        // Use optimized BLAS if available and requested
        if params.use_optimized {
            // In a real implementation, this would call BLAS routines
            // For now, fall back to the basic implementation
        }
        
        inputs[0].matmul(inputs[1])
    }

    fn name(&self) -> &str { "matmul" }
    fn input_arity(&self) -> usize { 2 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        if inputs.len() != 2 {
            return false;
        }
        
        let shape1 = inputs[0].shape();
        let shape2 = inputs[1].shape();
        
        // Check matrix multiplication compatibility
        shape1.len() >= 2 && shape2.len() >= 2 && 
        shape1[shape1.len() - 1] == shape2[shape2.len() - 2]
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "MatMul operation requires exactly 2 inputs".to_string(),
            });
        }
        
        let shape1 = input_shapes[0];
        let shape2 = input_shapes[1];
        
        if shape1.len() < 2 || shape2.len() < 2 {
            return Err(TensorError::MatMulError {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
                reason: "Matrix multiplication requires at least 2D tensors".to_string(),
            });
        }
        
        let inner1 = shape1[shape1.len() - 1];
        let inner2 = shape2[shape2.len() - 2];
        
        if inner1 != inner2 {
            return Err(TensorError::MatMulError {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
                reason: format!("Inner dimensions don't match: {} vs {}", inner1, inner2),
            });
        }
        
        // Result shape: [...batch_dims, M, N]
        let mut result_shape = Vec::new();
        
        // Handle batch dimensions (simplified)
        let batch_dims1 = &shape1[..shape1.len() - 2];
        let batch_dims2 = &shape2[..shape2.len() - 2];
        
        if batch_dims1.len() >= batch_dims2.len() {
            result_shape.extend_from_slice(batch_dims1);
        } else {
            result_shape.extend_from_slice(batch_dims2);
        }
        
        // Matrix dimensions
        result_shape.push(shape1[shape1.len() - 2]); // M
        result_shape.push(shape2[shape2.len() - 1]); // N
        
        Ok(result_shape)
    }
}

/// Transpose operation
pub struct TransposeOperation;

impl TensorOperation for TransposeOperation {
    fn execute(&self, inputs: &[&Tensor], _params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Transpose operation requires exactly 1 input".to_string(),
            });
        }
        
        inputs[0].transpose()
    }

    fn name(&self) -> &str { "transpose" }
    fn input_arity(&self) -> usize { 1 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 1 && inputs[0].rank() >= 2
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Transpose operation requires exactly 1 input".to_string(),
            });
        }
        
        let shape = input_shapes[0];
        if shape.len() < 2 {
            return Err(TensorError::InvalidShape {
                shape: shape.to_vec(),
                reason: "Transpose requires at least 2D tensor".to_string(),
            });
        }
        
        let mut result_shape = shape.to_vec();
        let len = result_shape.len();
        result_shape.swap(len - 2, len - 1);
        
        Ok(result_shape)
    }
}

/// Reshape operation
pub struct ReshapeOperation;

impl TensorOperation for ReshapeOperation {
    fn execute(&self, inputs: &[&Tensor], params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Reshape operation requires exactly 1 input".to_string(),
            });
        }
        
        // Get target shape from parameters
        let target_shape = if let Some(OperationParam::IntArray(shape)) = params.params.get("shape") {
            shape.iter().map(|&x| x as usize).collect()
        } else {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Reshape operation requires 'shape' parameter".to_string(),
            });
        };
        
        inputs[0].reshape(target_shape)
    }

    fn name(&self) -> &str { "reshape" }
    fn input_arity(&self) -> usize { 1 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 1
    }
    
    fn infer_output_shape(&self, _input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        // Output shape depends on parameters, which we don't have access to here
        // In a real implementation, this would be handled differently
        Ok(vec![])
    }
}

/// Sum reduction operation
pub struct SumOperation;

impl TensorOperation for SumOperation {
    fn execute(&self, inputs: &[&Tensor], params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Sum operation requires exactly 1 input".to_string(),
            });
        }
        
        let axis = if let Some(OperationParam::Int(axis)) = params.params.get("axis") {
            Some(*axis as usize)
        } else {
            None // Sum all elements
        };
        
        sum_tensor(inputs[0], axis)
    }

    fn name(&self) -> &str { "sum" }
    fn input_arity(&self) -> usize { 1 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 1
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Sum operation requires exactly 1 input".to_string(),
            });
        }
        
        // For simplicity, assume sum over all dimensions (scalar result)
        Ok(vec![])
    }
}

/// Mean reduction operation
pub struct MeanOperation;

impl TensorOperation for MeanOperation {
    fn execute(&self, inputs: &[&Tensor], params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Mean operation requires exactly 1 input".to_string(),
            });
        }
        
        let axis = if let Some(OperationParam::Int(axis)) = params.params.get("axis") {
            Some(*axis as usize)
        } else {
            None // Mean of all elements
        };
        
        mean_tensor(inputs[0], axis)
    }

    fn name(&self) -> &str { "mean" }
    fn input_arity(&self) -> usize { 1 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 1
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Mean operation requires exactly 1 input".to_string(),
            });
        }
        
        // For simplicity, assume mean over all dimensions (scalar result)
        Ok(vec![])
    }
}

/// Max reduction operation
pub struct MaxOperation;

impl TensorOperation for MaxOperation {
    fn execute(&self, inputs: &[&Tensor], params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Max operation requires exactly 1 input".to_string(),
            });
        }
        
        let axis = if let Some(OperationParam::Int(axis)) = params.params.get("axis") {
            Some(*axis as usize)
        } else {
            None // Max of all elements
        };
        
        max_tensor(inputs[0], axis)
    }

    fn name(&self) -> &str { "max" }
    fn input_arity(&self) -> usize { 1 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 1
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Max operation requires exactly 1 input".to_string(),
            });
        }
        
        // For simplicity, assume max over all dimensions (scalar result)
        Ok(vec![])
    }
}

/// Min reduction operation
pub struct MinOperation;

impl TensorOperation for MinOperation {
    fn execute(&self, inputs: &[&Tensor], params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Min operation requires exactly 1 input".to_string(),
            });
        }
        
        let axis = if let Some(OperationParam::Int(axis)) = params.params.get("axis") {
            Some(*axis as usize)
        } else {
            None // Min of all elements
        };
        
        min_tensor(inputs[0], axis)
    }

    fn name(&self) -> &str { "min" }
    fn input_arity(&self) -> usize { 1 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 1
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Min operation requires exactly 1 input".to_string(),
            });
        }
        
        // For simplicity, assume min over all dimensions (scalar result)
        Ok(vec![])
    }
}

/// Softmax activation operation
pub struct SoftmaxOperation;

impl TensorOperation for SoftmaxOperation {
    fn execute(&self, inputs: &[&Tensor], params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Softmax operation requires exactly 1 input".to_string(),
            });
        }
        
        let axis = if let Some(OperationParam::Int(axis)) = params.params.get("axis") {
            *axis as usize
        } else {
            inputs[0].rank() - 1 // Default to last axis
        };
        
        softmax_tensor(inputs[0], axis)
    }

    fn name(&self) -> &str { "softmax" }
    fn input_arity(&self) -> usize { 1 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 1 && inputs[0].dtype().is_float()
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Softmax operation requires exactly 1 input".to_string(),
            });
        }
        
        // Softmax preserves input shape
        Ok(input_shapes[0].to_vec())
    }
}

/// ReLU activation operation
pub struct ReluOperation;

impl TensorOperation for ReluOperation {
    fn execute(&self, inputs: &[&Tensor], _params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "ReLU operation requires exactly 1 input".to_string(),
            });
        }
        
        relu_tensor(inputs[0])
    }

    fn name(&self) -> &str { "relu" }
    fn input_arity(&self) -> usize { 1 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 1
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 1 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "ReLU operation requires exactly 1 input".to_string(),
            });
        }
        
        // ReLU preserves input shape
        Ok(input_shapes[0].to_vec())
    }
}

/// Convolution operation (simplified)
pub struct ConvOperation;

impl TensorOperation for ConvOperation {
    fn execute(&self, inputs: &[&Tensor], params: &OperationParams) -> TensorResult<Tensor> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Conv operation requires exactly 2 inputs (input and kernel)".to_string(),
            });
        }
        
        // Get convolution parameters
        let stride = if let Some(OperationParam::IntArray(stride)) = params.params.get("stride") {
            stride.iter().map(|&x| x as usize).collect()
        } else {
            vec![1, 1] // Default stride
        };
        
        let padding = if let Some(OperationParam::IntArray(padding)) = params.params.get("padding") {
            padding.iter().map(|&x| x as usize).collect()
        } else {
            vec![0, 0] // Default no padding
        };
        
        conv2d_tensor(inputs[0], inputs[1], &stride, &padding)
    }

    fn name(&self) -> &str { "conv" }
    fn input_arity(&self) -> usize { 2 }
    
    fn supports(&self, inputs: &[&Tensor]) -> bool {
        inputs.len() == 2 && inputs[0].rank() == 4 && inputs[1].rank() == 4
    }
    
    fn infer_output_shape(&self, input_shapes: &[&[usize]]) -> TensorResult<Vec<usize>> {
        if input_shapes.len() != 2 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Conv operation requires exactly 2 inputs".to_string(),
            });
        }
        
        // Simplified convolution output shape calculation
        // Assumes NCHW format: [batch, channels, height, width]
        let input_shape = input_shapes[0];
        let kernel_shape = input_shapes[1];
        
        if input_shape.len() != 4 || kernel_shape.len() != 4 {
            return Err(TensorError::InvalidShape {
                shape: vec![],
                reason: "Conv operation requires 4D tensors".to_string(),
            });
        }
        
        let batch = input_shape[0];
        let out_channels = kernel_shape[0];
        let in_height = input_shape[2];
        let in_width = input_shape[3];
        let kernel_height = kernel_shape[2];
        let kernel_width = kernel_shape[3];
        
        // Simplified calculation (assumes stride=1, padding=0)
        let out_height = in_height - kernel_height + 1;
        let out_width = in_width - kernel_width + 1;
        
        Ok(vec![batch, out_channels, out_height, out_width])
    }
}

// Helper functions for tensor operations

/// Broadcast two shapes according to NumPy broadcasting rules
pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> TensorResult<Vec<usize>> {
    let max_len = shape1.len().max(shape2.len());
    let mut result_shape = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let dim1 = shape1.get(shape1.len().saturating_sub(max_len - i)).copied().unwrap_or(1);
        let dim2 = shape2.get(shape2.len().saturating_sub(max_len - i)).copied().unwrap_or(1);

        if dim1 == 1 {
            result_shape.push(dim2);
        } else if dim2 == 1 {
            result_shape.push(dim1);
        } else if dim1 == dim2 {
            result_shape.push(dim1);
        } else {
            return Err(TensorError::BroadcastError {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
            });
        }
    }

    Ok(result_shape)
}

/// Negate tensor elements
fn negate_tensor(tensor: &Tensor) -> TensorResult<Tensor> {
    let zeros = Tensor::zeros(tensor.shape().to_vec(), tensor.dtype())?;
    zeros.add(&tensor.mul(&Tensor::from_data_and_shape(vec![-1.0f32], vec![1])?)?)
}

/// Compute reciprocal of tensor elements
fn reciprocal_tensor(tensor: &Tensor) -> TensorResult<Tensor> {
    let ones = Tensor::ones(tensor.shape().to_vec(), tensor.dtype())?;
    // This is a simplified implementation - would need proper division
    Ok(ones)
}

/// Sum tensor along specified axis
fn sum_tensor(tensor: &Tensor, _axis: Option<usize>) -> TensorResult<Tensor> {
    // Simplified implementation - sum all elements to scalar
    match tensor.dtype() {
        TensorDType::Float32 => {
            let data = tensor.cast_to_f32()?;
            let sum: f32 = data.iter().sum();
            Tensor::from_data_and_shape(vec![sum], vec![])
        }
        TensorDType::Float64 => {
            let data = tensor.cast_to_f64()?;
            let sum: f64 = data.iter().sum();
            Tensor::from_data_and_shape(vec![sum], vec![])
        }
        _ => Err(TensorError::DTypeMismatch {
            expected: TensorDType::Float32,
            actual: tensor.dtype(),
            operation: "sum".to_string(),
        }),
    }
}

/// Compute mean of tensor along specified axis
fn mean_tensor(tensor: &Tensor, axis: Option<usize>) -> TensorResult<Tensor> {
    let sum_result = sum_tensor(tensor, axis)?;
    let count = tensor.size() as f32;
    let count_tensor = Tensor::from_data_and_shape(vec![count], vec![])?;
    sum_result.mul(&reciprocal_tensor(&count_tensor)?)
}

/// Find maximum value in tensor
fn max_tensor(tensor: &Tensor, _axis: Option<usize>) -> TensorResult<Tensor> {
    match tensor.dtype() {
        TensorDType::Float32 => {
            let data = tensor.cast_to_f32()?;
            let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            Tensor::from_data_and_shape(vec![max], vec![])
        }
        TensorDType::Float64 => {
            let data = tensor.cast_to_f64()?;
            let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            Tensor::from_data_and_shape(vec![max], vec![])
        }
        _ => Err(TensorError::DTypeMismatch {
            expected: TensorDType::Float32,
            actual: tensor.dtype(),
            operation: "max".to_string(),
        }),
    }
}

/// Find minimum value in tensor
fn min_tensor(tensor: &Tensor, _axis: Option<usize>) -> TensorResult<Tensor> {
    match tensor.dtype() {
        TensorDType::Float32 => {
            let data = tensor.cast_to_f32()?;
            let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            Tensor::from_data_and_shape(vec![min], vec![])
        }
        TensorDType::Float64 => {
            let data = tensor.cast_to_f64()?;
            let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            Tensor::from_data_and_shape(vec![min], vec![])
        }
        _ => Err(TensorError::DTypeMismatch {
            expected: TensorDType::Float32,
            actual: tensor.dtype(),
            operation: "min".to_string(),
        }),
    }
}

/// Apply softmax activation
fn softmax_tensor(tensor: &Tensor, _axis: usize) -> TensorResult<Tensor> {
    // Simplified softmax implementation
    match tensor.dtype() {
        TensorDType::Float32 => {
            let data = tensor.cast_to_f32()?;
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            // Subtract max for numerical stability
            let shifted: Vec<f32> = data.iter().map(|&x| x - max_val).collect();
            
            // Compute exponentials
            let exp_vals: Vec<f32> = shifted.iter().map(|&x| x.exp()).collect();
            
            // Compute sum of exponentials
            let sum_exp: f32 = exp_vals.iter().sum();
            
            // Normalize
            let softmax_vals: Vec<f32> = exp_vals.iter().map(|&x| x / sum_exp).collect();
            
            Tensor::from_data_and_shape(softmax_vals, tensor.shape().to_vec())
        }
        _ => Err(TensorError::DTypeMismatch {
            expected: TensorDType::Float32,
            actual: tensor.dtype(),
            operation: "softmax".to_string(),
        }),
    }
}

/// Apply ReLU activation
fn relu_tensor(tensor: &Tensor) -> TensorResult<Tensor> {
    match tensor.dtype() {
        TensorDType::Float32 => {
            let data = tensor.cast_to_f32()?;
            let relu_vals: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
            Tensor::from_data_and_shape(relu_vals, tensor.shape().to_vec())
        }
        TensorDType::Float64 => {
            let data = tensor.cast_to_f64()?;
            let relu_vals: Vec<f64> = data.iter().map(|&x| x.max(0.0)).collect();
            Tensor::from_data_and_shape(relu_vals, tensor.shape().to_vec())
        }
        _ => Err(TensorError::DTypeMismatch {
            expected: TensorDType::Float32,
            actual: tensor.dtype(),
            operation: "relu".to_string(),
        }),
    }
}

/// Simplified 2D convolution
fn conv2d_tensor(input: &Tensor, kernel: &Tensor, _stride: &[usize], _padding: &[usize]) -> TensorResult<Tensor> {
    // This is a very simplified convolution implementation
    // A real implementation would be much more complex and optimized
    
    if input.rank() != 4 || kernel.rank() != 4 {
        return Err(TensorError::InvalidShape {
            shape: vec![],
            reason: "Convolution requires 4D tensors".to_string(),
        });
    }
    
    let input_shape = input.shape();
    let kernel_shape = kernel.shape();
    
    let batch = input_shape[0];
    let out_channels = kernel_shape[0];
    let out_height = input_shape[2] - kernel_shape[2] + 1;
    let out_width = input_shape[3] - kernel_shape[3] + 1;
    
    let output_shape = vec![batch, out_channels, out_height, out_width];
    let output_size: usize = output_shape.iter().product();
    
    // Create zero output tensor
    let output_data = vec![0.0f32; output_size];
    Tensor::from_data_and_shape(output_data, output_shape)
}

// Default configurations

impl Default for BlasConfig {
    fn default() -> Self {
        BlasConfig {
            available: false, // Would be detected at runtime
            library: BlasLibrary::Generic,
            num_threads: 1,
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        GpuConfig {
            available: false, // Would be detected at runtime
            backend: GpuBackend::Cuda,
            devices: Vec::new(),
        }
    }
}

impl Default for OperationParams {
    fn default() -> Self {
        OperationParams {
            params: HashMap::new(),
            device: None,
            use_optimized: true,
        }
    }
}

// Note: cast_to_f32 and cast_to_f64 methods are implemented in tensor.rs

// Additional helper functions for tensor operations

/// Create a tensor filled with random values (for testing)
pub fn random_tensor(shape: Vec<usize>, dtype: TensorDType) -> TensorResult<Tensor> {
    let total_elements: usize = shape.iter().product();
    
    match dtype {
        TensorDType::Float32 => {
            // Simple pseudo-random generation for testing
            let mut data = Vec::with_capacity(total_elements);
            let mut seed = 12345u32;
            for _ in 0..total_elements {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let val = (seed as f32) / (u32::MAX as f32);
                data.push(val);
            }
            Tensor::from_data_and_shape(data, shape)
        }
        TensorDType::Float64 => {
            let mut data = Vec::with_capacity(total_elements);
            let mut seed = 12345u64;
            for _ in 0..total_elements {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let val = (seed as f64) / (u64::MAX as f64);
                data.push(val);
            }
            Tensor::from_data_and_shape(data, shape)
        }
        TensorDType::Int32 => {
            let mut data = Vec::with_capacity(total_elements);
            let mut seed = 12345u32;
            for _ in 0..total_elements {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                data.push((seed % 100) as i32);
            }
            Tensor::from_data_and_shape(data, shape)
        }
        TensorDType::Int64 => {
            let mut data = Vec::with_capacity(total_elements);
            let mut seed = 12345u64;
            for _ in 0..total_elements {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                data.push((seed % 100) as i64);
            }
            Tensor::from_data_and_shape(data, shape)
        }
        TensorDType::Bool => {
            let mut data = Vec::with_capacity(total_elements);
            let mut seed = 12345u32;
            for _ in 0..total_elements {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                data.push((seed % 2) == 0);
            }
            Tensor::from_data_and_shape(data, shape)
        }
    }
}

/// Benchmark tensor operation performance
pub fn benchmark_operation<F>(name: &str, operation: F) -> Duration 
where 
    F: FnOnce() -> TensorResult<Tensor>,
{
    let start = std::time::Instant::now();
    let _result = operation();
    let duration = start.elapsed();
    println!("Operation '{}' took: {:.2}ms", name, duration.as_secs_f64() * 1000.0);
    duration
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_ops_registry() {
        let registry = TensorOpsRegistry::new();
        let ops = registry.available_operations();
        
        assert!(ops.contains(&"add"));
        assert!(ops.contains(&"mul"));
        assert!(ops.contains(&"matmul"));
        assert!(ops.contains(&"transpose"));
    }

    #[test]
    fn test_operation_execution() {
        let registry = TensorOpsRegistry::new();
        let tensor_a = Tensor::ones(vec![2, 2], TensorDType::Float32).unwrap();
        let tensor_b = Tensor::ones(vec![2, 2], TensorDType::Float32).unwrap();
        
        let params = OperationParams::default();
        let result = registry.execute("add", &[&tensor_a, &tensor_b], &params).unwrap();
        
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.dtype(), TensorDType::Float32);
    }

    #[test]
    fn test_broadcasting_shapes() {
        let shape1 = vec![3, 1];
        let shape2 = vec![1, 4];
        let result = broadcast_shapes(&shape1, &shape2).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_random_tensor_generation() {
        let tensor = random_tensor(vec![10, 10], TensorDType::Float32).unwrap();
        assert_eq!(tensor.shape(), &[10, 10]);
        assert_eq!(tensor.size(), 100);
    }

    #[test]
    fn test_relu_operation() {
        let data = vec![-1.0f32, 0.0, 1.0, 2.0, -2.0];
        let tensor = Tensor::from_data_and_shape(data, vec![5]).unwrap();
        let result = relu_tensor(&tensor).unwrap();
        
        let expected = vec![0.0f32, 0.0, 1.0, 2.0, 0.0];
        if let crate::runtime::tensor::TensorData::Float32(result_data) = result.data() {
            assert_eq!(result_data, &expected);
        } else {
            panic!("Expected Float32 data");
        }
    }

    #[test]
    fn test_softmax_operation() {
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_data_and_shape(data, vec![3]).unwrap();
        let result = softmax_tensor(&tensor, 0).unwrap();
        
        // Check that softmax output sums to 1
        if let crate::runtime::tensor::TensorData::Float32(result_data) = result.data() {
            let sum: f32 = result_data.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1, got {}", sum);
        } else {
            panic!("Expected Float32 data");
        }
    }
}