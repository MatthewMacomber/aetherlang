// Aether Native Tensor Computation System
// Implements tensor as primitive type with shape-aware operations

use std::fmt;
use crate::compiler::types::Type;

/// Native tensor data structure with shape-aware operations
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Raw data storage
    data: TensorData,
    /// Tensor shape (dimensions)
    shape: Vec<usize>,
    /// Element data type
    dtype: TensorDType,
    /// Memory layout information
    layout: TensorLayout,
    /// Device location (CPU, GPU, etc.)
    device: TensorDevice,
}

/// Tensor data storage variants
#[derive(Debug, Clone)]
pub enum TensorData {
    /// 32-bit floating point data
    Float32(Vec<f32>),
    /// 64-bit floating point data
    Float64(Vec<f64>),
    /// 32-bit signed integer data
    Int32(Vec<i32>),
    /// 64-bit signed integer data
    Int64(Vec<i64>),
    /// Boolean data
    Bool(Vec<bool>),
    /// GPU buffer reference (opaque handle)
    GpuBuffer(GpuBufferHandle),
}

/// GPU buffer handle for device memory
#[derive(Debug, Clone)]
pub struct GpuBufferHandle {
    pub device_id: u32,
    pub buffer_id: u64,
    pub size_bytes: usize,
}

/// Tensor data type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
}

/// Tensor memory layout for cache optimization
#[derive(Debug, Clone, PartialEq)]
pub enum TensorLayout {
    /// Row-major (C-style) layout
    RowMajor,
    /// Column-major (Fortran-style) layout
    ColumnMajor,
    /// Strided layout with custom strides
    Strided { strides: Vec<usize> },
    /// Blocked layout for cache efficiency
    Blocked { block_sizes: Vec<usize> },
}

/// Tensor device location
#[derive(Debug, Clone, PartialEq)]
pub enum TensorDevice {
    /// CPU memory
    Cpu,
    /// GPU memory with device ID
    Gpu(u32),
    /// Shared memory between CPU and GPU
    Shared,
}

/// Tensor operation result type
pub type TensorResult<T> = Result<T, TensorError>;

/// Tensor computation errors
#[derive(Debug, Clone)]
pub enum TensorError {
    /// Shape mismatch in operations
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        operation: String,
    },
    /// Data type mismatch
    DTypeMismatch {
        expected: TensorDType,
        actual: TensorDType,
        operation: String,
    },
    /// Device mismatch (CPU vs GPU)
    DeviceMismatch {
        expected: TensorDevice,
        actual: TensorDevice,
        operation: String,
    },
    /// Invalid tensor dimensions
    InvalidShape {
        shape: Vec<usize>,
        reason: String,
    },
    /// Memory allocation failure
    AllocationError {
        size_bytes: usize,
        device: TensorDevice,
    },
    /// GPU operation error
    GpuError {
        message: String,
        device_id: u32,
    },
    /// Broadcasting error
    BroadcastError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    },
    /// Matrix multiplication dimension error
    MatMulError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
        reason: String,
    },
}

impl Tensor {
    /// Create new tensor with given data and shape
    pub fn new(data: TensorData, shape: Vec<usize>) -> TensorResult<Self> {
        let total_elements: usize = shape.iter().product();
        let data_len = data.len();
        
        if total_elements != data_len {
            return Err(TensorError::InvalidShape {
                shape: shape.clone(),
                reason: format!("Shape implies {} elements but data has {} elements", 
                    total_elements, data_len),
            });
        }

        let dtype = data.dtype();
        
        Ok(Tensor {
            data,
            shape,
            dtype,
            layout: TensorLayout::RowMajor,
            device: TensorDevice::Cpu,
        })
    }

    /// Create tensor from flat data with shape
    pub fn from_data_and_shape<T: Into<TensorData>>(data: T, shape: Vec<usize>) -> TensorResult<Self> {
        Self::new(data.into(), shape)
    }

    /// Create zero tensor with given shape and dtype
    pub fn zeros(shape: Vec<usize>, dtype: TensorDType) -> TensorResult<Self> {
        let total_elements: usize = shape.iter().product();
        
        let data = match dtype {
            TensorDType::Float32 => TensorData::Float32(vec![0.0; total_elements]),
            TensorDType::Float64 => TensorData::Float64(vec![0.0; total_elements]),
            TensorDType::Int32 => TensorData::Int32(vec![0; total_elements]),
            TensorDType::Int64 => TensorData::Int64(vec![0; total_elements]),
            TensorDType::Bool => TensorData::Bool(vec![false; total_elements]),
        };

        Self::new(data, shape)
    }

    /// Create ones tensor with given shape and dtype
    pub fn ones(shape: Vec<usize>, dtype: TensorDType) -> TensorResult<Self> {
        let total_elements: usize = shape.iter().product();
        
        let data = match dtype {
            TensorDType::Float32 => TensorData::Float32(vec![1.0; total_elements]),
            TensorDType::Float64 => TensorData::Float64(vec![1.0; total_elements]),
            TensorDType::Int32 => TensorData::Int32(vec![1; total_elements]),
            TensorDType::Int64 => TensorData::Int64(vec![1; total_elements]),
            TensorDType::Bool => TensorData::Bool(vec![true; total_elements]),
        };

        Self::new(data, shape)
    }

    /// Create identity matrix
    pub fn eye(size: usize, dtype: TensorDType) -> TensorResult<Self> {
        let mut tensor = Self::zeros(vec![size, size], dtype)?;
        
        for i in 0..size {
            let idx = i * size + i; // Row-major index for diagonal
            match &mut tensor.data {
                TensorData::Float32(ref mut data) => data[idx] = 1.0,
                TensorData::Float64(ref mut data) => data[idx] = 1.0,
                TensorData::Int32(ref mut data) => data[idx] = 1,
                TensorData::Int64(ref mut data) => data[idx] = 1,
                TensorData::Bool(ref mut data) => data[idx] = true,
                TensorData::GpuBuffer(_) => {
                    return Err(TensorError::GpuError {
                        message: "GPU identity matrix creation not implemented".to_string(),
                        device_id: 0,
                    });
                }
            }
        }
        
        Ok(tensor)
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get tensor rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get tensor data type
    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    /// Get tensor device
    pub fn device(&self) -> &TensorDevice {
        &self.device
    }

    /// Get tensor layout
    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    /// Get tensor data (for testing purposes)
    #[cfg(test)]
    pub fn data(&self) -> &TensorData {
        &self.data
    }

    /// Check if tensor is on CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self.device, TensorDevice::Cpu)
    }

    /// Check if tensor is on GPU
    pub fn is_gpu(&self) -> bool {
        matches!(self.device, TensorDevice::Gpu(_))
    }

    /// Reshape tensor (must preserve total number of elements)
    pub fn reshape(&self, new_shape: Vec<usize>) -> TensorResult<Self> {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        
        if old_size != new_size {
            return Err(TensorError::InvalidShape {
                shape: new_shape,
                reason: format!("Cannot reshape tensor of size {} to size {}", old_size, new_size),
            });
        }

        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
            layout: self.layout.clone(),
            device: self.device.clone(),
        })
    }

    /// Transpose tensor (swap last two dimensions)
    pub fn transpose(&self) -> TensorResult<Self> {
        if self.rank() < 2 {
            return Err(TensorError::InvalidShape {
                shape: self.shape.clone(),
                reason: "Transpose requires at least 2D tensor".to_string(),
            });
        }

        let mut new_shape = self.shape.clone();
        let len = new_shape.len();
        new_shape.swap(len - 2, len - 1);

        // For now, we'll create a new tensor with transposed data
        // In a real implementation, this would be optimized with lazy evaluation
        let transposed_data = self.transpose_data()?;

        Ok(Tensor {
            data: transposed_data,
            shape: new_shape,
            dtype: self.dtype,
            layout: self.layout.clone(),
            device: self.device.clone(),
        })
    }

    /// Internal method to transpose data
    fn transpose_data(&self) -> TensorResult<TensorData> {
        if self.rank() != 2 {
            // For simplicity, only handle 2D transpose for now
            return Ok(self.data.clone());
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        match &self.data {
            TensorData::Float32(data) => {
                let mut transposed = vec![0.0f32; data.len()];
                for i in 0..rows {
                    for j in 0..cols {
                        transposed[j * rows + i] = data[i * cols + j];
                    }
                }
                Ok(TensorData::Float32(transposed))
            }
            TensorData::Float64(data) => {
                let mut transposed = vec![0.0f64; data.len()];
                for i in 0..rows {
                    for j in 0..cols {
                        transposed[j * rows + i] = data[i * cols + j];
                    }
                }
                Ok(TensorData::Float64(transposed))
            }
            TensorData::Int32(data) => {
                let mut transposed = vec![0i32; data.len()];
                for i in 0..rows {
                    for j in 0..cols {
                        transposed[j * rows + i] = data[i * cols + j];
                    }
                }
                Ok(TensorData::Int32(transposed))
            }
            TensorData::Int64(data) => {
                let mut transposed = vec![0i64; data.len()];
                for i in 0..rows {
                    for j in 0..cols {
                        transposed[j * rows + i] = data[i * cols + j];
                    }
                }
                Ok(TensorData::Int64(transposed))
            }
            TensorData::Bool(data) => {
                let mut transposed = vec![false; data.len()];
                for i in 0..rows {
                    for j in 0..cols {
                        transposed[j * rows + i] = data[i * cols + j];
                    }
                }
                Ok(TensorData::Bool(transposed))
            }
            TensorData::GpuBuffer(_) => {
                Err(TensorError::GpuError {
                    message: "GPU transpose not implemented".to_string(),
                    device_id: 0,
                })
            }
        }
    }

    /// Element-wise addition with broadcasting
    pub fn add(&self, other: &Tensor) -> TensorResult<Tensor> {
        self.check_device_compatibility(other, "add")?;
        
        let result_shape = self.broadcast_shapes(&self.shape, &other.shape)?;
        let result_dtype = self.promote_dtypes(self.dtype, other.dtype)?;
        
        let result_data = match (result_dtype, &self.data, &other.data) {
            (TensorDType::Float32, _, _) => {
                let self_f32 = self.cast_to_f32()?;
                let other_f32 = other.cast_to_f32()?;
                let broadcasted_self = self.broadcast_data_f32(&self_f32, &self.shape, &result_shape)?;
                let broadcasted_other = other.broadcast_data_f32(&other_f32, &other.shape, &result_shape)?;
                
                let result: Vec<f32> = broadcasted_self.iter()
                    .zip(broadcasted_other.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                TensorData::Float32(result)
            }
            (TensorDType::Float64, _, _) => {
                let self_f64 = self.cast_to_f64()?;
                let other_f64 = other.cast_to_f64()?;
                let broadcasted_self = self.broadcast_data_f64(&self_f64, &self.shape, &result_shape)?;
                let broadcasted_other = other.broadcast_data_f64(&other_f64, &other.shape, &result_shape)?;
                
                let result: Vec<f64> = broadcasted_self.iter()
                    .zip(broadcasted_other.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                TensorData::Float64(result)
            }
            (TensorDType::Int32, TensorData::Int32(a), TensorData::Int32(b)) => {
                let broadcasted_self = self.broadcast_data_i32(a, &self.shape, &result_shape)?;
                let broadcasted_other = other.broadcast_data_i32(b, &other.shape, &result_shape)?;
                
                let result: Vec<i32> = broadcasted_self.iter()
                    .zip(broadcasted_other.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                TensorData::Int32(result)
            }
            _ => return Err(TensorError::DTypeMismatch {
                expected: result_dtype,
                actual: self.dtype,
                operation: "add".to_string(),
            }),
        };

        Tensor::new(result_data, result_shape)
    }

    /// Element-wise multiplication with broadcasting
    pub fn mul(&self, other: &Tensor) -> TensorResult<Tensor> {
        self.check_device_compatibility(other, "mul")?;
        
        let result_shape = self.broadcast_shapes(&self.shape, &other.shape)?;
        let result_dtype = self.promote_dtypes(self.dtype, other.dtype)?;
        
        let result_data = match (result_dtype, &self.data, &other.data) {
            (TensorDType::Float32, _, _) => {
                let self_f32 = self.cast_to_f32()?;
                let other_f32 = other.cast_to_f32()?;
                let broadcasted_self = self.broadcast_data_f32(&self_f32, &self.shape, &result_shape)?;
                let broadcasted_other = other.broadcast_data_f32(&other_f32, &other.shape, &result_shape)?;
                
                let result: Vec<f32> = broadcasted_self.iter()
                    .zip(broadcasted_other.iter())
                    .map(|(a, b)| a * b)
                    .collect();
                TensorData::Float32(result)
            }
            (TensorDType::Float64, _, _) => {
                let self_f64 = self.cast_to_f64()?;
                let other_f64 = other.cast_to_f64()?;
                let broadcasted_self = self.broadcast_data_f64(&self_f64, &self.shape, &result_shape)?;
                let broadcasted_other = other.broadcast_data_f64(&other_f64, &other.shape, &result_shape)?;
                
                let result: Vec<f64> = broadcasted_self.iter()
                    .zip(broadcasted_other.iter())
                    .map(|(a, b)| a * b)
                    .collect();
                TensorData::Float64(result)
            }
            (TensorDType::Int32, TensorData::Int32(a), TensorData::Int32(b)) => {
                let broadcasted_self = self.broadcast_data_i32(a, &self.shape, &result_shape)?;
                let broadcasted_other = other.broadcast_data_i32(b, &other.shape, &result_shape)?;
                
                let result: Vec<i32> = broadcasted_self.iter()
                    .zip(broadcasted_other.iter())
                    .map(|(a, b)| a * b)
                    .collect();
                TensorData::Int32(result)
            }
            _ => return Err(TensorError::DTypeMismatch {
                expected: result_dtype,
                actual: self.dtype,
                operation: "mul".to_string(),
            }),
        };

        Tensor::new(result_data, result_shape)
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> TensorResult<Tensor> {
        self.check_device_compatibility(other, "matmul")?;
        self.check_matmul_compatibility(other)?;
        
        // For simplicity, handle only 2D matrix multiplication
        if self.rank() != 2 || other.rank() != 2 {
            return Err(TensorError::MatMulError {
                shape1: self.shape.clone(),
                shape2: other.shape.clone(),
                reason: "Only 2D matrix multiplication supported".to_string(),
            });
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        
        let result_shape = vec![m, n];
        let result_dtype = self.promote_dtypes(self.dtype, other.dtype)?;

        let result_data = match (result_dtype, &self.data, &other.data) {
            (TensorDType::Float32, _, _) => {
                let self_f32 = self.cast_to_f32()?;
                let other_f32 = other.cast_to_f32()?;
                let mut result = vec![0.0f32; m * n];
                
                // Basic matrix multiplication (can be optimized with BLAS)
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for l in 0..k {
                            sum += self_f32[i * k + l] * other_f32[l * n + j];
                        }
                        result[i * n + j] = sum;
                    }
                }
                TensorData::Float32(result)
            }
            (TensorDType::Float64, _, _) => {
                let self_f64 = self.cast_to_f64()?;
                let other_f64 = other.cast_to_f64()?;
                let mut result = vec![0.0f64; m * n];
                
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f64;
                        for l in 0..k {
                            sum += self_f64[i * k + l] * other_f64[l * n + j];
                        }
                        result[i * n + j] = sum;
                    }
                }
                TensorData::Float64(result)
            }
            _ => return Err(TensorError::DTypeMismatch {
                expected: result_dtype,
                actual: self.dtype,
                operation: "matmul".to_string(),
            }),
        };

        Tensor::new(result_data, result_shape)
    }

    /// Move tensor to specified device
    pub fn to_device(&self, device: TensorDevice) -> TensorResult<Tensor> {
        if self.device == device {
            return Ok(self.clone());
        }

        match (&self.device, &device) {
            (TensorDevice::Cpu, TensorDevice::Gpu(device_id)) => {
                // Simulate GPU transfer
                let gpu_handle = GpuBufferHandle {
                    device_id: *device_id,
                    buffer_id: (*device_id as u64) << 32 | (self.size() as u64), // Simulate GPU buffer ID
                    size_bytes: self.size() * self.dtype.size_bytes(),
                };
                
                Ok(Tensor {
                    data: TensorData::GpuBuffer(gpu_handle),
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                    layout: self.layout.clone(),
                    device,
                })
            }
            (TensorDevice::Gpu(_), TensorDevice::Cpu) => {
                // Simulate GPU to CPU transfer
                // In real implementation, this would copy data back from GPU
                Err(TensorError::GpuError {
                    message: "GPU to CPU transfer not implemented".to_string(),
                    device_id: 0,
                })
            }
            _ => Err(TensorError::DeviceMismatch {
                expected: device,
                actual: self.device.clone(),
                operation: "device transfer".to_string(),
            }),
        }
    }

    /// Optimize memory layout for cache efficiency
    pub fn optimize_layout(&self) -> TensorResult<Tensor> {
        // For now, just return the same tensor
        // In a real implementation, this would analyze access patterns and optimize layout
        Ok(self.clone())
    }

    // Helper methods for internal operations

    fn check_device_compatibility(&self, other: &Tensor, operation: &str) -> TensorResult<()> {
        if self.device != other.device {
            return Err(TensorError::DeviceMismatch {
                expected: self.device.clone(),
                actual: other.device.clone(),
                operation: operation.to_string(),
            });
        }
        Ok(())
    }

    fn check_matmul_compatibility(&self, other: &Tensor) -> TensorResult<()> {
        if self.rank() < 2 || other.rank() < 2 {
            return Err(TensorError::MatMulError {
                shape1: self.shape.clone(),
                shape2: other.shape.clone(),
                reason: "Matrix multiplication requires at least 2D tensors".to_string(),
            });
        }

        let self_inner = self.shape[self.rank() - 1];
        let other_inner = other.shape[other.rank() - 2];

        if self_inner != other_inner {
            return Err(TensorError::MatMulError {
                shape1: self.shape.clone(),
                shape2: other.shape.clone(),
                reason: format!("Inner dimensions don't match: {} vs {}", self_inner, other_inner),
            });
        }

        Ok(())
    }

    fn broadcast_shapes(&self, shape1: &[usize], shape2: &[usize]) -> TensorResult<Vec<usize>> {
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

    fn promote_dtypes(&self, dtype1: TensorDType, dtype2: TensorDType) -> TensorResult<TensorDType> {
        use TensorDType::*;
        
        match (dtype1, dtype2) {
            (Float64, _) | (_, Float64) => Ok(Float64),
            (Float32, _) | (_, Float32) => Ok(Float32),
            (Int64, _) | (_, Int64) => Ok(Int64),
            (Int32, Int32) => Ok(Int32),
            (Bool, Bool) => Ok(Bool),
            _ => Ok(Float32), // Default promotion
        }
    }

    // Data casting methods
    pub(crate) fn cast_to_f32(&self) -> TensorResult<Vec<f32>> {
        match &self.data {
            TensorData::Float32(data) => Ok(data.clone()),
            TensorData::Float64(data) => Ok(data.iter().map(|&x| x as f32).collect()),
            TensorData::Int32(data) => Ok(data.iter().map(|&x| x as f32).collect()),
            TensorData::Int64(data) => Ok(data.iter().map(|&x| x as f32).collect()),
            TensorData::Bool(data) => Ok(data.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect()),
            TensorData::GpuBuffer(_) => Err(TensorError::GpuError {
                message: "Cannot cast GPU buffer to f32".to_string(),
                device_id: 0,
            }),
        }
    }

    pub(crate) fn cast_to_f64(&self) -> TensorResult<Vec<f64>> {
        match &self.data {
            TensorData::Float32(data) => Ok(data.iter().map(|&x| x as f64).collect()),
            TensorData::Float64(data) => Ok(data.clone()),
            TensorData::Int32(data) => Ok(data.iter().map(|&x| x as f64).collect()),
            TensorData::Int64(data) => Ok(data.iter().map(|&x| x as f64).collect()),
            TensorData::Bool(data) => Ok(data.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect()),
            TensorData::GpuBuffer(_) => Err(TensorError::GpuError {
                message: "Cannot cast GPU buffer to f64".to_string(),
                device_id: 0,
            }),
        }
    }

    // Broadcasting helper methods
    fn broadcast_data_f32(&self, data: &[f32], from_shape: &[usize], to_shape: &[usize]) -> TensorResult<Vec<f32>> {
        if from_shape == to_shape {
            return Ok(data.to_vec());
        }

        let total_elements: usize = to_shape.iter().product();
        let mut result = Vec::with_capacity(total_elements);

        // Simplified broadcasting - only handle basic cases
        if from_shape.len() == 1 && from_shape[0] == 1 && to_shape.len() >= 1 {
            // Broadcast scalar to any shape
            let scalar = data[0];
            result.resize(total_elements, scalar);
        } else {
            // For more complex broadcasting, we'd need a more sophisticated algorithm
            result = data.to_vec();
            result.resize(total_elements, 0.0);
        }

        Ok(result)
    }

    fn broadcast_data_f64(&self, data: &[f64], from_shape: &[usize], to_shape: &[usize]) -> TensorResult<Vec<f64>> {
        if from_shape == to_shape {
            return Ok(data.to_vec());
        }

        let total_elements: usize = to_shape.iter().product();
        let mut result = Vec::with_capacity(total_elements);

        if from_shape.len() == 1 && from_shape[0] == 1 && to_shape.len() >= 1 {
            let scalar = data[0];
            result.resize(total_elements, scalar);
        } else {
            result = data.to_vec();
            result.resize(total_elements, 0.0);
        }

        Ok(result)
    }

    fn broadcast_data_i32(&self, data: &[i32], from_shape: &[usize], to_shape: &[usize]) -> TensorResult<Vec<i32>> {
        if from_shape == to_shape {
            return Ok(data.to_vec());
        }

        let total_elements: usize = to_shape.iter().product();
        let mut result = Vec::with_capacity(total_elements);

        if from_shape.len() == 1 && from_shape[0] == 1 && to_shape.len() >= 1 {
            let scalar = data[0];
            result.resize(total_elements, scalar);
        } else {
            result = data.to_vec();
            result.resize(total_elements, 0);
        }

        Ok(result)
    }
}

impl TensorData {
    /// Get the length of the data
    pub fn len(&self) -> usize {
        match self {
            TensorData::Float32(data) => data.len(),
            TensorData::Float64(data) => data.len(),
            TensorData::Int32(data) => data.len(),
            TensorData::Int64(data) => data.len(),
            TensorData::Bool(data) => data.len(),
            TensorData::GpuBuffer(handle) => handle.size_bytes / 4, // Assume 4-byte elements
        }
    }

    /// Check if data is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the data type
    pub fn dtype(&self) -> TensorDType {
        match self {
            TensorData::Float32(_) => TensorDType::Float32,
            TensorData::Float64(_) => TensorDType::Float64,
            TensorData::Int32(_) => TensorDType::Int32,
            TensorData::Int64(_) => TensorDType::Int64,
            TensorData::Bool(_) => TensorDType::Bool,
            TensorData::GpuBuffer(_) => TensorDType::Float32, // Default assumption
        }
    }
}

impl TensorDType {
    /// Get size in bytes for this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            TensorDType::Float32 => 4,
            TensorDType::Float64 => 8,
            TensorDType::Int32 => 4,
            TensorDType::Int64 => 8,
            TensorDType::Bool => 1,
        }
    }

    /// Check if data type is floating point
    pub fn is_float(&self) -> bool {
        matches!(self, TensorDType::Float32 | TensorDType::Float64)
    }

    /// Check if data type is integer
    pub fn is_integer(&self) -> bool {
        matches!(self, TensorDType::Int32 | TensorDType::Int64)
    }
}

// Conversion traits for easy tensor creation
impl From<Vec<f32>> for TensorData {
    fn from(data: Vec<f32>) -> Self {
        TensorData::Float32(data)
    }
}

impl From<Vec<f64>> for TensorData {
    fn from(data: Vec<f64>) -> Self {
        TensorData::Float64(data)
    }
}

impl From<Vec<i32>> for TensorData {
    fn from(data: Vec<i32>) -> Self {
        TensorData::Int32(data)
    }
}

impl From<Vec<i64>> for TensorData {
    fn from(data: Vec<i64>) -> Self {
        TensorData::Int64(data)
    }
}

impl From<Vec<bool>> for TensorData {
    fn from(data: Vec<bool>) -> Self {
        TensorData::Bool(data)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, dtype={:?}, device={:?})", 
               self.shape, self.dtype, self.device)
    }
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, actual, operation } => {
                write!(f, "Shape mismatch in {}: expected {:?}, got {:?}", operation, expected, actual)
            }
            TensorError::DTypeMismatch { expected, actual, operation } => {
                write!(f, "Data type mismatch in {}: expected {:?}, got {:?}", operation, expected, actual)
            }
            TensorError::DeviceMismatch { expected, actual, operation } => {
                write!(f, "Device mismatch in {}: expected {:?}, got {:?}", operation, expected, actual)
            }
            TensorError::InvalidShape { shape, reason } => {
                write!(f, "Invalid shape {:?}: {}", shape, reason)
            }
            TensorError::AllocationError { size_bytes, device } => {
                write!(f, "Failed to allocate {} bytes on device {:?}", size_bytes, device)
            }
            TensorError::GpuError { message, device_id } => {
                write!(f, "GPU error on device {}: {}", device_id, message)
            }
            TensorError::BroadcastError { shape1, shape2 } => {
                write!(f, "Cannot broadcast shapes {:?} and {:?}", shape1, shape2)
            }
            TensorError::MatMulError { shape1, shape2, reason } => {
                write!(f, "Matrix multiplication error between {:?} and {:?}: {}", shape1, shape2, reason)
            }
        }
    }
}

impl std::error::Error for TensorError {}

// Add a simple random number generator for GPU buffer IDs
mod rand {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn random<T: Hash + Default>() -> u64 {
        let mut hasher = DefaultHasher::new();
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos().hash(&mut hasher);
        hasher.finish()
    }
}