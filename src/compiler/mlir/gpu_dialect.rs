// GPU-specific MLIR dialect for Aether
// Handles CUDA and SPIR-V kernel generation from MLIR GPU dialects

use crate::compiler::mlir::mlir_context::{MLIRError, MLIRContext, MLIROperation, MLIRValue, MLIRType, MLIRAttribute};


/// GPU target platform
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuTarget {
    /// NVIDIA CUDA
    Cuda,
    /// AMD/Intel SPIR-V
    SpirV,
    /// WebGPU compute shaders
    WebGpu,
}

/// GPU kernel launch configuration
#[derive(Debug, Clone)]
pub struct KernelLaunchConfig {
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
    pub shared_memory_size: u32,
    pub stream: Option<String>,
}

/// GPU memory management configuration
#[derive(Debug, Clone)]
pub struct GpuMemoryConfig {
    pub memory_type: GpuMemoryType,
    pub alignment: u32,
    pub coalescing_hint: bool,
    pub cache_hint: GpuCacheHint,
}

/// GPU memory types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuMemoryType {
    /// Global device memory
    Global,
    /// Shared memory within thread block
    Shared,
    /// Constant memory (read-only)
    Constant,
    /// Texture memory
    Texture,
    /// Local/private memory
    Local,
}

/// GPU cache hints for memory access optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuCacheHint {
    /// No specific cache hint
    None,
    /// Cache at all levels
    CacheAll,
    /// Cache at global level only
    CacheGlobal,
    /// Streaming access (don't cache)
    Streaming,
}

/// GPU-specific MLIR operations
pub struct GpuOps;

impl GpuOps {
    /// Create GPU kernel function
    pub fn kernel_func(
        _context: &MLIRContext,
        name: &str,
        params: &[(&str, &str)], // (name, type) pairs
        target: GpuTarget,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("gpu.func".to_string());
        op.add_attribute("sym_name".to_string(), MLIRAttribute::String(name.to_string()));
        op.add_attribute("target".to_string(), MLIRAttribute::String(target.as_str().to_string()));
        op.add_attribute("kernel".to_string(), MLIRAttribute::Boolean(true));
        
        for (param_name, param_type) in params {
            op.add_attribute(
                format!("param_{}", param_name), 
                MLIRAttribute::String(param_type.to_string())
            );
        }
        
        Ok(op)
    }

    /// Create GPU kernel launch operation
    pub fn kernel_launch(
        _context: &MLIRContext,
        kernel_name: &str,
        config: &KernelLaunchConfig,
        args: &[&str],
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("gpu.launch_func".to_string());
        op.add_attribute("kernel".to_string(), MLIRAttribute::String(kernel_name.to_string()));
        op.add_attribute("gridSizeX".to_string(), MLIRAttribute::Integer(config.grid_size.0 as i64));
        op.add_attribute("gridSizeY".to_string(), MLIRAttribute::Integer(config.grid_size.1 as i64));
        op.add_attribute("gridSizeZ".to_string(), MLIRAttribute::Integer(config.grid_size.2 as i64));
        op.add_attribute("blockSizeX".to_string(), MLIRAttribute::Integer(config.block_size.0 as i64));
        op.add_attribute("blockSizeY".to_string(), MLIRAttribute::Integer(config.block_size.1 as i64));
        op.add_attribute("blockSizeZ".to_string(), MLIRAttribute::Integer(config.block_size.2 as i64));
        op.add_attribute("dynamicSharedMemorySize".to_string(), MLIRAttribute::Integer(config.shared_memory_size as i64));
        
        for arg in args {
            let operand = MLIRValue::new(arg.to_string(), MLIRType::Index);
            op.add_operand(operand);
        }
        
        Ok(op)
    }

    /// Create GPU memory allocation
    pub fn gpu_alloc(
        _context: &MLIRContext,
        size: &str,
        config: &GpuMemoryConfig,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("gpu.alloc".to_string());
        let size_operand = MLIRValue::new(size.to_string(), MLIRType::Index);
        op.add_operand(size_operand);
        op.add_attribute("memory_type".to_string(), MLIRAttribute::String(config.memory_type.as_str().to_string()));
        op.add_attribute("alignment".to_string(), MLIRAttribute::Integer(config.alignment as i64));
        op.add_attribute("coalescing".to_string(), MLIRAttribute::Boolean(config.coalescing_hint));
        op.add_attribute("cache_hint".to_string(), MLIRAttribute::String(config.cache_hint.as_str().to_string()));
        
        Ok(op)
    }

    /// Create GPU memory deallocation
    pub fn gpu_dealloc(
        _context: &MLIRContext,
        memref: &str,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("gpu.dealloc".to_string());
        let memref_operand = MLIRValue::new(memref.to_string(), MLIRType::Memref { 
            element_type: Box::new(MLIRType::Float { width: 32 }), 
            shape: vec![-1] 
        });
        op.add_operand(memref_operand);
        
        Ok(op)
    }

    /// Create GPU memory copy operation
    pub fn gpu_memcpy(
        _context: &MLIRContext,
        src: &str,
        dst: &str,
        size: &str,
        direction: GpuMemcpyDirection,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("gpu.memcpy".to_string());
        let memref_type = MLIRType::Memref { 
            element_type: Box::new(MLIRType::Float { width: 32 }), 
            shape: vec![-1] 
        };
        op.add_operand(MLIRValue::new(src.to_string(), memref_type.clone()));
        op.add_operand(MLIRValue::new(dst.to_string(), memref_type));
        op.add_operand(MLIRValue::new(size.to_string(), MLIRType::Index));
        op.add_attribute("direction".to_string(), MLIRAttribute::String(direction.as_str().to_string()));
        
        Ok(op)
    }

    /// Create GPU barrier/synchronization
    pub fn gpu_barrier(
        _context: &MLIRContext,
        barrier_type: GpuBarrierType,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("gpu.barrier".to_string());
        op.add_attribute("type".to_string(), MLIRAttribute::String(barrier_type.as_str().to_string()));
        
        Ok(op)
    }

    /// Create GPU thread index access
    pub fn thread_id(
        _context: &MLIRContext,
        dimension: GpuDimension,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("gpu.thread_id".to_string());
        op.add_attribute("dimension".to_string(), MLIRAttribute::String(dimension.as_str().to_string()));
        let result = MLIRValue::new("thread_id_result".to_string(), MLIRType::Index);
        op.add_result(result);
        
        Ok(op)
    }

    /// Create GPU block index access
    pub fn block_id(
        _context: &MLIRContext,
        dimension: GpuDimension,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("gpu.block_id".to_string());
        op.add_attribute("dimension".to_string(), MLIRAttribute::String(dimension.as_str().to_string()));
        let result = MLIRValue::new("block_id_result".to_string(), MLIRType::Index);
        op.add_result(result);
        
        Ok(op)
    }

    /// Create GPU grid/block dimension access
    pub fn grid_dim(
        _context: &MLIRContext,
        dimension: GpuDimension,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("gpu.grid_dim".to_string());
        op.add_attribute("dimension".to_string(), MLIRAttribute::String(dimension.as_str().to_string()));
        let result = MLIRValue::new("grid_dim_result".to_string(), MLIRType::Index);
        op.add_result(result);
        
        Ok(op)
    }

    /// Create GPU atomic operation
    pub fn atomic_op(
        _context: &MLIRContext,
        op_type: GpuAtomicOp,
        address: &str,
        value: &str,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("gpu.atomic".to_string());
        op.add_attribute("operation".to_string(), MLIRAttribute::String(op_type.as_str().to_string()));
        
        let address_type = MLIRType::Memref { 
            element_type: Box::new(MLIRType::Float { width: 32 }), 
            shape: vec![] 
        };
        op.add_operand(MLIRValue::new(address.to_string(), address_type));
        op.add_operand(MLIRValue::new(value.to_string(), MLIRType::Float { width: 32 }));
        
        let result = MLIRValue::new("atomic_result".to_string(), MLIRType::Float { width: 32 });
        op.add_result(result);
        
        Ok(op)
    }
}

/// GPU memory copy directions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuMemcpyDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost,
}

/// GPU barrier types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuBarrierType {
    /// Synchronize all threads in block
    Block,
    /// Synchronize all blocks in grid
    Grid,
    /// Memory fence
    Memory,
}

/// GPU dimension (x, y, z)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuDimension {
    X,
    Y,
    Z,
}

/// GPU atomic operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuAtomicOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    And,
    Or,
    Xor,
    Exchange,
    CompareAndSwap,
}



impl GpuTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuTarget::Cuda => "cuda",
            GpuTarget::SpirV => "spirv",
            GpuTarget::WebGpu => "webgpu",
        }
    }
}

impl GpuMemoryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuMemoryType::Global => "global",
            GpuMemoryType::Shared => "shared",
            GpuMemoryType::Constant => "constant",
            GpuMemoryType::Texture => "texture",
            GpuMemoryType::Local => "local",
        }
    }
}

impl GpuCacheHint {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuCacheHint::None => "none",
            GpuCacheHint::CacheAll => "cache_all",
            GpuCacheHint::CacheGlobal => "cache_global",
            GpuCacheHint::Streaming => "streaming",
        }
    }
}

impl GpuMemcpyDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuMemcpyDirection::HostToDevice => "host_to_device",
            GpuMemcpyDirection::DeviceToHost => "device_to_host",
            GpuMemcpyDirection::DeviceToDevice => "device_to_device",
            GpuMemcpyDirection::HostToHost => "host_to_host",
        }
    }
}

impl GpuBarrierType {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuBarrierType::Block => "block",
            GpuBarrierType::Grid => "grid",
            GpuBarrierType::Memory => "memory",
        }
    }
}

impl GpuDimension {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuDimension::X => "x",
            GpuDimension::Y => "y",
            GpuDimension::Z => "z",
        }
    }
}

impl GpuAtomicOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuAtomicOp::Add => "add",
            GpuAtomicOp::Sub => "sub",
            GpuAtomicOp::Mul => "mul",
            GpuAtomicOp::Div => "div",
            GpuAtomicOp::Min => "min",
            GpuAtomicOp::Max => "max",
            GpuAtomicOp::And => "and",
            GpuAtomicOp::Or => "or",
            GpuAtomicOp::Xor => "xor",
            GpuAtomicOp::Exchange => "exchange",
            GpuAtomicOp::CompareAndSwap => "compare_and_swap",
        }
    }
}

impl Default for KernelLaunchConfig {
    fn default() -> Self {
        KernelLaunchConfig {
            grid_size: (1, 1, 1),
            block_size: (256, 1, 1),
            shared_memory_size: 0,
            stream: None,
        }
    }
}

impl Default for GpuMemoryConfig {
    fn default() -> Self {
        GpuMemoryConfig {
            memory_type: GpuMemoryType::Global,
            alignment: 256, // Common GPU memory alignment
            coalescing_hint: true,
            cache_hint: GpuCacheHint::CacheAll,
        }
    }
}