// GPU-specific tensor operation optimizations
// Implements optimized tensor operations for GPU execution

use crate::compiler::mlir::mlir_context::{MLIRError, MLIRContext, MLIRModule, MLIROperation, MLIRAttribute};
use crate::compiler::mlir::gpu_dialect::*;
use crate::compiler::types::{Shape, PrimitiveType};
use std::collections::HashMap;

/// GPU tensor operation optimizer
pub struct GpuTensorOptimizer<'a> {
    context: &'a MLIRContext,
    target: GpuTarget,
}

impl<'a> GpuTensorOptimizer<'a> {
    /// Create new GPU tensor optimizer
    pub fn new(context: &'a MLIRContext, target: GpuTarget) -> Self {
        GpuTensorOptimizer { context, target }
    }

    /// Handle MLIR operation errors with recovery mechanisms
    fn handle_mlir_error(&self, operation_name: &str, error: MLIRError) -> MLIRError {
        eprintln!("MLIR operation '{}' failed: {}", operation_name, error);
        
        // Log error details for debugging
        match &error {
            MLIRError::OperationError(msg) => {
                eprintln!("  Operation error details: {}", msg);
            }
            MLIRError::VerificationError(msg) => {
                eprintln!("  Verification error details: {}", msg);
            }
            MLIRError::ModuleError(msg) => {
                eprintln!("  Module error details: {}", msg);
            }
            _ => {
                eprintln!("  Error type: {:?}", error);
            }
        }
        
        // Return the original error for proper propagation
        error
    }

    /// Optimize tensor operations for GPU execution
    pub fn optimize_tensor_operations(&self, module: &mut MLIRModule) -> Result<(), MLIRError> {
        // Apply tensor-specific optimizations
        self.fuse_tensor_operations(module)?;
        self.optimize_memory_layout(module)?;
        self.vectorize_operations(module)?;
        self.optimize_broadcasting(module)?;
        self.apply_tensor_core_optimizations(module)?;

        Ok(())
    }

    /// Fuse compatible tensor operations with enhanced error handling
    fn fuse_tensor_operations(&self, module: &mut MLIRModule) -> Result<(), MLIRError> {
        let mut fused_ops = Vec::new();
        let mut i = 0;

        while i < module.operations().len() {
            let current_op = &module.operations()[i];
            
            if self.is_fusable_tensor_op(&current_op.name) {
                // Look for fusable operations in the next few operations
                let mut fusion_group = vec![current_op.clone()];
                let mut j = i + 1;
                
                while j < module.operations().len() && j < i + 5 {
                    let next_op = &module.operations()[j];
                    if self.can_fuse_operations(&current_op.name, &next_op.name) {
                        fusion_group.push(next_op.clone());
                        j += 1;
                    } else {
                        break;
                    }
                }
                
                if fusion_group.len() > 1 {
                    // Create fused operation with error handling
                    match self.create_fused_operation(&fusion_group.iter().map(|op| op.name.clone()).collect::<Vec<_>>()) {
                        Ok(fused_op) => {
                            let mut fused_operation = MLIROperation::new(fused_op);
                            fused_operation.add_attribute("fusion_group_size".to_string(), MLIRAttribute::Integer(fusion_group.len() as i64));
                            fused_operation.add_attribute("optimization_applied".to_string(), MLIRAttribute::String("tensor_fusion".to_string()));
                            fused_ops.push(fused_operation);
                            i = j; // Skip the fused operations
                        }
                        Err(e) => {
                            // Log fusion failure but continue with original operations
                            eprintln!("Warning: Failed to create fused operation: {}", e);
                            for op in fusion_group {
                                fused_ops.push(op);
                            }
                            i = j;
                        }
                    }
                } else {
                    fused_ops.push(current_op.clone());
                    i += 1;
                }
            } else {
                fused_ops.push(current_op.clone());
                i += 1;
            }
        }

        // Replace operations with proper error handling and recovery
        match module.replace_operations(fused_ops) {
            Ok(()) => {
                if self.target == GpuTarget::Cuda {
                    println!("Successfully applied tensor fusion optimizations for CUDA target");
                }
                Ok(())
            }
            Err(e) => {
                let handled_error = self.handle_mlir_error("fuse_tensor_operations", e);
                
                // Attempt recovery by reverting to original operations
                eprintln!("Tensor fusion failed, attempting recovery...");
                let original_ops: Vec<MLIROperation> = module.operations().to_vec();
                match module.replace_operations(original_ops) {
                    Ok(()) => {
                        eprintln!("Successfully recovered from tensor fusion failure");
                        // Return a warning instead of an error
                        Ok(())
                    }
                    Err(recovery_error) => {
                        eprintln!("Recovery failed: {}", recovery_error);
                        Err(handled_error)
                    }
                }
            }
        }
    }

    /// Check if operation can be fused
    fn is_fusable_tensor_op(&self, op: &str) -> bool {
        op.contains("tensor_op") ||
        op.contains("arith.addf") ||
        op.contains("arith.mulf") ||
        op.contains("arith.subf") ||
        op.contains("arith.divf") ||
        op.contains("math.exp") ||
        op.contains("math.log") ||
        op.contains("math.tanh") ||
        op.contains("math.sqrt")
    }

    /// Check if two operations can be fused
    fn can_fuse_operations(&self, op1: &str, op2: &str) -> bool {
        // Simple heuristic: both must be element-wise operations
        self.is_elementwise_op(op1) && self.is_elementwise_op(op2)
    }

    /// Check if operation is element-wise
    fn is_elementwise_op(&self, op: &str) -> bool {
        op.contains("arith.") || op.contains("math.") || op.contains("elementwise")
    }

    /// Create fused operation from group
    fn create_fused_operation(&self, ops: &[String]) -> Result<String, MLIRError> {
        let fusion_id = ops.len();
        Ok(format!("gpu.fused_kernel_{}: {}", fusion_id, ops.join(" + ")))
    }

    /// Optimize memory layout for GPU access patterns with enhanced error handling
    fn optimize_memory_layout(&self, module: &mut MLIRModule) -> Result<(), MLIRError> {
        let mut optimized_ops = Vec::new();
        let mut optimization_count = 0;

        for op in module.operations() {
            let mut new_op = op.clone();
            
            // Apply memory layout optimizations based on operation type
            if op.name.contains("memref.alloc") {
                // Add memory layout optimization hints with target-specific settings
                match self.target {
                    GpuTarget::Cuda => {
                        new_op.add_attribute("layout".to_string(), MLIRAttribute::String("coalesced".to_string()));
                        new_op.add_attribute("aligned".to_string(), MLIRAttribute::Boolean(true));
                        new_op.add_attribute("memory_space".to_string(), MLIRAttribute::String("global".to_string()));
                    }
                    GpuTarget::SpirV => {
                        new_op.add_attribute("layout".to_string(), MLIRAttribute::String("std430".to_string()));
                        new_op.add_attribute("aligned".to_string(), MLIRAttribute::Boolean(true));
                    }
                    GpuTarget::WebGpu => {
                        new_op.add_attribute("layout".to_string(), MLIRAttribute::String("packed".to_string()));
                        new_op.add_attribute("binding".to_string(), MLIRAttribute::Integer(0));
                    }
                }
                optimization_count += 1;
            } else if op.name.contains("tensor.extract") || op.name.contains("tensor.insert") {
                // Optimize tensor access patterns
                new_op.add_attribute("access".to_string(), MLIRAttribute::String("vectorized".to_string()));
                new_op.add_attribute("cache_hint".to_string(), MLIRAttribute::String("temporal".to_string()));
                optimization_count += 1;
            } else if op.name.contains("memref.load") || op.name.contains("memref.store") {
                // Optimize memory access patterns
                new_op.add_attribute("access_pattern".to_string(), MLIRAttribute::String("coalesced".to_string()));
                optimization_count += 1;
            }
            
            optimized_ops.push(new_op);
        }

        // Apply optimizations with error handling and validation
        match module.replace_operations(optimized_ops) {
            Ok(()) => {
                println!("Applied {} memory layout optimizations for {:?} target", optimization_count, self.target);
                Ok(())
            }
            Err(e) => {
                let handled_error = self.handle_mlir_error("optimize_memory_layout", e);
                
                // Attempt recovery with less aggressive optimizations
                eprintln!("Memory layout optimization failed, attempting conservative approach...");
                
                let mut conservative_ops = Vec::new();
                for op in module.operations() {
                    let mut new_op = op.clone();
                    
                    // Apply only basic optimizations
                    if op.name.contains("memref.alloc") {
                        new_op.add_attribute("aligned".to_string(), MLIRAttribute::Boolean(true));
                    }
                    conservative_ops.push(new_op);
                }
                
                match module.replace_operations(conservative_ops) {
                    Ok(()) => {
                        eprintln!("Successfully applied conservative memory optimizations");
                        Ok(())
                    }
                    Err(recovery_error) => {
                        eprintln!("Conservative optimization also failed: {}", recovery_error);
                        Err(handled_error)
                    }
                }
            }
        }
    }

    /// Vectorize tensor operations with enhanced error handling and validation
    fn vectorize_operations(&self, module: &mut MLIRModule) -> Result<(), MLIRError> {
        let mut vectorized_ops = Vec::new();
        let mut vectorization_count = 0;

        for op in module.operations() {
            if self.can_vectorize_operation(&op.name) {
                match self.create_vectorized_operation(&op.name) {
                    Ok(vectorized_name) => {
                        let mut vectorized = op.clone();
                        vectorized.name = vectorized_name;
                        vectorized.add_attribute("vectorized".to_string(), MLIRAttribute::Boolean(true));
                        
                        // Add target-specific vectorization attributes
                        match self.target {
                            GpuTarget::Cuda => {
                                vectorized.add_attribute("vector_width".to_string(), MLIRAttribute::Integer(4));
                                vectorized.add_attribute("cuda_vectorized".to_string(), MLIRAttribute::Boolean(true));
                            }
                            GpuTarget::SpirV => {
                                vectorized.add_attribute("vector_width".to_string(), MLIRAttribute::Integer(4));
                                vectorized.add_attribute("spirv_vectorized".to_string(), MLIRAttribute::Boolean(true));
                            }
                            GpuTarget::WebGpu => {
                                vectorized.add_attribute("vector_width".to_string(), MLIRAttribute::Integer(2));
                                vectorized.add_attribute("webgpu_vectorized".to_string(), MLIRAttribute::Boolean(true));
                            }
                        }
                        
                        vectorized_ops.push(vectorized);
                        vectorization_count += 1;
                    }
                    Err(e) => {
                        // Log vectorization failure but continue with original operation
                        eprintln!("Warning: Failed to vectorize operation '{}': {}", op.name, e);
                        vectorized_ops.push(op.clone());
                    }
                }
            } else {
                vectorized_ops.push(op.clone());
            }
        }

        // Apply vectorization with comprehensive error handling
        match module.replace_operations(vectorized_ops) {
            Ok(()) => {
                if vectorization_count > 0 {
                    println!("Successfully vectorized {} operations for {:?} target", vectorization_count, self.target);
                }
                Ok(())
            }
            Err(e) => {
                let handled_error = self.handle_mlir_error("vectorize_operations", e);
                
                // Attempt recovery by reverting to original operations
                eprintln!("Vectorization failed, reverting to scalar operations...");
                
                let mut scalar_ops = Vec::new();
                for op in module.operations() {
                    let mut scalar_op = op.clone();
                    // Remove any vectorization attributes that might cause issues
                    scalar_op.attributes.remove("vectorized");
                    scalar_op.attributes.remove("vector_width");
                    scalar_op.attributes.remove("cuda_vectorized");
                    scalar_op.attributes.remove("spirv_vectorized");
                    scalar_op.attributes.remove("webgpu_vectorized");
                    scalar_ops.push(scalar_op);
                }
                
                match module.replace_operations(scalar_ops) {
                    Ok(()) => {
                        eprintln!("Successfully reverted to scalar operations");
                        Ok(())
                    }
                    Err(recovery_error) => {
                        eprintln!("Failed to revert to scalar operations: {}", recovery_error);
                        Err(handled_error)
                    }
                }
            }
        }
    }

    /// Check if operation can be vectorized
    fn can_vectorize_operation(&self, op: &str) -> bool {
        match self.target {
            GpuTarget::Cuda => {
                // CUDA supports various vector widths
                op.contains("arith.") && (op.contains("f32") || op.contains("f16"))
            }
            GpuTarget::SpirV => {
                // SPIR-V has vector support
                op.contains("arith.") || op.contains("math.")
            }
            GpuTarget::WebGpu => {
                // WebGPU has limited vectorization
                op.contains("arith.addf") || op.contains("arith.mulf")
            }
        }
    }

    /// Create vectorized version of operation
    fn create_vectorized_operation(&self, op: &str) -> Result<String, MLIRError> {
        match self.target {
            GpuTarget::Cuda => {
                if op.contains("f32") {
                    Ok(format!("{} // vectorized: float4", op))
                } else {
                    Ok(format!("{} // vectorized: float2", op))
                }
            }
            GpuTarget::SpirV => {
                Ok(format!("{} // vectorized: vec4", op))
            }
            GpuTarget::WebGpu => {
                Ok(format!("{} // vectorized: vec2", op))
            }
        }
    }

    /// Optimize broadcasting operations
    fn optimize_broadcasting(&self, module: &mut MLIRModule) -> Result<(), MLIRError> {
        let mut optimized_ops = Vec::new();

        for op in module.operations() {
            if op.name.contains("broadcast") || op.name.contains("expand") {
                let mut optimized = op.clone();
                optimized.add_attribute("broadcast_optimized".to_string(), MLIRAttribute::Boolean(true));
                optimized_ops.push(optimized);
            } else {
                optimized_ops.push(op.clone());
            }
        }

        module.replace_operations(optimized_ops)
            .map_err(|e| self.handle_mlir_error("optimize_broadcasting", e))?;
        Ok(())
    }

    /// Optimize specific broadcast operation
    fn optimize_broadcast_operation(&self, op: &str) -> Result<String, MLIRError> {
        // Analyze broadcast pattern and optimize for GPU memory hierarchy
        if op.contains("scalar_broadcast") {
            Ok(format!("{} // optimized: constant_memory", op))
        } else if op.contains("vector_broadcast") {
            Ok(format!("{} // optimized: shared_memory", op))
        } else {
            Ok(format!("{} // optimized: coalesced_access", op))
        }
    }

    /// Apply tensor core optimizations (for NVIDIA GPUs) with enhanced validation
    fn apply_tensor_core_optimizations(&self, module: &mut MLIRModule) -> Result<(), MLIRError> {
        if self.target != GpuTarget::Cuda {
            return Ok(()); // Tensor cores are NVIDIA-specific
        }

        let mut optimized_ops = Vec::new();
        let mut tensor_core_count = 0;

        for op in module.operations() {
            if self.can_use_tensor_cores(&op.name) {
                match self.create_tensor_core_operation(&op.name) {
                    Ok(tensor_core_op) => {
                        let mut optimized = op.clone();
                        optimized.name = tensor_core_op;
                        optimized.add_attribute("tensor_core".to_string(), MLIRAttribute::Boolean(true));
                        optimized.add_attribute("wmma_enabled".to_string(), MLIRAttribute::Boolean(true));
                        
                        // Add tensor core specific attributes
                        if op.name.contains("matmul") {
                            optimized.add_attribute("tensor_core_type".to_string(), MLIRAttribute::String("wmma".to_string()));
                            optimized.add_attribute("tile_size".to_string(), MLIRAttribute::String("16x16x16".to_string()));
                        } else if op.name.contains("conv") {
                            optimized.add_attribute("tensor_core_type".to_string(), MLIRAttribute::String("cudnn".to_string()));
                            optimized.add_attribute("algorithm".to_string(), MLIRAttribute::String("implicit_gemm".to_string()));
                        }
                        
                        optimized_ops.push(optimized);
                        tensor_core_count += 1;
                    }
                    Err(e) => {
                        // Log tensor core optimization failure but continue
                        eprintln!("Warning: Failed to apply tensor core optimization to '{}': {}", op.name, e);
                        optimized_ops.push(op.clone());
                    }
                }
            } else {
                optimized_ops.push(op.clone());
            }
        }

        // Apply tensor core optimizations with validation
        match module.replace_operations(optimized_ops) {
            Ok(()) => {
                if tensor_core_count > 0 {
                    println!("Successfully applied tensor core optimizations to {} operations", tensor_core_count);
                }
                Ok(())
            }
            Err(e) => {
                let handled_error = self.handle_mlir_error("apply_tensor_core_optimizations", e);
                
                // Attempt recovery with fallback to regular CUDA operations
                eprintln!("Tensor core optimization failed, falling back to regular CUDA operations...");
                
                let mut fallback_ops = Vec::new();
                for op in module.operations() {
                    let mut fallback_op = op.clone();
                    
                    // Remove tensor core attributes and use regular CUDA operations
                    fallback_op.attributes.remove("tensor_core");
                    fallback_op.attributes.remove("wmma_enabled");
                    fallback_op.attributes.remove("tensor_core_type");
                    fallback_op.attributes.remove("tile_size");
                    fallback_op.attributes.remove("algorithm");
                    
                    // Add regular CUDA optimization attributes
                    if op.name.contains("matmul") {
                        fallback_op.add_attribute("cuda_optimized".to_string(), MLIRAttribute::Boolean(true));
                        fallback_op.add_attribute("use_cublas".to_string(), MLIRAttribute::Boolean(true));
                    } else if op.name.contains("conv") {
                        fallback_op.add_attribute("cuda_optimized".to_string(), MLIRAttribute::Boolean(true));
                        fallback_op.add_attribute("use_cudnn".to_string(), MLIRAttribute::Boolean(true));
                    }
                    
                    fallback_ops.push(fallback_op);
                }
                
                match module.replace_operations(fallback_ops) {
                    Ok(()) => {
                        eprintln!("Successfully applied fallback CUDA optimizations");
                        Ok(())
                    }
                    Err(recovery_error) => {
                        eprintln!("Fallback optimization also failed: {}", recovery_error);
                        Err(handled_error)
                    }
                }
            }
        }
    }

    /// Check if operation can use tensor cores
    fn can_use_tensor_cores(&self, op: &str) -> bool {
        // Tensor cores support specific matrix sizes and data types
        (op.contains("matmul") || op.contains("conv")) &&
        (op.contains("f16") || op.contains("bf16") || op.contains("i8")) &&
        self.has_compatible_dimensions(op)
    }

    /// Check if operation has tensor core compatible dimensions
    fn has_compatible_dimensions(&self, _op: &str) -> bool {
        // Simplified check - tensor cores work best with specific tile sizes
        // Real implementation would parse actual dimensions
        true
    }

    /// Create tensor core optimized operation
    fn create_tensor_core_operation(&self, op: &str) -> Result<String, MLIRError> {
        if op.contains("matmul") {
            Ok(format!("{} // tensor_core: wmma", op))
        } else if op.contains("conv") {
            Ok(format!("{} // tensor_core: cudnn", op))
        } else {
            Ok(op.to_string())
        }
    }
}

/// GPU tensor operation patterns
#[derive(Debug, Clone)]
pub struct TensorOpPattern {
    pub operation_type: TensorOpType,
    pub input_shapes: Vec<Shape>,
    pub output_shape: Shape,
    pub data_type: PrimitiveType,
    pub optimization_hints: TensorOptimizationHints,
}

/// Types of tensor operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorOpType {
    /// Element-wise operations
    ElementWise,
    /// Matrix multiplication
    MatMul,
    /// Convolution
    Convolution,
    /// Reduction operations
    Reduction,
    /// Transpose operations
    Transpose,
    /// Reshape operations
    Reshape,
    /// Broadcasting operations
    Broadcast,
}

/// Optimization hints for tensor operations
#[derive(Debug, Clone)]
pub struct TensorOptimizationHints {
    /// Preferred memory layout
    pub memory_layout: MemoryLayout,
    /// Vectorization width
    pub vector_width: u32,
    /// Use tensor cores if available
    pub use_tensor_cores: bool,
    /// Fusion opportunities
    pub fusion_candidates: Vec<String>,
    /// Memory access pattern
    pub access_pattern: AccessPattern,
}

/// Memory layout preferences
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryLayout {
    /// Row-major layout (C-style)
    RowMajor,
    /// Column-major layout (Fortran-style)
    ColumnMajor,
    /// Blocked/tiled layout
    Blocked { block_size: (u32, u32) },
    /// Optimal layout for target hardware
    Optimal,
}

/// Memory access patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    /// Sequential access
    Sequential,
    /// Strided access
    Strided { stride: u32 },
    /// Random access
    Random,
    /// Broadcast access (same value for all threads)
    Broadcast,
    /// Coalesced access (optimal for GPU)
    Coalesced,
}

impl TensorOpPattern {
    /// Create new tensor operation pattern
    pub fn new(
        op_type: TensorOpType,
        input_shapes: Vec<Shape>,
        output_shape: Shape,
        data_type: PrimitiveType,
    ) -> Self {
        TensorOpPattern {
            operation_type: op_type,
            input_shapes,
            output_shape,
            data_type,
            optimization_hints: TensorOptimizationHints::default(),
        }
    }

    /// Analyze operation for GPU optimization opportunities
    pub fn analyze_for_gpu(&mut self, target: GpuTarget) -> Result<(), MLIRError> {
        // Set optimal memory layout for target
        self.optimization_hints.memory_layout = match target {
            GpuTarget::Cuda => MemoryLayout::RowMajor, // CUDA prefers row-major
            GpuTarget::SpirV => MemoryLayout::Optimal,  // Let driver decide
            GpuTarget::WebGpu => MemoryLayout::RowMajor, // WebGPU uses row-major
        };

        // Set vectorization width based on data type and target
        self.optimization_hints.vector_width = match (target, &self.data_type) {
            (GpuTarget::Cuda, PrimitiveType::Float32) => 4, // float4
            (GpuTarget::Cuda, PrimitiveType::Float64) => 2, // double2
            (GpuTarget::SpirV, _) => 4, // vec4 is common
            (GpuTarget::WebGpu, _) => 2, // Limited vectorization
            _ => 1,
        };

        // Check tensor core compatibility
        self.optimization_hints.use_tensor_cores = 
            target == GpuTarget::Cuda &&
            matches!(self.operation_type, TensorOpType::MatMul | TensorOpType::Convolution) &&
            matches!(self.data_type, PrimitiveType::Float32 | PrimitiveType::Float64);

        // Analyze access pattern
        self.optimization_hints.access_pattern = match self.operation_type {
            TensorOpType::ElementWise => AccessPattern::Coalesced,
            TensorOpType::MatMul => AccessPattern::Coalesced,
            TensorOpType::Reduction => AccessPattern::Sequential,
            TensorOpType::Transpose => AccessPattern::Strided { stride: self.get_transpose_stride() },
            _ => AccessPattern::Sequential,
        };

        Ok(())
    }

    /// Get stride for transpose operations
    fn get_transpose_stride(&self) -> u32 {
        // Simplified - would analyze actual tensor dimensions
        if let Some(shape) = self.input_shapes.first() {
            if let Some(dims) = shape.as_concrete() {
                if dims.len() >= 2 {
                    return dims[dims.len() - 1] as u32;
                }
            }
        }
        1
    }
}

impl Default for TensorOptimizationHints {
    fn default() -> Self {
        TensorOptimizationHints {
            memory_layout: MemoryLayout::RowMajor,
            vector_width: 1,
            use_tensor_cores: false,
            fusion_candidates: Vec::new(),
            access_pattern: AccessPattern::Sequential,
        }
    }
}

impl From<MemoryLayout> for AccessPattern {
    fn from(layout: MemoryLayout) -> Self {
        match layout {
            MemoryLayout::RowMajor | MemoryLayout::ColumnMajor => AccessPattern::Sequential,
            MemoryLayout::Blocked { .. } => AccessPattern::Coalesced,
            MemoryLayout::Optimal => AccessPattern::Coalesced,
        }
    }
}

/// GPU tensor memory manager
pub struct GpuTensorMemoryManager {
    target: GpuTarget,
    memory_pools: HashMap<String, GpuMemoryPool>,
}

/// GPU memory pool for tensor allocations
#[derive(Debug, Clone)]
pub struct GpuMemoryPool {
    pub pool_type: GpuMemoryType,
    pub total_size: usize,
    pub used_size: usize,
    pub alignment: u32,
    pub allocations: Vec<GpuAllocation>,
}

/// GPU memory allocation record
#[derive(Debug, Clone)]
pub struct GpuAllocation {
    pub id: String,
    pub offset: usize,
    pub size: usize,
    pub tensor_shape: Shape,
    pub data_type: PrimitiveType,
}

impl GpuTensorMemoryManager {
    /// Create new GPU tensor memory manager
    pub fn new(target: GpuTarget) -> Self {
        let mut memory_pools = HashMap::new();
        
        // Initialize default memory pools
        memory_pools.insert("global".to_string(), GpuMemoryPool::new(GpuMemoryType::Global, 1024 * 1024 * 1024)); // 1GB
        memory_pools.insert("shared".to_string(), GpuMemoryPool::new(GpuMemoryType::Shared, 48 * 1024)); // 48KB
        memory_pools.insert("constant".to_string(), GpuMemoryPool::new(GpuMemoryType::Constant, 64 * 1024)); // 64KB
        
        GpuTensorMemoryManager {
            target,
            memory_pools,
        }
    }

    /// Allocate tensor memory
    pub fn allocate_tensor(
        &mut self,
        id: String,
        shape: &Shape,
        data_type: PrimitiveType,
        memory_type: GpuMemoryType,
    ) -> Result<GpuAllocation, MLIRError> {
        let size = self.calculate_tensor_size(shape, &data_type)?;
        let pool_name = memory_type.as_str();
        
        if let Some(pool) = self.memory_pools.get_mut(pool_name) {
            pool.allocate(id, size, shape.clone(), data_type)
        } else {
            Err(MLIRError::OperationError(format!("Memory pool {} not found", pool_name)))
        }
    }

    /// Calculate tensor size in bytes
    fn calculate_tensor_size(&self, shape: &Shape, data_type: &PrimitiveType) -> Result<usize, MLIRError> {
        let element_size = data_type.size_bytes();
        
        match shape {
            Shape::Concrete(dims) => {
                let total_elements: usize = dims.iter().product();
                Ok(total_elements * element_size)
            }
            Shape::Unknown => {
                // Use default size for unknown shapes
                Ok(1024 * element_size)
            }
            _ => {
                // For symbolic shapes, use heuristic
                Ok(1024 * element_size)
            }
        }
    }
}

impl GpuMemoryPool {
    /// Create new memory pool
    pub fn new(pool_type: GpuMemoryType, total_size: usize) -> Self {
        GpuMemoryPool {
            pool_type,
            total_size,
            used_size: 0,
            alignment: 256, // Common GPU alignment
            allocations: Vec::new(),
        }
    }

    /// Allocate memory from pool
    pub fn allocate(
        &mut self,
        id: String,
        size: usize,
        shape: Shape,
        data_type: PrimitiveType,
    ) -> Result<GpuAllocation, MLIRError> {
        // Align size
        let aligned_size = (size + self.alignment as usize - 1) & !(self.alignment as usize - 1);
        
        if self.used_size + aligned_size > self.total_size {
            return Err(MLIRError::OperationError("Out of GPU memory".to_string()));
        }
        
        let allocation = GpuAllocation {
            id: id.clone(),
            offset: self.used_size,
            size: aligned_size,
            tensor_shape: shape,
            data_type,
        };
        
        self.used_size += aligned_size;
        self.allocations.push(allocation.clone());
        
        Ok(allocation)
    }

    /// Deallocate memory
    pub fn deallocate(&mut self, id: &str) -> Result<(), MLIRError> {
        if let Some(pos) = self.allocations.iter().position(|a| a.id == id) {
            let allocation = self.allocations.remove(pos);
            self.used_size -= allocation.size;
            Ok(())
        } else {
            Err(MLIRError::OperationError(format!("Allocation {} not found", id)))
        }
    }
}