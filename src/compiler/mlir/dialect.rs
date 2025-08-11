// Aether-specific MLIR dialect definition
// Defines operations and types specific to Aether language constructs

use crate::compiler::mlir::{MLIRError, MLIRContext, MLIROperation, MLIRValue, MLIRType, MLIRAttribute};
use crate::compiler::types::LinearOwnership;
use std::collections::HashMap;

/// Aether MLIR dialect
pub struct AetherDialect<'a> {
    context: &'a MLIRContext,
}

impl<'a> AetherDialect<'a> {
    /// Create new Aether dialect
    pub fn new(context: &'a MLIRContext) -> Self {
        AetherDialect { context }
    }

    /// Register Aether dialect with MLIR context
    pub fn register(&self) -> Result<(), MLIRError> {
        // In a real implementation, this would register the dialect with MLIR
        // For now, we'll simulate successful registration
        Ok(())
    }
}

/// Aether MLIR type system integration
#[derive(Debug, Clone)]
pub enum AetherMLIRType {
    /// Tensor type with shape information and differentiability
    Tensor {
        element_type: Box<MLIRType>,
        shape: Vec<i64>,
        is_differentiable: bool,
        device: String,
    },
    /// Probabilistic variable type
    ProbabilisticVariable {
        distribution: DistributionType,
        value_type: Box<MLIRType>,
    },
    /// Linear type for resource management
    LinearType {
        inner_type: Box<MLIRType>,
        ownership_info: LinearOwnershipInfo,
    },
    /// Actor type for concurrency
    ActorType {
        message_type: Box<MLIRType>,
        state_type: Box<MLIRType>,
    },
    /// Function type with Aether-specific attributes
    Function {
        inputs: Vec<MLIRType>,
        outputs: Vec<MLIRType>,
        attributes: AetherFunctionAttributes,
    },
}

/// Distribution types for probabilistic programming
#[derive(Debug, Clone)]
pub enum DistributionType {
    Normal { mean: f64, std: f64 },
    Uniform { min: f64, max: f64 },
    Bernoulli { p: f64 },
    Categorical { probs: Vec<f64> },
    Custom(String),
}

/// Linear ownership information
#[derive(Debug, Clone)]
pub struct LinearOwnershipInfo {
    pub ownership: LinearOwnership,
    pub allocation_site: Option<String>,
    pub lifetime: Option<String>,
}

/// Aether function attributes
#[derive(Debug, Clone)]
pub struct AetherFunctionAttributes {
    pub is_differentiable: bool,
    pub is_gpu_kernel: bool,
    pub is_actor_method: bool,
    pub optimization_hints: Vec<String>,
}

/// Aether-specific MLIR operations
pub struct AetherOps;

impl AetherOps {
    // ===== TENSOR OPERATIONS =====
    
    /// Create tensor creation operation
    pub fn tensor_create(
        _context: &MLIRContext,
        element_type: MLIRType,
        shape: &[i64],
        device: &str,
        is_differentiable: bool,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.tensor_create".to_string());
        
        // Add shape as attribute
        let shape_attr = MLIRAttribute::Array(
            shape.iter().map(|&dim| MLIRAttribute::Integer(dim)).collect()
        );
        op.add_attribute("shape".to_string(), shape_attr);
        
        // Add device attribute
        op.add_attribute("device".to_string(), MLIRAttribute::String(device.to_string()));
        
        // Add differentiability attribute
        op.add_attribute("is_differentiable".to_string(), MLIRAttribute::Boolean(is_differentiable));
        
        // Add result type
        let result_type = MLIRType::AetherTensor {
            element_type: Box::new(element_type),
            shape: shape.to_vec(),
            device: device.to_string(),
        };
        let result = MLIRValue::new("tensor_result".to_string(), result_type);
        op.add_result(result);
        
        Ok(op)
    }
    
    /// Create generic tensor operation
    pub fn tensor_op(
        _context: &MLIRContext,
        op_name: &str,
        operands: Vec<MLIRValue>,
        result_type: MLIRType,
        attributes: HashMap<String, MLIRAttribute>,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.tensor_op".to_string());
        
        // Add operation name
        op.add_attribute("op_name".to_string(), MLIRAttribute::String(op_name.to_string()));
        
        // Add operands
        for operand in operands {
            op.add_operand(operand);
        }
        
        // Add custom attributes
        for (key, value) in attributes {
            op.add_attribute(key, value);
        }
        
        // Add result
        let result = MLIRValue::new("tensor_op_result".to_string(), result_type);
        op.add_result(result);
        
        Ok(op)
    }
    
    /// Create matrix multiplication operation
    pub fn matmul(
        _context: &MLIRContext,
        lhs: MLIRValue,
        rhs: MLIRValue,
        result_type: MLIRType,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.matmul".to_string());
        
        // Add operands
        op.add_operand(lhs);
        op.add_operand(rhs);
        
        // Add transpose attributes
        op.add_attribute("transpose_a".to_string(), MLIRAttribute::Boolean(transpose_a));
        op.add_attribute("transpose_b".to_string(), MLIRAttribute::Boolean(transpose_b));
        
        // Add result
        let result = MLIRValue::new("matmul_result".to_string(), result_type);
        op.add_result(result);
        
        Ok(op)
    }
    
    // ===== AUTOMATIC DIFFERENTIATION OPERATIONS =====
    
    /// Create forward-mode automatic differentiation operation
    pub fn autodiff_forward(
        _context: &MLIRContext,
        function: MLIRValue,
        input: MLIRValue,
        tangent: MLIRValue,
        result_type: MLIRType,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.autodiff_forward".to_string());
        
        // Add operands
        op.add_operand(function);
        op.add_operand(input);
        op.add_operand(tangent);
        
        // Add mode attribute
        op.add_attribute("mode".to_string(), MLIRAttribute::String("forward".to_string()));
        
        // Add result (value and derivative)
        let result = MLIRValue::new("forward_result".to_string(), result_type);
        op.add_result(result);
        
        Ok(op)
    }
    
    /// Create reverse-mode automatic differentiation operation
    pub fn autodiff_reverse(
        _context: &MLIRContext,
        function: MLIRValue,
        input: MLIRValue,
        result_type: MLIRType,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.autodiff_reverse".to_string());
        
        // Add operands
        op.add_operand(function);
        op.add_operand(input);
        
        // Add mode attribute
        op.add_attribute("mode".to_string(), MLIRAttribute::String("reverse".to_string()));
        
        // Add result (value and gradient function)
        let result = MLIRValue::new("reverse_result".to_string(), result_type);
        op.add_result(result);
        
        Ok(op)
    }
    
    /// Create gradient computation operation
    pub fn gradient(
        _context: &MLIRContext,
        function: MLIRValue,
        input: MLIRValue,
        result_type: MLIRType,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.gradient".to_string());
        
        // Add operands
        op.add_operand(function);
        op.add_operand(input);
        
        // Add result (gradient tensor)
        let result = MLIRValue::new("gradient_result".to_string(), result_type);
        op.add_result(result);
        
        Ok(op)
    }
    
    // ===== PROBABILISTIC PROGRAMMING OPERATIONS =====
    
    /// Create probabilistic variable declaration
    pub fn prob_var(
        _context: &MLIRContext,
        name: &str,
        distribution: DistributionType,
        parameters: Vec<MLIRValue>,
        result_type: MLIRType,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.prob_var".to_string());
        
        // Add variable name
        op.add_attribute("name".to_string(), MLIRAttribute::String(name.to_string()));
        
        // Add distribution type
        let dist_attr = match distribution {
            DistributionType::Normal { mean, std } => {
                let mut dict = HashMap::new();
                dict.insert("type".to_string(), MLIRAttribute::String("normal".to_string()));
                dict.insert("mean".to_string(), MLIRAttribute::Float(mean));
                dict.insert("std".to_string(), MLIRAttribute::Float(std));
                MLIRAttribute::Dictionary(dict)
            }
            DistributionType::Uniform { min, max } => {
                let mut dict = HashMap::new();
                dict.insert("type".to_string(), MLIRAttribute::String("uniform".to_string()));
                dict.insert("min".to_string(), MLIRAttribute::Float(min));
                dict.insert("max".to_string(), MLIRAttribute::Float(max));
                MLIRAttribute::Dictionary(dict)
            }
            DistributionType::Bernoulli { p } => {
                let mut dict = HashMap::new();
                dict.insert("type".to_string(), MLIRAttribute::String("bernoulli".to_string()));
                dict.insert("p".to_string(), MLIRAttribute::Float(p));
                MLIRAttribute::Dictionary(dict)
            }
            DistributionType::Categorical { probs } => {
                let mut dict = HashMap::new();
                dict.insert("type".to_string(), MLIRAttribute::String("categorical".to_string()));
                let probs_attr = MLIRAttribute::Array(
                    probs.iter().map(|&p| MLIRAttribute::Float(p)).collect()
                );
                dict.insert("probs".to_string(), probs_attr);
                MLIRAttribute::Dictionary(dict)
            }
            DistributionType::Custom(name) => {
                let mut dict = HashMap::new();
                dict.insert("type".to_string(), MLIRAttribute::String("custom".to_string()));
                dict.insert("name".to_string(), MLIRAttribute::String(name));
                MLIRAttribute::Dictionary(dict)
            }
        };
        op.add_attribute("distribution".to_string(), dist_attr);
        
        // Add parameters
        for param in parameters {
            op.add_operand(param);
        }
        
        // Add result
        let result = MLIRValue::new("prob_var_result".to_string(), result_type);
        op.add_result(result);
        
        Ok(op)
    }
    
    /// Create sampling operation
    pub fn sample(
        _context: &MLIRContext,
        prob_var: MLIRValue,
        result_type: MLIRType,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.sample".to_string());
        
        // Add probabilistic variable operand
        op.add_operand(prob_var);
        
        // Add result
        let result = MLIRValue::new("sample_result".to_string(), result_type);
        op.add_result(result);
        
        Ok(op)
    }
    
    /// Create observation operation
    pub fn observe(
        _context: &MLIRContext,
        prob_var: MLIRValue,
        observed_value: MLIRValue,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.observe".to_string());
        
        // Add operands
        op.add_operand(prob_var);
        op.add_operand(observed_value);
        
        // No result for observe operation (side effect)
        
        Ok(op)
    }
    
    // ===== MEMORY MANAGEMENT OPERATIONS =====
    
    /// Create linear allocation operation
    pub fn linear_alloc(
        _context: &MLIRContext,
        inner_type: MLIRType,
        size: Option<MLIRValue>,
        allocation_site: &str,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.linear_alloc".to_string());
        
        // Add allocation site
        op.add_attribute("allocation_site".to_string(), MLIRAttribute::String(allocation_site.to_string()));
        
        // Add size operand if provided
        if let Some(size_val) = size {
            op.add_operand(size_val);
        }
        
        // Add result with linear type
        let result_type = MLIRType::AetherLinear {
            inner_type: Box::new(inner_type),
        };
        let result = MLIRValue::new("linear_alloc_result".to_string(), result_type);
        op.add_result(result);
        
        Ok(op)
    }
    
    /// Create linear move operation
    pub fn linear_move(
        _context: &MLIRContext,
        source: MLIRValue,
        result_type: MLIRType,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.linear_move".to_string());
        
        // Add source operand
        op.add_operand(source);
        
        // Add result
        let result = MLIRValue::new("linear_move_result".to_string(), result_type);
        op.add_result(result);
        
        Ok(op)
    }
    
    /// Create linear drop operation
    pub fn linear_drop(
        _context: &MLIRContext,
        value: MLIRValue,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.linear_drop".to_string());
        
        // Add value operand
        op.add_operand(value);
        
        // No result for drop operation (side effect)
        
        Ok(op)
    }
    
    // ===== CONCURRENCY OPERATIONS =====
    
    /// Create actor spawn operation
    pub fn spawn_actor(
        _context: &MLIRContext,
        actor_type: MLIRType,
        initial_state: Option<MLIRValue>,
        result_type: MLIRType,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.spawn_actor".to_string());
        
        // Add actor type attribute
        op.add_attribute("actor_type".to_string(), MLIRAttribute::String(format!("{:?}", actor_type)));
        
        // Add initial state if provided
        if let Some(state) = initial_state {
            op.add_operand(state);
        }
        
        // Add result (actor reference)
        let result = MLIRValue::new("actor_ref".to_string(), result_type);
        op.add_result(result);
        
        Ok(op)
    }
    
    /// Create message send operation
    pub fn send_message(
        _context: &MLIRContext,
        actor_ref: MLIRValue,
        message: MLIRValue,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.send_message".to_string());
        
        // Add operands
        op.add_operand(actor_ref);
        op.add_operand(message);
        
        // No result for send operation (side effect)
        
        Ok(op)
    }
    
    /// Create parallel for operation
    pub fn parallel_for(
        _context: &MLIRContext,
        lower_bound: MLIRValue,
        upper_bound: MLIRValue,
        step: Option<MLIRValue>,
        body_function: MLIRValue,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.parallel_for".to_string());
        
        // Add bounds
        op.add_operand(lower_bound);
        op.add_operand(upper_bound);
        
        // Add step if provided
        if let Some(step_val) = step {
            op.add_operand(step_val);
        }
        
        // Add body function
        op.add_operand(body_function);
        
        // No result for parallel_for (side effect)
        
        Ok(op)
    }
}

/// Automatic differentiation modes
#[derive(Debug, Clone, Copy)]
pub enum AutodiffMode {
    Forward,
    Reverse,
    Both,
}

impl AutodiffMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            AutodiffMode::Forward => "forward",
            AutodiffMode::Reverse => "reverse",
            AutodiffMode::Both => "both",
        }
    }
}

/// Operation builder utilities for Aether dialect
pub struct AetherOperationBuilder {
    context: *const MLIRContext,
}

impl AetherOperationBuilder {
    /// Create new operation builder
    pub fn new(context: &MLIRContext) -> Self {
        AetherOperationBuilder {
            context: context as *const MLIRContext,
        }
    }
    
    /// Build tensor creation operation with validation
    pub fn build_tensor_create(
        &self,
        element_type: MLIRType,
        shape: &[i64],
        device: &str,
        is_differentiable: bool,
    ) -> Result<MLIROperation, MLIRError> {
        // Validate shape
        if shape.is_empty() {
            return Err(MLIRError::OperationError("Tensor shape cannot be empty".to_string()));
        }
        
        for &dim in shape {
            if dim <= 0 {
                return Err(MLIRError::OperationError(format!("Invalid tensor dimension: {}", dim)));
            }
        }
        
        // Validate device
        let valid_devices = ["cpu", "gpu", "cuda", "opencl"];
        if !valid_devices.contains(&device) {
            return Err(MLIRError::OperationError(format!("Invalid device: {}", device)));
        }
        
        let context = unsafe { &*self.context };
        AetherOps::tensor_create(context, element_type, shape, device, is_differentiable)
    }
    
    /// Build matrix multiplication with shape validation
    pub fn build_matmul(
        &self,
        lhs: MLIRValue,
        rhs: MLIRValue,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<MLIROperation, MLIRError> {
        // Validate operand types are tensors
        if !lhs.value_type.is_tensor() || !rhs.value_type.is_tensor() {
            return Err(MLIRError::OperationError("Matrix multiplication requires tensor operands".to_string()));
        }
        
        // TODO: Add shape compatibility validation
        
        // Infer result type based on operand shapes
        let result_type = self.infer_matmul_result_type(&lhs.value_type, &rhs.value_type, transpose_a, transpose_b)?;
        
        let context = unsafe { &*self.context };
        AetherOps::matmul(context, lhs, rhs, result_type, transpose_a, transpose_b)
    }
    
    /// Build probabilistic variable with distribution validation
    pub fn build_prob_var(
        &self,
        name: &str,
        distribution: DistributionType,
        parameters: Vec<MLIRValue>,
    ) -> Result<MLIROperation, MLIRError> {
        // Validate distribution parameters
        match &distribution {
            DistributionType::Normal { mean: _, std } => {
                if *std <= 0.0 {
                    return Err(MLIRError::OperationError("Normal distribution std must be positive".to_string()));
                }
            }
            DistributionType::Uniform { min, max } => {
                if min >= max {
                    return Err(MLIRError::OperationError("Uniform distribution min must be less than max".to_string()));
                }
            }
            DistributionType::Bernoulli { p } => {
                if *p < 0.0 || *p > 1.0 {
                    return Err(MLIRError::OperationError("Bernoulli probability must be between 0 and 1".to_string()));
                }
            }
            DistributionType::Categorical { probs } => {
                if probs.is_empty() {
                    return Err(MLIRError::OperationError("Categorical distribution must have at least one probability".to_string()));
                }
                let sum: f64 = probs.iter().sum();
                if (sum - 1.0).abs() > 1e-6 {
                    return Err(MLIRError::OperationError("Categorical probabilities must sum to 1".to_string()));
                }
            }
            DistributionType::Custom(_) => {
                // Custom distributions are validated at runtime
            }
        }
        
        // Infer result type
        let result_type = MLIRType::AetherProbabilistic {
            distribution: format!("{:?}", distribution),
            inner_type: Box::new(MLIRType::Float { width: 64 }), // Default to f64
        };
        
        let context = unsafe { &*self.context };
        AetherOps::prob_var(context, name, distribution, parameters, result_type)
    }
    
    /// Infer result type for matrix multiplication
    fn infer_matmul_result_type(
        &self,
        lhs_type: &MLIRType,
        rhs_type: &MLIRType,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<MLIRType, MLIRError> {
        match (lhs_type, rhs_type) {
            (MLIRType::AetherTensor { element_type: lhs_elem, shape: lhs_shape, device: lhs_device },
             MLIRType::AetherTensor { element_type: rhs_elem, shape: rhs_shape, device: rhs_device }) => {
                
                // Check element types are compatible
                if format!("{:?}", lhs_elem) != format!("{:?}", rhs_elem) {
                    return Err(MLIRError::TypeError("Matrix multiplication requires compatible element types".to_string()));
                }
                
                // Check devices are compatible
                if lhs_device != rhs_device {
                    return Err(MLIRError::TypeError("Matrix multiplication requires tensors on same device".to_string()));
                }
                
                // Compute result shape
                let result_shape = self.compute_matmul_shape(lhs_shape, rhs_shape, transpose_a, transpose_b)?;
                
                Ok(MLIRType::AetherTensor {
                    element_type: lhs_elem.clone(),
                    shape: result_shape,
                    device: lhs_device.clone(),
                })
            }
            _ => Err(MLIRError::TypeError("Matrix multiplication requires Aether tensor types".to_string()))
        }
    }
    
    /// Compute result shape for matrix multiplication
    fn compute_matmul_shape(
        &self,
        lhs_shape: &[i64],
        rhs_shape: &[i64],
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Vec<i64>, MLIRError> {
        if lhs_shape.len() < 2 || rhs_shape.len() < 2 {
            return Err(MLIRError::TypeError("Matrix multiplication requires at least 2D tensors".to_string()));
        }
        
        // Get matrix dimensions considering transposes
        let (m, k1) = if transpose_a {
            (lhs_shape[lhs_shape.len() - 1], lhs_shape[lhs_shape.len() - 2])
        } else {
            (lhs_shape[lhs_shape.len() - 2], lhs_shape[lhs_shape.len() - 1])
        };
        
        let (k2, n) = if transpose_b {
            (rhs_shape[rhs_shape.len() - 1], rhs_shape[rhs_shape.len() - 2])
        } else {
            (rhs_shape[rhs_shape.len() - 2], rhs_shape[rhs_shape.len() - 1])
        };
        
        // Check inner dimensions match
        if k1 != k2 {
            return Err(MLIRError::TypeError(format!(
                "Matrix multiplication inner dimensions must match: {} vs {}", k1, k2
            )));
        }
        
        // Compute batch dimensions (broadcast)
        let lhs_batch = &lhs_shape[..lhs_shape.len() - 2];
        let rhs_batch = &rhs_shape[..rhs_shape.len() - 2];
        
        let mut result_shape = Vec::new();
        
        // Broadcast batch dimensions
        let max_batch_dims = lhs_batch.len().max(rhs_batch.len());
        for i in 0..max_batch_dims {
            let lhs_dim = lhs_batch.get(lhs_batch.len().saturating_sub(max_batch_dims - i)).copied().unwrap_or(1);
            let rhs_dim = rhs_batch.get(rhs_batch.len().saturating_sub(max_batch_dims - i)).copied().unwrap_or(1);
            
            if lhs_dim == 1 {
                result_shape.push(rhs_dim);
            } else if rhs_dim == 1 {
                result_shape.push(lhs_dim);
            } else if lhs_dim == rhs_dim {
                result_shape.push(lhs_dim);
            } else {
                return Err(MLIRError::TypeError(format!(
                    "Incompatible batch dimensions for broadcasting: {} vs {}", lhs_dim, rhs_dim
                )));
            }
        }
        
        // Add matrix result dimensions
        result_shape.push(m);
        result_shape.push(n);
        
        Ok(result_shape)
    }
}

/// Aether-specific type utilities
pub struct AetherTypes;

impl AetherTypes {
    /// Create Aether tensor type with shape information
    pub fn tensor_type(
        _context: &MLIRContext,
        element_type: MLIRType,
        shape: &[i64],
        device: &str,
        is_differentiable: bool,
    ) -> AetherMLIRType {
        AetherMLIRType::Tensor {
            element_type: Box::new(element_type),
            shape: shape.to_vec(),
            is_differentiable,
            device: device.to_string(),
        }
    }

    /// Create function type with Aether-specific annotations
    pub fn function_type(
        _context: &MLIRContext,
        inputs: Vec<MLIRType>,
        outputs: Vec<MLIRType>,
        is_differentiable: bool,
        is_gpu_kernel: bool,
        is_actor_method: bool,
    ) -> AetherMLIRType {
        AetherMLIRType::Function {
            inputs,
            outputs,
            attributes: AetherFunctionAttributes {
                is_differentiable,
                is_gpu_kernel,
                is_actor_method,
                optimization_hints: Vec::new(),
            },
        }
    }

    /// Create probabilistic type
    pub fn probabilistic_type(
        _context: &MLIRContext,
        distribution: DistributionType,
        value_type: MLIRType,
    ) -> AetherMLIRType {
        AetherMLIRType::ProbabilisticVariable {
            distribution,
            value_type: Box::new(value_type),
        }
    }

    /// Create linear type
    pub fn linear_type(
        _context: &MLIRContext,
        inner_type: MLIRType,
        ownership: LinearOwnership,
        allocation_site: Option<String>,
        lifetime: Option<String>,
    ) -> AetherMLIRType {
        AetherMLIRType::LinearType {
            inner_type: Box::new(inner_type),
            ownership_info: LinearOwnershipInfo {
                ownership,
                allocation_site,
                lifetime,
            },
        }
    }
    
    /// Create actor type
    pub fn actor_type(
        _context: &MLIRContext,
        message_type: MLIRType,
        state_type: MLIRType,
    ) -> AetherMLIRType {
        AetherMLIRType::ActorType {
            message_type: Box::new(message_type),
            state_type: Box::new(state_type),
        }
    }
    
    /// Convert Aether type to MLIR type
    pub fn convert_to_mlir_type(aether_type: &AetherMLIRType) -> MLIRType {
        match aether_type {
            AetherMLIRType::Tensor { element_type, shape, device, .. } => {
                MLIRType::AetherTensor {
                    element_type: element_type.clone(),
                    shape: shape.clone(),
                    device: device.clone(),
                }
            }
            AetherMLIRType::ProbabilisticVariable { distribution, value_type } => {
                MLIRType::AetherProbabilistic {
                    distribution: format!("{:?}", distribution),
                    inner_type: value_type.clone(),
                }
            }
            AetherMLIRType::LinearType { inner_type, .. } => {
                MLIRType::AetherLinear {
                    inner_type: inner_type.clone(),
                }
            }
            AetherMLIRType::ActorType { message_type, state_type } => {
                // For now, represent actor as a struct-like type
                MLIRType::Function {
                    inputs: vec![(**message_type).clone()],
                    outputs: vec![(**state_type).clone()],
                }
            }
            AetherMLIRType::Function { inputs, outputs, .. } => {
                MLIRType::Function {
                    inputs: inputs.clone(),
                    outputs: outputs.clone(),
                }
            }
        }
    }
    
    /// Check if type is differentiable
    pub fn is_differentiable(aether_type: &AetherMLIRType) -> bool {
        match aether_type {
            AetherMLIRType::Tensor { is_differentiable, .. } => *is_differentiable,
            AetherMLIRType::Function { attributes, .. } => attributes.is_differentiable,
            _ => false,
        }
    }
    
    /// Check if type is GPU-compatible
    pub fn is_gpu_compatible(aether_type: &AetherMLIRType) -> bool {
        match aether_type {
            AetherMLIRType::Tensor { device, .. } => device.contains("gpu") || device.contains("cuda"),
            AetherMLIRType::Function { attributes, .. } => attributes.is_gpu_kernel,
            _ => false,
        }
    }
    
    /// Get tensor shape if type is tensor
    pub fn get_tensor_shape(aether_type: &AetherMLIRType) -> Option<&Vec<i64>> {
        match aether_type {
            AetherMLIRType::Tensor { shape, .. } => Some(shape),
            _ => None,
        }
    }
    
    /// Get element type if type is tensor
    pub fn get_element_type(aether_type: &AetherMLIRType) -> Option<&MLIRType> {
        match aether_type {
            AetherMLIRType::Tensor { element_type, .. } => Some(element_type),
            AetherMLIRType::ProbabilisticVariable { value_type, .. } => Some(value_type),
            AetherMLIRType::LinearType { inner_type, .. } => Some(inner_type),
            _ => None,
        }
    }
}