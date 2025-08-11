// Aether MLIR Dialect Registration System
// Implements dialect registration with MLIR context and operation builders

use crate::compiler::mlir::{
    MLIRError, AetherMLIRContext, MLIROperation, MLIRValue, MLIRType, MLIRAttribute,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Aether dialect registration and management
pub struct AetherDialectRegistry {
    context: Arc<AetherMLIRContext>,
    registered_operations: HashMap<String, OperationDefinition>,
    operation_builders: HashMap<String, Box<dyn OperationBuilder>>,
}

/// Operation definition for dialect registration
#[derive(Debug, Clone)]
pub struct OperationDefinition {
    pub name: String,
    pub operand_types: Vec<MLIRType>,
    pub result_types: Vec<MLIRType>,
    pub attributes: HashMap<String, AttributeDefinition>,
    pub regions: Vec<RegionDefinition>,
    pub traits: Vec<OperationTrait>,
}

/// Attribute definition for operations
#[derive(Debug, Clone)]
pub struct AttributeDefinition {
    pub name: String,
    pub attribute_type: AttributeType,
    pub required: bool,
    pub default_value: Option<MLIRAttribute>,
}

/// Attribute types supported by Aether dialect
#[derive(Debug, Clone)]
pub enum AttributeType {
    String,
    Integer,
    Float,
    Boolean,
    Array(Box<AttributeType>),
    Dictionary,
    Shape,
    Device,
    Distribution,
}

/// Region definition for operations with nested blocks
#[derive(Debug, Clone)]
pub struct RegionDefinition {
    pub name: String,
    pub min_blocks: usize,
    pub max_blocks: Option<usize>,
}

/// Operation traits for verification and optimization
#[derive(Debug, Clone, PartialEq)]
pub enum OperationTrait {
    /// Operation has no side effects
    Pure,
    /// Operation is commutative
    Commutative,
    /// Operation is associative
    Associative,
    /// Operation preserves tensor shapes
    ShapePreserving,
    /// Operation is differentiable
    Differentiable,
    /// Operation requires GPU execution
    GpuOnly,
    /// Operation supports broadcasting
    Broadcastable,
    /// Operation is memory-safe
    MemorySafe,
}

/// Operation builder trait for creating operations
pub trait OperationBuilder: Send + Sync {
    fn build_operation(
        &self,
        context: &AetherMLIRContext,
        operands: Vec<MLIRValue>,
        attributes: HashMap<String, MLIRAttribute>,
    ) -> Result<MLIROperation, MLIRError>;
    
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError>;
    fn validate_attributes(&self, attributes: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError>;
    fn infer_result_types(&self, operands: &[MLIRValue], attributes: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError>;
}

impl AetherDialectRegistry {
    /// Create new dialect registry
    pub fn new(context: Arc<AetherMLIRContext>) -> Result<Self, MLIRError> {
        let mut registry = AetherDialectRegistry {
            context,
            registered_operations: HashMap::new(),
            operation_builders: HashMap::new(),
        };
        
        // Register all Aether dialect operations
        registry.register_all_operations()?;
        
        Ok(registry)
    }
    
    /// Register all Aether dialect operations
    fn register_all_operations(&mut self) -> Result<(), MLIRError> {
        // Register tensor operations
        self.register_tensor_operations()?;
        
        // Register automatic differentiation operations
        self.register_autodiff_operations()?;
        
        // Register probabilistic programming operations
        self.register_probabilistic_operations()?;
        
        // Register memory management operations
        self.register_memory_operations()?;
        
        // Register concurrency operations
        self.register_concurrency_operations()?;
        
        Ok(())
    }
    
    /// Register tensor operations
    fn register_tensor_operations(&mut self) -> Result<(), MLIRError> {
        // Register tensor_create operation
        let tensor_create_def = OperationDefinition {
            name: "aether.tensor_create".to_string(),
            operand_types: vec![], // No operands, shape comes from attributes
            result_types: vec![], // Inferred from attributes
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("shape".to_string(), AttributeDefinition {
                    name: "shape".to_string(),
                    attribute_type: AttributeType::Shape,
                    required: true,
                    default_value: None,
                });
                attrs.insert("device".to_string(), AttributeDefinition {
                    name: "device".to_string(),
                    attribute_type: AttributeType::Device,
                    required: true,
                    default_value: Some(MLIRAttribute::String("cpu".to_string())),
                });
                attrs.insert("is_differentiable".to_string(), AttributeDefinition {
                    name: "is_differentiable".to_string(),
                    attribute_type: AttributeType::Boolean,
                    required: false,
                    default_value: Some(MLIRAttribute::Boolean(false)),
                });
                attrs
            },
            regions: vec![],
            traits: vec![OperationTrait::Pure, OperationTrait::MemorySafe],
        };
        
        self.registered_operations.insert("aether.tensor_create".to_string(), tensor_create_def);
        self.operation_builders.insert("aether.tensor_create".to_string(), Box::new(TensorCreateBuilder));
        
        // Register tensor_op operation
        let tensor_op_def = OperationDefinition {
            name: "aether.tensor_op".to_string(),
            operand_types: vec![], // Variable number of tensor operands
            result_types: vec![], // Inferred from operands and operation type
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("op_name".to_string(), AttributeDefinition {
                    name: "op_name".to_string(),
                    attribute_type: AttributeType::String,
                    required: true,
                    default_value: None,
                });
                attrs
            },
            regions: vec![],
            traits: vec![OperationTrait::Broadcastable],
        };
        
        self.registered_operations.insert("aether.tensor_op".to_string(), tensor_op_def);
        self.operation_builders.insert("aether.tensor_op".to_string(), Box::new(TensorOpBuilder));
        
        // Register matmul operation
        let matmul_def = OperationDefinition {
            name: "aether.matmul".to_string(),
            operand_types: vec![
                MLIRType::AetherTensor { 
                    element_type: Box::new(MLIRType::Float { width: 32 }), 
                    shape: vec![], 
                    device: "any".to_string() 
                },
                MLIRType::AetherTensor { 
                    element_type: Box::new(MLIRType::Float { width: 32 }), 
                    shape: vec![], 
                    device: "any".to_string() 
                },
            ],
            result_types: vec![],
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("transpose_a".to_string(), AttributeDefinition {
                    name: "transpose_a".to_string(),
                    attribute_type: AttributeType::Boolean,
                    required: false,
                    default_value: Some(MLIRAttribute::Boolean(false)),
                });
                attrs.insert("transpose_b".to_string(), AttributeDefinition {
                    name: "transpose_b".to_string(),
                    attribute_type: AttributeType::Boolean,
                    required: false,
                    default_value: Some(MLIRAttribute::Boolean(false)),
                });
                attrs
            },
            regions: vec![],
            traits: vec![OperationTrait::Pure, OperationTrait::Differentiable],
        };
        
        self.registered_operations.insert("aether.matmul".to_string(), matmul_def);
        self.operation_builders.insert("aether.matmul".to_string(), Box::new(MatmulBuilder));
        
        Ok(())
    }
    
    /// Register automatic differentiation operations
    fn register_autodiff_operations(&mut self) -> Result<(), MLIRError> {
        // Register autodiff_forward operation
        let autodiff_forward_def = OperationDefinition {
            name: "aether.autodiff_forward".to_string(),
            operand_types: vec![
                MLIRType::Function { inputs: vec![], outputs: vec![] }, // Function to differentiate
                MLIRType::AetherTensor { element_type: Box::new(MLIRType::Float { width: 32 }), shape: vec![], device: "any".to_string() }, // Input
                MLIRType::AetherTensor { element_type: Box::new(MLIRType::Float { width: 32 }), shape: vec![], device: "any".to_string() }, // Tangent
            ],
            result_types: vec![], // Inferred from function signature
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("mode".to_string(), AttributeDefinition {
                    name: "mode".to_string(),
                    attribute_type: AttributeType::String,
                    required: false,
                    default_value: Some(MLIRAttribute::String("forward".to_string())),
                });
                attrs
            },
            regions: vec![],
            traits: vec![OperationTrait::Pure, OperationTrait::Differentiable],
        };
        
        self.registered_operations.insert("aether.autodiff_forward".to_string(), autodiff_forward_def);
        self.operation_builders.insert("aether.autodiff_forward".to_string(), Box::new(AutodiffForwardBuilder));
        
        // Register autodiff_reverse operation
        let autodiff_reverse_def = OperationDefinition {
            name: "aether.autodiff_reverse".to_string(),
            operand_types: vec![
                MLIRType::Function { inputs: vec![], outputs: vec![] }, // Function to differentiate
                MLIRType::AetherTensor { element_type: Box::new(MLIRType::Float { width: 32 }), shape: vec![], device: "any".to_string() }, // Input
            ],
            result_types: vec![], // Inferred from function signature
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("mode".to_string(), AttributeDefinition {
                    name: "mode".to_string(),
                    attribute_type: AttributeType::String,
                    required: false,
                    default_value: Some(MLIRAttribute::String("reverse".to_string())),
                });
                attrs
            },
            regions: vec![],
            traits: vec![OperationTrait::Pure, OperationTrait::Differentiable],
        };
        
        self.registered_operations.insert("aether.autodiff_reverse".to_string(), autodiff_reverse_def);
        self.operation_builders.insert("aether.autodiff_reverse".to_string(), Box::new(AutodiffReverseBuilder));
        
        // Register gradient operation
        let gradient_def = OperationDefinition {
            name: "aether.gradient".to_string(),
            operand_types: vec![
                MLIRType::Function { inputs: vec![], outputs: vec![] }, // Function to differentiate
                MLIRType::AetherTensor { element_type: Box::new(MLIRType::Float { width: 32 }), shape: vec![], device: "any".to_string() }, // Input
            ],
            result_types: vec![], // Gradient tensor
            attributes: HashMap::new(),
            regions: vec![],
            traits: vec![OperationTrait::Pure, OperationTrait::Differentiable],
        };
        
        self.registered_operations.insert("aether.gradient".to_string(), gradient_def);
        self.operation_builders.insert("aether.gradient".to_string(), Box::new(GradientBuilder));
        
        Ok(())
    }
    
    /// Register probabilistic programming operations
    fn register_probabilistic_operations(&mut self) -> Result<(), MLIRError> {
        // Register prob_var operation
        let prob_var_def = OperationDefinition {
            name: "aether.prob_var".to_string(),
            operand_types: vec![], // Parameters come from attributes
            result_types: vec![], // Inferred from distribution
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("name".to_string(), AttributeDefinition {
                    name: "name".to_string(),
                    attribute_type: AttributeType::String,
                    required: true,
                    default_value: None,
                });
                attrs.insert("distribution".to_string(), AttributeDefinition {
                    name: "distribution".to_string(),
                    attribute_type: AttributeType::Distribution,
                    required: true,
                    default_value: None,
                });
                attrs
            },
            regions: vec![],
            traits: vec![OperationTrait::Pure],
        };
        
        self.registered_operations.insert("aether.prob_var".to_string(), prob_var_def);
        self.operation_builders.insert("aether.prob_var".to_string(), Box::new(ProbVarBuilder));
        
        // Register sample operation
        let sample_def = OperationDefinition {
            name: "aether.sample".to_string(),
            operand_types: vec![
                MLIRType::AetherProbabilistic { distribution: "any".to_string(), inner_type: Box::new(MLIRType::Float { width: 32 }) },
            ],
            result_types: vec![], // Inferred from probabilistic variable
            attributes: HashMap::new(),
            regions: vec![],
            traits: vec![], // Not pure due to randomness
        };
        
        self.registered_operations.insert("aether.sample".to_string(), sample_def);
        self.operation_builders.insert("aether.sample".to_string(), Box::new(SampleBuilder));
        
        // Register observe operation
        let observe_def = OperationDefinition {
            name: "aether.observe".to_string(),
            operand_types: vec![
                MLIRType::AetherProbabilistic { distribution: "any".to_string(), inner_type: Box::new(MLIRType::Float { width: 32 }) },
                MLIRType::AetherTensor { element_type: Box::new(MLIRType::Float { width: 32 }), shape: vec![], device: "any".to_string() },
            ],
            result_types: vec![], // No result, side effect only
            attributes: HashMap::new(),
            regions: vec![],
            traits: vec![], // Not pure due to side effects
        };
        
        self.registered_operations.insert("aether.observe".to_string(), observe_def);
        self.operation_builders.insert("aether.observe".to_string(), Box::new(ObserveBuilder));
        
        Ok(())
    }
    
    /// Register memory management operations
    fn register_memory_operations(&mut self) -> Result<(), MLIRError> {
        // Register linear_alloc operation
        let linear_alloc_def = OperationDefinition {
            name: "aether.linear_alloc".to_string(),
            operand_types: vec![], // Size may be provided as operand
            result_types: vec![], // Inferred linear type
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("allocation_site".to_string(), AttributeDefinition {
                    name: "allocation_site".to_string(),
                    attribute_type: AttributeType::String,
                    required: true,
                    default_value: None,
                });
                attrs.insert("inner_type".to_string(), AttributeDefinition {
                    name: "inner_type".to_string(),
                    attribute_type: AttributeType::String, // Type representation
                    required: true,
                    default_value: None,
                });
                attrs
            },
            regions: vec![],
            traits: vec![OperationTrait::MemorySafe],
        };
        
        self.registered_operations.insert("aether.linear_alloc".to_string(), linear_alloc_def);
        self.operation_builders.insert("aether.linear_alloc".to_string(), Box::new(LinearAllocBuilder));
        
        // Register linear_move operation
        let linear_move_def = OperationDefinition {
            name: "aether.linear_move".to_string(),
            operand_types: vec![
                MLIRType::AetherLinear { inner_type: Box::new(MLIRType::Float { width: 32 }) },
            ],
            result_types: vec![], // Same as operand type
            attributes: HashMap::new(),
            regions: vec![],
            traits: vec![OperationTrait::MemorySafe],
        };
        
        self.registered_operations.insert("aether.linear_move".to_string(), linear_move_def);
        self.operation_builders.insert("aether.linear_move".to_string(), Box::new(LinearMoveBuilder));
        
        // Register linear_drop operation
        let linear_drop_def = OperationDefinition {
            name: "aether.linear_drop".to_string(),
            operand_types: vec![
                MLIRType::AetherLinear { inner_type: Box::new(MLIRType::Float { width: 32 }) },
            ],
            result_types: vec![], // No result, side effect only
            attributes: HashMap::new(),
            regions: vec![],
            traits: vec![OperationTrait::MemorySafe],
        };
        
        self.registered_operations.insert("aether.linear_drop".to_string(), linear_drop_def);
        self.operation_builders.insert("aether.linear_drop".to_string(), Box::new(LinearDropBuilder));
        
        Ok(())
    }
    
    /// Register concurrency operations
    fn register_concurrency_operations(&mut self) -> Result<(), MLIRError> {
        // Register spawn_actor operation
        let spawn_actor_def = OperationDefinition {
            name: "aether.spawn_actor".to_string(),
            operand_types: vec![], // Initial state may be provided
            result_types: vec![], // Actor reference type
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("actor_type".to_string(), AttributeDefinition {
                    name: "actor_type".to_string(),
                    attribute_type: AttributeType::String,
                    required: true,
                    default_value: None,
                });
                attrs
            },
            regions: vec![],
            traits: vec![],
        };
        
        self.registered_operations.insert("aether.spawn_actor".to_string(), spawn_actor_def);
        self.operation_builders.insert("aether.spawn_actor".to_string(), Box::new(SpawnActorBuilder));
        
        // Register send_message operation
        let send_message_def = OperationDefinition {
            name: "aether.send_message".to_string(),
            operand_types: vec![
                MLIRType::Function { inputs: vec![], outputs: vec![] }, // Actor reference
                MLIRType::AetherTensor { element_type: Box::new(MLIRType::Float { width: 32 }), shape: vec![], device: "any".to_string() }, // Message
            ],
            result_types: vec![], // No result, side effect only
            attributes: HashMap::new(),
            regions: vec![],
            traits: vec![],
        };
        
        self.registered_operations.insert("aether.send_message".to_string(), send_message_def);
        self.operation_builders.insert("aether.send_message".to_string(), Box::new(SendMessageBuilder));
        
        // Register parallel_for operation
        let parallel_for_def = OperationDefinition {
            name: "aether.parallel_for".to_string(),
            operand_types: vec![
                MLIRType::Index, // Lower bound
                MLIRType::Index, // Upper bound
                MLIRType::Function { inputs: vec![], outputs: vec![] }, // Body function
            ],
            result_types: vec![], // No result, side effect only
            attributes: {
                let mut attrs = HashMap::new();
                attrs.insert("step".to_string(), AttributeDefinition {
                    name: "step".to_string(),
                    attribute_type: AttributeType::Integer,
                    required: false,
                    default_value: Some(MLIRAttribute::Integer(1)),
                });
                attrs
            },
            regions: vec![],
            traits: vec![],
        };
        
        self.registered_operations.insert("aether.parallel_for".to_string(), parallel_for_def);
        self.operation_builders.insert("aether.parallel_for".to_string(), Box::new(ParallelForBuilder));
        
        Ok(())
    }
    
    /// Register dialect with MLIR context
    pub fn register_dialect(&self) -> Result<(), MLIRError> {
        // Register the Aether dialect with the MLIR context
        // In a real implementation, this would call MLIR C API functions
        // For now, we simulate successful registration
        
        // Check if dialect is already registered
        if self.context.is_dialect_registered("aether") {
            return Ok(());
        }
        
        // Register dialect (simulated)
        // mlir_register_dialect(context, "aether", operations, types, attributes)
        
        Ok(())
    }
    
    /// Get operation definition by name
    pub fn get_operation_definition(&self, name: &str) -> Option<&OperationDefinition> {
        self.registered_operations.get(name)
    }
    
    /// Get operation builder by name
    pub fn get_operation_builder(&self, name: &str) -> Option<&dyn OperationBuilder> {
        self.operation_builders.get(name).map(|b| b.as_ref())
    }
    
    /// Create operation using registered builder
    pub fn create_operation(
        &self,
        name: &str,
        operands: Vec<MLIRValue>,
        attributes: HashMap<String, MLIRAttribute>,
    ) -> Result<MLIROperation, MLIRError> {
        let builder = self.get_operation_builder(name)
            .ok_or_else(|| MLIRError::OperationError(format!("Unknown operation: {}", name)))?;
        
        // Validate operands and attributes
        builder.validate_operands(&operands)?;
        builder.validate_attributes(&attributes)?;
        
        // Build the operation
        builder.build_operation(&self.context, operands, attributes)
    }
    
    /// List all registered operations
    pub fn list_operations(&self) -> Vec<String> {
        self.registered_operations.keys().cloned().collect()
    }
    
    /// Verify operation against its definition
    pub fn verify_operation(&self, operation: &MLIROperation) -> Result<(), MLIRError> {
        let definition = self.get_operation_definition(&operation.name)
            .ok_or_else(|| MLIRError::VerificationError(format!("Unknown operation: {}", operation.name)))?;
        
        // Verify operand count and types
        if !definition.operand_types.is_empty() && operation.operands.len() != definition.operand_types.len() {
            return Err(MLIRError::VerificationError(format!(
                "Operation {} expects {} operands, got {}",
                operation.name, definition.operand_types.len(), operation.operands.len()
            )));
        }
        
        // Verify required attributes are present
        for (attr_name, attr_def) in &definition.attributes {
            if attr_def.required && !operation.attributes.contains_key(attr_name) {
                return Err(MLIRError::VerificationError(format!(
                    "Operation {} missing required attribute: {}",
                    operation.name, attr_name
                )));
            }
        }
        
        // Verify operation traits
        for trait_type in &definition.traits {
            self.verify_operation_trait(operation, trait_type)?;
        }
        
        Ok(())
    }
    
    /// Verify operation trait
    fn verify_operation_trait(&self, operation: &MLIROperation, trait_type: &OperationTrait) -> Result<(), MLIRError> {
        match trait_type {
            OperationTrait::Pure => {
                // Pure operations should not have side effects
                // This is a semantic check that would be more complex in practice
                Ok(())
            }
            OperationTrait::Differentiable => {
                // Differentiable operations should work with tensor types
                for operand in &operation.operands {
                    if !operand.value_type.is_tensor() {
                        return Err(MLIRError::VerificationError(format!(
                            "Differentiable operation {} requires tensor operands",
                            operation.name
                        )));
                    }
                }
                Ok(())
            }
            OperationTrait::MemorySafe => {
                // Memory-safe operations should properly handle linear types
                Ok(())
            }
            _ => {
                // Other traits are verified during lowering or optimization
                Ok(())
            }
        }
    }
}

// Operation builder implementations

/// Tensor creation operation builder
struct TensorCreateBuilder;

impl OperationBuilder for TensorCreateBuilder {
    fn build_operation(
        &self,
        _context: &AetherMLIRContext,
        operands: Vec<MLIRValue>,
        attributes: HashMap<String, MLIRAttribute>,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.tensor_create".to_string());
        
        // Add operands
        for operand in operands {
            op.add_operand(operand);
        }
        
        // Add attributes
        for (key, value) in attributes {
            op.add_attribute(key, value);
        }
        
        // Infer and add result type
        let result_types = self.infer_result_types(&op.operands, &op.attributes)?;
        for (i, result_type) in result_types.into_iter().enumerate() {
            let result = MLIRValue::new(format!("tensor_create_result_{}", i), result_type);
            op.add_result(result);
        }
        
        Ok(op)
    }
    
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        // tensor_create typically has no operands (shape comes from attributes)
        if !operands.is_empty() {
            return Err(MLIRError::OperationError("tensor_create should have no operands".to_string()));
        }
        Ok(())
    }
    
    fn validate_attributes(&self, attributes: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> {
        // Validate required attributes
        if !attributes.contains_key("shape") {
            return Err(MLIRError::OperationError("tensor_create requires 'shape' attribute".to_string()));
        }
        
        // Validate shape attribute
        if let Some(MLIRAttribute::Array(shape_attrs)) = attributes.get("shape") {
            for shape_attr in shape_attrs {
                if let MLIRAttribute::Integer(dim) = shape_attr {
                    if *dim <= 0 {
                        return Err(MLIRError::OperationError(format!("Invalid tensor dimension: {}", dim)));
                    }
                } else {
                    return Err(MLIRError::OperationError("Shape must be array of integers".to_string()));
                }
            }
        } else {
            return Err(MLIRError::OperationError("Shape attribute must be array".to_string()));
        }
        
        Ok(())
    }
    
    fn infer_result_types(&self, _operands: &[MLIRValue], attributes: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        // Extract shape from attributes
        let shape = if let Some(MLIRAttribute::Array(shape_attrs)) = attributes.get("shape") {
            shape_attrs.iter().map(|attr| {
                if let MLIRAttribute::Integer(dim) = attr {
                    Ok(*dim)
                } else {
                    Err(MLIRError::TypeError("Shape must be array of integers".to_string()))
                }
            }).collect::<Result<Vec<i64>, MLIRError>>()?
        } else {
            return Err(MLIRError::TypeError("Missing shape attribute".to_string()));
        };
        
        // Extract device
        let device = if let Some(MLIRAttribute::String(device_str)) = attributes.get("device") {
            device_str.clone()
        } else {
            "cpu".to_string()
        };
        
        // Create tensor type
        let tensor_type = MLIRType::AetherTensor {
            element_type: Box::new(MLIRType::Float { width: 32 }), // Default to f32
            shape,
            device,
        };
        
        Ok(vec![tensor_type])
    }
}

/// Generic tensor operation builder
struct TensorOpBuilder;

impl OperationBuilder for TensorOpBuilder {
    fn build_operation(
        &self,
        _context: &AetherMLIRContext,
        operands: Vec<MLIRValue>,
        attributes: HashMap<String, MLIRAttribute>,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.tensor_op".to_string());
        
        for operand in operands {
            op.add_operand(operand);
        }
        
        for (key, value) in attributes {
            op.add_attribute(key, value);
        }
        
        // Infer result type based on operands and operation
        let result_types = self.infer_result_types(&op.operands, &op.attributes)?;
        for (i, result_type) in result_types.into_iter().enumerate() {
            let result = MLIRValue::new(format!("tensor_op_result_{}", i), result_type);
            op.add_result(result);
        }
        
        Ok(op)
    }
    
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        if operands.is_empty() {
            return Err(MLIRError::OperationError("tensor_op requires at least one operand".to_string()));
        }
        
        // All operands should be tensors
        for operand in operands {
            if !operand.value_type.is_tensor() {
                return Err(MLIRError::OperationError("tensor_op requires tensor operands".to_string()));
            }
        }
        
        Ok(())
    }
    
    fn validate_attributes(&self, attributes: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> {
        if !attributes.contains_key("op_name") {
            return Err(MLIRError::OperationError("tensor_op requires 'op_name' attribute".to_string()));
        }
        Ok(())
    }
    
    fn infer_result_types(&self, operands: &[MLIRValue], _attributes: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        if operands.is_empty() {
            return Err(MLIRError::TypeError("Cannot infer result type without operands".to_string()));
        }
        
        // For now, return the type of the first operand
        // In practice, this would depend on the specific operation
        Ok(vec![operands[0].value_type.clone()])
    }
}

/// Matrix multiplication operation builder
struct MatmulBuilder;

impl OperationBuilder for MatmulBuilder {
    fn build_operation(
        &self,
        _context: &AetherMLIRContext,
        operands: Vec<MLIRValue>,
        attributes: HashMap<String, MLIRAttribute>,
    ) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.matmul".to_string());
        
        for operand in operands {
            op.add_operand(operand);
        }
        
        for (key, value) in attributes {
            op.add_attribute(key, value);
        }
        
        let result_types = self.infer_result_types(&op.operands, &op.attributes)?;
        for (i, result_type) in result_types.into_iter().enumerate() {
            let result = MLIRValue::new(format!("matmul_result_{}", i), result_type);
            op.add_result(result);
        }
        
        Ok(op)
    }
    
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        if operands.len() != 2 {
            return Err(MLIRError::OperationError("matmul requires exactly 2 operands".to_string()));
        }
        
        // Both operands should be tensors
        for operand in operands {
            if !operand.value_type.is_tensor() {
                return Err(MLIRError::OperationError("matmul requires tensor operands".to_string()));
            }
        }
        
        Ok(())
    }
    
    fn validate_attributes(&self, _attributes: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> {
        // transpose_a and transpose_b are optional
        Ok(())
    }
    
    fn infer_result_types(&self, operands: &[MLIRValue], attributes: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        if operands.len() != 2 {
            return Err(MLIRError::TypeError("matmul requires exactly 2 operands".to_string()));
        }
        
        // Extract transpose flags
        let transpose_a = attributes.get("transpose_a")
            .and_then(|attr| if let MLIRAttribute::Boolean(b) = attr { Some(*b) } else { None })
            .unwrap_or(false);
        let transpose_b = attributes.get("transpose_b")
            .and_then(|attr| if let MLIRAttribute::Boolean(b) = attr { Some(*b) } else { None })
            .unwrap_or(false);
        
        // Compute result shape based on operand shapes and transpose flags
        match (&operands[0].value_type, &operands[1].value_type) {
            (MLIRType::AetherTensor { element_type: lhs_elem, shape: lhs_shape, device: lhs_device },
             MLIRType::AetherTensor { element_type: rhs_elem, shape: rhs_shape, device: rhs_device }) => {
                
                // Check compatibility
                if format!("{:?}", lhs_elem) != format!("{:?}", rhs_elem) {
                    return Err(MLIRError::TypeError("matmul operands must have same element type".to_string()));
                }
                
                if lhs_device != rhs_device {
                    return Err(MLIRError::TypeError("matmul operands must be on same device".to_string()));
                }
                
                // Compute result shape
                let result_shape = self.compute_matmul_shape(lhs_shape, rhs_shape, transpose_a, transpose_b)?;
                
                let result_type = MLIRType::AetherTensor {
                    element_type: lhs_elem.clone(),
                    shape: result_shape,
                    device: lhs_device.clone(),
                };
                
                Ok(vec![result_type])
            }
            _ => Err(MLIRError::TypeError("matmul requires Aether tensor operands".to_string()))
        }
    }
}

impl MatmulBuilder {
    fn compute_matmul_shape(&self, lhs_shape: &[i64], rhs_shape: &[i64], transpose_a: bool, transpose_b: bool) -> Result<Vec<i64>, MLIRError> {
        if lhs_shape.len() < 2 || rhs_shape.len() < 2 {
            return Err(MLIRError::TypeError("matmul requires at least 2D tensors".to_string()));
        }
        
        // Get matrix dimensions
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
        
        if k1 != k2 {
            return Err(MLIRError::TypeError(format!("matmul inner dimensions must match: {} vs {}", k1, k2)));
        }
        
        // Compute batch dimensions
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
                return Err(MLIRError::TypeError(format!("Incompatible batch dimensions: {} vs {}", lhs_dim, rhs_dim)));
            }
        }
        
        result_shape.push(m);
        result_shape.push(n);
        
        Ok(result_shape)
    }
}

// Additional operation builders (simplified implementations)

struct AutodiffForwardBuilder;
impl OperationBuilder for AutodiffForwardBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.autodiff_forward".to_string());
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        let result = MLIRValue::new("autodiff_forward_result".to_string(), MLIRType::Float { width: 32 });
        op.add_result(result);
        Ok(op)
    }
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        if operands.len() != 3 { return Err(MLIRError::OperationError("autodiff_forward requires 3 operands".to_string())); }
        Ok(())
    }
    fn validate_attributes(&self, _: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> { Ok(()) }
    fn infer_result_types(&self, _: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        Ok(vec![MLIRType::Float { width: 32 }])
    }
}

struct AutodiffReverseBuilder;
impl OperationBuilder for AutodiffReverseBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.autodiff_reverse".to_string());
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        let result = MLIRValue::new("autodiff_reverse_result".to_string(), MLIRType::Float { width: 32 });
        op.add_result(result);
        Ok(op)
    }
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        if operands.len() != 2 { return Err(MLIRError::OperationError("autodiff_reverse requires 2 operands".to_string())); }
        Ok(())
    }
    fn validate_attributes(&self, _: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> { Ok(()) }
    fn infer_result_types(&self, _: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        Ok(vec![MLIRType::Float { width: 32 }])
    }
}

struct GradientBuilder;
impl OperationBuilder for GradientBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.gradient".to_string());
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        let result = MLIRValue::new("gradient_result".to_string(), MLIRType::AetherTensor { 
            element_type: Box::new(MLIRType::Float { width: 32 }), 
            shape: vec![1], 
            device: "cpu".to_string() 
        });
        op.add_result(result);
        Ok(op)
    }
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        if operands.len() != 2 { return Err(MLIRError::OperationError("gradient requires 2 operands".to_string())); }
        Ok(())
    }
    fn validate_attributes(&self, _: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> { Ok(()) }
    fn infer_result_types(&self, _: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        Ok(vec![MLIRType::AetherTensor { element_type: Box::new(MLIRType::Float { width: 32 }), shape: vec![1], device: "cpu".to_string() }])
    }
}

struct ProbVarBuilder;
impl OperationBuilder for ProbVarBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.prob_var".to_string());
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        let result = MLIRValue::new("prob_var_result".to_string(), MLIRType::AetherProbabilistic { 
            distribution: "normal".to_string(), 
            inner_type: Box::new(MLIRType::Float { width: 32 }) 
        });
        op.add_result(result);
        Ok(op)
    }
    fn validate_operands(&self, _: &[MLIRValue]) -> Result<(), MLIRError> { Ok(()) }
    fn validate_attributes(&self, attributes: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> {
        if !attributes.contains_key("name") || !attributes.contains_key("distribution") {
            return Err(MLIRError::OperationError("prob_var requires 'name' and 'distribution' attributes".to_string()));
        }
        Ok(())
    }
    fn infer_result_types(&self, _: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        Ok(vec![MLIRType::AetherProbabilistic { distribution: "normal".to_string(), inner_type: Box::new(MLIRType::Float { width: 32 }) }])
    }
}

struct SampleBuilder;
impl OperationBuilder for SampleBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.sample".to_string());
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        let result = MLIRValue::new("sample_result".to_string(), MLIRType::Float { width: 32 });
        op.add_result(result);
        Ok(op)
    }
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        if operands.len() != 1 { return Err(MLIRError::OperationError("sample requires 1 operand".to_string())); }
        Ok(())
    }
    fn validate_attributes(&self, _: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> { Ok(()) }
    fn infer_result_types(&self, _: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        Ok(vec![MLIRType::Float { width: 32 }])
    }
}

struct ObserveBuilder;
impl OperationBuilder for ObserveBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.observe".to_string());
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        Ok(op)
    }
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        if operands.len() != 2 { return Err(MLIRError::OperationError("observe requires 2 operands".to_string())); }
        Ok(())
    }
    fn validate_attributes(&self, _: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> { Ok(()) }
    fn infer_result_types(&self, _: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        Ok(vec![])
    }
}

struct LinearAllocBuilder;
impl OperationBuilder for LinearAllocBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.linear_alloc".to_string());
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        let result = MLIRValue::new("linear_alloc_result".to_string(), MLIRType::AetherLinear { 
            inner_type: Box::new(MLIRType::Float { width: 32 }) 
        });
        op.add_result(result);
        Ok(op)
    }
    fn validate_operands(&self, _: &[MLIRValue]) -> Result<(), MLIRError> { Ok(()) }
    fn validate_attributes(&self, attributes: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> {
        if !attributes.contains_key("allocation_site") || !attributes.contains_key("inner_type") {
            return Err(MLIRError::OperationError("linear_alloc requires 'allocation_site' and 'inner_type' attributes".to_string()));
        }
        Ok(())
    }
    fn infer_result_types(&self, _: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        Ok(vec![MLIRType::AetherLinear { inner_type: Box::new(MLIRType::Float { width: 32 }) }])
    }
}

struct LinearMoveBuilder;
impl OperationBuilder for LinearMoveBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.linear_move".to_string());
        
        // Get the first operand type before consuming the vector
        let result_type = if let Some(operand) = operands.first() {
            operand.value_type.clone()
        } else {
            return Err(MLIRError::OperationError("linear_move requires operand".to_string()));
        };
        
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        
        let result = MLIRValue::new("linear_move_result".to_string(), result_type);
        op.add_result(result);
        Ok(op)
    }
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        if operands.len() != 1 { return Err(MLIRError::OperationError("linear_move requires 1 operand".to_string())); }
        Ok(())
    }
    fn validate_attributes(&self, _: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> { Ok(()) }
    fn infer_result_types(&self, operands: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        if let Some(operand) = operands.first() {
            Ok(vec![operand.value_type.clone()])
        } else {
            Err(MLIRError::TypeError("linear_move requires operand to infer type".to_string()))
        }
    }
}

struct LinearDropBuilder;
impl OperationBuilder for LinearDropBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.linear_drop".to_string());
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        Ok(op)
    }
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        if operands.len() != 1 { return Err(MLIRError::OperationError("linear_drop requires 1 operand".to_string())); }
        Ok(())
    }
    fn validate_attributes(&self, _: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> { Ok(()) }
    fn infer_result_types(&self, _: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        Ok(vec![])
    }
}

struct SpawnActorBuilder;
impl OperationBuilder for SpawnActorBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.spawn_actor".to_string());
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        let result = MLIRValue::new("actor_ref".to_string(), MLIRType::Function { inputs: vec![], outputs: vec![] });
        op.add_result(result);
        Ok(op)
    }
    fn validate_operands(&self, _: &[MLIRValue]) -> Result<(), MLIRError> { Ok(()) }
    fn validate_attributes(&self, attributes: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> {
        if !attributes.contains_key("actor_type") {
            return Err(MLIRError::OperationError("spawn_actor requires 'actor_type' attribute".to_string()));
        }
        Ok(())
    }
    fn infer_result_types(&self, _: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        Ok(vec![MLIRType::Function { inputs: vec![], outputs: vec![] }])
    }
}

struct SendMessageBuilder;
impl OperationBuilder for SendMessageBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.send_message".to_string());
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        Ok(op)
    }
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        if operands.len() != 2 { return Err(MLIRError::OperationError("send_message requires 2 operands".to_string())); }
        Ok(())
    }
    fn validate_attributes(&self, _: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> { Ok(()) }
    fn infer_result_types(&self, _: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        Ok(vec![])
    }
}

struct ParallelForBuilder;
impl OperationBuilder for ParallelForBuilder {
    fn build_operation(&self, _context: &AetherMLIRContext, operands: Vec<MLIRValue>, attributes: HashMap<String, MLIRAttribute>) -> Result<MLIROperation, MLIRError> {
        let mut op = MLIROperation::new("aether.parallel_for".to_string());
        for operand in operands { op.add_operand(operand); }
        for (k, v) in attributes { op.add_attribute(k, v); }
        Ok(op)
    }
    fn validate_operands(&self, operands: &[MLIRValue]) -> Result<(), MLIRError> {
        if operands.len() < 3 { return Err(MLIRError::OperationError("parallel_for requires at least 3 operands".to_string())); }
        Ok(())
    }
    fn validate_attributes(&self, _: &HashMap<String, MLIRAttribute>) -> Result<(), MLIRError> { Ok(()) }
    fn infer_result_types(&self, _: &[MLIRValue], _: &HashMap<String, MLIRAttribute>) -> Result<Vec<MLIRType>, MLIRError> {
        Ok(vec![])
    }
}