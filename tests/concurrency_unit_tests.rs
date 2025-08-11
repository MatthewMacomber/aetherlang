// Unit tests for Aether structured concurrency system core functionality
// Tests the basic concurrency constructs without complex MLIR integration

use aether_language::compiler::concurrency::*;
use std::collections::HashMap;

#[test]
fn test_actor_state_creation() {
    let state = ActorState::new();
    assert_eq!(state.execution_counter, 0);
    assert_eq!(state.last_message_time, 0);
    assert!(state.variables.is_empty());
}

#[test]
fn test_execution_state_creation() {
    let state = ExecutionState::new();
    assert_eq!(state.timestamp, 0);
    assert_eq!(state.random_seed, 42);
    assert!(state.actor_states.is_empty());
    assert!(state.message_log.is_empty());
    assert!(state.snapshots.is_empty());
}

#[test]
fn test_typed_message_creation() {
    let message = TypedMessage {
        sender: 1,
        receiver: 2,
        message_type: "test_message".to_string(),
        payload: AetherValue::Integer(42),
        timestamp: 100,
    };
    
    assert_eq!(message.sender, 1);
    assert_eq!(message.receiver, 2);
    assert_eq!(message.message_type, "test_message");
    assert_eq!(message.timestamp, 100);
    
    match message.payload {
        AetherValue::Integer(42) => {},
        _ => panic!("Expected integer payload"),
    }
}

#[test]
fn test_aether_value_types() {
    // Test all AetherValue variants
    let int_val = AetherValue::Integer(42);
    let float_val = AetherValue::Float(3.14);
    let bool_val = AetherValue::Boolean(true);
    let string_val = AetherValue::String("hello".to_string());
    
    let array_val = AetherValue::Array(vec![
        AetherValue::Integer(1),
        AetherValue::Integer(2),
        AetherValue::Integer(3),
    ]);
    
    let mut struct_map = HashMap::new();
    struct_map.insert("x".to_string(), AetherValue::Float(1.0));
    struct_map.insert("y".to_string(), AetherValue::Float(2.0));
    let struct_val = AetherValue::Struct(struct_map);
    
    let tensor_val = AetherValue::Tensor {
        data: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![2, 2],
    };
    
    let function_val = AetherValue::Function {
        name: "test_function".to_string(),
        parameters: vec!["x".to_string(), "y".to_string()],
    };
    
    // Verify all values can be pattern matched
    match int_val {
        AetherValue::Integer(42) => {},
        _ => panic!("Expected integer value"),
    }
    
    match float_val {
        AetherValue::Float(f) if (f - 3.14).abs() < 0.001 => {},
        _ => panic!("Expected float value"),
    }
    
    match bool_val {
        AetherValue::Boolean(true) => {},
        _ => panic!("Expected boolean value"),
    }
    
    match string_val {
        AetherValue::String(ref s) if s == "hello" => {},
        _ => panic!("Expected string value"),
    }
    
    match array_val {
        AetherValue::Array(ref arr) => {
            assert_eq!(arr.len(), 3);
        }
        _ => panic!("Expected array value"),
    }
    
    match struct_val {
        AetherValue::Struct(ref map) => {
            assert_eq!(map.len(), 2);
            assert!(map.contains_key("x"));
            assert!(map.contains_key("y"));
        }
        _ => panic!("Expected struct value"),
    }
    
    match tensor_val {
        AetherValue::Tensor { ref data, ref shape } => {
            assert_eq!(data.len(), 4);
            assert_eq!(shape, &vec![2, 2]);
        }
        _ => panic!("Expected tensor value"),
    }
    
    match function_val {
        AetherValue::Function { ref name, ref parameters } => {
            assert_eq!(name, "test_function");
            assert_eq!(parameters.len(), 2);
        }
        _ => panic!("Expected function value"),
    }
}

#[test]
fn test_actor_types() {
    let compute_actor = ActorType::Compute { worker_count: 4 };
    let io_actor = ActorType::IO { buffer_size: 1024 };
    let gpu_actor = ActorType::GPU { device_id: 0 };
    let custom_actor = ActorType::Custom { type_name: "DatabaseActor".to_string() };
    
    match compute_actor {
        ActorType::Compute { worker_count: 4 } => {},
        _ => panic!("Expected compute actor"),
    }
    
    match io_actor {
        ActorType::IO { buffer_size: 1024 } => {},
        _ => panic!("Expected IO actor"),
    }
    
    match gpu_actor {
        ActorType::GPU { device_id: 0 } => {},
        _ => panic!("Expected GPU actor"),
    }
    
    match custom_actor {
        ActorType::Custom { ref type_name } => {
            assert_eq!(type_name, "DatabaseActor");
        }
        _ => panic!("Expected custom actor"),
    }
}

#[test]
fn test_parallel_task_types() {
    let parallel_for = ParallelTaskType::ParallelFor {
        range: (0, 100),
        step: 1,
        body: "arith.addi %i, %c1 : index".to_string(),
    };
    
    let map_task = ParallelTaskType::Map {
        input_collection: "input_array".to_string(),
        transform_function: "square_function".to_string(),
    };
    
    let reduce_task = ParallelTaskType::Reduce {
        input_collection: "input_array".to_string(),
        reduce_function: "sum_function".to_string(),
        initial_value: AetherValue::Integer(0),
    };
    
    let custom_task = ParallelTaskType::Custom {
        operation_name: "custom_parallel_op".to_string(),
        parameters: {
            let mut params = HashMap::new();
            params.insert("param1".to_string(), AetherValue::Integer(42));
            params
        },
    };
    
    match parallel_for {
        ParallelTaskType::ParallelFor { range: (0, 100), step: 1, .. } => {},
        _ => panic!("Expected parallel for task"),
    }
    
    match map_task {
        ParallelTaskType::Map { ref input_collection, ref transform_function } => {
            assert_eq!(input_collection, "input_array");
            assert_eq!(transform_function, "square_function");
        }
        _ => panic!("Expected map task"),
    }
    
    match reduce_task {
        ParallelTaskType::Reduce { ref input_collection, ref reduce_function, ref initial_value } => {
            assert_eq!(input_collection, "input_array");
            assert_eq!(reduce_function, "sum_function");
            match initial_value {
                AetherValue::Integer(0) => {},
                _ => panic!("Expected integer initial value"),
            }
        }
        _ => panic!("Expected reduce task"),
    }
    
    match custom_task {
        ParallelTaskType::Custom { ref operation_name, ref parameters } => {
            assert_eq!(operation_name, "custom_parallel_op");
            assert_eq!(parameters.len(), 1);
        }
        _ => panic!("Expected custom task"),
    }
}

#[test]
fn test_pipeline_stage_types() {
    let transform_stage = PipelineStageType::Transform {
        function: "map_function".to_string(),
    };
    
    let filter_stage = PipelineStageType::Filter {
        predicate: "is_positive".to_string(),
    };
    
    let aggregate_stage = PipelineStageType::Aggregate {
        aggregation_function: "sum".to_string(),
    };
    
    let branch_stage = PipelineStageType::Branch {
        condition: "value > threshold".to_string(),
    };
    
    match transform_stage {
        PipelineStageType::Transform { ref function } => {
            assert_eq!(function, "map_function");
        }
        _ => panic!("Expected transform stage"),
    }
    
    match filter_stage {
        PipelineStageType::Filter { ref predicate } => {
            assert_eq!(predicate, "is_positive");
        }
        _ => panic!("Expected filter stage"),
    }
    
    match aggregate_stage {
        PipelineStageType::Aggregate { ref aggregation_function } => {
            assert_eq!(aggregation_function, "sum");
        }
        _ => panic!("Expected aggregate stage"),
    }
    
    match branch_stage {
        PipelineStageType::Branch { ref condition } => {
            assert_eq!(condition, "value > threshold");
        }
        _ => panic!("Expected branch stage"),
    }
}

#[test]
fn test_gpu_kernel_types() {
    let compute_shader = GpuKernelType::ComputeShader {
        shader_code: "@compute @workgroup_size(64) fn main() {}".to_string(),
    };
    
    let cuda_kernel = GpuKernelType::CudaKernel {
        ptx_code: ".version 6.0\n.target sm_50\n.entry kernel() {}".to_string(),
    };
    
    let spirv_kernel = GpuKernelType::SpirvKernel {
        spirv_binary: vec![0x07, 0x23, 0x02, 0x03],
    };
    
    match compute_shader {
        GpuKernelType::ComputeShader { ref shader_code } => {
            assert!(shader_code.contains("@compute"));
        }
        _ => panic!("Expected compute shader"),
    }
    
    match cuda_kernel {
        GpuKernelType::CudaKernel { ref ptx_code } => {
            assert!(ptx_code.contains(".version"));
        }
        _ => panic!("Expected CUDA kernel"),
    }
    
    match spirv_kernel {
        GpuKernelType::SpirvKernel { ref spirv_binary } => {
            assert!(!spirv_binary.is_empty());
        }
        _ => panic!("Expected SPIR-V kernel"),
    }
}

#[test]
fn test_parallel_config() {
    let config = ParallelConfig {
        max_parallelism: 8,
        chunk_size: Some(256),
        load_balancing: LoadBalancingStrategy::WorkStealing,
        memory_model: MemoryModel::Shared,
    };
    
    assert_eq!(config.max_parallelism, 8);
    assert_eq!(config.chunk_size, Some(256));
    
    match config.load_balancing {
        LoadBalancingStrategy::WorkStealing => {},
        _ => panic!("Expected work stealing strategy"),
    }
    
    match config.memory_model {
        MemoryModel::Shared => {},
        _ => panic!("Expected shared memory model"),
    }
}

#[test]
fn test_load_balancing_strategies() {
    let static_strategy = LoadBalancingStrategy::Static;
    let work_stealing = LoadBalancingStrategy::WorkStealing;
    let guided = LoadBalancingStrategy::Guided { chunk_factor: 0.5 };
    
    match static_strategy {
        LoadBalancingStrategy::Static => {},
        _ => panic!("Expected static strategy"),
    }
    
    match work_stealing {
        LoadBalancingStrategy::WorkStealing => {},
        _ => panic!("Expected work stealing strategy"),
    }
    
    match guided {
        LoadBalancingStrategy::Guided { chunk_factor } => {
            assert_eq!(chunk_factor, 0.5);
        }
        _ => panic!("Expected guided strategy"),
    }
}

#[test]
fn test_memory_models() {
    let shared = MemoryModel::Shared;
    let distributed = MemoryModel::Distributed;
    let gpu = MemoryModel::GPU { device_memory: true, unified_memory: false };
    
    match shared {
        MemoryModel::Shared => {},
        _ => panic!("Expected shared memory model"),
    }
    
    match distributed {
        MemoryModel::Distributed => {},
        _ => panic!("Expected distributed memory model"),
    }
    
    match gpu {
        MemoryModel::GPU { device_memory: true, unified_memory: false } => {},
        _ => panic!("Expected GPU memory model"),
    }
}

#[test]
fn test_kernel_launch_config() {
    let config = KernelLaunchConfig {
        grid_size: (64, 1, 1),
        block_size: (256, 1, 1),
        shared_memory_size: 48 * 1024,
        stream_id: Some(1),
    };
    
    assert_eq!(config.grid_size, (64, 1, 1));
    assert_eq!(config.block_size, (256, 1, 1));
    assert_eq!(config.shared_memory_size, 48 * 1024);
    assert_eq!(config.stream_id, Some(1));
}

#[test]
fn test_gpu_memory_requirements() {
    let requirements = GpuMemoryRequirements {
        global_memory: 1024 * 1024,
        shared_memory: 48 * 1024,
        constant_memory: 64 * 1024,
        texture_memory: 512 * 1024,
    };
    
    assert_eq!(requirements.global_memory, 1024 * 1024);
    assert_eq!(requirements.shared_memory, 48 * 1024);
    assert_eq!(requirements.constant_memory, 64 * 1024);
    assert_eq!(requirements.texture_memory, 512 * 1024);
}

#[test]
fn test_concurrency_error_display() {
    let actor_not_found = ConcurrencyError::ActorNotFound(42);
    let message_delivery_failed = ConcurrencyError::MessageDeliveryFailed("Network error".to_string());
    let parallel_execution_error = ConcurrencyError::ParallelExecutionError("Thread panic".to_string());
    let gpu_kernel_error = ConcurrencyError::GpuKernelError("CUDA error".to_string());
    let snapshot_error = ConcurrencyError::SnapshotError("Checksum mismatch".to_string());
    let replay_error = ConcurrencyError::ReplayError("Invalid state".to_string());
    let type_mismatch = ConcurrencyError::TypeMismatch("Expected integer".to_string());
    let resource_exhaustion = ConcurrencyError::ResourceExhaustion("Out of memory".to_string());
    
    assert_eq!(format!("{}", actor_not_found), "Actor not found: 42");
    assert_eq!(format!("{}", message_delivery_failed), "Message delivery failed: Network error");
    assert_eq!(format!("{}", parallel_execution_error), "Parallel execution error: Thread panic");
    assert_eq!(format!("{}", gpu_kernel_error), "GPU kernel error: CUDA error");
    assert_eq!(format!("{}", snapshot_error), "Snapshot error: Checksum mismatch");
    assert_eq!(format!("{}", replay_error), "Replay error: Invalid state");
    assert_eq!(format!("{}", type_mismatch), "Type mismatch: Expected integer");
    assert_eq!(format!("{}", resource_exhaustion), "Resource exhaustion: Out of memory");
}

#[test]
fn test_default_implementations() {
    let default_config = ParallelConfig::default();
    assert!(default_config.max_parallelism > 0); // Should be number of CPUs
    assert_eq!(default_config.chunk_size, None);
    
    match default_config.load_balancing {
        LoadBalancingStrategy::WorkStealing => {},
        _ => panic!("Expected work stealing as default"),
    }
    
    match default_config.memory_model {
        MemoryModel::Shared => {},
        _ => panic!("Expected shared memory as default"),
    }
    
    let default_launch_config = KernelLaunchConfig::default();
    assert_eq!(default_launch_config.grid_size, (1, 1, 1));
    assert_eq!(default_launch_config.block_size, (256, 1, 1));
    assert_eq!(default_launch_config.shared_memory_size, 0);
    assert_eq!(default_launch_config.stream_id, None);
    
    let default_memory_requirements = GpuMemoryRequirements::default();
    assert_eq!(default_memory_requirements.global_memory, 1024 * 1024);
    assert_eq!(default_memory_requirements.shared_memory, 48 * 1024);
    assert_eq!(default_memory_requirements.constant_memory, 64 * 1024);
    assert_eq!(default_memory_requirements.texture_memory, 0);
}

#[test]
fn test_actor_registry_creation() {
    let registry = ActorRegistry::new();
    assert_eq!(registry.next_id, 1);
    assert!(registry.actors.is_empty());
    assert!(registry.message_queues.is_empty());
}

#[test]
fn test_task_scheduler_creation() {
    let scheduler = TaskScheduler::new();
    assert!(scheduler.parallel_tasks.is_empty());
    assert!(scheduler.pipeline_stages.is_empty());
    assert!(scheduler.gpu_kernels.is_empty());
}

#[test]
fn test_message_event_creation() {
    let message = TypedMessage {
        sender: 1,
        receiver: 2,
        message_type: "test".to_string(),
        payload: AetherValue::Integer(42),
        timestamp: 100,
    };
    
    let event = MessageEvent {
        message: message.clone(),
        processing_time: std::time::Duration::from_millis(10),
        result: MessageResult::Success(vec![]),
    };
    
    assert_eq!(event.message.sender, 1);
    assert_eq!(event.message.receiver, 2);
    assert_eq!(event.processing_time, std::time::Duration::from_millis(10));
    
    match event.result {
        MessageResult::Success(ref responses) => {
            assert!(responses.is_empty());
        }
        _ => panic!("Expected success result"),
    }
}

#[test]
fn test_state_snapshot_creation() {
    let execution_state = ExecutionState::new();
    let snapshot = StateSnapshot {
        timestamp: 100,
        execution_state: execution_state.clone(),
        checksum: 12345,
    };
    
    assert_eq!(snapshot.timestamp, 100);
    assert_eq!(snapshot.checksum, 12345);
    assert_eq!(snapshot.execution_state.timestamp, execution_state.timestamp);
}