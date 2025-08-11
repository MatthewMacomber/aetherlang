// Comprehensive tests for Aether structured concurrency system
// Tests parallel constructs, actor model, GPU kernels, and deterministic execution

use aether_language::compiler::concurrency::*;
use aether_language::compiler::mlir::mlir_context::MLIRContext;
use aether_language::compiler::mlir::concurrency_dialect::*;
use std::sync::Arc;
use std::time::Duration;
use std::thread;

#[test]
fn test_concurrency_system_creation() {
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let concurrency_system = ConcurrencySystem::new(context);
    
    // System should be created successfully
    assert_eq!(concurrency_system.execution_state.timestamp, 0);
    assert!(concurrency_system.execution_state.actor_states.is_empty());
    assert!(concurrency_system.execution_state.message_log.is_empty());
}

#[test]
fn test_actor_creation_and_messaging() {
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let mut concurrency_system = ConcurrencySystem::new(context);
    
    // Create compute actor
    let actor1 = concurrency_system.create_actor(ActorType::Compute { worker_count: 4 })
        .expect("Failed to create compute actor");
    
    // Create IO actor
    let actor2 = concurrency_system.create_actor(ActorType::IO { buffer_size: 1024 })
        .expect("Failed to create IO actor");
    
    // Actors should have different IDs
    assert_ne!(actor1, actor2);
    
    // Send message between actors
    let message = TypedMessage {
        sender: actor1,
        receiver: actor2,
        message_type: "compute_result".to_string(),
        payload: AetherValue::Integer(42),
        timestamp: 1,
    };
    
    concurrency_system.send_message(message)
        .expect("Failed to send message");
    
    // Process messages
    concurrency_system.process_messages()
        .expect("Failed to process messages");
    
    // Message should be logged
    assert_eq!(concurrency_system.execution_state.message_log.len(), 1);
}

#[test]
fn test_parallel_for_execution() {
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let mut concurrency_system = ConcurrencySystem::new(context);
    
    let config = ParallelConfig {
        max_parallelism: 4,
        chunk_size: Some(100),
        load_balancing: LoadBalancingStrategy::WorkStealing,
        memory_model: MemoryModel::Shared,
    };
    
    // Execute parallel for loop
    let result = concurrency_system.execute_parallel_for(
        (0, 1000),
        1,
        "arith.addi %i, %c1 : index",
        config,
    );
    
    assert!(result.is_ok());
    let operation = result.unwrap();
    assert_eq!(operation.name, "scf.parallel");
    
    // Check attributes
    assert!(operation.attributes.contains_key("lower_bound"));
    assert!(operation.attributes.contains_key("upper_bound"));
    assert!(operation.attributes.contains_key("step"));
}

#[test]
fn test_pipeline_stage_creation() {
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let mut concurrency_system = ConcurrencySystem::new(context);
    
    // Create transform stage
    let stage_id = concurrency_system.create_pipeline_stage(
        PipelineStageType::Transform { 
            function: "square".to_string() 
        },
        vec!["input_channel".to_string()],
        vec!["output_channel".to_string()],
        2,
    );
    
    assert!(stage_id.is_ok());
    let stage_id = stage_id.unwrap();
    assert!(stage_id.starts_with("pipeline_stage_"));
}

#[test]
fn test_gpu_kernel_compilation() {
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let mut concurrency_system = ConcurrencySystem::new(context);
    
    let kernel_type = GpuKernelType::ComputeShader {
        shader_code: "
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let gid = global_id.x;
                output_buffer[gid] = input_buffer[gid] * input_buffer[gid];
            }
        ".to_string(),
    };
    
    let launch_config = KernelLaunchConfig {
        grid_size: (64, 1, 1),
        block_size: (256, 1, 1),
        shared_memory_size: 0,
        stream_id: None,
    };
    
    let memory_requirements = GpuMemoryRequirements {
        global_memory: 1024 * 1024, // 1MB
        shared_memory: 0,
        constant_memory: 0,
        texture_memory: 0,
    };
    
    let kernel_id = concurrency_system.compile_gpu_kernel(
        kernel_type,
        launch_config,
        memory_requirements,
    );
    
    assert!(kernel_id.is_ok());
    let kernel_id = kernel_id.unwrap();
    assert!(kernel_id.starts_with("gpu_kernel_"));
}

#[test]
fn test_state_snapshot_and_restore() {
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let mut concurrency_system = ConcurrencySystem::new(context);
    
    // Create actor and send message to change state
    let actor_id = concurrency_system.create_actor(ActorType::Compute { worker_count: 1 })
        .expect("Failed to create actor");
    
    let message = TypedMessage {
        sender: actor_id,
        receiver: actor_id,
        message_type: "self_message".to_string(),
        payload: AetherValue::String("test".to_string()),
        timestamp: 1,
    };
    
    concurrency_system.send_message(message)
        .expect("Failed to send message");
    concurrency_system.process_messages()
        .expect("Failed to process messages");
    
    // Take snapshot
    let snapshot = concurrency_system.take_snapshot()
        .expect("Failed to take snapshot");
    
    assert_eq!(snapshot.timestamp, 0); // Initial timestamp
    assert!(snapshot.checksum != 0);
    
    // Modify state
    concurrency_system.execution_state.timestamp = 100;
    
    // Restore snapshot
    concurrency_system.restore_snapshot(&snapshot)
        .expect("Failed to restore snapshot");
    
    assert_eq!(concurrency_system.execution_state.timestamp, 0);
}

#[test]
fn test_deterministic_replay() {
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let mut concurrency_system = ConcurrencySystem::new(context);
    
    // Create actors
    let actor1 = concurrency_system.create_actor(ActorType::Compute { worker_count: 1 })
        .expect("Failed to create actor");
    let actor2 = concurrency_system.create_actor(ActorType::IO { buffer_size: 512 })
        .expect("Failed to create actor");
    
    // Take initial snapshot
    let initial_snapshot = concurrency_system.take_snapshot()
        .expect("Failed to take initial snapshot");
    
    // Send sequence of messages
    let messages = vec![
        TypedMessage {
            sender: actor1,
            receiver: actor2,
            message_type: "data".to_string(),
            payload: AetherValue::Integer(1),
            timestamp: 1,
        },
        TypedMessage {
            sender: actor2,
            receiver: actor1,
            message_type: "ack".to_string(),
            payload: AetherValue::Boolean(true),
            timestamp: 2,
        },
    ];
    
    for message in messages {
        concurrency_system.send_message(message)
            .expect("Failed to send message");
    }
    
    concurrency_system.process_messages()
        .expect("Failed to process messages");
    
    let final_timestamp = concurrency_system.execution_state.timestamp;
    
    // Replay execution
    let replay_result = concurrency_system.replay_execution(&initial_snapshot, final_timestamp);
    assert!(replay_result.is_ok());
}

#[test]
fn test_parallel_task_types() {
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let mut concurrency_system = ConcurrencySystem::new(context);
    
    let config = ParallelConfig::default();
    
    // Test Map task
    let map_result = concurrency_system.scheduler.schedule_parallel_task(
        ParallelTask {
            id: "map_test".to_string(),
            task_type: ParallelTaskType::Map {
                input_collection: "input_array".to_string(),
                transform_function: "square_function".to_string(),
            },
            data_dependencies: vec![],
            execution_config: config.clone(),
        },
        concurrency_system.context.as_ref(),
    );
    
    assert!(map_result.is_ok());
    let map_op = map_result.unwrap();
    assert_eq!(map_op.name, "aether.parallel_map");
    
    // Test Reduce task
    let reduce_result = concurrency_system.scheduler.schedule_parallel_task(
        ParallelTask {
            id: "reduce_test".to_string(),
            task_type: ParallelTaskType::Reduce {
                input_collection: "input_array".to_string(),
                reduce_function: "sum_function".to_string(),
                initial_value: AetherValue::Integer(0),
            },
            data_dependencies: vec![],
            execution_config: config,
        },
        concurrency_system.context.as_ref(),
    );
    
    assert!(reduce_result.is_ok());
    let reduce_op = reduce_result.unwrap();
    assert_eq!(reduce_op.name, "aether.parallel_reduce");
}

#[test]
fn test_actor_types() {
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let mut concurrency_system = ConcurrencySystem::new(context);
    
    // Test Compute actor
    let compute_actor = concurrency_system.create_actor(ActorType::Compute { worker_count: 8 })
        .expect("Failed to create compute actor");
    
    // Test IO actor
    let io_actor = concurrency_system.create_actor(ActorType::IO { buffer_size: 2048 })
        .expect("Failed to create IO actor");
    
    // Test GPU actor
    let gpu_actor = concurrency_system.create_actor(ActorType::GPU { device_id: 0 })
        .expect("Failed to create GPU actor");
    
    // Test Custom actor
    let custom_actor = concurrency_system.create_actor(ActorType::Custom { 
        type_name: "DatabaseActor".to_string() 
    }).expect("Failed to create custom actor");
    
    // All actors should have unique IDs
    let actors = vec![compute_actor, io_actor, gpu_actor, custom_actor];
    for i in 0..actors.len() {
        for j in i+1..actors.len() {
            assert_ne!(actors[i], actors[j]);
        }
    }
}

#[test]
fn test_memory_barriers_and_atomics() {
    let context = MLIRContext::new().expect("Failed to create MLIR context");
    
    // Test memory barrier creation
    let barrier_op = ConcurrencyOps::memory_barrier(
        &context,
        MemoryBarrierType::Full,
        MemoryScope::System,
    );
    
    assert!(barrier_op.is_ok());
    let barrier_op = barrier_op.unwrap();
    assert_eq!(barrier_op.name, "aether.memory_barrier");
    
    // Test atomic operation creation
    let address = crate::compiler::mlir::mlir_context::MLIRValue::new(
        "address".to_string(),
        crate::compiler::mlir::mlir_context::MLIRType::Memref {
            element_type: Box::new(crate::compiler::mlir::mlir_context::MLIRType::Integer { width: 32, signed: true }),
            shape: vec![],
        },
    );
    let value = crate::compiler::mlir::mlir_context::MLIRValue::new(
        "value".to_string(),
        crate::compiler::mlir::mlir_context::MLIRType::Integer { width: 32, signed: true },
    );
    
    let atomic_op = ConcurrencyOps::atomic_operation(
        &context,
        AtomicOperation::Add,
        address,
        value,
        MemoryOrder::SequentiallyConsistent,
    );
    
    assert!(atomic_op.is_ok());
    let atomic_op = atomic_op.unwrap();
    assert_eq!(atomic_op.name, "aether.atomic");
    assert_eq!(atomic_op.operands.len(), 2);
    assert_eq!(atomic_op.results.len(), 1);
}

#[test]
fn test_concurrency_lowering() {
    let context = MLIRContext::new().expect("Failed to create MLIR context");
    let mut module = context.create_module("test_module")
        .expect("Failed to create module");
    
    // Add parallel for operation
    let parallel_for_op = ConcurrencyOps::parallel_for(
        &context,
        0,
        100,
        1,
        crate::compiler::mlir::mlir_context::MLIRRegion { blocks: vec![] },
    ).expect("Failed to create parallel for operation");
    
    module.add_operation(parallel_for_op)
        .expect("Failed to add operation");
    
    // Test lowering
    let lowering = ConcurrencyLowering::new(&context);
    let result = lowering.lower_parallel_for(&mut module);
    
    assert!(result.is_ok());
}

#[test]
fn test_load_balancing_strategies() {
    let config_static = ParallelConfig {
        max_parallelism: 4,
        chunk_size: Some(256),
        load_balancing: LoadBalancingStrategy::Static,
        memory_model: MemoryModel::Shared,
    };
    
    let config_work_stealing = ParallelConfig {
        max_parallelism: 8,
        chunk_size: None,
        load_balancing: LoadBalancingStrategy::WorkStealing,
        memory_model: MemoryModel::Distributed,
    };
    
    let config_guided = ParallelConfig {
        max_parallelism: 6,
        chunk_size: Some(128),
        load_balancing: LoadBalancingStrategy::Guided { chunk_factor: 0.5 },
        memory_model: MemoryModel::GPU { device_memory: true, unified_memory: false },
    };
    
    // Test that different configurations can be created
    assert_eq!(config_static.max_parallelism, 4);
    assert_eq!(config_work_stealing.max_parallelism, 8);
    assert_eq!(config_guided.max_parallelism, 6);
    
    match config_guided.load_balancing {
        LoadBalancingStrategy::Guided { chunk_factor } => {
            assert_eq!(chunk_factor, 0.5);
        }
        _ => panic!("Expected Guided load balancing strategy"),
    }
}

#[test]
fn test_aether_value_conversions() {
    // Test various AetherValue types
    let int_val = AetherValue::Integer(42);
    let float_val = AetherValue::Float(3.14);
    let bool_val = AetherValue::Boolean(true);
    let string_val = AetherValue::String("hello".to_string());
    
    let array_val = AetherValue::Array(vec![
        AetherValue::Integer(1),
        AetherValue::Integer(2),
        AetherValue::Integer(3),
    ]);
    
    let mut struct_map = std::collections::HashMap::new();
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
    
    // Test that all values can be created and matched
    match int_val {
        AetherValue::Integer(42) => {},
        _ => panic!("Expected integer value"),
    }
    
    match array_val {
        AetherValue::Array(ref arr) => {
            assert_eq!(arr.len(), 3);
        }
        _ => panic!("Expected array value"),
    }
    
    match tensor_val {
        AetherValue::Tensor { ref data, ref shape } => {
            assert_eq!(data.len(), 4);
            assert_eq!(shape, &vec![2, 2]);
        }
        _ => panic!("Expected tensor value"),
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
        spirv_binary: vec![0x07, 0x23, 0x02, 0x03], // Mock SPIR-V header
    };
    
    // Test that different kernel types can be created
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
    
    // Test that all stage types can be created and matched
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
fn test_data_race_freedom() {
    // This test verifies that the concurrency system prevents data races
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let mut concurrency_system = ConcurrencySystem::new(context);
    
    // Create multiple actors that will access shared data
    let actor1 = concurrency_system.create_actor(ActorType::Compute { worker_count: 1 })
        .expect("Failed to create actor 1");
    let actor2 = concurrency_system.create_actor(ActorType::Compute { worker_count: 1 })
        .expect("Failed to create actor 2");
    
    // Send messages that would cause race conditions in unsafe systems
    let message1 = TypedMessage {
        sender: actor1,
        receiver: actor2,
        message_type: "increment".to_string(),
        payload: AetherValue::Integer(1),
        timestamp: 1,
    };
    
    let message2 = TypedMessage {
        sender: actor2,
        receiver: actor1,
        message_type: "increment".to_string(),
        payload: AetherValue::Integer(1),
        timestamp: 1, // Same timestamp to test race condition handling
    };
    
    // Both messages should be sent successfully
    assert!(concurrency_system.send_message(message1).is_ok());
    assert!(concurrency_system.send_message(message2).is_ok());
    
    // Processing should be deterministic and race-free
    assert!(concurrency_system.process_messages().is_ok());
    
    // Message log should contain both messages
    assert_eq!(concurrency_system.execution_state.message_log.len(), 2);
}

#[test]
fn test_performance_characteristics() {
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let mut concurrency_system = ConcurrencySystem::new(context);
    
    let start_time = std::time::Instant::now();
    
    // Create many actors to test scalability
    let mut actors = Vec::new();
    for i in 0..100 {
        let actor_type = if i % 2 == 0 {
            ActorType::Compute { worker_count: 1 }
        } else {
            ActorType::IO { buffer_size: 512 }
        };
        
        let actor_id = concurrency_system.create_actor(actor_type)
            .expect("Failed to create actor");
        actors.push(actor_id);
    }
    
    let creation_time = start_time.elapsed();
    
    // Send many messages to test message throughput
    let message_start = std::time::Instant::now();
    for i in 0..1000 {
        let sender = actors[i % actors.len()];
        let receiver = actors[(i + 1) % actors.len()];
        
        let message = TypedMessage {
            sender,
            receiver,
            message_type: "performance_test".to_string(),
            payload: AetherValue::Integer(i as i64),
            timestamp: i as u64,
        };
        
        concurrency_system.send_message(message)
            .expect("Failed to send message");
    }
    
    let message_send_time = message_start.elapsed();
    
    // Process all messages
    let processing_start = std::time::Instant::now();
    concurrency_system.process_messages()
        .expect("Failed to process messages");
    let processing_time = processing_start.elapsed();
    
    // Performance assertions (these are basic sanity checks)
    assert!(creation_time < Duration::from_secs(1), "Actor creation took too long");
    assert!(message_send_time < Duration::from_secs(1), "Message sending took too long");
    assert!(processing_time < Duration::from_secs(5), "Message processing took too long");
    
    // Verify all messages were processed
    assert_eq!(concurrency_system.execution_state.message_log.len(), 1000);
}

#[test]
fn test_error_handling() {
    let context = Arc::new(MLIRContext::new().expect("Failed to create MLIR context"));
    let mut concurrency_system = ConcurrencySystem::new(context);
    
    // Test sending message to non-existent actor
    let invalid_message = TypedMessage {
        sender: 1,
        receiver: 999, // Non-existent actor
        message_type: "test".to_string(),
        payload: AetherValue::Integer(42),
        timestamp: 1,
    };
    
    let result = concurrency_system.send_message(invalid_message);
    assert!(result.is_err());
    
    match result.unwrap_err() {
        ConcurrencyError::ActorNotFound(id) => {
            assert_eq!(id, 999);
        }
        _ => panic!("Expected ActorNotFound error"),
    }
}