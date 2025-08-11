// Tests for GPU kernel generation system
// Validates CUDA and SPIR-V kernel generation from MLIR GPU dialects

use aether_language::compiler::mlir::*;
use aether_language::compiler::ast::*;
use aether_language::compiler::types::*;

#[test]
fn test_gpu_dialect_operations() {
    let context = MockMLIRContext::new();
    
    // Test GPU kernel function creation
    let kernel_params = vec![("input", "memref<?xf32>"), ("output", "memref<?xf32>")];
    let kernel_op = GpuOps::kernel_func(&context, "test_kernel", &kernel_params, GpuTarget::Cuda);
    assert!(kernel_op.is_ok());
    
    let kernel = kernel_op.unwrap();
    assert_eq!(kernel.name, "gpu.func");
    assert_eq!(kernel.attributes.get("sym_name"), Some(&"test_kernel".to_string()));
    assert_eq!(kernel.attributes.get("target"), Some(&"cuda".to_string()));
    assert_eq!(kernel.attributes.get("kernel"), Some(&"true".to_string()));
}

#[test]
fn test_gpu_memory_operations() {
    let context = MockMLIRContext::new();
    
    // Test GPU memory allocation
    let memory_config = GpuMemoryConfig {
        memory_type: GpuMemoryType::Global,
        alignment: 256,
        coalescing_hint: true,
        cache_hint: GpuCacheHint::CacheAll,
    };
    
    let alloc_op = GpuOps::gpu_alloc(&context, "%size", &memory_config);
    assert!(alloc_op.is_ok());
    
    let alloc = alloc_op.unwrap();
    assert_eq!(alloc.name, "gpu.alloc");
    assert_eq!(alloc.attributes.get("memory_type"), Some(&"global".to_string()));
    assert_eq!(alloc.attributes.get("alignment"), Some(&"256".to_string()));
    
    // Test GPU memory deallocation
    let dealloc_op = GpuOps::gpu_dealloc(&context, "%memref");
    assert!(dealloc_op.is_ok());
    
    let dealloc = dealloc_op.unwrap();
    assert_eq!(dealloc.name, "gpu.dealloc");
    assert_eq!(dealloc.operands.len(), 1);
}

#[test]
fn test_gpu_kernel_launch() {
    let context = MockMLIRContext::new();
    
    let launch_config = KernelLaunchConfig {
        grid_size: (32, 32, 1),
        block_size: (16, 16, 1),
        shared_memory_size: 1024,
        stream: Some("stream0".to_string()),
    };
    
    let args = vec!["%input", "%output", "%size"];
    let launch_op = GpuOps::kernel_launch(&context, "test_kernel", &launch_config, &args);
    assert!(launch_op.is_ok());
    
    let launch = launch_op.unwrap();
    assert_eq!(launch.name, "gpu.launch_func");
    assert_eq!(launch.attributes.get("kernel"), Some(&"test_kernel".to_string()));
    assert_eq!(launch.attributes.get("gridSizeX"), Some(&"32".to_string()));
    assert_eq!(launch.attributes.get("blockSizeX"), Some(&"16".to_string()));
    assert_eq!(launch.operands.len(), 3);
}

#[test]
fn test_gpu_thread_operations() {
    let context = MockMLIRContext::new();
    
    // Test thread ID access
    let thread_id_op = GpuOps::thread_id(&context, GpuDimension::X);
    assert!(thread_id_op.is_ok());
    
    let thread_id = thread_id_op.unwrap();
    assert_eq!(thread_id.name, "gpu.thread_id");
    assert_eq!(thread_id.attributes.get("dimension"), Some(&"x".to_string()));
    assert_eq!(thread_id.results.len(), 1);
    
    // Test block ID access
    let block_id_op = GpuOps::block_id(&context, GpuDimension::Y);
    assert!(block_id_op.is_ok());
    
    let block_id = block_id_op.unwrap();
    assert_eq!(block_id.name, "gpu.block_id");
    assert_eq!(block_id.attributes.get("dimension"), Some(&"y".to_string()));
}

#[test]
fn test_gpu_atomic_operations() {
    let context = MockMLIRContext::new();
    
    let atomic_op = GpuOps::atomic_op(&context, GpuAtomicOp::Add, "%address", "%value");
    assert!(atomic_op.is_ok());
    
    let atomic = atomic_op.unwrap();
    assert_eq!(atomic.name, "gpu.atomic");
    assert_eq!(atomic.attributes.get("operation"), Some(&"add".to_string()));
    assert_eq!(atomic.operands.len(), 2);
    assert_eq!(atomic.results.len(), 1);
}

#[test]
fn test_gpu_lowering_pipeline() {
    let context = MockMLIRContext::new();
    let mut module = MockMLIRModule::new();
    
    // Add some operations that can be parallelized
    module.add_operation("linalg.generic".to_string());
    module.add_operation("aether.tensor_op: elementwise_add".to_string());
    module.add_operation("aether.tensor_op: matmul".to_string());
    
    // Test CUDA lowering
    let cuda_lowering = GpuLowering::new(&context, GpuTarget::Cuda);
    let result = cuda_lowering.lower_to_gpu_kernels(&mut module);
    assert!(result.is_ok());
    
    // Check that parallel markers were added
    let has_parallel_marker = module.operations.iter()
        .any(|op| op.contains("gpu.parallel_marker"));
    assert!(has_parallel_marker);
    
    // Check that kernels were generated
    let has_kernel = module.operations.iter()
        .any(|op| op.contains("gpu.func") && op.contains("kernel"));
    assert!(has_kernel);
}

#[test]
fn test_gpu_lowering_different_targets() {
    let context = MockMLIRContext::new();
    
    // Test CUDA target
    let mut cuda_module = MockMLIRModule::new();
    cuda_module.add_operation("aether.tensor_op: elementwise".to_string());
    
    let cuda_lowering = GpuLowering::new(&context, GpuTarget::Cuda);
    let cuda_result = cuda_lowering.lower_to_gpu_kernels(&mut cuda_module);
    assert!(cuda_result.is_ok());
    
    // Test SPIR-V target
    let mut spirv_module = MockMLIRModule::new();
    spirv_module.add_operation("aether.tensor_op: elementwise".to_string());
    
    let spirv_lowering = GpuLowering::new(&context, GpuTarget::SpirV);
    let spirv_result = spirv_lowering.lower_to_gpu_kernels(&mut spirv_module);
    assert!(spirv_result.is_ok());
    
    // Test WebGPU target
    let mut webgpu_module = MockMLIRModule::new();
    webgpu_module.add_operation("aether.tensor_op: elementwise".to_string());
    
    let webgpu_lowering = GpuLowering::new(&context, GpuTarget::WebGpu);
    let webgpu_result = webgpu_lowering.lower_to_gpu_kernels(&mut webgpu_module);
    assert!(webgpu_result.is_ok());
}

#[test]
fn test_parallelization_pattern_detection() {
    let context = MockMLIRContext::new();
    let lowering = GpuLowering::new(&context, GpuTarget::Cuda);
    
    // Test different operation patterns
    assert!(lowering.is_parallelizable_operation("linalg.generic"));
    assert!(lowering.is_parallelizable_operation("tensor_op: map"));
    assert!(lowering.is_parallelizable_operation("aether.tensor_op: reduce"));
    assert!(lowering.is_parallelizable_operation("@parallel for"));
    assert!(lowering.is_parallelizable_operation("matmul"));
    
    // Test non-parallelizable operations
    assert!(!lowering.is_parallelizable_operation("func.call"));
    assert!(!lowering.is_parallelizable_operation("scf.if"));
    assert!(!lowering.is_parallelizable_operation("memref.alloc"));
}

#[test]
fn test_cuda_code_generation() {
    let context = MockMLIRContext::new();
    let mut module = MockMLIRModule::new();
    
    // Add GPU kernel operations
    module.add_operation("gpu.func @elementwise_kernel kernel".to_string());
    module.add_operation("gpu.func @matmul_kernel kernel".to_string());
    
    let codegen = GpuCodeGenerator::new(&context, GpuTarget::Cuda);
    let cuda_code = codegen.generate_kernel_code(&module);
    assert!(cuda_code.is_ok());
    
    let code = cuda_code.unwrap();
    assert!(code.contains("#include <cuda_runtime.h>"));
    assert!(code.contains("__global__ void"));
    assert!(code.contains("CUDA_CHECK"));
    assert!(code.contains("launch_"));
}

#[test]
fn test_spirv_code_generation() {
    let context = MockMLIRContext::new();
    let mut module = MockMLIRModule::new();
    
    // Add SPIR-V kernel operations
    module.add_operation("spirv.func @elementwise_kernel \"Kernel\"".to_string());
    
    let codegen = GpuCodeGenerator::new(&context, GpuTarget::SpirV);
    let spirv_code = codegen.generate_kernel_code(&module);
    assert!(spirv_code.is_ok());
    
    let code = spirv_code.unwrap();
    assert!(code.contains("; SPIR-V"));
    assert!(code.contains("OpCapability"));
    assert!(code.contains("OpMemoryModel"));
    assert!(code.contains("OpEntryPoint"));
}

#[test]
fn test_webgpu_code_generation() {
    let context = MockMLIRContext::new();
    let mut module = MockMLIRModule::new();
    
    // Add WebGPU compute shader operations
    module.add_operation("webgpu.compute_shader @elementwise_kernel".to_string());
    
    let codegen = GpuCodeGenerator::new(&context, GpuTarget::WebGpu);
    let webgpu_code = codegen.generate_kernel_code(&module);
    assert!(webgpu_code.is_ok());
    
    let code = webgpu_code.unwrap();
    assert!(code.contains("// WebGPU Compute Shader"));
    assert!(code.contains("@group(0) @binding"));
    assert!(code.contains("@compute @workgroup_size"));
    assert!(code.contains("@builtin(global_invocation_id)"));
}

#[test]
fn test_tensor_operation_optimization() {
    let context = MockMLIRContext::new();
    let mut module = MockMLIRModule::new();
    
    // Add tensor operations that can be optimized
    module.add_operation("arith.addf %a, %b : f32".to_string());
    module.add_operation("arith.mulf %c, %d : f32".to_string());
    module.add_operation("math.exp %e : f32".to_string());
    module.add_operation("linalg.matmul".to_string());
    
    let optimizer = GpuTensorOptimizer::new(&context, GpuTarget::Cuda);
    let result = optimizer.optimize_tensor_operations(&mut module);
    assert!(result.is_ok());
    
    // Check that operations were optimized
    let has_fused_ops = module.operations.iter()
        .any(|op| op.contains("gpu.fused_kernel"));
    assert!(has_fused_ops);
    
    let has_vectorized_ops = module.operations.iter()
        .any(|op| op.contains("vectorized"));
    assert!(has_vectorized_ops);
}

#[test]
fn test_tensor_operation_fusion() {
    let context = MockMLIRContext::new();
    let mut module = MockMLIRModule::new();
    
    // Add fusable operations
    module.add_operation("arith.addf %a, %b : f32".to_string());
    module.add_operation("arith.mulf %result0, %c : f32".to_string());
    module.add_operation("math.tanh %result1 : f32".to_string());
    
    let optimizer = GpuTensorOptimizer::new(&context, GpuTarget::Cuda);
    let result = optimizer.optimize_tensor_operations(&mut module);
    assert!(result.is_ok());
    
    // Check that fusion occurred
    let fused_count = module.operations.iter()
        .filter(|op| op.contains("gpu.fused_kernel"))
        .count();
    assert!(fused_count > 0);
}

#[test]
fn test_memory_layout_optimization() {
    let context = MockMLIRContext::new();
    let mut module = MockMLIRModule::new();
    
    // Add memory operations
    module.add_operation("memref.alloc() : memref<1024xf32>".to_string());
    module.add_operation("tensor.extract %tensor[%i] : tensor<?xf32>".to_string());
    module.add_operation("tensor.insert %val, %tensor[%i] : tensor<?xf32>".to_string());
    
    let optimizer = GpuTensorOptimizer::new(&context, GpuTarget::Cuda);
    let result = optimizer.optimize_tensor_operations(&mut module);
    assert!(result.is_ok());
    
    // Check that memory layout hints were added
    let has_layout_hints = module.operations.iter()
        .any(|op| op.contains("layout: coalesced") || op.contains("access: vectorized"));
    assert!(has_layout_hints);
}

#[test]
fn test_tensor_core_optimization() {
    let context = MockMLIRContext::new();
    let mut module = MockMLIRModule::new();
    
    // Add operations that can use tensor cores
    module.add_operation("linalg.matmul ins(%A, %B : tensor<16x16xf16>, tensor<16x16xf16>)".to_string());
    module.add_operation("linalg.conv_2d ins(%input, %filter : tensor<1x224x224x3xf16>, tensor<64x7x7x3xf16>)".to_string());
    
    let optimizer = GpuTensorOptimizer::new(&context, GpuTarget::Cuda);
    let result = optimizer.optimize_tensor_operations(&mut module);
    assert!(result.is_ok());
    
    // Check that tensor core optimizations were applied
    let has_tensor_core_ops = module.operations.iter()
        .any(|op| op.contains("tensor_core"));
    assert!(has_tensor_core_ops);
}

#[test]
fn test_gpu_memory_manager() {
    let mut memory_manager = GpuTensorMemoryManager::new(GpuTarget::Cuda);
    
    // Test tensor allocation
    let shape = Shape::concrete(vec![1024, 1024]);
    let allocation = memory_manager.allocate_tensor(
        "tensor_a".to_string(),
        &shape,
        PrimitiveType::Float32,
        GpuMemoryType::Global,
    );
    assert!(allocation.is_ok());
    
    let alloc = allocation.unwrap();
    assert_eq!(alloc.id, "tensor_a");
    assert_eq!(alloc.size, 1024 * 1024 * 4); // 4 bytes per float32
    assert_eq!(alloc.tensor_shape, shape);
    assert_eq!(alloc.data_type, PrimitiveType::Float32);
}

#[test]
fn test_gpu_memory_pool() {
    let mut pool = GpuMemoryPool::new(GpuMemoryType::Global, 1024 * 1024); // 1MB pool
    
    // Test allocation
    let shape = Shape::concrete(vec![256, 256]);
    let allocation = pool.allocate(
        "test_tensor".to_string(),
        256 * 256 * 4, // 256KB
        shape.clone(),
        PrimitiveType::Float32,
    );
    assert!(allocation.is_ok());
    
    let alloc = allocation.unwrap();
    assert_eq!(alloc.id, "test_tensor");
    assert_eq!(alloc.tensor_shape, shape);
    
    // Test deallocation
    let dealloc_result = pool.deallocate("test_tensor");
    assert!(dealloc_result.is_ok());
    assert_eq!(pool.used_size, 0);
}

#[test]
fn test_tensor_operation_pattern_analysis() {
    let input_shapes = vec![
        Shape::concrete(vec![1024, 1024]),
        Shape::concrete(vec![1024, 512]),
    ];
    let output_shape = Shape::concrete(vec![1024, 512]);
    
    let mut pattern = TensorOpPattern::new(
        TensorOpType::MatMul,
        input_shapes,
        output_shape,
        PrimitiveType::Float32,
    );
    
    // Analyze for CUDA target
    let result = pattern.analyze_for_gpu(GpuTarget::Cuda);
    assert!(result.is_ok());
    
    assert_eq!(pattern.optimization_hints.memory_layout, MemoryLayout::RowMajor);
    assert_eq!(pattern.optimization_hints.vector_width, 4); // float4 for CUDA
    assert!(pattern.optimization_hints.use_tensor_cores); // MatMul with f32 should use tensor cores
}

#[test]
fn test_end_to_end_gpu_compilation() {
    let context = MockMLIRContext::new();
    let pipeline = TestMLIRPipeline::new();
    
    // Create a simple AST with tensor operations
    let ast = AST::new(ASTNode::List(vec![
        ASTNodeRef::direct(ASTNode::symbol("tensor_add".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("a".to_string())),
        ASTNodeRef::direct(ASTNode::symbol("b".to_string())),
    ]));
    
    // Compile to MLIR
    let mlir_result = pipeline.compile_ast(&ast);
    assert!(mlir_result.is_ok());
    
    let mut module = mlir_result.unwrap();
    
    // Add some tensor operations for testing
    module.add_operation("aether.tensor_op: elementwise_add".to_string());
    module.add_operation("aether.tensor_op: matmul".to_string());
    
    // Lower to GPU kernels
    let gpu_result = pipeline.lower_to_gpu_kernels(&mut module, GpuTarget::Cuda);
    assert!(gpu_result.is_ok());
    
    // Optimize tensor operations
    let opt_result = pipeline.optimize_gpu_tensors(&mut module, GpuTarget::Cuda);
    assert!(opt_result.is_ok());
    
    // Generate GPU code
    let code_result = pipeline.generate_gpu_code(&module, GpuTarget::Cuda);
    assert!(code_result.is_ok());
    
    let gpu_code = code_result.unwrap();
    assert!(gpu_code.contains("__global__"));
    assert!(gpu_code.contains("CUDA_CHECK"));
}

#[test]
fn test_gpu_kernel_performance_validation() {
    let context = MockMLIRContext::new();
    let mut module = MockMLIRModule::new();
    
    // Add performance-critical operations
    module.add_operation("linalg.matmul ins(%A, %B : tensor<4096x4096xf32>, tensor<4096x4096xf32>)".to_string());
    module.add_operation("linalg.generic elementwise_add".to_string());
    module.add_operation("linalg.reduce sum".to_string());
    
    // Test optimization for different targets
    for target in [GpuTarget::Cuda, GpuTarget::SpirV, GpuTarget::WebGpu] {
        let optimizer = GpuTensorOptimizer::new(&context, target);
        let result = optimizer.optimize_tensor_operations(&mut module.clone());
        assert!(result.is_ok(), "Optimization failed for target: {:?}", target);
        
        let codegen = GpuCodeGenerator::new(&context, target);
        let code_result = codegen.generate_kernel_code(&module);
        assert!(code_result.is_ok(), "Code generation failed for target: {:?}", target);
    }
}

#[test]
fn test_gpu_memory_safety_validation() {
    let mut memory_manager = GpuTensorMemoryManager::new(GpuTarget::Cuda);
    
    // Test memory bounds checking
    let large_shape = Shape::concrete(vec![100000, 100000]); // Very large tensor (40GB)
    let allocation = memory_manager.allocate_tensor(
        "large_tensor".to_string(),
        &large_shape,
        PrimitiveType::Float64,
        GpuMemoryType::Global,
    );
    
    // Should fail due to memory constraints
    assert!(allocation.is_err());
    
    // Test valid allocation
    let small_shape = Shape::concrete(vec![100, 100]);
    let small_allocation = memory_manager.allocate_tensor(
        "small_tensor".to_string(),
        &small_shape,
        PrimitiveType::Float32,
        GpuMemoryType::Global,
    );
    assert!(small_allocation.is_ok());
}

#[test]
fn test_gpu_kernel_correctness_validation() {
    let context = MockMLIRContext::new();
    
    // Test kernel generation for different operation types
    let operation_types = vec![
        ("elementwise", "arith.addf %a, %b : f32"),
        ("matmul", "linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)"),
        ("reduction", "linalg.reduce add ins(%input : tensor<?xf32>)"),
        ("convolution", "linalg.conv_2d ins(%input, %filter : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)"),
    ];
    
    for (op_name, op_mlir) in operation_types {
        let mut module = MockMLIRModule::new();
        module.add_operation(op_mlir.to_string());
        
        let lowering = GpuLowering::new(&context, GpuTarget::Cuda);
        let result = lowering.lower_to_gpu_kernels(&mut module);
        assert!(result.is_ok(), "Failed to lower {} operation", op_name);
        
        // Verify kernel was generated (either gpu.func or fused_kernel)
        let has_kernel = module.operations.iter()
            .any(|op| op.contains("gpu.func") && op.contains("kernel") || op.contains("gpu.fused_kernel"));
        assert!(has_kernel, "No kernel generated for {} operation", op_name);
    }
}