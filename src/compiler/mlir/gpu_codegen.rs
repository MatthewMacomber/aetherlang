// GPU code generation for CUDA and SPIR-V targets
// Generates actual GPU kernel code from MLIR GPU dialect

use crate::compiler::mlir::mlir_context::{MLIRError, MLIRContext, MLIRModule, MLIROperation, MLIRType, MLIRAttribute};
use crate::compiler::mlir::gpu_dialect::*;

/// GPU code generator
pub struct GpuCodeGenerator<'a> {
    context: &'a MLIRContext,
    target: GpuTarget,
}

impl<'a> GpuCodeGenerator<'a> {
    /// Create new GPU code generator
    pub fn new(context: &'a MLIRContext, target: GpuTarget) -> Self {
        GpuCodeGenerator { context, target }
    }

    /// Generate GPU kernel code from MLIR module
    pub fn generate_kernel_code(&self, module: &MLIRModule) -> Result<String, MLIRError> {
        match self.target {
            GpuTarget::Cuda => self.generate_cuda_code(module),
            GpuTarget::SpirV => self.generate_spirv_code(module),
            GpuTarget::WebGpu => self.generate_webgpu_code(module),
        }
    }

    /// Generate CUDA kernel code
    fn generate_cuda_code(&self, module: &MLIRModule) -> Result<String, MLIRError> {
        let mut code = String::new();
        
        // Add CUDA headers
        code.push_str("#include <cuda_runtime.h>\n");
        code.push_str("#include <device_launch_parameters.h>\n");
        code.push_str("#include <cooperative_groups.h>\n\n");
        
        // Add helper macros
        code.push_str("// GPU kernel helper macros\n");
        code.push_str("#define CUDA_CHECK(call) \\\n");
        code.push_str("    do { \\\n");
        code.push_str("        cudaError_t err = call; \\\n");
        code.push_str("        if (err != cudaSuccess) { \\\n");
        code.push_str("            fprintf(stderr, \"CUDA error at %s:%d: %s\\n\", __FILE__, __LINE__, cudaGetErrorString(err)); \\\n");
        code.push_str("            exit(1); \\\n");
        code.push_str("        } \\\n");
        code.push_str("    } while(0)\n\n");
        
        // Generate kernels from MLIR operations
        for op in module.operations() {
            if op.name == "gpu.func" && self.is_kernel_operation(op) {
                code.push_str(&self.generate_cuda_kernel_from_op(op)?);
                code.push_str("\n\n");
            }
        }
        
        // Generate host launch functions
        code.push_str(&self.generate_cuda_launch_functions(module)?);
        
        Ok(code)
    }

    /// Generate CUDA kernel from MLIR operation
    fn generate_cuda_kernel_from_op(&self, op: &MLIROperation) -> Result<String, MLIRError> {
        let mut kernel_code = String::new();
        
        // Extract kernel name
        let kernel_name = self.extract_kernel_name(op)?;
        
        // Generate kernel signature based on operands
        kernel_code.push_str(&format!("__global__ void {}(\n", kernel_name));
        
        // Generate parameters from operands
        for (_i, operand) in op.operands.iter().enumerate() {
            let param_type = self.mlir_type_to_cuda_type(&operand.value_type)?;
            kernel_code.push_str(&format!("    {} {},\n", param_type, operand.id));
        }
        
        // Remove trailing comma and add closing parenthesis
        if !op.operands.is_empty() {
            kernel_code.truncate(kernel_code.len() - 2);
            kernel_code.push_str("\n");
        }
        kernel_code.push_str(") {\n");
        
        // Generate kernel body based on operation type
        let op_type = self.get_operation_type(op);
        match op_type.as_str() {
            "elementwise" => kernel_code.push_str(&self.generate_cuda_elementwise_body()?),
            "matmul" => kernel_code.push_str(&self.generate_cuda_matmul_body()?),
            "reduction" => kernel_code.push_str(&self.generate_cuda_reduction_body()?),
            _ => kernel_code.push_str(&self.generate_cuda_generic_body()?),
        }
        
        kernel_code.push_str("}\n");
        
        Ok(kernel_code)
    }

    /// Generate CUDA elementwise kernel body
    fn generate_cuda_elementwise_body(&self) -> Result<String, MLIRError> {
        let mut body = String::new();
        
        body.push_str("    // Calculate global thread index\n");
        body.push_str("    int tid = threadIdx.x;\n");
        body.push_str("    int bid = blockIdx.x;\n");
        body.push_str("    int bsz = blockDim.x;\n");
        body.push_str("    int gid = bid * bsz + tid;\n\n");
        
        body.push_str("    // Bounds check\n");
        body.push_str("    if (gid < size) {\n");
        body.push_str("        // Perform elementwise operation\n");
        body.push_str("        float val = input[gid];\n");
        body.push_str("        output[gid] = val * val; // Example: square operation\n");
        body.push_str("    }\n");
        
        Ok(body)
    }

    /// Generate CUDA matrix multiplication kernel body
    fn generate_cuda_matmul_body(&self) -> Result<String, MLIRError> {
        let mut body = String::new();
        
        body.push_str("    // Tiled matrix multiplication with shared memory\n");
        body.push_str("    __shared__ float As[16][16];\n");
        body.push_str("    __shared__ float Bs[16][16];\n\n");
        
        body.push_str("    int tx = threadIdx.x;\n");
        body.push_str("    int ty = threadIdx.y;\n");
        body.push_str("    int bx = blockIdx.x;\n");
        body.push_str("    int by = blockIdx.y;\n\n");
        
        body.push_str("    int row = by * 16 + ty;\n");
        body.push_str("    int col = bx * 16 + tx;\n\n");
        
        body.push_str("    float sum = 0.0f;\n");
        body.push_str("    \n");
        body.push_str("    // Loop over tiles\n");
        body.push_str("    for (int k = 0; k < size; k += 16) {\n");
        body.push_str("        // Load tile into shared memory\n");
        body.push_str("        if (row < size && (k + tx) < size)\n");
        body.push_str("            As[ty][tx] = input[row * size + k + tx];\n");
        body.push_str("        else\n");
        body.push_str("            As[ty][tx] = 0.0f;\n\n");
        
        body.push_str("        if ((k + ty) < size && col < size)\n");
        body.push_str("            Bs[ty][tx] = input[(k + ty) * size + col];\n");
        body.push_str("        else\n");
        body.push_str("            Bs[ty][tx] = 0.0f;\n\n");
        
        body.push_str("        __syncthreads();\n\n");
        
        body.push_str("        // Compute partial result\n");
        body.push_str("        for (int i = 0; i < 16; i++) {\n");
        body.push_str("            sum += As[ty][i] * Bs[i][tx];\n");
        body.push_str("        }\n\n");
        
        body.push_str("        __syncthreads();\n");
        body.push_str("    }\n\n");
        
        body.push_str("    // Write result\n");
        body.push_str("    if (row < size && col < size) {\n");
        body.push_str("        output[row * size + col] = sum;\n");
        body.push_str("    }\n");
        
        Ok(body)
    }

    /// Generate CUDA reduction kernel body
    fn generate_cuda_reduction_body(&self) -> Result<String, MLIRError> {
        let mut body = String::new();
        
        body.push_str("    // Block-level reduction with shared memory\n");
        body.push_str("    __shared__ float sdata[256];\n\n");
        
        body.push_str("    int tid = threadIdx.x;\n");
        body.push_str("    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n\n");
        
        body.push_str("    // Load data into shared memory\n");
        body.push_str("    sdata[tid] = (gid < size) ? input[gid] : 0.0f;\n");
        body.push_str("    __syncthreads();\n\n");
        
        body.push_str("    // Tree reduction in shared memory\n");
        body.push_str("    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n");
        body.push_str("        if (tid < s) {\n");
        body.push_str("            sdata[tid] += sdata[tid + s];\n");
        body.push_str("        }\n");
        body.push_str("        __syncthreads();\n");
        body.push_str("    }\n\n");
        
        body.push_str("    // Write block result\n");
        body.push_str("    if (tid == 0) {\n");
        body.push_str("        atomicAdd(output, sdata[0]);\n");
        body.push_str("    }\n");
        
        Ok(body)
    }

    /// Generate CUDA generic kernel body
    fn generate_cuda_generic_body(&self) -> Result<String, MLIRError> {
        let mut body = String::new();
        
        body.push_str("    // Generic parallel kernel\n");
        body.push_str("    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n");
        body.push_str("    \n");
        body.push_str("    if (gid < size) {\n");
        body.push_str("        // Generic operation placeholder\n");
        body.push_str("        output[gid] = input[gid];\n");
        body.push_str("    }\n");
        
        Ok(body)
    }

    /// Check if operation is a kernel
    fn is_kernel_operation(&self, op: &MLIROperation) -> bool {
        if let Some(MLIRAttribute::Boolean(is_kernel)) = op.attributes.get("kernel") {
            *is_kernel
        } else {
            false
        }
    }

    /// Get operation type from attributes
    fn get_operation_type(&self, op: &MLIROperation) -> String {
        if let Some(MLIRAttribute::String(op_type)) = op.attributes.get("operation_type") {
            op_type.clone()
        } else {
            "generic".to_string()
        }
    }

    /// Convert MLIR type to CUDA type string
    fn mlir_type_to_cuda_type(&self, mlir_type: &MLIRType) -> Result<String, MLIRError> {
        match mlir_type {
            MLIRType::Float { width: 32 } => Ok("float".to_string()),
            MLIRType::Float { width: 64 } => Ok("double".to_string()),
            MLIRType::Integer { width: 32, signed: true } => Ok("int".to_string()),
            MLIRType::Integer { width: 32, signed: false } => Ok("unsigned int".to_string()),
            MLIRType::Index => Ok("size_t".to_string()),
            MLIRType::Memref { element_type, .. } => {
                let elem_type = self.mlir_type_to_cuda_type(element_type)?;
                Ok(format!("{}*", elem_type))
            }
            MLIRType::Tensor { element_type, .. } => {
                let elem_type = self.mlir_type_to_cuda_type(element_type)?;
                Ok(format!("{}*", elem_type))
            }
            _ => Ok("void*".to_string()),
        }
    }

    /// Generate CUDA host launch functions
    fn generate_cuda_launch_functions(&self, module: &MLIRModule) -> Result<String, MLIRError> {
        let mut code = String::new();
        
        code.push_str("// Host launch functions\n");
        code.push_str("extern \"C\" {\n\n");
        
        // Generate launch function for each kernel
        for op in module.operations() {
            if op.name == "gpu.func" && self.is_kernel_operation(op) {
                let kernel_name = self.extract_kernel_name(op)?;
                code.push_str(&self.generate_cuda_launch_function(&kernel_name)?);
                code.push_str("\n");
            }
        }
        
        code.push_str("}\n");
        
        Ok(code)
    }

    /// Generate CUDA launch function for specific kernel
    fn generate_cuda_launch_function(&self, kernel_name: &str) -> Result<String, MLIRError> {
        let mut func = String::new();
        
        func.push_str(&format!("void launch_{}(\n", kernel_name));
        func.push_str("    float* d_input,\n");
        func.push_str("    float* d_output,\n");
        func.push_str("    int size,\n");
        func.push_str("    int block_size = 256\n");
        func.push_str(") {\n");
        func.push_str("    // Calculate grid dimensions\n");
        func.push_str("    int grid_size = (size + block_size - 1) / block_size;\n\n");
        
        func.push_str("    // Launch kernel\n");
        func.push_str(&format!("    {}<<<grid_size, block_size>>>(\n", kernel_name));
        func.push_str("        d_input,\n");
        func.push_str("        d_output,\n");
        func.push_str("        size\n");
        func.push_str("    );\n\n");
        
        func.push_str("    // Check for launch errors\n");
        func.push_str("    CUDA_CHECK(cudaGetLastError());\n");
        func.push_str("    CUDA_CHECK(cudaDeviceSynchronize());\n");
        func.push_str("}\n");
        
        Ok(func)
    }

    /// Generate SPIR-V code
    fn generate_spirv_code(&self, module: &MLIRModule) -> Result<String, MLIRError> {
        let mut code = String::new();
        
        // SPIR-V assembly header
        code.push_str("; SPIR-V\n");
        code.push_str("; Version: 1.0\n");
        code.push_str("; Generator: Aether Compiler\n");
        code.push_str("; Bound: 1000\n");
        code.push_str("; Schema: 0\n\n");
        
        // OpCapability declarations
        code.push_str("OpCapability Shader\n");
        code.push_str("OpCapability Float64\n");
        code.push_str("OpCapability Int64\n\n");
        
        // OpExtension declarations
        code.push_str("OpExtension \"SPV_KHR_storage_buffer_storage_class\"\n\n");
        
        // OpMemoryModel
        code.push_str("OpMemoryModel Logical GLSL450\n\n");
        
        // Generate compute shader entry points
        for op in module.operations() {
            if op.name.contains("spirv.func") && self.is_kernel_operation(op) {
                let kernel_name = self.extract_kernel_name(op)?;
                code.push_str(&format!("OpEntryPoint GLCompute %{} \"main\"\n", kernel_name));
            }
        }
        code.push_str("\n");
        
        // Generate SPIR-V kernel implementations
        for op in module.operations() {
            if op.name.contains("spirv.func") && self.is_kernel_operation(op) {
                code.push_str(&self.generate_spirv_kernel_from_op(op)?);
                code.push_str("\n");
            }
        }
        
        Ok(code)
    }

    /// Generate SPIR-V kernel from MLIR operation
    fn generate_spirv_kernel_from_op(&self, op: &MLIROperation) -> Result<String, MLIRError> {
        let mut kernel_code = String::new();
        let kernel_name = self.extract_kernel_name(op)?;
        
        // Type declarations
        kernel_code.push_str("// Type declarations\n");
        kernel_code.push_str("%void = OpTypeVoid\n");
        kernel_code.push_str("%func_type = OpTypeFunction %void\n");
        kernel_code.push_str("%float = OpTypeFloat 32\n");
        kernel_code.push_str("%int = OpTypeInt 32 1\n");
        kernel_code.push_str("%uint = OpTypeInt 32 0\n");
        kernel_code.push_str("%v3uint = OpTypeVector %uint 3\n\n");
        
        // Function definition
        kernel_code.push_str(&format!("%{} = OpFunction %void None %func_type\n", kernel_name));
        kernel_code.push_str("%entry = OpLabel\n\n");
        
        // Get global invocation ID
        kernel_code.push_str("// Get global invocation ID\n");
        kernel_code.push_str("%gid_ptr = OpVariable %_ptr_Input_v3uint Input\n");
        kernel_code.push_str("%gid = OpLoad %v3uint %gid_ptr\n");
        kernel_code.push_str("%gid_x = OpCompositeExtract %uint %gid 0\n\n");
        
        // Kernel body based on operation type
        let op_type = self.get_operation_type(op);
        if op_type.contains("elementwise") {
            kernel_code.push_str(&self.generate_spirv_elementwise_body()?);
        } else {
            kernel_code.push_str(&self.generate_spirv_generic_body()?);
        }
        
        kernel_code.push_str("OpReturn\n");
        kernel_code.push_str("OpFunctionEnd\n");
        
        Ok(kernel_code)
    }

    /// Generate SPIR-V elementwise kernel body
    fn generate_spirv_elementwise_body(&self) -> Result<String, MLIRError> {
        let mut body = String::new();
        
        body.push_str("// Elementwise operation\n");
        body.push_str("// Load input value\n");
        body.push_str("%input_ptr = OpAccessChain %_ptr_StorageBuffer_float %input_buffer %uint_0 %gid_x\n");
        body.push_str("%input_val = OpLoad %float %input_ptr\n\n");
        
        body.push_str("// Perform operation (square)\n");
        body.push_str("%result = OpFMul %float %input_val %input_val\n\n");
        
        body.push_str("// Store result\n");
        body.push_str("%output_ptr = OpAccessChain %_ptr_StorageBuffer_float %output_buffer %uint_0 %gid_x\n");
        body.push_str("OpStore %output_ptr %result\n\n");
        
        Ok(body)
    }

    /// Generate SPIR-V generic kernel body
    fn generate_spirv_generic_body(&self) -> Result<String, MLIRError> {
        let mut body = String::new();
        
        body.push_str("// Generic operation\n");
        body.push_str("// Copy input to output\n");
        body.push_str("%input_ptr = OpAccessChain %_ptr_StorageBuffer_float %input_buffer %uint_0 %gid_x\n");
        body.push_str("%input_val = OpLoad %float %input_ptr\n");
        body.push_str("%output_ptr = OpAccessChain %_ptr_StorageBuffer_float %output_buffer %uint_0 %gid_x\n");
        body.push_str("OpStore %output_ptr %input_val\n\n");
        
        Ok(body)
    }

    /// Generate WebGPU compute shader code
    fn generate_webgpu_code(&self, module: &MLIRModule) -> Result<String, MLIRError> {
        let mut code = String::new();
        
        // WebGPU compute shader header
        code.push_str("// WebGPU Compute Shader (WGSL)\n\n");
        
        // Generate compute shaders from MLIR operations
        for op in module.operations() {
            if op.name.contains("webgpu.compute_shader") {
                code.push_str(&self.generate_webgpu_shader_from_op(op)?);
                code.push_str("\n\n");
            }
        }
        
        Ok(code)
    }

    /// Generate WebGPU compute shader from MLIR operation
    fn generate_webgpu_shader_from_op(&self, op: &MLIROperation) -> Result<String, MLIRError> {
        let mut shader_code = String::new();
        let kernel_name = self.extract_kernel_name(op)?;
        
        // Binding declarations
        shader_code.push_str("// Buffer bindings\n");
        shader_code.push_str("@group(0) @binding(0) var<storage, read> input_buffer: array<f32>;\n");
        shader_code.push_str("@group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;\n");
        shader_code.push_str("@group(0) @binding(2) var<uniform> params: Params;\n\n");
        
        shader_code.push_str("struct Params {\n");
        shader_code.push_str("    size: u32,\n");
        shader_code.push_str("};\n\n");
        
        // Compute shader function
        shader_code.push_str(&format!("@compute @workgroup_size(256)\n"));
        shader_code.push_str(&format!("fn {}(@builtin(global_invocation_id) global_id: vec3<u32>) {{\n", kernel_name));
        
        // Shader body
        shader_code.push_str("    let gid = global_id.x;\n");
        shader_code.push_str("    \n");
        shader_code.push_str("    if (gid >= params.size) {\n");
        shader_code.push_str("        return;\n");
        shader_code.push_str("    }\n\n");
        
        let op_type = self.get_operation_type(op);
        if op_type.contains("elementwise") {
            shader_code.push_str("    // Elementwise operation\n");
            shader_code.push_str("    let input_val = input_buffer[gid];\n");
            shader_code.push_str("    output_buffer[gid] = input_val * input_val;\n");
        } else {
            shader_code.push_str("    // Generic operation\n");
            shader_code.push_str("    output_buffer[gid] = input_buffer[gid];\n");
        }
        
        shader_code.push_str("}\n");
        
        Ok(shader_code)
    }

    /// Extract kernel name from MLIR operation
    fn extract_kernel_name(&self, op: &MLIROperation) -> Result<String, MLIRError> {
        if let Some(MLIRAttribute::String(name)) = op.attributes.get("sym_name") {
            Ok(name.clone())
        } else {
            Ok("kernel_0".to_string()) // Default name
        }
    }
}

/// GPU memory management utilities
pub struct GpuMemoryManager {
    target: GpuTarget,
}

impl GpuMemoryManager {
    pub fn new(target: GpuTarget) -> Self {
        GpuMemoryManager { target }
    }

    /// Generate memory allocation code
    pub fn generate_alloc_code(&self, size: &str, _config: &GpuMemoryConfig) -> String {
        match self.target {
            GpuTarget::Cuda => {
                format!("CUDA_CHECK(cudaMalloc(&ptr, {}));", size)
            }
            GpuTarget::SpirV => {
                format!("// SPIR-V buffer allocation: size = {}", size)
            }
            GpuTarget::WebGpu => {
                format!("// WebGPU buffer creation: size = {}", size)
            }
        }
    }

    /// Generate memory deallocation code
    pub fn generate_dealloc_code(&self, ptr: &str) -> String {
        match self.target {
            GpuTarget::Cuda => {
                format!("CUDA_CHECK(cudaFree({}));", ptr)
            }
            GpuTarget::SpirV => {
                format!("// SPIR-V buffer cleanup: {}", ptr)
            }
            GpuTarget::WebGpu => {
                format!("// WebGPU buffer destroy: {}", ptr)
            }
        }
    }

    /// Generate memory copy code
    pub fn generate_memcpy_code(&self, src: &str, dst: &str, size: &str, direction: GpuMemcpyDirection) -> String {
        match self.target {
            GpuTarget::Cuda => {
                let cuda_kind = match direction {
                    GpuMemcpyDirection::HostToDevice => "cudaMemcpyHostToDevice",
                    GpuMemcpyDirection::DeviceToHost => "cudaMemcpyDeviceToHost",
                    GpuMemcpyDirection::DeviceToDevice => "cudaMemcpyDeviceToDevice",
                    GpuMemcpyDirection::HostToHost => "cudaMemcpyHostToHost",
                };
                format!("CUDA_CHECK(cudaMemcpy({}, {}, {}, {}));", dst, src, size, cuda_kind)
            }
            GpuTarget::SpirV => {
                format!("// SPIR-V buffer copy: {} -> {}, size = {}", src, dst, size)
            }
            GpuTarget::WebGpu => {
                format!("// WebGPU buffer copy: {} -> {}, size = {}", src, dst, size)
            }
        }
    }
}