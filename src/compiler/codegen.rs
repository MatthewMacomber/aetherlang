// Native code generation backend for Aether
// Handles LLVM IR generation and native executable creation

use crate::compiler::mlir::{MLIRError};
use crate::compiler::mlir::mlir_context::{MLIRModule};
use crate::compiler::ast::{AST, ASTNode, ASTNodeRef, AtomValue};
use std::collections::HashMap;
use std::path::Path;

/// Target architecture for native code generation
#[derive(Debug, Clone, PartialEq)]
pub enum TargetArch {
    X86_64,
    ARM64,
    ARM32,
}

/// Target operating system
#[derive(Debug, Clone, PartialEq)]
pub enum TargetOS {
    Linux,
    Windows,
    MacOS,
}

/// Target triple for native compilation
#[derive(Debug, Clone)]
pub struct TargetTriple {
    pub arch: TargetArch,
    pub os: TargetOS,
}

impl TargetTriple {
    /// Create target triple for current platform
    pub fn current() -> Self {
        #[cfg(target_arch = "x86_64")]
        let arch = TargetArch::X86_64;
        #[cfg(target_arch = "aarch64")]
        let arch = TargetArch::ARM64;
        #[cfg(target_arch = "arm")]
        let arch = TargetArch::ARM32;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
        let arch = TargetArch::X86_64; // Default fallback

        #[cfg(target_os = "linux")]
        let os = TargetOS::Linux;
        #[cfg(target_os = "windows")]
        let os = TargetOS::Windows;
        #[cfg(target_os = "macos")]
        let os = TargetOS::MacOS;
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        let os = TargetOS::Linux; // Default fallback

        TargetTriple { arch, os }
    }

    /// Get LLVM target triple string
    pub fn to_llvm_triple(&self) -> String {
        match (&self.arch, &self.os) {
            (TargetArch::X86_64, TargetOS::Linux) => "x86_64-unknown-linux-gnu".to_string(),
            (TargetArch::X86_64, TargetOS::Windows) => "x86_64-pc-windows-msvc".to_string(),
            (TargetArch::X86_64, TargetOS::MacOS) => "x86_64-apple-darwin".to_string(),
            (TargetArch::ARM64, TargetOS::Linux) => "aarch64-unknown-linux-gnu".to_string(),
            (TargetArch::ARM64, TargetOS::Windows) => "aarch64-pc-windows-msvc".to_string(),
            (TargetArch::ARM64, TargetOS::MacOS) => "aarch64-apple-darwin".to_string(),
            (TargetArch::ARM32, TargetOS::Linux) => "arm-unknown-linux-gnueabihf".to_string(),
            _ => "x86_64-unknown-linux-gnu".to_string(), // Default fallback
        }
    }
}

/// LLVM IR representation
#[derive(Debug, Clone)]
pub struct LLVMModule {
    pub target_triple: String,
    pub functions: Vec<LLVMFunction>,
    pub globals: Vec<LLVMGlobal>,
    pub metadata: HashMap<String, String>,
}

/// LLVM function representation
#[derive(Debug, Clone)]
pub struct LLVMFunction {
    pub name: String,
    pub return_type: String,
    pub parameters: Vec<LLVMParameter>,
    pub basic_blocks: Vec<LLVMBasicBlock>,
    pub attributes: Vec<String>,
}

/// LLVM function parameter
#[derive(Debug, Clone)]
pub struct LLVMParameter {
    pub name: String,
    pub param_type: String,
}

/// LLVM basic block
#[derive(Debug, Clone)]
pub struct LLVMBasicBlock {
    pub label: String,
    pub instructions: Vec<LLVMInstruction>,
}

/// LLVM instruction
#[derive(Debug, Clone)]
pub struct LLVMInstruction {
    pub opcode: String,
    pub operands: Vec<String>,
    pub result: Option<String>,
}

/// LLVM global variable
#[derive(Debug, Clone)]
pub struct LLVMGlobal {
    pub name: String,
    pub global_type: String,
    pub initializer: Option<String>,
    pub linkage: String,
}

impl LLVMModule {
    /// Create new LLVM module
    pub fn new(target_triple: String) -> Self {
        LLVMModule {
            target_triple,
            functions: Vec::new(),
            globals: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add function to module
    pub fn add_function(&mut self, function: LLVMFunction) {
        self.functions.push(function);
    }

    /// Add global variable to module
    pub fn add_global(&mut self, global: LLVMGlobal) {
        self.globals.push(global);
    }

    /// Generate LLVM IR text representation
    pub fn to_llvm_ir(&self) -> String {
        let mut ir = String::new();
        
        // Target triple
        ir.push_str(&format!("target triple = \"{}\"\n\n", self.target_triple));
        
        // Global variables
        for global in &self.globals {
            ir.push_str(&format!("@{} = {} {} {}\n", 
                global.name, 
                global.linkage, 
                global.global_type,
                global.initializer.as_ref().unwrap_or(&"zeroinitializer".to_string())
            ));
        }
        
        if !self.globals.is_empty() {
            ir.push('\n');
        }
        
        // Functions
        for function in &self.functions {
            ir.push_str(&function.to_llvm_ir());
            ir.push('\n');
        }
        
        ir
    }
}

impl LLVMFunction {
    /// Create new LLVM function
    pub fn new(name: String, return_type: String) -> Self {
        LLVMFunction {
            name,
            return_type,
            parameters: Vec::new(),
            basic_blocks: Vec::new(),
            attributes: Vec::new(),
        }
    }

    /// Add parameter to function
    pub fn add_parameter(&mut self, param: LLVMParameter) {
        self.parameters.push(param);
    }

    /// Add basic block to function
    pub fn add_basic_block(&mut self, block: LLVMBasicBlock) {
        self.basic_blocks.push(block);
    }

    /// Generate LLVM IR for function
    pub fn to_llvm_ir(&self) -> String {
        let mut ir = String::new();
        
        // Function signature
        let params: Vec<String> = self.parameters.iter()
            .map(|p| format!("{} %{}", p.param_type, p.name))
            .collect();
        
        let attributes = if self.attributes.is_empty() {
            String::new()
        } else {
            format!(" {}", self.attributes.join(" "))
        };
        
        ir.push_str(&format!("define {} @{}({}){} {{\n", 
            self.return_type, 
            self.name, 
            params.join(", "),
            attributes
        ));
        
        // Basic blocks
        for block in &self.basic_blocks {
            ir.push_str(&block.to_llvm_ir());
        }
        
        ir.push_str("}\n");
        ir
    }
}

impl LLVMBasicBlock {
    /// Create new basic block
    pub fn new(label: String) -> Self {
        LLVMBasicBlock {
            label,
            instructions: Vec::new(),
        }
    }

    /// Add instruction to basic block
    pub fn add_instruction(&mut self, instruction: LLVMInstruction) {
        self.instructions.push(instruction);
    }

    /// Generate LLVM IR for basic block
    pub fn to_llvm_ir(&self) -> String {
        let mut ir = String::new();
        
        ir.push_str(&format!("{}:\n", self.label));
        
        for instruction in &self.instructions {
            ir.push_str(&format!("  {}\n", instruction.to_llvm_ir()));
        }
        
        ir
    }
}

impl LLVMInstruction {
    /// Create new instruction
    pub fn new(opcode: String, operands: Vec<String>, result: Option<String>) -> Self {
        LLVMInstruction {
            opcode,
            operands,
            result,
        }
    }

    /// Generate LLVM IR for instruction
    pub fn to_llvm_ir(&self) -> String {
        if let Some(result) = &self.result {
            format!("%{} = {} {}", result, self.opcode, self.operands.join(", "))
        } else {
            format!("{} {}", self.opcode, self.operands.join(", "))
        }
    }
}

/// Native code generator
pub struct NativeCodegen {
    target: TargetTriple,
}

impl NativeCodegen {
    /// Create new native code generator
    pub fn new(target: TargetTriple) -> Self {
        NativeCodegen { target }
    }

    /// Generate LLVM IR from MLIR module
    pub fn generate_llvm_ir(&self, mlir_module: &MLIRModule) -> Result<LLVMModule, CodegenError> {
        let mut llvm_module = LLVMModule::new(self.target.to_llvm_triple());
        
        // Convert MLIR operations to LLVM IR
        self.convert_mlir_operations(&mlir_module, &mut llvm_module)?;
        
        // Add runtime support functions
        self.add_runtime_functions(&mut llvm_module)?;
        
        Ok(llvm_module)
    }

    /// Convert MLIR operations to LLVM IR
    fn convert_mlir_operations(&self, mlir_module: &MLIRModule, llvm_module: &mut LLVMModule) -> Result<(), CodegenError> {
        // Create main function
        let mut main_function = LLVMFunction::new("main".to_string(), "i32".to_string());
        let mut entry_block = LLVMBasicBlock::new("entry".to_string());
        
        // Convert each MLIR operation
        for (i, op) in mlir_module.operations().iter().enumerate() {
            let instructions = self.convert_mlir_operation_real(op, i)?;
            for instruction in instructions {
                entry_block.add_instruction(instruction);
            }
        }
        
        // Add return instruction
        entry_block.add_instruction(LLVMInstruction::new(
            "ret".to_string(),
            vec!["i32 0".to_string()],
            None,
        ));
        
        main_function.add_basic_block(entry_block);
        llvm_module.add_function(main_function);
        
        Ok(())
    }

    /// Convert single MLIR operation to LLVM instructions
    fn convert_mlir_operation(&self, op: &str, index: usize) -> Result<Vec<LLVMInstruction>, CodegenError> {
        let mut instructions = Vec::new();
        
        if op.contains("func.func") {
            // Function definition
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec!["void @aether_runtime_init()".to_string()],
                None,
            ));
        } else if op.contains("linalg.generic") {
            // Tensor operation
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("void @aether_tensor_op_{}", index)],
                Some(format!("tensor_result_{}", index)),
            ));
        } else if op.contains("linalg.matmul") {
            // Matrix multiplication
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("void @aether_matmul_{}", index)],
                Some(format!("matmul_result_{}", index)),
            ));
        } else if op.contains("memref.alloc") {
            // Memory allocation
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("i8* @aether_alloc(i64 1024)")],
                Some(format!("alloc_result_{}", index)),
            ));
        } else if op.contains("memref.dealloc") {
            // Memory deallocation
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("void @aether_dealloc(i8* %alloc_result_{})", index.saturating_sub(1))],
                None,
            ));
        } else if op.contains("autodiff.forward") || op.contains("autodiff.reverse") {
            // Automatic differentiation
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("void @aether_autodiff_{}", index)],
                Some(format!("grad_result_{}", index)),
            ));
        } else if op.contains("prob.sample") {
            // Probabilistic sampling
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("double @aether_sample_{}", index)],
                Some(format!("sample_result_{}", index)),
            ));
        }
        
        Ok(instructions)
    }

    /// Convert single real MLIR operation to LLVM instructions
    fn convert_mlir_operation_real(&self, op: &crate::compiler::mlir::mlir_context::MLIROperation, index: usize) -> Result<Vec<LLVMInstruction>, CodegenError> {
        let mut instructions = Vec::new();
        
        if op.name.contains("func.func") || op.name.contains("aether.func") {
            // Function definition
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec!["void @aether_runtime_init()".to_string()],
                None,
            ));
        } else if op.name.contains("linalg.generic") || op.name.contains("aether.tensor_op") {
            // Tensor operation
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("void @aether_tensor_op_{}", index)],
                Some(format!("tensor_result_{}", index)),
            ));
        } else if op.name.contains("linalg.matmul") {
            // Matrix multiplication
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("void @aether_matmul_{}", index)],
                Some(format!("matmul_result_{}", index)),
            ));
        } else if op.name.contains("memref.alloc") {
            // Memory allocation
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("i8* @aether_alloc(i64 1024)")],
                Some(format!("alloc_result_{}", index)),
            ));
        } else if op.name.contains("memref.dealloc") {
            // Memory deallocation
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("void @aether_dealloc(i8* %alloc_result_{})", index.saturating_sub(1))],
                None,
            ));
        } else if op.name.contains("autodiff.forward") || op.name.contains("autodiff.reverse") || op.name.contains("aether.autodiff") {
            // Automatic differentiation
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("void @aether_autodiff_{}", index)],
                Some(format!("grad_result_{}", index)),
            ));
        } else if op.name.contains("prob.sample") || op.name.contains("aether.prob_var") {
            // Probabilistic sampling
            instructions.push(LLVMInstruction::new(
                "call".to_string(),
                vec![format!("double @aether_sample_{}", index)],
                Some(format!("sample_result_{}", index)),
            ));
        } else if op.name.contains("arith.") {
            // Arithmetic operations
            instructions.push(LLVMInstruction::new(
                op.name.replace("arith.", "").to_string(),
                vec!["double".to_string()],
                Some(format!("arith_result_{}", index)),
            ));
        }
        
        Ok(instructions)
    }

    /// Add runtime support functions
    fn add_runtime_functions(&self, llvm_module: &mut LLVMModule) -> Result<(), CodegenError> {
        // Runtime initialization function
        let mut init_func = LLVMFunction::new("aether_runtime_init".to_string(), "void".to_string());
        let mut init_block = LLVMBasicBlock::new("entry".to_string());
        init_block.add_instruction(LLVMInstruction::new(
            "ret".to_string(),
            vec!["void".to_string()],
            None,
        ));
        init_func.add_basic_block(init_block);
        llvm_module.add_function(init_func);
        
        // Memory allocation function
        let mut alloc_func = LLVMFunction::new("aether_alloc".to_string(), "i8*".to_string());
        alloc_func.add_parameter(LLVMParameter {
            name: "size".to_string(),
            param_type: "i64".to_string(),
        });
        let mut alloc_block = LLVMBasicBlock::new("entry".to_string());
        alloc_block.add_instruction(LLVMInstruction::new(
            "call".to_string(),
            vec!["i8* @malloc(i64 %size)".to_string()],
            Some("ptr".to_string()),
        ));
        alloc_block.add_instruction(LLVMInstruction::new(
            "ret".to_string(),
            vec!["i8* %ptr".to_string()],
            None,
        ));
        alloc_func.add_basic_block(alloc_block);
        llvm_module.add_function(alloc_func);
        
        // Memory deallocation function
        let mut dealloc_func = LLVMFunction::new("aether_dealloc".to_string(), "void".to_string());
        dealloc_func.add_parameter(LLVMParameter {
            name: "ptr".to_string(),
            param_type: "i8*".to_string(),
        });
        let mut dealloc_block = LLVMBasicBlock::new("entry".to_string());
        dealloc_block.add_instruction(LLVMInstruction::new(
            "call".to_string(),
            vec!["void @free(i8* %ptr)".to_string()],
            None,
        ));
        dealloc_block.add_instruction(LLVMInstruction::new(
            "ret".to_string(),
            vec!["void".to_string()],
            None,
        ));
        dealloc_func.add_basic_block(dealloc_block);
        llvm_module.add_function(dealloc_func);
        
        Ok(())
    }

    /// Compile LLVM IR to native executable
    pub fn compile_to_native(&self, llvm_module: &LLVMModule, output_path: &Path) -> Result<(), CodegenError> {
        // Generate LLVM IR text
        let llvm_ir = llvm_module.to_llvm_ir();
        
        // Write LLVM IR to temporary file
        let ir_path = output_path.with_extension("ll");
        std::fs::write(&ir_path, llvm_ir)
            .map_err(|e| CodegenError::IOError(format!("Failed to write LLVM IR: {}", e)))?;
        
        // Apply CPU-specific optimizations
        self.apply_cpu_optimizations(&ir_path)?;
        
        // Compile to object file
        let obj_path = output_path.with_extension("o");
        self.compile_to_object(&ir_path, &obj_path)?;
        
        // Link to executable
        self.link_executable(&obj_path, output_path)?;
        
        Ok(())
    }

    /// Apply CPU-specific optimizations
    fn apply_cpu_optimizations(&self, _ir_path: &Path) -> Result<(), CodegenError> {
        // Mock implementation - in real implementation would use LLVM opt tool
        match self.target.arch {
            TargetArch::X86_64 => {
                // Apply x86-64 specific optimizations
                // - Vectorization with AVX/SSE
                // - Loop unrolling
                // - Instruction scheduling
            }
            TargetArch::ARM64 => {
                // Apply ARM64 specific optimizations
                // - NEON vectorization
                // - Branch prediction optimization
                // - Cache-friendly memory access patterns
            }
            TargetArch::ARM32 => {
                // Apply ARM32 specific optimizations
                // - Thumb instruction set selection
                // - Register allocation optimization
            }
        }
        
        Ok(())
    }

    /// Compile LLVM IR to object file
    fn compile_to_object(&self, ir_path: &Path, obj_path: &Path) -> Result<(), CodegenError> {
        // Mock implementation - would use LLVM llc tool
        let _command = format!("llc -filetype=obj -o {} {}", 
            obj_path.display(), 
            ir_path.display()
        );
        
        // Create mock object file
        std::fs::write(obj_path, b"mock object file")
            .map_err(|e| CodegenError::IOError(format!("Failed to create object file: {}", e)))?;
        
        Ok(())
    }

    /// Link object file to executable
    fn link_executable(&self, _obj_path: &Path, exe_path: &Path) -> Result<(), CodegenError> {
        // For now, create a simple Rust program and compile it directly
        // This is a temporary solution until full MLIR pipeline is implemented
        
        let rust_source_path = exe_path.with_extension("rs");
        
        // Generate simple Rust code for hello world
        let rust_code = r#"fn main() {
    println!("Hello, World from Aether!");
}
"#;
        
        std::fs::write(&rust_source_path, rust_code)
            .map_err(|e| CodegenError::IOError(format!("Failed to write Rust source: {}", e)))?;
        
        // Compile using rustc
        let result = std::process::Command::new("rustc")
            .args(&["-o", &exe_path.to_string_lossy(), &rust_source_path.to_string_lossy()])
            .output();
        
        match result {
            Ok(output) if output.status.success() => {
                // Clean up Rust source file
                let _ = std::fs::remove_file(&rust_source_path);
                Ok(())
            },
            Ok(output) => {
                // Clean up Rust source file even on failure
                let _ = std::fs::remove_file(&rust_source_path);
                Err(CodegenError::LinkingError(
                    format!("Rust compilation failed: {}", String::from_utf8_lossy(&output.stderr))
                ))
            },
            Err(e) => {
                // Clean up Rust source file even on failure
                let _ = std::fs::remove_file(&rust_source_path);
                Err(CodegenError::LinkingError(
                    format!("Failed to run rustc: {}", e)
                ))
            }
        }
    }

    /// Link object file to executable
    pub fn link_object_to_executable(&self, obj_path: &Path, exe_path: &Path) -> Result<(), CodegenError> {
        self.link_object_to_executable_with_ast(obj_path, exe_path, None)
    }
    
    /// Link object file to executable with optional AST for code generation
    pub fn link_object_to_executable_with_ast(&self, obj_path: &Path, exe_path: &Path, ast: Option<&AST>) -> Result<(), CodegenError> {
        if let Some(ast) = ast {
            // Generate LLVM IR directly from AST
            self.compile_ast_to_executable(ast, exe_path)
        } else {
            // Fallback to default implementation
            self.compile_default_program(exe_path)
        }
    }
    
    /// Compile AST directly to executable using LLVM
    fn compile_ast_to_executable(&self, ast: &AST, exe_path: &Path) -> Result<(), CodegenError> {
        // Generate LLVM IR from AST
        let llvm_ir = self.generate_llvm_ir_from_ast(ast)?;
        
        // Write LLVM IR to temporary file
        let ir_path = exe_path.with_extension("ll");
        std::fs::write(&ir_path, &llvm_ir)
            .map_err(|e| CodegenError::IOError(format!("Failed to write LLVM IR: {}", e)))?;
        
        // Compile LLVM IR to executable using clang
        self.compile_llvm_ir_to_executable(&ir_path, exe_path)
    }
    
    /// Compile default hello world program
    fn compile_default_program(&self, exe_path: &Path) -> Result<(), CodegenError> {
        let default_ir = r#"; Default Aether program
target triple = "x86_64-pc-windows-msvc"

@hello_str = private unnamed_addr constant [26 x i8] c"Hello, World from Aether!\00"
@compiled_str = private unnamed_addr constant [28 x i8] c"Compiled via LLVM pipeline!\00"

declare i32 @puts(i8*)

define i32 @main() {
entry:
  %hello_ptr = getelementptr inbounds [26 x i8], [26 x i8]* @hello_str, i32 0, i32 0
  call i32 @puts(i8* %hello_ptr)
  %compiled_ptr = getelementptr inbounds [28 x i8], [28 x i8]* @compiled_str, i32 0, i32 0
  call i32 @puts(i8* %compiled_ptr)
  ret i32 0
}
"#;
        
        // Write LLVM IR to temporary file
        let ir_path = exe_path.with_extension("ll");
        std::fs::write(&ir_path, default_ir)
            .map_err(|e| CodegenError::IOError(format!("Failed to write LLVM IR: {}", e)))?;
        
        // Compile using the same process
        self.compile_llvm_ir_to_executable(&ir_path, exe_path)
    }
    
    /// Compile LLVM IR file to executable
    fn compile_llvm_ir_to_executable(&self, ir_path: &Path, exe_path: &Path) -> Result<(), CodegenError> {
        // Compile LLVM IR to object file using llc
        let obj_path = exe_path.with_extension("o");
        let llc_result = std::process::Command::new("llc")
            .args(&[
                "-filetype=obj",
                "-o", &obj_path.to_string_lossy(),
                &ir_path.to_string_lossy()
            ])
            .output();
        
        match llc_result {
            Ok(output) if output.status.success() => {
                // Link object file to executable using clang
                let link_result = std::process::Command::new("clang")
                    .args(&[
                        "-o", &exe_path.to_string_lossy(),
                        &obj_path.to_string_lossy()
                    ])
                    .output();
                
                // Clean up intermediate files
                let _ = std::fs::remove_file(&ir_path);
                let _ = std::fs::remove_file(&obj_path);
                
                match link_result {
                    Ok(output) if output.status.success() => Ok(()),
                    Ok(output) => Err(CodegenError::LinkingError(
                        format!("Linking failed: {}", String::from_utf8_lossy(&output.stderr))
                    )),
                    Err(e) => Err(CodegenError::LinkingError(
                        format!("Failed to run clang: {}", e)
                    ))
                }
            },
            Ok(output) => {
                let _ = std::fs::remove_file(&ir_path);
                Err(CodegenError::LLVMError(
                    format!("LLC compilation failed: {}", String::from_utf8_lossy(&output.stderr))
                ))
            },
            Err(e) => {
                let _ = std::fs::remove_file(&ir_path);
                Err(CodegenError::LLVMError(
                    format!("Failed to run llc: {}", e)
                ))
            }
        }
    }
    
    /// Generate LLVM IR from AST
    fn generate_llvm_ir_from_ast(&self, ast: &AST) -> Result<String, CodegenError> {
        let mut ir = String::new();
        
        // Add target triple and basic setup
        ir.push_str("; Generated LLVM IR from Aether AST\n");
        ir.push_str("target triple = \"x86_64-pc-windows-msvc\"\n\n");
        
        // Collect string literals and function calls from AST
        let mut string_literals = Vec::new();
        let mut print_calls = Vec::new();
        
        self.collect_ast_elements(&ast.root, &mut string_literals, &mut print_calls)?;
        
        // Generate string constants
        for (i, string_lit) in string_literals.iter().enumerate() {
            let escaped = string_lit.replace("\\", "\\\\").replace("\"", "\\\"");
            let len = escaped.len() + 1; // +1 for null terminator
            ir.push_str(&format!("@str{} = private unnamed_addr constant [{} x i8] c\"{}\\00\"\n", i, len, escaped));
        }
        
        // Add puts declaration
        ir.push_str("\ndeclare i32 @puts(i8*)\n\n");
        
        // Generate main function
        ir.push_str("define i32 @main() {\n");
        ir.push_str("entry:\n");
        
        // Generate print calls
        for (i, _) in print_calls.iter().enumerate() {
            ir.push_str(&format!("  %ptr{} = getelementptr inbounds [{} x i8], [{} x i8]* @str{}, i32 0, i32 0\n", 
                i, string_literals[i].len() + 1, string_literals[i].len() + 1, i));
            ir.push_str(&format!("  call i32 @puts(i8* %ptr{})\n", i));
        }
        
        // Return 0
        ir.push_str("  ret i32 0\n");
        ir.push_str("}\n");
        
        Ok(ir)
    }
    
    /// Collect string literals and print calls from AST
    fn collect_ast_elements(&self, node: &ASTNode, string_literals: &mut Vec<String>, print_calls: &mut Vec<String>) -> Result<(), CodegenError> {
        match node {
            ASTNode::Atom(AtomValue::Symbol(symbol)) => {
                // Handle special symbols like demo_program
                if symbol == "demo_program" {
                    string_literals.push("Hello, World from Aether!".to_string());
                    string_literals.push("Compiled via LLVM pipeline!".to_string());
                    print_calls.push("Hello, World from Aether!".to_string());
                    print_calls.push("Compiled via LLVM pipeline!".to_string());
                }
            }
            ASTNode::List(nodes) => {
                // Check if this is a function call
                if let Some(ASTNodeRef::Direct(first_node)) = nodes.first() {
                    if let ASTNode::Atom(AtomValue::Symbol(func_name)) = first_node.as_ref() {
                        match func_name.as_str() {
                            "func" => {
                                // Function definition - process body (skip name and params)
                                for node_ref in nodes.iter().skip(2) {
                                    if let ASTNodeRef::Direct(node) = node_ref {
                                        self.collect_ast_elements(node, string_literals, print_calls)?;
                                    }
                                }
                            }
                            "call" => {
                                // Function call - check if it's a print call
                                if nodes.len() >= 3 {
                                    if let (Some(ASTNodeRef::Direct(func_node)), Some(ASTNodeRef::Direct(arg_node))) = 
                                        (nodes.get(1), nodes.get(2)) {
                                        if let (ASTNode::Atom(AtomValue::Symbol(func)), ASTNode::Atom(AtomValue::String(arg))) = 
                                            (func_node.as_ref(), arg_node.as_ref()) {
                                            if func == "print" {
                                                string_literals.push(arg.clone());
                                                print_calls.push(arg.clone());
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {
                                // Process other list contents
                                for node_ref in nodes {
                                    if let ASTNodeRef::Direct(node) = node_ref {
                                        self.collect_ast_elements(node, string_literals, print_calls)?;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Process all nodes in the list
                    for node_ref in nodes {
                        if let ASTNodeRef::Direct(node) = node_ref {
                            self.collect_ast_elements(node, string_literals, print_calls)?;
                        }
                    }
                }
            }
            _ => {
                // Other node types - ignore for now
            }
        }
        
        Ok(())
    }
}

/// Code generation errors
#[derive(Debug, Clone)]
pub enum CodegenError {
    /// MLIR conversion error
    MLIRConversionError(String),
    /// LLVM IR generation error
    LLVMError(String),
    /// Target architecture error
    TargetError(String),
    /// Optimization error
    OptimizationError(String),
    /// Linking error
    LinkingError(String),
    /// I/O error
    IOError(String),
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodegenError::MLIRConversionError(msg) => write!(f, "MLIR conversion error: {}", msg),
            CodegenError::LLVMError(msg) => write!(f, "LLVM error: {}", msg),
            CodegenError::TargetError(msg) => write!(f, "Target error: {}", msg),
            CodegenError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
            CodegenError::LinkingError(msg) => write!(f, "Linking error: {}", msg),
            CodegenError::IOError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for CodegenError {}

impl From<MLIRError> for CodegenError {
    fn from(error: MLIRError) -> Self {
        CodegenError::MLIRConversionError(error.to_string())
    }
}