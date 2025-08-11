// LLVM IR runtime function declarations for Aether
// Provides LLVM IR declarations for runtime functions

use std::collections::HashMap;

/// LLVM IR runtime function declarations
pub struct LLVMRuntimeDeclarations {
    declarations: HashMap<String, String>,
}

impl LLVMRuntimeDeclarations {
    /// Create new runtime declarations
    pub fn new() -> Self {
        let mut declarations = HashMap::new();
        Self::register_builtin_declarations(&mut declarations);
        
        LLVMRuntimeDeclarations { declarations }
    }

    /// Register built-in runtime function declarations
    fn register_builtin_declarations(declarations: &mut HashMap<String, String>) {
        // Runtime initialization and cleanup
        declarations.insert(
            "aether_runtime_init".to_string(),
            "declare i32 @aether_runtime_init()".to_string()
        );
        declarations.insert(
            "aether_runtime_shutdown".to_string(),
            "declare i32 @aether_runtime_shutdown()".to_string()
        );

        // Memory management functions
        declarations.insert(
            "aether_alloc".to_string(),
            "declare ptr @aether_alloc(i64 %size, i64 %align)".to_string()
        );
        declarations.insert(
            "aether_dealloc".to_string(),
            "declare void @aether_dealloc(ptr %ptr)".to_string()
        );
        declarations.insert(
            "aether_realloc".to_string(),
            "declare ptr @aether_realloc(ptr %ptr, i64 %old_size, i64 %new_size, i64 %align)".to_string()
        );

        // Tensor operations
        declarations.insert(
            "aether_tensor_create".to_string(),
            "declare ptr @aether_tensor_create(ptr %data, ptr %shape, i64 %shape_len, i32 %dtype)".to_string()
        );
        declarations.insert(
            "aether_tensor_destroy".to_string(),
            "declare void @aether_tensor_destroy(ptr %tensor)".to_string()
        );
        declarations.insert(
            "aether_tensor_add".to_string(),
            "declare ptr @aether_tensor_add(ptr %lhs, ptr %rhs)".to_string()
        );
        declarations.insert(
            "aether_tensor_mul".to_string(),
            "declare ptr @aether_tensor_mul(ptr %lhs, ptr %rhs)".to_string()
        );
        declarations.insert(
            "aether_tensor_matmul".to_string(),
            "declare ptr @aether_tensor_matmul(ptr %lhs, ptr %rhs)".to_string()
        );

        // Automatic differentiation functions
        declarations.insert(
            "aether_autodiff_forward".to_string(),
            "declare ptr @aether_autodiff_forward(ptr %input, ptr %operation, ptr %params)".to_string()
        );
        declarations.insert(
            "aether_autodiff_backward".to_string(),
            "declare ptr @aether_autodiff_backward(ptr %output_grad, ptr %computation_graph)".to_string()
        );
        declarations.insert(
            "aether_gradient_compute".to_string(),
            "declare ptr @aether_gradient_compute(ptr %loss, ptr %variables, i64 %var_count)".to_string()
        );

        // Probabilistic programming functions
        declarations.insert(
            "aether_prob_sample".to_string(),
            "declare double @aether_prob_sample(ptr %distribution, ptr %params, i64 %param_count)".to_string()
        );
        declarations.insert(
            "aether_prob_observe".to_string(),
            "declare i32 @aether_prob_observe(ptr %variable, double %observed_value)".to_string()
        );
        declarations.insert(
            "aether_prob_infer".to_string(),
            "declare ptr @aether_prob_infer(ptr %model, ptr %evidence, ptr %query)".to_string()
        );

        // Linear type management
        declarations.insert(
            "aether_linear_move".to_string(),
            "declare ptr @aether_linear_move(ptr %source)".to_string()
        );
        declarations.insert(
            "aether_linear_drop".to_string(),
            "declare void @aether_linear_drop(ptr %ptr)".to_string()
        );

        // Concurrency primitives
        declarations.insert(
            "aether_actor_spawn".to_string(),
            "declare ptr @aether_actor_spawn(ptr %actor_fn, ptr %initial_state)".to_string()
        );
        declarations.insert(
            "aether_message_send".to_string(),
            "declare i32 @aether_message_send(ptr %actor, ptr %message)".to_string()
        );
        declarations.insert(
            "aether_parallel_for".to_string(),
            "declare void @aether_parallel_for(i64 %start, i64 %end, ptr %body_fn, ptr %context)".to_string()
        );

        // Utility functions
        declarations.insert(
            "aether_runtime_stats".to_string(),
            "declare ptr @aether_runtime_stats()".to_string()
        );
        declarations.insert(
            "aether_handle_error".to_string(),
            "declare void @aether_handle_error(i32 %error_code, ptr %message)".to_string()
        );

        // Math library functions (commonly used in AI)
        declarations.insert(
            "aether_math_exp".to_string(),
            "declare double @exp(double %x)".to_string()
        );
        declarations.insert(
            "aether_math_log".to_string(),
            "declare double @log(double %x)".to_string()
        );
        declarations.insert(
            "aether_math_sin".to_string(),
            "declare double @sin(double %x)".to_string()
        );
        declarations.insert(
            "aether_math_cos".to_string(),
            "declare double @cos(double %x)".to_string()
        );
        declarations.insert(
            "aether_math_sqrt".to_string(),
            "declare double @sqrt(double %x)".to_string()
        );
        declarations.insert(
            "aether_math_pow".to_string(),
            "declare double @pow(double %x, double %y)".to_string()
        );

        // BLAS/LAPACK integration (for optimized tensor operations)
        declarations.insert(
            "aether_blas_sgemm".to_string(),
            "declare void @cblas_sgemm(i32 %order, i32 %transa, i32 %transb, i32 %m, i32 %n, i32 %k, float %alpha, ptr %a, i32 %lda, ptr %b, i32 %ldb, float %beta, ptr %c, i32 %ldc)".to_string()
        );
        declarations.insert(
            "aether_blas_dgemm".to_string(),
            "declare void @cblas_dgemm(i32 %order, i32 %transa, i32 %transb, i32 %m, i32 %n, i32 %k, double %alpha, ptr %a, i32 %lda, ptr %b, i32 %ldb, double %beta, ptr %c, i32 %ldc)".to_string()
        );

        // GPU/CUDA runtime functions (when available)
        declarations.insert(
            "aether_cuda_malloc".to_string(),
            "declare i32 @cudaMalloc(ptr %devPtr, i64 %size)".to_string()
        );
        declarations.insert(
            "aether_cuda_free".to_string(),
            "declare i32 @cudaFree(ptr %devPtr)".to_string()
        );
        declarations.insert(
            "aether_cuda_memcpy".to_string(),
            "declare i32 @cudaMemcpy(ptr %dst, ptr %src, i64 %count, i32 %kind)".to_string()
        );

        // OpenMP runtime functions (for parallel execution)
        declarations.insert(
            "aether_omp_get_num_threads".to_string(),
            "declare i32 @omp_get_num_threads()".to_string()
        );
        declarations.insert(
            "aether_omp_get_thread_num".to_string(),
            "declare i32 @omp_get_thread_num()".to_string()
        );
    }

    /// Get LLVM IR declaration for a function
    pub fn get_declaration(&self, function_name: &str) -> Option<&String> {
        self.declarations.get(function_name)
    }

    /// Get all function declarations as LLVM IR
    pub fn get_all_declarations(&self) -> String {
        let mut ir = String::new();
        ir.push_str("; Aether Runtime Function Declarations\n");
        ir.push_str("; Generated automatically - do not edit\n\n");

        // Group declarations by category
        let categories = vec![
            ("Runtime Management", vec!["aether_runtime_init", "aether_runtime_shutdown"]),
            ("Memory Management", vec!["aether_alloc", "aether_dealloc", "aether_realloc"]),
            ("Tensor Operations", vec![
                "aether_tensor_create", "aether_tensor_destroy", 
                "aether_tensor_add", "aether_tensor_mul", "aether_tensor_matmul"
            ]),
            ("Automatic Differentiation", vec![
                "aether_autodiff_forward", "aether_autodiff_backward", "aether_gradient_compute"
            ]),
            ("Probabilistic Programming", vec![
                "aether_prob_sample", "aether_prob_observe", "aether_prob_infer"
            ]),
            ("Linear Types", vec!["aether_linear_move", "aether_linear_drop"]),
            ("Concurrency", vec![
                "aether_actor_spawn", "aether_message_send", "aether_parallel_for"
            ]),
            ("Utilities", vec!["aether_runtime_stats", "aether_handle_error"]),
            ("Math Library", vec![
                "aether_math_exp", "aether_math_log", "aether_math_sin", 
                "aether_math_cos", "aether_math_sqrt", "aether_math_pow"
            ]),
            ("BLAS Integration", vec!["aether_blas_sgemm", "aether_blas_dgemm"]),
            ("CUDA Runtime", vec!["aether_cuda_malloc", "aether_cuda_free", "aether_cuda_memcpy"]),
            ("OpenMP Runtime", vec!["aether_omp_get_num_threads", "aether_omp_get_thread_num"]),
        ];

        for (category_name, function_names) in categories {
            ir.push_str(&format!("; {}\n", category_name));
            for function_name in function_names {
                if let Some(declaration) = self.declarations.get(function_name) {
                    ir.push_str(&format!("{}\n", declaration));
                }
            }
            ir.push('\n');
        }

        ir
    }

    /// Generate LLVM IR module with runtime declarations
    pub fn generate_runtime_module(&self) -> String {
        let mut module = String::new();
        
        module.push_str("; Aether Runtime Module\n");
        module.push_str("; Contains all runtime function declarations\n\n");
        
        // Add target information
        module.push_str("target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n");
        module.push_str("target triple = \"x86_64-unknown-linux-gnu\"\n\n");
        
        // Add runtime declarations
        module.push_str(&self.get_all_declarations());
        
        // Add runtime initialization function
        module.push_str("; Runtime initialization wrapper\n");
        module.push_str("define i32 @_aether_init_runtime() {\n");
        module.push_str("entry:\n");
        module.push_str("  %result = call i32 @aether_runtime_init()\n");
        module.push_str("  ret i32 %result\n");
        module.push_str("}\n\n");
        
        // Add runtime cleanup function
        module.push_str("; Runtime cleanup wrapper\n");
        module.push_str("define i32 @_aether_cleanup_runtime() {\n");
        module.push_str("entry:\n");
        module.push_str("  %result = call i32 @aether_runtime_shutdown()\n");
        module.push_str("  ret i32 %result\n");
        module.push_str("}\n\n");
        
        // Add main function template
        module.push_str("; Main function template\n");
        module.push_str("define i32 @main() {\n");
        module.push_str("entry:\n");
        module.push_str("  ; Initialize runtime\n");
        module.push_str("  %init_result = call i32 @_aether_init_runtime()\n");
        module.push_str("  %init_failed = icmp ne i32 %init_result, 0\n");
        module.push_str("  br i1 %init_failed, label %init_error, label %main_body\n\n");
        
        module.push_str("init_error:\n");
        module.push_str("  ret i32 -1\n\n");
        
        module.push_str("main_body:\n");
        module.push_str("  ; User code goes here\n");
        module.push_str("  br label %cleanup\n\n");
        
        module.push_str("cleanup:\n");
        module.push_str("  ; Cleanup runtime\n");
        module.push_str("  %cleanup_result = call i32 @_aether_cleanup_runtime()\n");
        module.push_str("  ret i32 0\n");
        module.push_str("}\n");
        
        module
    }

    /// Get function names by category
    pub fn get_functions_by_category(&self) -> HashMap<String, Vec<String>> {
        let mut categories = HashMap::new();
        
        // Categorize functions based on their prefixes
        for function_name in self.declarations.keys() {
            let category = if function_name.starts_with("aether_runtime") {
                "Runtime Management"
            } else if function_name.starts_with("aether_alloc") || function_name.starts_with("aether_dealloc") || function_name.starts_with("aether_realloc") {
                "Memory Management"
            } else if function_name.starts_with("aether_tensor") {
                "Tensor Operations"
            } else if function_name.starts_with("aether_autodiff") || function_name.starts_with("aether_gradient") {
                "Automatic Differentiation"
            } else if function_name.starts_with("aether_prob") {
                "Probabilistic Programming"
            } else if function_name.starts_with("aether_linear") {
                "Linear Types"
            } else if function_name.starts_with("aether_actor") || function_name.starts_with("aether_message") || function_name.starts_with("aether_parallel") {
                "Concurrency"
            } else if function_name.starts_with("aether_math") {
                "Math Library"
            } else if function_name.starts_with("aether_blas") {
                "BLAS Integration"
            } else if function_name.starts_with("aether_cuda") {
                "CUDA Runtime"
            } else if function_name.starts_with("aether_omp") {
                "OpenMP Runtime"
            } else {
                "Utilities"
            };
            
            categories.entry(category.to_string())
                .or_insert_with(Vec::new)
                .push(function_name.clone());
        }
        
        categories
    }

    /// Add custom runtime function declaration
    pub fn add_declaration(&mut self, name: String, declaration: String) {
        self.declarations.insert(name, declaration);
    }

    /// Remove runtime function declaration
    pub fn remove_declaration(&mut self, name: &str) -> Option<String> {
        self.declarations.remove(name)
    }

    /// Check if function is declared
    pub fn has_declaration(&self, name: &str) -> bool {
        self.declarations.contains_key(name)
    }

    /// Get count of declared functions
    pub fn declaration_count(&self) -> usize {
        self.declarations.len()
    }
}

impl Default for LLVMRuntimeDeclarations {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_declarations_creation() {
        let declarations = LLVMRuntimeDeclarations::new();
        assert!(declarations.declaration_count() > 0);
    }

    #[test]
    fn test_get_declaration() {
        let declarations = LLVMRuntimeDeclarations::new();
        let alloc_decl = declarations.get_declaration("aether_alloc");
        assert!(alloc_decl.is_some());
        assert!(alloc_decl.unwrap().contains("@aether_alloc"));
    }

    #[test]
    fn test_generate_runtime_module() {
        let declarations = LLVMRuntimeDeclarations::new();
        let module = declarations.generate_runtime_module();
        assert!(module.contains("target datalayout"));
        assert!(module.contains("@aether_runtime_init"));
        assert!(module.contains("define i32 @main()"));
    }

    #[test]
    fn test_functions_by_category() {
        let declarations = LLVMRuntimeDeclarations::new();
        let categories = declarations.get_functions_by_category();
        assert!(categories.contains_key("Memory Management"));
        assert!(categories.contains_key("Tensor Operations"));
        assert!(categories.contains_key("Automatic Differentiation"));
    }

    #[test]
    fn test_add_custom_declaration() {
        let mut declarations = LLVMRuntimeDeclarations::new();
        let initial_count = declarations.declaration_count();
        
        declarations.add_declaration(
            "custom_function".to_string(),
            "declare i32 @custom_function(i32 %arg)".to_string()
        );
        
        assert_eq!(declarations.declaration_count(), initial_count + 1);
        assert!(declarations.has_declaration("custom_function"));
    }
}