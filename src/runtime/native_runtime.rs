// Native runtime system for Aether
// Provides memory management, error handling, and system integration

use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::ptr::NonNull;

/// Native runtime for Aether programs
pub struct AetherRuntime {
    memory_manager: Arc<Mutex<MemoryManager>>,
    error_handler: ErrorHandler,
    tensor_runtime: TensorRuntime,
    autodiff_runtime: AutodiffRuntime,
    prob_runtime: ProbabilisticRuntime,
}

impl AetherRuntime {
    /// Initialize the Aether runtime
    pub fn new() -> Self {
        AetherRuntime {
            memory_manager: Arc::new(Mutex::new(MemoryManager::new())),
            error_handler: ErrorHandler::new(),
            tensor_runtime: TensorRuntime::new(),
            autodiff_runtime: AutodiffRuntime::new(),
            prob_runtime: ProbabilisticRuntime::new(),
        }
    }

    /// Initialize runtime (called from generated code)
    pub fn init() -> Result<(), RuntimeError> {
        // Initialize global runtime state
        GLOBAL_RUNTIME.lock().unwrap().replace(AetherRuntime::new());
        Ok(())
    }

    /// Shutdown runtime and cleanup resources
    pub fn shutdown() -> Result<(), RuntimeError> {
        if let Some(runtime) = GLOBAL_RUNTIME.lock().unwrap().take() {
            runtime.cleanup()?;
        }
        Ok(())
    }

    /// Cleanup all runtime resources
    fn cleanup(&self) -> Result<(), RuntimeError> {
        // Cleanup memory manager
        let mut memory_manager = self.memory_manager.lock().unwrap();
        memory_manager.cleanup()?;
        
        // Cleanup tensor runtime
        self.tensor_runtime.cleanup()?;
        
        // Cleanup autodiff runtime
        self.autodiff_runtime.cleanup()?;
        
        // Cleanup probabilistic runtime
        self.prob_runtime.cleanup()?;
        
        Ok(())
    }

    /// Get memory manager
    pub fn memory_manager(&self) -> Arc<Mutex<MemoryManager>> {
        self.memory_manager.clone()
    }

    /// Get error handler
    pub fn error_handler(&self) -> &ErrorHandler {
        &self.error_handler
    }
}

/// Global runtime instance
static GLOBAL_RUNTIME: Mutex<Option<AetherRuntime>> = Mutex::new(None);

/// Get global runtime instance
pub fn get_runtime() -> Result<Arc<AetherRuntime>, RuntimeError> {
    let runtime_guard = GLOBAL_RUNTIME.lock().unwrap();
    runtime_guard.as_ref()
        .map(|_| Arc::new(AetherRuntime::new())) // Simplified for mock
        .ok_or(RuntimeError::RuntimeNotInitialized)
}

/// Memory manager for deterministic resource management
pub struct MemoryManager {
    allocations: HashMap<usize, AllocationInfo>,
    total_allocated: usize,
    peak_allocated: usize,
    allocation_counter: usize,
}

/// Information about a memory allocation
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    size: usize,
    layout: Layout,
    allocation_id: usize,
    stack_trace: Vec<String>, // Simplified stack trace
}

impl MemoryManager {
    /// Create new memory manager
    pub fn new() -> Self {
        MemoryManager {
            allocations: HashMap::new(),
            total_allocated: 0,
            peak_allocated: 0,
            allocation_counter: 0,
        }
    }

    /// Allocate memory with linear type tracking
    pub fn allocate(&mut self, size: usize, align: usize) -> Result<NonNull<u8>, RuntimeError> {
        let layout = Layout::from_size_align(size, align)
            .map_err(|e| RuntimeError::MemoryError(format!("Invalid layout: {}", e)))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(RuntimeError::MemoryError("Allocation failed".to_string()));
        }

        let non_null_ptr = NonNull::new(ptr)
            .ok_or_else(|| RuntimeError::MemoryError("Null pointer returned".to_string()))?;

        // Track allocation
        self.allocation_counter += 1;
        let allocation_info = AllocationInfo {
            size,
            layout,
            allocation_id: self.allocation_counter,
            stack_trace: self.capture_stack_trace(),
        };

        self.allocations.insert(ptr as usize, allocation_info);
        self.total_allocated += size;
        if self.total_allocated > self.peak_allocated {
            self.peak_allocated = self.total_allocated;
        }

        Ok(non_null_ptr)
    }

    /// Deallocate memory with ownership verification
    pub fn deallocate(&mut self, ptr: NonNull<u8>) -> Result<(), RuntimeError> {
        let ptr_addr = ptr.as_ptr() as usize;
        
        if let Some(allocation_info) = self.allocations.remove(&ptr_addr) {
            unsafe {
                dealloc(ptr.as_ptr(), allocation_info.layout);
            }
            self.total_allocated -= allocation_info.size;
            Ok(())
        } else {
            Err(RuntimeError::MemoryError(
                "Attempted to deallocate untracked pointer".to_string()
            ))
        }
    }

    /// Check for memory leaks
    pub fn check_leaks(&self) -> Vec<AllocationInfo> {
        self.allocations.values().cloned().collect()
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.total_allocated,
            peak_allocated: self.peak_allocated,
            active_allocations: self.allocations.len(),
        }
    }

    /// Cleanup all allocations (for shutdown)
    fn cleanup(&mut self) -> Result<(), RuntimeError> {
        let leaked_allocations = self.check_leaks();
        if !leaked_allocations.is_empty() {
            eprintln!("Warning: {} memory leaks detected during shutdown", leaked_allocations.len());
            for leak in &leaked_allocations {
                eprintln!("  Leak: {} bytes, allocation #{}", leak.size, leak.allocation_id);
            }
        }

        // Force cleanup of all remaining allocations
        for (ptr_addr, allocation_info) in self.allocations.drain() {
            unsafe {
                dealloc(ptr_addr as *mut u8, allocation_info.layout);
            }
        }
        
        self.total_allocated = 0;
        Ok(())
    }

    /// Capture simplified stack trace
    fn capture_stack_trace(&self) -> Vec<String> {
        // Simplified implementation - in real runtime would use backtrace
        vec!["aether_alloc".to_string(), "user_code".to_string()]
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub active_allocations: usize,
}

/// Error handler for runtime errors
pub struct ErrorHandler {
    error_callbacks: Vec<Box<dyn Fn(&RuntimeError) + Send + Sync>>,
}

impl ErrorHandler {
    /// Create new error handler
    pub fn new() -> Self {
        ErrorHandler {
            error_callbacks: Vec::new(),
        }
    }

    /// Register error callback
    pub fn register_callback<F>(&mut self, callback: F)
    where
        F: Fn(&RuntimeError) + Send + Sync + 'static,
    {
        self.error_callbacks.push(Box::new(callback));
    }

    /// Handle runtime error
    pub fn handle_error(&self, error: &RuntimeError) {
        // Call all registered callbacks
        for callback in &self.error_callbacks {
            callback(error);
        }

        // Default error handling
        match error {
            RuntimeError::MemoryError(_) => {
                eprintln!("Memory error: {}", error);
                // Could trigger cleanup or recovery
            }
            RuntimeError::TensorError(_) => {
                eprintln!("Tensor error: {}", error);
                // Could reset tensor state
            }
            RuntimeError::AutodiffError(_) => {
                eprintln!("Autodiff error: {}", error);
                // Could clear gradient tape
            }
            RuntimeError::ProbabilisticError(_) => {
                eprintln!("Probabilistic error: {}", error);
                // Could reset random state
            }
            _ => {
                eprintln!("Runtime error: {}", error);
            }
        }
    }
}

/// Tensor runtime support
pub struct TensorRuntime {
    tensor_cache: HashMap<usize, TensorInfo>,
}

#[derive(Debug, Clone)]
struct TensorInfo {
    shape: Vec<usize>,
    dtype: String,
    data_ptr: usize,
}

impl TensorRuntime {
    pub fn new() -> Self {
        TensorRuntime {
            tensor_cache: HashMap::new(),
        }
    }

    pub fn cleanup(&self) -> Result<(), RuntimeError> {
        // Cleanup tensor resources
        Ok(())
    }
}

/// Automatic differentiation runtime support
pub struct AutodiffRuntime {
    gradient_tape: Vec<GradientOp>,
}

#[derive(Debug, Clone)]
struct GradientOp {
    op_type: String,
    inputs: Vec<usize>,
    output: usize,
}

impl AutodiffRuntime {
    pub fn new() -> Self {
        AutodiffRuntime {
            gradient_tape: Vec::new(),
        }
    }

    pub fn cleanup(&self) -> Result<(), RuntimeError> {
        // Cleanup autodiff resources
        Ok(())
    }
}

/// Probabilistic programming runtime support
pub struct ProbabilisticRuntime {
    random_state: u64,
    distributions: HashMap<String, DistributionInfo>,
}

#[derive(Debug, Clone)]
struct DistributionInfo {
    dist_type: String,
    parameters: Vec<f64>,
}

impl ProbabilisticRuntime {
    pub fn new() -> Self {
        ProbabilisticRuntime {
            random_state: 12345, // Default seed
            distributions: HashMap::new(),
        }
    }

    pub fn cleanup(&self) -> Result<(), RuntimeError> {
        // Cleanup probabilistic resources
        Ok(())
    }
}

/// Runtime error types
#[derive(Debug, Clone)]
pub enum RuntimeError {
    /// Runtime not initialized
    RuntimeNotInitialized,
    /// Memory management error
    MemoryError(String),
    /// Tensor operation error
    TensorError(String),
    /// Automatic differentiation error
    AutodiffError(String),
    /// Probabilistic programming error
    ProbabilisticError(String),
    /// System error
    SystemError(String),
    /// I/O error
    IOError(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::RuntimeNotInitialized => write!(f, "Runtime not initialized"),
            RuntimeError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            RuntimeError::TensorError(msg) => write!(f, "Tensor error: {}", msg),
            RuntimeError::AutodiffError(msg) => write!(f, "Autodiff error: {}", msg),
            RuntimeError::ProbabilisticError(msg) => write!(f, "Probabilistic error: {}", msg),
            RuntimeError::SystemError(msg) => write!(f, "System error: {}", msg),
            RuntimeError::IOError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for RuntimeError {}

// C-compatible runtime functions are now in runtime_library.rs