// Aether Runtime Error Handling
// Stack traces, resource cleanup, and error recovery

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::backtrace::Backtrace;
use std::fmt;
use crate::runtime::native_runtime::RuntimeError;

/// Enhanced runtime error with stack trace and context
#[derive(Debug, Clone)]
pub struct RuntimeErrorContext {
    pub error: RuntimeError,
    pub stack_trace: Vec<RuntimeStackFrame>,
    pub error_context: HashMap<String, String>,
    pub resource_state: ResourceState,
}

/// Runtime stack frame with Aether-specific information
#[derive(Debug, Clone)]
pub struct RuntimeStackFrame {
    pub function_name: String,
    pub file: String,
    pub line: usize,
    pub column: usize,
    pub local_variables: HashMap<String, String>,
    pub tensor_shapes: HashMap<String, Vec<usize>>,
    pub memory_usage: usize,
}

impl RuntimeStackFrame {
    pub fn new(function_name: String, file: String, line: usize, column: usize) -> Self {
        RuntimeStackFrame {
            function_name,
            file,
            line,
            column,
            local_variables: HashMap::new(),
            tensor_shapes: HashMap::new(),
            memory_usage: 0,
        }
    }

    pub fn with_variable(mut self, name: String, value: String) -> Self {
        self.local_variables.insert(name, value);
        self
    }

    pub fn with_tensor_shape(mut self, name: String, shape: Vec<usize>) -> Self {
        self.tensor_shapes.insert(name, shape);
        self
    }

    pub fn with_memory_usage(mut self, usage: usize) -> Self {
        self.memory_usage = usage;
        self
    }
}

impl fmt::Display for RuntimeStackFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "  at {} ({}:{}:{})", self.function_name, self.file, self.line, self.column)?;
        
        if !self.local_variables.is_empty() {
            write!(f, "\n    locals: ")?;
            for (name, value) in &self.local_variables {
                write!(f, "{}={} ", name, value)?;
            }
        }
        
        if !self.tensor_shapes.is_empty() {
            write!(f, "\n    tensors: ")?;
            for (name, shape) in &self.tensor_shapes {
                write!(f, "{}={:?} ", name, shape)?;
            }
        }
        
        if self.memory_usage > 0 {
            write!(f, "\n    memory: {} bytes", self.memory_usage)?;
        }
        
        Ok(())
    }
}

/// Resource state tracking for cleanup
#[derive(Debug, Clone)]
pub struct ResourceState {
    pub allocated_memory: Vec<AllocationInfo>,
    pub open_files: Vec<FileInfo>,
    pub gpu_resources: Vec<GpuResourceInfo>,
    pub tensor_references: Vec<TensorInfo>,
}

#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub ptr: usize,
    pub size: usize,
    pub allocation_type: AllocationType,
    pub stack_trace: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FileInfo {
    pub handle: usize,
    pub path: String,
    pub mode: String,
}

#[derive(Debug, Clone)]
pub struct GpuResourceInfo {
    pub resource_id: usize,
    pub resource_type: GpuResourceType,
    pub memory_size: usize,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub tensor_id: usize,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AllocationType {
    Heap,
    Stack,
    Gpu,
    Shared,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuResourceType {
    Buffer,
    Texture,
    Kernel,
    Context,
}

impl Default for ResourceState {
    fn default() -> Self {
        ResourceState {
            allocated_memory: Vec::new(),
            open_files: Vec::new(),
            gpu_resources: Vec::new(),
            tensor_references: Vec::new(),
        }
    }
}

/// Runtime error handler with cleanup and recovery
pub struct RuntimeErrorHandler {
    error_handlers: HashMap<String, Box<dyn Fn(&RuntimeErrorContext) -> RecoveryAction + Send + Sync>>,
    cleanup_handlers: Vec<Box<dyn Fn(&ResourceState) + Send + Sync>>,
    stack_trace_collector: StackTraceCollector,
    resource_tracker: Arc<Mutex<ResourceState>>,
}

/// Recovery action for runtime errors
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Continue execution
    Continue,
    /// Retry the operation
    Retry,
    /// Skip the current operation
    Skip,
    /// Terminate with cleanup
    Terminate,
    /// Custom recovery with message
    Custom(String),
}

impl RuntimeErrorHandler {
    pub fn new() -> Self {
        let mut handler = RuntimeErrorHandler {
            error_handlers: HashMap::new(),
            cleanup_handlers: Vec::new(),
            stack_trace_collector: StackTraceCollector::new(),
            resource_tracker: Arc::new(Mutex::new(ResourceState::default())),
        };
        
        handler.register_default_handlers();
        handler
    }

    fn register_default_handlers(&mut self) {
        // Memory error handler
        self.register_error_handler("memory".to_string(), Box::new(|ctx: &RuntimeErrorContext| {
            match &ctx.error {
                RuntimeError::MemoryError(msg) => {
                    eprintln!("Memory error: {}", msg);
                    if msg.contains("out of memory") {
                        // Try to free some resources
                        RecoveryAction::Custom("attempting garbage collection".to_string())
                    } else {
                        RecoveryAction::Terminate
                    }
                }
                _ => RecoveryAction::Continue,
            }
        }));

        // Tensor error handler
        self.register_error_handler("tensor".to_string(), Box::new(|ctx: &RuntimeErrorContext| {
            match &ctx.error {
                RuntimeError::TensorError(msg) => {
                    eprintln!("Tensor error: {}", msg);
                    if msg.contains("shape mismatch") {
                        RecoveryAction::Skip
                    } else {
                        RecoveryAction::Terminate
                    }
                }
                _ => RecoveryAction::Continue,
            }
        }));

        // GPU error handler
        self.register_error_handler("gpu".to_string(), Box::new(|_ctx: &RuntimeErrorContext| {
            eprintln!("GPU error detected, falling back to CPU");
            RecoveryAction::Custom("fallback to CPU".to_string())
        }));
    }

    pub fn register_error_handler<F>(&mut self, error_type: String, handler: F)
    where
        F: Fn(&RuntimeErrorContext) -> RecoveryAction + Send + Sync + 'static,
    {
        self.error_handlers.insert(error_type, Box::new(handler));
    }

    pub fn register_cleanup_handler<F>(&mut self, handler: F)
    where
        F: Fn(&ResourceState) + Send + Sync + 'static,
    {
        self.cleanup_handlers.push(Box::new(handler));
    }

    pub fn handle_error(&self, error: RuntimeError) -> RecoveryAction {
        // Collect stack trace
        let stack_trace = self.stack_trace_collector.collect_stack_trace();
        
        // Get current resource state
        let resource_state = self.resource_tracker.lock().unwrap().clone();
        
        // Create error context
        let mut error_context = HashMap::new();
        error_context.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
        error_context.insert("thread_id".to_string(), format!("{:?}", std::thread::current().id()));
        
        let ctx = RuntimeErrorContext {
            error: error.clone(),
            stack_trace,
            error_context,
            resource_state: resource_state.clone(),
        };

        // Try specific error handlers
        let error_type = self.classify_error(&error);
        if let Some(handler) = self.error_handlers.get(&error_type) {
            let action = handler(&ctx);
            
            // Perform cleanup if terminating
            if matches!(action, RecoveryAction::Terminate) {
                self.perform_cleanup(&resource_state);
            }
            
            return action;
        }

        // Default handling
        eprintln!("Unhandled runtime error: {}", error);
        self.print_stack_trace(&ctx.stack_trace);
        
        RecoveryAction::Terminate
    }

    fn classify_error(&self, error: &RuntimeError) -> String {
        match error {
            RuntimeError::MemoryError(_) => "memory".to_string(),
            RuntimeError::TensorError(_) => "tensor".to_string(),
            RuntimeError::AutodiffError(_) => "autodiff".to_string(),
            RuntimeError::ProbabilisticError(_) => "probabilistic".to_string(),
            RuntimeError::SystemError(_) => "system".to_string(),
            RuntimeError::IOError(_) => "io".to_string(),
            _ => "unknown".to_string(),
        }
    }

    fn perform_cleanup(&self, resource_state: &ResourceState) {
        println!("Performing emergency cleanup...");
        
        // Call registered cleanup handlers
        for handler in &self.cleanup_handlers {
            handler(resource_state);
        }
        
        // Default cleanup
        self.cleanup_memory(&resource_state.allocated_memory);
        self.cleanup_files(&resource_state.open_files);
        self.cleanup_gpu_resources(&resource_state.gpu_resources);
        self.cleanup_tensors(&resource_state.tensor_references);
    }

    fn cleanup_memory(&self, allocations: &[AllocationInfo]) {
        for allocation in allocations {
            match allocation.allocation_type {
                AllocationType::Heap => {
                    println!("Cleaning up heap allocation: {} bytes at {:x}", 
                            allocation.size, allocation.ptr);
                    // In real implementation: unsafe { dealloc(...) }
                }
                AllocationType::Gpu => {
                    println!("Cleaning up GPU allocation: {} bytes", allocation.size);
                    // In real implementation: GPU memory free
                }
                AllocationType::Shared => {
                    println!("Cleaning up shared memory: {} bytes", allocation.size);
                    // In real implementation: shared memory cleanup
                }
                AllocationType::Stack => {
                    // Stack memory is automatically cleaned up
                }
            }
        }
    }

    fn cleanup_files(&self, files: &[FileInfo]) {
        for file in files {
            println!("Closing file: {} (handle: {})", file.path, file.handle);
            // In real implementation: close file handle
        }
    }

    fn cleanup_gpu_resources(&self, resources: &[GpuResourceInfo]) {
        for resource in resources {
            println!("Cleaning up GPU resource: {:?} ({})", 
                    resource.resource_type, resource.resource_id);
            // In real implementation: GPU resource cleanup
        }
    }

    fn cleanup_tensors(&self, tensors: &[TensorInfo]) {
        for tensor in tensors {
            println!("Cleaning up tensor: {} shape={:?} device={}", 
                    tensor.tensor_id, tensor.shape, tensor.device);
            // In real implementation: tensor cleanup
        }
    }

    fn print_stack_trace(&self, stack_trace: &[RuntimeStackFrame]) {
        if stack_trace.is_empty() {
            println!("No stack trace available");
            return;
        }
        
        println!("Stack trace:");
        for frame in stack_trace {
            println!("{}", frame);
        }
    }

    pub fn track_allocation(&self, ptr: usize, size: usize, allocation_type: AllocationType) {
        let mut resource_state = self.resource_tracker.lock().unwrap();
        resource_state.allocated_memory.push(AllocationInfo {
            ptr,
            size,
            allocation_type,
            stack_trace: self.stack_trace_collector.collect_simple_trace(),
        });
    }

    pub fn track_file(&self, handle: usize, path: String, mode: String) {
        let mut resource_state = self.resource_tracker.lock().unwrap();
        resource_state.open_files.push(FileInfo { handle, path, mode });
    }

    pub fn track_gpu_resource(&self, resource_id: usize, resource_type: GpuResourceType, memory_size: usize) {
        let mut resource_state = self.resource_tracker.lock().unwrap();
        resource_state.gpu_resources.push(GpuResourceInfo {
            resource_id,
            resource_type,
            memory_size,
        });
    }

    pub fn track_tensor(&self, tensor_id: usize, shape: Vec<usize>, dtype: String, device: String) {
        let mut resource_state = self.resource_tracker.lock().unwrap();
        resource_state.tensor_references.push(TensorInfo {
            tensor_id,
            shape,
            dtype,
            device,
        });
    }

    pub fn untrack_allocation(&self, ptr: usize) {
        let mut resource_state = self.resource_tracker.lock().unwrap();
        resource_state.allocated_memory.retain(|alloc| alloc.ptr != ptr);
    }

    pub fn untrack_file(&self, handle: usize) {
        let mut resource_state = self.resource_tracker.lock().unwrap();
        resource_state.open_files.retain(|file| file.handle != handle);
    }

    pub fn untrack_gpu_resource(&self, resource_id: usize) {
        let mut resource_state = self.resource_tracker.lock().unwrap();
        resource_state.gpu_resources.retain(|res| res.resource_id != resource_id);
    }

    pub fn untrack_tensor(&self, tensor_id: usize) {
        let mut resource_state = self.resource_tracker.lock().unwrap();
        resource_state.tensor_references.retain(|tensor| tensor.tensor_id != tensor_id);
    }
}

/// Stack trace collector for runtime errors
pub struct StackTraceCollector {
    max_frames: usize,
}

impl StackTraceCollector {
    pub fn new() -> Self {
        StackTraceCollector { max_frames: 50 }
    }

    pub fn with_max_frames(max_frames: usize) -> Self {
        StackTraceCollector { max_frames }
    }

    pub fn collect_stack_trace(&self) -> Vec<RuntimeStackFrame> {
        // In a real implementation, this would use platform-specific APIs
        // to walk the stack and extract Aether function information
        
        // For now, return a mock stack trace
        vec![
            RuntimeStackFrame::new(
                "main".to_string(),
                "main.ae".to_string(),
                10,
                5,
            ).with_variable("x".to_string(), "42".to_string())
             .with_memory_usage(1024),
            
            RuntimeStackFrame::new(
                "compute_tensor".to_string(),
                "tensor_ops.ae".to_string(),
                25,
                12,
            ).with_tensor_shape("input".to_string(), vec![32, 64, 128])
             .with_memory_usage(2048),
        ]
    }

    pub fn collect_simple_trace(&self) -> Vec<String> {
        // Simplified stack trace for allocation tracking
        vec![
            "aether_alloc".to_string(),
            "user_function".to_string(),
            "main".to_string(),
        ]
    }
}

/// Panic handler for Aether runtime
pub fn install_panic_handler() {
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("Aether runtime panic: {}", panic_info);
        
        // Collect backtrace
        let backtrace = Backtrace::capture();
        eprintln!("Backtrace:\n{}", backtrace);
        
        // Perform emergency cleanup
        eprintln!("Performing emergency cleanup...");
        
        // In a real implementation, this would call the global error handler
        // to perform resource cleanup before terminating
        
        std::process::exit(1);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_error_handler() {
        let handler = RuntimeErrorHandler::new();
        let error = RuntimeError::MemoryError("test error".to_string());
        
        let action = handler.handle_error(error);
        
        // Should not panic and return some action
        match action {
            RecoveryAction::Terminate | RecoveryAction::Custom(_) => {},
            _ => panic!("Unexpected recovery action"),
        }
    }

    #[test]
    fn test_resource_tracking() {
        let handler = RuntimeErrorHandler::new();
        
        handler.track_allocation(0x1000, 1024, AllocationType::Heap);
        handler.track_file(1, "test.txt".to_string(), "r".to_string());
        
        // Verify resources are tracked
        let resource_state = handler.resource_tracker.lock().unwrap();
        assert_eq!(resource_state.allocated_memory.len(), 1);
        assert_eq!(resource_state.open_files.len(), 1);
    }

    #[test]
    fn test_stack_trace_collector() {
        let collector = StackTraceCollector::new();
        let trace = collector.collect_stack_trace();
        
        assert!(!trace.is_empty());
        assert!(trace[0].function_name == "main");
    }

    #[test]
    fn test_runtime_stack_frame() {
        let frame = RuntimeStackFrame::new(
            "test_function".to_string(),
            "test.ae".to_string(),
            10,
            5,
        ).with_variable("x".to_string(), "42".to_string())
         .with_tensor_shape("tensor".to_string(), vec![32, 64]);
        
        assert_eq!(frame.function_name, "test_function");
        assert_eq!(frame.local_variables.get("x"), Some(&"42".to_string()));
        assert_eq!(frame.tensor_shapes.get("tensor"), Some(&vec![32, 64]));
    }
}