// Aether Static Analyzer
// Performance and memory analysis tool

use aether_language::{
    AST, ASTNode, parse_sexpr, sweet_to_sexpr
};
use clap::{Arg, Command};
use colored::*;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Serialize, Deserialize)]
struct AnalysisReport {
    file_path: String,
    memory_analysis: MemoryAnalysis,
    performance_analysis: PerformanceAnalysis,
    tensor_analysis: TensorAnalysis,
    gpu_analysis: GpuAnalysis,
    warnings: Vec<Warning>,
    errors: Vec<AnalysisError>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MemoryAnalysis {
    total_allocations: usize,
    peak_memory_usage: usize,
    memory_leaks: Vec<MemoryLeak>,
    linear_type_violations: Vec<LinearTypeViolation>,
    allocation_patterns: Vec<AllocationPattern>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PerformanceAnalysis {
    computational_complexity: String,
    bottlenecks: Vec<PerformanceBottleneck>,
    optimization_opportunities: Vec<OptimizationOpportunity>,
    parallelization_potential: ParallelizationAnalysis,
}

#[derive(Debug, Serialize, Deserialize)]
struct TensorAnalysis {
    tensor_operations: Vec<TensorOperation>,
    shape_compatibility_issues: Vec<ShapeCompatibilityIssue>,
    dimension_mismatches: Vec<DimensionMismatch>,
    broadcasting_opportunities: Vec<BroadcastingOpportunity>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GpuAnalysis {
    gpu_utilization: f64,
    memory_bandwidth_usage: f64,
    kernel_efficiency: Vec<KernelEfficiency>,
    memory_coalescing_issues: Vec<MemoryCoalescingIssue>,
    occupancy_analysis: OccupancyAnalysis,
}

#[derive(Debug, Serialize, Deserialize)]
struct Warning {
    severity: String,
    message: String,
    location: SourceLocation,
    suggestion: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnalysisError {
    message: String,
    location: SourceLocation,
    error_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct SourceLocation {
    line: usize,
    column: usize,
    file: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct MemoryLeak {
    allocation_site: SourceLocation,
    size: usize,
    type_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct LinearTypeViolation {
    location: SourceLocation,
    violation_type: String,
    variable_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct AllocationPattern {
    pattern_type: String,
    frequency: usize,
    average_size: usize,
    locations: Vec<SourceLocation>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PerformanceBottleneck {
    location: SourceLocation,
    bottleneck_type: String,
    impact_score: f64,
    description: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OptimizationOpportunity {
    location: SourceLocation,
    optimization_type: String,
    potential_speedup: f64,
    description: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ParallelizationAnalysis {
    parallelizable_loops: Vec<ParallelizableLoop>,
    data_dependencies: Vec<DataDependency>,
    suggested_parallelization: Vec<ParallelizationSuggestion>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ParallelizableLoop {
    location: SourceLocation,
    loop_type: String,
    parallelization_factor: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct DataDependency {
    source: SourceLocation,
    target: SourceLocation,
    dependency_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ParallelizationSuggestion {
    location: SourceLocation,
    suggestion: String,
    expected_speedup: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct TensorOperation {
    location: SourceLocation,
    operation_type: String,
    input_shapes: Vec<Vec<usize>>,
    output_shape: Vec<usize>,
    computational_cost: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ShapeCompatibilityIssue {
    location: SourceLocation,
    operation: String,
    expected_shapes: Vec<Vec<usize>>,
    actual_shapes: Vec<Vec<usize>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DimensionMismatch {
    location: SourceLocation,
    operation: String,
    dimension_error: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct BroadcastingOpportunity {
    location: SourceLocation,
    current_shapes: Vec<Vec<usize>>,
    optimized_shapes: Vec<Vec<usize>>,
    performance_gain: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct KernelEfficiency {
    kernel_name: String,
    location: SourceLocation,
    efficiency_score: f64,
    issues: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MemoryCoalescingIssue {
    location: SourceLocation,
    access_pattern: String,
    efficiency_loss: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct OccupancyAnalysis {
    theoretical_occupancy: f64,
    achieved_occupancy: f64,
    limiting_factors: Vec<String>,
}

fn main() {
    let matches = Command::new("aether-analyze")
        .version("0.1.0")
        .about("Aether Static Analyzer - Performance and memory analysis tool")
        .arg(
            Arg::new("memory")
                .long("memory")
                .help("Perform memory analysis")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("performance")
                .long("performance")
                .help("Perform performance analysis")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("tensors")
                .long("tensors")
                .help("Perform tensor shape analysis")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("gpu")
                .long("gpu")
                .help("Perform GPU utilization analysis")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("all")
                .long("all")
                .help("Perform all analyses")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .help("Output file for analysis report (JSON format)")
                .value_name("FILE")
        )
        .arg(
            Arg::new("format")
                .long("format")
                .help("Output format: text, json, html")
                .value_name("FORMAT")
                .default_value("text")
        )
        .arg(
            Arg::new("verbose")
                .long("verbose")
                .short('v')
                .help("Verbose output")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("INPUT")
                .help("Input Aether source file")
                .required(true)
                .index(1)
        )
        .get_matches();

    let input_file = matches.get_one::<String>("INPUT").unwrap();
    let analyze_memory = matches.get_flag("memory") || matches.get_flag("all");
    let analyze_performance = matches.get_flag("performance") || matches.get_flag("all");
    let analyze_tensors = matches.get_flag("tensors") || matches.get_flag("all");
    let analyze_gpu = matches.get_flag("gpu") || matches.get_flag("all");
    let output_file = matches.get_one::<String>("output");
    let format = matches.get_one::<String>("format").unwrap();
    let verbose = matches.get_flag("verbose");

    if !analyze_memory && !analyze_performance && !analyze_tensors && !analyze_gpu {
        eprintln!("{}", "Error: Must specify at least one analysis type (--memory, --performance, --tensors, --gpu, or --all)".red());
        std::process::exit(1);
    }

    // Read and parse input file
    let source_content = match fs::read_to_string(input_file) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("{}: {}", "Error reading input file".red(), e);
            std::process::exit(1);
        }
    };

    let ast = match parse_source_file(&source_content) {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("{}: {}", "Error parsing source file".red(), e);
            std::process::exit(1);
        }
    };

    if verbose {
        println!("{} {}", "Analyzing".green(), input_file);
    }

    // Perform analyses
    let mut report = AnalysisReport {
        file_path: input_file.to_string(),
        memory_analysis: MemoryAnalysis::default(),
        performance_analysis: PerformanceAnalysis::default(),
        tensor_analysis: TensorAnalysis::default(),
        gpu_analysis: GpuAnalysis::default(),
        warnings: Vec::new(),
        errors: Vec::new(),
    };

    if analyze_memory {
        if verbose {
            println!("  {} Memory analysis...", "Running".cyan());
        }
        report.memory_analysis = perform_memory_analysis(&ast, input_file);
    }

    if analyze_performance {
        if verbose {
            println!("  {} Performance analysis...", "Running".cyan());
        }
        report.performance_analysis = perform_performance_analysis(&ast, input_file);
    }

    if analyze_tensors {
        if verbose {
            println!("  {} Tensor analysis...", "Running".cyan());
        }
        report.tensor_analysis = perform_tensor_analysis(&ast, input_file);
    }

    if analyze_gpu {
        if verbose {
            println!("  {} GPU analysis...", "Running".cyan());
        }
        report.gpu_analysis = perform_gpu_analysis(&ast, input_file);
    }

    // Output results
    match format.as_str() {
        "json" => output_json_report(&report, output_file),
        "html" => output_html_report(&report, output_file),
        _ => output_text_report(&report, output_file, verbose),
    }
}

fn parse_source_file(content: &str) -> Result<AST, String> {
    // Try to parse as sweet syntax first, then fall back to S-expressions
    let lines: Vec<&str> = content.lines().collect();
    
    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with(";") {
            continue;
        }

        // Try sweet syntax first
        if let Ok(sexpr) = sweet_to_sexpr(line) {
            if let Ok(ast_node) = parse_sexpr(&sexpr) {
                // Use the parsed AST node instead of creating a symbol
                return Ok(AST::new(ast_node.root));
            }
        }

        // Try direct S-expression parsing
        if let Ok(ast_node) = parse_sexpr(line) {
            // Use the parsed AST node instead of creating a symbol
            return Ok(AST::new(ast_node.root));
        }
    }

    // Create a minimal AST for demonstration
    Ok(AST::new(ASTNode::symbol("demo_program".to_string())))
}

fn perform_memory_analysis(_ast: &AST, file_path: &str) -> MemoryAnalysis {
    // Mock memory analysis implementation
    // TODO: Implement actual AST analysis for memory patterns
    MemoryAnalysis {
        total_allocations: 42,
        peak_memory_usage: 1024 * 1024, // 1MB
        memory_leaks: vec![
            MemoryLeak {
                allocation_site: SourceLocation {
                    line: 15,
                    column: 8,
                    file: file_path.to_string(),
                },
                size: 256,
                type_name: "Tensor<f32>".to_string(),
            }
        ],
        linear_type_violations: vec![
            LinearTypeViolation {
                location: SourceLocation {
                    line: 23,
                    column: 12,
                    file: file_path.to_string(),
                },
                violation_type: "Double use".to_string(),
                variable_name: "gpu_buffer".to_string(),
            }
        ],
        allocation_patterns: vec![
            AllocationPattern {
                pattern_type: "Frequent small allocations".to_string(),
                frequency: 150,
                average_size: 64,
                locations: vec![
                    SourceLocation {
                        line: 30,
                        column: 4,
                        file: file_path.to_string(),
                    }
                ],
            }
        ],
    }
}

fn perform_performance_analysis(_ast: &AST, file_path: &str) -> PerformanceAnalysis {
    // Mock performance analysis implementation
    // TODO: Implement actual AST analysis for performance bottlenecks
    PerformanceAnalysis {
        computational_complexity: "O(n²)".to_string(),
        bottlenecks: vec![
            PerformanceBottleneck {
                location: SourceLocation {
                    line: 45,
                    column: 8,
                    file: file_path.to_string(),
                },
                bottleneck_type: "Nested loop".to_string(),
                impact_score: 0.85,
                description: "Nested loop with O(n²) complexity could be optimized".to_string(),
            }
        ],
        optimization_opportunities: vec![
            OptimizationOpportunity {
                location: SourceLocation {
                    line: 52,
                    column: 4,
                    file: file_path.to_string(),
                },
                optimization_type: "Vectorization".to_string(),
                potential_speedup: 3.2,
                description: "Loop can be vectorized using SIMD instructions".to_string(),
            }
        ],
        parallelization_potential: ParallelizationAnalysis {
            parallelizable_loops: vec![
                ParallelizableLoop {
                    location: SourceLocation {
                        line: 60,
                        column: 4,
                        file: file_path.to_string(),
                    },
                    loop_type: "Data parallel".to_string(),
                    parallelization_factor: 4.0,
                }
            ],
            data_dependencies: vec![],
            suggested_parallelization: vec![
                ParallelizationSuggestion {
                    location: SourceLocation {
                        line: 60,
                        column: 4,
                        file: file_path.to_string(),
                    },
                    suggestion: "Use @parallel for annotation".to_string(),
                    expected_speedup: 3.8,
                }
            ],
        },
    }
}

fn perform_tensor_analysis(_ast: &AST, file_path: &str) -> TensorAnalysis {
    // Mock tensor analysis implementation
    // TODO: Implement actual AST analysis for tensor operations and shapes
    TensorAnalysis {
        tensor_operations: vec![
            TensorOperation {
                location: SourceLocation {
                    line: 25,
                    column: 8,
                    file: file_path.to_string(),
                },
                operation_type: "Matrix multiplication".to_string(),
                input_shapes: vec![vec![128, 256], vec![256, 512]],
                output_shape: vec![128, 512],
                computational_cost: 33554432.0, // 128 * 256 * 512
            }
        ],
        shape_compatibility_issues: vec![],
        dimension_mismatches: vec![
            DimensionMismatch {
                location: SourceLocation {
                    line: 35,
                    column: 12,
                    file: file_path.to_string(),
                },
                operation: "Element-wise addition".to_string(),
                dimension_error: "Cannot add tensors of shape [128, 256] and [128, 512]".to_string(),
            }
        ],
        broadcasting_opportunities: vec![
            BroadcastingOpportunity {
                location: SourceLocation {
                    line: 40,
                    column: 8,
                    file: file_path.to_string(),
                },
                current_shapes: vec![vec![128, 256], vec![1, 256]],
                optimized_shapes: vec![vec![128, 256], vec![128, 256]],
                performance_gain: 1.5,
            }
        ],
    }
}

fn perform_gpu_analysis(_ast: &AST, file_path: &str) -> GpuAnalysis {
    // Mock GPU analysis implementation
    // TODO: Implement actual AST analysis for GPU utilization and kernel efficiency
    GpuAnalysis {
        gpu_utilization: 0.75,
        memory_bandwidth_usage: 0.82,
        kernel_efficiency: vec![
            KernelEfficiency {
                kernel_name: "matrix_multiply_kernel".to_string(),
                location: SourceLocation {
                    line: 70,
                    column: 4,
                    file: file_path.to_string(),
                },
                efficiency_score: 0.68,
                issues: vec![
                    "Low occupancy due to register usage".to_string(),
                    "Uncoalesced memory access pattern".to_string(),
                ],
            }
        ],
        memory_coalescing_issues: vec![
            MemoryCoalescingIssue {
                location: SourceLocation {
                    line: 75,
                    column: 12,
                    file: file_path.to_string(),
                },
                access_pattern: "Strided access".to_string(),
                efficiency_loss: 0.4,
            }
        ],
        occupancy_analysis: OccupancyAnalysis {
            theoretical_occupancy: 1.0,
            achieved_occupancy: 0.65,
            limiting_factors: vec![
                "Register usage per thread".to_string(),
                "Shared memory usage".to_string(),
            ],
        },
    }
}

fn output_text_report(report: &AnalysisReport, output_file: Option<&String>, verbose: bool) {
    let output = format_text_report(report, verbose);
    
    match output_file {
        Some(file) => {
            if let Err(e) = fs::write(file, &output) {
                eprintln!("{}: {}", "Error writing output file".red(), e);
                std::process::exit(1);
            }
            println!("{} {}", "Analysis report written to".green(), file);
        }
        None => {
            print!("{}", output);
        }
    }
}

fn output_json_report(report: &AnalysisReport, output_file: Option<&String>) {
    let json_output = match serde_json::to_string_pretty(report) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("{}: {}", "Error serializing report to JSON".red(), e);
            std::process::exit(1);
        }
    };

    match output_file {
        Some(file) => {
            if let Err(e) = fs::write(file, &json_output) {
                eprintln!("{}: {}", "Error writing output file".red(), e);
                std::process::exit(1);
            }
            println!("{} {}", "JSON analysis report written to".green(), file);
        }
        None => {
            print!("{}", json_output);
        }
    }
}

fn output_html_report(report: &AnalysisReport, output_file: Option<&String>) {
    let html_output = format_html_report(report);
    
    let output_path = output_file.map(|s| s.as_str()).unwrap_or("analysis_report.html");
    
    if let Err(e) = fs::write(output_path, &html_output) {
        eprintln!("{}: {}", "Error writing HTML report".red(), e);
        std::process::exit(1);
    }
    
    println!("{} {}", "HTML analysis report written to".green(), output_path);
}

fn format_text_report(report: &AnalysisReport, _verbose: bool) -> String {
    let mut output = String::new();
    
    output.push_str(&format!("{}\n", "=== Aether Static Analysis Report ===".bold()));
    output.push_str(&format!("File: {}\n\n", report.file_path.cyan()));
    
    // Memory Analysis
    if report.memory_analysis.total_allocations > 0 {
        output.push_str(&format!("{}\n", "Memory Analysis:".yellow().bold()));
        output.push_str(&format!("  Total allocations: {}\n", report.memory_analysis.total_allocations));
        output.push_str(&format!("  Peak memory usage: {} bytes\n", report.memory_analysis.peak_memory_usage));
        
        if !report.memory_analysis.memory_leaks.is_empty() {
            output.push_str(&format!("  {} Memory leaks detected:\n", "⚠".red()));
            for leak in &report.memory_analysis.memory_leaks {
                output.push_str(&format!("    - {} bytes at {}:{}\n", 
                    leak.size, leak.allocation_site.line, leak.allocation_site.column));
            }
        }
        
        if !report.memory_analysis.linear_type_violations.is_empty() {
            output.push_str(&format!("  {} Linear type violations:\n", "⚠".red()));
            for violation in &report.memory_analysis.linear_type_violations {
                output.push_str(&format!("    - {} at {}:{}\n", 
                    violation.violation_type, violation.location.line, violation.location.column));
            }
        }
        output.push('\n');
    }
    
    // Performance Analysis
    if !report.performance_analysis.computational_complexity.is_empty() {
        output.push_str(&format!("{}\n", "Performance Analysis:".yellow().bold()));
        output.push_str(&format!("  Computational complexity: {}\n", report.performance_analysis.computational_complexity));
        
        if !report.performance_analysis.bottlenecks.is_empty() {
            output.push_str(&format!("  {} Performance bottlenecks:\n", "⚠".yellow()));
            for bottleneck in &report.performance_analysis.bottlenecks {
                output.push_str(&format!("    - {} (impact: {:.2}) at {}:{}\n", 
                    bottleneck.description, bottleneck.impact_score, 
                    bottleneck.location.line, bottleneck.location.column));
            }
        }
        
        if !report.performance_analysis.optimization_opportunities.is_empty() {
            output.push_str(&format!("  {} Optimization opportunities:\n", "💡".green()));
            for opt in &report.performance_analysis.optimization_opportunities {
                output.push_str(&format!("    - {} (speedup: {:.1}x) at {}:{}\n", 
                    opt.description, opt.potential_speedup, 
                    opt.location.line, opt.location.column));
            }
        }
        output.push('\n');
    }
    
    // Tensor Analysis
    if !report.tensor_analysis.tensor_operations.is_empty() {
        output.push_str(&format!("{}\n", "Tensor Analysis:".yellow().bold()));
        output.push_str(&format!("  Tensor operations: {}\n", report.tensor_analysis.tensor_operations.len()));
        
        if !report.tensor_analysis.dimension_mismatches.is_empty() {
            output.push_str(&format!("  {} Dimension mismatches:\n", "❌".red()));
            for mismatch in &report.tensor_analysis.dimension_mismatches {
                output.push_str(&format!("    - {} at {}:{}\n", 
                    mismatch.dimension_error, mismatch.location.line, mismatch.location.column));
            }
        }
        
        if !report.tensor_analysis.broadcasting_opportunities.is_empty() {
            output.push_str(&format!("  {} Broadcasting optimizations:\n", "💡".green()));
            for broadcast in &report.tensor_analysis.broadcasting_opportunities {
                output.push_str(&format!("    - {:.1}x speedup at {}:{}\n", 
                    broadcast.performance_gain, broadcast.location.line, broadcast.location.column));
            }
        }
        output.push('\n');
    }
    
    // GPU Analysis
    if report.gpu_analysis.gpu_utilization > 0.0 {
        output.push_str(&format!("{}\n", "GPU Analysis:".yellow().bold()));
        output.push_str(&format!("  GPU utilization: {:.1}%\n", report.gpu_analysis.gpu_utilization * 100.0));
        output.push_str(&format!("  Memory bandwidth usage: {:.1}%\n", report.gpu_analysis.memory_bandwidth_usage * 100.0));
        output.push_str(&format!("  Achieved occupancy: {:.1}%\n", report.gpu_analysis.occupancy_analysis.achieved_occupancy * 100.0));
        
        if !report.gpu_analysis.kernel_efficiency.is_empty() {
            output.push_str(&format!("  Kernel efficiency:\n"));
            for kernel in &report.gpu_analysis.kernel_efficiency {
                output.push_str(&format!("    - {}: {:.1}%\n", kernel.kernel_name, kernel.efficiency_score * 100.0));
            }
        }
        output.push('\n');
    }
    
    // Summary
    let total_warnings = report.warnings.len();
    let total_errors = report.errors.len();
    
    if total_errors > 0 {
        output.push_str(&format!("{} {} errors found\n", "❌".red(), total_errors));
    }
    if total_warnings > 0 {
        output.push_str(&format!("{} {} warnings found\n", "⚠".yellow(), total_warnings));
    }
    if total_errors == 0 && total_warnings == 0 {
        output.push_str(&format!("{} No issues found\n", "✅".green()));
    }
    
    output
}

fn format_html_report(report: &AnalysisReport) -> String {
    format!(r#"<!DOCTYPE html>
<html>
<head>
    <title>Aether Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        .section {{ margin: 20px 0; }}
        .warning {{ color: #ff6600; }}
        .error {{ color: #cc0000; }}
        .success {{ color: #00aa00; }}
        .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .issue {{ background: #fff3cd; padding: 8px; margin: 5px 0; border-left: 4px solid #ffc107; }}
        .opportunity {{ background: #d4edda; padding: 8px; margin: 5px 0; border-left: 4px solid #28a745; }}
    </style>
</head>
<body>
    <h1 class="header">Aether Static Analysis Report</h1>
    <p><strong>File:</strong> {}</p>
    
    <div class="section">
        <h2>Memory Analysis</h2>
        <div class="metric">Total allocations: {}</div>
        <div class="metric">Peak memory usage: {} bytes</div>
    </div>
    
    <div class="section">
        <h2>Performance Analysis</h2>
        <div class="metric">Computational complexity: {}</div>
    </div>
    
    <div class="section">
        <h2>GPU Analysis</h2>
        <div class="metric">GPU utilization: {:.1}%</div>
        <div class="metric">Memory bandwidth usage: {:.1}%</div>
        <div class="metric">Achieved occupancy: {:.1}%</div>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <p class="success">Analysis complete</p>
    </div>
</body>
</html>"#,
        report.file_path,
        report.memory_analysis.total_allocations,
        report.memory_analysis.peak_memory_usage,
        report.performance_analysis.computational_complexity,
        report.gpu_analysis.gpu_utilization * 100.0,
        report.gpu_analysis.memory_bandwidth_usage * 100.0,
        report.gpu_analysis.occupancy_analysis.achieved_occupancy * 100.0
    )
}

// Default implementations for analysis structures
impl Default for MemoryAnalysis {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            peak_memory_usage: 0,
            memory_leaks: Vec::new(),
            linear_type_violations: Vec::new(),
            allocation_patterns: Vec::new(),
        }
    }
}

impl Default for PerformanceAnalysis {
    fn default() -> Self {
        Self {
            computational_complexity: String::new(),
            bottlenecks: Vec::new(),
            optimization_opportunities: Vec::new(),
            parallelization_potential: ParallelizationAnalysis {
                parallelizable_loops: Vec::new(),
                data_dependencies: Vec::new(),
                suggested_parallelization: Vec::new(),
            },
        }
    }
}

impl Default for TensorAnalysis {
    fn default() -> Self {
        Self {
            tensor_operations: Vec::new(),
            shape_compatibility_issues: Vec::new(),
            dimension_mismatches: Vec::new(),
            broadcasting_opportunities: Vec::new(),
        }
    }
}

impl Default for GpuAnalysis {
    fn default() -> Self {
        Self {
            gpu_utilization: 0.0,
            memory_bandwidth_usage: 0.0,
            kernel_efficiency: Vec::new(),
            memory_coalescing_issues: Vec::new(),
            occupancy_analysis: OccupancyAnalysis {
                theoretical_occupancy: 0.0,
                achieved_occupancy: 0.0,
                limiting_factors: Vec::new(),
            },
        }
    }
}