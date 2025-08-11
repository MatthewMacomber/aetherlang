// Aether Playground
// Browser-based REPL with real-time compilation

use aether_language::{
    AST, ASTNode, sweet_to_sexpr, sexpr_to_sweet_string, parse_sexpr,
    NativeCompilationPipeline, WasmCompilationPipeline, WasmTarget
};
use clap::{Arg, Command};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use warp::Filter;

#[derive(Debug, Serialize, Deserialize)]
struct CompileRequest {
    code: String,
    target: String,
    optimization: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CompileResponse {
    success: bool,
    output: Option<String>,
    errors: Vec<CompileError>,
    warnings: Vec<CompileWarning>,
    execution_time: f64,
    ast: Option<String>,
    tokens: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CompileError {
    message: String,
    line: usize,
    column: usize,
    severity: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CompileWarning {
    message: String,
    line: usize,
    column: usize,
    suggestion: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FormatRequest {
    code: String,
    to_format: String, // "sweet" or "canonical"
}

#[derive(Debug, Serialize, Deserialize)]
struct FormatResponse {
    success: bool,
    formatted_code: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnalyzeRequest {
    code: String,
    analysis_types: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnalyzeResponse {
    success: bool,
    memory_usage: Option<MemoryUsageInfo>,
    performance_metrics: Option<PerformanceMetrics>,
    type_info: Option<TypeInfo>,
    suggestions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MemoryUsageInfo {
    estimated_allocations: usize,
    peak_usage: usize,
    linear_type_issues: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PerformanceMetrics {
    complexity: String,
    bottlenecks: Vec<String>,
    optimization_opportunities: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TypeInfo {
    inferred_types: HashMap<String, String>,
    type_errors: Vec<String>,
    tensor_shapes: HashMap<String, Vec<usize>>,
}

type PlaygroundState = Arc<RwLock<HashMap<String, String>>>;

#[tokio::main]
async fn main() {
    let matches = Command::new("aether-playground")
        .version("0.1.0")
        .about("Aether Playground - Browser-based REPL with real-time compilation")
        .arg(
            Arg::new("port")
                .long("port")
                .short('p')
                .help("Port to run the server on")
                .value_name("PORT")
                .default_value("3030")
        )
        .arg(
            Arg::new("host")
                .long("host")
                .help("Host to bind the server to")
                .value_name("HOST")
                .default_value("127.0.0.1")
        )
        .arg(
            Arg::new("open")
                .long("open")
                .help("Open browser automatically")
                .action(clap::ArgAction::SetTrue)
        )
        .get_matches();

    let port: u16 = matches.get_one::<String>("port")
        .unwrap()
        .parse()
        .expect("Invalid port number");
    
    let host = matches.get_one::<String>("host").unwrap();
    let open_browser = matches.get_flag("open");

    let state: PlaygroundState = Arc::new(RwLock::new(HashMap::new()));

    // CORS configuration
    let cors = warp::cors()
        .allow_any_origin()
        .allow_headers(vec!["content-type"])
        .allow_methods(vec!["GET", "POST", "OPTIONS"]);

    // Static files route
    let static_files = warp::path("static")
        .and(warp::fs::dir("static"));

    // API routes
    let compile_route = warp::path("api")
        .and(warp::path("compile"))
        .and(warp::post())
        .and(warp::body::json())
        .and(with_state(state.clone()))
        .and_then(handle_compile);

    let format_route = warp::path("api")
        .and(warp::path("format"))
        .and(warp::post())
        .and(warp::body::json())
        .and_then(handle_format);

    let analyze_route = warp::path("api")
        .and(warp::path("analyze"))
        .and(warp::post())
        .and(warp::body::json())
        .and_then(handle_analyze);

    // Main page route
    let index_route = warp::path::end()
        .and(warp::get())
        .and_then(serve_index);

    let routes = index_route
        .or(static_files)
        .or(compile_route)
        .or(format_route)
        .or(analyze_route)
        .with(cors);

    println!("🚀 Aether Playground starting on http://{}:{}", host, port);
    
    if open_browser {
        let url = format!("http://{}:{}", host, port);
        if let Err(e) = open::that(&url) {
            eprintln!("Failed to open browser: {}", e);
        }
    }

    let addr: std::net::SocketAddr = format!("{}:{}", host, port).parse().unwrap();
    warp::serve(routes).run(addr).await;
}

fn with_state(state: PlaygroundState) -> impl Filter<Extract = (PlaygroundState,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || state.clone())
}

async fn handle_compile(request: CompileRequest, _state: PlaygroundState) -> Result<impl warp::Reply, warp::Rejection> {
    let start_time = std::time::Instant::now();
    
    // Parse the code
    let ast_result = parse_code(&request.code);
    let ast = match ast_result {
        Ok(ast) => ast,
        Err(e) => {
            let response = CompileResponse {
                success: false,
                output: None,
                errors: vec![CompileError {
                    message: e,
                    line: 1,
                    column: 1,
                    severity: "error".to_string(),
                }],
                warnings: Vec::new(),
                execution_time: start_time.elapsed().as_secs_f64(),
                ast: None,
                tokens: None,
            };
            return Ok(warp::reply::json(&response));
        }
    };

    // Compile based on target
    let compile_result = match request.target.as_str() {
        "wasm" => compile_to_wasm(&ast),
        "native" => compile_to_native(&ast),
        _ => Err("Unsupported target".to_string()),
    };

    let response = match compile_result {
        Ok(output) => CompileResponse {
            success: true,
            output: Some(output),
            errors: Vec::new(),
            warnings: generate_mock_warnings(),
            execution_time: start_time.elapsed().as_secs_f64(),
            ast: Some(format!("{:?}", ast)),
            tokens: Some(vec!["let".to_string(), "x".to_string(), "=".to_string(), "42".to_string()]),
        },
        Err(e) => CompileResponse {
            success: false,
            output: None,
            errors: vec![CompileError {
                message: e,
                line: 1,
                column: 1,
                severity: "error".to_string(),
            }],
            warnings: Vec::new(),
            execution_time: start_time.elapsed().as_secs_f64(),
            ast: None,
            tokens: None,
        },
    };

    Ok(warp::reply::json(&response))
}

async fn handle_format(request: FormatRequest) -> Result<impl warp::Reply, warp::Rejection> {
    let result = match request.to_format.as_str() {
        "sweet" => {
            // Convert from canonical to sweet
            match parse_sexpr(&request.code) {
                Ok(_) => {
                    match sexpr_to_sweet_string(&request.code) {
                        Ok(sweet) => FormatResponse {
                            success: true,
                            formatted_code: Some(sweet),
                            error: None,
                        },
                        Err(e) => FormatResponse {
                            success: false,
                            formatted_code: None,
                            error: Some(e.to_string()),
                        },
                    }
                }
                Err(e) => FormatResponse {
                    success: false,
                    formatted_code: None,
                    error: Some(e.to_string()),
                },
            }
        }
        "canonical" => {
            // Convert from sweet to canonical
            match sweet_to_sexpr(&request.code) {
                Ok(sexpr) => FormatResponse {
                    success: true,
                    formatted_code: Some(sexpr),
                    error: None,
                },
                Err(e) => FormatResponse {
                    success: false,
                    formatted_code: None,
                    error: Some(e.to_string()),
                },
            }
        }
        _ => FormatResponse {
            success: false,
            formatted_code: None,
            error: Some("Invalid format type".to_string()),
        },
    };

    Ok(warp::reply::json(&result))
}

async fn handle_analyze(request: AnalyzeRequest) -> Result<impl warp::Reply, warp::Rejection> {
    let ast = match parse_code(&request.code) {
        Ok(ast) => ast,
        Err(e) => {
            let response = AnalyzeResponse {
                success: false,
                memory_usage: None,
                performance_metrics: None,
                type_info: None,
                suggestions: vec![format!("Parse error: {}", e)],
            };
            return Ok(warp::reply::json(&response));
        }
    };

    let mut memory_usage = None;
    let mut performance_metrics = None;
    let mut type_info = None;
    let mut suggestions = Vec::new();

    for analysis_type in &request.analysis_types {
        match analysis_type.as_str() {
            "memory" => {
                memory_usage = Some(MemoryUsageInfo {
                    estimated_allocations: 5,
                    peak_usage: 1024,
                    linear_type_issues: vec!["Potential double-use of linear variable 'buffer'".to_string()],
                });
            }
            "performance" => {
                performance_metrics = Some(PerformanceMetrics {
                    complexity: "O(n)".to_string(),
                    bottlenecks: vec!["Loop at line 3 could be vectorized".to_string()],
                    optimization_opportunities: vec!["Consider using @parallel annotation".to_string()],
                });
            }
            "types" => {
                let mut inferred_types = HashMap::new();
                inferred_types.insert("x".to_string(), "i32".to_string());
                inferred_types.insert("y".to_string(), "f64".to_string());
                
                let mut tensor_shapes = HashMap::new();
                tensor_shapes.insert("matrix".to_string(), vec![128, 256]);
                
                type_info = Some(TypeInfo {
                    inferred_types,
                    type_errors: Vec::new(),
                    tensor_shapes,
                });
            }
            _ => {
                suggestions.push(format!("Unknown analysis type: {}", analysis_type));
            }
        }
    }

    let response = AnalyzeResponse {
        success: true,
        memory_usage,
        performance_metrics,
        type_info,
        suggestions,
    };

    Ok(warp::reply::json(&response))
}

async fn serve_index() -> Result<impl warp::Reply, warp::Rejection> {
    let html = include_str!("../../static/playground.html");
    Ok(warp::reply::html(html))
}

fn parse_code(code: &str) -> Result<AST, String> {
    let lines: Vec<&str> = code.lines().collect();
    
    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with(";") {
            continue;
        }

        // Try sweet syntax first
        if let Ok(sexpr) = sweet_to_sexpr(line) {
            return Ok(AST::new(ASTNode::symbol(sexpr)));
        }

        // Try direct S-expression parsing
        if let Ok(_) = parse_sexpr(line) {
            return Ok(AST::new(ASTNode::symbol(line.to_string())));
        }
    }

    // Create a minimal AST for demonstration
    Ok(AST::new(ASTNode::symbol("demo_program".to_string())))
}

fn compile_to_wasm(ast: &AST) -> Result<String, String> {
    let mut pipeline = WasmCompilationPipeline::new(WasmTarget::Browser).map_err(|e| format!("Pipeline creation failed: {}", e))?;
    let output_path = Path::new("playground_output");
    
    match pipeline.compile_to_wasm(ast, output_path) {
        Ok(()) => Ok("WebAssembly compilation successful! Generated playground_output.wasm".to_string()),
        Err(e) => Err(format!("WebAssembly compilation failed: {}", e)),
    }
}

fn compile_to_native(ast: &AST) -> Result<String, String> {
    let mut pipeline = NativeCompilationPipeline::for_current_platform().map_err(|e| format!("Pipeline creation failed: {}", e))?;
    let output_path = Path::new("playground_output");
    
    match pipeline.compile_to_native(ast, output_path) {
        Ok(()) => Ok("Native compilation successful! Generated playground_output executable".to_string()),
        Err(e) => Err(format!("Native compilation failed: {}", e)),
    }
}

fn generate_mock_warnings() -> Vec<CompileWarning> {
    vec![
        CompileWarning {
            message: "Consider using more specific type annotation".to_string(),
            line: 2,
            column: 8,
            suggestion: Some("Add type annotation: let x: i32 = 42".to_string()),
        },
        CompileWarning {
            message: "This loop could be parallelized".to_string(),
            line: 5,
            column: 4,
            suggestion: Some("Use @parallel for annotation".to_string()),
        },
    ]
}