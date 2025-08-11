use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Run file compilation tests if requested
    if env::var("AETHER_RUN_FILE_TESTS").is_ok() {
        run_file_compilation_tests();
    }
    // Only build MLIR bindings if the mlir feature is enabled
    if !cfg!(feature = "mlir") {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MLIR_SYS_180_PREFIX");
    println!("cargo:rerun-if-env-changed=LLVM_SYS_180_PREFIX");

    // Try to find MLIR/LLVM installation
    let mlir_config = find_mlir_config();
    
    match mlir_config {
        Ok(config) => {
            println!("cargo:rustc-link-search=native={}", config.lib_dir.display());
            
            // Link MLIR libraries
            let mlir_libs = [
                "MLIR",
                "MLIRAnalysis",
                "MLIRBuiltinToLLVMIRTranslation",
                "MLIRCallInterfaces",
                "MLIRCastInterfaces",
                "MLIRControlFlowInterfaces",
                "MLIRDataLayoutInterfaces",
                "MLIRDerivedAttributeOpInterface",
                "MLIRDialect",
                "MLIRExecutionEngine",
                "MLIRFuncDialect",
                "MLIRFunctionInterfaces",
                "MLIRIR",
                "MLIRInferTypeOpInterface",
                "MLIRLLVMCommonConversion",
                "MLIRLLVMDialect",
                "MLIRLLVMToLLVMIRTranslation",
                "MLIRLinalgDialect",
                "MLIRMemRefDialect",
                "MLIRParser",
                "MLIRPass",
                "MLIRSCFDialect",
                "MLIRSideEffectInterfaces",
                "MLIRSupport",
                "MLIRTensorDialect",
                "MLIRTransforms",
                "MLIRTranslateLib",
                "MLIRArithDialect",
                "MLIRGPUDialect",
                "MLIRSPIRVDialect",
            ];

            for lib in &mlir_libs {
                println!("cargo:rustc-link-lib=static={}", lib);
            }

            // Link LLVM libraries
            let llvm_libs = [
                "LLVMCore",
                "LLVMSupport",
                "LLVMBitReader",
                "LLVMBitWriter",
                "LLVMTransformUtils",
                "LLVMAnalysis",
                "LLVMTarget",
                "LLVMMC",
                "LLVMObject",
                "LLVMCodeGen",
                "LLVMExecutionEngine",
                "LLVMJIT",
                "LLVMInterpreter",
                "LLVMX86CodeGen",
                "LLVMX86AsmParser",
                "LLVMX86Desc",
                "LLVMX86Info",
                "LLVMAArch64CodeGen",
                "LLVMAArch64AsmParser",
                "LLVMAArch64Desc",
                "LLVMAArch64Info",
            ];

            for lib in &llvm_libs {
                println!("cargo:rustc-link-lib=static={}", lib);
            }

            // Link system libraries
            #[cfg(target_os = "linux")]
            {
                println!("cargo:rustc-link-lib=dylib=stdc++");
                println!("cargo:rustc-link-lib=dylib=m");
                println!("cargo:rustc-link-lib=dylib=z");
                println!("cargo:rustc-link-lib=dylib=zstd");
            }

            #[cfg(target_os = "macos")]
            {
                println!("cargo:rustc-link-lib=dylib=c++");
                println!("cargo:rustc-link-lib=dylib=z");
                println!("cargo:rustc-link-lib=dylib=zstd");
            }

            #[cfg(target_os = "windows")]
            {
                println!("cargo:rustc-link-lib=dylib=msvcrt");
                println!("cargo:rustc-link-lib=dylib=shell32");
                println!("cargo:rustc-link-lib=dylib=ole32");
            }

            println!("cargo:rustc-env=MLIR_INCLUDE_DIR={}", config.include_dir.display());
        }
        Err(e) => {
            println!("cargo:warning=Could not find MLIR installation: {}", e);
            println!("cargo:warning=Building with stub MLIR implementation");
            println!("cargo:warning=To use real MLIR, install MLIR/LLVM and set MLIR_SYS_190_PREFIX");
        }
    }
}

struct MLIRConfig {
    lib_dir: PathBuf,
    include_dir: PathBuf,
}

fn find_mlir_config() -> Result<MLIRConfig, String> {
    // Try environment variable first
    if let Ok(prefix) = env::var("MLIR_SYS_180_PREFIX") {
        let prefix_path = PathBuf::from(prefix);
        return Ok(MLIRConfig {
            lib_dir: prefix_path.join("lib"),
            include_dir: prefix_path.join("include"),
        });
    }

    // Try LLVM prefix as fallback
    if let Ok(prefix) = env::var("LLVM_SYS_180_PREFIX") {
        let prefix_path = PathBuf::from(prefix);
        return Ok(MLIRConfig {
            lib_dir: prefix_path.join("lib"),
            include_dir: prefix_path.join("include"),
        });
    }

    // Try pkg-config
    if let Ok(library) = pkg_config::Config::new()
        .atleast_version("18.0")
        .probe("mlir")
    {
        if let (Some(lib_dir), Some(include_dir)) = (
            library.link_paths.first(),
            library.include_paths.first(),
        ) {
            return Ok(MLIRConfig {
                lib_dir: lib_dir.clone(),
                include_dir: include_dir.clone(),
            });
        }
    }

    // Try common installation paths
    let common_paths = [
        "/usr/local",
        "/usr",
        "/opt/homebrew", // macOS Homebrew
        "/opt/local",    // macOS MacPorts
    ];

    for path in &common_paths {
        let lib_dir = PathBuf::from(path).join("lib");
        let include_dir = PathBuf::from(path).join("include");
        
        if lib_dir.exists() && include_dir.exists() {
            // Check if MLIR headers exist
            if include_dir.join("mlir-c").join("IR.h").exists() {
                return Ok(MLIRConfig { lib_dir, include_dir });
            }
        }
    }

    Err("MLIR installation not found. Please install MLIR/LLVM or set MLIR_SYS_180_PREFIX environment variable.".to_string())
}

/// Run file compilation tests during build if requested
fn run_file_compilation_tests() {
    println!("cargo:warning=Running Aether file compilation tests...");
    
    // Check if the aether-file-test binary exists
    let test_binary = if cfg!(windows) {
        "target/debug/aether-file-test.exe"
    } else {
        "target/debug/aether-file-test"
    };
    
    // Only run if the binary exists (to avoid circular dependency)
    if std::path::Path::new(test_binary).exists() {
        let output = Command::new(test_binary)
            .args(&["run", "--format", "console", "--verbose"])
            .output();
            
        match output {
            Ok(result) => {
                if result.status.success() {
                    println!("cargo:warning=✓ File compilation tests passed");
                } else {
                    println!("cargo:warning=⚠ File compilation tests failed");
                    if !result.stderr.is_empty() {
                        let stderr = String::from_utf8_lossy(&result.stderr);
                        println!("cargo:warning=Test errors: {}", stderr);
                    }
                }
            }
            Err(e) => {
                println!("cargo:warning=Failed to run file compilation tests: {}", e);
            }
        }
    } else {
        println!("cargo:warning=Skipping file compilation tests (binary not found)");
    }
}