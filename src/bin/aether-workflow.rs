// Aether Build Workflow CLI Binary
// Command-line tool for orchestrating Aether build workflows

use aether_language::cli::run_workflow_cli;
use std::process;

fn main() {
    if let Err(e) = run_workflow_cli() {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}