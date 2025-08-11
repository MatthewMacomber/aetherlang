// Aether Format Tool
// Bidirectional transpiler between human-readable and canonical syntax

use aether_language::{sweet_to_sexpr, sexpr_to_sweet_string, parse_sexpr};
use clap::{Arg, Command};
use colored::*;
use std::fs;
use std::io::{self, Read};

fn main() {
    let matches = Command::new("aetherfmt")
        .version("0.1.0")
        .about("Aether Format Tool - Bidirectional transpiler between human-readable and canonical syntax")
        .arg(
            Arg::new("to-sweet")
                .long("to-sweet")
                .help("Convert from canonical S-expressions to human-readable sweet syntax")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("to-canonical")
                .long("to-canonical")
                .help("Convert from sweet syntax to canonical S-expressions")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("pretty")
                .long("pretty")
                .short('p')
                .help("Enable pretty-printing with syntax highlighting")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("check")
                .long("check")
                .help("Check syntax without modifying files")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("output")
                .long("output")
                .short('o')
                .help("Output file (default: stdout)")
                .value_name("FILE")
        )
        .arg(
            Arg::new("INPUT")
                .help("Input file (default: stdin)")
                .index(1)
        )
        .get_matches();

    let to_sweet = matches.get_flag("to-sweet");
    let to_canonical = matches.get_flag("to-canonical");
    let pretty = matches.get_flag("pretty");
    let check_only = matches.get_flag("check");
    let input_file = matches.get_one::<String>("INPUT");
    let output_file = matches.get_one::<String>("output");

    if to_sweet && to_canonical {
        eprintln!("{}", "Error: Cannot specify both --to-sweet and --to-canonical".red());
        std::process::exit(1);
    }

    if !to_sweet && !to_canonical {
        eprintln!("{}", "Error: Must specify either --to-sweet or --to-canonical".red());
        std::process::exit(1);
    }

    // Read input
    let input_content = match input_file {
        Some(file) => {
            match fs::read_to_string(file) {
                Ok(content) => content,
                Err(e) => {
                    eprintln!("{}: {}", "Error reading input file".red(), e);
                    std::process::exit(1);
                }
            }
        }
        None => {
            let mut buffer = String::new();
            match io::stdin().read_to_string(&mut buffer) {
                Ok(_) => buffer,
                Err(e) => {
                    eprintln!("{}: {}", "Error reading from stdin".red(), e);
                    std::process::exit(1);
                }
            }
        }
    };

    // Process the content
    let result = if to_sweet {
        convert_to_sweet(&input_content, pretty)
    } else {
        convert_to_canonical(&input_content, pretty)
    };

    let output_content = match result {
        Ok(content) => content,
        Err(e) => {
            eprintln!("{}: {}", "Conversion error".red(), e);
            std::process::exit(1);
        }
    };

    if check_only {
        println!("{}", "Syntax check passed".green());
        return;
    }

    // Write output
    match output_file {
        Some(file) => {
            match fs::write(file, &output_content) {
                Ok(_) => {
                    println!("{} {}", "Successfully wrote to".green(), file);
                }
                Err(e) => {
                    eprintln!("{}: {}", "Error writing output file".red(), e);
                    std::process::exit(1);
                }
            }
        }
        None => {
            print!("{}", output_content);
        }
    }
}

fn convert_to_sweet(input: &str, pretty: bool) -> Result<String, String> {
    // Parse S-expressions and convert to sweet syntax
    let lines: Vec<&str> = input.lines().collect();
    let mut results = Vec::new();

    for (line_num, line) in lines.iter().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') {
            results.push(line.to_string());
            continue;
        }

        match parse_sexpr(line) {
            Ok(sexpr) => {
                let sexpr_str = format!("{:?}", sexpr); // Convert AST to string representation
                match sexpr_to_sweet_string(&sexpr_str) {
                    Ok(sweet) => {
                        if pretty {
                            results.push(format_sweet_syntax(&sweet));
                        } else {
                            results.push(sweet);
                        }
                    }
                    Err(e) => {
                        return Err(format!("Line {}: {}", line_num + 1, e));
                    }
                }
            }
            Err(e) => {
                return Err(format!("Line {}: {}", line_num + 1, e));
            }
        }
    }

    Ok(results.join("\n"))
}

fn convert_to_canonical(input: &str, pretty: bool) -> Result<String, String> {
    // Parse sweet syntax and convert to S-expressions
    let lines: Vec<&str> = input.lines().collect();
    let mut results = Vec::new();

    for (line_num, line) in lines.iter().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            results.push(line.to_string());
            continue;
        }

        match sweet_to_sexpr(line) {
            Ok(sexpr) => {
                if pretty {
                    results.push(format_sexpr_syntax(&sexpr));
                } else {
                    results.push(sexpr);
                }
            }
            Err(e) => {
                return Err(format!("Line {}: {}", line_num + 1, e));
            }
        }
    }

    Ok(results.join("\n"))
}

fn format_sweet_syntax(sweet: &str) -> String {
    // Apply syntax highlighting for sweet syntax
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        return sweet.to_string();
    }

    let mut result = String::new();
    let mut chars = sweet.chars().peekable();
    let mut in_braces = false;
    let mut in_string = false;
    let mut current_token = String::new();

    while let Some(ch) = chars.next() {
        match ch {
            '{' if !in_string => {
                if !current_token.is_empty() {
                    result.push_str(&highlight_token(&current_token, false));
                    current_token.clear();
                }
                result.push_str(&"{".cyan().to_string());
                in_braces = true;
            }
            '}' if !in_string => {
                if !current_token.is_empty() {
                    result.push_str(&highlight_token(&current_token, true));
                    current_token.clear();
                }
                result.push_str(&"}".cyan().to_string());
                in_braces = false;
            }
            '"' => {
                current_token.push(ch);
                in_string = !in_string;
                if !in_string {
                    result.push_str(&current_token.green().to_string());
                    current_token.clear();
                }
            }
            ' ' | '\t' | '\n' if !in_string => {
                if !current_token.is_empty() {
                    result.push_str(&highlight_token(&current_token, in_braces));
                    current_token.clear();
                }
                result.push(ch);
            }
            _ => {
                current_token.push(ch);
            }
        }
    }

    if !current_token.is_empty() {
        result.push_str(&highlight_token(&current_token, in_braces));
    }

    result
}

fn format_sexpr_syntax(sexpr: &str) -> String {
    // Apply syntax highlighting for S-expressions
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        return sexpr.to_string();
    }

    let mut result = String::new();
    let mut chars = sexpr.chars().peekable();
    let mut in_string = false;
    let mut current_token = String::new();

    while let Some(ch) = chars.next() {
        match ch {
            '(' if !in_string => {
                if !current_token.is_empty() {
                    result.push_str(&highlight_sexpr_token(&current_token));
                    current_token.clear();
                }
                result.push_str(&"(".blue().to_string());
            }
            ')' if !in_string => {
                if !current_token.is_empty() {
                    result.push_str(&highlight_sexpr_token(&current_token));
                    current_token.clear();
                }
                result.push_str(&")".blue().to_string());
            }
            '"' => {
                current_token.push(ch);
                in_string = !in_string;
                if !in_string {
                    result.push_str(&current_token.green().to_string());
                    current_token.clear();
                }
            }
            ' ' | '\t' | '\n' if !in_string => {
                if !current_token.is_empty() {
                    result.push_str(&highlight_sexpr_token(&current_token));
                    current_token.clear();
                }
                result.push(ch);
            }
            _ => {
                current_token.push(ch);
            }
        }
    }

    if !current_token.is_empty() {
        result.push_str(&highlight_sexpr_token(&current_token));
    }

    result
}

fn highlight_token(token: &str, in_expression: bool) -> String {
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        return token.to_string();
    }

    // Keywords
    if matches!(token, "let" | "fn" | "if" | "else" | "while" | "for" | "return" | "match" | "struct" | "enum" | "impl" | "trait" | "use" | "mod" | "pub") {
        return token.purple().to_string();
    }

    // Operators in expressions
    if in_expression && matches!(token, "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | ">" | "<=" | ">=" | "&&" | "||" | "!" | "&" | "|" | "^" | "<<" | ">>" | "=" | "+=" | "-=" | "*=" | "/=") {
        return token.yellow().to_string();
    }

    // Numbers
    if token.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '_') {
        return token.cyan().to_string();
    }

    // Default
    token.to_string()
}

fn highlight_sexpr_token(token: &str) -> String {
    if !colored::control::SHOULD_COLORIZE.should_colorize() {
        return token.to_string();
    }

    // Keywords and operators
    if matches!(token, "let" | "fn" | "if" | "else" | "while" | "for" | "return" | "match" | "+" | "-" | "*" | "/" | "=" | "<" | ">") {
        return token.purple().to_string();
    }

    // Numbers
    if token.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '_') {
        return token.cyan().to_string();
    }

    // Default
    token.to_string()
}