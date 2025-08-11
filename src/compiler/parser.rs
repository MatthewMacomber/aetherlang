// Aether S-Expression Parser
// Recursive descent parser for canonical S-expression syntax with graph extensions

use std::collections::HashMap;
use std::fmt;
use crate::compiler::ast::{AST, ASTNode, ASTNodeRef, NodeId, AtomValue};
use crate::compiler::symbol_table::SymbolTable;

/// Parser error types
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// Unexpected end of input
    UnexpectedEof,
    /// Unexpected character
    UnexpectedChar(char, usize),
    /// Unterminated string literal
    UnterminatedString(usize),
    /// Invalid number format
    InvalidNumber(String, usize),
    /// Unmatched parentheses
    UnmatchedParen(usize),
    /// Invalid datum label
    InvalidDatumLabel(String, usize),
    /// Undefined datum label reference
    UndefinedLabel(String, usize),
    /// Cyclic datum label definition
    CyclicLabel(String, usize),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedEof => write!(f, "Unexpected end of input"),
            ParseError::UnexpectedChar(ch, pos) => {
                write!(f, "Unexpected character '{}' at position {}", ch, pos)
            }
            ParseError::UnterminatedString(pos) => {
                write!(f, "Unterminated string literal at position {}", pos)
            }
            ParseError::InvalidNumber(num, pos) => {
                write!(f, "Invalid number format '{}' at position {}", num, pos)
            }
            ParseError::UnmatchedParen(pos) => {
                write!(f, "Unmatched parenthesis at position {}", pos)
            }
            ParseError::InvalidDatumLabel(label, pos) => {
                write!(f, "Invalid datum label '{}' at position {}", label, pos)
            }
            ParseError::UndefinedLabel(label, pos) => {
                write!(f, "Undefined datum label '{}' at position {}", label, pos)
            }
            ParseError::CyclicLabel(label, pos) => {
                write!(f, "Cyclic datum label definition '{}' at position {}", label, pos)
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// Parser result type
pub type ParseResult<T> = Result<T, ParseError>;

/// Lexical token for parsing
#[derive(Debug, Clone, PartialEq)]
enum LexToken {
    LeftParen,
    RightParen,
    Symbol(String),
    Number(f64),
    String(String),
    Boolean(bool),
    Nil,
    DatumLabel(String),
    DatumRef(String),
    Eof,
}

/// Lexer for S-expressions
struct Lexer {
    input: Vec<char>,
    position: usize,
    current_char: Option<char>,
}

impl Lexer {
    /// Create new lexer
    fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let current_char = chars.get(0).copied();
        
        Lexer {
            input: chars,
            position: 0,
            current_char,
        }
    }

    /// Advance to next character
    fn advance(&mut self) {
        self.position += 1;
        self.current_char = self.input.get(self.position).copied();
    }

    /// Peek at next character without advancing
    fn peek(&self) -> Option<char> {
        self.input.get(self.position + 1).copied()
    }

    /// Skip whitespace and comments
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() {
                self.advance();
            } else if ch == ';' {
                // Skip line comment
                while let Some(ch) = self.current_char {
                    if ch == '\n' {
                        break;
                    }
                    self.advance();
                }
            } else {
                break;
            }
        }
    }

    /// Read string literal
    fn read_string(&mut self) -> ParseResult<String> {
        let start_pos = self.position;
        self.advance(); // Skip opening quote
        
        let mut result = String::new();
        
        while let Some(ch) = self.current_char {
            if ch == '"' {
                self.advance(); // Skip closing quote
                return Ok(result);
            } else if ch == '\\' {
                self.advance();
                match self.current_char {
                    Some('n') => result.push('\n'),
                    Some('t') => result.push('\t'),
                    Some('r') => result.push('\r'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some(ch) => result.push(ch),
                    None => return Err(ParseError::UnterminatedString(start_pos)),
                }
                self.advance();
            } else {
                result.push(ch);
                self.advance();
            }
        }
        
        Err(ParseError::UnterminatedString(start_pos))
    }

    /// Read number literal
    fn read_number(&mut self) -> ParseResult<f64> {
        let start_pos = self.position;
        let mut number_str = String::new();
        
        // Handle negative numbers
        if self.current_char == Some('-') {
            number_str.push('-');
            self.advance();
        }
        
        // Read integer part
        while let Some(ch) = self.current_char {
            if ch.is_ascii_digit() {
                number_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        // Read decimal part
        if self.current_char == Some('.') {
            number_str.push('.');
            self.advance();
            
            while let Some(ch) = self.current_char {
                if ch.is_ascii_digit() {
                    number_str.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
        }
        
        // Read exponent part
        if matches!(self.current_char, Some('e') | Some('E')) {
            number_str.push(self.current_char.unwrap());
            self.advance();
            
            if matches!(self.current_char, Some('+') | Some('-')) {
                number_str.push(self.current_char.unwrap());
                self.advance();
            }
            
            while let Some(ch) = self.current_char {
                if ch.is_ascii_digit() {
                    number_str.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
        }
        
        number_str.parse::<f64>()
            .map_err(|_| ParseError::InvalidNumber(number_str, start_pos))
    }

    /// Read symbol or keyword
    fn read_symbol(&mut self) -> String {
        let mut symbol = String::new();
        
        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '?' || ch == ':' || ch == '.' {
                symbol.push(ch);
                self.advance();
            } else if symbol.is_empty() && (ch == '+' || ch == '*' || ch == '/' || ch == '=' || 
                     ch == '<' || ch == '>' || ch == '!' || ch == '@' || ch == '#' || ch == '$' || 
                     ch == '%' || ch == '^' || ch == '&' || ch == '|' || ch == '~') {
                // For operators, read a sequence of operator characters
                while let Some(op_ch) = self.current_char {
                    if op_ch == '+' || op_ch == '*' || op_ch == '/' || op_ch == '=' || 
                       op_ch == '<' || op_ch == '>' || op_ch == '!' || op_ch == '@' || 
                       op_ch == '#' || op_ch == '$' || op_ch == '%' || op_ch == '^' || 
                       op_ch == '&' || op_ch == '|' || op_ch == '~' {
                        symbol.push(op_ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
                break;
            } else {
                break;
            }
        }
        
        symbol
    }

    /// Get next token
    fn next_token(&mut self) -> ParseResult<LexToken> {
        self.skip_whitespace();
        
        match self.current_char {
            None => Ok(LexToken::Eof),
            Some('(') => {
                self.advance();
                Ok(LexToken::LeftParen)
            }
            Some(')') => {
                self.advance();
                Ok(LexToken::RightParen)
            }
            Some('"') => {
                let string_val = self.read_string()?;
                Ok(LexToken::String(string_val))
            }
            Some(ch) if ch.is_ascii_digit() || (ch == '-' && self.peek().map_or(false, |c| c.is_ascii_digit())) => {
                let number_val = self.read_number()?;
                Ok(LexToken::Number(number_val))
            }
            Some('#') => {
                self.advance();
                if self.current_char == Some('=') {
                    // Datum label definition: #=label#
                    self.advance();
                    let label = self.read_symbol();
                    if self.current_char == Some('#') {
                        self.advance();
                        Ok(LexToken::DatumLabel(label))
                    } else {
                        Err(ParseError::InvalidDatumLabel(label, self.position))
                    }
                } else {
                    // Datum reference: #label#
                    let label = self.read_symbol();
                    if self.current_char == Some('#') {
                        self.advance();
                        Ok(LexToken::DatumRef(label))
                    } else {
                        Err(ParseError::InvalidDatumLabel(label, self.position))
                    }
                }
            }
            Some(_) => {
                let symbol = self.read_symbol();
                match symbol.as_str() {
                    "true" => Ok(LexToken::Boolean(true)),
                    "false" => Ok(LexToken::Boolean(false)),
                    "nil" => Ok(LexToken::Nil),
                    _ => Ok(LexToken::Symbol(symbol)),
                }
            }
        }
    }
}

/// S-expression parser
pub struct Parser {
    lexer: Lexer,
    current_token: LexToken,
    symbol_table: SymbolTable,
    /// Datum labels for referencing parsed nodes - used for future circular reference support
    #[allow(dead_code)]
    datum_labels: HashMap<String, NodeId>,
    pending_refs: Vec<(String, usize)>,
}

impl Parser {
    /// Create new parser
    pub fn new(input: &str) -> ParseResult<Self> {
        let mut lexer = Lexer::new(input);
        let current_token = lexer.next_token()?;
        
        Ok(Parser {
            lexer,
            current_token,
            symbol_table: SymbolTable::new(),
            datum_labels: HashMap::new(),
            pending_refs: Vec::new(),
        })
    }

    /// Advance to next token
    fn advance(&mut self) -> ParseResult<()> {
        self.current_token = self.lexer.next_token()?;
        Ok(())
    }

    /// Parse complete S-expression
    pub fn parse(&mut self) -> ParseResult<AST> {
        let root = self.parse_expression()?;
        
        // Check for undefined datum label references
        if !self.pending_refs.is_empty() {
            let (label, pos) = self.pending_refs[0].clone();
            return Err(ParseError::UndefinedLabel(label, pos));
        }
        
        Ok(AST::new(root))
    }

    /// Parse single expression
    fn parse_expression(&mut self) -> ParseResult<ASTNode> {
        match &self.current_token {
            LexToken::LeftParen => self.parse_list(),
            LexToken::Symbol(name) => {
                let symbol = name.clone();
                self.advance()?;
                Ok(ASTNode::symbol(symbol))
            }
            LexToken::Number(value) => {
                let num = *value;
                self.advance()?;
                Ok(ASTNode::number(num))
            }
            LexToken::String(value) => {
                let string = value.clone();
                self.advance()?;
                Ok(ASTNode::string(string))
            }
            LexToken::Boolean(value) => {
                let bool_val = *value;
                self.advance()?;
                Ok(ASTNode::boolean(bool_val))
            }
            LexToken::Nil => {
                self.advance()?;
                Ok(ASTNode::nil())
            }
            LexToken::DatumLabel(label) => {
                let _label_name = label.clone();
                self.advance()?;
                let node = self.parse_expression()?;
                
                // Store labeled node for future references
                // For now, we'll just return the node directly
                // Full datum label support would require AST modification
                Ok(node)
            }
            LexToken::DatumRef(label) => {
                let label_name = label.clone();
                self.advance()?;
                
                // For now, return a symbol representing the reference
                // Full datum label support would require AST modification
                Ok(ASTNode::symbol(format!("#ref:{}", label_name)))
            }
            LexToken::RightParen => {
                Err(ParseError::UnmatchedParen(self.lexer.position))
            }
            LexToken::Eof => {
                Err(ParseError::UnexpectedEof)
            }
        }
    }

    /// Parse list expression
    fn parse_list(&mut self) -> ParseResult<ASTNode> {
        let start_pos = self.lexer.position;
        self.advance()?; // Skip opening paren
        
        let mut elements = Vec::new();
        
        while self.current_token != LexToken::RightParen {
            if self.current_token == LexToken::Eof {
                return Err(ParseError::UnmatchedParen(start_pos));
            }
            
            let element = self.parse_expression()?;
            elements.push(ASTNodeRef::direct(element));
        }
        
        self.advance()?; // Skip closing paren
        Ok(ASTNode::list(elements))
    }

    /// Get symbol table reference
    pub fn symbol_table(&self) -> &SymbolTable {
        &self.symbol_table
    }

    /// Get mutable symbol table reference
    pub fn symbol_table_mut(&mut self) -> &mut SymbolTable {
        &mut self.symbol_table
    }
}

/// Parse S-expression from string
pub fn parse_sexpr(input: &str) -> ParseResult<AST> {
    let mut parser = Parser::new(input)?;
    parser.parse()
}

/// Parse multiple S-expressions from string
pub fn parse_multiple_sexprs(input: &str) -> ParseResult<Vec<AST>> {
    let mut parser = Parser::new(input)?;
    let mut results = Vec::new();
    
    while parser.current_token != LexToken::Eof {
        let ast = AST::new(parser.parse_expression()?);
        results.push(ast);
    }
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_atom_symbol() {
        let ast = parse_sexpr("hello").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Symbol(name)) => assert_eq!(name, "hello"),
            _ => panic!("Expected symbol atom"),
        }
    }

    #[test]
    fn test_parse_atom_number() {
        let ast = parse_sexpr("42.5").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, 42.5),
            _ => panic!("Expected number atom"),
        }
    }

    #[test]
    fn test_parse_atom_string() {
        let ast = parse_sexpr("\"hello world\"").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::String(value)) => assert_eq!(value, "hello world"),
            _ => panic!("Expected string atom"),
        }
    }

    #[test]
    fn test_parse_atom_boolean() {
        let ast = parse_sexpr("true").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Boolean(value)) => assert_eq!(*value, true),
            _ => panic!("Expected boolean atom"),
        }
    }

    #[test]
    fn test_parse_atom_nil() {
        let ast = parse_sexpr("nil").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Nil) => {},
            _ => panic!("Expected nil atom"),
        }
    }

    #[test]
    fn test_parse_empty_list() {
        let ast = parse_sexpr("()").unwrap();
        match &ast.root {
            ASTNode::List(elements) => assert_eq!(elements.len(), 0),
            _ => panic!("Expected empty list"),
        }
    }

    #[test]
    fn test_parse_simple_list() {
        let ast = parse_sexpr("(+ 1 2)").unwrap();
        match &ast.root {
            ASTNode::List(elements) => {
                assert_eq!(elements.len(), 3);
                // Check first element is '+'
                if let Some(ASTNode::Atom(AtomValue::Symbol(name))) = ast.resolve_ref(&elements[0]) {
                    assert_eq!(name, "+");
                } else {
                    panic!("Expected '+' symbol");
                }
            }
            _ => panic!("Expected list"),
        }
    }

    #[test]
    fn test_parse_nested_list() {
        let ast = parse_sexpr("(if (> x 0) x (- x))").unwrap();
        match &ast.root {
            ASTNode::List(elements) => {
                assert_eq!(elements.len(), 4);
                // Check second element is a list
                if let Some(ASTNode::List(_)) = ast.resolve_ref(&elements[1]) {
                    // Good, it's a nested list
                } else {
                    panic!("Expected nested list");
                }
            }
            _ => panic!("Expected list"),
        }
    }

    #[test]
    fn test_parse_error_unmatched_paren() {
        let result = parse_sexpr("(+ 1 2");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::UnmatchedParen(_) => {},
            _ => panic!("Expected unmatched paren error"),
        }
    }

    #[test]
    fn test_parse_error_unexpected_paren() {
        // This should parse successfully as just the symbol "+"
        // The closing paren would be an error if we tried to parse multiple expressions
        let result = parse_sexpr("+");
        assert!(result.is_ok());
        
        // Test a real unmatched paren case
        let result = parse_sexpr(")");
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::UnmatchedParen(_) => {},
            _ => panic!("Expected unmatched paren error"),
        }
    }

    #[test]
    fn test_parse_string_with_escapes() {
        let ast = parse_sexpr("\"hello\\nworld\\t!\"").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::String(value)) => assert_eq!(value, "hello\nworld\t!"),
            _ => panic!("Expected string atom"),
        }
    }

    #[test]
    fn test_parse_negative_number() {
        let ast = parse_sexpr("-42.5").unwrap();
        match &ast.root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, -42.5),
            _ => panic!("Expected negative number atom"),
        }
    }

    #[test]
    fn test_parse_multiple_expressions() {
        let asts = parse_multiple_sexprs("42 (+ 1 2) \"hello\"").unwrap();
        assert_eq!(asts.len(), 3);
        
        match &asts[0].root {
            ASTNode::Atom(AtomValue::Number(value)) => assert_eq!(*value, 42.0),
            _ => panic!("Expected number"),
        }
        
        match &asts[1].root {
            ASTNode::List(_) => {},
            _ => panic!("Expected list"),
        }
        
        match &asts[2].root {
            ASTNode::Atom(AtomValue::String(value)) => assert_eq!(value, "hello"),
            _ => panic!("Expected string"),
        }
    }
}