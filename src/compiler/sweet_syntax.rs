// Aether Sweet Syntax Transpiler
// Bidirectional transpiler between human-readable "sweet" syntax and S-expressions

use std::fmt;
use crate::compiler::ast::{AST, ASTNode, ASTNodeRef, AtomValue};
use crate::compiler::parser::ParseError;

/// Sweet syntax parsing errors
#[derive(Debug, Clone, PartialEq)]
pub enum SweetSyntaxError {
    /// Indentation error
    IndentationError(String, usize),
    /// Invalid infix expression
    InvalidInfixExpression(String, usize),
    /// Unmatched delimiters
    UnmatchedDelimiters(char, usize),
    /// Invalid function call syntax
    InvalidFunctionCall(String, usize),
    /// Parse error from underlying S-expression parser
    ParseError(ParseError),
}

impl fmt::Display for SweetSyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SweetSyntaxError::IndentationError(msg, pos) => {
                write!(f, "Indentation error: {} at position {}", msg, pos)
            }
            SweetSyntaxError::InvalidInfixExpression(expr, pos) => {
                write!(f, "Invalid infix expression '{}' at position {}", expr, pos)
            }
            SweetSyntaxError::UnmatchedDelimiters(delim, pos) => {
                write!(f, "Unmatched delimiter '{}' at position {}", delim, pos)
            }
            SweetSyntaxError::InvalidFunctionCall(call, pos) => {
                write!(f, "Invalid function call '{}' at position {}", call, pos)
            }
            SweetSyntaxError::ParseError(err) => write!(f, "Parse error: {}", err),
        }
    }
}

impl std::error::Error for SweetSyntaxError {}

impl From<ParseError> for SweetSyntaxError {
    fn from(err: ParseError) -> Self {
        SweetSyntaxError::ParseError(err)
    }
}

/// Sweet syntax result type
pub type SweetResult<T> = Result<T, SweetSyntaxError>;

/// Operator precedence levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Precedence {
    Lowest = 0,
    Assignment = 1,    // =, +=, -=, etc.
    LogicalOr = 2,     // ||
    LogicalAnd = 3,    // &&
    Equality = 4,      // ==, !=
    Comparison = 5,    // <, >, <=, >=
    BitwiseOr = 6,     // |
    BitwiseXor = 7,    // ^
    BitwiseAnd = 8,    // &
    Shift = 9,         // <<, >>
    Addition = 10,     // +, -
    Multiplication = 11, // *, /, %
    Unary = 12,        // !, -, +, ~
    Call = 13,         // function calls, array access
    Primary = 14,      // literals, identifiers, parentheses
}

/// Get operator precedence
fn get_precedence(op: &str) -> Precedence {
    match op {
        "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>=" => Precedence::Assignment,
        "||" => Precedence::LogicalOr,
        "&&" => Precedence::LogicalAnd,
        "==" | "!=" => Precedence::Equality,
        "<" | ">" | "<=" | ">=" => Precedence::Comparison,
        "|" => Precedence::BitwiseOr,
        "^" => Precedence::BitwiseXor,
        "&" => Precedence::BitwiseAnd,
        "<<" | ">>" => Precedence::Shift,
        "+" | "-" => Precedence::Addition,
        "*" | "/" | "%" => Precedence::Multiplication,
        _ => Precedence::Lowest,
    }
}

/// Check if operator is right-associative
fn is_right_associative(op: &str) -> bool {
    matches!(op, "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>=")
}

/// Lexical token for sweet syntax
#[derive(Debug, Clone, PartialEq)]
enum SweetToken {
    // Literals and identifiers
    Identifier(String),
    Number(f64),
    String(String),
    Boolean(bool),
    Nil,
    
    // Operators
    Operator(String),
    
    // Delimiters
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Comma,
    Colon,
    Semicolon,
    
    // Special tokens
    InfixStart,  // {
    InfixEnd,    // }
    Newline,
    Indent(usize),
    Dedent(usize),
    Eof,
}

/// Sweet syntax lexer
struct SweetLexer {
    input: Vec<char>,
    position: usize,
    current_char: Option<char>,
    line: usize,
    column: usize,
    indent_stack: Vec<usize>,
}

impl SweetLexer {
    /// Create new sweet syntax lexer
    fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let current_char = chars.get(0).copied();
        
        SweetLexer {
            input: chars,
            position: 0,
            current_char,
            line: 1,
            column: 1,
            indent_stack: vec![0],
        }
    }

    /// Advance to next character
    fn advance(&mut self) {
        if self.current_char == Some('\n') {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        
        self.position += 1;
        self.current_char = self.input.get(self.position).copied();
    }

    /// Peek at next character
    fn peek(&self) -> Option<char> {
        self.input.get(self.position + 1).copied()
    }

    /// Skip whitespace (but not newlines)
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch == ' ' || ch == '\t' || ch == '\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Handle indentation at start of line
    fn handle_indentation(&mut self) -> Vec<SweetToken> {
        let mut tokens = Vec::new();
        let mut indent_level = 0;
        
        // Count indentation
        while let Some(ch) = self.current_char {
            if ch == ' ' {
                indent_level += 1;
                self.advance();
            } else if ch == '\t' {
                indent_level += 4; // Treat tab as 4 spaces
                self.advance();
            } else {
                break;
            }
        }
        
        let current_indent = *self.indent_stack.last().unwrap();
        
        if indent_level > current_indent {
            // Increased indentation
            self.indent_stack.push(indent_level);
            tokens.push(SweetToken::Indent(indent_level));
        } else if indent_level < current_indent {
            // Decreased indentation - may need multiple dedents
            while let Some(&stack_indent) = self.indent_stack.last() {
                if stack_indent <= indent_level {
                    break;
                }
                self.indent_stack.pop();
                tokens.push(SweetToken::Dedent(stack_indent));
            }
        }
        
        tokens
    }

    /// Read string literal
    fn read_string(&mut self) -> SweetResult<String> {
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
                    None => return Err(SweetSyntaxError::UnmatchedDelimiters('"', start_pos)),
                }
                self.advance();
            } else {
                result.push(ch);
                self.advance();
            }
        }
        
        Err(SweetSyntaxError::UnmatchedDelimiters('"', start_pos))
    }

    /// Read number literal
    fn read_number(&mut self) -> SweetResult<f64> {
        let start_pos = self.position;
        let mut number_str = String::new();
        
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
        if self.current_char == Some('.') && self.peek().map_or(false, |c| c.is_ascii_digit()) {
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
            .map_err(|_| SweetSyntaxError::InvalidInfixExpression(number_str, start_pos))
    }

    /// Read identifier or keyword
    fn read_identifier(&mut self) -> String {
        let mut identifier = String::new();
        
        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' {
                identifier.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        identifier
    }

    /// Read operator
    fn read_operator(&mut self) -> String {
        let mut operator = String::new();
        
        // Multi-character operators
        let two_char_ops = [
            "==", "!=", "<=", ">=", "&&", "||", "<<", ">>",
            "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=",
            "->", "=>", "::", "..", "...",
        ];
        
        // Check for two-character operators first
        if let (Some(ch1), Some(ch2)) = (self.current_char, self.peek()) {
            let two_char = format!("{}{}", ch1, ch2);
            if two_char_ops.contains(&two_char.as_str()) {
                operator.push(ch1);
                self.advance();
                operator.push(ch2);
                self.advance();
                return operator;
            }
        }
        
        // Single character operators
        if let Some(ch) = self.current_char {
            if "+-*/%=<>!&|^~@#$".contains(ch) {
                operator.push(ch);
                self.advance();
            }
        }
        
        operator
    }

    /// Get next token
    fn next_token(&mut self) -> SweetResult<SweetToken> {
        // Handle indentation at start of line
        if self.column == 1 && self.current_char.is_some() && self.current_char != Some('\n') {
            let indent_tokens = self.handle_indentation();
            if !indent_tokens.is_empty() {
                // For simplicity, return the first indent/dedent token
                // In a full implementation, we'd need to queue multiple tokens
                return Ok(indent_tokens[0].clone());
            }
        }
        
        self.skip_whitespace();
        
        match self.current_char {
            None => Ok(SweetToken::Eof),
            Some('\n') => {
                self.advance();
                Ok(SweetToken::Newline)
            }
            Some('(') => {
                self.advance();
                Ok(SweetToken::LeftParen)
            }
            Some(')') => {
                self.advance();
                Ok(SweetToken::RightParen)
            }
            Some('{') => {
                self.advance();
                Ok(SweetToken::InfixStart)
            }
            Some('}') => {
                self.advance();
                Ok(SweetToken::InfixEnd)
            }
            Some('[') => {
                self.advance();
                Ok(SweetToken::LeftBracket)
            }
            Some(']') => {
                self.advance();
                Ok(SweetToken::RightBracket)
            }
            Some(',') => {
                self.advance();
                Ok(SweetToken::Comma)
            }
            Some(':') => {
                self.advance();
                Ok(SweetToken::Colon)
            }
            Some(';') => {
                self.advance();
                Ok(SweetToken::Semicolon)
            }
            Some('"') => {
                let string_val = self.read_string()?;
                Ok(SweetToken::String(string_val))
            }
            Some(ch) if ch.is_ascii_digit() => {
                let number_val = self.read_number()?;
                Ok(SweetToken::Number(number_val))
            }
            Some(ch) if ch.is_alphabetic() || ch == '_' => {
                let identifier = self.read_identifier();
                match identifier.as_str() {
                    "true" => Ok(SweetToken::Boolean(true)),
                    "false" => Ok(SweetToken::Boolean(false)),
                    "nil" => Ok(SweetToken::Nil),
                    _ => Ok(SweetToken::Identifier(identifier)),
                }
            }
            Some(_) => {
                let operator = self.read_operator();
                if operator.is_empty() {
                    let ch = self.current_char.unwrap();
                    self.advance();
                    Err(SweetSyntaxError::InvalidInfixExpression(ch.to_string(), self.position - 1))
                } else {
                    Ok(SweetToken::Operator(operator))
                }
            }
        }
    }
}

/// Sweet syntax transpiler
pub struct SweetSyntaxTranspiler {
    lexer: SweetLexer,
    current_token: SweetToken,
}

impl SweetSyntaxTranspiler {
    /// Create new transpiler
    pub fn new(input: &str) -> SweetResult<Self> {
        let mut lexer = SweetLexer::new(input);
        let current_token = lexer.next_token()?;
        
        Ok(SweetSyntaxTranspiler {
            lexer,
            current_token,
        })
    }

    /// Advance to next token
    fn advance(&mut self) -> SweetResult<()> {
        self.current_token = self.lexer.next_token()?;
        Ok(())
    }

    /// Parse sweet syntax to S-expression AST
    pub fn parse_to_sexpr(&mut self) -> SweetResult<AST> {
        let root = self.parse_expression()?;
        Ok(AST::new(root))
    }

    /// Parse expression with precedence climbing
    fn parse_expression(&mut self) -> SweetResult<ASTNode> {
        self.parse_expression_with_precedence(Precedence::Lowest)
    }

    /// Parse expression with given minimum precedence
    fn parse_expression_with_precedence(&mut self, min_precedence: Precedence) -> SweetResult<ASTNode> {
        let mut left = self.parse_primary()?;

        while let SweetToken::Operator(ref op) = self.current_token {
            let precedence = get_precedence(op);
            if precedence < min_precedence {
                break;
            }

            let operator = op.clone();
            self.advance()?;

            let next_min_precedence = if is_right_associative(&operator) {
                precedence
            } else {
                Precedence::from(precedence as u8 + 1)
            };

            let right = self.parse_expression_with_precedence(next_min_precedence)?;

            // Convert infix to S-expression: (op left right)
            left = ASTNode::list(vec![
                ASTNodeRef::direct(ASTNode::symbol(operator)),
                ASTNodeRef::direct(left),
                ASTNodeRef::direct(right),
            ]);
        }

        Ok(left)
    }

    /// Parse primary expression
    fn parse_primary(&mut self) -> SweetResult<ASTNode> {
        match &self.current_token.clone() {
            SweetToken::Identifier(name) => {
                let identifier = name.clone();
                self.advance()?;
                
                // Check for function call syntax: identifier(args...)
                if self.current_token == SweetToken::LeftParen {
                    self.parse_function_call(identifier)
                } else {
                    Ok(ASTNode::symbol(identifier))
                }
            }
            SweetToken::Number(value) => {
                let num = *value;
                self.advance()?;
                Ok(ASTNode::number(num))
            }
            SweetToken::String(value) => {
                let string = value.clone();
                self.advance()?;
                Ok(ASTNode::string(string))
            }
            SweetToken::Boolean(value) => {
                let bool_val = *value;
                self.advance()?;
                Ok(ASTNode::boolean(bool_val))
            }
            SweetToken::Nil => {
                self.advance()?;
                Ok(ASTNode::nil())
            }
            SweetToken::LeftParen => {
                self.advance()?; // Skip '('
                let expr = self.parse_expression()?;
                if self.current_token != SweetToken::RightParen {
                    return Err(SweetSyntaxError::UnmatchedDelimiters('(', self.lexer.position));
                }
                self.advance()?; // Skip ')'
                Ok(expr)
            }
            SweetToken::InfixStart => {
                // Parse infix expression: {x + y * z}
                self.advance()?; // Skip '{'
                let expr = self.parse_expression()?;
                if self.current_token != SweetToken::InfixEnd {
                    return Err(SweetSyntaxError::UnmatchedDelimiters('{', self.lexer.position));
                }
                self.advance()?; // Skip '}'
                Ok(expr)
            }
            SweetToken::Operator(op) if matches!(op.as_str(), "!" | "-" | "~") => {
                // Unary operators (note: + is not included as unary in this context)
                let operator = op.clone();
                self.advance()?;
                let operand = self.parse_expression_with_precedence(Precedence::Unary)?;
                Ok(ASTNode::list(vec![
                    ASTNodeRef::direct(ASTNode::symbol(operator)),
                    ASTNodeRef::direct(operand),
                ]))
            }
            _ => Err(SweetSyntaxError::InvalidInfixExpression(
                format!("{:?}", self.current_token),
                self.lexer.position,
            )),
        }
    }

    /// Parse function call: function_name(arg1, arg2, ...)
    fn parse_function_call(&mut self, function_name: String) -> SweetResult<ASTNode> {
        self.advance()?; // Skip '('
        
        let mut args = vec![ASTNodeRef::direct(ASTNode::symbol(function_name))];
        
        // Parse arguments
        while self.current_token != SweetToken::RightParen {
            if self.current_token == SweetToken::Eof {
                return Err(SweetSyntaxError::UnmatchedDelimiters('(', self.lexer.position));
            }
            
            let arg = self.parse_expression()?;
            args.push(ASTNodeRef::direct(arg));
            
            if self.current_token == SweetToken::Comma {
                self.advance()?; // Skip ','
            } else if self.current_token != SweetToken::RightParen {
                return Err(SweetSyntaxError::InvalidFunctionCall(
                    format!("Expected ',' or ')' in function call"),
                    self.lexer.position,
                ));
            }
        }
        
        self.advance()?; // Skip ')'
        Ok(ASTNode::list(args))
    }
}

impl From<u8> for Precedence {
    fn from(value: u8) -> Self {
        match value {
            0 => Precedence::Lowest,
            1 => Precedence::Assignment,
            2 => Precedence::LogicalOr,
            3 => Precedence::LogicalAnd,
            4 => Precedence::Equality,
            5 => Precedence::Comparison,
            6 => Precedence::BitwiseOr,
            7 => Precedence::BitwiseXor,
            8 => Precedence::BitwiseAnd,
            9 => Precedence::Shift,
            10 => Precedence::Addition,
            11 => Precedence::Multiplication,
            12 => Precedence::Unary,
            13 => Precedence::Call,
            14 => Precedence::Primary,
            _ => Precedence::Lowest,
        }
    }
}

/// Convert S-expression AST back to sweet syntax
pub fn sexpr_to_sweet(ast: &AST) -> String {
    ast_node_to_sweet(&ast.root, 0)
}

/// Convert AST node to sweet syntax string
fn ast_node_to_sweet(node: &ASTNode, indent_level: usize) -> String {
    match node {
        ASTNode::Atom(atom) => atom_to_sweet(atom),
        ASTNode::List(elements) if elements.is_empty() => "()".to_string(),
        ASTNode::List(elements) => {
            // Check if this is a unary expression
            if elements.len() == 2 {
                if let ASTNodeRef::Direct(boxed_node) = &elements[0] {
                    if let ASTNode::Atom(AtomValue::Symbol(op)) = boxed_node.as_ref() {
                        if is_unary_operator(op) {
                            let operand = ast_node_ref_to_sweet(&elements[1], indent_level);
                            return format!("({}{})", op, operand);
                        }
                    }
                }
            }
            
            // Check if this is an infix expression
            if elements.len() == 3 {
                if let ASTNodeRef::Direct(boxed_node) = &elements[0] {
                    if let ASTNode::Atom(AtomValue::Symbol(op)) = boxed_node.as_ref() {
                        if is_infix_operator(op) {
                            let left = ast_node_ref_to_sweet(&elements[1], indent_level);
                            let right = ast_node_ref_to_sweet(&elements[2], indent_level);
                            return format!("{{{} {} {}}}", left, op, right);
                        }
                    }
                }
            }
            
            // Check if this is a function call
            if !elements.is_empty() {
                if let ASTNodeRef::Direct(boxed_node) = &elements[0] {
                    if let ASTNode::Atom(AtomValue::Symbol(func_name)) = boxed_node.as_ref() {
                        if !is_special_form(func_name) {
                            let args: Vec<String> = elements[1..]
                                .iter()
                                .map(|arg| ast_node_ref_to_sweet(arg, indent_level))
                                .collect();
                            return format!("{}({})", func_name, args.join(", "));
                        }
                    }
                }
            }
            
            // Default S-expression format
            let elements_str: Vec<String> = elements
                .iter()
                .map(|elem| ast_node_ref_to_sweet(elem, indent_level))
                .collect();
            format!("({})", elements_str.join(" "))
        }
        ASTNode::Graph { .. } => {
            // For now, just use S-expression format for graphs
            format!("{}", node)
        }
    }
}

/// Convert AST node reference to sweet syntax
fn ast_node_ref_to_sweet(node_ref: &ASTNodeRef, indent_level: usize) -> String {
    match node_ref {
        ASTNodeRef::Direct(node) => ast_node_to_sweet(node, indent_level),
        ASTNodeRef::Id(id) => format!("#ref({})", id),
        ASTNodeRef::Label(label) => format!("#label({})", label),
    }
}

/// Convert atom value to sweet syntax
fn atom_to_sweet(atom: &AtomValue) -> String {
    match atom {
        AtomValue::Symbol(name) => name.clone(),
        AtomValue::Number(value) => {
            if value.fract() == 0.0 {
                format!("{}", *value as i64)
            } else {
                format!("{}", value)
            }
        }
        AtomValue::String(value) => format!("\"{}\"", value),
        AtomValue::Boolean(value) => value.to_string(),
        AtomValue::Nil => "nil".to_string(),
        AtomValue::Token(token) => format!("#token({})", token),
    }
}

/// Check if operator should be displayed in infix notation
fn is_infix_operator(op: &str) -> bool {
    matches!(op, 
        "+" | "-" | "*" | "/" | "%" | 
        "==" | "!=" | "<" | ">" | "<=" | ">=" |
        "&&" | "||" | "&" | "|" | "^" |
        "<<" | ">>" | "=" | "+=" | "-=" | "*=" | "/=" | "%="
    )
}

/// Check if operator is a unary operator
fn is_unary_operator(op: &str) -> bool {
    matches!(op, "!" | "-" | "~")
}

/// Check if symbol is a special form (should not be converted to function call syntax)
fn is_special_form(symbol: &str) -> bool {
    matches!(symbol,
        "if" | "let" | "fn" | "while" | "for" | "match" | "return" |
        "type" | "struct" | "enum" | "impl" | "pub" | "mod" | "use"
    )
}

/// Parse sweet syntax to S-expression string
pub fn sweet_to_sexpr(input: &str) -> SweetResult<String> {
    let mut transpiler = SweetSyntaxTranspiler::new(input)?;
    let ast = transpiler.parse_to_sexpr()?;
    Ok(format!("{}", ast.root))
}

/// Parse S-expression to sweet syntax string
pub fn sexpr_to_sweet_string(input: &str) -> SweetResult<String> {
    let ast = crate::compiler::parser::parse_sexpr(input)?;
    Ok(sexpr_to_sweet(&ast))
}