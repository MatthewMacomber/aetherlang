// Aether Token System
// 32-bit token representation optimized for LLM generation

use std::fmt;

/// Token type encoding (4 bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TokenType {
    Keyword = 0,    // Language keywords (let, fn, if, etc.)
    OpCode = 1,     // Operators (+, -, *, /, etc.)
    Literal = 2,    // Numeric and string literals
    VarRef = 3,     // Variable references
    NodeRef = 4,    // AST node references
    Meta = 5,       // Metadata tokens (comments, annotations)
}

impl TokenType {
    /// Convert from 4-bit value
    pub fn from_bits(bits: u8) -> Option<Self> {
        match bits & 0xF {
            0 => Some(TokenType::Keyword),
            1 => Some(TokenType::OpCode),
            2 => Some(TokenType::Literal),
            3 => Some(TokenType::VarRef),
            4 => Some(TokenType::NodeRef),
            5 => Some(TokenType::Meta),
            _ => None,
        }
    }

    /// Convert to 4-bit value
    pub fn to_bits(self) -> u8 {
        self as u8
    }
}

/// 32-bit token structure
/// Layout: [Type: 4 bits][Payload: 28 bits]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Token {
    data: u32,
}

impl Token {
    /// Create a new token
    pub fn new(token_type: TokenType, payload: u32) -> Self {
        // Ensure payload fits in 28 bits
        let payload = payload & 0x0FFF_FFFF;
        let type_bits = (token_type.to_bits() as u32) << 28;
        
        Token {
            data: type_bits | payload,
        }
    }

    /// Get token type
    pub fn token_type(&self) -> TokenType {
        let type_bits = (self.data >> 28) as u8;
        TokenType::from_bits(type_bits).unwrap_or(TokenType::Meta)
    }

    /// Get payload (28 bits)
    pub fn payload(&self) -> u32 {
        self.data & 0x0FFF_FFFF
    }

    /// Get raw 32-bit data
    pub fn raw_data(&self) -> u32 {
        self.data
    }

    /// Create token from raw 32-bit data
    pub fn from_raw(data: u32) -> Self {
        Token { data }
    }

    /// Create keyword token
    pub fn keyword(keyword_id: u32) -> Self {
        Token::new(TokenType::Keyword, keyword_id)
    }

    /// Create operator token
    pub fn operator(op_id: u32) -> Self {
        Token::new(TokenType::OpCode, op_id)
    }

    /// Create literal token
    pub fn literal(literal_id: u32) -> Self {
        Token::new(TokenType::Literal, literal_id)
    }

    /// Create variable reference token
    pub fn var_ref(var_id: u32) -> Self {
        Token::new(TokenType::VarRef, var_id)
    }

    /// Create AST node reference token
    pub fn node_ref(node_id: u32) -> Self {
        Token::new(TokenType::NodeRef, node_id)
    }

    /// Create metadata token
    pub fn meta(meta_id: u32) -> Self {
        Token::new(TokenType::Meta, meta_id)
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Token({:?}, {})", self.token_type(), self.payload())
    }
}

/// Token sequence for representing complete programs
#[derive(Debug, Clone)]
pub struct TokenSequence {
    tokens: Vec<Token>,
}

impl TokenSequence {
    /// Create new empty token sequence
    pub fn new() -> Self {
        TokenSequence {
            tokens: Vec::new(),
        }
    }

    /// Create from vector of tokens
    pub fn from_tokens(tokens: Vec<Token>) -> Self {
        TokenSequence { tokens }
    }

    /// Add token to sequence
    pub fn push(&mut self, token: Token) {
        self.tokens.push(token);
    }

    /// Get token at index
    pub fn get(&self, index: usize) -> Option<&Token> {
        self.tokens.get(index)
    }

    /// Get length of sequence
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if sequence is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get iterator over tokens
    pub fn iter(&self) -> std::slice::Iter<Token> {
        self.tokens.iter()
    }

    /// Convert to vector
    pub fn into_vec(self) -> Vec<Token> {
        self.tokens
    }
}

impl Default for TokenSequence {
    fn default() -> Self {
        Self::new()
    }
}