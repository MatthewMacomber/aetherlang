// Token System Unit Tests

use aether_language::{Token, TokenType, TokenSequence, SymbolTable, SymbolType};

#[cfg(test)]
mod token_tests {
    use super::*;

    #[test]
    fn test_token_type_encoding() {
        // Test all token types can be encoded and decoded
        let types = [
            TokenType::Keyword,
            TokenType::OpCode,
            TokenType::Literal,
            TokenType::VarRef,
            TokenType::NodeRef,
            TokenType::Meta,
        ];

        for token_type in &types {
            let bits = token_type.to_bits();
            let decoded = TokenType::from_bits(bits).unwrap();
            assert_eq!(*token_type, decoded);
        }
    }

    #[test]
    fn test_token_creation() {
        let token = Token::new(TokenType::Keyword, 42);
        assert_eq!(token.token_type(), TokenType::Keyword);
        assert_eq!(token.payload(), 42);
    }

    #[test]
    fn test_token_payload_limits() {
        // Test maximum payload (28 bits = 0x0FFF_FFFF)
        let max_payload = 0x0FFF_FFFF;
        let token = Token::new(TokenType::Literal, max_payload);
        assert_eq!(token.payload(), max_payload);

        // Test payload overflow is masked
        let overflow_payload = 0x1FFF_FFFF; // 29 bits
        let token = Token::new(TokenType::Literal, overflow_payload);
        assert_eq!(token.payload(), max_payload); // Should be masked to 28 bits
    }

    #[test]
    fn test_token_convenience_constructors() {
        let keyword = Token::keyword(1);
        assert_eq!(keyword.token_type(), TokenType::Keyword);
        assert_eq!(keyword.payload(), 1);

        let operator = Token::operator(2);
        assert_eq!(operator.token_type(), TokenType::OpCode);
        assert_eq!(operator.payload(), 2);

        let literal = Token::literal(3);
        assert_eq!(literal.token_type(), TokenType::Literal);
        assert_eq!(literal.payload(), 3);

        let var_ref = Token::var_ref(4);
        assert_eq!(var_ref.token_type(), TokenType::VarRef);
        assert_eq!(var_ref.payload(), 4);

        let node_ref = Token::node_ref(5);
        assert_eq!(node_ref.token_type(), TokenType::NodeRef);
        assert_eq!(node_ref.payload(), 5);

        let meta = Token::meta(6);
        assert_eq!(meta.token_type(), TokenType::Meta);
        assert_eq!(meta.payload(), 6);
    }

    #[test]
    fn test_token_raw_data() {
        let token = Token::new(TokenType::Keyword, 42);
        let raw = token.raw_data();
        let reconstructed = Token::from_raw(raw);
        
        assert_eq!(token.token_type(), reconstructed.token_type());
        assert_eq!(token.payload(), reconstructed.payload());
    }

    #[test]
    fn test_token_sequence() {
        let mut sequence = TokenSequence::new();
        assert!(sequence.is_empty());
        assert_eq!(sequence.len(), 0);

        sequence.push(Token::keyword(1));
        sequence.push(Token::literal(42));
        sequence.push(Token::operator(3));

        assert!(!sequence.is_empty());
        assert_eq!(sequence.len(), 3);

        assert_eq!(sequence.get(0).unwrap().token_type(), TokenType::Keyword);
        assert_eq!(sequence.get(1).unwrap().payload(), 42);
        assert_eq!(sequence.get(2).unwrap().token_type(), TokenType::OpCode);

        // Test iterator
        let types: Vec<TokenType> = sequence.iter()
            .map(|token| token.token_type())
            .collect();
        assert_eq!(types, vec![TokenType::Keyword, TokenType::Literal, TokenType::OpCode]);
    }

    #[test]
    fn test_token_sequence_from_tokens() {
        let tokens = vec![
            Token::keyword(1),
            Token::literal(42),
            Token::operator(3),
        ];
        
        let sequence = TokenSequence::from_tokens(tokens.clone());
        assert_eq!(sequence.len(), 3);
        
        let reconstructed = sequence.into_vec();
        assert_eq!(reconstructed, tokens);
    }
}

#[cfg(test)]
mod symbol_table_tests {
    use super::*;

    #[test]
    fn test_symbol_table_creation() {
        let table = SymbolTable::new();
        assert!(!table.is_empty()); // Should have builtins
        assert!(table.len() > 0);
    }

    #[test]
    fn test_builtin_symbols() {
        let table = SymbolTable::new();
        
        // Test keyword lookup
        let let_symbol = table.lookup("let").unwrap();
        assert_eq!(let_symbol.name, "let");
        assert_eq!(let_symbol.symbol_type, SymbolType::Keyword);

        // Test operator lookup
        let plus_symbol = table.lookup("+").unwrap();
        assert_eq!(plus_symbol.name, "+");
        assert_eq!(plus_symbol.symbol_type, SymbolType::Operator);
    }

    #[test]
    fn test_symbol_addition() {
        let mut table = SymbolTable::new();
        
        let var_id = table.add_symbol("my_var".to_string(), SymbolType::Variable);
        let func_id = table.add_symbol("my_func".to_string(), SymbolType::Function);
        
        assert_ne!(var_id, func_id);
        
        let var_symbol = table.get_symbol(var_id).unwrap();
        assert_eq!(var_symbol.name, "my_var");
        assert_eq!(var_symbol.symbol_type, SymbolType::Variable);
        
        let func_symbol = table.lookup("my_func").unwrap();
        assert_eq!(func_symbol.id, func_id);
        assert_eq!(func_symbol.symbol_type, SymbolType::Function);
    }

    #[test]
    fn test_scope_management() {
        let mut table = SymbolTable::new();
        
        // Add symbol in global scope
        let global_var = table.add_symbol("global_var".to_string(), SymbolType::Variable);
        assert_eq!(table.current_scope_level(), 0);
        
        // Enter new scope
        table.enter_scope();
        assert_eq!(table.current_scope_level(), 1);
        
        // Add symbol in nested scope
        let local_var = table.add_symbol("local_var".to_string(), SymbolType::Variable);
        
        // Both symbols should be accessible
        assert!(table.lookup("global_var").is_some());
        assert!(table.lookup("local_var").is_some());
        
        // Exit scope
        table.exit_scope();
        assert_eq!(table.current_scope_level(), 0);
        
        // Global symbol still accessible, local symbol removed
        assert!(table.lookup("global_var").is_some());
        assert!(table.lookup("local_var").is_none());
        
        // Verify symbol was actually removed
        assert!(table.get_symbol(local_var).is_none());
    }

    #[test]
    fn test_scope_symbol_filtering() {
        let mut table = SymbolTable::new();
        
        table.add_symbol("global1".to_string(), SymbolType::Variable);
        table.add_symbol("global2".to_string(), SymbolType::Function);
        
        table.enter_scope();
        table.add_symbol("local1".to_string(), SymbolType::Variable);
        table.add_symbol("local2".to_string(), SymbolType::Type);
        
        let current_scope_symbols = table.current_scope_symbols();
        assert_eq!(current_scope_symbols.len(), 2);
        
        let names: Vec<&String> = current_scope_symbols.iter()
            .map(|symbol| &symbol.name)
            .collect();
        assert!(names.contains(&&"local1".to_string()));
        assert!(names.contains(&&"local2".to_string()));
        
        table.exit_scope();
        
        let global_scope_symbols = table.current_scope_symbols();
        assert_eq!(global_scope_symbols.len(), 2);
    }

    #[test]
    fn test_symbol_name_retrieval() {
        let mut table = SymbolTable::new();
        
        let var_id = table.add_symbol("test_var".to_string(), SymbolType::Variable);
        let name = table.get_name(var_id).unwrap();
        assert_eq!(name, "test_var");
        
        // Test non-existent ID
        assert!(table.get_name(99999).is_none());
    }

    #[test]
    fn test_current_scope_existence_check() {
        let mut table = SymbolTable::new();
        
        table.add_symbol("global_var".to_string(), SymbolType::Variable);
        assert!(!table.exists_in_current_scope("global_var")); // Built in different scope
        
        table.enter_scope();
        table.add_symbol("local_var".to_string(), SymbolType::Variable);
        assert!(table.exists_in_current_scope("local_var"));
        assert!(!table.exists_in_current_scope("global_var"));
    }
}