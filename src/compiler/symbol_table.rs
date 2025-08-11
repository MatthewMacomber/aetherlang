// Aether Symbol Table
// Manages variable and function name resolution

use std::collections::HashMap;
use std::fmt;

/// Symbol identifier type
pub type SymbolId = u32;

/// Symbol information
#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub id: SymbolId,
    pub name: String,
    pub symbol_type: SymbolType,
    pub scope_level: u32,
}

/// Types of symbols in the language
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolType {
    Variable,
    Function,
    Type,
    Keyword,
    Operator,
}

/// Symbol table for name resolution
#[derive(Debug, Clone)]
pub struct SymbolTable {
    symbols: HashMap<SymbolId, Symbol>,
    name_to_id: HashMap<String, SymbolId>,
    next_id: SymbolId,
    scope_stack: Vec<u32>,
    current_scope: u32,
}

impl SymbolTable {
    /// Create new symbol table
    pub fn new() -> Self {
        let mut table = SymbolTable {
            symbols: HashMap::new(),
            name_to_id: HashMap::new(),
            next_id: 0,
            scope_stack: vec![0],
            current_scope: 0,
        };
        
        // Initialize with built-in keywords and operators
        table.init_builtins();
        table
    }

    /// Initialize built-in symbols
    fn init_builtins(&mut self) {
        // Keywords
        let keywords = [
            "let", "fn", "if", "else", "while", "for", "match", "return",
            "true", "false", "nil", "type", "struct", "enum", "impl",
            "pub", "mod", "use", "as", "where", "self", "Self",
        ];

        for keyword in &keywords {
            self.add_symbol(keyword.to_string(), SymbolType::Keyword);
        }

        // Operators
        let operators = [
            "+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=",
            "&&", "||", "!", "&", "|", "^", "<<", ">>", "~",
            "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=",
            "->", "=>", "::", ".", "..", "...", "?", "@",
        ];

        for operator in &operators {
            self.add_symbol(operator.to_string(), SymbolType::Operator);
        }
    }

    /// Add new symbol to table
    pub fn add_symbol(&mut self, name: String, symbol_type: SymbolType) -> SymbolId {
        let id = self.next_id;
        self.next_id += 1;

        let symbol = Symbol {
            id,
            name: name.clone(),
            symbol_type,
            scope_level: self.current_scope,
        };

        self.symbols.insert(id, symbol);
        self.name_to_id.insert(name, id);
        id
    }

    /// Look up symbol by name
    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.name_to_id.get(name)
            .and_then(|id| self.symbols.get(id))
    }

    /// Look up symbol by ID
    pub fn get_symbol(&self, id: SymbolId) -> Option<&Symbol> {
        self.symbols.get(&id)
    }

    /// Get symbol name by ID
    pub fn get_name(&self, id: SymbolId) -> Option<&String> {
        self.symbols.get(&id).map(|symbol| &symbol.name)
    }

    /// Enter new scope
    pub fn enter_scope(&mut self) {
        self.current_scope += 1;
        self.scope_stack.push(self.current_scope);
    }

    /// Exit current scope
    pub fn exit_scope(&mut self) {
        if let Some(scope_level) = self.scope_stack.pop() {
            // Remove symbols from the exited scope
            let symbols_to_remove: Vec<SymbolId> = self.symbols
                .iter()
                .filter(|(_, symbol)| symbol.scope_level == scope_level)
                .map(|(id, _)| *id)
                .collect();

            for id in symbols_to_remove {
                if let Some(symbol) = self.symbols.remove(&id) {
                    self.name_to_id.remove(&symbol.name);
                }
            }

            // Update current scope
            self.current_scope = self.scope_stack.last().copied().unwrap_or(0);
        }
    }

    /// Get current scope level
    pub fn current_scope_level(&self) -> u32 {
        self.current_scope
    }

    /// Check if symbol exists in current scope
    pub fn exists_in_current_scope(&self, name: &str) -> bool {
        self.lookup(name)
            .map(|symbol| symbol.scope_level == self.current_scope)
            .unwrap_or(false)
    }

    /// Get all symbols in current scope
    pub fn current_scope_symbols(&self) -> Vec<&Symbol> {
        self.symbols
            .values()
            .filter(|symbol| symbol.scope_level == self.current_scope)
            .collect()
    }

    /// Get total number of symbols
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Check if table is empty (excluding builtins)
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SymbolTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Symbol Table ({} symbols):", self.symbols.len())?;
        for symbol in self.symbols.values() {
            writeln!(f, "  {}: {} ({:?}, scope {})", 
                symbol.id, symbol.name, symbol.symbol_type, symbol.scope_level)?;
        }
        Ok(())
    }
}