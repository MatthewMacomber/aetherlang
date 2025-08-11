// Aether Abstract Syntax Tree
// AST node data structures supporting atoms, lists, and graph extensions

use std::collections::HashMap;
use std::fmt;
use crate::compiler::token::Token;

/// AST node identifier for graph references
pub type NodeId = u32;

/// AST node supporting atoms, lists, and graph structures with cycles
#[derive(Debug, Clone, PartialEq)]
pub enum ASTNode {
    /// Atomic values (symbols, numbers, strings, booleans)
    Atom(AtomValue),
    /// List of child nodes
    List(Vec<ASTNodeRef>),
    /// Graph node with labeled references for cycle support
    Graph {
        nodes: Vec<ASTNodeRef>,
        edges: Vec<GraphEdge>,
        labels: HashMap<String, NodeId>,
    },
}

/// Atomic value types
#[derive(Debug, Clone, PartialEq)]
pub enum AtomValue {
    /// Symbol/identifier
    Symbol(String),
    /// Numeric literal (stored as f64 for simplicity)
    Number(f64),
    /// String literal
    String(String),
    /// Boolean literal
    Boolean(bool),
    /// Nil/null value
    Nil,
    /// Token reference for machine representation
    Token(Token),
}

/// Reference to an AST node (either direct or by ID for graph structures)
#[derive(Debug, Clone, PartialEq)]
pub enum ASTNodeRef {
    /// Direct node reference
    Direct(Box<ASTNode>),
    /// Reference by node ID (for graph structures and cycles)
    Id(NodeId),
    /// Labeled reference for datum labels
    Label(String),
}

/// Graph edge for representing relationships in graph structures
#[derive(Debug, Clone, PartialEq)]
pub struct GraphEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub label: Option<String>,
}

/// AST with support for graph structures and datum labels
#[derive(Debug, Clone)]
pub struct AST {
    /// Root node of the AST
    pub root: ASTNode,
    /// Node storage for graph references
    pub nodes: HashMap<NodeId, ASTNode>,
    /// Label to node ID mapping
    pub labels: HashMap<String, NodeId>,
    /// Next available node ID
    next_id: NodeId,
}

impl AST {
    /// Create new AST with given root node
    pub fn new(root: ASTNode) -> Self {
        AST {
            root,
            nodes: HashMap::new(),
            labels: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add node to AST and return its ID
    pub fn add_node(&mut self, node: ASTNode) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, node);
        id
    }

    /// Add labeled node to AST
    pub fn add_labeled_node(&mut self, label: String, node: ASTNode) -> NodeId {
        let id = self.add_node(node);
        self.labels.insert(label, id);
        id
    }

    /// Get node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&ASTNode> {
        self.nodes.get(&id)
    }

    /// Get node by label
    pub fn get_labeled_node(&self, label: &str) -> Option<&ASTNode> {
        self.labels.get(label)
            .and_then(|id| self.nodes.get(id))
    }

    /// Resolve node reference to actual node
    pub fn resolve_ref<'a>(&'a self, node_ref: &'a ASTNodeRef) -> Option<&'a ASTNode> {
        match node_ref {
            ASTNodeRef::Direct(node) => Some(node.as_ref()),
            ASTNodeRef::Id(id) => self.get_node(*id),
            ASTNodeRef::Label(label) => self.get_labeled_node(label),
        }
    }

    /// Check if AST contains cycles
    pub fn has_cycles(&self) -> bool {
        // Simple cycle detection using DFS
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();
        
        self.has_cycles_util(&self.root, &mut visited, &mut rec_stack)
    }

    fn has_cycles_util(
        &self,
        node: &ASTNode,
        visited: &mut std::collections::HashSet<*const ASTNode>,
        rec_stack: &mut std::collections::HashSet<*const ASTNode>,
    ) -> bool {
        let node_ptr = node as *const ASTNode;
        
        if rec_stack.contains(&node_ptr) {
            return true;
        }
        
        if visited.contains(&node_ptr) {
            return false;
        }
        
        visited.insert(node_ptr);
        rec_stack.insert(node_ptr);
        
        match node {
            ASTNode::List(children) => {
                for child_ref in children {
                    if let Some(child) = self.resolve_ref(child_ref) {
                        if self.has_cycles_util(child, visited, rec_stack) {
                            return true;
                        }
                    }
                }
            }
            ASTNode::Graph { nodes, .. } => {
                for node_ref in nodes {
                    if let Some(child) = self.resolve_ref(node_ref) {
                        if self.has_cycles_util(child, visited, rec_stack) {
                            return true;
                        }
                    }
                }
            }
            ASTNode::Atom(_) => {}
        }
        
        rec_stack.remove(&node_ptr);
        false
    }
}

impl ASTNode {
    /// Create atom node from symbol
    pub fn symbol(name: String) -> Self {
        ASTNode::Atom(AtomValue::Symbol(name))
    }

    /// Create atom node from number
    pub fn number(value: f64) -> Self {
        ASTNode::Atom(AtomValue::Number(value))
    }

    /// Create atom node from string
    pub fn string(value: String) -> Self {
        ASTNode::Atom(AtomValue::String(value))
    }

    /// Create atom node from boolean
    pub fn boolean(value: bool) -> Self {
        ASTNode::Atom(AtomValue::Boolean(value))
    }

    /// Create nil atom node
    pub fn nil() -> Self {
        ASTNode::Atom(AtomValue::Nil)
    }

    /// Create atom node from token
    pub fn token(token: Token) -> Self {
        ASTNode::Atom(AtomValue::Token(token))
    }

    /// Create list node
    pub fn list(children: Vec<ASTNodeRef>) -> Self {
        ASTNode::List(children)
    }

    /// Create empty list node
    pub fn empty_list() -> Self {
        ASTNode::List(Vec::new())
    }

    /// Check if node is an atom
    pub fn is_atom(&self) -> bool {
        matches!(self, ASTNode::Atom(_))
    }

    /// Check if node is a list
    pub fn is_list(&self) -> bool {
        matches!(self, ASTNode::List(_))
    }

    /// Check if node is a graph
    pub fn is_graph(&self) -> bool {
        matches!(self, ASTNode::Graph { .. })
    }

    /// Get atom value if node is an atom
    pub fn as_atom(&self) -> Option<&AtomValue> {
        match self {
            ASTNode::Atom(value) => Some(value),
            _ => None,
        }
    }

    /// Get list children if node is a list
    pub fn as_list(&self) -> Option<&Vec<ASTNodeRef>> {
        match self {
            ASTNode::List(children) => Some(children),
            _ => None,
        }
    }
}

impl ASTNodeRef {
    /// Create direct node reference
    pub fn direct(node: ASTNode) -> Self {
        ASTNodeRef::Direct(Box::new(node))
    }

    /// Create ID reference
    pub fn id(id: NodeId) -> Self {
        ASTNodeRef::Id(id)
    }

    /// Create label reference
    pub fn label(label: String) -> Self {
        ASTNodeRef::Label(label)
    }
}

impl AtomValue {
    /// Check if atom is a symbol
    pub fn is_symbol(&self) -> bool {
        matches!(self, AtomValue::Symbol(_))
    }

    /// Check if atom is a number
    pub fn is_number(&self) -> bool {
        matches!(self, AtomValue::Number(_))
    }

    /// Check if atom is a string
    pub fn is_string(&self) -> bool {
        matches!(self, AtomValue::String(_))
    }

    /// Check if atom is a boolean
    pub fn is_boolean(&self) -> bool {
        matches!(self, AtomValue::Boolean(_))
    }

    /// Check if atom is nil
    pub fn is_nil(&self) -> bool {
        matches!(self, AtomValue::Nil)
    }

    /// Get symbol name if atom is a symbol
    pub fn as_symbol(&self) -> Option<&String> {
        match self {
            AtomValue::Symbol(name) => Some(name),
            _ => None,
        }
    }

    /// Get number value if atom is a number
    pub fn as_number(&self) -> Option<f64> {
        match self {
            AtomValue::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// Get string value if atom is a string
    pub fn as_string(&self) -> Option<&String> {
        match self {
            AtomValue::String(value) => Some(value),
            _ => None,
        }
    }

    /// Get boolean value if atom is a boolean
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            AtomValue::Boolean(value) => Some(*value),
            _ => None,
        }
    }
}

impl fmt::Display for ASTNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ASTNode::Atom(atom) => write!(f, "{}", atom),
            ASTNode::List(children) => {
                write!(f, "(")?;
                for (i, child) in children.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", child)?;
                }
                write!(f, ")")
            }
            ASTNode::Graph { nodes, labels, .. } => {
                write!(f, "#graph(")?;
                for (i, node) in nodes.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", node)?;
                }
                if !labels.is_empty() {
                    write!(f, " :labels {:?}", labels)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for AtomValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AtomValue::Symbol(name) => write!(f, "{}", name),
            AtomValue::Number(value) => write!(f, "{}", value),
            AtomValue::String(value) => write!(f, "\"{}\"", value),
            AtomValue::Boolean(value) => write!(f, "{}", value),
            AtomValue::Nil => write!(f, "nil"),
            AtomValue::Token(token) => write!(f, "#token({})", token),
        }
    }
}

impl fmt::Display for ASTNodeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ASTNodeRef::Direct(node) => write!(f, "{}", node),
            ASTNodeRef::Id(id) => write!(f, "#ref({})", id),
            ASTNodeRef::Label(label) => write!(f, "#label({})", label),
        }
    }
}