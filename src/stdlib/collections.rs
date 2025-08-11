// Aether Standard Library - Core Collections
// Implements vectors, maps, sets with linear type support

use crate::compiler::types::LinearOwnership;
use crate::runtime::memory_layout::MemoryLayoutOptimizer;
use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;

/// Linear vector implementation with ownership tracking
#[derive(Debug, Clone)]
pub struct LinearVec<T> {
    data: Vec<T>,
    ownership: LinearOwnership,
    capacity_hint: Option<usize>,
}

impl<T> LinearVec<T> {
    /// Create a new linear vector with owned data
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            ownership: LinearOwnership::Owned,
            capacity_hint: None,
        }
    }

    /// Create a new linear vector with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            ownership: LinearOwnership::Owned,
            capacity_hint: Some(capacity),
        }
    }

    /// Push an element, consuming ownership
    pub fn push(mut self, item: T) -> Self {
        self.data.push(item);
        self
    }

    /// Pop an element, returning new vector and element
    pub fn pop(mut self) -> (Self, Option<T>) {
        let item = self.data.pop();
        (self, item)
    }

    /// Get length without consuming
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty without consuming
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get element by index (immutable borrow)
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Convert to owned Vec, consuming the LinearVec
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Map function over elements, preserving linear ownership
    pub fn map<U, F>(self, f: F) -> LinearVec<U>
    where
        F: FnMut(T) -> U,
    {
        LinearVec {
            data: self.data.into_iter().map(f).collect(),
            ownership: self.ownership,
            capacity_hint: self.capacity_hint,
        }
    }

    /// Filter elements, preserving linear ownership
    pub fn filter<F>(self, f: F) -> Self
    where
        F: FnMut(&T) -> bool,
    {
        Self {
            data: self.data.into_iter().filter(f).collect(),
            ownership: self.ownership,
            capacity_hint: self.capacity_hint,
        }
    }

    /// Fold over elements, consuming the vector
    pub fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, T) -> B,
    {
        self.data.into_iter().fold(init, f)
    }
}

impl<T> Default for LinearVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear map implementation with ownership tracking
#[derive(Debug, Clone)]
pub struct LinearMap<K, V> 
where
    K: Eq + Hash,
{
    data: HashMap<K, V>,
    ownership: LinearOwnership,
}

impl<K, V> LinearMap<K, V>
where
    K: Eq + Hash,
{
    /// Create a new linear map
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            ownership: LinearOwnership::Owned,
        }
    }

    /// Create a new linear map with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
            ownership: LinearOwnership::Owned,
        }
    }

    /// Insert a key-value pair, consuming and returning the map
    pub fn insert(mut self, key: K, value: V) -> (Self, Option<V>) {
        let old_value = self.data.insert(key, value);
        (self, old_value)
    }

    /// Remove a key, consuming and returning the map
    pub fn remove(mut self, key: &K) -> (Self, Option<V>) {
        let value = self.data.remove(key);
        (self, value)
    }

    /// Get a value by key (immutable borrow)
    pub fn get(&self, key: &K) -> Option<&V> {
        self.data.get(key)
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &K) -> bool {
        self.data.contains_key(key)
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get all keys as a vector
    pub fn keys(&self) -> LinearVec<&K> {
        LinearVec {
            data: self.data.keys().collect(),
            ownership: LinearOwnership::Owned,
            capacity_hint: Some(self.data.len()),
        }
    }

    /// Get all values as a vector
    pub fn values(&self) -> LinearVec<&V> {
        LinearVec {
            data: self.data.values().collect(),
            ownership: LinearOwnership::Owned,
            capacity_hint: Some(self.data.len()),
        }
    }

    /// Convert to owned HashMap, consuming the LinearMap
    pub fn into_hashmap(self) -> HashMap<K, V> {
        self.data
    }
}

impl<K, V> Default for LinearMap<K, V>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Linear set implementation with ownership tracking
#[derive(Debug, Clone)]
pub struct LinearSet<T>
where
    T: Eq + Hash,
{
    data: std::collections::HashSet<T>,
    ownership: LinearOwnership,
}

impl<T> LinearSet<T>
where
    T: Eq + Hash + Clone,
{
    /// Create a new linear set
    pub fn new() -> Self {
        Self {
            data: std::collections::HashSet::new(),
            ownership: LinearOwnership::Owned,
        }
    }

    /// Create a new linear set with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: std::collections::HashSet::with_capacity(capacity),
            ownership: LinearOwnership::Owned,
        }
    }

    /// Insert an element, consuming and returning the set
    pub fn insert(mut self, value: T) -> (Self, bool) {
        let was_new = self.data.insert(value);
        (self, was_new)
    }

    /// Remove an element, consuming and returning the set
    pub fn remove(mut self, value: &T) -> (Self, bool) {
        let was_present = self.data.remove(value);
        (self, was_present)
    }

    /// Check if element exists
    pub fn contains(&self, value: &T) -> bool {
        self.data.contains(value)
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Union with another set, consuming both
    pub fn union(self, other: Self) -> Self {
        Self {
            data: self.data.union(&other.data).cloned().collect(),
            ownership: LinearOwnership::Owned,
        }
    }

    /// Intersection with another set, consuming both
    pub fn intersection(self, other: Self) -> Self {
        Self {
            data: self.data.intersection(&other.data).cloned().collect(),
            ownership: LinearOwnership::Owned,
        }
    }

    /// Difference with another set, consuming both
    pub fn difference(self, other: Self) -> Self {
        Self {
            data: self.data.difference(&other.data).cloned().collect(),
            ownership: LinearOwnership::Owned,
        }
    }

    /// Convert to vector, consuming the set
    pub fn into_vec(self) -> LinearVec<T> {
        LinearVec {
            data: self.data.into_iter().collect(),
            ownership: self.ownership,
            capacity_hint: None,
        }
    }
}

impl<T> Default for LinearSet<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Collection utilities for memory optimization
pub struct CollectionUtils;

impl CollectionUtils {
    /// Optimize memory layout for a vector
    pub fn optimize_vector_layout<T>(_vec: &LinearVec<T>) -> MemoryLayoutOptimizer {
        MemoryLayoutOptimizer::new()
    }

    /// Optimize memory layout for a map
    pub fn optimize_map_layout<K, V>(_map: &LinearMap<K, V>) -> MemoryLayoutOptimizer
    where
        K: Eq + Hash,
    {
        MemoryLayoutOptimizer::new()
    }

    /// Estimate memory usage for collections
    pub fn estimate_memory_usage<T>(count: usize) -> usize {
        count * std::mem::size_of::<T>() + std::mem::size_of::<Vec<T>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_vec_basic_operations() {
        let vec = LinearVec::new()
            .push(1)
            .push(2)
            .push(3);
        
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(1), Some(&2));
        
        let (vec, popped) = vec.pop();
        assert_eq!(popped, Some(3));
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_linear_map_basic_operations() {
        let (map, _) = LinearMap::new().insert("key1", 10);
        let (map, _) = map.insert("key2", 20);
        
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&"key1"), Some(&10));
        assert!(map.contains_key(&"key2"));
        
        let (map, removed) = map.remove(&"key1");
        assert_eq!(removed, Some(10));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_linear_set_basic_operations() {
        let (set, _) = LinearSet::new().insert(1);
        let (set, _) = set.insert(2);
        let (set, was_new) = set.insert(1); // Duplicate
        
        assert!(!was_new);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        
        let (set, was_present) = set.remove(&1);
        assert!(was_present);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_set_operations() {
        let (set1, _) = LinearSet::new().insert(1);
        let (set1, _) = set1.insert(2);
        
        let (set2, _) = LinearSet::new().insert(2);
        let (set2, _) = set2.insert(3);
        
        let union = set1.clone().union(set2.clone());
        assert_eq!(union.len(), 3);
        
        let intersection = set1.intersection(set2);
        assert_eq!(intersection.len(), 1);
        assert!(intersection.contains(&2));
    }

    #[test]
    fn test_functional_operations() {
        let vec = LinearVec::new()
            .push(1)
            .push(2)
            .push(3);
        
        let doubled = vec.map(|x| x * 2);
        assert_eq!(doubled.get(0), Some(&2));
        assert_eq!(doubled.get(1), Some(&4));
        assert_eq!(doubled.get(2), Some(&6));
        
        let filtered = LinearVec::new()
            .push(1)
            .push(2)
            .push(3)
            .push(4)
            .filter(|&x| x % 2 == 0);
        assert_eq!(filtered.len(), 2);
        
        let sum = LinearVec::new()
            .push(1)
            .push(2)
            .push(3)
            .fold(0, |acc, x| acc + x);
        assert_eq!(sum, 6);
    }
}