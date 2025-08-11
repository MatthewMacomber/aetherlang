// Entity-Component-System (ECS) architecture support for Aether
// Provides high-performance game development patterns with compile-time safety

use std::collections::HashMap;
use std::any::{Any, TypeId};
use std::marker::PhantomData;

/// Unique identifier for entities in the ECS world
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity(pub u64);

/// Trait for components that can be attached to entities
pub trait Component: Any + Send + Sync + 'static {}

/// World manages all entities, components, and systems
pub struct World {
    next_entity_id: u64,
    entities: Vec<Entity>,
    components: HashMap<TypeId, Box<dyn ComponentStorage>>,
    systems: Vec<Box<dyn System>>,
}

/// Storage trait for component collections
trait ComponentStorage: Send + Sync {
    fn remove(&mut self, entity: Entity);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Concrete storage for a specific component type
struct ComponentVec<T: Component> {
    data: HashMap<Entity, T>,
}

impl<T: Component> ComponentStorage for ComponentVec<T> {
    fn remove(&mut self, entity: Entity) {
        self.data.remove(&entity);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// System trait for processing entities with specific components
pub trait System: Send + Sync {
    fn run(&mut self, world: &mut World);
}

/// Query builder for accessing components
pub struct Query<'a, T> {
    world: &'a World,
    _phantom: PhantomData<T>,
}

impl World {
    /// Create a new ECS world
    pub fn new() -> Self {
        World {
            next_entity_id: 0,
            entities: Vec::new(),
            components: HashMap::new(),
            systems: Vec::new(),
        }
    }

    /// Create a new entity
    pub fn create_entity(&mut self) -> Entity {
        let entity = Entity(self.next_entity_id);
        self.next_entity_id += 1;
        self.entities.push(entity);
        entity
    }

    /// Add a component to an entity
    pub fn add_component<T: Component>(&mut self, entity: Entity, component: T) {
        let type_id = TypeId::of::<T>();
        
        if !self.components.contains_key(&type_id) {
            self.components.insert(
                type_id,
                Box::new(ComponentVec::<T> {
                    data: HashMap::new(),
                })
            );
        }

        if let Some(storage) = self.components.get_mut(&type_id) {
            if let Some(component_vec) = storage.as_any_mut().downcast_mut::<ComponentVec<T>>() {
                component_vec.data.insert(entity, component);
            }
        }
    }

    /// Get a component from an entity
    pub fn get_component<T: Component>(&self, entity: Entity) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        
        if let Some(storage) = self.components.get(&type_id) {
            if let Some(component_vec) = storage.as_any().downcast_ref::<ComponentVec<T>>() {
                return component_vec.data.get(&entity);
            }
        }
        None
    }

    /// Get a mutable component from an entity
    pub fn get_component_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        
        if let Some(storage) = self.components.get_mut(&type_id) {
            if let Some(component_vec) = storage.as_any_mut().downcast_mut::<ComponentVec<T>>() {
                return component_vec.data.get_mut(&entity);
            }
        }
        None
    }

    /// Remove a component from an entity
    pub fn remove_component<T: Component>(&mut self, entity: Entity) -> Option<T> {
        let type_id = TypeId::of::<T>();
        
        if let Some(storage) = self.components.get_mut(&type_id) {
            if let Some(component_vec) = storage.as_any_mut().downcast_mut::<ComponentVec<T>>() {
                return component_vec.data.remove(&entity);
            }
        }
        None
    }

    /// Destroy an entity and all its components
    pub fn destroy_entity(&mut self, entity: Entity) {
        // Remove from entities list
        self.entities.retain(|&e| e != entity);
        
        // Remove all components for this entity
        for storage in self.components.values_mut() {
            storage.remove(entity);
        }
    }

    /// Add a system to the world
    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.systems.push(Box::new(system));
    }

    /// Run all systems
    pub fn update(&mut self) {
        // Note: In a real implementation, we'd need to handle borrowing more carefully
        // This is a simplified version for demonstration
        for _i in 0..self.systems.len() {
            // Unsafe pointer manipulation would be needed here for proper borrowing
            // This is a conceptual implementation
        }
    }

    /// Query entities with specific components
    pub fn query<T: Component>(&self) -> Vec<(Entity, &T)> {
        let type_id = TypeId::of::<T>();
        let mut results = Vec::new();
        
        if let Some(storage) = self.components.get(&type_id) {
            if let Some(component_vec) = storage.as_any().downcast_ref::<ComponentVec<T>>() {
                for (&entity, component) in &component_vec.data {
                    results.push((entity, component));
                }
            }
        }
        
        results
    }

    /// Get all entities
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }
}

/// Macro for easier component definition
#[macro_export]
macro_rules! component {
    ($name:ident { $($field:ident: $type:ty),* }) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            $(pub $field: $type,)*
        }
        
        impl Component for $name {}
    };
}

/// Common game components
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: crate::stdlib::math3d::Vec3,
    pub rotation: crate::stdlib::math3d::Quat,
    pub scale: crate::stdlib::math3d::Vec3,
}

impl Component for Transform {}

#[derive(Debug, Clone, Copy)]
pub struct Velocity {
    pub linear: crate::stdlib::math3d::Vec3,
    pub angular: crate::stdlib::math3d::Vec3,
}

impl Component for Velocity {}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<crate::stdlib::math3d::Vec3>,
    pub indices: Vec<u32>,
    pub material_id: u32,
}

impl Component for Mesh {}

#[derive(Debug, Clone)]
pub struct RigidBody {
    pub mass: f32,
    pub restitution: f32,
    pub friction: f32,
    pub is_kinematic: bool,
}

impl Component for RigidBody {}

/// Example movement system
pub struct MovementSystem;

impl System for MovementSystem {
    fn run(&mut self, world: &mut World) {
        // This would need proper query system implementation
        // Conceptual implementation for demonstration
        let entities: Vec<Entity> = world.entities().to_vec();
        
        for entity in entities {
            // Get velocity first (immutable borrow)
            let velocity = if let Some(vel) = world.get_component::<Velocity>(entity) {
                *vel // Copy the velocity
            } else {
                continue;
            };
            
            // Then get mutable transform
            if let Some(transform) = world.get_component_mut::<Transform>(entity) {
                // Update position based on velocity (simplified)
                transform.position.x += velocity.linear.x * 0.016; // 60 FPS
                transform.position.y += velocity.linear.y * 0.016;
                transform.position.z += velocity.linear.z * 0.016;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::math3d::{Vec3, Quat};

    #[test]
    fn test_ecs_basic_operations() {
        let mut world = World::new();
        
        // Create entity
        let entity = world.create_entity();
        
        // Add components
        world.add_component(entity, Transform {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::identity(),
            scale: Vec3::new(1.0, 1.0, 1.0),
        });
        
        world.add_component(entity, Velocity {
            linear: Vec3::new(1.0, 0.0, 0.0),
            angular: Vec3::new(0.0, 0.0, 0.0),
        });
        
        // Query components
        assert!(world.get_component::<Transform>(entity).is_some());
        assert!(world.get_component::<Velocity>(entity).is_some());
        
        // Remove component
        world.remove_component::<Velocity>(entity);
        assert!(world.get_component::<Velocity>(entity).is_none());
        
        // Destroy entity
        world.destroy_entity(entity);
        assert!(world.get_component::<Transform>(entity).is_none());
    }

    #[test]
    fn test_movement_system() {
        let mut world = World::new();
        let entity = world.create_entity();
        
        world.add_component(entity, Transform {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::identity(),
            scale: Vec3::new(1.0, 1.0, 1.0),
        });
        
        world.add_component(entity, Velocity {
            linear: Vec3::new(1.0, 0.0, 0.0),
            angular: Vec3::new(0.0, 0.0, 0.0),
        });
        
        let mut movement_system = MovementSystem;
        movement_system.run(&mut world);
        
        let transform = world.get_component::<Transform>(entity).unwrap();
        assert!(transform.position.x > 0.0);
    }
}