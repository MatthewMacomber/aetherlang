// Aether Standard Library Module
// Core data structures, mathematical functions, and I/O operations

pub mod collections;
pub mod math;
pub mod io;
pub mod string;
pub mod ecs;
pub mod math3d;
pub mod graphics;
pub mod physics;

// Re-export core collections
pub use collections::{LinearVec, LinearMap, LinearSet, CollectionUtils};

// Re-export mathematical functions
pub use math::{
    DiffScalar, constants, trig, exp_log, hyperbolic, stats
};

// Re-export I/O operations
pub use io::{
    LinearFile, LinearReader, LinearWriter, LinearDirectory, 
    IOError, IOResult, async_io, serialization
};

// Re-export string processing
pub use string::{LinearString, StringUtils, regex};

// Re-export game development types
pub use ecs::{World, Entity, Component, System};
pub use math3d::{Vec2, Vec3, Vec4, Quat, Mat3, Mat4};
pub use graphics::{GraphicsAPI, GraphicsBackend, create_graphics_context};
pub use physics::{PhysicsWorld, RigidBody, CollisionShape, PhysicsMaterial};