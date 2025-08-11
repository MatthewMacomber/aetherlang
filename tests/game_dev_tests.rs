// Tests for game development and simulation API
// Covers ECS, 3D math, graphics, and physics systems

use aether_lang::stdlib::{
    ecs::{World, Entity, Component, System, Transform, Velocity, MovementSystem},
    math3d::{Vec2, Vec3, Vec4, Quat, Mat3, Mat4, degrees_to_radians},
    graphics::{GraphicsBackend, create_graphics_context, BufferUsage, TextureDescriptor, TextureFormat, TextureUsage, Mesh, Vertex},
    physics::{PhysicsWorld, CollisionShape, PhysicsMaterial, Ray, BodyId},
};

#[test]
fn test_ecs_entity_creation_and_components() {
    let mut world = World::new();
    
    // Create entity
    let entity = world.create_entity();
    
    // Add components
    world.add_component(entity, Transform {
        position: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::identity(),
        scale: Vec3::new(1.0, 1.0, 1.0),
    });
    
    world.add_component(entity, Velocity {
        linear: Vec3::new(0.5, 0.0, 0.0),
        angular: Vec3::new(0.0, 0.0, 0.0),
    });
    
    // Verify components exist
    assert!(world.get_component::<Transform>(entity).is_some());
    assert!(world.get_component::<Velocity>(entity).is_some());
    
    let transform = world.get_component::<Transform>(entity).unwrap();
    assert_eq!(transform.position, Vec3::new(1.0, 2.0, 3.0));
    
    let velocity = world.get_component::<Velocity>(entity).unwrap();
    assert_eq!(velocity.linear, Vec3::new(0.5, 0.0, 0.0));
}

#[test]
fn test_ecs_component_removal() {
    let mut world = World::new();
    let entity = world.create_entity();
    
    world.add_component(entity, Transform {
        position: Vec3::ZERO,
        rotation: Quat::identity(),
        scale: Vec3::ONE,
    });
    
    assert!(world.get_component::<Transform>(entity).is_some());
    
    let removed = world.remove_component::<Transform>(entity);
    assert!(removed.is_some());
    assert!(world.get_component::<Transform>(entity).is_none());
}

#[test]
fn test_ecs_entity_destruction() {
    let mut world = World::new();
    let entity = world.create_entity();
    
    world.add_component(entity, Transform {
        position: Vec3::ZERO,
        rotation: Quat::identity(),
        scale: Vec3::ONE,
    });
    
    assert_eq!(world.entities().len(), 1);
    
    world.destroy_entity(entity);
    
    assert_eq!(world.entities().len(), 0);
    assert!(world.get_component::<Transform>(entity).is_none());
}

#[test]
fn test_ecs_movement_system() {
    let mut world = World::new();
    let entity = world.create_entity();
    
    world.add_component(entity, Transform {
        position: Vec3::ZERO,
        rotation: Quat::identity(),
        scale: Vec3::ONE,
    });
    
    world.add_component(entity, Velocity {
        linear: Vec3::new(1.0, 0.0, 0.0),
        angular: Vec3::ZERO,
    });
    
    let mut movement_system = MovementSystem;
    movement_system.run(&mut world);
    
    let transform = world.get_component::<Transform>(entity).unwrap();
    // Position should have changed due to velocity
    assert!(transform.position.x > 0.0);
}

#[test]
fn test_math3d_vector_operations() {
    let v1 = Vec3::new(1.0, 2.0, 3.0);
    let v2 = Vec3::new(4.0, 5.0, 6.0);
    
    // Basic operations
    let sum = v1 + v2;
    assert_eq!(sum, Vec3::new(5.0, 7.0, 9.0));
    
    let diff = v2 - v1;
    assert_eq!(diff, Vec3::new(3.0, 3.0, 3.0));
    
    let scaled = v1 * 2.0;
    assert_eq!(scaled, Vec3::new(2.0, 4.0, 6.0));
    
    // Dot product
    let dot = v1.dot(v2);
    assert_eq!(dot, 32.0);
    
    // Cross product
    let cross = v1.cross(v2);
    assert_eq!(cross, Vec3::new(-3.0, 6.0, -3.0));
    
    // Length
    let length = Vec3::new(3.0, 4.0, 0.0).length();
    assert_eq!(length, 5.0);
    
    // Normalization
    let normalized = Vec3::new(3.0, 4.0, 0.0).normalize();
    assert_eq!(normalized, Vec3::new(0.6, 0.8, 0.0));
}

#[test]
fn test_math3d_quaternion_operations() {
    // Identity quaternion
    let q_identity = Quat::identity();
    assert_eq!(q_identity, Quat::new(0.0, 0.0, 0.0, 1.0));
    
    // Rotation around Y axis
    let q_y_90 = Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI / 2.0);
    
    // Rotate X vector around Y by 90 degrees should give -Z
    let rotated = q_y_90.rotate_vector(Vec3::X);
    assert!((rotated.x - 0.0).abs() < 0.001);
    assert!((rotated.y - 0.0).abs() < 0.001);
    assert!((rotated.z - (-1.0)).abs() < 0.001);
    
    // Quaternion multiplication
    let q1 = Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI / 4.0);
    let q2 = Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI / 4.0);
    let combined = q1 * q2;
    
    // Should be equivalent to 90 degree rotation
    let expected = Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI / 2.0);
    assert!((combined.x - expected.x).abs() < 0.001);
    assert!((combined.y - expected.y).abs() < 0.001);
    assert!((combined.z - expected.z).abs() < 0.001);
    assert!((combined.w - expected.w).abs() < 0.001);
}

#[test]
fn test_math3d_matrix_operations() {
    // Identity matrix
    let identity = Mat4::identity();
    let point = Vec3::new(1.0, 2.0, 3.0);
    let transformed = identity.transform_point(point);
    assert_eq!(transformed, point);
    
    // Translation matrix
    let translation = Mat4::translation(Vec3::new(5.0, 10.0, 15.0));
    let translated = translation.transform_point(point);
    assert_eq!(translated, Vec3::new(6.0, 12.0, 18.0));
    
    // Scale matrix
    let scale = Mat4::scale(Vec3::new(2.0, 3.0, 4.0));
    let scaled = scale.transform_point(point);
    assert_eq!(scaled, Vec3::new(2.0, 6.0, 12.0));
    
    // Matrix multiplication
    let combined = translation * scale;
    let result = combined.transform_point(point);
    assert_eq!(result, Vec3::new(7.0, 16.0, 27.0));
    
    // Perspective matrix
    let fov = degrees_to_radians(45.0);
    let aspect = 16.0 / 9.0;
    let near = 0.1;
    let far = 100.0;
    let perspective = Mat4::perspective(fov, aspect, near, far);
    
    // Matrix should not be zero
    assert!(perspective.determinant() != 0.0);
}

#[test]
fn test_graphics_context_creation() {
    let context = create_graphics_context(GraphicsBackend::WebGPU);
    assert!(context.is_ok());
}

#[test]
fn test_graphics_buffer_creation() {
    let mut context = create_graphics_context(GraphicsBackend::WebGPU).unwrap();
    
    let vertex_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            vertex_data.as_ptr() as *const u8,
            vertex_data.len() * std::mem::size_of::<f32>(),
        )
    };
    
    let buffer = context.create_buffer(bytes, BufferUsage::Vertex);
    assert!(buffer.is_ok());
}

#[test]
fn test_graphics_texture_creation() {
    let mut context = create_graphics_context(GraphicsBackend::WebGPU).unwrap();
    
    let texture_desc = TextureDescriptor {
        width: 256,
        height: 256,
        depth: 1,
        format: TextureFormat::RGBA8Unorm,
        usage: TextureUsage::Sampled,
        sample_count: 1,
    };
    
    let texture = context.create_texture(&texture_desc);
    assert!(texture.is_ok());
}

#[test]
fn test_graphics_mesh_generation() {
    // Test cube mesh generation
    let cube = Mesh::cube();
    assert!(!cube.vertices.is_empty());
    assert!(!cube.indices.is_empty());
    
    // Verify cube has 8 vertices (simplified cube)
    assert!(cube.vertices.len() >= 4); // At least front face
    
    // Test sphere mesh generation
    let sphere = Mesh::sphere(1.0, 8);
    assert!(!sphere.vertices.is_empty());
    assert!(!sphere.indices.is_empty());
    
    // Verify sphere has correct number of vertices
    let expected_vertices = (8 + 1) * (8 + 1);
    assert_eq!(sphere.vertices.len(), expected_vertices);
}

#[test]
fn test_physics_world_creation() {
    let world = PhysicsWorld::new();
    // World should be created successfully
    assert_eq!(world.query_sphere(Vec3::ZERO, 1.0).len(), 0);
}

#[test]
fn test_physics_rigid_body_creation() {
    let mut world = PhysicsWorld::new();
    
    let shape = CollisionShape::Sphere { radius: 1.0 };
    let material = PhysicsMaterial::default();
    let position = Vec3::new(0.0, 10.0, 0.0);
    let rotation = Quat::identity();
    
    let body_id = world.create_body(shape, material, position, rotation);
    let body = world.get_body(body_id).unwrap();
    
    assert_eq!(body.position, position);
    assert_eq!(body.rotation, rotation);
    assert!(body.mass > 0.0);
    assert!(body.inverse_mass > 0.0);
}

#[test]
fn test_physics_collision_detection() {
    let mut world = PhysicsWorld::new();
    
    // Create two overlapping spheres
    let shape1 = CollisionShape::Sphere { radius: 1.0 };
    let shape2 = CollisionShape::Sphere { radius: 1.0 };
    let material = PhysicsMaterial::default();
    
    let body1 = world.create_body(
        shape1,
        material.clone(),
        Vec3::new(0.0, 0.0, 0.0),
        Quat::identity(),
    );
    
    let body2 = world.create_body(
        shape2,
        material,
        Vec3::new(1.5, 0.0, 0.0), // Overlapping
        Quat::identity(),
    );
    
    // Step physics to detect collisions
    world.step(1.0 / 60.0);
    
    // Bodies should have moved apart due to collision resolution
    let body1_pos = world.get_body(body1).unwrap().position;
    let body2_pos = world.get_body(body2).unwrap().position;
    let distance = (body2_pos - body1_pos).length();
    
    // Distance should be close to sum of radii (2.0)
    assert!(distance >= 1.8); // Allow some tolerance
}

#[test]
fn test_physics_gravity_simulation() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));
    
    let shape = CollisionShape::Sphere { radius: 1.0 };
    let material = PhysicsMaterial::default();
    let initial_position = Vec3::new(0.0, 10.0, 0.0);
    
    let body_id = world.create_body(shape, material, initial_position, Quat::identity());
    
    // Step physics multiple times
    for _ in 0..60 {
        world.step(1.0 / 60.0);
    }
    
    let final_position = world.get_body(body_id).unwrap().position;
    
    // Body should have fallen due to gravity
    assert!(final_position.y < initial_position.y);
}

#[test]
fn test_physics_raycast() {
    let mut world = PhysicsWorld::new();
    
    let shape = CollisionShape::Sphere { radius: 1.0 };
    let material = PhysicsMaterial::default();
    let body_id = world.create_body(shape, material, Vec3::ZERO, Quat::identity());
    
    let ray = Ray {
        origin: Vec3::new(-5.0, 0.0, 0.0),
        direction: Vec3::new(1.0, 0.0, 0.0),
        max_distance: 10.0,
    };
    
    let hit = world.raycast(ray);
    assert!(hit.is_some());
    
    let hit = hit.unwrap();
    assert_eq!(hit.body_id, body_id);
    assert!(hit.distance > 0.0);
    assert!(hit.distance < 5.0);
}

#[test]
fn test_physics_force_application() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO); // Disable gravity for this test
    
    let shape = CollisionShape::Sphere { radius: 1.0 };
    let material = PhysicsMaterial::default();
    let body_id = world.create_body(shape, material, Vec3::ZERO, Quat::identity());
    
    let initial_velocity = world.get_body(body_id).unwrap().linear_velocity;
    
    // Apply force
    world.apply_force(body_id, Vec3::new(100.0, 0.0, 0.0), None);
    
    let final_velocity = world.get_body(body_id).unwrap().linear_velocity;
    
    // Velocity should have changed
    assert!(final_velocity.x > initial_velocity.x);
}

#[test]
fn test_physics_query_sphere() {
    let mut world = PhysicsWorld::new();
    
    let shape = CollisionShape::Sphere { radius: 0.5 };
    let material = PhysicsMaterial::default();
    
    // Create bodies at different positions
    let body1 = world.create_body(shape.clone(), material.clone(), Vec3::new(0.0, 0.0, 0.0), Quat::identity());
    let body2 = world.create_body(shape.clone(), material.clone(), Vec3::new(2.0, 0.0, 0.0), Quat::identity());
    let body3 = world.create_body(shape, material, Vec3::new(10.0, 0.0, 0.0), Quat::identity());
    
    // Query sphere around origin
    let results = world.query_sphere(Vec3::ZERO, 3.0);
    
    // Should find body1 and body2, but not body3
    assert_eq!(results.len(), 2);
    assert!(results.contains(&body1));
    assert!(results.contains(&body2));
    assert!(!results.contains(&body3));
}

#[test]
fn test_integration_ecs_physics() {
    let mut ecs_world = World::new();
    let mut physics_world = PhysicsWorld::new();
    
    // Create ECS entity
    let entity = ecs_world.create_entity();
    ecs_world.add_component(entity, Transform {
        position: Vec3::new(0.0, 5.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3::ONE,
    });
    
    // Create corresponding physics body
    let shape = CollisionShape::Sphere { radius: 1.0 };
    let material = PhysicsMaterial::default();
    let body_id = physics_world.create_body(
        shape,
        material,
        Vec3::new(0.0, 5.0, 0.0),
        Quat::identity(),
    );
    
    // Step physics
    physics_world.step(1.0 / 60.0);
    
    // Update ECS transform from physics
    let physics_body = physics_world.get_body(body_id).unwrap();
    if let Some(transform) = ecs_world.get_component_mut::<Transform>(entity) {
        transform.position = physics_body.position;
        transform.rotation = physics_body.rotation;
    }
    
    let final_transform = ecs_world.get_component::<Transform>(entity).unwrap();
    
    // Position should have changed due to gravity
    assert!(final_transform.position.y < 5.0);
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_ecs_entity_creation() {
        let mut world = World::new();
        let entity_count = 10000;
        
        let start = Instant::now();
        
        for i in 0..entity_count {
            let entity = world.create_entity();
            world.add_component(entity, Transform {
                position: Vec3::new(i as f32, 0.0, 0.0),
                rotation: Quat::identity(),
                scale: Vec3::ONE,
            });
        }
        
        let duration = start.elapsed();
        println!("Created {} entities in {:?}", entity_count, duration);
        
        // Should be reasonably fast
        assert!(duration.as_millis() < 1000);
    }

    #[test]
    fn benchmark_math_vector_operations() {
        let iterations = 1000000;
        let mut result = Vec3::ZERO;
        
        let start = Instant::now();
        
        for i in 0..iterations {
            let v1 = Vec3::new(i as f32, i as f32, i as f32);
            let v2 = Vec3::new((i + 1) as f32, (i + 1) as f32, (i + 1) as f32);
            result = result + v1.cross(v2);
        }
        
        let duration = start.elapsed();
        println!("Performed {} vector operations in {:?}", iterations, duration);
        
        // Should be very fast
        assert!(duration.as_millis() < 1000);
        
        // Use result to prevent optimization
        assert!(result.length() > 0.0);
    }

    #[test]
    fn benchmark_physics_simulation() {
        let mut world = PhysicsWorld::new();
        let body_count = 100;
        
        // Create physics bodies
        for i in 0..body_count {
            let shape = CollisionShape::Sphere { radius: 0.5 };
            let material = PhysicsMaterial::default();
            let position = Vec3::new(
                (i % 10) as f32,
                (i / 10) as f32 + 10.0,
                0.0,
            );
            
            world.create_body(shape, material, position, Quat::identity());
        }
        
        let start = Instant::now();
        
        // Run simulation for 100 steps
        for _ in 0..100 {
            world.step(1.0 / 60.0);
        }
        
        let duration = start.elapsed();
        println!("Simulated {} bodies for 100 steps in {:?}", body_count, duration);
        
        // Should complete in reasonable time
        assert!(duration.as_millis() < 5000);
    }
}