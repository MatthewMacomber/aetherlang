// Unit tests for game development API components
// Tests individual modules without dependencies on MLIR

#[cfg(test)]
mod ecs_tests {
    use aether_lang::stdlib::ecs::*;
    use aether_lang::stdlib::math3d::*;

    #[test]
    fn test_world_creation() {
        let world = World::new();
        assert_eq!(world.entities().len(), 0);
    }

    #[test]
    fn test_entity_creation() {
        let mut world = World::new();
        let entity = world.create_entity();
        assert_eq!(world.entities().len(), 1);
        assert_eq!(world.entities()[0], entity);
    }

    #[test]
    fn test_component_operations() {
        let mut world = World::new();
        let entity = world.create_entity();
        
        // Add transform component
        world.add_component(entity, Transform {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::identity(),
            scale: Vec3::new(1.0, 1.0, 1.0),
        });
        
        // Verify component exists
        assert!(world.get_component::<Transform>(entity).is_some());
        let transform = world.get_component::<Transform>(entity).unwrap();
        assert_eq!(transform.position, Vec3::new(1.0, 2.0, 3.0));
        
        // Remove component
        let removed = world.remove_component::<Transform>(entity);
        assert!(removed.is_some());
        assert!(world.get_component::<Transform>(entity).is_none());
    }
}

#[cfg(test)]
mod math3d_tests {
    use aether_lang::stdlib::math3d::*;

    #[test]
    fn test_vec3_basic_operations() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        
        // Addition
        let sum = v1 + v2;
        assert_eq!(sum, Vec3::new(5.0, 7.0, 9.0));
        
        // Subtraction
        let diff = v2 - v1;
        assert_eq!(diff, Vec3::new(3.0, 3.0, 3.0));
        
        // Scalar multiplication
        let scaled = v1 * 2.0;
        assert_eq!(scaled, Vec3::new(2.0, 4.0, 6.0));
        
        // Dot product
        let dot = v1.dot(v2);
        assert_eq!(dot, 32.0);
        
        // Cross product
        let cross = v1.cross(v2);
        assert_eq!(cross, Vec3::new(-3.0, 6.0, -3.0));
    }

    #[test]
    fn test_vec3_length_and_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(v.length(), 5.0);
        
        let normalized = v.normalize();
        assert_eq!(normalized, Vec3::new(0.6, 0.8, 0.0));
        assert!((normalized.length() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quaternion_identity() {
        let q = Quat::identity();
        assert_eq!(q, Quat::new(0.0, 0.0, 0.0, 1.0));
        
        // Identity quaternion should not change vectors
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate_vector(v);
        assert_eq!(rotated, v);
    }

    #[test]
    fn test_quaternion_rotation() {
        // 90 degree rotation around Y axis
        let q = Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI / 2.0);
        let v = Vec3::X;
        let rotated = q.rotate_vector(v);
        
        // Should rotate X to -Z
        assert!((rotated.x - 0.0).abs() < 0.001);
        assert!((rotated.y - 0.0).abs() < 0.001);
        assert!((rotated.z - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_matrix_identity() {
        let identity = Mat4::identity();
        let point = Vec3::new(1.0, 2.0, 3.0);
        let transformed = identity.transform_point(point);
        assert_eq!(transformed, point);
    }

    #[test]
    fn test_matrix_translation() {
        let translation = Mat4::translation(Vec3::new(5.0, 10.0, 15.0));
        let point = Vec3::new(1.0, 2.0, 3.0);
        let transformed = translation.transform_point(point);
        assert_eq!(transformed, Vec3::new(6.0, 12.0, 18.0));
    }
}

#[cfg(test)]
mod graphics_tests {
    use aether_lang::stdlib::graphics::*;

    #[test]
    fn test_graphics_context_creation() {
        let context = create_graphics_context(GraphicsBackend::WebGPU);
        assert!(context.is_ok());
    }

    #[test]
    fn test_mesh_cube_generation() {
        let cube = Mesh::cube();
        assert!(!cube.vertices.is_empty());
        assert!(!cube.indices.is_empty());
    }

    #[test]
    fn test_mesh_sphere_generation() {
        let sphere = Mesh::sphere(1.0, 8);
        assert!(!sphere.vertices.is_empty());
        assert!(!sphere.indices.is_empty());
        
        // Check that vertices are approximately on unit sphere
        for vertex in &sphere.vertices {
            let distance = vertex.position.length();
            assert!((distance - 1.0).abs() < 0.1); // Allow some tolerance
        }
    }
}

#[cfg(test)]
mod physics_tests {
    use aether_lang::stdlib::physics::*;
    use aether_lang::stdlib::math3d::*;

    #[test]
    fn test_physics_world_creation() {
        let world = PhysicsWorld::new();
        // Should create without errors
        assert_eq!(world.query_sphere(Vec3::ZERO, 1.0).len(), 0);
    }

    #[test]
    fn test_rigid_body_creation() {
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
    }

    #[test]
    fn test_physics_material_default() {
        let material = PhysicsMaterial::default();
        assert_eq!(material.density, 1.0);
        assert_eq!(material.restitution, 0.5);
        assert_eq!(material.static_friction, 0.6);
        assert_eq!(material.dynamic_friction, 0.4);
    }

    #[test]
    fn test_raycast_basic() {
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
    fn test_sphere_query() {
        let mut world = PhysicsWorld::new();
        
        let shape = CollisionShape::Sphere { radius: 0.5 };
        let material = PhysicsMaterial::default();
        
        // Create bodies at different positions
        let body1 = world.create_body(shape.clone(), material.clone(), Vec3::new(0.0, 0.0, 0.0), Quat::identity());
        let body2 = world.create_body(shape.clone(), material.clone(), Vec3::new(2.0, 0.0, 0.0), Quat::identity());
        let _body3 = world.create_body(shape, material, Vec3::new(10.0, 0.0, 0.0), Quat::identity());
        
        // Query sphere around origin
        let results = world.query_sphere(Vec3::ZERO, 3.0);
        
        // Should find body1 and body2, but not body3
        assert_eq!(results.len(), 2);
        assert!(results.contains(&body1));
        assert!(results.contains(&body2));
    }
}

#[cfg(test)]
mod integration_tests {
    use aether_lang::stdlib::ecs::*;
    use aether_lang::stdlib::math3d::*;
    use aether_lang::stdlib::physics::*;

    #[test]
    fn test_ecs_physics_integration() {
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
        
        // Verify initial state
        let initial_transform = ecs_world.get_component::<Transform>(entity).unwrap();
        let initial_body = physics_world.get_body(body_id).unwrap();
        
        assert_eq!(initial_transform.position, initial_body.position);
        assert_eq!(initial_transform.rotation, initial_body.rotation);
    }

    #[test]
    fn test_movement_system() {
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
}