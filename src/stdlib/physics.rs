// Real-time physics simulation primitives for game development
// Provides collision detection, rigid body dynamics, and constraint solving

use crate::stdlib::math3d::{Vec3, Quat, Mat3, Mat4};
use std::collections::HashMap;

/// Physics world containing all rigid bodies and constraints
pub struct PhysicsWorld {
    gravity: Vec3,
    bodies: HashMap<BodyId, RigidBody>,
    constraints: Vec<Constraint>,
    collision_pairs: Vec<CollisionPair>,
    next_body_id: u64,
    time_step: f32,
    substeps: u32,
}

/// Unique identifier for rigid bodies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyId(pub u64);

/// Rigid body representation
#[derive(Debug, Clone)]
pub struct RigidBody {
    pub id: BodyId,
    pub position: Vec3,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass: f32,
    pub inverse_mass: f32,
    pub inertia_tensor: Mat3,
    pub inverse_inertia_tensor: Mat3,
    pub restitution: f32,
    pub friction: f32,
    pub is_kinematic: bool,
    pub is_sleeping: bool,
    pub shape: CollisionShape,
    pub material: PhysicsMaterial,
}

/// Collision shapes
#[derive(Debug, Clone)]
pub enum CollisionShape {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Capsule { radius: f32, height: f32 },
    Plane { normal: Vec3, distance: f32 },
    Mesh { vertices: Vec<Vec3>, indices: Vec<u32> },
}

/// Physics material properties
#[derive(Debug, Clone)]
pub struct PhysicsMaterial {
    pub density: f32,
    pub restitution: f32,
    pub static_friction: f32,
    pub dynamic_friction: f32,
}

/// Constraint types for connecting bodies
#[derive(Debug, Clone)]
pub enum Constraint {
    Distance {
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest_length: f32,
        stiffness: f32,
        damping: f32,
    },
    Hinge {
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        min_angle: f32,
        max_angle: f32,
    },
    Fixed {
        body_a: BodyId,
        body_b: BodyId,
        offset: Mat4,
    },
}

/// Collision detection result
#[derive(Debug, Clone)]
pub struct CollisionPair {
    pub body_a: BodyId,
    pub body_b: BodyId,
    pub contacts: Vec<ContactPoint>,
}

#[derive(Debug, Clone)]
pub struct ContactPoint {
    pub position: Vec3,
    pub normal: Vec3,
    pub penetration: f32,
    pub impulse: f32,
}

/// Ray casting
#[derive(Debug, Clone)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub max_distance: f32,
}

#[derive(Debug, Clone)]
pub struct RaycastHit {
    pub body_id: BodyId,
    pub position: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

impl PhysicsWorld {
    /// Create a new physics world
    pub fn new() -> Self {
        PhysicsWorld {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            bodies: HashMap::new(),
            constraints: Vec::new(),
            collision_pairs: Vec::new(),
            next_body_id: 1,
            time_step: 1.0 / 60.0,
            substeps: 4,
        }
    }

    /// Set gravity vector
    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.gravity = gravity;
    }

    /// Create a new rigid body
    pub fn create_body(&mut self, shape: CollisionShape, material: PhysicsMaterial, position: Vec3, rotation: Quat) -> BodyId {
        let id = BodyId(self.next_body_id);
        self.next_body_id += 1;

        let mass = Self::calculate_mass(&shape, material.density);
        let inverse_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let inertia_tensor = Self::calculate_inertia_tensor(&shape, mass);
        let inverse_inertia_tensor = Self::invert_matrix3(inertia_tensor);

        let body = RigidBody {
            id,
            position,
            rotation,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass,
            inverse_mass,
            inertia_tensor,
            inverse_inertia_tensor,
            restitution: material.restitution,
            friction: (material.static_friction + material.dynamic_friction) * 0.5,
            is_kinematic: false,
            is_sleeping: false,
            shape,
            material,
        };

        self.bodies.insert(id, body);
        id
    }

    /// Remove a rigid body
    pub fn remove_body(&mut self, id: BodyId) {
        self.bodies.remove(&id);
        self.constraints.retain(|constraint| {
            match constraint {
                Constraint::Distance { body_a, body_b, .. } => *body_a != id && *body_b != id,
                Constraint::Hinge { body_a, body_b, .. } => *body_a != id && *body_b != id,
                Constraint::Fixed { body_a, body_b, .. } => *body_a != id && *body_b != id,
            }
        });
    }

    /// Get a rigid body by ID
    pub fn get_body(&self, id: BodyId) -> Option<&RigidBody> {
        self.bodies.get(&id)
    }

    /// Get a mutable rigid body by ID
    pub fn get_body_mut(&mut self, id: BodyId) -> Option<&mut RigidBody> {
        self.bodies.get_mut(&id)
    }

    /// Add a constraint between two bodies
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Apply force to a rigid body
    pub fn apply_force(&mut self, id: BodyId, force: Vec3, point: Option<Vec3>) {
        if let Some(body) = self.bodies.get_mut(&id) {
            if !body.is_kinematic && body.inverse_mass > 0.0 {
                body.linear_velocity = body.linear_velocity + force * body.inverse_mass * self.time_step;
                
                if let Some(point) = point {
                    let torque = (point - body.position).cross(force);
                    let world_inertia = body.inverse_inertia_tensor; // Use stored tensor directly
                    let angular_acceleration = Self::transform_vector_by_matrix3(world_inertia, torque);
                    body.angular_velocity = body.angular_velocity + angular_acceleration * self.time_step;
                }
            }
        }
    }

    /// Apply impulse to a rigid body
    pub fn apply_impulse(&mut self, id: BodyId, impulse: Vec3, point: Option<Vec3>) {
        if let Some(body) = self.bodies.get_mut(&id) {
            if !body.is_kinematic && body.inverse_mass > 0.0 {
                body.linear_velocity = body.linear_velocity + impulse * body.inverse_mass;
                
                if let Some(point) = point {
                    let torque = (point - body.position).cross(impulse);
                    let world_inertia = body.inverse_inertia_tensor; // Use stored tensor directly
                    let angular_impulse = Self::transform_vector_by_matrix3(world_inertia, torque);
                    body.angular_velocity = body.angular_velocity + angular_impulse;
                }
            }
        }
    }

    /// Step the physics simulation
    pub fn step(&mut self, delta_time: f32) {
        let substep_time = delta_time / self.substeps as f32;
        
        for _ in 0..self.substeps {
            self.integrate_forces(substep_time);
            self.detect_collisions();
            self.solve_constraints(substep_time);
            self.integrate_velocities(substep_time);
        }
    }

    /// Raycast against all bodies in the world
    pub fn raycast(&self, ray: Ray) -> Option<RaycastHit> {
        let mut closest_hit: Option<RaycastHit> = None;
        let mut closest_distance = ray.max_distance;

        for body in self.bodies.values() {
            if let Some(hit) = self.raycast_shape(&ray, body) {
                if hit.distance < closest_distance {
                    closest_distance = hit.distance;
                    closest_hit = Some(hit);
                }
            }
        }

        closest_hit
    }

    /// Query bodies within a sphere
    pub fn query_sphere(&self, center: Vec3, radius: f32) -> Vec<BodyId> {
        let mut results = Vec::new();
        
        for body in self.bodies.values() {
            if self.sphere_intersects_shape(center, radius, body) {
                results.push(body.id);
            }
        }
        
        results
    }

    // Private implementation methods

    fn integrate_forces(&mut self, dt: f32) {
        for body in self.bodies.values_mut() {
            if !body.is_kinematic && body.inverse_mass > 0.0 && !body.is_sleeping {
                // Apply gravity
                body.linear_velocity = body.linear_velocity + self.gravity * dt;
                
                // Apply damping
                body.linear_velocity = body.linear_velocity * 0.99;
                body.angular_velocity = body.angular_velocity * 0.99;
            }
        }
    }

    fn integrate_velocities(&mut self, dt: f32) {
        for body in self.bodies.values_mut() {
            if !body.is_kinematic && !body.is_sleeping {
                // Update position
                body.position = body.position + body.linear_velocity * dt;
                
                // Update rotation
                if body.angular_velocity.length() > 0.001 {
                    let angular_speed = body.angular_velocity.length();
                    let axis = body.angular_velocity.normalize();
                    let rotation_delta = Quat::from_axis_angle(axis, angular_speed * dt);
                    body.rotation = rotation_delta * body.rotation;
                    body.rotation = body.rotation.normalize();
                }
            }
        }
    }

    fn detect_collisions(&mut self) {
        self.collision_pairs.clear();
        
        let body_ids: Vec<BodyId> = self.bodies.keys().cloned().collect();
        
        for i in 0..body_ids.len() {
            for j in (i + 1)..body_ids.len() {
                let id_a = body_ids[i];
                let id_b = body_ids[j];
                
                if let (Some(body_a), Some(body_b)) = (self.bodies.get(&id_a), self.bodies.get(&id_b)) {
                    if let Some(contacts) = self.detect_collision(body_a, body_b) {
                        self.collision_pairs.push(CollisionPair {
                            body_a: id_a,
                            body_b: id_b,
                            contacts,
                        });
                    }
                }
            }
        }
    }

    fn detect_collision(&self, body_a: &RigidBody, body_b: &RigidBody) -> Option<Vec<ContactPoint>> {
        match (&body_a.shape, &body_b.shape) {
            (CollisionShape::Sphere { radius: r1 }, CollisionShape::Sphere { radius: r2 }) => {
                self.sphere_sphere_collision(body_a, *r1, body_b, *r2)
            }
            (CollisionShape::Box { half_extents: he1 }, CollisionShape::Box { half_extents: he2 }) => {
                self.box_box_collision(body_a, *he1, body_b, *he2)
            }
            (CollisionShape::Sphere { radius }, CollisionShape::Box { half_extents }) => {
                self.sphere_box_collision(body_a, *radius, body_b, *half_extents)
            }
            (CollisionShape::Box { half_extents }, CollisionShape::Sphere { radius }) => {
                self.sphere_box_collision(body_b, *radius, body_a, *half_extents)
                    .map(|mut contacts| {
                        for contact in &mut contacts {
                            contact.normal = -contact.normal;
                        }
                        contacts
                    })
            }
            _ => None, // Other collision combinations not implemented
        }
    }

    fn sphere_sphere_collision(&self, body_a: &RigidBody, radius_a: f32, body_b: &RigidBody, radius_b: f32) -> Option<Vec<ContactPoint>> {
        let distance_vec = body_b.position - body_a.position;
        let distance = distance_vec.length();
        let combined_radius = radius_a + radius_b;
        
        if distance < combined_radius && distance > 0.001 {
            let normal = distance_vec.normalize();
            let penetration = combined_radius - distance;
            let contact_point = body_a.position + normal * radius_a;
            
            Some(vec![ContactPoint {
                position: contact_point,
                normal,
                penetration,
                impulse: 0.0,
            }])
        } else {
            None
        }
    }

    fn box_box_collision(&self, _body_a: &RigidBody, _he_a: Vec3, _body_b: &RigidBody, _he_b: Vec3) -> Option<Vec<ContactPoint>> {
        // Simplified box-box collision (SAT algorithm would be implemented here)
        None
    }

    fn sphere_box_collision(&self, _sphere_body: &RigidBody, _radius: f32, _box_body: &RigidBody, _half_extents: Vec3) -> Option<Vec<ContactPoint>> {
        // Simplified sphere-box collision
        None
    }

    fn solve_constraints(&mut self, dt: f32) {
        // Solve collision constraints
        let collision_pairs = std::mem::take(&mut self.collision_pairs);
        for mut collision in collision_pairs {
            self.solve_collision_constraint(&mut collision, dt);
        }
        
        // Solve user constraints
        let constraints = self.constraints.clone();
        for constraint in &constraints {
            self.solve_user_constraint(constraint, dt);
        }
    }

    fn solve_collision_constraint(&mut self, collision: &mut CollisionPair, _dt: f32) {
        let body_a_id = collision.body_a;
        let body_b_id = collision.body_b;
        
        for contact in &mut collision.contacts {
            // Get bodies (we need to be careful about borrowing here)
            let (restitution, _friction) = {
                let body_a = self.bodies.get(&body_a_id).unwrap();
                let body_b = self.bodies.get(&body_b_id).unwrap();
                ((body_a.restitution + body_b.restitution) * 0.5, (body_a.friction + body_b.friction) * 0.5)
            };
            
            // Calculate relative velocity
            let (rel_velocity, inv_mass_sum) = {
                let body_a = self.bodies.get(&body_a_id).unwrap();
                let body_b = self.bodies.get(&body_b_id).unwrap();
                
                let vel_a = body_a.linear_velocity;
                let vel_b = body_b.linear_velocity;
                let rel_vel = vel_b - vel_a;
                let inv_mass = body_a.inverse_mass + body_b.inverse_mass;
                
                (rel_vel, inv_mass)
            };
            
            // Calculate impulse
            let normal_velocity = rel_velocity.dot(contact.normal);
            if normal_velocity > 0.0 {
                continue; // Bodies separating
            }
            
            let impulse_magnitude = -(1.0 + restitution) * normal_velocity / inv_mass_sum;
            let impulse = contact.normal * impulse_magnitude;
            
            // Apply impulse
            {
                let body_a = self.bodies.get_mut(&body_a_id).unwrap();
                if !body_a.is_kinematic {
                    body_a.linear_velocity = body_a.linear_velocity - impulse * body_a.inverse_mass;
                }
            }
            
            {
                let body_b = self.bodies.get_mut(&body_b_id).unwrap();
                if !body_b.is_kinematic {
                    body_b.linear_velocity = body_b.linear_velocity + impulse * body_b.inverse_mass;
                }
            }
            
            // Position correction for penetration
            let correction = contact.normal * (contact.penetration * 0.8 / inv_mass_sum);
            
            {
                let body_a = self.bodies.get_mut(&body_a_id).unwrap();
                if !body_a.is_kinematic {
                    body_a.position = body_a.position - correction * body_a.inverse_mass;
                }
            }
            
            {
                let body_b = self.bodies.get_mut(&body_b_id).unwrap();
                if !body_b.is_kinematic {
                    body_b.position = body_b.position + correction * body_b.inverse_mass;
                }
            }
            
            contact.impulse = impulse_magnitude;
        }
    }

    fn solve_user_constraint(&mut self, constraint: &Constraint, _dt: f32) {
        match constraint {
            Constraint::Distance { body_a, body_b, anchor_a, anchor_b, rest_length, stiffness, damping: _ } => {
                // Simplified distance constraint solving
                if let (Some(body_a_data), Some(body_b_data)) = (self.bodies.get(body_a), self.bodies.get(body_b)) {
                    let pos_a = body_a_data.position + *anchor_a;
                    let pos_b = body_b_data.position + *anchor_b;
                    let distance_vec = pos_b - pos_a;
                    let current_length = distance_vec.length();
                    
                    if current_length > 0.001 {
                        let difference = current_length - rest_length;
                        let force_magnitude = difference * stiffness;
                        let _force = distance_vec.normalize() * force_magnitude;
                        
                        // Apply forces (simplified)
                        // In a real implementation, we'd properly handle the constraint
                    }
                }
            }
            _ => {} // Other constraint types not fully implemented
        }
    }

    fn raycast_shape(&self, ray: &Ray, body: &RigidBody) -> Option<RaycastHit> {
        match &body.shape {
            CollisionShape::Sphere { radius } => {
                self.raycast_sphere(ray, body.position, *radius, body.id)
            }
            CollisionShape::Box { half_extents } => {
                self.raycast_box(ray, body.position, body.rotation, *half_extents, body.id)
            }
            _ => None, // Other shapes not implemented
        }
    }

    fn raycast_sphere(&self, ray: &Ray, center: Vec3, radius: f32, body_id: BodyId) -> Option<RaycastHit> {
        let oc = ray.origin - center;
        let a = ray.direction.dot(ray.direction);
        let b = 2.0 * oc.dot(ray.direction);
        let c = oc.dot(oc) - radius * radius;
        let discriminant: f32 = b * b - 4.0 * a * c;
        
        if discriminant >= 0.0 {
            let t = (-b - discriminant.sqrt()) / (2.0 * a);
            if t >= 0.0 && t <= ray.max_distance {
                let hit_point = ray.origin + ray.direction * t;
                let normal = (hit_point - center).normalize();
                
                return Some(RaycastHit {
                    body_id,
                    position: hit_point,
                    normal,
                    distance: t,
                });
            }
        }
        
        None
    }

    fn raycast_box(&self, _ray: &Ray, _position: Vec3, _rotation: Quat, _half_extents: Vec3, _body_id: BodyId) -> Option<RaycastHit> {
        // Box raycast not implemented
        None
    }

    fn sphere_intersects_shape(&self, center: Vec3, radius: f32, body: &RigidBody) -> bool {
        match &body.shape {
            CollisionShape::Sphere { radius: body_radius } => {
                let distance = (body.position - center).length();
                distance <= radius + body_radius
            }
            _ => false, // Other shapes not implemented
        }
    }

    // Utility functions

    fn calculate_mass(shape: &CollisionShape, density: f32) -> f32 {
        match shape {
            CollisionShape::Sphere { radius } => {
                (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3) * density
            }
            CollisionShape::Box { half_extents } => {
                8.0 * half_extents.x * half_extents.y * half_extents.z * density
            }
            CollisionShape::Capsule { radius, height } => {
                let cylinder_volume = std::f32::consts::PI * radius.powi(2) * height;
                let sphere_volume = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
                (cylinder_volume + sphere_volume) * density
            }
            _ => 1.0, // Default mass
        }
    }

    fn calculate_inertia_tensor(shape: &CollisionShape, mass: f32) -> Mat3 {
        match shape {
            CollisionShape::Sphere { radius } => {
                let inertia = 0.4 * mass * radius.powi(2);
                Mat3 {
                    m: [
                        [inertia, 0.0, 0.0],
                        [0.0, inertia, 0.0],
                        [0.0, 0.0, inertia],
                    ],
                }
            }
            CollisionShape::Box { half_extents } => {
                let x2 = half_extents.x * half_extents.x;
                let y2 = half_extents.y * half_extents.y;
                let z2 = half_extents.z * half_extents.z;
                
                Mat3 {
                    m: [
                        [(mass / 3.0) * (y2 + z2), 0.0, 0.0],
                        [0.0, (mass / 3.0) * (x2 + z2), 0.0],
                        [0.0, 0.0, (mass / 3.0) * (x2 + y2)],
                    ],
                }
            }
            _ => Mat3 {
                m: [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        }
    }

    fn invert_matrix3(m: Mat3) -> Mat3 {
        // Simplified 3x3 matrix inversion
        let det = m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1])
                - m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0])
                + m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);
        
        if det.abs() < 0.001 {
            return Mat3 {
                m: [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            };
        }
        
        let inv_det = 1.0 / det;
        
        Mat3 {
            m: [
                [(m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) * inv_det,
                 (m.m[0][2] * m.m[2][1] - m.m[0][1] * m.m[2][2]) * inv_det,
                 (m.m[0][1] * m.m[1][2] - m.m[0][2] * m.m[1][1]) * inv_det],
                [(m.m[1][2] * m.m[2][0] - m.m[1][0] * m.m[2][2]) * inv_det,
                 (m.m[0][0] * m.m[2][2] - m.m[0][2] * m.m[2][0]) * inv_det,
                 (m.m[0][2] * m.m[1][0] - m.m[0][0] * m.m[1][2]) * inv_det],
                [(m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]) * inv_det,
                 (m.m[0][1] * m.m[2][0] - m.m[0][0] * m.m[2][1]) * inv_det,
                 (m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0]) * inv_det],
            ],
        }
    }



    fn transform_vector_by_matrix3(matrix: Mat3, vector: Vec3) -> Vec3 {
        Vec3::new(
            matrix.m[0][0] * vector.x + matrix.m[0][1] * vector.y + matrix.m[0][2] * vector.z,
            matrix.m[1][0] * vector.x + matrix.m[1][1] * vector.y + matrix.m[1][2] * vector.z,
            matrix.m[2][0] * vector.x + matrix.m[2][1] * vector.y + matrix.m[2][2] * vector.z,
        )
    }
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        PhysicsMaterial {
            density: 1.0,
            restitution: 0.5,
            static_friction: 0.6,
            dynamic_friction: 0.4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_world_creation() {
        let world = PhysicsWorld::new();
        assert_eq!(world.gravity, Vec3::new(0.0, -9.81, 0.0));
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
    fn test_sphere_collision_detection() {
        let mut world = PhysicsWorld::new();
        
        let shape1 = CollisionShape::Sphere { radius: 1.0 };
        let shape2 = CollisionShape::Sphere { radius: 1.0 };
        let material = PhysicsMaterial::default();
        
        let body1 = world.create_body(shape1, material.clone(), Vec3::new(0.0, 0.0, 0.0), Quat::identity());
        let body2 = world.create_body(shape2, material, Vec3::new(1.5, 0.0, 0.0), Quat::identity());
        
        world.detect_collisions();
        
        assert_eq!(world.collision_pairs.len(), 1);
        assert_eq!(world.collision_pairs[0].body_a, body1);
        assert_eq!(world.collision_pairs[0].body_b, body2);
    }

    #[test]
    fn test_raycast() {
        let mut world = PhysicsWorld::new();
        
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let material = PhysicsMaterial::default();
        let body_id = world.create_body(shape, material, Vec3::new(0.0, 0.0, 0.0), Quat::identity());
        
        let ray = Ray {
            origin: Vec3::new(-5.0, 0.0, 0.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
            max_distance: 10.0,
        };
        
        let hit = world.raycast(ray);
        assert!(hit.is_some());
        
        let hit = hit.unwrap();
        assert_eq!(hit.body_id, body_id);
        assert!(hit.distance < 5.0);
    }

    #[test]
    fn test_physics_step() {
        let mut world = PhysicsWorld::new();
        
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let material = PhysicsMaterial::default();
        let body_id = world.create_body(shape, material, Vec3::new(0.0, 10.0, 0.0), Quat::identity());
        
        let initial_position = world.get_body(body_id).unwrap().position;
        
        world.step(1.0 / 60.0);
        
        let final_position = world.get_body(body_id).unwrap().position;
        
        // Body should have fallen due to gravity
        assert!(final_position.y < initial_position.y);
    }
}