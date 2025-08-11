// 3D mathematics library for game development and simulation
// Provides vectors, matrices, quaternions with SIMD optimizations

use std::ops::{Add, Sub, Mul, Div, Neg};

/// 3D Vector with SIMD-friendly alignment
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    _padding: f32, // For SIMD alignment
}

/// 4D Vector for homogeneous coordinates
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

/// 2D Vector for UI and texture coordinates
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

/// Quaternion for rotations
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

/// 4x4 Matrix for transformations
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub m: [[f32; 4]; 4],
}

/// 3x3 Matrix for rotations and scaling
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m: [[f32; 3]; 3],
}

// Vec3 Implementation
impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0, _padding: 0.0 };
    pub const ONE: Vec3 = Vec3 { x: 1.0, y: 1.0, z: 1.0, _padding: 0.0 };
    pub const X: Vec3 = Vec3 { x: 1.0, y: 0.0, z: 0.0, _padding: 0.0 };
    pub const Y: Vec3 = Vec3 { x: 0.0, y: 1.0, z: 0.0, _padding: 0.0 };
    pub const Z: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 1.0, _padding: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z, _padding: 0.0 }
    }

    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalize(self) -> Vec3 {
        let len = self.length();
        if len > 0.0 {
            Vec3::new(self.x / len, self.y / len, self.z / len)
        } else {
            Vec3::ZERO
        }
    }

    pub fn distance(self, other: Vec3) -> f32 {
        (self - other).length()
    }

    pub fn lerp(self, other: Vec3, t: f32) -> Vec3 {
        self + (other - self) * t
    }

    pub fn reflect(self, normal: Vec3) -> Vec3 {
        self - normal * (2.0 * self.dot(normal))
    }

    pub fn project(self, onto: Vec3) -> Vec3 {
        let dot = self.dot(onto);
        let len_sq = onto.length_squared();
        if len_sq > 0.0 {
            onto * (dot / len_sq)
        } else {
            Vec3::ZERO
        }
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, scalar: f32) -> Vec3 {
        Vec3::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, vec: Vec3) -> Vec3 {
        vec * self
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, scalar: f32) -> Vec3 {
        Vec3::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

// Vec4 Implementation
impl Vec4 {
    pub const ZERO: Vec4 = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    pub const ONE: Vec4 = Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };

    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vec4 { x, y, z, w }
    }

    pub fn from_vec3(v: Vec3, w: f32) -> Self {
        Vec4::new(v.x, v.y, v.z, w)
    }

    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    pub fn dot(self, other: Vec4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Vec4 {
        let len = self.length();
        if len > 0.0 {
            Vec4::new(self.x / len, self.y / len, self.z / len, self.w / len)
        } else {
            Vec4::ZERO
        }
    }
}

// Vec2 Implementation
impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };
    pub const ONE: Vec2 = Vec2 { x: 1.0, y: 1.0 };
    pub const X: Vec2 = Vec2 { x: 1.0, y: 0.0 };
    pub const Y: Vec2 = Vec2 { x: 0.0, y: 1.0 };

    pub fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }

    pub fn dot(self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalize(self) -> Vec2 {
        let len = self.length();
        if len > 0.0 {
            Vec2::new(self.x / len, self.y / len)
        } else {
            Vec2::ZERO
        }
    }
}

// Quaternion Implementation
impl Quat {
    pub const IDENTITY: Quat = Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Quat { x, y, z, w }
    }

    pub fn identity() -> Self {
        Quat::IDENTITY
    }

    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let half_angle = angle * 0.5;
        let sin_half = half_angle.sin();
        let cos_half = half_angle.cos();
        let normalized_axis = axis.normalize();

        Quat::new(
            normalized_axis.x * sin_half,
            normalized_axis.y * sin_half,
            normalized_axis.z * sin_half,
            cos_half,
        )
    }

    pub fn from_euler(pitch: f32, yaw: f32, roll: f32) -> Self {
        let half_pitch = pitch * 0.5;
        let half_yaw = yaw * 0.5;
        let half_roll = roll * 0.5;

        let cp = half_pitch.cos();
        let sp = half_pitch.sin();
        let cy = half_yaw.cos();
        let sy = half_yaw.sin();
        let cr = half_roll.cos();
        let sr = half_roll.sin();

        Quat::new(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    }

    pub fn dot(self, other: Quat) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Quat {
        let len = self.length();
        if len > 0.0 {
            Quat::new(self.x / len, self.y / len, self.z / len, self.w / len)
        } else {
            Quat::IDENTITY
        }
    }

    pub fn conjugate(self) -> Quat {
        Quat::new(-self.x, -self.y, -self.z, self.w)
    }

    pub fn inverse(self) -> Quat {
        let len_sq = self.dot(self);
        if len_sq > 0.0 {
            let conj = self.conjugate();
            Quat::new(conj.x / len_sq, conj.y / len_sq, conj.z / len_sq, conj.w / len_sq)
        } else {
            Quat::IDENTITY
        }
    }

    pub fn rotate_vector(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let uv = qv.cross(v);
        let uuv = qv.cross(uv);
        v + (uv * self.w + uuv) * 2.0
    }

    pub fn slerp(self, other: Quat, t: f32) -> Quat {
        let mut dot = self.dot(other);
        let mut other = other;

        // If dot product is negative, slerp won't take the shorter path
        if dot < 0.0 {
            other = Quat::new(-other.x, -other.y, -other.z, -other.w);
            dot = -dot;
        }

        if dot > 0.9995 {
            // Linear interpolation for very close quaternions
            let result = Quat::new(
                self.x + t * (other.x - self.x),
                self.y + t * (other.y - self.y),
                self.z + t * (other.z - self.z),
                self.w + t * (other.w - self.w),
            );
            result.normalize()
        } else {
            let theta_0 = dot.acos();
            let theta = theta_0 * t;
            let sin_theta = theta.sin();
            let sin_theta_0 = theta_0.sin();

            let s0 = (theta_0 - theta).cos() - dot * sin_theta / sin_theta_0;
            let s1 = sin_theta / sin_theta_0;

            Quat::new(
                s0 * self.x + s1 * other.x,
                s0 * self.y + s1 * other.y,
                s0 * self.z + s1 * other.z,
                s0 * self.w + s1 * other.w,
            )
        }
    }
}

impl Mul for Quat {
    type Output = Quat;
    fn mul(self, other: Quat) -> Quat {
        Quat::new(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )
    }
}

// Mat3 Implementation
impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3 {
        m: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    };

    pub const ZERO: Mat3 = Mat3 {
        m: [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    };

    pub fn identity() -> Self {
        Mat3::IDENTITY
    }

    pub fn transpose(self) -> Mat3 {
        Mat3 {
            m: [
                [self.m[0][0], self.m[1][0], self.m[2][0]],
                [self.m[0][1], self.m[1][1], self.m[2][1]],
                [self.m[0][2], self.m[1][2], self.m[2][2]],
            ],
        }
    }
}

impl Mul for Mat3 {
    type Output = Mat3;
    fn mul(self, other: Mat3) -> Mat3 {
        let mut result = Mat3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    result.m[i][j] += self.m[i][k] * other.m[k][j];
                }
            }
        }
        result
    }
}

// Mat4 Implementation
impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4 {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    pub const ZERO: Mat4 = Mat4 {
        m: [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    };

    pub fn identity() -> Self {
        Mat4::IDENTITY
    }

    pub fn translation(v: Vec3) -> Self {
        let mut result = Mat4::IDENTITY;
        result.m[0][3] = v.x;
        result.m[1][3] = v.y;
        result.m[2][3] = v.z;
        result
    }

    pub fn scale(v: Vec3) -> Self {
        let mut result = Mat4::IDENTITY;
        result.m[0][0] = v.x;
        result.m[1][1] = v.y;
        result.m[2][2] = v.z;
        result
    }

    pub fn from_quat(q: Quat) -> Self {
        let q = q.normalize();
        let xx = q.x * q.x;
        let yy = q.y * q.y;
        let zz = q.z * q.z;
        let xy = q.x * q.y;
        let xz = q.x * q.z;
        let yz = q.y * q.z;
        let wx = q.w * q.x;
        let wy = q.w * q.y;
        let wz = q.w * q.z;

        Mat4 {
            m: [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let tan_half_fov = (fov * 0.5).tan();
        let mut result = Mat4::ZERO;
        
        result.m[0][0] = 1.0 / (aspect * tan_half_fov);
        result.m[1][1] = 1.0 / tan_half_fov;
        result.m[2][2] = -(far + near) / (far - near);
        result.m[2][3] = -(2.0 * far * near) / (far - near);
        result.m[3][2] = -1.0;
        
        result
    }

    pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Mat4 {
            m: [
                [s.x, s.y, s.z, -s.dot(eye)],
                [u.x, u.y, u.z, -u.dot(eye)],
                [-f.x, -f.y, -f.z, f.dot(eye)],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn transpose(self) -> Mat4 {
        Mat4 {
            m: [
                [self.m[0][0], self.m[1][0], self.m[2][0], self.m[3][0]],
                [self.m[0][1], self.m[1][1], self.m[2][1], self.m[3][1]],
                [self.m[0][2], self.m[1][2], self.m[2][2], self.m[3][2]],
                [self.m[0][3], self.m[1][3], self.m[2][3], self.m[3][3]],
            ],
        }
    }

    pub fn determinant(self) -> f32 {
        let m = &self.m;
        m[0][0] * (
            m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) -
            m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) +
            m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])
        ) - m[0][1] * (
            m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) -
            m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0]) +
            m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])
        ) + m[0][2] * (
            m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) -
            m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0]) +
            m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0])
        ) - m[0][3] * (
            m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]) -
            m[1][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]) +
            m[1][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0])
        )
    }

    pub fn transform_point(self, point: Vec3) -> Vec3 {
        let v = Vec4::from_vec3(point, 1.0);
        let result = self * v;
        if result.w != 0.0 {
            result.xyz() / result.w
        } else {
            result.xyz()
        }
    }

    pub fn transform_vector(self, vector: Vec3) -> Vec3 {
        let v = Vec4::from_vec3(vector, 0.0);
        let result = self * v;
        result.xyz()
    }
}

impl Mul for Mat4 {
    type Output = Mat4;
    fn mul(self, other: Mat4) -> Mat4 {
        let mut result = Mat4::ZERO;
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result.m[i][j] += self.m[i][k] * other.m[k][j];
                }
            }
        }
        result
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    fn mul(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z + self.m[0][3] * v.w,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z + self.m[1][3] * v.w,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z + self.m[2][3] * v.w,
            self.m[3][0] * v.x + self.m[3][1] * v.y + self.m[3][2] * v.z + self.m[3][3] * v.w,
        )
    }
}

// Utility functions
pub fn degrees_to_radians(degrees: f32) -> f32 {
    degrees * std::f32::consts::PI / 180.0
}

pub fn radians_to_degrees(radians: f32) -> f32 {
    radians * 180.0 / std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_operations() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        
        assert_eq!(v1 + v2, Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(v1 - v2, Vec3::new(-3.0, -3.0, -3.0));
        assert_eq!(v1 * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(v1.dot(v2), 32.0);
        
        let cross = v1.cross(v2);
        assert_eq!(cross, Vec3::new(-3.0, 6.0, -3.0));
    }

    #[test]
    fn test_quaternion_rotation() {
        let q = Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI / 2.0);
        let v = Vec3::X;
        let rotated = q.rotate_vector(v);
        
        // Rotating X around Y by 90 degrees should give -Z
        assert!((rotated.x - 0.0).abs() < 0.001);
        assert!((rotated.y - 0.0).abs() < 0.001);
        assert!((rotated.z - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_matrix_multiplication() {
        let m1 = Mat4::identity();
        let m2 = Mat4::translation(Vec3::new(1.0, 2.0, 3.0));
        let result = m1 * m2;
        
        assert_eq!(result.m[0][3], 1.0);
        assert_eq!(result.m[1][3], 2.0);
        assert_eq!(result.m[2][3], 3.0);
    }

    #[test]
    fn test_perspective_matrix() {
        let fov = degrees_to_radians(45.0);
        let aspect = 16.0 / 9.0;
        let near = 0.1;
        let far = 100.0;
        
        let proj = Mat4::perspective(fov, aspect, near, far);
        
        // Test that the matrix is not zero
        assert!(proj.determinant() != 0.0);
    }
}