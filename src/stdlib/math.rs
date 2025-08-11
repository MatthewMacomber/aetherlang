// Aether Standard Library - Mathematical Functions
// Core mathematical operations with automatic differentiation support

use crate::compiler::autodiff::{DiffMarker, DiffMode, DiffDirection};

/// Mathematical constants
pub mod constants {
    pub const PI: f64 = std::f64::consts::PI;
    pub const E: f64 = std::f64::consts::E;
    pub const TAU: f64 = std::f64::consts::TAU;
    pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
    pub const LN_2: f64 = std::f64::consts::LN_2;
    pub const LN_10: f64 = std::f64::consts::LN_10;
}

/// Differentiable scalar mathematical functions
pub struct DiffScalar {
    value: f64,
    gradient: Option<f64>,
    diff_marker: Option<DiffMarker>,
}

impl DiffScalar {
    /// Create a new differentiable scalar
    pub fn new(value: f64) -> Self {
        Self {
            value,
            gradient: None,
            diff_marker: None,
        }
    }

    /// Create a differentiable scalar with gradient tracking
    pub fn with_gradient(value: f64, gradient: f64) -> Self {
        Self {
            value,
            gradient: Some(gradient),
            diff_marker: Some(DiffMarker {
                wrt_vars: vec!["x".to_string()],
                direction: DiffDirection::Forward,
                order: 1,
                mode: DiffMode::Dynamic,
            }),
        }
    }

    /// Get the value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get the gradient if available
    pub fn gradient(&self) -> Option<f64> {
        self.gradient
    }

    /// Mark for differentiation
    pub fn mark_for_diff(mut self, marker: DiffMarker) -> Self {
        self.diff_marker = Some(marker);
        self
    }
}

/// Basic arithmetic operations with AD support
impl std::ops::Add for DiffScalar {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let value = self.value + other.value;
        let gradient = match (self.gradient, other.gradient) {
            (Some(g1), Some(g2)) => Some(g1 + g2),
            (Some(g), None) | (None, Some(g)) => Some(g),
            (None, None) => None,
        };
        
        Self {
            value,
            gradient,
            diff_marker: self.diff_marker.or(other.diff_marker),
        }
    }
}

impl std::ops::Mul for DiffScalar {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let value = self.value * other.value;
        let gradient = match (self.gradient, other.gradient) {
            (Some(g1), Some(g2)) => Some(g1 * other.value + self.value * g2),
            (Some(g), None) => Some(g * other.value),
            (None, Some(g)) => Some(self.value * g),
            (None, None) => None,
        };
        
        Self {
            value,
            gradient,
            diff_marker: self.diff_marker.or(other.diff_marker),
        }
    }
}

/// Trigonometric functions with AD support
pub mod trig {
    use super::DiffScalar;

    /// Sine function with automatic differentiation
    pub fn sin(x: DiffScalar) -> DiffScalar {
        let value = x.value().sin();
        let gradient = x.gradient().map(|g| g * x.value().cos());
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Cosine function with automatic differentiation
    pub fn cos(x: DiffScalar) -> DiffScalar {
        let value = x.value().cos();
        let gradient = x.gradient().map(|g| -g * x.value().sin());
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Tangent function with automatic differentiation
    pub fn tan(x: DiffScalar) -> DiffScalar {
        let value = x.value().tan();
        let cos_val = x.value().cos();
        let gradient = x.gradient().map(|g| g / (cos_val * cos_val));
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Arcsine function with automatic differentiation
    pub fn asin(x: DiffScalar) -> DiffScalar {
        let value = x.value().asin();
        let gradient = x.gradient().map(|g| g / (1.0 - x.value() * x.value()).sqrt());
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Arccosine function with automatic differentiation
    pub fn acos(x: DiffScalar) -> DiffScalar {
        let value = x.value().acos();
        let gradient = x.gradient().map(|g| -g / (1.0 - x.value() * x.value()).sqrt());
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Arctangent function with automatic differentiation
    pub fn atan(x: DiffScalar) -> DiffScalar {
        let value = x.value().atan();
        let gradient = x.gradient().map(|g| g / (1.0 + x.value() * x.value()));
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Two-argument arctangent function
    pub fn atan2(y: DiffScalar, x: DiffScalar) -> DiffScalar {
        let value = y.value().atan2(x.value());
        let denom = x.value() * x.value() + y.value() * y.value();
        
        let gradient = match (y.gradient(), x.gradient()) {
            (Some(dy), Some(dx)) => Some((x.value() * dy - y.value() * dx) / denom),
            (Some(dy), None) => Some(x.value() * dy / denom),
            (None, Some(dx)) => Some(-y.value() * dx / denom),
            (None, None) => None,
        };
        
        DiffScalar {
            value,
            gradient,
            diff_marker: y.diff_marker.or(x.diff_marker),
        }
    }
}

/// Exponential and logarithmic functions with AD support
pub mod exp_log {
    use super::DiffScalar;

    /// Exponential function with automatic differentiation
    pub fn exp(x: DiffScalar) -> DiffScalar {
        let value = x.value().exp();
        let gradient = x.gradient().map(|g| g * value);
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Natural logarithm with automatic differentiation
    pub fn ln(x: DiffScalar) -> DiffScalar {
        let value = x.value().ln();
        let gradient = x.gradient().map(|g| g / x.value());
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Base-10 logarithm with automatic differentiation
    pub fn log10(x: DiffScalar) -> DiffScalar {
        let value = x.value().log10();
        let gradient = x.gradient().map(|g| g / (x.value() * super::constants::LN_10));
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Base-2 logarithm with automatic differentiation
    pub fn log2(x: DiffScalar) -> DiffScalar {
        let value = x.value().log2();
        let gradient = x.gradient().map(|g| g / (x.value() * super::constants::LN_2));
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Power function with automatic differentiation
    pub fn pow(base: DiffScalar, exponent: DiffScalar) -> DiffScalar {
        let value = base.value().powf(exponent.value());
        
        let gradient = match (base.gradient(), exponent.gradient()) {
            (Some(db), Some(de)) => {
                let base_term = de * value * base.value().ln();
                let exp_term = db * exponent.value() * base.value().powf(exponent.value() - 1.0);
                Some(base_term + exp_term)
            },
            (Some(db), None) => {
                Some(db * exponent.value() * base.value().powf(exponent.value() - 1.0))
            },
            (None, Some(de)) => {
                Some(de * value * base.value().ln())
            },
            (None, None) => None,
        };
        
        DiffScalar {
            value,
            gradient,
            diff_marker: base.diff_marker.or(exponent.diff_marker),
        }
    }

    /// Square root with automatic differentiation
    pub fn sqrt(x: DiffScalar) -> DiffScalar {
        let value = x.value().sqrt();
        let gradient = x.gradient().map(|g| g / (2.0 * value));
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }
}

/// Hyperbolic functions with AD support
pub mod hyperbolic {
    use super::DiffScalar;

    /// Hyperbolic sine with automatic differentiation
    pub fn sinh(x: DiffScalar) -> DiffScalar {
        let value = x.value().sinh();
        let gradient = x.gradient().map(|g| g * x.value().cosh());
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Hyperbolic cosine with automatic differentiation
    pub fn cosh(x: DiffScalar) -> DiffScalar {
        let value = x.value().cosh();
        let gradient = x.gradient().map(|g| g * x.value().sinh());
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }

    /// Hyperbolic tangent with automatic differentiation
    pub fn tanh(x: DiffScalar) -> DiffScalar {
        let value = x.value().tanh();
        let gradient = x.gradient().map(|g| g * (1.0 - value * value));
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }
}

/// Statistical functions
pub mod stats {
    use super::DiffScalar;
    use crate::stdlib::collections::LinearVec;

    /// Calculate mean of a vector
    pub fn mean(values: &LinearVec<f64>) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.clone().fold(0.0, |acc, x| acc + x) / values.len() as f64
    }

    /// Calculate variance of a vector
    pub fn variance(values: &LinearVec<f64>) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean_val = mean(values);
        let sum_sq_diff = values.clone().fold(0.0, |acc, x| {
            let diff = x - mean_val;
            acc + diff * diff
        });
        
        sum_sq_diff / (values.len() - 1) as f64
    }

    /// Calculate standard deviation
    pub fn std_dev(values: &LinearVec<f64>) -> f64 {
        variance(values).sqrt()
    }

    /// Normal distribution probability density function
    pub fn normal_pdf(x: DiffScalar, mean: f64, std_dev: f64) -> DiffScalar {
        let coefficient = 1.0 / (std_dev * (2.0 * super::constants::PI).sqrt());
        let exponent = -0.5 * ((x.value() - mean) / std_dev).powi(2);
        
        let value = coefficient * exponent.exp();
        let gradient = x.gradient().map(|g| {
            -g * value * (x.value() - mean) / (std_dev * std_dev)
        });
        
        DiffScalar {
            value,
            gradient,
            diff_marker: x.diff_marker,
        }
    }
}

/// Tensor mathematical operations with AD support
/// Note: These operations would require access to tensor internals
/// and are placeholder implementations for the standard library design
pub mod tensor_math {
    /// Placeholder for tensor mathematical operations
    /// In a full implementation, these would integrate with the tensor runtime
    pub struct TensorMath;

    impl TensorMath {
        /// Placeholder for element-wise sine operation
        pub fn sin_info() -> &'static str {
            "Element-wise sine operation on tensors with automatic differentiation support"
        }

        /// Placeholder for element-wise exponential operation
        pub fn exp_info() -> &'static str {
            "Element-wise exponential operation on tensors with automatic differentiation support"
        }

        /// Placeholder for element-wise logarithm operation
        pub fn log_info() -> &'static str {
            "Element-wise logarithm operation on tensors with automatic differentiation support"
        }

        /// Placeholder for matrix multiplication
        pub fn matmul_info() -> &'static str {
            "Matrix multiplication with gradient support for automatic differentiation"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::trig::*;
    use super::exp_log::*;

    #[test]
    fn test_diff_scalar_arithmetic() {
        let x = DiffScalar::with_gradient(2.0, 1.0);
        let y = DiffScalar::with_gradient(3.0, 1.0);
        
        let sum = x + y;
        assert_eq!(sum.value(), 5.0);
        assert_eq!(sum.gradient(), Some(2.0));
        
        let x = DiffScalar::with_gradient(2.0, 1.0);
        let y = DiffScalar::with_gradient(3.0, 1.0);
        let product = x * y;
        assert_eq!(product.value(), 6.0);
        assert_eq!(product.gradient(), Some(5.0)); // 1*3 + 2*1
    }

    #[test]
    fn test_trig_functions() {
        let x = DiffScalar::with_gradient(0.0, 1.0);
        
        let sin_result = sin(x);
        assert_eq!(sin_result.value(), 0.0);
        assert_eq!(sin_result.gradient(), Some(1.0)); // cos(0) = 1
        
        let x = DiffScalar::with_gradient(0.0, 1.0);
        let cos_result = cos(x);
        assert_eq!(cos_result.value(), 1.0);
        assert_eq!(cos_result.gradient(), Some(0.0)); // -sin(0) = 0
    }

    #[test]
    fn test_exp_log_functions() {
        let x = DiffScalar::with_gradient(0.0, 1.0);
        
        let exp_result = exp(x);
        assert_eq!(exp_result.value(), 1.0);
        assert_eq!(exp_result.gradient(), Some(1.0)); // exp(0) = 1
        
        let x = DiffScalar::with_gradient(1.0, 1.0);
        let ln_result = ln(x);
        assert_eq!(ln_result.value(), 0.0);
        assert_eq!(ln_result.gradient(), Some(1.0)); // 1/1 = 1
    }

    #[test]
    fn test_power_function() {
        let base = DiffScalar::with_gradient(2.0, 1.0);
        let exp = DiffScalar::new(3.0);
        
        let result = pow(base, exp);
        assert_eq!(result.value(), 8.0); // 2^3 = 8
        assert_eq!(result.gradient(), Some(12.0)); // 3 * 2^2 = 12
    }

    #[test]
    fn test_statistical_functions() {
        use super::stats::*;
        use crate::stdlib::collections::LinearVec;
        
        let values = LinearVec::new()
            .push(1.0)
            .push(2.0)
            .push(3.0)
            .push(4.0)
            .push(5.0);
        
        assert_eq!(mean(&values), 3.0);
        assert_eq!(variance(&values), 2.5);
        assert!((std_dev(&values) - 1.5811388300841898).abs() < 1e-10);
    }
}