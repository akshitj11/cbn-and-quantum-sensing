//! Cl(2,0) Euclidean geometric algebra: multivectors `a + b*e1 + c*e2 + d*e12`.

use std::ops::{Add, Mul, Neg, Sub};

/// Basis order: scalar, e1, e2, e12.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Multivector2 {
    pub s: f32,
    pub e1: f32,
    pub e2: f32,
    pub e12: f32,
}

impl Multivector2 {
    pub const ZERO: Self = Self {
        s: 0.0,
        e1: 0.0,
        e2: 0.0,
        e12: 0.0,
    };

    pub const ONE: Self = Self {
        s: 1.0,
        e1: 0.0,
        e2: 0.0,
        e12: 0.0,
    };

    pub fn scalar(s: f32) -> Self {
        Self {
            s,
            e1: 0.0,
            e2: 0.0,
            e12: 0.0,
        }
    }

    pub fn vector(x: f32, y: f32) -> Self {
        Self {
            s: 0.0,
            e1: x,
            e2: y,
            e12: 0.0,
        }
    }

    pub fn bivector(xy: f32) -> Self {
        Self {
            s: 0.0,
            e1: 0.0,
            e2: 0.0,
            e12: xy,
        }
    }

    pub fn from_scalar_vector(scalar: f32, vx: f32, vy: f32) -> Self {
        Self {
            s: scalar,
            e1: vx,
            e2: vy,
            e12: 0.0,
        }
    }

    /// Reverse: sign flip on bivector (and trivectors if any).
    pub fn reverse(self) -> Self {
        Self {
            s: self.s,
            e1: self.e1,
            e2: self.e2,
            e12: -self.e12,
        }
    }

    pub fn norm_sq(self) -> f32 {
        self.s * self.s + self.e1 * self.e1 + self.e2 * self.e2 + self.e12 * self.e12
    }

    pub fn norm(self) -> f32 {
        self.norm_sq().sqrt()
    }

    pub fn vector_part(self) -> (f32, f32) {
        (self.e1, self.e2)
    }

    pub fn as_coeffs(self) -> [f32; 4] {
        [self.s, self.e1, self.e2, self.e12]
    }

    pub fn from_coeffs(c: [f32; 4]) -> Self {
        Self {
            s: c[0],
            e1: c[1],
            e2: c[2],
            e12: c[3],
        }
    }
}

impl Add for Multivector2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            s: self.s + rhs.s,
            e1: self.e1 + rhs.e1,
            e2: self.e2 + rhs.e2,
            e12: self.e12 + rhs.e12,
        }
    }
}

impl Sub for Multivector2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Neg for Multivector2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            s: -self.s,
            e1: -self.e1,
            e2: -self.e2,
            e12: -self.e12,
        }
    }
}

impl Mul<f32> for Multivector2 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            s: self.s * rhs,
            e1: self.e1 * rhs,
            e2: self.e2 * rhs,
            e12: self.e12 * rhs,
        }
    }
}

/// Geometric product in Cl(2,0): e1² = e2² = 1, e1 e2 = -e2 e1 = e12.
pub fn geometric_product(a: Multivector2, b: Multivector2) -> Multivector2 {
    let (a0, a1, a2, a12) = (a.s, a.e1, a.e2, a.e12);
    let (b0, b1, b2, b12) = (b.s, b.e1, b.e2, b.e12);

    Multivector2 {
        s: a0 * b0 + a1 * b1 + a2 * b2 - a12 * b12,
        e1: a0 * b1 + a1 * b0 + a12 * b2 - a2 * b12,
        e2: a0 * b2 + a2 * b0 + a1 * b12 - a12 * b1,
        e12: a0 * b12 + a12 * b0 + a1 * b2 - a2 * b1,
    }
}

/// Unit rotor from angle θ (rotation in the e1-e2 plane): cos(θ/2) + sin(θ/2) e12.
pub fn rotor_from_angle(theta: f32) -> Multivector2 {
    let half = theta * 0.5;
    Multivector2 {
        s: half.cos(),
        e1: 0.0,
        e2: 0.0,
        e12: -half.sin(),
    }
}

/// Sandwich product R x R̃ for rotors (even subalgebra).
pub fn sandwich(rotor: Multivector2, x: Multivector2) -> Multivector2 {
    geometric_product(geometric_product(rotor, x), rotor.reverse())
}

/// Rotate a 2D vector (x, y) by angle θ using a rotor.
pub fn rotate_vector2(vx: f32, vy: f32, theta: f32) -> (f32, f32) {
    let r = rotor_from_angle(theta);
    let v = Multivector2::vector(vx, vy);
    let out = sandwich(r, v);
    (out.e1, out.e2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn lesson_basis_anticommute() {
        let e1 = Multivector2::vector(1.0, 0.0);
        let e2 = Multivector2::vector(0.0, 1.0);
        let gp = geometric_product(e1, e2);
        assert_relative_eq!(gp.e12, 1.0);
        assert_relative_eq!(geometric_product(e2, e1).e12, -1.0);
    }

    #[test]
    fn lesson_rotor_preserves_vector_norm() {
        let r = rotor_from_angle(0.7);
        let v = Multivector2::vector(3.0, 4.0);
        let rotated = sandwich(r, v);
        assert_relative_eq!(rotated.norm(), v.norm(), epsilon = 1e-5);
    }

    #[test]
    fn lesson_rotor_matches_matrix_rotation() {
        let theta = 1.2;
        let (rx, ry) = rotate_vector2(1.0, 0.0, theta);
        assert_relative_eq!(rx, theta.cos(), epsilon = 1e-5);
        assert_relative_eq!(ry, theta.sin(), epsilon = 1e-5);
    }

    #[test]
    fn geometric_product_associative() {
        let a = Multivector2::from_scalar_vector(1.0, 2.0, -1.0);
        let b = Multivector2::from_coeffs([0.5, 1.0, 0.0, 0.3]);
        let c = Multivector2::vector(1.0, 1.0);
        let left = geometric_product(geometric_product(a, b), c);
        let right = geometric_product(a, geometric_product(b, c));
        for i in 0..4 {
            assert_relative_eq!(left.as_coeffs()[i], right.as_coeffs()[i], epsilon = 1e-4);
        }
    }
}
