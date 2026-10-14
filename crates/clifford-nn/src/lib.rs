//! Minimal Clifford neural building blocks in Cl(2,0).

use clifford_math::{geometric_product, rotate_vector2, Multivector2, sandwich, rotor_from_angle};

pub use clifford_math;

/// One grid cell: scalar + 2D velocity encoded as multivector (no bivector in raw data).
pub fn encode_scalar_vector(scalar: f32, vx: f32, vy: f32) -> Multivector2 {
    Multivector2::from_scalar_vector(scalar, vx, vy)
}

/// Apply the same plane rotation to every cell's vector part; scalar unchanged.
pub fn rotate_field(field: &[Multivector2], theta: f32) -> Vec<Multivector2> {
    let r = rotor_from_angle(theta);
    field
        .iter()
        .map(|mv| sandwich(r, *mv))
        .collect()
}

/// Synthetic 2D rotation flow: v(x,y) proportional to (-y, x) on a small grid.
pub fn rotation_flow_field(grid: usize, omega: f32) -> Vec<Multivector2> {
    let mut out = Vec::with_capacity(grid * grid);
    let center = (grid as f32 - 1.0) * 0.5;
    for iy in 0..grid {
        for ix in 0..grid {
            let x = ix as f32 - center;
            let y = iy as f32 - center;
            let vx = -omega * y;
            let vy = omega * x;
            let pressure = (x * x + y * y).sqrt() * 0.1;
            out.push(encode_scalar_vector(pressure, vx, vy));
        }
    }
    out
}

/// Advance rotation flow one Euler step (analytic for this field).
pub fn rotation_flow_step(field: &[Multivector2], dt: f32, omega: f32) -> Vec<Multivector2> {
    field
        .iter()
        .map(|mv| {
            let (vx, vy) = mv.vector_part();
            let (rx, ry) = rotate_vector2(vx, vy, omega * dt);
            Multivector2::from_scalar_vector(mv.s, rx, ry)
        })
        .collect()
}

/// Clifford linear map: for each output channel, weighted sum of GP(input_channel_j, weight_ij).
/// Weights are full multivectors per (out, in) pair.
#[derive(Clone, Debug)]
pub struct CliffordLinear {
    /// weights[out][in] = multivector kernel
    pub weights: Vec<Vec<Multivector2>>,
    pub in_channels: usize,
    pub out_channels: usize,
}

impl CliffordLinear {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        let mut weights = Vec::with_capacity(out_channels);
        for _o in 0..out_channels {
            let mut row = Vec::with_capacity(in_channels);
            for _i in 0..in_channels {
                let scale = 0.1 / (in_channels as f32).sqrt();
                row.push(Multivector2::from_coeffs([
                    scale,
                    scale * 0.5,
                    scale * 0.5,
                    scale * 0.25,
                ]));
            }
            weights.push(row);
        }
        Self {
            weights,
            in_channels,
            out_channels,
        }
    }

    /// `inputs[c]` = multivector per channel at one spatial index; same length for all cells in batch.
    pub fn forward_cell(&self, inputs: &[Multivector2]) -> Vec<Multivector2> {
        assert_eq!(inputs.len(), self.in_channels);
        let mut outputs = Vec::with_capacity(self.out_channels);
        for o in 0..self.out_channels {
            let mut acc = Multivector2::ZERO;
            for i in 0..self.in_channels {
                acc = acc + geometric_product(inputs[i], self.weights[o][i]);
            }
            outputs.push(acc);
        }
        outputs
    }

    pub fn forward_batch(&self, batch: &[Vec<Multivector2>]) -> Vec<Vec<Multivector2>> {
        batch
            .iter()
            .map(|cell_inputs| self.forward_cell(cell_inputs))
            .collect()
    }

    pub fn weight_coeffs(&self) -> Vec<f32> {
        let mut v = Vec::new();
        for row in &self.weights {
            for w in row {
                v.extend(w.as_coeffs());
            }
        }
        v
    }

    pub fn set_weight_coeffs(&mut self, coeffs: &[f32]) {
        let n = self.in_channels * self.out_channels * 4;
        assert_eq!(coeffs.len(), n);
        let mut k = 0;
        for o in 0..self.out_channels {
            for i in 0..self.in_channels {
                self.weights[o][i] =
                    Multivector2::from_coeffs([coeffs[k], coeffs[k + 1], coeffs[k + 2], coeffs[k + 3]]);
                k += 4;
            }
        }
    }

    pub fn num_scalar_params(&self) -> usize {
        self.in_channels * self.out_channels * 4
    }
}

/// Flat MLP-style linear on stacked coefficients (baseline).
#[derive(Clone, Debug)]
pub struct FlatLinear {
    pub w: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl FlatLinear {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let n = in_dim * out_dim;
        let w = (0..n)
            .map(|i| 0.1 / (in_dim as f32).sqrt() * ((i % 7) as f32 * 0.13 - 0.3))
            .collect();
        Self { w, in_dim, out_dim }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim);
        let mut y = vec![0.0; self.out_dim];
        for o in 0..self.out_dim {
            let mut sum = 0.0;
            for i in 0..self.in_dim {
                sum += self.w[o * self.in_dim + i] * x[i];
            }
            y[o] = sum;
        }
        y
    }
}

/// MSE on vector parts (e1, e2) across batch; ignores scalar for simplicity.
pub fn mse_vector(pred: &[Multivector2], target: &[Multivector2]) -> f32 {
    assert_eq!(pred.len(), target.len());
    let mut sum = 0.0;
    for (p, t) in pred.iter().zip(target.iter()) {
        let (px, py) = p.vector_part();
        let (tx, ty) = t.vector_part();
        sum += (px - tx).powi(2) + (py - ty).powi(2);
    }
    sum / pred.len() as f32
}

/// One SGD step via symmetric finite differences on CliffordLinear weights.
pub fn train_clifford_linear_fd(
    layer: &mut CliffordLinear,
    batch_in: &[Vec<Multivector2>],
    batch_tgt: &[Multivector2],
    lr: f32,
    eps: f32,
) -> f32 {
    let n = layer.num_scalar_params();
    let coeffs = layer.weight_coeffs();
    let mut grad = vec![0.0; n];

    for k in 0..n {
        let mut plus = coeffs.clone();
        let mut minus = coeffs.clone();
        plus[k] += eps;
        minus[k] -= eps;
        layer.set_weight_coeffs(&plus);
        let loss_plus = eval_batch_loss(layer, batch_in, batch_tgt);
        layer.set_weight_coeffs(&minus);
        let loss_minus = eval_batch_loss(layer, batch_in, batch_tgt);
        grad[k] = (loss_plus - loss_minus) / (2.0 * eps);
    }

    let mut new_coeffs = coeffs;
    for k in 0..n {
        new_coeffs[k] -= lr * grad[k];
    }
    layer.set_weight_coeffs(&new_coeffs);
    eval_batch_loss(layer, batch_in, batch_tgt)
}

fn eval_batch_loss(
    layer: &CliffordLinear,
    batch_in: &[Vec<Multivector2>],
    batch_tgt: &[Multivector2],
) -> f32 {
    let pred_cells: Vec<Multivector2> = batch_in
        .iter()
        .map(|inputs| {
            let outs = layer.forward_cell(inputs);
            outs[0]
        })
        .collect();
    mse_vector(&pred_cells, batch_tgt)
}

/// Build single-channel batch: each spatial cell is `[mv]`.
pub fn field_as_single_channel_batch(field: &[Multivector2]) -> Vec<Vec<Multivector2>> {
    field.iter().map(|mv| vec![*mv]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn encode_rotate_matches_sandwich_field() {
        let field = rotation_flow_field(4, 1.0);
        let theta = 0.5;
        let rotated = rotate_field(&field, theta);
        let r = rotor_from_angle(theta);
        for (a, b) in rotated.iter().zip(field.iter()) {
            let expected = sandwich(r, *b);
            for i in 0..4 {
                assert_relative_eq!(a.as_coeffs()[i], expected.as_coeffs()[i], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn training_reduces_loss_slightly() {
        let grid = 3;
        let field = rotation_flow_field(grid, 1.0);
        let target = rotation_flow_step(&field, 0.1, 1.0);
        let batch_in = field_as_single_channel_batch(&field);
        let mut layer = CliffordLinear::new(1, 1);
        let loss0 = eval_batch_loss(&layer, &batch_in, &target);
        for _ in 0..8 {
            train_clifford_linear_fd(&mut layer, &batch_in, &target, 0.05, 1e-3);
        }
        let loss1 = eval_batch_loss(&layer, &batch_in, &target);
        assert!(loss1 < loss0, "loss0={} loss1={}", loss0, loss1);
    }
}
