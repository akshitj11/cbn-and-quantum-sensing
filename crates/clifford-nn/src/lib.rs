//! Minimal Clifford neural building blocks in Cl(2,0).

use clifford_math::{
    geometric_product, gp_vjp, rotate_vector2, rotor_from_angle, sandwich, Multivector2,
};

pub use clifford_math;

pub mod eval;
pub mod mag_models;
pub mod magnetometer;
pub mod models;
pub mod nbody;

/// One grid cell: scalar + 2D velocity encoded as multivector (no bivector in raw data).
pub fn encode_scalar_vector(scalar: f32, vx: f32, vy: f32) -> Multivector2 {
    Multivector2::from_scalar_vector(scalar, vx, vy)
}

/// Apply the same plane rotation to every cell's vector part; scalar unchanged.
pub fn rotate_field(field: &[Multivector2], theta: f32) -> Vec<Multivector2> {
    let r = rotor_from_angle(theta);
    field.iter().map(|mv| sandwich(r, *mv)).collect()
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
    /// `weights[out][in]` = multivector kernel
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
        for row in &self.weights {
            let mut acc = Multivector2::ZERO;
            for (input, weight) in inputs.iter().zip(row.iter()) {
                acc = acc + geometric_product(*input, *weight);
            }
            outputs.push(acc);
        }
        outputs
    }

    /// Analytic backward for `forward_cell`.
    ///
    /// Given the inputs used in the forward pass and the upstream gradient on
    /// each output channel, returns the gradient w.r.t. every weight kernel and
    /// w.r.t. each input channel (the latter enables stacking layers).
    pub fn backward_cell(
        &self,
        inputs: &[Multivector2],
        grad_out: &[Multivector2],
    ) -> (Vec<Vec<Multivector2>>, Vec<Multivector2>) {
        assert_eq!(inputs.len(), self.in_channels);
        assert_eq!(grad_out.len(), self.out_channels);

        let mut grad_w = vec![vec![Multivector2::ZERO; self.in_channels]; self.out_channels];
        let mut grad_in = vec![Multivector2::ZERO; self.in_channels];

        for (o, grad_row) in grad_w.iter_mut().enumerate() {
            for (i, gw) in grad_row.iter_mut().enumerate() {
                let (g_in, g_w) = gp_vjp(inputs[i], self.weights[o][i], grad_out[o]);
                *gw = g_w;
                grad_in[i] = grad_in[i] + g_in;
            }
        }
        (grad_w, grad_in)
    }

    /// In-place SGD update: `weights -= lr * grad_w`.
    pub fn apply_grad(&mut self, grad_w: &[Vec<Multivector2>], lr: f32) {
        for (row, grad_row) in self.weights.iter_mut().zip(grad_w.iter()) {
            for (w, g) in row.iter_mut().zip(grad_row.iter()) {
                *w = *w - *g * lr;
            }
        }
    }

    /// Accumulate `src` grads into `dst` (both shaped `[out][in]`).
    pub fn accumulate_grad(dst: &mut [Vec<Multivector2>], src: &[Vec<Multivector2>]) {
        for (drow, srow) in dst.iter_mut().zip(src.iter()) {
            for (d, s) in drow.iter_mut().zip(srow.iter()) {
                *d = *d + *s;
            }
        }
    }

    /// A zero gradient buffer shaped like the weights.
    pub fn zero_grad(&self) -> Vec<Vec<Multivector2>> {
        vec![vec![Multivector2::ZERO; self.in_channels]; self.out_channels]
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
                self.weights[o][i] = Multivector2::from_coeffs([
                    coeffs[k],
                    coeffs[k + 1],
                    coeffs[k + 2],
                    coeffs[k + 3],
                ]);
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
        for (o, yo) in y.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (i, xi) in x.iter().enumerate() {
                sum += self.w[o * self.in_dim + i] * xi;
            }
            *yo = sum;
        }
        y
    }

    /// Analytic backward for `forward`.
    ///
    /// Returns `(grad_w, grad_x)` where `grad_w` is flattened like `self.w`.
    pub fn backward(&self, x: &[f32], grad_out: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(x.len(), self.in_dim);
        assert_eq!(grad_out.len(), self.out_dim);
        let mut grad_w = vec![0.0; self.w.len()];
        let mut grad_x = vec![0.0; self.in_dim];
        for (o, &go) in grad_out.iter().enumerate() {
            for (i, &xi) in x.iter().enumerate() {
                grad_w[o * self.in_dim + i] = go * xi;
                grad_x[i] += self.w[o * self.in_dim + i] * go;
            }
        }
        (grad_w, grad_x)
    }

    pub fn apply_grad(&mut self, grad_w: &[f32], lr: f32) {
        assert_eq!(grad_w.len(), self.w.len());
        for (w, g) in self.w.iter_mut().zip(grad_w.iter()) {
            *w -= lr * g;
        }
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

/// Gradient of `mse_vector` w.r.t. a single prediction multivector.
///
/// Only the vector grades (e1, e2) contribute; scalar and bivector get zero.
/// `n` is the batch size used to normalize the mean.
pub fn mse_vector_grad(pred: Multivector2, target: Multivector2, n: usize) -> Multivector2 {
    let (px, py) = pred.vector_part();
    let (tx, ty) = target.vector_part();
    let scale = 2.0 / n as f32;
    Multivector2::from_coeffs([0.0, scale * (px - tx), scale * (py - ty), 0.0])
}

/// One analytic-gradient SGD step for a single-output `CliffordLinear` on the
/// vector-MSE objective (same task as `train_clifford_linear_fd`).
pub fn train_clifford_linear_analytic(
    layer: &mut CliffordLinear,
    batch_in: &[Vec<Multivector2>],
    batch_tgt: &[Multivector2],
    lr: f32,
) -> f32 {
    let n = batch_in.len();
    let mut grad_acc = layer.zero_grad();
    for (inputs, tgt) in batch_in.iter().zip(batch_tgt.iter()) {
        let outs = layer.forward_cell(inputs);
        let mut grad_out = vec![Multivector2::ZERO; layer.out_channels];
        grad_out[0] = mse_vector_grad(outs[0], *tgt, n);
        let (grad_w, _grad_in) = layer.backward_cell(inputs, &grad_out);
        CliffordLinear::accumulate_grad(&mut grad_acc, &grad_w);
    }
    layer.apply_grad(&grad_acc, lr);
    eval_batch_loss(layer, batch_in, batch_tgt)
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
    fn analytic_grad_matches_finite_difference() {
        let grid = 3;
        let field = rotation_flow_field(grid, 1.0);
        let target = rotation_flow_step(&field, 0.1, 1.0);
        let batch_in = field_as_single_channel_batch(&field);
        let layer = CliffordLinear::new(1, 1);

        // Analytic gradient accumulated over the batch.
        let n = batch_in.len();
        let mut grad_acc = layer.zero_grad();
        for (inputs, tgt) in batch_in.iter().zip(target.iter()) {
            let outs = layer.forward_cell(inputs);
            let mut grad_out = vec![Multivector2::ZERO; layer.out_channels];
            grad_out[0] = mse_vector_grad(outs[0], *tgt, n);
            let (grad_w, _) = layer.backward_cell(inputs, &grad_out);
            CliffordLinear::accumulate_grad(&mut grad_acc, &grad_w);
        }
        let analytic: Vec<f32> = grad_acc[0][0].as_coeffs().to_vec();

        // Finite-difference gradient on the same loss.
        let coeffs = layer.weight_coeffs();
        let eps = 1e-3;
        let mut l = layer.clone();
        for k in 0..coeffs.len() {
            let mut plus = coeffs.clone();
            let mut minus = coeffs.clone();
            plus[k] += eps;
            minus[k] -= eps;
            l.set_weight_coeffs(&plus);
            let lp = eval_batch_loss(&l, &batch_in, &target);
            l.set_weight_coeffs(&minus);
            let lm = eval_batch_loss(&l, &batch_in, &target);
            let fd = (lp - lm) / (2.0 * eps);
            assert_relative_eq!(analytic[k], fd, epsilon = 1e-3);
        }
    }

    #[test]
    fn analytic_training_reduces_loss() {
        let grid = 3;
        let field = rotation_flow_field(grid, 1.0);
        let target = rotation_flow_step(&field, 0.1, 1.0);
        let batch_in = field_as_single_channel_batch(&field);
        let mut layer = CliffordLinear::new(1, 1);
        let loss0 = eval_batch_loss(&layer, &batch_in, &target);
        let mut loss1 = loss0;
        for _ in 0..8 {
            loss1 = train_clifford_linear_analytic(&mut layer, &batch_in, &target, 0.05);
        }
        assert!(loss1 < loss0, "loss0={} loss1={}", loss0, loss1);
    }

    #[test]
    fn flat_backward_matches_finite_difference() {
        let layer = FlatLinear::new(4, 3);
        let x = vec![0.5, -1.0, 0.25, 2.0];
        let grad_out = vec![0.7, -0.2, 1.1];
        let (grad_w, grad_x) = layer.backward(&x, &grad_out);

        let loss = |l: &FlatLinear, x: &[f32]| -> f32 {
            let y = l.forward(x);
            y.iter().zip(grad_out.iter()).map(|(a, b)| a * b).sum()
        };
        let eps = 1e-4;
        // grad wrt weights
        for (k, &gw) in grad_w.iter().enumerate() {
            let mut lp = layer.clone();
            let mut lm = layer.clone();
            lp.w[k] += eps;
            lm.w[k] -= eps;
            let fd = (loss(&lp, &x) - loss(&lm, &x)) / (2.0 * eps);
            assert_relative_eq!(gw, fd, epsilon = 1e-3);
        }
        // grad wrt inputs
        for k in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += eps;
            xm[k] -= eps;
            let fd = (loss(&layer, &xp) - loss(&layer, &xm)) / (2.0 * eps);
            assert_relative_eq!(grad_x[k], fd, epsilon = 1e-3);
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
