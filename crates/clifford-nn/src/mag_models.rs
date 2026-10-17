//! Field-estimation models for the planar magnetometer.
//!
//! Three estimators share one task: recover the world-frame field `B` from a
//! Ramsey probability record and the measurement axes used.
//!
//!   * [`CliffordMag`]   - message-passing network carrying `Multivector2`
//!     channels through geometric products. Each measurement is one "node":
//!     a vector-grade channel `v_k = (P_k - 0.5) * u_k`, the measured signal
//!     scaled onto its axis direction. This is the geometric observation the
//!     field estimate is (approximately) a linear combination of, and it
//!     rotates with the apparatus, so the estimate tends to rotate with it
//!     (measured, not assumed).
//!   * [`FlatMag`]       - parameter-matched baseline on raw real features.
//!   * [`ClassicalEstimator`] - non-learned least-squares inversion of the
//!     Ramsey model; the clean-data accuracy reference. `with_assumptions`
//!     turns it into an oracle that undoes a believed hard/soft-iron defect.
//!   * [`CliffordMagRaw`] / [`FlatMlpMag`] - Experiment 2 variants that are
//!     *not* handed the vector observable and must form `(P_k-0.5)*u_k`
//!     themselves (geometric product vs a tanh MLP).
//!
//! Both learned models mirror the n-body topology (shared per-item message
//! layer, permutation-invariant sum, shared update layer) so the comparison is
//! fair, and both report `num_params`.

use crate::magnetometer::{MagConfig, MagSample};
use crate::{CliffordLinear, FlatLinear};
use clifford_math::{geometric_product, Multivector2};

/// Squared error between an estimated and true 2D field.
pub fn field_sq_err(pred: [f32; 2], tgt: [f32; 2]) -> f32 {
    (pred[0] - tgt[0]).powi(2) + (pred[1] - tgt[1]).powi(2)
}

/// Mean squared error over a slice of (pred, tgt) field pairs.
pub fn field_mse(pairs: &[([f32; 2], [f32; 2])]) -> f32 {
    if pairs.is_empty() {
        return 0.0;
    }
    let sum: f32 = pairs.iter().map(|(p, t)| field_sq_err(*p, *t)).sum();
    sum / pairs.len() as f32
}

// A multivector gradient carrying only vector-grade error.
fn vec_grad(ex: f32, ey: f32) -> Multivector2 {
    Multivector2::from_coeffs([0.0, ex, ey, 0.0])
}

// ----------------------------------------------------------------------------
// Clifford model
// ----------------------------------------------------------------------------

/// Message-passing Clifford magnetometer network.
///
/// Message input (1 channel): `scalar(P_k - 0.5) + vector(u_k)` per measurement.
/// Update input (`hidden + 1` channels): aggregated messages ++ a constant
/// `ONE` channel (lets the model learn an additive bias, e.g. hard-iron).
/// Output (1 channel): the field estimate, read from the vector grade.
#[derive(Clone, Debug)]
pub struct CliffordMag {
    pub msg: CliffordLinear,
    pub upd: CliffordLinear,
    pub hidden: usize,
}

impl CliffordMag {
    pub fn new(hidden: usize) -> Self {
        Self {
            msg: CliffordLinear::new(1, hidden),
            upd: CliffordLinear::new(hidden + 1, 1),
            hidden,
        }
    }

    pub fn num_params(&self) -> usize {
        self.msg.num_scalar_params() + self.upd.num_scalar_params()
    }

    fn measurement_channel(p: f32, u: [f32; 2]) -> Vec<Multivector2> {
        // Vector-grade observation: measured signal scaled onto its axis.
        let s = p - 0.5;
        vec![Multivector2::vector(s * u[0], s * u[1])]
    }

    // Aggregated messages ++ constant channel.
    fn update_input(&self, sample: &MagSample) -> Vec<Multivector2> {
        let mut agg = vec![Multivector2::ZERO; self.hidden];
        for (p, u) in sample.record.iter().zip(sample.axes.iter()) {
            let chan = Self::measurement_channel(*p, *u);
            let out = self.msg.forward_cell(&chan);
            for c in 0..self.hidden {
                agg[c] = agg[c] + out[c];
            }
        }
        agg.push(Multivector2::ONE);
        agg
    }

    pub fn estimate(&self, sample: &MagSample) -> [f32; 2] {
        let upd_in = self.update_input(sample);
        let out = self.upd.forward_cell(&upd_in);
        let (bx, by) = out[0].vector_part();
        [bx, by]
    }

    /// One full-batch analytic SGD step. Returns the pre-update batch MSE.
    pub fn train_step(&mut self, batch: &[MagSample], lr: f32) -> f32 {
        let mut grad_msg = self.msg.zero_grad();
        let mut grad_upd = self.upd.zero_grad();
        let mut total_loss = 0.0;
        let n = batch.len().max(1);
        // Loss is mean over samples of the 2-component squared field error.
        let scale = 2.0 / (n as f32 * 2.0);

        for sample in batch {
            let upd_in = self.update_input(sample);
            let out = self.upd.forward_cell(&upd_in);
            let (bx, by) = out[0].vector_part();
            let ex = bx - sample.field[0];
            let ey = by - sample.field[1];
            total_loss += ex * ex + ey * ey;

            let grad_out = vec![vec_grad(scale * ex, scale * ey)];
            let (gw_upd, g_upd_in) = self.upd.backward_cell(&upd_in, &grad_out);
            CliffordLinear::accumulate_grad(&mut grad_upd, &gw_upd);

            // First `hidden` input grads flow back into each measurement message.
            let grad_agg = &g_upd_in[0..self.hidden];
            for u in sample.axes.iter().zip(sample.record.iter()) {
                let (axis, p) = (u.0, u.1);
                let chan = Self::measurement_channel(*p, *axis);
                let (gw_msg, _g_in) = self.msg.backward_cell(&chan, grad_agg);
                CliffordLinear::accumulate_grad(&mut grad_msg, &gw_msg);
            }
        }

        self.msg.apply_grad(&grad_msg, lr);
        self.upd.apply_grad(&grad_upd, lr);
        total_loss / (n as f32 * 2.0)
    }
}

// ----------------------------------------------------------------------------
// Flat baseline (matched topology on real features)
// ----------------------------------------------------------------------------

/// Per-measurement features: the vector observation `(P_k - 0.5) * u_k`.
const MSG_DIM: usize = 2;
/// Output features: `[Bx, By]`.
const OUT_DIM: usize = 2;

/// Flat message-passing baseline mirroring [`CliffordMag`]'s structure.
#[derive(Clone, Debug)]
pub struct FlatMag {
    pub msg: FlatLinear,
    pub upd: FlatLinear,
    pub hidden: usize,
}

impl FlatMag {
    pub fn new(hidden: usize) -> Self {
        Self {
            msg: FlatLinear::new(MSG_DIM, hidden),
            // +1 for a constant bias input (matches Clifford's ONE channel).
            upd: FlatLinear::new(hidden + 1, OUT_DIM),
            hidden,
        }
    }

    pub fn num_params(&self) -> usize {
        self.msg.w.len() + self.upd.w.len()
    }

    /// Pick the hidden width whose param count is closest to `target_params`.
    pub fn hidden_for_params(target_params: usize) -> usize {
        // params(h) = MSG_DIM*h + (h+1)*OUT_DIM = (MSG_DIM+OUT_DIM)*h + OUT_DIM
        let slope = (MSG_DIM + OUT_DIM) as f32;
        let intercept = OUT_DIM as f32;
        let h = ((target_params as f32 - intercept) / slope).round();
        (h as isize).max(1) as usize
    }

    fn measurement_features(p: f32, u: [f32; 2]) -> Vec<f32> {
        let s = p - 0.5;
        vec![s * u[0], s * u[1]]
    }

    fn update_input(&self, sample: &MagSample) -> Vec<f32> {
        let mut agg = vec![0.0; self.hidden];
        for (p, u) in sample.record.iter().zip(sample.axes.iter()) {
            let out = self.msg.forward(&Self::measurement_features(*p, *u));
            for c in 0..self.hidden {
                agg[c] += out[c];
            }
        }
        agg.push(1.0);
        agg
    }

    pub fn estimate(&self, sample: &MagSample) -> [f32; 2] {
        let out = self.upd.forward(&self.update_input(sample));
        [out[0], out[1]]
    }

    pub fn train_step(&mut self, batch: &[MagSample], lr: f32) -> f32 {
        let mut grad_msg = vec![0.0; self.msg.w.len()];
        let mut grad_upd = vec![0.0; self.upd.w.len()];
        let mut total_loss = 0.0;
        let n = batch.len().max(1);
        let scale = 2.0 / (n as f32 * 2.0);

        for sample in batch {
            let upd_in = self.update_input(sample);
            let out = self.upd.forward(&upd_in);
            let err = [out[0] - sample.field[0], out[1] - sample.field[1]];
            total_loss += err[0] * err[0] + err[1] * err[1];
            let grad_out: Vec<f32> = err.iter().map(|e| scale * e).collect();

            let (gw_upd, g_upd_in) = self.upd.backward(&upd_in, &grad_out);
            for (d, s) in grad_upd.iter_mut().zip(gw_upd.iter()) {
                *d += s;
            }
            let grad_agg = &g_upd_in[0..self.hidden];
            for (p, u) in sample.record.iter().zip(sample.axes.iter()) {
                let feat = Self::measurement_features(*p, *u);
                let (gw_msg, _g_in) = self.msg.backward(&feat, grad_agg);
                for (d, s) in grad_msg.iter_mut().zip(gw_msg.iter()) {
                    *d += s;
                }
            }
        }

        self.msg.apply_grad(&grad_msg, lr);
        self.upd.apply_grad(&grad_upd, lr);
        total_loss / (n as f32 * 2.0)
    }
}

// ----------------------------------------------------------------------------
// Classical estimator (non-learned)
// ----------------------------------------------------------------------------

/// Least-squares inversion of the Ramsey model.
///
/// Each measurement gives `P_k = 0.5(1 + V sin(gamma T (B_eff.u_k)))`, so the
/// projection is recovered as `proj_k = asin(clamp((2 P_k - 1)/V)) / (gamma T)`.
/// Stacking the axes into `A` (rows `u_k`), the distorted field is the
/// normal-equation solution `B_eff = (A^T A)^{-1} A^T proj`, then the believed
/// sensor defect is undone: `B = (B_eff - assume_hard) / assume_soft`.
///
/// The `cfg` it inverts with and the `assume_*` corrections encode what the fit
/// *believes* about the sensor. An oracle sets them to the truth; a naive fit
/// leaves them at the no-defect / nominal values and is beatable when the real
/// sensor is distorted.
#[derive(Clone, Debug)]
pub struct ClassicalEstimator {
    pub cfg: MagConfig,
    pub assume_hard: [f32; 2],
    pub assume_soft: [f32; 2],
}

impl ClassicalEstimator {
    /// A fit that assumes no hard/soft-iron defect (naive by default).
    pub fn new(cfg: MagConfig) -> Self {
        Self {
            cfg,
            assume_hard: [0.0, 0.0],
            assume_soft: [1.0, 1.0],
        }
    }

    /// A fit that believes the given hard/soft-iron defect (an oracle when the
    /// values match the simulator's true defect).
    pub fn with_assumptions(cfg: MagConfig, assume_hard: [f32; 2], assume_soft: [f32; 2]) -> Self {
        Self {
            cfg,
            assume_hard,
            assume_soft,
        }
    }

    pub fn estimate(&self, sample: &MagSample) -> [f32; 2] {
        let v = self.cfg.contrast().max(1e-6);
        let gt = self.cfg.gamma * self.cfg.t_evolve;

        // 2x2 normal equations: (A^T A) B_eff = A^T proj.
        let (mut axx, mut axy, mut ayy) = (0.0f32, 0.0f32, 0.0f32);
        let (mut bx, mut by) = (0.0f32, 0.0f32);
        for (p, u) in sample.record.iter().zip(sample.axes.iter()) {
            let s = ((2.0 * p - 1.0) / v).clamp(-1.0, 1.0);
            let proj = s.asin() / gt;
            axx += u[0] * u[0];
            axy += u[0] * u[1];
            ayy += u[1] * u[1];
            bx += u[0] * proj;
            by += u[1] * proj;
        }
        let det = axx * ayy - axy * axy;
        if det.abs() < 1e-8 {
            return [0.0, 0.0];
        }
        let inv = 1.0 / det;
        let eff_x = inv * (ayy * bx - axy * by);
        let eff_y = inv * (-axy * bx + axx * by);

        // Undo the believed sensor defect: B = (B_eff - assume_hard) / assume_soft.
        let sx = if self.assume_soft[0].abs() < 1e-6 {
            1.0
        } else {
            self.assume_soft[0]
        };
        let sy = if self.assume_soft[1].abs() < 1e-6 {
            1.0
        } else {
            self.assume_soft[1]
        };
        [
            (eff_x - self.assume_hard[0]) / sx,
            (eff_y - self.assume_hard[1]) / sy,
        ]
    }
}

// ----------------------------------------------------------------------------
// Experiment 2 models: form the observable inside the model
// ----------------------------------------------------------------------------
//
// The models above are handed the vector observable `(P_k - 0.5) * u_k`. These
// two instead receive the raw, separated pieces - the scalar readout `P_k - 0.5`
// and the axis vector `u_k` - and must *build* the product themselves. A single
// `CliffordLinear` cannot (it maps each channel linearly and sums), so this
// isolates the value of the bilinear geometric product as an inductive bias
// versus a generic nonlinearity.

/// Clifford model that forms the observable with an explicit geometric product.
///
/// Per measurement it computes `g_k = geometric_product(scalar(P_k - 0.5),
/// vector(u_k))`, which is exactly `(P_k - 0.5) * u_k` - but produced *by the
/// algebra*, not handed in. The product carries no weights, so training is
/// identical to [`CliffordMag`] from the message layer onward; the point is
/// that the architecture knows to multiply the scalar readout onto its axis.
#[derive(Clone, Debug)]
pub struct CliffordMagRaw {
    pub msg: CliffordLinear,
    pub upd: CliffordLinear,
    pub hidden: usize,
}

impl CliffordMagRaw {
    pub fn new(hidden: usize) -> Self {
        Self {
            msg: CliffordLinear::new(1, hidden),
            upd: CliffordLinear::new(hidden + 1, 1),
            hidden,
        }
    }

    pub fn num_params(&self) -> usize {
        self.msg.num_scalar_params() + self.upd.num_scalar_params()
    }

    // The bilinear interaction: scalar readout times axis vector, via the GP.
    fn interaction(p: f32, u: [f32; 2]) -> Vec<Multivector2> {
        let a = Multivector2::scalar(p - 0.5);
        let b = Multivector2::vector(u[0], u[1]);
        vec![geometric_product(a, b)]
    }

    fn update_input(&self, sample: &MagSample) -> Vec<Multivector2> {
        let mut agg = vec![Multivector2::ZERO; self.hidden];
        for (p, u) in sample.record.iter().zip(sample.axes.iter()) {
            let chan = Self::interaction(*p, *u);
            let out = self.msg.forward_cell(&chan);
            for c in 0..self.hidden {
                agg[c] = agg[c] + out[c];
            }
        }
        agg.push(Multivector2::ONE);
        agg
    }

    pub fn estimate(&self, sample: &MagSample) -> [f32; 2] {
        let out = self.upd.forward_cell(&self.update_input(sample));
        let (bx, by) = out[0].vector_part();
        [bx, by]
    }

    pub fn train_step(&mut self, batch: &[MagSample], lr: f32) -> f32 {
        let mut grad_msg = self.msg.zero_grad();
        let mut grad_upd = self.upd.zero_grad();
        let mut total_loss = 0.0;
        let n = batch.len().max(1);
        let scale = 2.0 / (n as f32 * 2.0);

        for sample in batch {
            let upd_in = self.update_input(sample);
            let out = self.upd.forward_cell(&upd_in);
            let (bx, by) = out[0].vector_part();
            let ex = bx - sample.field[0];
            let ey = by - sample.field[1];
            total_loss += ex * ex + ey * ey;

            let grad_out = vec![vec_grad(scale * ex, scale * ey)];
            let (gw_upd, g_upd_in) = self.upd.backward_cell(&upd_in, &grad_out);
            CliffordLinear::accumulate_grad(&mut grad_upd, &gw_upd);

            let grad_agg = &g_upd_in[0..self.hidden];
            for (p, u) in sample.record.iter().zip(sample.axes.iter()) {
                let chan = Self::interaction(*p, *u);
                let (gw_msg, _g_in) = self.msg.backward_cell(&chan, grad_agg);
                CliffordLinear::accumulate_grad(&mut grad_msg, &gw_msg);
            }
        }

        self.msg.apply_grad(&grad_msg, lr);
        self.upd.apply_grad(&grad_upd, lr);
        total_loss / (n as f32 * 2.0)
    }
}

/// Flat MLP baseline that must *learn* the product from raw inputs.
///
/// Per measurement it sees `[P_k - 0.5, u_kx, u_ky]` and passes it through a
/// `tanh` hidden layer; the measurements are summed and a linear update reads
/// out the field. Unlike [`FlatMag`], nothing hands it `(P_k - 0.5) * u_k`, so
/// it has to approximate that bilinear map with the nonlinearity - the generic
/// alternative to the geometric product.
#[derive(Clone, Debug)]
pub struct FlatMlpMag {
    pub msg: FlatLinear,
    pub upd: FlatLinear,
    pub hidden: usize,
}

/// Raw per-measurement features for the MLP baseline.
const RAW_DIM: usize = 3;

impl FlatMlpMag {
    pub fn new(hidden: usize) -> Self {
        Self {
            msg: FlatLinear::new(RAW_DIM, hidden),
            upd: FlatLinear::new(hidden + 1, OUT_DIM),
            hidden,
        }
    }

    pub fn num_params(&self) -> usize {
        self.msg.w.len() + self.upd.w.len()
    }

    /// Match the Clifford raw model's parameter count.
    pub fn hidden_for_params(target_params: usize) -> usize {
        // params(h) = RAW_DIM*h + (h+1)*OUT_DIM = (RAW_DIM+OUT_DIM)*h + OUT_DIM
        let slope = (RAW_DIM + OUT_DIM) as f32;
        let intercept = OUT_DIM as f32;
        let h = ((target_params as f32 - intercept) / slope).round();
        (h as isize).max(1) as usize
    }

    fn raw_features(p: f32, u: [f32; 2]) -> Vec<f32> {
        vec![p - 0.5, u[0], u[1]]
    }

    // Returns (update_input, per-measurement pre-activations z_k) for backprop.
    fn forward_cache(&self, sample: &MagSample) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut agg = vec![0.0; self.hidden];
        let mut zs = Vec::with_capacity(sample.record.len());
        for (p, u) in sample.record.iter().zip(sample.axes.iter()) {
            let z = self.msg.forward(&Self::raw_features(*p, *u));
            for c in 0..self.hidden {
                agg[c] += z[c].tanh();
            }
            zs.push(z);
        }
        agg.push(1.0);
        (agg, zs)
    }

    pub fn estimate(&self, sample: &MagSample) -> [f32; 2] {
        let (upd_in, _) = self.forward_cache(sample);
        let out = self.upd.forward(&upd_in);
        [out[0], out[1]]
    }

    pub fn train_step(&mut self, batch: &[MagSample], lr: f32) -> f32 {
        let mut grad_msg = vec![0.0; self.msg.w.len()];
        let mut grad_upd = vec![0.0; self.upd.w.len()];
        let mut total_loss = 0.0;
        let n = batch.len().max(1);
        let scale = 2.0 / (n as f32 * 2.0);

        for sample in batch {
            let (upd_in, zs) = self.forward_cache(sample);
            let out = self.upd.forward(&upd_in);
            let err = [out[0] - sample.field[0], out[1] - sample.field[1]];
            total_loss += err[0] * err[0] + err[1] * err[1];
            let grad_out: Vec<f32> = err.iter().map(|e| scale * e).collect();

            let (gw_upd, g_upd_in) = self.upd.backward(&upd_in, &grad_out);
            for (d, s) in grad_upd.iter_mut().zip(gw_upd.iter()) {
                *d += s;
            }
            let grad_agg = &g_upd_in[0..self.hidden];

            // Through the tanh: d/dz = grad_agg * (1 - tanh(z)^2).
            for ((p, u), z) in sample.record.iter().zip(sample.axes.iter()).zip(zs.iter()) {
                let grad_z: Vec<f32> = z
                    .iter()
                    .zip(grad_agg.iter())
                    .map(|(zi, g)| g * (1.0 - zi.tanh() * zi.tanh()))
                    .collect();
                let feat = Self::raw_features(*p, *u);
                let (gw_msg, _g_in) = self.msg.backward(&feat, &grad_z);
                for (d, s) in grad_msg.iter_mut().zip(gw_msg.iter()) {
                    *d += s;
                }
            }
        }

        self.msg.apply_grad(&grad_msg, lr);
        self.upd.apply_grad(&grad_upd, lr);
        total_loss / (n as f32 * 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::magnetometer::{generate_dataset, MagConfig};
    use clifford_math::gp_vjp;

    fn cfg() -> MagConfig {
        MagConfig::default()
    }

    #[test]
    fn clifford_estimate_shape() {
        let data = generate_dataset(&cfg(), 1, 1);
        let model = CliffordMag::new(4);
        let out = model.estimate(&data[0]);
        assert!(out[0].is_finite() && out[1].is_finite());
    }

    #[test]
    fn clifford_training_reduces_loss() {
        let data = generate_dataset(&cfg(), 128, 7);
        let mut model = CliffordMag::new(8);
        let l0 = model.train_step(&data, 0.0);
        for _ in 0..200 {
            model.train_step(&data, 0.1);
        }
        let l1 = model.train_step(&data, 0.0);
        assert!(l1 < l0, "clifford l0={l0} l1={l1}");
    }

    #[test]
    fn flat_training_reduces_loss() {
        let data = generate_dataset(&cfg(), 128, 7);
        let mut model = FlatMag::new(8);
        let l0 = model.train_step(&data, 0.0);
        for _ in 0..200 {
            model.train_step(&data, 0.1);
        }
        let l1 = model.train_step(&data, 0.0);
        assert!(l1 < l0, "flat l0={l0} l1={l1}");
    }

    #[test]
    fn flat_param_matching_is_close() {
        let c = CliffordMag::new(8);
        let target = c.num_params();
        let h = FlatMag::hidden_for_params(target);
        let f = FlatMag::new(h);
        let rel = (f.num_params() as f32 - target as f32).abs() / target as f32;
        assert!(rel < 0.2, "clifford={} flat={}", target, f.num_params());
    }

    #[test]
    fn classical_is_accurate_on_clean_data() {
        let c = MagConfig {
            sigma: 0.0,
            ..cfg()
        };
        let data = generate_dataset(&c, 200, 3);
        let est = ClassicalEstimator::new(c);
        let pairs: Vec<([f32; 2], [f32; 2])> =
            data.iter().map(|s| (est.estimate(s), s.field)).collect();
        let mse = field_mse(&pairs);
        assert!(mse < 1e-6, "classical clean mse={mse}");
    }

    #[test]
    fn interaction_gp_vjp_matches_finite_difference() {
        // The bilinear interaction is a geometric product of two data
        // multivectors (no weights). `gp_vjp` is its reverse: given dL/dg it
        // returns (dL/da, dL/db). Check that against a finite-difference probe.
        let p = 0.62f32;
        let u = [0.6f32, -0.8];
        let a = Multivector2::scalar(p - 0.5);
        let b = Multivector2::vector(u[0], u[1]);
        let g = geometric_product(a, b);
        // Probe loss L = g.e1 + 0.3 g.e2.
        let grad = Multivector2::from_coeffs([0.0, 1.0, 0.3, 0.0]);
        let (ga, gb) = gp_vjp(a, b, grad);
        let eps = 1e-3f32;
        let loss = |mv: Multivector2| mv.e1 + 0.3 * mv.e2;
        let da = (loss(geometric_product(Multivector2::scalar(p - 0.5 + eps), b))
            - loss(geometric_product(Multivector2::scalar(p - 0.5 - eps), b)))
            / (2.0 * eps);
        assert!((ga.s - da).abs() < 1e-3, "ga.s={} fd={}", ga.s, da);
        let dbx = (loss(geometric_product(a, Multivector2::vector(u[0] + eps, u[1])))
            - loss(geometric_product(a, Multivector2::vector(u[0] - eps, u[1]))))
            / (2.0 * eps);
        let dby = (loss(geometric_product(a, Multivector2::vector(u[0], u[1] + eps)))
            - loss(geometric_product(a, Multivector2::vector(u[0], u[1] - eps))))
            / (2.0 * eps);
        assert!((gb.e1 - dbx).abs() < 1e-3, "gb.e1={} fd={}", gb.e1, dbx);
        assert!((gb.e2 - dby).abs() < 1e-3, "gb.e2={} fd={}", gb.e2, dby);
        let _ = g;
    }

    #[test]
    fn clifford_raw_matches_handed_observable() {
        // Forming (P-0.5)*u via the geometric product must equal handing it in.
        let data = generate_dataset(&cfg(), 4, 1);
        let raw = CliffordMagRaw::new(5);
        for s in &data {
            // Compare the internal interaction to the direct product.
            for (p, u) in s.record.iter().zip(s.axes.iter()) {
                let g = CliffordMagRaw::interaction(*p, *u)[0];
                let (gx, gy) = g.vector_part();
                assert!((gx - (p - 0.5) * u[0]).abs() < 1e-6);
                assert!((gy - (p - 0.5) * u[1]).abs() < 1e-6);
                assert!(g.s.abs() < 1e-6 && g.e12.abs() < 1e-6);
            }
        }
        let _ = raw.num_params();
    }

    #[test]
    fn clifford_raw_training_reduces_loss() {
        let data = generate_dataset(&cfg(), 128, 7);
        let mut model = CliffordMagRaw::new(8);
        let l0 = model.train_step(&data, 0.0);
        for _ in 0..200 {
            model.train_step(&data, 0.1);
        }
        let l1 = model.train_step(&data, 0.0);
        assert!(l1 < l0, "clifford_raw l0={l0} l1={l1}");
    }

    #[test]
    fn flat_mlp_training_reduces_loss() {
        let data = generate_dataset(&cfg(), 128, 7);
        let mut model = FlatMlpMag::new(12);
        let l0 = model.train_step(&data, 0.0);
        for _ in 0..400 {
            model.train_step(&data, 0.1);
        }
        let l1 = model.train_step(&data, 0.0);
        assert!(l1 < l0, "flat_mlp l0={l0} l1={l1}");
    }

    #[test]
    fn flat_mlp_param_matching_is_close() {
        let c = CliffordMagRaw::new(8);
        let target = c.num_params();
        let h = FlatMlpMag::hidden_for_params(target);
        let f = FlatMlpMag::new(h);
        let rel = (f.num_params() as f32 - target as f32).abs() / target as f32;
        assert!(
            rel < 0.2,
            "clifford_raw={} flat_mlp={}",
            target,
            f.num_params()
        );
    }

    #[test]
    fn oracle_beats_naive_under_defect() {
        // A distorted sensor: the naive fit ignores the defect, the oracle
        // corrects for it and should be far more accurate.
        let hard = [0.15, -0.1];
        let soft = [1.25, 0.8];
        let c = MagConfig {
            sigma: 0.0,
            hard_iron: hard,
            soft_iron: soft,
            ..cfg()
        };
        let data = generate_dataset(&c, 200, 3);

        let naive = ClassicalEstimator::new(c);
        let oracle = ClassicalEstimator::with_assumptions(c, hard, soft);

        let naive_pairs: Vec<([f32; 2], [f32; 2])> =
            data.iter().map(|s| (naive.estimate(s), s.field)).collect();
        let oracle_pairs: Vec<([f32; 2], [f32; 2])> =
            data.iter().map(|s| (oracle.estimate(s), s.field)).collect();

        let naive_mse = field_mse(&naive_pairs);
        let oracle_mse = field_mse(&oracle_pairs);
        assert!(oracle_mse < 1e-6, "oracle mse={oracle_mse}");
        assert!(
            naive_mse > 100.0 * oracle_mse.max(1e-12),
            "naive={naive_mse} oracle={oracle_mse}"
        );
    }
}
