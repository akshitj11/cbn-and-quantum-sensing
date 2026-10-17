//! Reusable evaluation harness for the magnetometer stream: a common model
//! trait plus metrics for field-estimation error and rotation equivariance.
//! Mirrors [`crate::eval`] so the two streams report in the same shape.

use crate::mag_models::{
    field_mse, ClassicalEstimator, CliffordMag, CliffordMagRaw, FlatMag, FlatMlpMag,
};
use crate::magnetometer::{rotate_field, MagConfig, MagSample};

/// Common interface so metrics work for every estimator. The classical
/// estimator has no trainable parameters, so its `train_batch` is a no-op.
pub trait MagModel {
    fn estimate_field(&self, sample: &MagSample) -> [f32; 2];
    fn train_batch(&mut self, batch: &[MagSample], lr: f32) -> f32;
    fn param_count(&self) -> usize;
}

impl MagModel for CliffordMag {
    fn estimate_field(&self, sample: &MagSample) -> [f32; 2] {
        self.estimate(sample)
    }
    fn train_batch(&mut self, batch: &[MagSample], lr: f32) -> f32 {
        self.train_step(batch, lr)
    }
    fn param_count(&self) -> usize {
        self.num_params()
    }
}

impl MagModel for FlatMag {
    fn estimate_field(&self, sample: &MagSample) -> [f32; 2] {
        self.estimate(sample)
    }
    fn train_batch(&mut self, batch: &[MagSample], lr: f32) -> f32 {
        self.train_step(batch, lr)
    }
    fn param_count(&self) -> usize {
        self.num_params()
    }
}

impl MagModel for CliffordMagRaw {
    fn estimate_field(&self, sample: &MagSample) -> [f32; 2] {
        self.estimate(sample)
    }
    fn train_batch(&mut self, batch: &[MagSample], lr: f32) -> f32 {
        self.train_step(batch, lr)
    }
    fn param_count(&self) -> usize {
        self.num_params()
    }
}

impl MagModel for FlatMlpMag {
    fn estimate_field(&self, sample: &MagSample) -> [f32; 2] {
        self.estimate(sample)
    }
    fn train_batch(&mut self, batch: &[MagSample], lr: f32) -> f32 {
        self.train_step(batch, lr)
    }
    fn param_count(&self) -> usize {
        self.num_params()
    }
}

impl MagModel for ClassicalEstimator {
    fn estimate_field(&self, sample: &MagSample) -> [f32; 2] {
        self.estimate(sample)
    }
    fn train_batch(&mut self, _batch: &[MagSample], _lr: f32) -> f32 {
        0.0
    }
    fn param_count(&self) -> usize {
        0
    }
}

/// Mean field-estimation MSE over a dataset.
pub fn estimate_mse<M: MagModel>(model: &M, data: &[MagSample]) -> f32 {
    let pairs: Vec<([f32; 2], [f32; 2])> = data
        .iter()
        .map(|s| (model.estimate_field(s), s.field))
        .collect();
    field_mse(&pairs)
}

/// Rotation equivariance error: RMS gap between rotating the model's estimate
/// and estimating on the rotated input. The group acts on the geometric part of
/// the input (the measurement axes) while the probability record stays fixed, so
/// rotating the apparatus by `theta` should rotate the estimate by `theta`.
/// Zero means exactly equivariant. Averaged over angles and the dataset.
pub fn equivariance_error<M: MagModel>(model: &M, data: &[MagSample], thetas: &[f32]) -> f32 {
    if data.is_empty() || thetas.is_empty() {
        return 0.0;
    }
    let mut sq = 0.0;
    let mut cnt = 0usize;
    for s in data {
        let base = model.estimate_field(s);
        for &theta in thetas {
            let rotated = MagSample {
                record: s.record.clone(),
                axes: s.axes.iter().map(|&u| rotate_field(u, theta)).collect(),
                field: rotate_field(s.field, theta),
                theta: s.theta + theta,
            };
            let pred_rot = model.estimate_field(&rotated);
            let expect = rotate_field(base, theta);
            sq += (expect[0] - pred_rot[0]).powi(2) + (expect[1] - pred_rot[1]).powi(2);
            cnt += 2;
        }
    }
    (sq / cnt as f32).sqrt()
}

/// Cramer-Rao lower bound on the field-estimation MSE for the Ramsey +
/// binomial-readout protocol, evaluated at the true `field` with the given
/// `axes` and `n_shots`.
///
/// For axis `u_k`: `phi_k = gamma T (B.u_k)`, `P_k = 0.5(1 + V sin(phi_k))`,
/// so `dP_k/dB = 0.5 V gamma T cos(phi_k) * u_k`, and a binomial estimate from
/// `n_shots` shots has variance `Var_k = P_k(1-P_k)/n_shots`. The Fisher matrix
/// is `F = sum_k (dP_k/dB)(dP_k/dB)^T / Var_k`, and the bound on the summed
/// two-component MSE is `trace(F^{-1})`. Returns that trace (an MSE); take its
/// square root for an RMS field error.
pub fn cramer_rao_bound(
    cfg: &MagConfig,
    axes: &[[f32; 2]],
    field: [f32; 2],
    n_shots: usize,
) -> f32 {
    let v = cfg.contrast();
    let gt = cfg.gamma * cfg.t_evolve;
    let b = [
        cfg.soft_iron[0] * field[0] + cfg.hard_iron[0],
        cfg.soft_iron[1] * field[1] + cfg.hard_iron[1],
    ];

    // Accumulate the 2x2 Fisher information matrix.
    let (mut fxx, mut fxy, mut fyy) = (0.0f32, 0.0f32, 0.0f32);
    for u in axes {
        let proj = b[0] * u[0] + b[1] * u[1];
        let phi = gt * proj;
        let p = 0.5 * (1.0 + v * phi.sin());
        let dp = 0.5 * v * gt * phi.cos(); // dP/dproj
        let var = (p * (1.0 - p) / n_shots as f32).max(1e-12);
        let w = dp * dp / var;
        fxx += w * u[0] * u[0];
        fxy += w * u[0] * u[1];
        fyy += w * u[1] * u[1];
    }
    let det = fxx * fyy - fxy * fxy;
    if det.abs() < 1e-12 {
        return f32::INFINITY;
    }
    // trace(F^{-1}) = (Fxx + Fyy) / det  for a 2x2 matrix.
    (fxx + fyy) / det
}

/// Mean Cramer-Rao MSE bound over a dataset (one bound per sample's geometry).
pub fn mean_crb(cfg: &MagConfig, data: &[MagSample], n_shots: usize) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f32 = data
        .iter()
        .map(|s| cramer_rao_bound(cfg, &s.axes, s.field, n_shots))
        .sum();
    sum / data.len() as f32
}

/// Shot-scaled field sensitivity: `gamma * T * sqrt(n_shots) * rms_error`. A
/// shot-count-independent figure of merit (lower is better) that lets runs at
/// different `n_shots` be compared on one axis.
pub fn sensitivity(cfg: &MagConfig, n_shots: usize, rms_error: f32) -> f32 {
    cfg.gamma * cfg.t_evolve * (n_shots as f32).sqrt() * rms_error
}

/// Train a model for `epochs` full-batch steps; returns (final_loss, seconds).
pub fn train_model<M: MagModel>(
    model: &mut M,
    data: &[MagSample],
    epochs: usize,
    lr: f32,
) -> (f32, f64) {
    let start = std::time::Instant::now();
    let mut loss = 0.0;
    for _ in 0..epochs {
        loss = model.train_batch(data, lr);
    }
    (loss, start.elapsed().as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mag_models::{ClassicalEstimator, CliffordMag, FlatMag};
    use crate::magnetometer::{generate_dataset, MagConfig};

    fn cfg() -> MagConfig {
        MagConfig::default()
    }

    #[test]
    fn metrics_run_for_all_models() {
        let data = generate_dataset(&cfg(), 16, 1);
        let thetas = [0.3, 1.1, 2.0];

        let c = CliffordMag::new(6);
        let f = FlatMag::new(FlatMag::hidden_for_params(c.num_params()));
        let k = ClassicalEstimator::new(cfg());

        assert!(estimate_mse(&c, &data) >= 0.0);
        assert!(estimate_mse(&f, &data) >= 0.0);
        assert!(estimate_mse(&k, &data) >= 0.0);
        assert!(equivariance_error(&c, &data, &thetas) >= 0.0);
        assert!(equivariance_error(&f, &data, &thetas) >= 0.0);
    }

    #[test]
    fn classical_is_near_equivariant() {
        // The classical estimator inverts the geometry exactly, so rotating the
        // apparatus rotates its estimate: equivariance error should be tiny.
        let c = MagConfig {
            sigma: 0.0,
            ..cfg()
        };
        let data = generate_dataset(&c, 64, 2);
        let k = ClassicalEstimator::new(c);
        let err = equivariance_error(&k, &data, &[0.5, 1.7, 3.0]);
        assert!(err < 1e-4, "classical equivariance err={err}");
    }

    #[test]
    fn crb_decreases_with_more_shots() {
        use crate::magnetometer::base_axes;
        let c = cfg();
        let axes = base_axes(c.m);
        let field = [0.3, -0.2];
        let crb_32 = cramer_rao_bound(&c, &axes, field, 32);
        let crb_512 = cramer_rao_bound(&c, &axes, field, 512);
        assert!(crb_32.is_finite() && crb_512.is_finite());
        // Fisher information is linear in n_shots, so the bound scales as 1/n.
        let ratio = crb_32 / crb_512;
        assert!((ratio - 16.0).abs() < 0.5, "ratio={ratio}");
    }

    #[test]
    fn classical_approaches_crb_under_shot_noise() {
        // With many shots the classical estimator should sit near the bound.
        let c = MagConfig {
            n_shots: Some(512),
            ..cfg()
        };
        let data = generate_dataset(&c, 400, 2);
        let est = ClassicalEstimator::new(c);
        let rms = estimate_mse(&est, &data).sqrt();
        let crb_rms = mean_crb(&c, &data, 512).sqrt();
        // Classical should be within a modest factor of the bound (asin
        // nonlinearity and finite sampling keep it slightly above).
        assert!(rms < 3.0 * crb_rms, "rms={rms} crb_rms={crb_rms}");
        assert!(rms >= 0.5 * crb_rms, "rms={rms} crb_rms={crb_rms}");
    }

    #[test]
    fn training_improves_estimate_mse() {
        let data = generate_dataset(&cfg(), 128, 11);
        let mut c = CliffordMag::new(8);
        let before = estimate_mse(&c, &data);
        train_model(&mut c, &data, 200, 0.1);
        let after = estimate_mse(&c, &data);
        assert!(after < before, "before={before} after={after}");
    }
}
