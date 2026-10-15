//! Reusable evaluation harness: a common model trait plus metrics for one-step
//! error, rotation equivariance, and autoregressive rollout drift.

use crate::models::{delta_mse, target_deltas, CliffordNBody, Delta, FlatNBody};
use crate::nbody::{rotate_state, simulate, NBodyConfig, Sample, State};

/// Common interface so metrics work for both the Clifford and flat models.
pub trait NBodyModel {
    fn forward_state(&self, state: &State) -> Vec<Delta>;
    fn train_batch(&mut self, batch: &[Sample], lr: f32) -> f32;
    fn rollout_state(&self, initial: &State, steps: usize) -> State;
    fn param_count(&self) -> usize;
}

impl NBodyModel for CliffordNBody {
    fn forward_state(&self, state: &State) -> Vec<Delta> {
        self.forward(state)
    }
    fn train_batch(&mut self, batch: &[Sample], lr: f32) -> f32 {
        self.train_step(batch, lr)
    }
    fn rollout_state(&self, initial: &State, steps: usize) -> State {
        self.rollout(initial, steps)
    }
    fn param_count(&self) -> usize {
        self.num_params()
    }
}

impl NBodyModel for FlatNBody {
    fn forward_state(&self, state: &State) -> Vec<Delta> {
        self.forward(state)
    }
    fn train_batch(&mut self, batch: &[Sample], lr: f32) -> f32 {
        self.train_step(batch, lr)
    }
    fn rollout_state(&self, initial: &State, steps: usize) -> State {
        self.rollout(initial, steps)
    }
    fn param_count(&self) -> usize {
        self.num_params()
    }
}

/// Mean one-step delta MSE over a dataset.
pub fn one_step_mse<M: NBodyModel>(model: &M, data: &[Sample]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    for s in data {
        let pred = model.forward_state(&s.input);
        let tgt = target_deltas(&s.input, &s.target);
        acc += delta_mse(&pred, &tgt);
    }
    acc / data.len() as f32
}

fn rotate_vec(v: [f32; 2], theta: f32) -> [f32; 2] {
    let (c, s) = (theta.cos(), theta.sin());
    [c * v[0] - s * v[1], s * v[0] + c * v[1]]
}

/// Rotation equivariance error: RMS difference between rotating the model's
/// prediction and predicting on the rotated input. Zero means exactly
/// equivariant. Averaged over the given rotation angles and the dataset.
pub fn equivariance_error<M: NBodyModel>(model: &M, data: &[Sample], thetas: &[f32]) -> f32 {
    if data.is_empty() || thetas.is_empty() {
        return 0.0;
    }
    let mut sq = 0.0;
    let mut cnt = 0usize;
    for s in data {
        let base = model.forward_state(&s.input);
        for &theta in thetas {
            let rotated_in = rotate_state(&s.input, theta);
            let pred_rot = model.forward_state(&rotated_in);
            for (b, pr) in base.iter().zip(pred_rot.iter()) {
                let rp = rotate_vec(b.dpos, theta);
                let rv = rotate_vec(b.dvel, theta);
                sq += (rp[0] - pr.dpos[0]).powi(2) + (rp[1] - pr.dpos[1]).powi(2);
                sq += (rv[0] - pr.dvel[0]).powi(2) + (rv[1] - pr.dvel[1]).powi(2);
                cnt += 4;
            }
        }
    }
    (sq / cnt as f32).sqrt()
}

/// Autoregressive rollout error: RMS position error after `steps` model steps
/// vs the ground-truth leapfrog trajectory. Each initial state is integrated
/// one true step per model step (model predicts a one-step delta).
pub fn rollout_error<M: NBodyModel>(
    model: &M,
    inits: &[State],
    cfg: &NBodyConfig,
    steps: usize,
) -> f32 {
    if inits.is_empty() {
        return 0.0;
    }
    let mut sq = 0.0;
    let mut cnt = 0usize;
    for init in inits {
        let truth = simulate(init, cfg, steps);
        let pred = model.rollout_state(init, steps);
        let truth_end = &truth[steps];
        for (p, t) in pred.iter().zip(truth_end.iter()) {
            sq += (p.pos[0] - t.pos[0]).powi(2) + (p.pos[1] - t.pos[1]).powi(2);
            cnt += 2;
        }
    }
    (sq / cnt as f32).sqrt()
}

/// Train a model for `epochs` full-batch steps; returns (final_loss, seconds).
pub fn train_model<M: NBodyModel>(
    model: &mut M,
    data: &[Sample],
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
    use crate::models::{CliffordNBody, FlatNBody};
    use crate::nbody::{generate_dataset, random_system, Rng};

    fn cfg() -> NBodyConfig {
        NBodyConfig::default()
    }

    #[test]
    fn metrics_run_for_both_models() {
        let data = generate_dataset(&cfg(), 16, 1, 3);
        let thetas = [0.3, 1.1, 2.0];

        let c = CliffordNBody::new(4);
        let f = FlatNBody::new(FlatNBody::hidden_for_params(c.num_params()));

        // Metrics should produce finite, non-negative numbers.
        assert!(one_step_mse(&c, &data) >= 0.0);
        assert!(one_step_mse(&f, &data) >= 0.0);
        assert!(equivariance_error(&c, &data, &thetas) >= 0.0);
        assert!(equivariance_error(&f, &data, &thetas) >= 0.0);

        let mut rng = Rng::new(1);
        let inits: Vec<State> = (0..4).map(|_| random_system(&cfg(), &mut rng)).collect();
        assert!(rollout_error(&c, &inits, &cfg(), 5).is_finite());
        assert!(rollout_error(&f, &inits, &cfg(), 5).is_finite());
    }

    #[test]
    fn training_improves_one_step_mse() {
        let data = generate_dataset(&cfg(), 64, 1, 11);
        let mut c = CliffordNBody::new(6);
        let before = one_step_mse(&c, &data);
        train_model(&mut c, &data, 60, 0.05);
        let after = one_step_mse(&c, &data);
        assert!(after < before, "before={before} after={after}");
    }
}
