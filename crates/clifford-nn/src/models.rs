//! Interaction-aware models for n-body delta-state prediction.
//!
//! Both models share the same message-passing topology so the comparison is
//! fair:
//!   1. a shared per-pair *message* layer over relative features,
//!   2. a permutation-invariant sum over neighbours,
//!   3. a shared *update* layer mapping (aggregated message + self) to a
//!      per-particle delta (position and velocity change).
//!
//! The Clifford variant carries multivector channels through `CliffordLinear`
//! geometric products; the flat baseline carries plain real vectors through
//! `FlatLinear`. Parameter counts are reported by both so they can be matched.

use crate::nbody::State;
use crate::{CliffordLinear, FlatLinear};
use clifford_math::Multivector2;

/// Per-particle prediction: change in position and velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Delta {
    pub dpos: [f32; 2],
    pub dvel: [f32; 2],
}

/// Ground-truth delta between an input state and its target state.
pub fn target_deltas(input: &State, target: &State) -> Vec<Delta> {
    input
        .iter()
        .zip(target.iter())
        .map(|(a, b)| Delta {
            dpos: [b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]],
            dvel: [b.vel[0] - a.vel[0], b.vel[1] - a.vel[1]],
        })
        .collect()
}

/// MSE over predicted vs target deltas (position and velocity components).
pub fn delta_mse(pred: &[Delta], tgt: &[Delta]) -> f32 {
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for (p, t) in pred.iter().zip(tgt.iter()) {
        sum += (p.dpos[0] - t.dpos[0]).powi(2) + (p.dpos[1] - t.dpos[1]).powi(2);
        sum += (p.dvel[0] - t.dvel[0]).powi(2) + (p.dvel[1] - t.dvel[1]).powi(2);
        cnt += 4;
    }
    sum / cnt as f32
}

// A small multivector gradient carrying only vector-grade error.
fn vec_grad(ex: f32, ey: f32) -> Multivector2 {
    Multivector2::from_coeffs([0.0, ex, ey, 0.0])
}

// ----------------------------------------------------------------------------
// Clifford model
// ----------------------------------------------------------------------------

/// Message-passing Clifford network.
///
/// Message input channels (3): `[scalar(mass_j), vector(r_ij), vector(v_ij)]`.
/// Update input channels (`hidden + 3`): aggregated messages ++ self state
/// `[scalar(mass_i), vector(pos_i), vector(vel_i)]`.
/// Output channels (2): delta-position and delta-velocity (vector parts used).
#[derive(Clone, Debug)]
pub struct CliffordNBody {
    pub msg: CliffordLinear,
    pub upd: CliffordLinear,
    pub hidden: usize,
}

impl CliffordNBody {
    pub fn new(hidden: usize) -> Self {
        Self {
            msg: CliffordLinear::new(3, hidden),
            upd: CliffordLinear::new(hidden + 3, 2),
            hidden,
        }
    }

    pub fn num_params(&self) -> usize {
        self.msg.num_scalar_params() + self.upd.num_scalar_params()
    }

    fn self_channels(state: &State, i: usize) -> Vec<Multivector2> {
        let p = state[i];
        vec![
            Multivector2::scalar(p.mass),
            Multivector2::vector(p.pos[0], p.pos[1]),
            Multivector2::vector(p.vel[0], p.vel[1]),
        ]
    }

    fn pair_channels(state: &State, i: usize, j: usize) -> Vec<Multivector2> {
        let (a, b) = (state[i], state[j]);
        vec![
            Multivector2::scalar(b.mass),
            Multivector2::vector(b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]),
            Multivector2::vector(b.vel[0] - a.vel[0], b.vel[1] - a.vel[1]),
        ]
    }

    // Aggregated messages + full update input for particle i.
    fn update_input(&self, state: &State, i: usize) -> Vec<Multivector2> {
        let n = state.len();
        let mut agg = vec![Multivector2::ZERO; self.hidden];
        for j in 0..n {
            if j == i {
                continue;
            }
            let pair = Self::pair_channels(state, i, j);
            let out = self.msg.forward_cell(&pair);
            for c in 0..self.hidden {
                agg[c] = agg[c] + out[c];
            }
        }
        let mut upd_in = agg;
        upd_in.extend(Self::self_channels(state, i));
        upd_in
    }

    pub fn forward(&self, state: &State) -> Vec<Delta> {
        let n = state.len();
        (0..n)
            .map(|i| {
                let upd_in = self.update_input(state, i);
                let out = self.upd.forward_cell(&upd_in);
                let (dpx, dpy) = out[0].vector_part();
                let (dvx, dvy) = out[1].vector_part();
                Delta {
                    dpos: [dpx, dpy],
                    dvel: [dvx, dvy],
                }
            })
            .collect()
    }

    /// One full-batch analytic SGD step. Returns the pre-update batch loss.
    pub fn train_step(&mut self, batch: &[crate::nbody::Sample], lr: f32) -> f32 {
        let mut grad_msg = self.msg.zero_grad();
        let mut grad_upd = self.upd.zero_grad();
        let mut total_loss = 0.0;
        let total_targets: usize = batch.iter().map(|s| s.input.len() * 4).sum();

        for sample in batch {
            let state = &sample.input;
            let tgt = target_deltas(&sample.input, &sample.target);
            let n = state.len();
            for (i, t) in tgt.iter().enumerate() {
                let upd_in = self.update_input(state, i);
                let out = self.upd.forward_cell(&upd_in);
                let (dpx, dpy) = out[0].vector_part();
                let (dvx, dvy) = out[1].vector_part();

                total_loss += (dpx - t.dpos[0]).powi(2)
                    + (dpy - t.dpos[1]).powi(2)
                    + (dvx - t.dvel[0]).powi(2)
                    + (dvy - t.dvel[1]).powi(2);

                let scale = 2.0 / total_targets as f32;
                let mut grad_out = vec![Multivector2::ZERO; 2];
                grad_out[0] = vec_grad(scale * (dpx - t.dpos[0]), scale * (dpy - t.dpos[1]));
                grad_out[1] = vec_grad(scale * (dvx - t.dvel[0]), scale * (dvy - t.dvel[1]));

                let (gw_upd, g_upd_in) = self.upd.backward_cell(&upd_in, &grad_out);
                CliffordLinear::accumulate_grad(&mut grad_upd, &gw_upd);

                // First `hidden` input grads flow back into each message.
                let grad_agg = &g_upd_in[0..self.hidden];
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let pair = Self::pair_channels(state, i, j);
                    let (gw_msg, _g_pair) = self.msg.backward_cell(pair.as_slice(), grad_agg);
                    CliffordLinear::accumulate_grad(&mut grad_msg, &gw_msg);
                }
            }
        }

        self.msg.apply_grad(&grad_msg, lr);
        self.upd.apply_grad(&grad_upd, lr);
        total_loss / total_targets as f32
    }

    /// Autoregressive rollout: apply predicted deltas `steps` times.
    pub fn rollout(&self, initial: &State, steps: usize) -> State {
        let mut cur = initial.clone();
        for _ in 0..steps {
            let d = self.forward(&cur);
            for (p, delta) in cur.iter_mut().zip(d.iter()) {
                p.pos[0] += delta.dpos[0];
                p.pos[1] += delta.dpos[1];
                p.vel[0] += delta.dvel[0];
                p.vel[1] += delta.dvel[1];
            }
        }
        cur
    }
}

// ----------------------------------------------------------------------------
// Flat baseline (matched topology on real vectors)
// ----------------------------------------------------------------------------

/// Per-particle self features: `[mass, px, py, vx, vy]`.
const SELF_DIM: usize = 5;
/// Per-pair features: `[mass_j, rpx, rpy, rvx, rvy]`.
const PAIR_DIM: usize = 5;
/// Output features: `[dpx, dpy, dvx, dvy]`.
const OUT_DIM: usize = 4;

/// Flat message-passing baseline mirroring `CliffordNBody`'s structure.
#[derive(Clone, Debug)]
pub struct FlatNBody {
    pub msg: FlatLinear,
    pub upd: FlatLinear,
    pub hidden: usize,
}

impl FlatNBody {
    pub fn new(hidden: usize) -> Self {
        Self {
            msg: FlatLinear::new(PAIR_DIM, hidden),
            upd: FlatLinear::new(hidden + SELF_DIM, OUT_DIM),
            hidden,
        }
    }

    pub fn num_params(&self) -> usize {
        self.msg.w.len() + self.upd.w.len()
    }

    /// Pick the hidden width whose param count is closest to `target_params`.
    pub fn hidden_for_params(target_params: usize) -> usize {
        // params(h) = PAIR_DIM*h + (h+SELF_DIM)*OUT_DIM
        //           = (PAIR_DIM + OUT_DIM) * h + SELF_DIM*OUT_DIM
        let slope = (PAIR_DIM + OUT_DIM) as f32;
        let intercept = (SELF_DIM * OUT_DIM) as f32;
        let h = ((target_params as f32 - intercept) / slope).round();
        (h as isize).max(1) as usize
    }

    fn self_features(state: &State, i: usize) -> Vec<f32> {
        let p = state[i];
        vec![p.mass, p.pos[0], p.pos[1], p.vel[0], p.vel[1]]
    }

    fn pair_features(state: &State, i: usize, j: usize) -> Vec<f32> {
        let (a, b) = (state[i], state[j]);
        vec![
            b.mass,
            b.pos[0] - a.pos[0],
            b.pos[1] - a.pos[1],
            b.vel[0] - a.vel[0],
            b.vel[1] - a.vel[1],
        ]
    }

    fn update_input(&self, state: &State, i: usize) -> Vec<f32> {
        let n = state.len();
        let mut agg = vec![0.0; self.hidden];
        for j in 0..n {
            if j == i {
                continue;
            }
            let out = self.msg.forward(&Self::pair_features(state, i, j));
            for c in 0..self.hidden {
                agg[c] += out[c];
            }
        }
        agg.extend(Self::self_features(state, i));
        agg
    }

    pub fn forward(&self, state: &State) -> Vec<Delta> {
        let n = state.len();
        (0..n)
            .map(|i| {
                let out = self.upd.forward(&self.update_input(state, i));
                Delta {
                    dpos: [out[0], out[1]],
                    dvel: [out[2], out[3]],
                }
            })
            .collect()
    }

    pub fn train_step(&mut self, batch: &[crate::nbody::Sample], lr: f32) -> f32 {
        let mut grad_msg = vec![0.0; self.msg.w.len()];
        let mut grad_upd = vec![0.0; self.upd.w.len()];
        let mut total_loss = 0.0;
        let total_targets: usize = batch.iter().map(|s| s.input.len() * 4).sum();
        let scale = 2.0 / total_targets as f32;

        for sample in batch {
            let state = &sample.input;
            let tgt = target_deltas(&sample.input, &sample.target);
            let n = state.len();
            for (i, t) in tgt.iter().enumerate() {
                let upd_in = self.update_input(state, i);
                let out = self.upd.forward(&upd_in);
                let err = [
                    out[0] - t.dpos[0],
                    out[1] - t.dpos[1],
                    out[2] - t.dvel[0],
                    out[3] - t.dvel[1],
                ];
                for e in err {
                    total_loss += e * e;
                }
                let grad_out: Vec<f32> = err.iter().map(|e| scale * e).collect();

                let (gw_upd, g_upd_in) = self.upd.backward(&upd_in, &grad_out);
                for (d, s) in grad_upd.iter_mut().zip(gw_upd.iter()) {
                    *d += s;
                }
                let grad_agg = &g_upd_in[0..self.hidden];
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let pair = Self::pair_features(state, i, j);
                    let (gw_msg, _g_pair) = self.msg.backward(&pair, grad_agg);
                    for (d, s) in grad_msg.iter_mut().zip(gw_msg.iter()) {
                        *d += s;
                    }
                }
            }
        }

        self.msg.apply_grad(&grad_msg, lr);
        self.upd.apply_grad(&grad_upd, lr);
        total_loss / total_targets as f32
    }

    pub fn rollout(&self, initial: &State, steps: usize) -> State {
        let mut cur = initial.clone();
        for _ in 0..steps {
            let d = self.forward(&cur);
            for (p, delta) in cur.iter_mut().zip(d.iter()) {
                p.pos[0] += delta.dpos[0];
                p.pos[1] += delta.dpos[1];
                p.vel[0] += delta.dvel[0];
                p.vel[1] += delta.dvel[1];
            }
        }
        cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nbody::{generate_dataset, NBodyConfig};

    #[test]
    fn clifford_forward_shape() {
        let cfg = NBodyConfig::default();
        let data = generate_dataset(&cfg, 1, 1, 1);
        let model = CliffordNBody::new(4);
        let out = model.forward(&data[0].input);
        assert_eq!(out.len(), cfg.n);
    }

    #[test]
    fn clifford_training_reduces_loss() {
        let cfg = NBodyConfig::default();
        let data = generate_dataset(&cfg, 64, 1, 7);
        let mut model = CliffordNBody::new(6);
        let l0 = model.train_step(&data, 0.0); // measure only
        for _ in 0..50 {
            model.train_step(&data, 0.05);
        }
        let l1 = model.train_step(&data, 0.0);
        assert!(l1 < l0, "clifford l0={l0} l1={l1}");
    }

    #[test]
    fn flat_training_reduces_loss() {
        let cfg = NBodyConfig::default();
        let data = generate_dataset(&cfg, 64, 1, 7);
        let mut model = FlatNBody::new(8);
        let l0 = model.train_step(&data, 0.0);
        for _ in 0..50 {
            model.train_step(&data, 0.05);
        }
        let l1 = model.train_step(&data, 0.0);
        assert!(l1 < l0, "flat l0={l0} l1={l1}");
    }

    #[test]
    fn flat_param_matching_is_close() {
        let c = CliffordNBody::new(6);
        let target = c.num_params();
        let h = FlatNBody::hidden_for_params(target);
        let f = FlatNBody::new(h);
        let rel = (f.num_params() as f32 - target as f32).abs() / target as f32;
        assert!(rel < 0.15, "clifford={} flat={}", target, f.num_params());
    }
}
