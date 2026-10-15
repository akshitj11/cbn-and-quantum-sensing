//! 2D gravitational n-body system: leapfrog integrator and a deterministic
//! dataset builder. Dependency-free (hand-rolled PRNG) so it stays CPU-cheap
//! and fully reproducible from a seed.

/// Minimal splitmix64-based PRNG. Deterministic, no external deps.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero state degeneracy.
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    pub fn unit(&mut self) -> f32 {
        // Top 24 bits -> [0,1).
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Uniform f32 in [lo, hi).
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// A single particle in 2D: mass is a scalar, position/velocity are vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub mass: f32,
    pub pos: [f32; 2],
    pub vel: [f32; 2],
}

/// Full system state: one entry per particle.
pub type State = Vec<Particle>;

/// Physical and integrator parameters.
#[derive(Clone, Copy, Debug)]
pub struct NBodyConfig {
    pub n: usize,
    /// Gravitational constant.
    pub g: f32,
    /// Plummer softening length; prevents singular accelerations at close range.
    pub softening: f32,
    /// Integrator time step.
    pub dt: f32,
}

impl Default for NBodyConfig {
    fn default() -> Self {
        Self {
            n: 3,
            g: 1.0,
            softening: 0.1,
            dt: 0.01,
        }
    }
}

/// Softened gravitational accelerations for every particle.
pub fn accelerations(state: &State, cfg: &NBodyConfig) -> Vec<[f32; 2]> {
    let n = state.len();
    let mut acc = vec![[0.0f32; 2]; n];
    let soft2 = cfg.softening * cfg.softening;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = state[j].pos[0] - state[i].pos[0];
            let dy = state[j].pos[1] - state[i].pos[1];
            let r2 = dx * dx + dy * dy + soft2;
            let inv_r3 = 1.0 / (r2 * r2.sqrt());
            let s = cfg.g * state[j].mass * inv_r3;
            acc[i][0] += s * dx;
            acc[i][1] += s * dy;
        }
    }
    acc
}

/// One velocity-Verlet (leapfrog) step. Masses are preserved.
pub fn leapfrog_step(state: &State, cfg: &NBodyConfig) -> State {
    let dt = cfg.dt;
    let a0 = accelerations(state, cfg);

    // Drift positions a full step using current velocity + half-step accel.
    let mut next: State = state.clone();
    for i in 0..next.len() {
        next[i].pos[0] += state[i].vel[0] * dt + 0.5 * a0[i][0] * dt * dt;
        next[i].pos[1] += state[i].vel[1] * dt + 0.5 * a0[i][1] * dt * dt;
    }

    // Kick velocities using the average of old and new accelerations.
    let a1 = accelerations(&next, cfg);
    for i in 0..next.len() {
        next[i].vel[0] = state[i].vel[0] + 0.5 * (a0[i][0] + a1[i][0]) * dt;
        next[i].vel[1] = state[i].vel[1] + 0.5 * (a0[i][1] + a1[i][1]) * dt;
    }
    next
}

/// Integrate `steps` leapfrog steps, returning the full trajectory including
/// the initial state (length == steps + 1).
pub fn simulate(initial: &State, cfg: &NBodyConfig, steps: usize) -> Vec<State> {
    let mut traj = Vec::with_capacity(steps + 1);
    traj.push(initial.clone());
    let mut cur = initial.clone();
    for _ in 0..steps {
        cur = leapfrog_step(&cur, cfg);
        traj.push(cur.clone());
    }
    traj
}

/// Total mechanical energy (kinetic + softened potential). Useful as a sanity
/// check: leapfrog should keep this roughly constant over a trajectory.
pub fn total_energy(state: &State, cfg: &NBodyConfig) -> f32 {
    let n = state.len();
    let mut ke = 0.0;
    for p in state {
        ke += 0.5 * p.mass * (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]);
    }
    let soft2 = cfg.softening * cfg.softening;
    let mut pe = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = state[j].pos[0] - state[i].pos[0];
            let dy = state[j].pos[1] - state[i].pos[1];
            let r = (dx * dx + dy * dy + soft2).sqrt();
            pe -= cfg.g * state[i].mass * state[j].mass / r;
        }
    }
    ke + pe
}

/// Sample a random initial system: particles near the origin with small
/// velocities and unit-ish masses.
pub fn random_system(cfg: &NBodyConfig, rng: &mut Rng) -> State {
    (0..cfg.n)
        .map(|_| Particle {
            mass: rng.range(0.5, 1.5),
            pos: [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)],
            vel: [rng.range(-0.3, 0.3), rng.range(-0.3, 0.3)],
        })
        .collect()
}

/// One supervised example: predict the system `k` steps ahead.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub input: State,
    pub target: State,
}

/// Build a deterministic dataset of `(state_t, state_{t+k})` pairs.
///
/// For each of `n_samples` random initial conditions we integrate `k_steps`
/// leapfrog steps and pair the start with the end.
pub fn generate_dataset(
    cfg: &NBodyConfig,
    n_samples: usize,
    k_steps: usize,
    seed: u64,
) -> Vec<Sample> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let init = random_system(cfg, &mut rng);
        let traj = simulate(&init, cfg, k_steps);
        out.push(Sample {
            input: init,
            target: traj[k_steps].clone(),
        });
    }
    out
}

/// Rotate an entire state about the origin by `theta` (positions and
/// velocities; masses unchanged). Used for equivariance testing.
pub fn rotate_state(state: &State, theta: f32) -> State {
    let (c, s) = (theta.cos(), theta.sin());
    state
        .iter()
        .map(|p| Particle {
            mass: p.mass,
            pos: [c * p.pos[0] - s * p.pos[1], s * p.pos[0] + c * p.pos[1]],
            vel: [c * p.vel[0] - s * p.vel[1], s * p.vel[0] + c * p.vel[1]],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn rng_unit_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u), "u={u}");
        }
    }

    #[test]
    fn dataset_is_reproducible() {
        let cfg = NBodyConfig::default();
        let a = generate_dataset(&cfg, 10, 5, 123);
        let b = generate_dataset(&cfg, 10, 5, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn energy_is_approximately_conserved() {
        let cfg = NBodyConfig {
            n: 3,
            g: 1.0,
            softening: 0.2,
            dt: 0.005,
        };
        let mut rng = Rng::new(99);
        let init = random_system(&cfg, &mut rng);
        let traj = simulate(&init, &cfg, 200);
        let e0 = total_energy(&traj[0], &cfg);
        let e1 = total_energy(&traj[traj.len() - 1], &cfg);
        let rel = (e1 - e0).abs() / e0.abs().max(1e-6);
        assert!(rel < 0.05, "energy drift too large: e0={e0} e1={e1}");
    }

    #[test]
    fn rotation_commutes_with_dynamics() {
        // Physics is rotation-equivariant: rotating the initial state then
        // integrating equals integrating then rotating.
        let cfg = NBodyConfig {
            n: 4,
            g: 1.0,
            softening: 0.2,
            dt: 0.005,
        };
        let mut rng = Rng::new(5);
        let init = random_system(&cfg, &mut rng);
        let theta = 0.6;

        let rot_then_sim = simulate(&rotate_state(&init, theta), &cfg, 50);
        let sim_then_rot = rotate_state(&simulate(&init, &cfg, 50)[50], theta);

        let end = &rot_then_sim[50];
        for (p, q) in end.iter().zip(sim_then_rot.iter()) {
            assert!((p.pos[0] - q.pos[0]).abs() < 1e-3);
            assert!((p.pos[1] - q.pos[1]).abs() < 1e-3);
            assert!((p.vel[0] - q.vel[0]).abs() < 1e-3);
            assert!((p.vel[1] - q.vel[1]).abs() < 1e-3);
        }
    }
}
