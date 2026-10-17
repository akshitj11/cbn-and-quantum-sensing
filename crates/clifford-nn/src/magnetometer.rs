//! Planar qubit-magnetometer simulator (Cl(2,0) stream).
//!
//! A spin-1/2 sensor estimates a 2D magnetic field `B = (Bx, By)` from a set of
//! Ramsey-interferometry measurements taken along several sensor axes. The
//! physics is deliberately minimal and dependency-free (hand-rolled PRNG) so the
//! whole study stays CPU-cheap and reproducible from a seed, mirroring
//! [`crate::nbody`].
//!
//! # Observation model
//!
//! For a measurement along a unit axis `u_k`, the spin precesses (Larmor) about
//! the field. In a Ramsey sequence biased to the steepest point of the fringe
//! (control phase `-pi/2`), the excited-state probability is
//!
//! ```text
//!   phi_k = gamma * T * (B . u_k)                 (accumulated phase)
//!   P_k   = 0.5 * (1 + V * sin(phi_k)) + noise    (V = exp(-T / T2))
//! ```
//!
//! `V` is the fringe contrast reduced by T2 decoherence, and `noise` is Gaussian
//! readout noise with standard deviation `sigma`. The field magnitude is kept
//! small enough that `phi_k` stays within roughly `[-1, 1]` rad, so `sin` is
//! monotonic and the record is invertible (the regime real sensors operate in).
//!
//! # Geometry
//!
//! The field is a vector (grade-1) object. A measurement records the projection
//! of `B` onto a sensor axis, so projections are invariant under a simultaneous
//! rotation of the field and the axes. Equivalently, if the whole apparatus
//! rotates by `R`, the record is unchanged and the correct field estimate
//! rotates by `R`. That is the ground-truth symmetry the models are measured
//! against in [`crate::mag_eval`].

/// Splitmix64-based PRNG with a Gaussian helper. Deterministic, no deps.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
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
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Uniform f32 in [lo, hi).
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    /// Standard normal sample via Box-Muller. Returns one of the pair.
    pub fn normal(&mut self) -> f32 {
        // Guard against log(0).
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Number of successes in `n` Bernoulli trials with success probability
    /// `p`. Direct simulation (fine for the small shot counts used here), the
    /// physical readout statistic of a projective qubit measurement.
    pub fn binomial(&mut self, n: usize, p: f32) -> usize {
        let p = p.clamp(0.0, 1.0);
        (0..n).filter(|_| self.unit() < p).count()
    }
}

/// Physical and protocol parameters for the magnetometer.
#[derive(Clone, Copy, Debug)]
pub struct MagConfig {
    /// Number of measurement axes (and measurements) per sample.
    pub m: usize,
    /// Gyromagnetic constant.
    pub gamma: f32,
    /// Ramsey free-evolution time.
    pub t_evolve: f32,
    /// T2 decoherence time (sets fringe contrast `V = exp(-T/T2)`).
    pub t2: f32,
    /// Gaussian readout-noise standard deviation on each probability.
    pub sigma: f32,
    /// Maximum field magnitude sampled (kept so `gamma*T*|B| <~ 1`).
    pub b_max: f32,
    /// Hard-iron bias added to the field before projection (sensor defect).
    /// Default zero; used by the calibration variant of the study.
    pub hard_iron: [f32; 2],
    /// Soft-iron per-axis gain distortion applied to the field before the bias:
    /// `B_eff = diag(soft_iron) * B + hard_iron`. Default `[1, 1]` (no defect).
    pub soft_iron: [f32; 2],
    /// Projective readout shot count. `Some(k)` makes each probability a
    /// binomial estimate from `k` shots (`count / k`); `None` uses the Gaussian
    /// `sigma` model instead. Default `None`.
    pub n_shots: Option<usize>,
}

impl Default for MagConfig {
    fn default() -> Self {
        // gamma*T*b_max = 0.8 rad keeps sin() monotonic / invertible.
        Self {
            m: 6,
            gamma: 1.0,
            t_evolve: 1.0,
            t2: 5.0,
            sigma: 0.02,
            b_max: 0.8,
            hard_iron: [0.0, 0.0],
            soft_iron: [1.0, 1.0],
            n_shots: None,
        }
    }
}

impl MagConfig {
    /// Fringe contrast after T2 decoherence.
    pub fn contrast(&self) -> f32 {
        (-self.t_evolve / self.t2).exp()
    }
}

/// Rotate a 2D vector by `theta` (counter-clockwise).
pub fn rotate_field(v: [f32; 2], theta: f32) -> [f32; 2] {
    let (c, s) = (theta.cos(), theta.sin());
    [c * v[0] - s * v[1], s * v[0] + c * v[1]]
}

/// The `m` base measurement axes, evenly spread over the half-circle. Spreading
/// over `pi` (not `2*pi`) avoids redundant antipodal axes that carry the same
/// projection information.
pub fn base_axes(m: usize) -> Vec<[f32; 2]> {
    (0..m)
        .map(|k| {
            let a = std::f32::consts::PI * k as f32 / m as f32;
            [a.cos(), a.sin()]
        })
        .collect()
}

/// Noiseless Ramsey probability for a projection `b_proj = B . u_k`.
pub fn ramsey_probability(b_proj: f32, cfg: &MagConfig) -> f32 {
    let phi = cfg.gamma * cfg.t_evolve * b_proj;
    0.5 * (1.0 + cfg.contrast() * phi.sin())
}

/// One supervised example: estimate the world-frame field `field` from the
/// probability `record`, given the measurement `axes` used (world frame).
///
/// `theta` is the sensor orientation applied to the base axes for this sample;
/// it is stored for reference (the axes already incorporate it).
#[derive(Clone, Debug, PartialEq)]
pub struct MagSample {
    pub record: Vec<f32>,
    pub axes: Vec<[f32; 2]>,
    pub field: [f32; 2],
    pub theta: f32,
}

impl MagSample {
    /// Produce the noiseless record for this sample's field and axes.
    ///
    /// Applies the sensor distortion `B_eff = diag(soft_iron) * B + hard_iron`
    /// before projecting onto each axis.
    pub fn noiseless_record(field: [f32; 2], axes: &[[f32; 2]], cfg: &MagConfig) -> Vec<f32> {
        let b = [
            cfg.soft_iron[0] * field[0] + cfg.hard_iron[0],
            cfg.soft_iron[1] * field[1] + cfg.hard_iron[1],
        ];
        axes.iter()
            .map(|u| ramsey_probability(b[0] * u[0] + b[1] * u[1], cfg))
            .collect()
    }
}

/// Sample a random field inside the disk of radius `b_max`.
fn random_field(cfg: &MagConfig, rng: &mut Rng) -> [f32; 2] {
    // Uniform in the disk: r = b_max * sqrt(u) for area-uniform sampling.
    let r = cfg.b_max * rng.unit().sqrt();
    let a = rng.range(0.0, std::f32::consts::TAU);
    [r * a.cos(), r * a.sin()]
}

/// Build a deterministic dataset of magnetometer samples.
///
/// Each sample draws a random field and a random sensor orientation, rotates the
/// base axes by that orientation, and records the noisy Ramsey probabilities.
pub fn generate_dataset(cfg: &MagConfig, n_samples: usize, seed: u64) -> Vec<MagSample> {
    let mut rng = Rng::new(seed);
    let base = base_axes(cfg.m);
    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let field = random_field(cfg, &mut rng);
        let theta = rng.range(0.0, std::f32::consts::TAU);
        let axes: Vec<[f32; 2]> = base.iter().map(|&u| rotate_field(u, theta)).collect();
        let clean = MagSample::noiseless_record(field, &axes, cfg);
        let record: Vec<f32> = match cfg.n_shots {
            // Projective shot noise: each probability is a binomial estimate.
            Some(k) if k > 0 => clean
                .iter()
                .map(|p| rng.binomial(k, *p) as f32 / k as f32)
                .collect(),
            // Gaussian readout-noise model.
            _ => clean.iter().map(|p| p + cfg.sigma * rng.normal()).collect(),
        };
        out.push(MagSample {
            record,
            axes,
            field,
            theta,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MagConfig {
        MagConfig::default()
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn normal_is_roughly_standard() {
        let mut r = Rng::new(7);
        let n = 20_000;
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for _ in 0..n {
            let x = r.normal() as f64;
            sum += x;
            sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean={mean}");
        assert!((var - 1.0).abs() < 0.1, "var={var}");
    }

    #[test]
    fn dataset_is_reproducible() {
        let a = generate_dataset(&cfg(), 20, 123);
        let b = generate_dataset(&cfg(), 20, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_noise_matches_model() {
        let c = MagConfig {
            sigma: 0.0,
            ..cfg()
        };
        let data = generate_dataset(&c, 5, 1);
        for s in &data {
            let clean = MagSample::noiseless_record(s.field, &s.axes, &c);
            for (r, e) in s.record.iter().zip(clean.iter()) {
                assert!((r - e).abs() < 1e-6, "record={r} expected={e}");
            }
        }
    }

    #[test]
    fn noise_mean_is_small() {
        // Averaging many noisy records of a fixed field recovers the clean one.
        let c = MagConfig {
            sigma: 0.05,
            ..cfg()
        };
        let axes = base_axes(c.m);
        let field = [0.3, -0.2];
        let clean = MagSample::noiseless_record(field, &axes, &c);
        let mut rng = Rng::new(99);
        let reps = 5000;
        let mut acc = vec![0.0f32; c.m];
        for _ in 0..reps {
            for (a, cl) in acc.iter_mut().zip(clean.iter()) {
                *a += cl + c.sigma * rng.normal();
            }
        }
        for (a, cl) in acc.iter().zip(clean.iter()) {
            let mean = a / reps as f32;
            assert!((mean - cl).abs() < 0.01, "mean={mean} clean={cl}");
        }
    }

    #[test]
    fn projection_is_rotation_invariant() {
        // Ground-truth symmetry: rotating the field and the axes by the same
        // angle leaves the record unchanged.
        let c = MagConfig {
            sigma: 0.0,
            hard_iron: [0.0, 0.0],
            ..cfg()
        };
        let axes = base_axes(c.m);
        let field = [0.4, 0.15];
        let base = MagSample::noiseless_record(field, &axes, &c);

        let theta = 0.7;
        let rot_axes: Vec<[f32; 2]> = axes.iter().map(|&u| rotate_field(u, theta)).collect();
        let rot_field = rotate_field(field, theta);
        let rotated = MagSample::noiseless_record(rot_field, &rot_axes, &c);

        for (a, b) in base.iter().zip(rotated.iter()) {
            assert!((a - b).abs() < 1e-5, "base={a} rotated={b}");
        }
    }

    #[test]
    fn phase_stays_in_monotonic_range() {
        // Validity of the sin() linearization: |phi| should stay below ~1 rad.
        let c = cfg();
        let data = generate_dataset(&c, 200, 5);
        for s in &data {
            for u in &s.axes {
                let proj = s.field[0] * u[0] + s.field[1] * u[1];
                let phi = c.gamma * c.t_evolve * proj;
                assert!(phi.abs() <= 1.0, "phi={phi}");
            }
        }
    }

    #[test]
    fn binomial_matches_mean_and_variance() {
        let mut rng = Rng::new(13);
        let (n, p) = (128usize, 0.4f32);
        let reps = 20_000;
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for _ in 0..reps {
            let frac = rng.binomial(n, p) as f64 / n as f64;
            sum += frac;
            sq += frac * frac;
        }
        let mean = sum / reps as f64;
        let var = sq / reps as f64 - mean * mean;
        let expected_var = (p as f64 * (1.0 - p as f64)) / n as f64;
        assert!((mean - p as f64).abs() < 0.005, "mean={mean}");
        assert!((var - expected_var).abs() < expected_var * 0.2, "var={var}");
    }

    #[test]
    fn soft_iron_changes_record() {
        let base = MagConfig {
            sigma: 0.0,
            ..cfg()
        };
        let distorted = MagConfig {
            sigma: 0.0,
            soft_iron: [1.3, 0.7],
            ..cfg()
        };
        let axes = base_axes(base.m);
        let field = [0.3, 0.25];
        let a = MagSample::noiseless_record(field, &axes, &base);
        let b = MagSample::noiseless_record(field, &axes, &distorted);
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(
            diff > 1e-3,
            "soft iron should change the record, diff={diff}"
        );
    }

    #[test]
    fn defaults_reproduce_gaussian_dataset() {
        // Guard: the new fields default to the original behavior, so the
        // Gaussian path is unchanged (n_shots = None, soft_iron = [1,1]).
        let c = cfg();
        assert_eq!(c.soft_iron, [1.0, 1.0]);
        assert_eq!(c.n_shots, None);
        let a = generate_dataset(&c, 10, 7);
        let b = generate_dataset(&c, 10, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn shot_noise_dataset_is_reproducible_and_discrete() {
        let c = MagConfig {
            n_shots: Some(64),
            ..cfg()
        };
        let a = generate_dataset(&c, 10, 7);
        let b = generate_dataset(&c, 10, 7);
        assert_eq!(a, b);
        // Every reading must be a multiple of 1/64.
        for s in &a {
            for r in &s.record {
                let scaled = r * 64.0;
                assert!((scaled - scaled.round()).abs() < 1e-4, "r={r}");
            }
        }
    }
}
