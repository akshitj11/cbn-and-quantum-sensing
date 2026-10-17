# Clifford magnetometer research stream — design

## Framing

A second research stream in the same spirit as the n-body one: a clean,
reproducible, CPU-friendly substrate for studying Clifford vs flat encodings,
this time on a 2D qubit-magnetometer field-estimation problem. We measure and
report; there is no objective to make either learned model come out ahead, and
a strong classical estimator is included precisely so the comparison stays
honest.

The physics motivation: a magnetic field is a geometric (vector) object, and a
sensor reports it in its own frame. When the apparatus rotates, the readings
rotate, so field estimation is a rotation-equivariant regression — the exact
structure a Clifford layer is built to carry.

All work lives in the existing Rust workspace. No Python, no GPU. The whole
sweep runs in seconds-to-minutes on a CPU laptop.

## Hardware budget (hard constraints)

- Cl(2,0) only for this stream; Cl(3,0) / 3D is deferred.
- Small measurement sets (default `m = 6` axes), datasets up to ~1k samples in
  the default sweep.
- Single-threaded, no new heavy dependencies; gradients are hand-written and
  the PRNG is a small splitmix64, reusing the n-body patterns.

## Observation model

A spin-1/2 sensor runs a Ramsey sequence along each of `m` unit axes `u_k`,
biased to the steepest point of the fringe (control phase `-pi/2`):

```
phi_k = gamma * T * (B . u_k)                 accumulated Larmor phase
P_k   = 0.5 * (1 + V * sin(phi_k)) + noise    V = exp(-T / T2)
```

`V` is the fringe contrast reduced by T2 decoherence, and `noise` is Gaussian
readout noise with standard deviation `sigma`. The sampled field magnitude is
kept so `|phi_k| <~ 0.8` rad (a test enforces `<= 1`), so `sin` is monotonic and
the record is invertible — the regime real sensors operate in. Optional
hard-iron bias and soft-iron gain distort the field before projection
(`B_eff = diag(soft_iron) * B + hard_iron`); both default to identity. Readout
is Gaussian (`sigma`) unless `n_shots = Some(k)`, in which case each
probability is a binomial estimate from `k` projective shots.

Because a measurement records the projection of `B` onto an axis, the record is
invariant under a simultaneous rotation of the field and the axes. Equivalently:
rotate the apparatus by `R` and the record is unchanged while the correct
estimate rotates by `R`. That is the ground-truth symmetry the models are
measured against.

## Architecture

```
clifford-math        (unchanged) Cl(2,0) multivectors, geometric product, rotors

clifford-nn
  magnetometer.rs    splitmix64 RNG + Gaussian/binomial, MagConfig (soft_iron,
                     n_shots), Ramsey simulator, MagSample, generate_dataset
  mag_models.rs      CliffordMag + FlatMag, ClassicalEstimator (naive/oracle
                     defect correction), CliffordMagRaw + FlatMlpMag
  mag_eval.rs        MagModel trait + estimate_mse, equivariance_error,
                     cramer_rao_bound, sensitivity, train_model
  bin/train_magnetometer.rs      original sweep -> results/magnetometer_sweep.csv
  bin/train_mag_misspec.rs       defect sweep -> results/mag_misspec.csv
  bin/train_mag_efficiency.rs    low-N sweep -> results/mag_efficiency.csv
  bin/train_mag_shots.rs         shot-noise vs CRB -> results/mag_shots.csv
```

## Component details

### Simulator (`magnetometer.rs`)

Dependency-free. `Rng` adds a Box-Muller `normal()` to the n-body splitmix64.
`generate_dataset` draws, per sample, an area-uniform field in the disk of
radius `b_max` and a random sensor orientation, rotates the `m` base axes
(evenly spread over the half-circle) by that orientation, and records the noisy
Ramsey probabilities. Tests cover RNG determinism, approximately-standard
normals, dataset reproducibility, zero-noise exactness, the small noise mean,
projection rotation-invariance (the ground-truth symmetry), and that the phase
stays in the monotonic range.

### Models (`mag_models.rs`)

All three estimators share one task: recover the world-frame field `B` from the
record and the axes. The learned pair mirrors the n-body topology (shared
per-item message layer, permutation-invariant sum, shared update layer) so the
comparison is fair, and both report `num_params`.

The per-measurement feature is the vector observation `v_k = (P_k - 0.5) * u_k`:
the measured signal scaled onto its axis direction. This matters — the
least-squares solution is approximately a linear combination of the `v_k`, so a
single linear message layer can represent it. (An earlier encoding that fed
`(P_k - 0.5)` and `u_k` as separate grades could not, because the needed
product is bilinear in one input channel; both learned models then collapsed to
predicting zero. The vector encoding fixes this and is the more honest geometric
observation.)

- `CliffordMag`: `v_k` as a vector-grade `Multivector2` channel through
  `CliffordLinear` geometric products; a constant `ONE` update channel lets it
  learn an additive bias. Reuses the analytic `backward_cell` / `apply_grad`.
- `FlatMag`: `FlatLinear` baseline on the raw `v_k` components;
  `hidden_for_params` matches the Clifford param count.
- `ClassicalEstimator`: inverts the Ramsey model
  (`proj_k = asin(clamp((2 P_k - 1)/V)) / (gamma T)`) and solves the 2x2 normal
  equations for `B`. Non-learned; the clean-data accuracy reference.

### Evaluation (`mag_eval.rs`)

The `MagModel` trait lets all metrics run against every estimator (the classical
one has a no-op `train_batch`). Metrics:

- field-estimation MSE on a held-out set,
- rotation equivariance error (rotate the apparatus, the estimate should
  rotate),
- data-size sweep across seeds (in the binary).

`train_magnetometer` writes `results/magnetometer_sweep.csv` and prints a
mean +/- std summary. Numbers are reported regardless of which model is ahead.

## Observed first result

With default config (`m=6`, `sigma=0.02`, `T2=5`), 400 epochs, lr 0.1:

- Both learned models reach the classical estimator's accuracy
  (`field_mse ~= 0.0019`, near the readout-noise floor).
- The Clifford net has materially lower equivariance error than the flat
  baseline at every data size (e.g. ~0.0009 vs ~0.0015 at N=1000), and is
  slightly more data-efficient at small N.

This is the predicted shape: on clean data the classical estimator is the
accuracy ceiling, and the Clifford structure shows up as better equivariance,
not better raw accuracy.

## Honest caveats

- Simulation only. No lab data; noise and T2 decoherence are modeled explicitly
  and the operating regime is stated.
- Strong classical baseline. Least squares on a known likelihood is hard to beat
  on clean data; a net has to win on data efficiency, robustness, or on sensor
  defects the classical model does not include (the hard-iron / soft-iron /
  misspecified-T2 experiments below).
- Symmetry can be broken. A real NV-center sensor has fixed crystal axes, so
  full rotation equivariance does not hold; this stream uses an idealized planar
  vector sensor.
- Novelty check. Equivariant networks for magnetometry exist; Clifford networks
  on quantum-sensor data look much less explored. A literature pass is required
  before any novelty claim.

## Deeper experiments

The first sweep pins every estimator to the same readout-noise floor, because
all three invert a pre-computed observable. Three follow-on experiments each
break that tie. New `MagConfig` fields (`soft_iron`, `n_shots`) default so the
original sweep is unchanged.

### Experiment 1 — misspecified classical estimator (`train_mag_misspec`)

A device has a fixed hard-iron bias, soft-iron gain, and a true T2 that may
differ from the nominal value. Two classical fits run on the same data:

- `classical_naive` assumes no defect and the nominal T2.
- `classical_oracle` is given the true defect and T2.

`CliffordMag` and `FlatMag` train on labeled records from the distorted sensor.
Expected shape: at zero defect all four coincide; as defect grows, the naive
fit's error rises while the networks track the oracle. If the flat net matches
the Clifford net, the honest conclusion is that *defect-correction is not where
geometry helps*. Writes `results/mag_misspec.csv`.

Observed: at defect 0 all four sit at `field_mse ~= 0.0019`. At defect 1.0 the
naive fit is at 0.076, the oracle at 0.0024, the flat net at 0.0028, and the
Clifford net at 0.017. Learned models beat the misspecified classical; the
flat net tracks the oracle. Geometry does not help this (world-frame,
anisotropic) calibration task.

### Experiment 2 — form the observable (`train_mag_efficiency`)

Neither model is handed `(P_k - 0.5) u_k`. `CliffordMagRaw` builds it with an
explicit geometric product of `scalar(P_k - 0.5)` and `vector(u_k)` (the VJP is
`gp_vjp`); `FlatMlpMag` sees the raw triple `[P_k-0.5, u_x, u_y]` through a
`tanh` hidden layer and must learn the product. Low-N sweep
`20 / 50 / 100 / 500`. Expected shape: Clifford reaches the noise floor at
smaller N; the gap closes as N grows. Report the crossover. Writes
`results/mag_efficiency.csv`.

Observed: `CliffordMagRaw` is at the noise floor from N=20
(`field_mse ~= 0.002`). `FlatMlpMag` stays near 0.32–0.43 through N=500 — it
never learns the bilinear map in this budget. No crossover in range; this is
the setting where the geometric product is doing real work.

### Experiment 3 — shot noise vs the Cramer-Rao bound (`train_mag_shots`)

Gaussian `sigma` is replaced by binomial readout from `n_shots` projective
measurements per axis. `cramer_rao_bound` evaluates the Fisher information of
the Ramsey + binomial model; we report RMS error, `sqrt(CRB)`,
`rms_over_crb`, and the shot-scaled sensitivity
`gamma T sqrt(n_shots) * rms`. Expected shape: every estimator sits at or
above `sqrt(CRB)` and scales as `1/sqrt(n_shots)`. The classical estimator
should saturate the bound (`rms_over_crb` near 1); the networks should match
it. Caveat: the CRB assumes the observation model is correct, and is a bound
on unbiased estimators; a slightly biased network can sit a few percent
below it at high noise. Writes `results/mag_shots.csv`.

Observed: at 512 shots, classical / Clifford / flat all have
`rms_over_crb ~= 0.99` and sensitivity `~1.01`. Error scales as `1/sqrt(n)`.
At 32 shots the networks are ~7% below the unbiased CRB (shrinkage toward
the mean field); the classical estimator stays just above it (1.06).

## Deferred (not in this stream)

- Cl(3,0) / 3D Bloch magnetometer.
- Magnetic-map (anomaly) navigation.
- Multiple evolution times (fringe wrapping) on top of Experiment 3.

## Reproducing

```bash
cargo test
cargo run -p clifford-nn --release --bin train_magnetometer
cargo run -p clifford-nn --release --bin train_mag_misspec
cargo run -p clifford-nn --release --bin train_mag_efficiency
cargo run -p clifford-nn --release --bin train_mag_shots
```
