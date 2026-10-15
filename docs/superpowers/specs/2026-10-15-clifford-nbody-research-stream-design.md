# Clifford n-body research stream — design

## Framing

This is a research stream, not a benchmark to win. The goal is a clean,
reproducible, CPU-friendly substrate where Clifford vs flat encodings can be
studied honestly over time on a 2D gravitational n-body problem. We measure and
report; there is no objective to make either model come out ahead.

All work lives in the existing Rust workspace. No Python, no GPU. The full sweep
runs in minutes on an Intel i5 / 16 GB laptop.

## Hardware budget (hard constraints)

- 2D only (Cl(2,0)) for the whole first stream. Cl(3,0) is deferred.
- Small N (3-5 particles), short rollouts (tens of steps), datasets up to ~1k
  trajectories in the default sweep.
- Single-threaded. No new heavy dependencies; gradients are hand-written and
  the PRNG is a small splitmix64, so iteration stays fast and legible.

## Architecture

```
clifford-math   Cl(2,0) multivectors, geometric product, rotors
  + gp_vjp               analytic vector-Jacobian product for the GP

clifford-nn
  + CliffordLinear::backward_cell / apply_grad / zero_grad / accumulate_grad
  + FlatLinear::backward / apply_grad
  + mse_vector_grad, train_clifford_linear_analytic
  nbody.rs      splitmix64 RNG, Particle/State, leapfrog, dataset builder,
                total_energy, rotate_state
  models.rs     CliffordNBody and FlatNBody message-passing models
  eval.rs       NBodyModel trait + one_step_mse, equivariance_error,
                rollout_error, train_model
  bin/train_nbody.rs   data-size x seed sweep -> results/nbody_sweep.csv

clifford-net
  + average_many         N-node weight averaging
  bin/federated_nbody.rs  federated vs single-node Clifford study
```

## Component details

### Analytic gradients (Milestone 1)

The geometric product is bilinear, so `gp_vjp(a, b, grad)` returns `(dL/da,
dL/db)` as fixed linear maps of the opposite operand and the upstream gradient.
`CliffordLinear::backward_cell` chains this across channels and also returns
input gradients, enabling multi-layer backprop. `FlatLinear::backward` is the
standard dense-layer VJP. Both are verified against symmetric finite differences
in tests; the original finite-difference trainer is retained as an oracle.

### n-body data (Milestone 2)

Softened gravity (Plummer softening `r^2 + eps^2`) integrated with
velocity-Verlet (leapfrog). The dataset builder produces deterministic
`(state_t, state_{t+k})` pairs from a seed. Tests cover RNG determinism,
energy conservation, and the fact that rotation commutes with the dynamics
(the ground-truth symmetry we later measure models against).

### Models (Milestone 3)

Both models share one topology so the comparison is fair:

1. shared per-pair message layer over relative features `(mass_j, r_ij, v_ij)`,
2. permutation-invariant sum over neighbours,
3. shared update layer mapping (aggregated message ++ self state) to a
   per-particle delta `(dpos, dvel)`.

`CliffordNBody` carries multivector channels through `CliffordLinear` geometric
products. `FlatNBody` carries plain real vectors through `FlatLinear`.
`FlatNBody::hidden_for_params` picks the flat hidden width whose parameter count
is closest to the Clifford model's, and both report `num_params()` so the match
is explicit.

### Evaluation (Milestone 4)

The `NBodyModel` trait lets all metrics run against either model:

- one-step delta MSE on a held-out set,
- rotation equivariance error (RMS gap between rotating the prediction and
  predicting on the rotated input),
- autoregressive rollout drift vs the true leapfrog trajectory,
- data-size sweep (100 / 500 / 1000) across 3 seeds,
- parameter counts, per-epoch wall-clock, and inference time.

`train_nbody` writes everything to `results/nbody_sweep.csv` and prints a
mean +/- std summary. Numbers are reported regardless of which model is ahead.

### Federated study (Milestone 5)

`average_many` in `clifford-net` averages weights across N nodes.
`federated_nbody` splits one dataset into shards, trains one model per shard with
periodic weight averaging, and compares held-out MSE to a single-node model
trained on the union.

## Deferred (not in this stream)

- Cl(3,0) / 3D n-body.
- Any PDE / Navier-Stokes work.
- A GNN-with-explicit-edges baseline (the invariant-readout MLP is the honest
  first baseline; a GNN can be a later milestone).

## Reproducing

```bash
cargo test
cargo run -p clifford-nn --release --bin train_nbody
cargo run -p clifford-net --release --bin federated_nbody
```
