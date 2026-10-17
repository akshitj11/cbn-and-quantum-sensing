# Clifford networks on qubit magnetometry

A Clifford layer does not improve quantum sensitivity. On a simulated planar Ramsey magnetometer it matches a least-squares inversion of the known likelihood, saturates the Cramér–Rao bound at 512 shots (`rms / √CRB ≈ 0.99`), and loses to a plain linear net when the job is hard-iron / soft-iron calibration. The one place it is decisive is the product a vector sensor actually computes: `(readout − ½) × axis`. Given the raw pieces and not the product, a 68-parameter Cl(2,0) net is at the noise floor with 20 labeled fields (`field MSE ≈ 0.002`). A parameter-matched tanh MLP is still at `0.32` after 500 fields.

That is a decoder result, not a sensing result. The qubit does not get quieter. The map from a directed measurement to a field vector gets cheaper to learn.

All numbers below come from a CPU-only Rust stack (`clifford-math` + `clifford-nn`), single-threaded, hand-written geometric-product VJPs, splitmix64 data. The full sweeps finish in minutes. Simulation only.

## Why this pairing exists

A magnetic field is a grade-1 object. A vector magnetometer reports it in the sensor frame, so a platform rotation `R` sends `B ↦ R B` and the measurement axes with it. The record of projections is unchanged; the correct world-frame estimate rotates. That is the symmetry a geometric product is built to carry: `R (a b) R̃ = (R a R̃)(R b R̃)`.

A spin-1/2 sensor is already talking the same algebra. Bloch precession about `B` is a rotor in Cl(3,0). This study stays in Cl(2,0) on purpose. The first stream of the same codebase is 2D n-body. A 3D Bloch magnetometer is a later algebra, not a better first experiment.

The honest risk, visible in the first sweep, is that once you hand any model the vector observation `v_k = (P_k − ½) u_k`, every estimator is doing the same linear inversion. Clifford, flat, and least squares then sit on the same floor. You have to take the product away, or break the likelihood the classical fit believes, before the architectures can come apart.

## Sensor model

A spin-1/2 runs a Ramsey sequence along each of `m = 6` unit axes `u_k`, biased to the steepest point of the fringe (control phase `−π/2`):

```
φ_k = γ T (B · u_k)
P_k = ½ (1 + V sin φ_k) + noise
V   = exp(−T / T2)
```

Default protocol: `γ = 1`, `T = 1`, `T2 = 5` so `V ≈ 0.82`, Gaussian readout `σ = 0.02` unless replaced by binomial shot noise. Fields are drawn uniformly in the disk `|B| ≤ 0.8`, which keeps `|φ| ≤ 0.8` rad. `sin` is monotonic there; the record is invertible. That is the regime the classical estimator is allowed to be strong in.

An optional device defect is applied before projection:

```
B_eff = diag(s) B + h
```

with soft-iron gain `s` (default `(1,1)`) and hard-iron bias `h` (default `0`). Defects are a property of the sensor, constant across samples. Fields and orientations vary.

The classical estimator inverts the fringe,

```
proj_k = arcsin(clip((2 P_k − 1) / V)) / (γ T)
```

and solves the 2×2 normal equations for `B_eff`. An oracle then undoes the true `(s, h)`. A naive fit assumes `s = 1`, `h = 0`, and the nominal `T2`.

Learned models share one topology: a shared per-axis message, a sum over the six axes, a shared update that reads a 2D field. `CliffordMag` carries `Multivector2` channels through geometric products. `FlatMag` is a parameter-matched `FlatLinear` net (`68` vs `70` scalars). Training is full-batch analytic SGD, 400 epochs, learning rate `0.1`, three seeds unless noted.

`field MSE` is the mean of `(ΔB_x)² + (ΔB_y)²` on 500 held-out samples. RMS field error is its square root. Equivariance error is the RMS gap between rotating the estimate and estimating on the rotated apparatus.

## Clean inversion is a tie

On the default Gaussian protocol, least squares, Clifford, and flat all report `field MSE ≈ 0.00191` on the held-out set. At 1000 training fields the two nets are `0.001877` (mean over seeds). That is the readout-noise floor of this protocol, not a model ranking.

The slope of the fringe near zero field is `dP/dB ≈ ½ V ≈ 0.41`. `σ = 0.02` is then about `0.05` of field error on one axis. Six axes, least-squares combined, put the two-component MSE near `0.0016` in the linear approximation. The measured `0.00191` is a little worse because `sin` flattens at larger phase and T2 has already cut 18% of the contrast.

What splits the nets is rotation, not accuracy. At N = 1000, equivariance error is `0.00092` (Clifford) vs `0.00152` (flat). At N = 100 the same gap is `0.0044` vs `0.0067`. The classical inversion is at `6 × 10⁻⁸`. Clifford is the closer of the two learned models to the group action. It is not a more accurate field estimate.

If you stop here you will conclude, correctly, that a Clifford net can invert a Ramsey record and that a textbook estimator is enough. The next three runs exist because that conclusion is true and incomplete.

## Experiment 1. A wrong classical model is beatable. Geometry is not why.

A real device has a bias, a gain, and a T2 you do not know exactly. The honest baseline is a lab fit that assumes an ideal sensor.

Defect level `d` sets

```
h = (0.2 d, −0.15 d),   s = (1 + 0.3 d, 1 − 0.2 d),   T2 = 5 (1 − 0.2 d)
```

and the naive fit still believes `h = 0`, `s = 1`, `T2 = 5`.

| d | naive LS | oracle LS | Clifford | flat |
|---|---:|---:|---:|---:|
| 0.0 | 0.00191 | 0.00191 | 0.00188 | 0.00188 |
| 0.5 | 0.0221 | 0.00199 | 0.00611 | 0.00207 |
| 1.0 | 0.0761 | 0.00240 | 0.0171 | 0.00278 |

At `d = 1` the naive fit is about 40× the noise floor. The oracle is essentially back on the floor. The flat net tracks the oracle (`0.00278`). Clifford beats the naive fit (`0.017` vs `0.076`) and loses to the flat net by about 6×.

The defect is a world-frame diagonal scale plus an offset. That is not a rotor. A model that wants to rotate its features is the wrong inductive bias for this calibration. A linear map on `v_k` plus a constant channel can absorb `s` and `h`. The flat net does that. Report the result that way: learned models beat a misspecified likelihood, and the architecture that won is the one without geometric structure.

Equivariance error rises with `d` for every estimator that tries to undo the defect (`0.30` for the oracle at `d = 1`). Soft-iron in world axes breaks the rotation symmetry of the record. The naive fit stays "equivariant" because it is consistently wrong in a rotation-covariant way.

## Experiment 2. Take the product away.

`CliffordMag` and `FlatMag` were handed `v_k = (P_k − ½) u_k`. A single linear layer can finish the job. That is why they tied at N = 100.

`CliffordMagRaw` is given two separate channels, `scalar(P_k − ½)` and `vector(u_k)`, and forms the observable with one geometric product. No weights in that product. The rest of the net is the same 68-parameter message/update pair. `FlatMlpMag` sees `[P_k − ½, u_x, u_y]` through a tanh hidden layer (67 parameters) and has to learn the bilinear map. Five seeds, 600 epochs, training sizes 20 / 50 / 100 / 500.

| N | Clifford raw, field MSE | tanh MLP, field MSE |
|---:|---:|---:|
| 20 | 0.00196 ± 0.00004 | 0.432 ± 0.057 |
| 50 | 0.00195 ± 0.00005 | 0.353 ± 0.022 |
| 100 | 0.00192 ± 0.00003 | 0.332 ± 0.006 |
| 500 | 0.00188 ± 0.00001 | 0.323 ± 0.002 |

`0.32` is the MSE of a model that predicts near-zero field on this disk. The MLP never forms the product in this budget. Clifford is on the floor at the first point in the sweep.

Equivariance follows the same split. At N = 20, Clifford is `0.004–0.010`; the MLP is `0.18–0.40`. At N = 500, Clifford is `0.0002–0.002`; the MLP is still `0.02–0.06`.

This is the experiment that justifies the algebra. The geometric product is the map the sensor physics already uses. A generic nonlinearity of the same width does not find it from a few hundred examples. There is no crossover in the range we ran. Saying "the gap closes as N grows" would be a prediction, not a measurement.

## Experiment 3. Shot noise, and a number a sensing reader will accept.

Gaussian `σ` is a convenience. A projective readout of `n` shots is a binomial count. Fisher information for this Ramsey model, per axis, is

```
dP/dB_proj = ½ V γ T cos φ_k
Var(P̂)     = P_k (1 − P_k) / n
F           = Σ_k (dP/dB)(dP/dB)ᵀ / Var_k
CRB         = tr(F⁻¹)
```

The figure that should travel is `rms / √CRB`, plus the shot-scaled sensitivity `γ T √n · rms`. A ratio near 1 means the estimator used the shots. A ratio of 0.5 would be a claim of super-classical information and a reason to stop and find the bug.

| n | classical rms | Clifford rms | flat rms | √CRB | classical / bound | Clifford / bound |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.192 | 0.169 | 0.169 | 0.182 | 1.056 | 0.929 |
| 128 | 0.0925 | 0.0898 | 0.0899 | 0.0910 | 1.016 | 0.987 |
| 512 | 0.0448 | 0.0449 | 0.0449 | 0.0454 | 0.986 | 0.988 |

Error scales as `1/√n`. Sixteen times more shots (32 → 512) cut RMS by about four (0.192 → 0.045). Sensitivity sits at `1.01–1.09` in these units. At 512 shots all three estimators are on the bound.

At 32 shots the nets sit about 7% under the unbiased CRB. The Cramér–Rao statement is for unbiased estimators. A net trained on MSE shrinks toward the mean of the training disk. That can beat an unbiased bound on squared error without extracting more information than the likelihood contains. The classical inversion stays just above the bound (`1.056`), which is the shape you want from a correctly derived Fisher matrix.

Nobody here improved the sensor. They used the shots.

## What this is allowed to claim

Clifford networks applied to quantum-sensor *data* are a plausible research object. Equivariant networks for magnetometry already exist; this stack is a small Cl(2,0) implementation with a Ramsey simulator, a classical estimator, and three tests that can falsify the usual story. A novelty sentence needs a literature pass that this document does not do.

The claim that survives the numbers:

1. On a known Ramsey likelihood, a Clifford net is an efficient function class for the inverse, not a better estimator than least squares.
2. When the inverse requires multiplying a scalar readout onto a direction, that product is cheap in Cl(2,0) and expensive for a small flat MLP.
3. When the inverse requires undoing a world-frame gain and bias, a flat net is the better function class.
4. Under binomial readout, both nets match the Cramér–Rao scaling. Sensitivity is set by `n`, `V`, and `T`, not by the layer type.

The claim that does not survive: that a Clifford network is an effective upgrade to quantum sensing. Sensing is the extraction of `B` from shots. That extraction is already solved, in this regime, by `arcsin` and a 2×2 solve.

## Failure modes the study does not hide

Simulation only. There is no NV-center, fluxgate, or SQUID dataset in this repo. Noise and T2 are modeled; they are not measured.

The sensor is an ideal planar vector magnetometer. An NV-center has fixed crystal axes. Full rotation equivariance does not hold for that device. You would pass the axes in as data, which this model already does, and you would stop claiming the network is equivariant by construction.

The operating point stays inside one fringe. Multiple evolution times, phase wrapping, and adaptive Ramsey are out of scope. Those are the regimes where a learned inverse sometimes earns its keep against a misspecified or multimodal likelihood. They are also the regimes where a classical estimator has to become a proper maximum-likelihood search, not a 2×2 solve.

Cl(3,0) is not implemented. A 3D Bloch magnetometer on the same easy inverse would reproduce the clean-data tie. It becomes interesting only after Experiments 2 and 3 are restated in 3D.

The n-body stream in the same repo is a reminder that geometry is not an automatic win. On 2D gravity the flat message-passing net currently reports lower one-step and rollout error than the Clifford net. Measure, then decide where the product is the computation.

## How to reproduce

```
cargo test
cargo run -p clifford-nn --release --bin train_magnetometer
cargo run -p clifford-nn --release --bin train_mag_misspec
cargo run -p clifford-nn --release --bin train_mag_efficiency
cargo run -p clifford-nn --release --bin train_mag_shots
```

CSVs land in `results/`. The design note that specifies the protocol is `docs/superpowers/specs/2026-10-17-clifford-magnetometer-stream-design.md`.

If you want a next measurement that could still change the story, it is not "add Cl(3,0) and rerun the easy inverse." It is a wrapped-fringe, multi-`T` protocol, or a real (even public) vector-magnetometer log, with the same three estimators and the same refusal to call a noise-floor tie a sensing win.
