# clifford-based-network

Learn geometric algebra by building a tiny **Clifford neural** stack in Rust, then sync weights across two logical nodes.

## Clifford vs usual networks

See [docs/clifford-vs-usual.md](docs/clifford-vs-usual.md) for a short comparison. In practice: layers use the **geometric product** on **multivectors** (not plain matmul on flattened vectors), which can respect **rotation/reflection structure** in the data.

## Workspace

| Crate | Role |
|-------|------|
| `clifford-math` | Cl(2,0) multivectors, geometric product, rotors, GP VJP (`gp_vjp`) |
| `clifford-nn` | Flow data, `CliffordLinear`/`FlatLinear` with analytic gradients, 2D n-body (`nbody`), message-passing models (`models`), evaluation harness (`eval`); planar qubit magnetometer (`magnetometer`, `mag_models`, `mag_eval`) |
| `clifford-net` | `StateSnapshot` encode/decode + N-node weight averaging + federated n-body study |

## n-body research stream

A reproducible, CPU-only comparison of a Clifford message-passing network against
a parameter-matched flat baseline on 2D gravitational n-body dynamics. Metrics:
one-step error, rotation equivariance, rollout drift, params, and timing, across
a data-size x seed sweep. See
[docs/superpowers/specs/2026-10-15-clifford-nbody-research-stream-design.md](docs/superpowers/specs/2026-10-15-clifford-nbody-research-stream-design.md).

## Magnetometer research stream

A second CPU-only stream on planar qubit-magnetometer field estimation: a
Clifford message-passing network, a parameter-matched flat baseline, and a
classical least-squares estimator all recover a 2D field `B` from noisy Ramsey
measurements. Metrics: field-estimation MSE, rotation equivariance, params, and
timing, across a data-size x seed sweep. On clean data the classical estimator
is the accuracy reference; the Clifford structure shows up as lower equivariance
error. See
[docs/superpowers/specs/2026-10-17-clifford-magnetometer-stream-design.md](docs/superpowers/specs/2026-10-17-clifford-magnetometer-stream-design.md).

Honest caveats: simulation only (noise + T2 decoherence modeled), strong
classical baseline, idealized planar vector sensor (NV-center crystal axes would
break full equivariance), and a literature pass is required before any novelty
claim.

## Quick start

```bash
cargo test
cargo run -p clifford-nn --bin train_flow
cargo run -p clifford-nn --release --bin train_nbody           # writes results/nbody_sweep.csv
cargo run -p clifford-net --release --bin federated_nbody      # federated vs single-node
cargo run -p clifford-nn --release --bin train_magnetometer    # writes results/magnetometer_sweep.csv
```

## Docs

- [Glossary](docs/00-glossary.md)
- [Clifford vs usual](docs/clifford-vs-usual.md)
