# clifford-based-network

Learn geometric algebra by building a tiny **Clifford neural** stack in Rust, then sync weights across two logical nodes.

## Clifford vs usual networks

See [docs/clifford-vs-usual.md](docs/clifford-vs-usual.md) for a short comparison. In practice: layers use the **geometric product** on **multivectors** (not plain matmul on flattened vectors), which can respect **rotation/reflection structure** in the data.

## Workspace

| Crate | Role |
|-------|------|
| `clifford-math` | Cl(2,0) multivectors, geometric product, rotors |
| `clifford-nn` | Synthetic flow data, `CliffordLinear`, training |
| `clifford-net` | `StateSnapshot` encode/decode + weight averaging |

## Quick start

```bash
cargo test
cargo run -p clifford-nn --bin train_flow
```

## Docs

- [Glossary](docs/00-glossary.md)
- [Clifford vs usual](docs/clifford-vs-usual.md)
