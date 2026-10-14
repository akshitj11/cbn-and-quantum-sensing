# Clifford network vs a usual network

## What stays the same

- You still have **parameters**, a **forward pass**, a **loss**, and **optimization**.
- You still stack **layers** and train on data.
- A “distributed network” still means **nodes**, **messages**, and **synchronized state** — that part is independent of Clifford algebra.

## What changes (the important part)

| Usual network | Clifford-based network |
|---------------|------------------------|
| Features are a flat vector in ℝⁿ | Features are **multivectors** (typed grades: scalar, vector, bivector, …) |
| Layer = matrix multiply + bias | Layer = **geometric product** with learnable multivector weights, then grade-aware combine |
| Dot product mixes components linearly | GP is **non-commutative**; composition matches rotation/reflection structure |
| Rotation of input is data augmentation only | With the right layer, **rotate input ≈ rotate output** (equivariance) |
| One number per neuron | One **multivector** per channel; each grade has meaning |

## Intuition in one sentence

A usual net treats `(vx, vy)` as two unrelated floats. A Clifford net treats them as a **vector in an algebra** whose products encode geometry (e.g. area as bivector), so the **same operation** can represent physically consistent transforms.

## In this repo

- **Math:** `clifford-math` — Cl(2,0), rotors, sandwich.
- **NN:** `clifford-nn` — encode flow fields, `CliffordLinear`, tiny training loop.
- **Net:** `clifford-net` — serialize weights and average across two nodes (federated-style step).

Read [Clifford Neural Layers (Brandstetter et al.)](https://arxiv.org/abs/2209.04934) when you want the full PDE/CNN story; this sprint is the minimal slice.
