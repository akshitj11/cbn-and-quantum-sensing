//! Two-node weight averaging over serialized Clifford model state.

use clifford_nn::CliffordLinear;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StateSnapshot {
    pub round: u64,
    pub weight_coeffs: Vec<f32>,
}

impl StateSnapshot {
    pub fn from_layer(layer: &CliffordLinear, round: u64) -> Self {
        Self {
            round,
            weight_coeffs: layer.weight_coeffs(),
        }
    }

    pub fn apply_to(&self, layer: &mut CliffordLinear) {
        layer.set_weight_coeffs(&self.weight_coeffs);
    }

    pub fn encode(&self) -> Vec<u8> {
        postcard::to_allocvec(self).expect("serialize StateSnapshot")
    }

    pub fn decode(bytes: &[u8]) -> Self {
        postcard::from_bytes(bytes).expect("deserialize StateSnapshot")
    }
}

/// Elementwise average of two snapshots (same shape required).
pub fn average_snapshots(a: &StateSnapshot, b: &StateSnapshot) -> StateSnapshot {
    assert_eq!(a.weight_coeffs.len(), b.weight_coeffs.len());
    let weight_coeffs = a
        .weight_coeffs
        .iter()
        .zip(b.weight_coeffs.iter())
        .map(|(x, y)| (*x + *y) * 0.5)
        .collect();
    StateSnapshot {
        round: a.round.max(b.round) + 1,
        weight_coeffs,
    }
}

/// Elementwise mean of many snapshots (all same shape). Panics if empty.
pub fn average_many(snaps: &[StateSnapshot]) -> StateSnapshot {
    assert!(
        !snaps.is_empty(),
        "average_many needs at least one snapshot"
    );
    let len = snaps[0].weight_coeffs.len();
    let mut acc = vec![0.0f32; len];
    for s in snaps {
        assert_eq!(s.weight_coeffs.len(), len, "snapshot shape mismatch");
        for (a, w) in acc.iter_mut().zip(s.weight_coeffs.iter()) {
            *a += *w;
        }
    }
    let inv = 1.0 / snaps.len() as f32;
    for a in acc.iter_mut() {
        *a *= inv;
    }
    StateSnapshot {
        round: snaps.iter().map(|s| s.round).max().unwrap() + 1,
        weight_coeffs: acc,
    }
}

/// In-process federated round: each node mutates coeffs locally, then mean-sync.
pub fn federated_round(
    local_a: &mut CliffordLinear,
    local_b: &mut CliffordLinear,
    round: u64,
    local_train_fn: impl Fn(&mut CliffordLinear),
) {
    local_train_fn(local_a);
    local_train_fn(local_b);
    let snap_a = StateSnapshot::from_layer(local_a, round);
    let snap_b = StateSnapshot::from_layer(local_b, round);
    let merged = average_snapshots(&snap_a, &snap_b);
    merged.apply_to(local_a);
    merged.apply_to(local_b);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn snapshot_roundtrip() {
        let layer = CliffordLinear::new(1, 1);
        let snap = StateSnapshot::from_layer(&layer, 0);
        let bytes = snap.encode();
        let decoded = StateSnapshot::decode(&bytes);
        assert_eq!(snap, decoded);
    }

    #[test]
    fn average_many_matches_pairwise() {
        let mut a = CliffordLinear::new(1, 2);
        let mut b = CliffordLinear::new(1, 2);
        let mut cb = b.weight_coeffs();
        for (i, v) in cb.iter_mut().enumerate() {
            *v += i as f32 * 0.1;
        }
        b.set_weight_coeffs(&cb);
        let snaps = [
            StateSnapshot::from_layer(&a, 0),
            StateSnapshot::from_layer(&b, 0),
        ];
        let merged = average_many(&snaps);
        let pair = average_snapshots(&snaps[0], &snaps[1]);
        for i in 0..merged.weight_coeffs.len() {
            assert_relative_eq!(
                merged.weight_coeffs[i],
                pair.weight_coeffs[i],
                epsilon = 1e-6
            );
        }
        // silence unused mut warnings
        a.set_weight_coeffs(&a.weight_coeffs());
    }

    #[test]
    fn federated_sync_equalizes_weights() {
        let mut a = CliffordLinear::new(1, 1);
        let mut b = CliffordLinear::new(1, 1);
        let ca = a.weight_coeffs();
        let mut cb = b.weight_coeffs();
        cb[0] += 1.0;
        b.set_weight_coeffs(&cb);

        let merged = average_snapshots(
            &StateSnapshot::from_layer(&a, 0),
            &StateSnapshot::from_layer(&b, 0),
        );
        merged.apply_to(&mut a);
        merged.apply_to(&mut b);

        for i in 0..ca.len() {
            assert_relative_eq!(a.weight_coeffs()[i], b.weight_coeffs()[i], epsilon = 1e-6);
        }
    }
}
