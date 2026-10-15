//! Repo-native novelty: does federated averaging of Clifford n-body models
//! across data shards approach single-node training?
//!
//! We split one dataset into `S` shards. The single-node model trains on the
//! union; the federated nodes each train locally on their shard for a few
//! epochs per round, then synchronize by averaging layer weights (reusing
//! `clifford-net`'s `average_many`). We report held-out one-step MSE for both.
//!
//! Run: `cargo run -p clifford-net --release --bin federated_nbody`

use clifford_net::{average_many, StateSnapshot};
use clifford_nn::eval::{one_step_mse, train_model};
use clifford_nn::models::CliffordNBody;
use clifford_nn::nbody::{generate_dataset, NBodyConfig, Sample};

/// Average the weights of several models in place (both layers) via
/// `clifford-net` snapshots, then broadcast the merged weights back to all.
fn federated_sync(models: &mut [CliffordNBody], round: u64) {
    let msg_snaps: Vec<StateSnapshot> = models
        .iter()
        .map(|m| StateSnapshot::from_layer(&m.msg, round))
        .collect();
    let upd_snaps: Vec<StateSnapshot> = models
        .iter()
        .map(|m| StateSnapshot::from_layer(&m.upd, round))
        .collect();
    let msg_merged = average_many(&msg_snaps);
    let upd_merged = average_many(&upd_snaps);
    for m in models.iter_mut() {
        msg_merged.apply_to(&mut m.msg);
        upd_merged.apply_to(&mut m.upd);
    }
}

fn shards(data: &[Sample], s: usize) -> Vec<Vec<Sample>> {
    let mut out: Vec<Vec<Sample>> = vec![Vec::new(); s];
    for (i, sample) in data.iter().enumerate() {
        out[i % s].push(sample.clone());
    }
    out
}

fn main() {
    let cfg = NBodyConfig {
        n: 4,
        g: 1.0,
        softening: 0.2,
        dt: 0.01,
    };
    let hidden = 8usize;
    let lr = 0.05f32;
    let num_shards = 4usize;
    let total_epochs = 150usize;
    let local_epochs = 5usize; // per federated round
    let rounds = total_epochs / local_epochs;

    let train = generate_dataset(&cfg, 800, 1, 2024);
    let test = generate_dataset(&cfg, 400, 1, 9_999);

    // Single-node baseline on the full dataset.
    let mut single = CliffordNBody::new(hidden);
    train_model(&mut single, &train, total_epochs, lr);
    let single_mse = one_step_mse(&single, &test);

    // Federated: one model per shard, sync every `local_epochs`.
    let parts = shards(&train, num_shards);
    let mut nodes: Vec<CliffordNBody> = (0..num_shards)
        .map(|_| CliffordNBody::new(hidden))
        .collect();
    // Start all nodes from the same initial weights.
    let init = StateSnapshot::from_layer(&nodes[0].msg, 0);
    let init_upd = StateSnapshot::from_layer(&nodes[0].upd, 0);
    for m in nodes.iter_mut() {
        init.apply_to(&mut m.msg);
        init_upd.apply_to(&mut m.upd);
    }

    for round in 0..rounds {
        for (node, shard) in nodes.iter_mut().zip(parts.iter()) {
            if !shard.is_empty() {
                train_model(node, shard, local_epochs, lr);
            }
        }
        federated_sync(&mut nodes, round as u64);
    }
    // After syncing, all nodes share weights; evaluate node 0.
    let fed_mse = one_step_mse(&nodes[0], &test);

    println!("=== Federated vs single-node Clifford n-body ===");
    println!("shards={num_shards} rounds={rounds} local_epochs={local_epochs} total_epochs={total_epochs}");
    println!("single-node test MSE : {single_mse:.8}");
    println!("federated  test MSE : {fed_mse:.8}");
    let gap = (fed_mse - single_mse) / single_mse * 100.0;
    println!("federated is {gap:+.2}% relative to single-node (positive = worse)");
}
