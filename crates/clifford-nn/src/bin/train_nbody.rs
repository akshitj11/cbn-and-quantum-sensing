//! Research harness: Clifford vs matched flat baseline on 2D n-body one-step
//! prediction. Runs a data-size sweep across multiple seeds and reports one-step
//! error, equivariance error, rollout drift, parameter counts, and timing.
//!
//! Results are written as CSV to `results/nbody_sweep.csv`. This reports numbers
//! honestly; it does not optimize for either model to "win".
//!
//! Run: `cargo run -p clifford-nn --release --bin train_nbody`

use clifford_nn::eval::{equivariance_error, one_step_mse, rollout_error, train_model, NBodyModel};
use clifford_nn::models::{CliffordNBody, FlatNBody};
use clifford_nn::nbody::{generate_dataset, random_system, NBodyConfig, Rng, State};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

struct RunResult {
    model: &'static str,
    train_size: usize,
    seed: u64,
    params: usize,
    train_loss: f32,
    test_mse: f32,
    equiv_err: f32,
    rollout_err: f32,
    train_secs: f64,
    infer_us: f64,
}

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len().max(1) as f32
}

fn std_dev(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (xs.len() - 1) as f32;
    var.sqrt()
}

fn infer_time_us<M: NBodyModel>(model: &M, data: &[clifford_nn::nbody::Sample]) -> f64 {
    let reps = 50usize;
    let start = std::time::Instant::now();
    for _ in 0..reps {
        for s in data.iter().take(32) {
            std::hint::black_box(model.forward_state(&s.input));
        }
    }
    let total = start.elapsed().as_secs_f64();
    let calls = reps * data.len().min(32);
    total / calls as f64 * 1e6
}

fn main() {
    let cfg = NBodyConfig {
        n: 4,
        g: 1.0,
        softening: 0.2,
        dt: 0.01,
    };

    // Kept modest so the whole sweep finishes in minutes on a CPU-only laptop.
    let train_sizes = [100usize, 500, 1000];
    let seeds = [0u64, 1, 2];
    let epochs = 150usize;
    let lr = 0.05f32;
    let clifford_hidden = 8usize;
    let thetas = [0.4f32, 1.2, 2.5, 4.0];
    let rollout_steps = 10usize;
    let test_size = 500usize;
    let rollout_count = 50usize;

    // Fixed held-out test and rollout sets (seed chosen far from train seeds).
    let test = generate_dataset(&cfg, test_size, 1, 9_999);
    let mut roll_rng = Rng::new(8_888);
    let rollout_inits: Vec<State> = (0..rollout_count)
        .map(|_| random_system(&cfg, &mut roll_rng))
        .collect();

    let probe = CliffordNBody::new(clifford_hidden);
    let flat_hidden = FlatNBody::hidden_for_params(probe.num_params());
    println!(
        "Clifford params={} | Flat hidden={} params={}",
        probe.num_params(),
        flat_hidden,
        FlatNBody::new(flat_hidden).num_params()
    );

    let mut results: Vec<RunResult> = Vec::new();

    for &train_size in &train_sizes {
        for &seed in &seeds {
            // Each (size, seed) gets its own training set.
            let train = generate_dataset(&cfg, train_size, 1, 1000 + seed);

            // Clifford
            {
                let mut m = CliffordNBody::new(clifford_hidden);
                let (loss, secs) = train_model(&mut m, &train, epochs, lr);
                results.push(RunResult {
                    model: "clifford",
                    train_size,
                    seed,
                    params: m.num_params(),
                    train_loss: loss,
                    test_mse: one_step_mse(&m, &test),
                    equiv_err: equivariance_error(&m, &test, &thetas),
                    rollout_err: rollout_error(&m, &rollout_inits, &cfg, rollout_steps),
                    train_secs: secs / epochs as f64,
                    infer_us: infer_time_us(&m, &test),
                });
            }
            // Flat baseline
            {
                let mut m = FlatNBody::new(flat_hidden);
                let (loss, secs) = train_model(&mut m, &train, epochs, lr);
                results.push(RunResult {
                    model: "flat",
                    train_size,
                    seed,
                    params: m.num_params(),
                    train_loss: loss,
                    test_mse: one_step_mse(&m, &test),
                    equiv_err: equivariance_error(&m, &test, &thetas),
                    rollout_err: rollout_error(&m, &rollout_inits, &cfg, rollout_steps),
                    train_secs: secs / epochs as f64,
                    infer_us: infer_time_us(&m, &test),
                });
            }
            println!("done train_size={train_size} seed={seed}");
        }
    }

    write_csv(&results);
    print_summary(&results, &train_sizes);
}

fn write_csv(results: &[RunResult]) {
    let dir = Path::new("results");
    if let Err(e) = fs::create_dir_all(dir) {
        eprintln!("warning: could not create results/: {e}");
    }
    let mut csv = String::from(
        "model,train_size,seed,params,train_loss,test_mse,equiv_err,rollout_err,train_secs_per_epoch,infer_us\n",
    );
    for r in results {
        let _ = writeln!(
            csv,
            "{},{},{},{},{:.8},{:.8},{:.8},{:.8},{:.6},{:.4}",
            r.model,
            r.train_size,
            r.seed,
            r.params,
            r.train_loss,
            r.test_mse,
            r.equiv_err,
            r.rollout_err,
            r.train_secs,
            r.infer_us,
        );
    }
    let path = dir.join("nbody_sweep.csv");
    match fs::write(&path, csv) {
        Ok(_) => println!("\nwrote {}", path.display()),
        Err(e) => eprintln!("error writing CSV: {e}"),
    }
}

fn print_summary(results: &[RunResult], train_sizes: &[usize]) {
    println!("\n=== Summary (mean +/- std over seeds) ===");
    println!(
        "{:<9} {:>6} {:>22} {:>22} {:>16}",
        "model", "N", "test_mse", "equiv_err", "rollout_err"
    );
    for model in ["clifford", "flat"] {
        for &n in train_sizes {
            let rows: Vec<&RunResult> = results
                .iter()
                .filter(|r| r.model == model && r.train_size == n)
                .collect();
            let mse: Vec<f32> = rows.iter().map(|r| r.test_mse).collect();
            let eq: Vec<f32> = rows.iter().map(|r| r.equiv_err).collect();
            let ro: Vec<f32> = rows.iter().map(|r| r.rollout_err).collect();
            println!(
                "{:<9} {:>6} {:>12.6} +/- {:>5.4} {:>12.6} +/- {:>5.4} {:>8.4} +/- {:>5.4}",
                model,
                n,
                mean(&mse),
                std_dev(&mse),
                mean(&eq),
                std_dev(&eq),
                mean(&ro),
                std_dev(&ro),
            );
        }
    }
}
