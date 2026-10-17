//! Research harness: Clifford vs matched flat baseline vs a classical
//! least-squares estimator on planar qubit-magnetometer field estimation.
//! Runs a data-size sweep across multiple seeds and reports field-estimation
//! error, rotation equivariance error, parameter counts, and timing.
//!
//! Results are written as CSV to `results/magnetometer_sweep.csv`. This reports
//! numbers honestly; it does not optimize for any model to "win". On clean data
//! the classical estimator is the accuracy reference; the interesting question
//! is whether the Clifford net reaches a given error with less data than the
//! flat net and stays closer to the classical estimator under rotated fields.
//!
//! Run: `cargo run -p clifford-nn --release --bin train_magnetometer`

use clifford_nn::mag_eval::{equivariance_error, estimate_mse, train_model, MagModel};
use clifford_nn::mag_models::{ClassicalEstimator, CliffordMag, FlatMag};
use clifford_nn::magnetometer::{generate_dataset, MagConfig, MagSample};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

struct RunResult {
    model: &'static str,
    train_size: usize,
    seed: u64,
    params: usize,
    train_loss: f32,
    field_mse: f32,
    equiv_err: f32,
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

fn infer_time_us<M: MagModel>(model: &M, data: &[MagSample]) -> f64 {
    let reps = 50usize;
    let take = data.len().min(32);
    let start = std::time::Instant::now();
    for _ in 0..reps {
        for s in data.iter().take(take) {
            std::hint::black_box(model.estimate_field(s));
        }
    }
    let total = start.elapsed().as_secs_f64();
    let calls = (reps * take).max(1);
    total / calls as f64 * 1e6
}

fn main() {
    let cfg = MagConfig::default();

    // Kept modest so the whole sweep finishes in minutes on a CPU-only laptop.
    let train_sizes = [100usize, 500, 1000];
    let seeds = [0u64, 1, 2];
    let epochs = 400usize;
    let lr = 0.1f32;
    let clifford_hidden = 8usize;
    let thetas = [0.4f32, 1.2, 2.5, 4.0];
    let test_size = 500usize;

    // Fixed held-out test set (seed chosen far from train seeds).
    let test = generate_dataset(&cfg, test_size, 9_999);

    let probe = CliffordMag::new(clifford_hidden);
    let flat_hidden = FlatMag::hidden_for_params(probe.num_params());
    println!(
        "Clifford params={} | Flat hidden={} params={} | classical params=0",
        probe.num_params(),
        flat_hidden,
        FlatMag::new(flat_hidden).num_params()
    );

    let mut results: Vec<RunResult> = Vec::new();

    // The classical estimator does not train; evaluate it once per test set.
    {
        let k = ClassicalEstimator::new(cfg);
        results.push(RunResult {
            model: "classical",
            train_size: 0,
            seed: 0,
            params: 0,
            train_loss: 0.0,
            field_mse: estimate_mse(&k, &test),
            equiv_err: equivariance_error(&k, &test, &thetas),
            train_secs: 0.0,
            infer_us: infer_time_us(&k, &test),
        });
    }

    for &train_size in &train_sizes {
        for &seed in &seeds {
            let train = generate_dataset(&cfg, train_size, 1000 + seed);

            // Clifford
            {
                let mut m = CliffordMag::new(clifford_hidden);
                let (loss, secs) = train_model(&mut m, &train, epochs, lr);
                results.push(RunResult {
                    model: "clifford",
                    train_size,
                    seed,
                    params: m.num_params(),
                    train_loss: loss,
                    field_mse: estimate_mse(&m, &test),
                    equiv_err: equivariance_error(&m, &test, &thetas),
                    train_secs: secs / epochs as f64,
                    infer_us: infer_time_us(&m, &test),
                });
            }
            // Flat baseline
            {
                let mut m = FlatMag::new(flat_hidden);
                let (loss, secs) = train_model(&mut m, &train, epochs, lr);
                results.push(RunResult {
                    model: "flat",
                    train_size,
                    seed,
                    params: m.num_params(),
                    train_loss: loss,
                    field_mse: estimate_mse(&m, &test),
                    equiv_err: equivariance_error(&m, &test, &thetas),
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
        "model,train_size,seed,params,train_loss,field_mse,equiv_err,train_secs_per_epoch,infer_us\n",
    );
    for r in results {
        let _ = writeln!(
            csv,
            "{},{},{},{},{:.8},{:.8},{:.8},{:.6},{:.4}",
            r.model,
            r.train_size,
            r.seed,
            r.params,
            r.train_loss,
            r.field_mse,
            r.equiv_err,
            r.train_secs,
            r.infer_us,
        );
    }
    let path = dir.join("magnetometer_sweep.csv");
    match fs::write(&path, csv) {
        Ok(_) => println!("\nwrote {}", path.display()),
        Err(e) => eprintln!("error writing CSV: {e}"),
    }
}

fn print_summary(results: &[RunResult], train_sizes: &[usize]) {
    println!("\n=== Summary (mean +/- std over seeds) ===");
    println!(
        "{:<10} {:>6} {:>24} {:>24}",
        "model", "N", "field_mse", "equiv_err"
    );

    // Classical: a single orientation-independent row.
    if let Some(k) = results.iter().find(|r| r.model == "classical") {
        println!(
            "{:<10} {:>6} {:>24.8} {:>24.8}",
            "classical", "-", k.field_mse, k.equiv_err
        );
    }

    for model in ["clifford", "flat"] {
        for &n in train_sizes {
            let rows: Vec<&RunResult> = results
                .iter()
                .filter(|r| r.model == model && r.train_size == n)
                .collect();
            let mse: Vec<f32> = rows.iter().map(|r| r.field_mse).collect();
            let eq: Vec<f32> = rows.iter().map(|r| r.equiv_err).collect();
            println!(
                "{:<10} {:>6} {:>14.8} +/- {:>7.5} {:>14.8} +/- {:>7.5}",
                model,
                n,
                mean(&mse),
                std_dev(&mse),
                mean(&eq),
                std_dev(&eq),
            );
        }
    }
}
