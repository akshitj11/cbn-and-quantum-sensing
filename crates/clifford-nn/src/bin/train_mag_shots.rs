//! Experiment 3: realistic projective shot noise, reported against the
//! Cramer-Rao bound.
//!
//! The ad-hoc Gaussian `sigma` is replaced by binomial readout from `n_shots`
//! projective measurements per axis - the actual quantum readout statistic. We
//! report each estimator's RMS field error, the mean Cramer-Rao bound (RMS),
//! and the ratio `rms / sqrt(crb)`. A ratio near 1 means the estimator
//! saturates the bound; it is the authentic efficiency statement, not a
//! unitless MSE.
//!
//! Run: `cargo run -p clifford-nn --release --bin train_mag_shots`

use clifford_nn::mag_eval::{estimate_mse, mean_crb, sensitivity, train_model};
use clifford_nn::mag_models::{ClassicalEstimator, CliffordMag, FlatMag};
use clifford_nn::magnetometer::{generate_dataset, MagConfig};

struct RunResult {
    n_shots: usize,
    model: &'static str,
    field_mse: f32,
    rms_err: f32,
    crb_rms: f32,
    rms_over_crb: f32,
    sensitivity: f32,
}

fn main() {
    let shot_counts = [32usize, 128, 512];
    let seeds = [0u64, 1, 2];
    let epochs = 400usize;
    let lr = 0.1f32;
    let clifford_hidden = 8usize;
    let train_size = 1000usize;
    let test_size = 500usize;

    let flat_hidden = FlatMag::hidden_for_params(CliffordMag::new(clifford_hidden).num_params());
    println!("shot sweep: {shot_counts:?}");

    let mut results: Vec<RunResult> = Vec::new();

    for &n_shots in &shot_counts {
        let cfg = MagConfig {
            n_shots: Some(n_shots),
            ..MagConfig::default()
        };
        let test = generate_dataset(&cfg, test_size, 9_999);
        let crb_rms = mean_crb(&cfg, &test, n_shots).sqrt();

        // Classical estimator (no training).
        {
            let est = ClassicalEstimator::new(cfg);
            let mse = estimate_mse(&est, &test);
            results.push(make(n_shots, "classical", mse, crb_rms, &cfg));
        }

        // Learned models, averaged over seeds.
        let mut cliff = Vec::new();
        let mut flat = Vec::new();
        for &seed in &seeds {
            let train = generate_dataset(&cfg, train_size, 1000 + seed);
            let mut c = CliffordMag::new(clifford_hidden);
            train_model(&mut c, &train, epochs, lr);
            cliff.push(estimate_mse(&c, &test));

            let mut f = FlatMag::new(flat_hidden);
            train_model(&mut f, &train, epochs, lr);
            flat.push(estimate_mse(&f, &test));
        }
        results.push(make(n_shots, "clifford", mean(&cliff), crb_rms, &cfg));
        results.push(make(n_shots, "flat", mean(&flat), crb_rms, &cfg));
        println!("done n_shots={n_shots}");
    }

    write_csv(&results);
    print_summary(&results);
}

fn make(n_shots: usize, model: &'static str, mse: f32, crb_rms: f32, cfg: &MagConfig) -> RunResult {
    let rms = mse.sqrt();
    RunResult {
        n_shots,
        model,
        field_mse: mse,
        rms_err: rms,
        crb_rms,
        rms_over_crb: rms / crb_rms.max(1e-12),
        sensitivity: sensitivity(cfg, n_shots, rms),
    }
}

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len().max(1) as f32
}

fn write_csv(results: &[RunResult]) {
    use std::fmt::Write as _;
    let dir = std::path::Path::new("results");
    let _ = std::fs::create_dir_all(dir);
    let mut csv =
        String::from("n_shots,model,field_mse,rms_err,crb_rms,rms_over_crb,sensitivity\n");
    for r in results {
        let _ = writeln!(
            csv,
            "{},{},{:.8},{:.8},{:.8},{:.6},{:.6}",
            r.n_shots, r.model, r.field_mse, r.rms_err, r.crb_rms, r.rms_over_crb, r.sensitivity
        );
    }
    let path = dir.join("mag_shots.csv");
    match std::fs::write(&path, csv) {
        Ok(_) => println!("\nwrote {}", path.display()),
        Err(e) => eprintln!("error writing CSV: {e}"),
    }
}

fn print_summary(results: &[RunResult]) {
    println!("\n=== RMS field error vs Cramer-Rao bound ===");
    println!(
        "{:>8} {:>10} {:>12} {:>12} {:>14} {:>12}",
        "n_shots", "model", "rms_err", "crb_rms", "rms_over_crb", "sensitivity"
    );
    for r in results {
        println!(
            "{:>8} {:>10} {:>12.6} {:>12.6} {:>14.3} {:>12.4}",
            r.n_shots, r.model, r.rms_err, r.crb_rms, r.rms_over_crb, r.sensitivity
        );
    }
}
