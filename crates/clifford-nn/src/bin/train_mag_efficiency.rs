//! Experiment 2: data efficiency when the model must *form* the observable.
//!
//! Neither model is handed `(P_k - 0.5) * u_k`. The Clifford model builds it
//! with a geometric product (a structural inductive bias); the flat MLP must
//! learn the bilinear map through a tanh nonlinearity. We sweep small training
//! sizes and look for the crossover N below which the geometric prior wins.
//!
//! Results go to `results/mag_efficiency.csv`. The honest claim: the Clifford
//! model reaches the noise floor at smaller N; the gap closes as N grows.
//!
//! Run: `cargo run -p clifford-nn --release --bin train_mag_efficiency`

use clifford_nn::mag_eval::{equivariance_error, estimate_mse, train_model};
use clifford_nn::mag_models::{CliffordMagRaw, FlatMlpMag};
use clifford_nn::magnetometer::{generate_dataset, MagConfig};

struct Row {
    train_size: usize,
    clifford_mse: f32,
    flat_mse: f32,
}

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len().max(1) as f32
}

fn std_dev(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (xs.len() - 1) as f32).sqrt()
}

fn main() {
    let cfg = MagConfig::default();
    let train_sizes = [20usize, 50, 100, 500];
    let seeds = [0u64, 1, 2, 3, 4];
    let epochs = 600usize;
    let lr = 0.1f32;
    let clifford_hidden = 8usize;

    let flat_hidden =
        FlatMlpMag::hidden_for_params(CliffordMagRaw::new(clifford_hidden).num_params());
    println!(
        "clifford_raw params={} | flat_mlp hidden={} params={}",
        CliffordMagRaw::new(clifford_hidden).num_params(),
        flat_hidden,
        FlatMlpMag::new(flat_hidden).num_params()
    );

    let test = generate_dataset(&cfg, 500, 9_999);
    let thetas = [0.4f32, 1.2, 2.5, 4.0];

    let mut csv = String::from("model,train_size,seed,field_mse,equiv_err\n");
    let mut rows: Vec<Row> = Vec::new();

    for &n in &train_sizes {
        let mut cliff = Vec::new();
        let mut flat = Vec::new();
        for &seed in &seeds {
            let train = generate_dataset(&cfg, n, 1000 + seed);

            let mut c = CliffordMagRaw::new(clifford_hidden);
            train_model(&mut c, &train, epochs, lr);
            let cm = estimate_mse(&c, &test);
            let ce = equivariance_error(&c, &test, &thetas);
            cliff.push(cm);

            let mut f = FlatMlpMag::new(flat_hidden);
            train_model(&mut f, &train, epochs, lr);
            let fm = estimate_mse(&f, &test);
            let fe = equivariance_error(&f, &test, &thetas);
            flat.push(fm);

            use std::fmt::Write as _;
            let _ = writeln!(csv, "clifford_raw,{n},{seed},{cm:.8},{ce:.8}");
            let _ = writeln!(csv, "flat_mlp,{n},{seed},{fm:.8},{fe:.8}");
        }
        rows.push(Row {
            train_size: n,
            clifford_mse: mean(&cliff),
            flat_mse: mean(&flat),
        });
        println!(
            "N={n:>4}  clifford={:.6} +/- {:.5}  flat_mlp={:.6} +/- {:.5}",
            mean(&cliff),
            std_dev(&cliff),
            mean(&flat),
            std_dev(&flat)
        );
    }

    let dir = std::path::Path::new("results");
    let _ = std::fs::create_dir_all(dir);
    let path = dir.join("mag_efficiency.csv");
    match std::fs::write(&path, csv) {
        Ok(_) => println!("\nwrote {}", path.display()),
        Err(e) => eprintln!("error writing CSV: {e}"),
    }

    // Report the crossover: largest N at which Clifford still beats the MLP.
    let last_n = rows.last().map(|r| r.train_size);
    let crossover = rows
        .iter()
        .filter(|r| r.clifford_mse < r.flat_mse)
        .map(|r| r.train_size)
        .max();
    match (crossover, last_n) {
        (Some(n), Some(last)) if n == last => {
            println!(
                "Clifford beats the flat MLP at every N in the sweep (through N={n}); no crossover in range."
            );
        }
        (Some(n), _) => {
            println!("Clifford beats the flat MLP up to N={n} (gap closes above that).");
        }
        (None, _) => {
            println!("No crossover: the flat MLP was at least as good at every N.");
        }
    }
}
