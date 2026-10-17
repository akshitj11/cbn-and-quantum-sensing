//! Experiment 1: can a learned model beat a *misspecified* classical estimator?
//!
//! A real sensor has an unknown hard-iron bias, soft-iron gain, and imperfectly
//! known T2. A lab least-squares fit that assumes an ideal sensor (the "naive"
//! classical estimator) is then biased. A network trained on labeled data should
//! learn the correction; an oracle fit that knows the true defect is the
//! accuracy reference.
//!
//! We sweep defect severity and report field MSE for four estimators:
//! `clifford`, `flat`, `classical_naive`, `classical_oracle`. Results go to
//! `results/mag_misspec.csv`. Reported honestly: if the flat net matches the
//! Clifford net, the conclusion is that defect-correction is not where geometry
//! helps.
//!
//! Run: `cargo run -p clifford-nn --release --bin train_mag_misspec`

use clifford_nn::mag_eval::{equivariance_error, estimate_mse, train_model, MagModel};
use clifford_nn::mag_models::{ClassicalEstimator, CliffordMag, FlatMag};
use clifford_nn::magnetometer::{generate_dataset, MagConfig};

struct RunResult {
    defect: f32,
    model: &'static str,
    field_mse: f32,
    equiv_err: f32,
}

fn main() {
    // Nominal (believed) sensor: no defect, T2 = 5.0.
    let nominal_t2 = 5.0f32;
    let defects = [0.0f32, 0.5, 1.0];
    let seeds = [0u64, 1, 2];
    let epochs = 400usize;
    let lr = 0.1f32;
    let clifford_hidden = 8usize;
    let thetas = [0.4f32, 1.2, 2.5, 4.0];
    let train_size = 1000usize;
    let test_size = 500usize;

    let flat_hidden = FlatMag::hidden_for_params(CliffordMag::new(clifford_hidden).num_params());
    println!("defect sweep: {defects:?} | nominal T2={nominal_t2}");

    let mut results: Vec<RunResult> = Vec::new();

    for &d in &defects {
        // True (physical) sensor for this defect level.
        let true_hard = [0.2 * d, -0.15 * d];
        let true_soft = [1.0 + 0.3 * d, 1.0 - 0.2 * d];
        let true_t2 = nominal_t2 * (1.0 - 0.2 * d);
        let true_cfg = MagConfig {
            t2: true_t2,
            hard_iron: true_hard,
            soft_iron: true_soft,
            ..MagConfig::default()
        };

        let test = generate_dataset(&true_cfg, test_size, 9_999);

        // Classical estimators: naive believes an ideal sensor + nominal T2;
        // oracle knows the true defect and T2.
        let naive_cfg = MagConfig {
            t2: nominal_t2,
            ..MagConfig::default()
        };
        let naive = ClassicalEstimator::new(naive_cfg);
        let oracle = ClassicalEstimator::with_assumptions(true_cfg, true_hard, true_soft);
        push_static(&mut results, d, "classical_naive", &naive, &test, &thetas);
        push_static(&mut results, d, "classical_oracle", &oracle, &test, &thetas);

        // Learned models, averaged over seeds.
        let mut cliff_mse = Vec::new();
        let mut cliff_eq = Vec::new();
        let mut flat_mse = Vec::new();
        let mut flat_eq = Vec::new();
        for &seed in &seeds {
            let train = generate_dataset(&true_cfg, train_size, 1000 + seed);

            let mut c = CliffordMag::new(clifford_hidden);
            train_model(&mut c, &train, epochs, lr);
            cliff_mse.push(estimate_mse(&c, &test));
            cliff_eq.push(equivariance_error(&c, &test, &thetas));

            let mut f = FlatMag::new(flat_hidden);
            train_model(&mut f, &train, epochs, lr);
            flat_mse.push(estimate_mse(&f, &test));
            flat_eq.push(equivariance_error(&f, &test, &thetas));
        }
        results.push(RunResult {
            defect: d,
            model: "clifford",
            field_mse: mean(&cliff_mse),
            equiv_err: mean(&cliff_eq),
        });
        results.push(RunResult {
            defect: d,
            model: "flat",
            field_mse: mean(&flat_mse),
            equiv_err: mean(&flat_eq),
        });
        println!("done defect={d}");
    }

    write_csv(&results);
    print_summary(&results, &defects);
}

fn push_static<M: MagModel>(
    out: &mut Vec<RunResult>,
    defect: f32,
    model: &'static str,
    est: &M,
    test: &[clifford_nn::magnetometer::MagSample],
    thetas: &[f32],
) {
    out.push(RunResult {
        defect,
        model,
        field_mse: estimate_mse(est, test),
        equiv_err: equivariance_error(est, test, thetas),
    });
}

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len().max(1) as f32
}

fn write_csv(results: &[RunResult]) {
    use std::fmt::Write as _;
    let dir = std::path::Path::new("results");
    let _ = std::fs::create_dir_all(dir);
    let mut csv = String::from("defect,model,field_mse,equiv_err\n");
    for r in results {
        let _ = writeln!(
            csv,
            "{:.3},{},{:.8},{:.8}",
            r.defect, r.model, r.field_mse, r.equiv_err
        );
    }
    let path = dir.join("mag_misspec.csv");
    match std::fs::write(&path, csv) {
        Ok(_) => println!("\nwrote {}", path.display()),
        Err(e) => eprintln!("error writing CSV: {e}"),
    }
}

fn print_summary(results: &[RunResult], defects: &[f32]) {
    println!("\n=== field_mse by defect level ===");
    println!(
        "{:>7} {:>16} {:>16} {:>18} {:>18}",
        "defect", "clifford", "flat", "classical_naive", "classical_oracle"
    );
    for &d in defects {
        let get = |m: &str| {
            results
                .iter()
                .find(|r| r.model == m && (r.defect - d).abs() < 1e-6)
                .map(|r| r.field_mse)
                .unwrap_or(f32::NAN)
        };
        println!(
            "{:>7.3} {:>16.8} {:>16.8} {:>18.8} {:>18.8}",
            d,
            get("clifford"),
            get("flat"),
            get("classical_naive"),
            get("classical_oracle"),
        );
    }
}
