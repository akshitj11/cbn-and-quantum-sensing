//! Train CliffordLinear on one-step rotation flow prediction.

use clifford_nn::{
    field_as_single_channel_batch, mse_vector, rotation_flow_field, rotation_flow_step,
    train_clifford_linear_fd, CliffordLinear,
};

fn main() {
    let grid = 5;
    let omega = 1.0;
    let dt = 0.1;
    let field = rotation_flow_field(grid, omega);
    let target = rotation_flow_step(&field, dt, omega);
    let batch_in = field_as_single_channel_batch(&field);

    let mut layer = CliffordLinear::new(1, 1);
    let mut loss = mse_vector(
        &batch_in
            .iter()
            .map(|c| layer.forward_cell(c)[0])
            .collect::<Vec<_>>(),
        &target,
    );
    println!("initial loss: {loss:.6}");

    for step in 0..20 {
        loss = train_clifford_linear_fd(&mut layer, &batch_in, &target, 0.08, 1e-3);
        if step % 5 == 0 {
            println!("step {step} loss: {loss:.6}");
        }
    }
    println!("final loss: {loss:.6}");
}
