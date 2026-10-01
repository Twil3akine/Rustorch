#![allow(unused)]

use rustorch::nn::*;
use rustorch::tensor::*;

fn main() {
    let model = Linear::new(1, 1);
    let loss_fn = MSELoss;

    let batches = vec![
        (
            Tensor::new(vec![-1., 0.], vec![2, 1]).unwrap(),
            Tensor::new(vec![-1., 1.], vec![2, 1]).unwrap(),
        ),
        (
            Tensor::new(vec![1., 2.], vec![2, 1]).unwrap(),
            Tensor::new(vec![3., 5.], vec![2, 1]).unwrap(),
        ),
    ];

    let mut optimizer = SGD::new(model.parameters(), 0.05);

    for epoch in 0..100 {
        let mut epoch_loss = 0.;

        for (input, target) in &batches {
            optimizer.zero_grad();

            let prediction = model.forward(input);
            let loss = loss_fn.forward(&prediction, target);

            epoch_loss += loss.data()[0];

            loss.backward();
            optimizer.step();
        }

        if epoch % 10 == 0 {
            println!(
                "epoch: {epoch}, loss: {}",
                epoch_loss / batches.len() as f32
            )
        };
    }

    let parameters = model.parameters();

    println!("weight: {:?}", parameters[0].tensor().data());
    println!("bias  : {:?}", parameters[1].tensor().data());
}
