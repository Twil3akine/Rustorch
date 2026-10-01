#![allow(unused)]

use rustorch::nn::*;
use rustorch::tensor::*;

fn main() {
    let model = Linear::new(1, 1);
    let loss_fn = MSELoss;

    let input = Tensor::new(vec![1.], vec![1, 1]).unwrap();

    let target = Tensor::new(vec![2.], vec![1, 1]).unwrap();

    let mut optimizer = SGD::new(model.parameters(), 0.1);

    for epoch in 0..20 {
        optimizer.zero_grad();

        let prediction = model.forward(&input);
        let loss = loss_fn.forward(&prediction, &target);

        println!("epoch: {}, loss: {}", epoch + 1, loss.data()[0]);

        loss.backward();
        optimizer.step();
    }
}
