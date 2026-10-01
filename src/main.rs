#![allow(unused)]

use rustorch::nn::*;
use rustorch::tensor::*;

fn main() {
    let model = Linear::new(1, 1);
    let loss_fn = MSELoss;

    let inputs = Tensor::new(vec![-1., 0., 1., 2.], vec![4, 1]).unwrap();
    let targets = Tensor::new(vec![-1., 1., 3., 5.], vec![4, 1]).unwrap();

    let dataset = TensorDataset::new(inputs, targets).unwrap();

    let mut optimizer = SGD::new(model.parameters(), 0.001);

    for epoch in 0..100 {
        let loader = DataLoader::new(&dataset, 2, true);

        let mut epoch_loss = 0.;
        let mut batch_count = 0;

        for (input, target) in loader {
            optimizer.zero_grad();

            let prediction = model.forward(&input);
            let loss = loss_fn.forward(&prediction, &target);

            epoch_loss += loss.data()[0];
            batch_count += 1;

            loss.backward();
            optimizer.step();
        }

        if epoch % 10 == 0 {
            println!("epoch: {epoch}, loss: {}", epoch_loss / batch_count as f32)
        };
    }

    let parameters = model.parameters();

    println!("weight: {:?}", parameters[0].tensor().data());
    println!("bias  : {:?}", parameters[1].tensor().data());
}
