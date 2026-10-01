#![allow(unused)]

use rustorch::data::*;
use rustorch::nn::*;
use rustorch::optim::*;
use rustorch::tensor::*;

fn main() {
    let images = load_images("data/mnist/train-images-idx3-ubyte");
    let labels = load_labels("data/mnist/train-labels-idx1-ubyte");

    println!("images shape: {:?}", images.shape());
    println!("labels shape: {}", labels.len());
    println!("first labels: {:?}", &labels[..10]);

    let dataset = ClassificationDataset::new(images, labels).unwrap();

    println!("dataset len: {}", dataset.len());

    let mut loader = DataLoader::new(&dataset, 32, true);

    let (inputs, targets) = loader.next().unwrap();

    println!("batch input shape: {:?}", inputs.shape());
    println!("batch targets len: {}", targets.len());
    println!("batch target: {:?}", targets);

    let model = Sequential::new(vec![
        Box::new(Linear::new(784, 128)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10)),
    ]);

    let output = model.forward(&inputs);

    println!("output shape: {:?}", output.shape());

    let loss_fn = CrossEntropyLoss;

    let loss = loss_fn.forward(&output, &targets);

    println!("loss: {:?}", loss.data());

    loss.backward();

    for (i, parameter) in model.parameters().iter().enumerate() {
        println!(
            "parameter {i} grad exists: {}",
            parameter.tensor().grad().is_some()
        );
    }

    let mut optimizer = Adam::new(model.parameters(), 0.001);

    for epoch in 0..5 {
        let loader = DataLoader::new(&dataset, 128, true);

        let mut total_loss = 0.;
        let mut sample_count = 0;

        let mut correct = 0;
        let mut total = 0;

        for (inputs, targets) in loader {
            optimizer.zero_grad();

            let output = model.forward(&inputs);
            let loss = loss_fn.forward(&output, &targets);

            let predictions = output.argmax();

            for (prediction, target) in predictions.iter().zip(&targets) {
                if prediction == target {
                    correct += 1;
                }
            }

            total += targets.len();

            let batch_size = targets.len();

            total_loss += loss.data()[0] * batch_size as f32;
            sample_count += batch_size;

            loss.backward();
            optimizer.step();
        }

        let average_loss = total_loss / sample_count as f32;
        let accuracy = correct as f32 / total as f32;

        println!(
            "epoch: {epoch}, loss: {average_loss}, accuracy: {:.2}%",
            accuracy * 100.
        );
    }

    let test_images = load_images("data/mnist/t10k-images-idx3-ubyte");
    let test_labels = load_labels("data/mnist/t10k-labels-idx1-ubyte");

    let test_dataset = ClassificationDataset::new(test_images, test_labels).unwrap();

    let test_loader = DataLoader::new(&test_dataset, 128, true);

    let mut correct = 0;
    let mut total = 0;

    for (inputs, targets) in test_loader {
        let output = model.forward(&inputs);
        let predictions = output.argmax();

        for (prediction, target) in predictions.iter().zip(&targets) {
            if prediction == target {
                correct += 1;
            }
        }

        total += targets.len();
    }

    let accuracy = correct as f32 / total as f32;

    println!("test accuracy: {:.2}%", accuracy * 100.);
}
