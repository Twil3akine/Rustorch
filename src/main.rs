use std::env;
use std::process;

use rustorch::autograd::*;
use rustorch::data::*;
use rustorch::model_io::*;
use rustorch::nn::*;
use rustorch::optim::*;

const MODEL_PATH: &str = "mnist.rth";
const TRAIN_IMAGES: &str = "data/mnist/train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "data/mnist/train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "data/mnist/t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "data/mnist/t10k-labels-idx1-ubyte";

fn build_model() -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(784, 128)),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10)),
    ])
}

fn load_dataset(images_path: &str, labels_path: &str) -> ClassificationDataset {
    let images = load_images(images_path);
    let labels = load_labels(labels_path);

    ClassificationDataset::new(images, labels).unwrap()
}

fn evaluate(model: &Sequential, dataset: &ClassificationDataset) -> f32 {
    let loader = DataLoader::new(dataset, 128, false);

    no_grad(|| {
        let mut correct = 0;
        let mut total = 0;

        for (inputs, targets) in loader {
            let output = model.forward(&inputs);
            let predictions = output.argmax();

            for (prediction, target) in predictions.iter().zip(&targets) {
                if prediction == target {
                    correct += 1;
                }
            }

            total += targets.len();
        }

        correct as f32 / total as f32
    })
}

fn train() {
    let dataset = load_dataset(TRAIN_IMAGES, TRAIN_LABELS);
    let model = build_model();

    let loss_fn = CrossEntropyLoss;
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

    save(&model, MODEL_PATH);
    println!("saved model to {MODEL_PATH}");

    let test_dataset = load_dataset(TEST_IMAGES, TEST_LABELS);
    let test_accuracy = evaluate(&model, &test_dataset);

    println!("test accuracy: {:.2}%", test_accuracy * 100.);
}

fn eval() {
    let model = build_model();

    load(&model, MODEL_PATH);

    let test_dataset = load_dataset(TEST_IMAGES, TEST_LABELS);
    let accuracy = evaluate(&model, &test_dataset);

    println!("test accuracy: {:.2}%", accuracy * 100.);
}

fn main() {
    let mode = env::args().nth(1).unwrap_or_else(|| "eval".to_string());

    match mode.as_str() {
        "train" => train(),
        "eval" => eval(),
        _ => {
            eprintln!("usage: rustorch [train|eval]");
            process::exit(2);
        }
    }
}
