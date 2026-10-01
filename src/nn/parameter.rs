use crate::tensor::Tensor;

pub struct Parameter {
    tensor: Tensor,
}

impl Parameter {
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn zero_grad(&self) {
        self.tensor.zero_grad();
    }
}

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<&Parameter>;
}

pub struct Linear {
    weight: Parameter,
    bias: Parameter,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let bound = 1. / (in_features as f32).sqrt();

        let weight_data = (0..in_features * out_features)
            .map(|_| rand::random_range(-bound..=bound))
            .collect::<Vec<f32>>();
        let bias_data = (0..out_features)
            .map(|_| rand::random_range(-bound..=bound))
            .collect::<Vec<f32>>();

        let weight = Tensor::new(weight_data, vec![in_features, out_features]).unwrap();
        let bias = Tensor::new(bias_data, vec![out_features]).unwrap();

        Self {
            weight: Parameter::new(weight),
            bias: Parameter::new(bias),
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let result = input.matmul(&self.weight.tensor).unwrap();
        let result = result.add(&self.bias.tensor).unwrap();

        result
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight, &self.bias]
    }
}

pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![]
    }
}

pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(modules: Vec<Box<dyn Module>>) -> Self {
        Self { modules }
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut output = input.clone();

        for module in &self.modules {
            output = module.forward(&output);
        }

        output
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut parameters = Vec::new();

        for module in &self.modules {
            parameters.extend(module.parameters());
        }

        parameters
    }
}

pub trait Loss {
    type Target;
    fn forward(&self, prediction: &Tensor, target: &Self::Target) -> Tensor;
}

pub struct MSELoss;

impl Loss for MSELoss {
    type Target = Tensor;
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let diff = prediction.sub(target).unwrap();
        let squared = diff.mul(&diff).unwrap();

        squared.mean()
    }
}

pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    type Target = Vec<usize>;
    fn forward(&self, prediction: &Tensor, target: &Self::Target) -> Tensor {
        let log_probs = prediction.log_softmax();
        let selected = log_probs.gather(target);
        let negative = selected.neg();

        negative.mean()
    }
}

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct SGD<'a> {
    parameters: Vec<&'a Parameter>,
    lr: f32,
}

impl<'a> SGD<'a> {
    pub fn new(parameters: Vec<&'a Parameter>, lr: f32) -> Self {
        Self { parameters, lr }
    }
}

impl Optimizer for SGD<'_> {
    fn step(&mut self) {
        for parameter in &self.parameters {
            parameter.tensor().apply_grad(self.lr);
        }
    }

    fn zero_grad(&mut self) {
        for parameter in &self.parameters {
            parameter.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear() {
        let linear = Linear::new(3, 2);

        let input = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();

        let output = linear.forward(&input);

        // [2, 3] @ [3, 2] + [2] -> [2, 2]
        assert_eq!(output.shape(), vec![2, 2]);

        let parameters = linear.parameters();

        assert_eq!(parameters.len(), 2);
        assert_eq!(parameters[0].tensor().shape(), vec![3, 2]);
        assert_eq!(parameters[1].tensor().shape(), vec![2]);

        // scalarにしてbackward
        let loss = output.sum();
        loss.backward();

        // weight:
        //
        // 入力
        // [1 2 3]
        // [4 5 6]
        //
        // 各出力について入力方向に足されるので
        //
        // [1+4, 1+4]
        // [2+5, 2+5]
        // [3+6, 3+6]
        assert_eq!(
            parameters[0].tensor().grad(),
            Some(vec![5., 5., 7., 7., 9., 9.,])
        );

        // biasは2つの入力データそれぞれから1ずつ来る
        assert_eq!(parameters[1].tensor().grad(), Some(vec![2., 2.]));
    }

    #[test]
    fn relu() {
        let relu = ReLU;
        let input = Tensor::new(vec![-1., 2., -3., 4.], vec![2, 2]).unwrap();

        let output = relu.forward(&input);

        assert_eq!(
            output,
            Tensor::new(vec![0., 2., 0., 4.], vec![2, 2]).unwrap()
        );

        assert!(relu.parameters().is_empty());
    }

    #[test]
    fn sequential() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(3, 4)),
            Box::new(ReLU),
            Box::new(Linear::new(4, 2)),
        ]);

        let input = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();

        let output = model.forward(&input);

        // [2,3] -> [2,4] -> [2,4] -> [2,2]
        assert_eq!(output.shape(), vec![2, 2]);

        // Linear 2層 × (weight + bias)
        let parameters = model.parameters();
        assert_eq!(parameters.len(), 4);

        let loss = output.sum();
        loss.backward();

        // 全Parameterまで勾配が到達していることを確認
        for parameter in parameters {
            assert!(parameter.tensor().grad().is_some());
        }
    }

    #[test]
    fn mse_loss() {
        let mse = MSELoss;

        let prediction = Tensor::new(vec![2., 4.], vec![2]).unwrap();

        let target = Tensor::new(vec![1., 6.], vec![2]).unwrap();

        let loss = mse.forward(&prediction, &target);

        assert_eq!(loss, Tensor::new(vec![2.5], vec![1]).unwrap());

        loss.backward();

        assert_eq!(prediction.grad(), Some(vec![1., -2.]));
    }

    #[test]
    fn cross_entropy_loss() {
        let loss_fn = CrossEntropyLoss;

        let logits = Tensor::new(vec![2., 1., 0., 0., 1., 2.], vec![2, 3]).unwrap();

        let target = vec![0, 2];

        let loss = loss_fn.forward(&logits, &target);

        assert_eq!(loss.shape(), vec![1]);

        loss.backward();

        assert!(logits.grad().is_some());
    }

    #[test]
    fn sgd_step() {
        let tensor = Tensor::new(vec![1., 2.], vec![2]).unwrap();
        let parameter = Parameter::new(tensor);

        let loss = parameter.tensor().mul(parameter.tensor()).unwrap().sum();

        loss.backward();

        let mut optimizer = SGD::new(vec![&parameter], 0.1);

        optimizer.step();

        assert_eq!(parameter.tensor().data(), vec![0.8, 1.6]);

        optimizer.zero_grad();

        assert_eq!(parameter.tensor().grad(), None);
    }

    #[test]
    fn sgd_reduces_loss() {
        let model = Linear::new(1, 1);
        let loss_fn = MSELoss;

        let input = Tensor::new(vec![1.], vec![1, 1]).unwrap();
        let target = Tensor::new(vec![0.], vec![1, 1]).unwrap();

        let parameters = model.parameters();
        let mut optimizer = SGD::new(parameters, 0.1);

        let prediction = model.forward(&input);
        let loss_before = loss_fn.forward(&prediction, &target);
        let before = loss_before.data()[0];

        optimizer.zero_grad();
        loss_before.backward();
        optimizer.step();

        let prediction = model.forward(&input);
        let loss_before = loss_fn.forward(&prediction, &target);
        let after = loss_before.data()[0];

        assert!(after < before)
    }
}
