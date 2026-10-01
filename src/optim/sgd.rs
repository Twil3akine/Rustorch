use crate::nn::*;
use crate::optim::*;
use crate::tensor::*;

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
            let Some(grad) = parameter.tensor().grad() else {
                continue;
            };

            let update = grad.iter().map(|g| self.lr * g).collect::<Vec<_>>();

            parameter.tensor().apply_update(&update);
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
