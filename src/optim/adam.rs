use crate::nn::*;
use crate::optim::*;
use crate::tensor::*;

pub struct Adam<'a> {
    parameters: Vec<&'a Parameter>,

    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,

    t: usize,

    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl<'a> Adam<'a> {
    pub fn new(parameters: Vec<&'a Parameter>, lr: f32) -> Self {
        let m = parameters
            .iter()
            .map(|prm| vec![0.; prm.tensor().numel()])
            .collect::<Vec<_>>();

        let v = parameters
            .iter()
            .map(|prm| vec![0.; prm.tensor().numel()])
            .collect::<Vec<_>>();

        Self {
            parameters,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m,
            v,
        }
    }
}

impl<'a> Optimizer for Adam<'a> {
    fn step(&mut self) {
        self.t += 1;

        for i in 0..self.parameters.len() {
            let parameter = self.parameters[i];

            let Some(grad) = parameter.tensor().grad() else {
                continue;
            };

            let mut update = Vec::with_capacity(grad.len());

            for (j, g) in grad.iter().enumerate() {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1. - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1. - self.beta2) * g * g;

                let m_hat = self.m[i][j] / (1. - self.beta1.powi(self.t as i32));
                let v_hat = self.v[i][j] / (1. - self.beta2.powi(self.t as i32));

                let delta = self.lr * m_hat / (v_hat.sqrt() + self.eps);

                update.push(delta);
            }

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
    fn adam_step() {
        let tensor = Tensor::new(vec![1., 2.], vec![2]).unwrap();
        let parameter = Parameter::new(tensor);

        let loss = parameter.tensor().mul(parameter.tensor()).unwrap().sum();

        loss.backward();

        assert_eq!(parameter.tensor().grad(), Some(vec![2., 4.]));

        let mut optimizer = Adam::new(vec![&parameter], 0.001);

        optimizer.step();

        let data = parameter.tensor().data();

        assert!((data[0] - 0.999).abs() < 1e-6);
        assert!((data[1] - 1.999).abs() < 1e-6);

        optimizer.zero_grad();

        assert_eq!(parameter.tensor().grad(), None);
    }
}
