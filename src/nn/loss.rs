use crate::tensor::*;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
