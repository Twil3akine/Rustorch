use crate::tensor::*;

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

    pub(crate) fn set_data(&self, data: &[f32]) {
        self.tensor().set_data(data);
    }
}
