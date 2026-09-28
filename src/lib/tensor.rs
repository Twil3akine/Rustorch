#[derive(Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Option<Self> {
        if data.len() == shape.iter().product() {
            Some(Self { data, shape })
        } else {
            None
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn get(&self, index: &[usize]) -> Option<f32> {
        if index.len() != self.ndim() {
            return None;
        }

        if self.shape.iter().zip(index).all(|(si, idx)| idx < si) {
            let mut offset: usize = 0;
            for i in 0..self.ndim() {
                offset = offset * self.shape[i] + index[i];
            }

            Some(self.data[offset])
        } else {
            None
        }
    }

    pub fn reshape(mut self, shape: Vec<usize>) -> Option<Self> {
        let new_shape_element_number: usize = shape.iter().product();

        if self.numel() == new_shape_element_number {
            self.shape = shape;
            Some(self)
        } else {
            None
        }
    }

    fn binary_op<F>(&self, other: &Self, op: F) -> Option<Tensor>
    where
        F: Fn(f32, f32) -> f32,
    {
        let output_shape = Self::broadcast_shape(&self.shape, &other.shape)?;

        let output_numel: usize = output_shape.iter().product();

        let mut data = Vec::with_capacity(output_numel);

        for output_offset in 0..output_numel {
            let output_index = Self::unravel_index(output_offset, &output_shape);

            let self_offset = Self::broadcast_offset(&output_index, &self.shape);

            let other_offset = Self::broadcast_offset(&output_index, &other.shape);

            data.push(op(self.data[self_offset], other.data[other_offset]));
        }

        Some(Self {
            data,
            shape: output_shape,
        })
    }

    pub fn add(&self, other: &Self) -> Option<Tensor> {
        self.binary_op(other, |x, y| x + y)
    }

    pub fn sub(&self, other: &Self) -> Option<Tensor> {
        self.binary_op(other, |x, y| x - y)
    }

    pub fn mul(&self, other: &Self) -> Option<Tensor> {
        self.binary_op(other, |x, y| x * y)
    }

    pub fn sum(&self) -> Tensor {
        Self {
            data: vec![self.data.iter().sum::<f32>()],
            shape: vec![1],
        }
    }

    pub fn mean(&self) -> Tensor {
        Self {
            data: vec![self.data.iter().sum::<f32>() / self.numel() as f32],
            shape: vec![1],
        }
    }

    pub fn relu(&self) -> Tensor {
        Self {
            data: self.data.iter().map(|x| x.max(0.)).collect(),
            shape: self.shape.clone(),
        }
    }

    fn unravel_index(mut offset: usize, shape: &[usize]) -> Vec<usize> {
        let mut index = vec![0; shape.len()];

        for i in (0..shape.len()).rev() {
            index[i] = offset % shape[i];
            offset /= shape[i];
        }

        index
    }

    fn broadcast_shape(x: &[usize], y: &[usize]) -> Option<Vec<usize>> {
        let ndim = x.len().max(y.len());
        let mut result = Vec::new();

        for i in 0..ndim {
            let xi = if x.len() <= i { 1 } else { x[x.len() - i - 1] };
            let yi = if y.len() <= i { 1 } else { y[y.len() - i - 1] };

            if xi == 1 || yi == 1 || xi == yi {
                result.push(xi.max(yi));
            } else {
                return None;
            }
        }

        result.reverse();

        Some(result)
    }

    fn broadcast_offset(output_index: &[usize], input_shape: &[usize]) -> usize {
        let shift = output_index.len() - input_shape.len();

        let mut offset = 0;

        for i in 0..input_shape.len() {
            let dim = input_shape[i];

            let output_i = output_index[shift + i];

            let input_i = if dim == 1 { 0 } else { output_i };

            offset = offset * dim + input_i;
        }

        offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_tensor() {
        let tensor = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

        assert!(tensor.is_some());
    }

    #[test]
    fn reject_invalid_shape() {
        let tensor = Tensor::new(vec![1., 2., 3.], vec![2, 2]);

        assert!(tensor.is_none());
    }

    #[test]
    fn tensor_properties() {
        let tensor = Tensor::new(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
            vec![2, 3, 4],
        )
        .unwrap();

        assert_eq!(tensor.ndim(), 3);
        assert_eq!(tensor.numel(), 24);
    }

    #[test]
    fn get_valid_index() {
        let tensor = Tensor::new(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
            vec![2, 3, 4],
        )
        .unwrap();

        assert_eq!(tensor.get(&[0, 0, 0]), Some(1.));
        assert_eq!(tensor.get(&[1, 2, 3]), Some(24.));
    }

    #[test]
    fn get_invalid_index() {
        let tensor = Tensor::new(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
            vec![2, 3, 4],
        )
        .unwrap();

        assert!(tensor.get(&[0, 0, 4]).is_none());
        assert!(tensor.get(&[2, 0, 0]).is_none());
        assert!(tensor.get(&[0, 3, 0]).is_none());
    }

    #[test]
    fn get_wrong_dimension() {
        let tensor = Tensor::new(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
            vec![2, 3, 4],
        )
        .unwrap();

        assert!(tensor.get(&[0, 0]).is_none());
        assert!(tensor.get(&[0, 0, 0, 0]).is_none());
    }

    #[test]
    fn reshape_success() {
        let tensor = Tensor::new(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
            vec![2, 3, 4],
        )
        .unwrap();

        let reshaped = tensor.reshape(vec![4, 3, 2]);

        assert!(reshaped.is_some());

        let reshaped = reshaped.unwrap();

        assert_eq!(reshaped.shape(), &[4, 3, 2]);
        assert_eq!(reshaped.ndim(), 3);
        assert_eq!(reshaped.numel(), 24);

        assert_eq!(reshaped.get(&[0, 0, 0]), Some(1.));
        assert_eq!(reshaped.get(&[0, 0, 1]), Some(2.));
        assert_eq!(reshaped.get(&[0, 1, 0]), Some(3.));
    }

    #[test]
    fn reshape_fail() {
        let tensor = Tensor::new(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
            vec![2, 3, 4],
        )
        .unwrap();

        let reshaped = tensor.reshape(vec![5, 5]);

        assert!(reshaped.is_none());
    }

    #[test]
    fn binary_ops() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();

        assert_eq!(a.add(&b), Tensor::new(vec![2., 4., 6., 8.], vec![2, 2]));

        assert_eq!(a.sub(&b), Tensor::new(vec![0., 0., 0., 0.], vec![2, 2]));

        assert_eq!(a.mul(&b), Tensor::new(vec![1., 4., 9., 16.], vec![2, 2]));
    }

    #[test]
    fn sum() {
        let tensor = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();

        assert_eq!(tensor.sum(), Tensor::new(vec![10f32], vec![1]).unwrap())
    }

    #[test]
    fn mean() {
        let tensor = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();

        assert_eq!(tensor.mean(), Tensor::new(vec![2.5], vec![1]).unwrap())
    }

    #[test]
    fn relu() {
        let tensor = Tensor::new(vec![-1., -2., 1., 2.], vec![2, 2]).unwrap();

        assert_eq!(
            tensor.relu(),
            Tensor::new(vec![0., 0., 1., 2.], vec![2, 2]).unwrap()
        )
    }

    #[test]
    fn broadcast_shape_test() {
        assert_eq!(Tensor::broadcast_shape(&[2, 3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(Tensor::broadcast_shape(&[2, 3], &[3]), Some(vec![2, 3]));
        assert_eq!(Tensor::broadcast_shape(&[2, 3], &[1, 3]), Some(vec![2, 3]));
        assert_eq!(
            Tensor::broadcast_shape(&[2, 1, 4], &[3, 4]),
            Some(vec![2, 3, 4])
        );
        assert_eq!(Tensor::broadcast_shape(&[2, 3], &[4, 3]), None);
    }
}
