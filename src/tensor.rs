use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone)]
pub struct Tensor {
    inner: Rc<RefCell<TensorInner>>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.borrow();

        f.debug_struct("Tensor")
            .field("data", &inner.data)
            .field("shape", &inner.shape)
            .finish()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        let self_inner = self.inner.borrow();
        let other_inner = other.inner.borrow();

        self_inner.data == other_inner.data && self_inner.shape == other_inner.shape
    }
}

struct TensorInner {
    data: Vec<f32>,
    shape: Vec<usize>,
    grad: Option<Vec<f32>>,
    operation: Option<Operation>,
    parents: Vec<Tensor>,
}

#[derive(Clone, Debug)]
enum Operation {
    Add,
    Sub,
    Mul,
    MatMul,
    Sum,
    Mean,
    ReLU,
    Exp,
    Log,
    Reshape,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Option<Self> {
        if data.len() != shape.iter().product() {
            return None;
        }

        let inner = TensorInner {
            data,
            shape,
            grad: None,
            operation: None,
            parents: Vec::new(),
        };

        Some(Self {
            inner: Rc::new(RefCell::new(inner)),
        })
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().shape.clone()
    }

    pub fn ndim(&self) -> usize {
        self.inner.borrow().shape.len()
    }

    pub fn numel(&self) -> usize {
        self.inner.borrow().shape.iter().product()
    }

    pub fn get(&self, index: &[usize]) -> Option<f32> {
        let inner = self.inner.borrow();

        if index.len() != inner.shape.len() {
            return None;
        }

        if !inner.shape.iter().zip(index).all(|(si, idx)| idx < si) {
            return None;
        }

        let mut offset: usize = 0;
        for i in 0..self.ndim() {
            offset = offset * inner.shape[i] + index[i];
        }

        Some(inner.data[offset])
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Option<Self> {
        let new_numel: usize = shape.iter().product();

        if self.numel() != new_numel {
            return None;
        }

        let data = self.inner.borrow().data.clone();

        Some(Self::from_operation(
            data,
            shape,
            Operation::Reshape,
            vec![self.clone()],
        ))
    }

    fn from_operation(
        data: Vec<f32>,
        shape: Vec<usize>,
        operation: Operation,
        parents: Vec<Tensor>,
    ) -> Self {
        Self {
            inner: Rc::new(RefCell::new(TensorInner {
                data,
                shape,
                grad: None,
                operation: Some(operation),
                parents,
            })),
        }
    }

    fn binary_op<F>(&self, other: &Self, operation: Operation, op: F) -> Option<Tensor>
    where
        F: Fn(f32, f32) -> f32,
    {
        let self_shape = self.inner.borrow().shape.clone();
        let other_shape = other.inner.borrow().shape.clone();

        let output_shape = Self::broadcast_shape(&self_shape, &other_shape)?;
        let output_numel: usize = output_shape.iter().product();

        let data = {
            let self_inner = self.inner.borrow();
            let other_inner = other.inner.borrow();

            let mut data = Vec::with_capacity(output_numel);

            for output_offset in 0..output_numel {
                let output_index = Self::unravel_index(output_offset, &output_shape);
                let self_offset = Self::broadcast_offset(&output_index, &self_shape);
                let other_offset = Self::broadcast_offset(&output_index, &other_shape);

                data.push(op(
                    self_inner.data[self_offset],
                    other_inner.data[other_offset],
                ));
            }

            data
        };

        Some(Self::from_operation(
            data,
            output_shape,
            operation,
            vec![self.clone(), other.clone()],
        ))
    }

    pub fn add(&self, other: &Self) -> Option<Tensor> {
        self.binary_op(other, Operation::Add, |x, y| x + y)
    }

    pub fn sub(&self, other: &Self) -> Option<Tensor> {
        self.binary_op(other, Operation::Sub, |x, y| x - y)
    }

    pub fn mul(&self, other: &Self) -> Option<Tensor> {
        self.binary_op(other, Operation::Mul, |x, y| x * y)
    }

    pub fn matmul(&self, other: &Self) -> Option<Tensor> {
        if !(self.ndim() == 2
            && other.ndim() == 2
            && self.inner.borrow().shape[1] == other.inner.borrow().shape[0])
        {
            return None;
        }

        let (result, n, m) = {
            let self_inner = self.inner.borrow();
            let other_inner = other.inner.borrow();

            let n = self_inner.shape[0];
            let k = self_inner.shape[1];
            let m = other_inner.shape[1];

            let mut result = Vec::with_capacity(n * m);

            for i in 0..n {
                for j in 0..m {
                    let mut sum: f32 = 0.;

                    for t in 0..k {
                        sum += self_inner.data[k * i + t] * other_inner.data[m * t + j];
                    }

                    result.push(sum);
                }
            }

            (result, n, m)
        };

        Some(Self::from_operation(
            result,
            vec![n, m],
            Operation::MatMul,
            vec![self.clone(), other.clone()],
        ))
    }

    pub fn sum(&self) -> Tensor {
        let data = {
            let inner = self.inner.borrow();
            vec![inner.data.iter().sum()]
        };

        Self::from_operation(data, vec![1], Operation::Sum, vec![self.clone()])
    }

    pub fn mean(&self) -> Tensor {
        let data = {
            let inner = self.inner.borrow();
            vec![inner.data.iter().sum::<f32>() / inner.data.len() as f32]
        };

        Self::from_operation(data, vec![1], Operation::Mean, vec![self.clone()])
    }

    pub fn relu(&self) -> Tensor {
        let (data, shape) = {
            let inner = self.inner.borrow();

            let data = inner.data.iter().map(|x| x.max(0.)).collect::<Vec<f32>>();

            let shape = inner.shape.clone();

            (data, shape)
        };

        Self::from_operation(data, shape, Operation::ReLU, vec![self.clone()])
    }

    pub fn exp(&self) -> Tensor {
        let (data, shape) = {
            let inner = self.inner.borrow();

            let data = inner
                .data
                .iter()
                .map(|x| f32::exp(*x))
                .collect::<Vec<f32>>();

            let shape = inner.shape.clone();

            (data, shape)
        };

        Self::from_operation(data, shape, Operation::Exp, vec![self.clone()])
    }

    pub fn log(&self) -> Tensor {
        let (data, shape) = {
            let inner = self.inner.borrow();

            let data = inner.data.iter().map(|x| f32::ln(*x)).collect::<Vec<f32>>();

            let shape = inner.shape.clone();

            (data, shape)
        };

        Self::from_operation(data, shape, Operation::Log, vec![self.clone()])
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

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.inner.borrow().grad.clone()
    }

    fn accumulate_grad(&self, grad: &[f32]) {
        let mut inner = self.inner.borrow_mut();

        assert_eq!(inner.data.len(), grad.len());

        match &mut inner.grad {
            Some(current_grad) => {
                for (current, incoming) in current_grad.iter_mut().zip(grad) {
                    *current += incoming;
                }
            }
            None => {
                inner.grad = Some(grad.to_vec());
            }
        }
    }

    pub fn backward(&self) {
        assert_eq!(
            self.numel(),
            1,
            "backward() currently requires a scalar tensor."
        );

        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        let mut stack = vec![(self.clone(), false)];

        // 親方向にDFSして、トポロジカルソート
        while let Some((tensor, expanded)) = stack.pop() {
            let ptr = Rc::as_ptr(&tensor.inner);

            if expanded {
                topo.push(tensor);
                continue;
            }

            if !visited.insert(ptr) {
                continue;
            }

            stack.push((tensor.clone(), true));

            let parents = tensor.inner.borrow().parents.clone();

            for parent in parents {
                stack.push((parent, false));
            }
        }

        // dy/dx = 1
        self.accumulate_grad(&[1.]);

        for tensor in topo.into_iter().rev() {
            let (grad, operation, parents) = {
                let inner = tensor.inner.borrow();

                (
                    inner.grad.clone(),
                    inner.operation.clone(),
                    inner.parents.clone(),
                )
            };

            let Some(grad) = grad else {
                continue;
            };

            match operation {
                Some(Operation::Add) => {
                    let output_shape = tensor.shape();

                    for parent in &parents {
                        let parent_shape = parent.shape();
                        let mut parent_grad = vec![0.; parent.numel()];

                        for output_offset in 0..grad.len() {
                            let output_index = Self::unravel_index(output_offset, &output_shape);
                            let parent_offset =
                                Self::broadcast_offset(&output_index, &parent_shape);

                            parent_grad[parent_offset] += grad[output_offset];
                        }

                        parent.accumulate_grad(&parent_grad);
                    }
                }

                Some(Operation::Sub) => {
                    let lhs = &parents[0];
                    let rhs = &parents[1];

                    let lhs_shape = lhs.shape();
                    let rhs_shape = rhs.shape();
                    let output_shape = tensor.shape();

                    let mut lhs_grad = vec![0.; lhs.numel()];
                    let mut rhs_grad = vec![0.; rhs.numel()];

                    for output_offset in 0..grad.len() {
                        let output_index = Self::unravel_index(output_offset, &output_shape);

                        let lhs_offset = Self::broadcast_offset(&output_index, &lhs_shape);
                        let rhs_offset = Self::broadcast_offset(&output_index, &rhs_shape);

                        lhs_grad[lhs_offset] += grad[output_offset];
                        rhs_grad[rhs_offset] -= grad[output_offset];
                    }

                    lhs.accumulate_grad(&lhs_grad);
                    rhs.accumulate_grad(&rhs_grad);
                }

                Some(Operation::Mul) => {
                    let lhs = &parents[0];
                    let rhs = &parents[1];

                    let lhs_shape = lhs.shape();
                    let rhs_shape = rhs.shape();
                    let output_shape = tensor.shape();

                    let lhs_data = lhs.inner.borrow().data.clone();
                    let rhs_data = rhs.inner.borrow().data.clone();

                    let mut lhs_grad = vec![0.; lhs.numel()];
                    let mut rhs_grad = vec![0.; rhs.numel()];

                    for output_offset in 0..grad.len() {
                        let output_index = Self::unravel_index(output_offset, &output_shape);

                        let lhs_offset = Self::broadcast_offset(&output_index, &lhs_shape);
                        let rhs_offset = Self::broadcast_offset(&output_index, &rhs_shape);

                        lhs_grad[lhs_offset] += grad[output_offset] * rhs_data[rhs_offset];
                        rhs_grad[rhs_offset] += grad[output_offset] * lhs_data[lhs_offset];
                    }

                    lhs.accumulate_grad(&lhs_grad);
                    rhs.accumulate_grad(&rhs_grad);
                }

                Some(Operation::MatMul) => {
                    let lhs = &parents[0];
                    let rhs = &parents[1];

                    let lhs_shape = lhs.shape();
                    let rhs_shape = rhs.shape();

                    let n = lhs_shape[0];
                    let k = lhs_shape[1];
                    let m = rhs_shape[1];

                    let lhs_data = lhs.inner.borrow().data.clone();
                    let rhs_data = rhs.inner.borrow().data.clone();

                    let mut lhs_grad = vec![0.0; lhs.numel()];
                    let mut rhs_grad = vec![0.0; rhs.numel()];

                    // dL/dA = G @ B^T
                    for i in 0..n {
                        for t in 0..k {
                            let mut sum = 0.0;

                            for j in 0..m {
                                sum += grad[i * m + j] * rhs_data[t * m + j];
                            }

                            lhs_grad[i * k + t] = sum;
                        }
                    }

                    // dL/dB = A^T @ G
                    for t in 0..k {
                        for j in 0..m {
                            let mut sum = 0.0;

                            for i in 0..n {
                                sum += lhs_data[i * k + t] * grad[i * m + j];
                            }

                            rhs_grad[t * m + j] = sum;
                        }
                    }

                    lhs.accumulate_grad(&lhs_grad);
                    rhs.accumulate_grad(&rhs_grad);
                }

                Some(Operation::Sum) => {
                    let parent = &parents[0];
                    let parent_grad = vec![grad[0]; parent.numel()];

                    parent.accumulate_grad(&parent_grad);
                }

                Some(Operation::Mean) => {
                    let parent = &parents[0];
                    let n = parent.numel() as f32;

                    let parent_grad = vec![grad[0] / n; parent.numel()];

                    parent.accumulate_grad(&parent_grad);
                }

                Some(Operation::ReLU) => {
                    let parent = &parents[0];

                    let parent_data = parent.inner.borrow().data.clone();

                    let parent_grad = grad
                        .iter()
                        .zip(parent_data.iter())
                        .map(|(g, x)| if *x > 0. { *g } else { 0. })
                        .collect::<Vec<f32>>();

                    parent.accumulate_grad(&parent_grad);
                }

                Some(Operation::Exp) => {
                    let parent = &parents[0];

                    let output_data = tensor.inner.borrow().data.clone();

                    let parent_grad = grad
                        .iter()
                        .zip(output_data.iter())
                        .map(|(g, x)| g * x)
                        .collect::<Vec<f32>>();

                    parent.accumulate_grad(&parent_grad);
                }

                Some(Operation::Log) => {
                    let parent = &parents[0];

                    let parent_data = parent.inner.borrow().data.clone();

                    let parent_grad = grad
                        .iter()
                        .zip(parent_data.iter())
                        .map(|(g, x)| g / x)
                        .collect::<Vec<f32>>();

                    parent.accumulate_grad(&parent_grad);
                }

                Some(Operation::Reshape) => {
                    parents[0].accumulate_grad(&grad);
                }

                _ => {}
            }
        }
    }

    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = None;
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

        assert_eq!(reshaped.shape(), vec![4, 3, 2]);
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

    #[test]
    fn matmul() {
        let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();

        let b = Tensor::new(vec![7., 8., 9., 10., 11., 12.], vec![3, 2]).unwrap();

        assert_eq!(
            a.matmul(&b),
            Tensor::new(vec![58., 64., 139., 154.], vec![2, 2],)
        );
    }

    #[test]
    fn computation_graph() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5., 6., 7., 8.], vec![2, 2]).unwrap();

        let c = a.mul(&b).unwrap();
        let d = c.add(&a).unwrap();
        let e = d.relu();
        let f = e.sum();

        {
            let inner = c.inner.borrow();

            assert!(matches!(inner.operation, Some(Operation::Mul)));
            assert_eq!(inner.parents.len(), 2);

            assert!(Rc::ptr_eq(&inner.parents[0].inner, &a.inner,));
            assert!(Rc::ptr_eq(&inner.parents[1].inner, &b.inner,));
        }

        {
            let inner = d.inner.borrow();

            assert!(matches!(inner.operation, Some(Operation::Add)));
            assert_eq!(inner.parents.len(), 2);

            assert!(Rc::ptr_eq(&inner.parents[0].inner, &c.inner,));
            assert!(Rc::ptr_eq(&inner.parents[1].inner, &a.inner,));
        }

        {
            let inner = e.inner.borrow();

            assert!(matches!(inner.operation, Some(Operation::ReLU)));
            assert_eq!(inner.parents.len(), 1);

            assert!(Rc::ptr_eq(&inner.parents[0].inner, &d.inner,));
        }

        {
            let inner = f.inner.borrow();

            assert!(matches!(inner.operation, Some(Operation::Sum)));
            assert_eq!(inner.parents.len(), 1);

            assert!(Rc::ptr_eq(&inner.parents[0].inner, &e.inner,));
        }
    }

    #[test]
    fn accumulate_gradient() {
        let x = Tensor::new(vec![1., 2.], vec![2]).unwrap();

        assert_eq!(x.grad(), None);

        x.accumulate_grad(&[3., 4.]);
        assert_eq!(x.grad(), Some(vec![3., 4.]));

        x.accumulate_grad(&[1., 2.]);
        assert_eq!(x.grad(), Some(vec![4., 6.]));

        x.zero_grad();
        assert_eq!(x.grad(), None);
    }

    #[test]
    fn backward() {
        let a = Tensor::new(vec![1., 2.], vec![2]).unwrap();
        let b = Tensor::new(vec![3., 4.], vec![2]).unwrap();

        let c = a.add(&b).unwrap();
        let y = c.sum();

        // Add + Sum
        {
            let a = Tensor::new(vec![1., 2.], vec![2]).unwrap();
            let b = Tensor::new(vec![3., 4.], vec![2]).unwrap();

            let c = a.add(&b).unwrap();
            let y = c.sum();

            y.backward();

            assert_eq!(a.grad(), Some(vec![1., 1.]));
            assert_eq!(b.grad(), Some(vec![1., 1.]));
        }

        // Sub + Sum
        {
            let a = Tensor::new(vec![1., 2.], vec![2]).unwrap();
            let b = Tensor::new(vec![3., 4.], vec![2]).unwrap();

            let c = a.sub(&b).unwrap();
            let y = c.sum();

            y.backward();

            assert_eq!(a.grad(), Some(vec![1., 1.]));
            assert_eq!(b.grad(), Some(vec![-1., -1.]));
        }

        // x * x
        {
            let x = Tensor::new(vec![2.], vec![1]).unwrap();

            let y = x.mul(&x).unwrap();

            y.backward();

            assert_eq!(x.grad(), Some(vec![4.]));
        }

        // Mean
        {
            let x = Tensor::new(vec![2., 4.], vec![2]).unwrap();

            let y = x.mean();
            y.backward();

            assert_eq!(x.grad(), Some(vec![0.5, 0.5]));
        }

        // ReLU + Sum
        {
            let x = Tensor::new(vec![-2., 3., -1., 5.], vec![4]).unwrap();

            let y = x.relu().sum();
            y.backward();

            assert_eq!(x.grad(), Some(vec![0., 1., 0., 1.]));
        }

        // broadcasting
        {
            let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();

            let b = Tensor::new(vec![10., 20., 30.], vec![3]).unwrap();

            let y = a.add(&b).unwrap().sum();

            y.backward();

            assert_eq!(a.grad(), Some(vec![1., 1., 1., 1., 1., 1.]));

            assert_eq!(b.grad(), Some(vec![2., 2., 2.]));
        }

        {
            let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();

            let b = Tensor::new(vec![5., 6.], vec![2, 1]).unwrap();

            let y = a.matmul(&b).unwrap().sum();

            y.backward();

            assert_eq!(a.grad(), Some(vec![5., 6., 5., 6.,]));

            assert_eq!(b.grad(), Some(vec![4., 6.,]));
        }

        {
            let x = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();

            let y = x.reshape(vec![4]).unwrap().sum();
            y.backward();

            assert_eq!(x.grad(), Some(vec![1., 1., 1., 1.]));
        }

        // Exp
        {
            let x = Tensor::new(vec![0., 1.], vec![2]).unwrap();

            let y = x.exp().sum();
            y.backward();

            assert_eq!(x.grad(), Some(vec![1., std::f32::consts::E]));
        }

        // Log
        {
            let x = Tensor::new(vec![1., std::f32::consts::E], vec![2]).unwrap();

            let y = x.log().sum();
            y.backward();

            assert_eq!(x.grad(), Some(vec![1., 1. / std::f32::consts::E]));
        }
    }
}
