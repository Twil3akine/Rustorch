#![allow(unused)]

use rustorch::tensor::Tensor;

fn main() {
    let tensor = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();

    assert_eq!(tensor.ndim(), 2);
    assert_eq!(tensor.numel(), 4);
    assert_eq!(tensor.shape(), vec![2, 2]);
}
