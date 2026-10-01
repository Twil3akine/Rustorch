use super::Backend;

use crate::shape::*;

pub struct Cpu;

impl Cpu {
    fn binary_op<F>(
        lhs: &[f32],
        rhs: &[f32],
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        output_shape: &[usize],
        op: F,
    ) -> Vec<f32>
    where
        F: Fn(f32, f32) -> f32,
    {
        let output_numel = output_shape.iter().product();
        let mut data = Vec::with_capacity(output_numel);

        for output_offset in 0..output_numel {
            let output_index = unravel_index(output_offset, &output_shape);
            let lhs_offset = broadcast_offset(&output_index, lhs_shape);
            let rhs_offset = broadcast_offset(&output_index, rhs_shape);

            data.push(op(lhs[lhs_offset], rhs[rhs_offset]));
        }

        data
    }
}
impl Backend for Cpu {
    fn add(
        lhs: &[f32],
        rhs: &[f32],
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        output_shape: &[usize],
    ) -> Vec<f32> {
        Self::binary_op(lhs, rhs, lhs_shape, rhs_shape, output_shape, |x, y| x + y)
    }
    fn sub(
        lhs: &[f32],
        rhs: &[f32],
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        output_shape: &[usize],
    ) -> Vec<f32> {
        Self::binary_op(lhs, rhs, lhs_shape, rhs_shape, output_shape, |x, y| x - y)
    }
    fn mul(
        lhs: &[f32],
        rhs: &[f32],
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        output_shape: &[usize],
    ) -> Vec<f32> {
        Self::binary_op(lhs, rhs, lhs_shape, rhs_shape, output_shape, |x, y| x * y)
    }
    fn matmul(lhs: &[f32], rhs: &[f32], n: usize, k: usize, m: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(n * m);

        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.;

                for t in 0..k {
                    sum += lhs[i * k + t] * rhs[t * m + j];
                }

                result.push(sum);
            }
        }

        result
    }

    fn relu(input: &[f32]) -> Vec<f32> {
        input.iter().map(|x| x.max(0.)).collect()
    }

    fn exp(input: &[f32]) -> Vec<f32> {
        input.iter().map(|x| f32::exp(*x)).collect::<Vec<f32>>()
    }

    fn log(input: &[f32]) -> Vec<f32> {
        input.iter().map(|x| f32::ln(*x)).collect::<Vec<f32>>()
    }

    fn neg(input: &[f32]) -> Vec<f32> {
        input.iter().map(|x| -x).collect::<Vec<f32>>()
    }
}
