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
    fn matmul(
        lhs: &[f32],
        rhs: &[f32],
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        output_shape: &[usize],
    ) -> Vec<f32> {
        let n = lhs_shape[lhs_shape.len() - 2];
        let k = lhs_shape[lhs_shape.len() - 1];
        let m = rhs_shape[rhs_shape.len() - 1];

        let lhs_batch_shape = &lhs_shape[..lhs_shape.len() - 2];

        let rhs_batch_shape = &rhs_shape[..rhs_shape.len() - 2];

        let output_batch_shape = &output_shape[..output_shape.len() - 2];

        let batch_count: usize = output_batch_shape.iter().product();

        let mut result = Vec::with_capacity(batch_count * n * m);

        for batch in 0..batch_count {
            let batch_index = unravel_index(batch, output_batch_shape);

            let lhs_batch = broadcast_offset(&batch_index, lhs_batch_shape);

            let rhs_batch = broadcast_offset(&batch_index, rhs_batch_shape);

            let lhs_base = lhs_batch * n * k;
            let rhs_base = rhs_batch * k * m;

            for i in 0..n {
                for j in 0..m {
                    let mut sum = 0.;

                    for t in 0..k {
                        sum += lhs[lhs_base + i * k + t] * rhs[rhs_base + t * m + j];
                    }

                    result.push(sum);
                }
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

    fn sum(input: &[f32]) -> f32 {
        input.iter().sum()
    }

    fn mean(input: &[f32]) -> f32 {
        Self::sum(input) / input.len() as f32
    }

    fn log_softmax(input: &[f32], n: usize, c: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(input.len());

        for i in 0..n {
            let row = &input[i * c..(i + 1) * c];

            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let exp_sum = row.iter().map(|x| (*x - max).exp()).sum::<f32>();

            let log_sum_exp = exp_sum.ln();

            for x in row {
                data.push((*x - max) - log_sum_exp);
            }
        }

        data
    }

    fn gather(input: &[f32], indices: &[usize], n: usize, c: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n);

        for i in 0..n {
            let j = indices[i];

            assert!(j < c);

            data.push(input[i * c + j]);
        }

        data
    }
}
