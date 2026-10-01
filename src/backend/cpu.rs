use super::Backend;

pub struct Cpu;

impl Backend for Cpu {
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
}
