pub mod cpu;
pub mod storage;

pub trait Backend {
    fn add(
        lhs: &[f32],
        rhs: &[f32],
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        output_shape: &[usize],
    ) -> Vec<f32>;
    fn sub(
        lhs: &[f32],
        rhs: &[f32],
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        output_shape: &[usize],
    ) -> Vec<f32>;
    fn mul(
        lhs: &[f32],
        rhs: &[f32],
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        output_shape: &[usize],
    ) -> Vec<f32>;
    fn matmul(lhs: &[f32], rhs: &[f32], n: usize, k: usize, m: usize) -> Vec<f32>;
    fn relu(input: &[f32]) -> Vec<f32>;
    fn exp(input: &[f32]) -> Vec<f32>;
    fn log(input: &[f32]) -> Vec<f32>;
    fn neg(input: &[f32]) -> Vec<f32>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
}
