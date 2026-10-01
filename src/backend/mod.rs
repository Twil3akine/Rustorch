pub mod cpu;
pub mod storage;

pub trait Backend {
    fn matmul(lhs: &[f32], rhs: &[f32], n: usize, k: usize, m: usize) -> Vec<f32>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
}
