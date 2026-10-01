pub mod adam;
pub mod sgd;

pub use adam::*;
pub use sgd::*;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}
