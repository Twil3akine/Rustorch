use crate::backend::Device;

#[derive(Clone)]
pub enum Storage {
    Cpu(Vec<f32>),
}

impl Storage {
    pub fn new(data: Vec<f32>, device: Device) -> Self {
        match device {
            Device::Cpu => Storage::Cpu(data),
        }
    }

    pub fn data(&self) -> &[f32] {
        match self {
            Storage::Cpu(data) => data,
        }
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        match self {
            Storage::Cpu(data) => data,
        }
    }

    pub fn len(&self) -> usize {
        self.data().len()
    }

    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
        }
    }
}
