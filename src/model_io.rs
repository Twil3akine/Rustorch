use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use crate::nn::Module;

pub fn save<M: Module>(model: &M, path: &str) {
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);

    let parameters = model.parameters();

    writer
        .write_all(&(parameters.len() as u64).to_le_bytes())
        .unwrap();

    for parameter in parameters {
        let data = parameter.tensor().data();

        writer
            .write_all(&(data.len() as u64).to_le_bytes())
            .unwrap();

        for value in data {
            writer.write_all(&value.to_le_bytes()).unwrap();
        }
    }
}

pub fn load<M: Module>(model: &M, path: &str) {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);

    let parameters = model.parameters();

    let mut buffer = [0u8; 8];
    reader.read_exact(&mut buffer).unwrap();

    let parameter_count = u64::from_le_bytes(buffer) as usize;

    assert_eq!(parameter_count, parameters.len());

    for parameter in parameters {
        let mut buffer = [0u8; 8];
        reader.read_exact(&mut buffer).unwrap();

        let len = u64::from_le_bytes(buffer) as usize;

        assert_eq!(len, parameter.tensor().numel());

        let mut data = Vec::with_capacity(len);

        for _ in 0..len {
            let mut buffer = [0u8; 4];

            reader.read_exact(&mut buffer).unwrap();

            let value = f32::from_le_bytes(buffer);

            data.push(value);
        }

        parameter.set_data(&data);
    }
}
