use crate::tensor::Tensor;

fn read_u32_be(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

pub fn load_images(path: &str) -> Tensor {
    let bytes = std::fs::read(path).unwrap();

    let magic = read_u32_be(&bytes, 0);

    assert_eq!(magic, 2051);

    let count = read_u32_be(&bytes, 4);
    let rows = read_u32_be(&bytes, 8);
    let cols = read_u32_be(&bytes, 12);

    let pixels = &bytes[16..];

    let data = pixels
        .iter()
        .map(|&pixel| pixel as f32 / 255.)
        .collect::<Vec<f32>>();

    assert_eq!(pixels.len(), count as usize * rows as usize * cols as usize);

    Tensor::new(data, vec![count as usize, (rows * cols) as usize]).unwrap()
}

pub fn load_labels(path: &str) -> Vec<usize> {
    let bytes = std::fs::read(path).unwrap();

    let magic = read_u32_be(&bytes, 0);

    assert_eq!(magic, 2049);

    let count = read_u32_be(&bytes, 4);

    let labels = &bytes[8..];

    let data = labels
        .iter()
        .map(|&label| label as usize)
        .collect::<Vec<usize>>();

    assert_eq!(data.len(), count as usize);

    data
}
