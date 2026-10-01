pub(crate) fn unravel_index(mut offset: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];

    for i in (0..shape.len()).rev() {
        index[i] = offset % shape[i];
        offset /= shape[i];
    }

    index
}

pub(crate) fn broadcast_offset(output_index: &[usize], input_shape: &[usize]) -> usize {
    let shift = output_index.len() - input_shape.len();

    let mut offset = 0;

    for i in 0..input_shape.len() {
        let dim = input_shape[i];
        let output_i = output_index[shift + i];
        let input_i = if dim == 1 { 0 } else { output_i };

        offset = offset * dim + input_i;
    }

    offset
}
