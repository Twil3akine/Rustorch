use crate::tensor::*;
use rand::seq::SliceRandom;

pub trait Dataset {
    type Target;

    fn len(&self) -> usize;

    fn batch(&self, indices: &[usize]) -> Option<(Tensor, Self::Target)>;
}

pub struct TensorDataset {
    inputs: Tensor,
    targets: Tensor,
}

impl TensorDataset {
    pub fn new(inputs: Tensor, targets: Tensor) -> Option<Self> {
        if inputs.ndim() != 2 || targets.ndim() != 2 {
            return None;
        }

        if inputs.shape()[0] != targets.shape()[0] {
            return None;
        }

        Some(Self { inputs, targets })
    }
}

impl Dataset for TensorDataset {
    type Target = Tensor;

    fn len(&self) -> usize {
        self.inputs.shape()[0]
    }

    fn batch(&self, indices: &[usize]) -> Option<(Tensor, Self::Target)> {
        let inputs = self.inputs.take_rows(indices)?;
        let targets = self.targets.take_rows(indices)?;

        Some((inputs, targets))
    }
}

pub struct ClassificationDataset {
    inputs: Tensor,
    targets: Vec<usize>,
}

impl ClassificationDataset {
    pub fn new(inputs: Tensor, targets: Vec<usize>) -> Option<Self> {
        if inputs.ndim() != 2 {
            return None;
        }

        if inputs.shape()[0] != targets.len() {
            return None;
        }

        Some(Self { inputs, targets })
    }
}

impl Dataset for ClassificationDataset {
    type Target = Vec<usize>;

    fn len(&self) -> usize {
        self.inputs.shape()[0]
    }

    fn batch(&self, indices: &[usize]) -> Option<(Tensor, Self::Target)> {
        let inputs = self.inputs.take_rows(indices)?;
        let targets = indices.iter().map(|&i| self.targets[i]).collect::<Vec<_>>();

        Some((inputs, targets))
    }
}

pub struct DataLoader<'a, D: Dataset> {
    dataset: &'a D,
    batch_size: usize,
    indices: Vec<usize>,
    position: usize,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        assert!(batch_size > 0);

        let mut indices = (0..dataset.len()).collect::<Vec<usize>>();

        if shuffle {
            let mut rng = rand::rng();
            indices.shuffle(&mut rng);
        }

        Self {
            dataset,
            batch_size,
            indices,
            position: 0usize,
        }
    }
}

impl<D: Dataset> Iterator for DataLoader<'_, D> {
    type Item = (Tensor, D::Target);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }

        let start = self.position;
        let end = (self.position + self.batch_size).min(self.dataset.len());

        let batch_indices = &self.indices[start..end];
        let batch = self.dataset.batch(batch_indices);

        self.position = end;

        batch
    }
}

#[test]
fn dataloader() {
    let inputs = Tensor::new(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], vec![5, 2]).unwrap();

    let targets = Tensor::new(vec![1., 2., 3., 4., 5.], vec![5, 1]).unwrap();

    let dataset = TensorDataset::new(inputs, targets).unwrap();
    let mut loader = DataLoader::new(&dataset, 2, false);

    let (x1, y1) = loader.next().unwrap();
    assert_eq!(x1.shape(), vec![2, 2]);
    assert_eq!(y1.shape(), vec![2, 1]);

    let (x2, y2) = loader.next().unwrap();
    assert_eq!(x2.shape(), vec![2, 2]);
    assert_eq!(y2.shape(), vec![2, 1]);

    let (x3, y3) = loader.next().unwrap();
    assert_eq!(x3.shape(), vec![1, 2]);
    assert_eq!(y3.shape(), vec![1, 1]);

    assert!(loader.next().is_none());
}

#[test]
fn dataloader_with_shuffle() {
    let inputs = Tensor::new(vec![1., 2., 3., 4.], vec![4, 1]).unwrap();

    let targets = Tensor::new(vec![10., 20., 30., 40.], vec![4, 1]).unwrap();

    let dataset = TensorDataset::new(inputs, targets).unwrap();
    let loader = DataLoader::new(&dataset, 2, true);

    let mut pairs = Vec::new();

    for (inputs, targets) in loader {
        for (input, target) in inputs.data().into_iter().zip(targets.data()) {
            pairs.push((input as i32, target as i32));
        }
    }

    pairs.sort();

    assert_eq!(pairs, vec![(1, 10), (2, 20), (3, 30), (4, 40),]);
}

#[test]
fn classification_dataloader() {
    let inputs = Tensor::new(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![4, 2]).unwrap();

    let targets = vec![3, 1, 0, 2];

    let dataset = ClassificationDataset::new(inputs, targets).unwrap();

    let mut loader = DataLoader::new(&dataset, 2, false);

    let (inputs, targets) = loader.next().unwrap();

    assert_eq!(inputs.data(), vec![1., 2., 3., 4.]);
    assert_eq!(targets, vec![3, 1]);

    let (inputs, targets) = loader.next().unwrap();

    assert_eq!(inputs.data(), vec![5., 6., 7., 8.]);
    assert_eq!(targets, vec![0, 2]);

    assert!(loader.next().is_none());
}
