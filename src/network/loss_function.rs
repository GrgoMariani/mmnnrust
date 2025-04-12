#[derive(Debug)]
pub enum LossFunction {
    LossSquared,
}

impl LossFunction {
    pub fn new() -> Self {
        LossFunction::LossSquared
    }

    pub fn get_error(&self, out: &Vec<f64>, expected: &Vec<f64>) -> f64 {
        if out.len() != expected.len() {
            panic!(
                "Sizes not matching when calculating error function. {} vs {}",
                out.len(),
                expected.len()
            )
        }
        match self {
            LossFunction::LossSquared => out
                .iter()
                .zip(expected.iter())
                .map(|(x, y)| (x - y).powi(2) )
                .sum(),
        }
    }

    pub fn get_derivative(&self, out: f64, expected: f64) -> f64 {
        match self {
            Self::LossSquared => (out - expected)*2.0,
        }
    }
}
