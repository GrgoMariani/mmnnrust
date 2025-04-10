#[derive(Debug)]
pub enum ErrorFunction {
    LossSquared,
}

impl ErrorFunction {
    pub fn new() -> Self {
        ErrorFunction::LossSquared
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
            ErrorFunction::LossSquared => out
                .iter()
                .zip(expected.iter())
                .map(|(x, y)| (x - y).powi(2) / 2.0)
                .sum(),
        }
    }

    pub fn get_derivative(&self, out: f64, expected: f64) -> f64 {
        match self {
            Self::LossSquared => out - expected,
        }
    }
}
