#[derive(Debug)]
pub enum ActivationFunction {
    ArcTan,
    Binary,
    ISRU,
    LeakyReLU,
    Linear,
    ReLU,
    SoftSign,
    SotfStep,
    TanH,
}

impl ActivationFunction {
    pub fn new(name: &str) -> ActivationFunction {
        match name.to_lowercase().as_str() {
            "arctan" => ActivationFunction::ArcTan,
            "binary" => ActivationFunction::Binary,
            "isru" => ActivationFunction::ISRU,
            "leakyrelu" => ActivationFunction::LeakyReLU,
            "linear" => ActivationFunction::Linear,
            "relu" => ActivationFunction::ReLU,
            "softsign" => ActivationFunction::SoftSign,
            "softstep" => ActivationFunction::SotfStep,
            "tanh" => ActivationFunction::TanH,
            _ => panic!("Unknown activation function '{}'", name),
        }
    }

    pub fn activation(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::ArcTan => x.atan(),
            ActivationFunction::Binary => {
                if x > 0.0 {
                    1_f64
                } else {
                    0_f64
                }
            }
            ActivationFunction::ISRU => x / (1.0 + x.powi(2)).sqrt(),
            ActivationFunction::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            ActivationFunction::Linear => x,
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0_f64
                }
            }
            ActivationFunction::SoftSign => x * (1_f64 + x.abs()),
            ActivationFunction::SotfStep => 1_f64 / (1_f64 + std::f64::consts::E.powf(-x)), //1/(1+e^-x)
            ActivationFunction::TanH => {
                2_f64 / (1_f64 + std::f64::consts::E.powf(-2.0 * x)) - 1_f64
            }
        }
    }

    pub fn inverse(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::ArcTan => x.tan(),
            ActivationFunction::Binary => {
                if x > 0.0 {
                    1_f64
                } else {
                    0_f64
                }
            }
            ActivationFunction::ISRU => {
                if x == 1.0_f64 {
                    1_000_000_f64
                } else {
                    x / (1_f64 - x.powi(2))
                }
            }
            ActivationFunction::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    100.0 * x
                }
            }
            ActivationFunction::Linear => x,
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0_f64
                }
            }
            ActivationFunction::SoftSign => x.tan() * 2.0 / std::f64::consts::PI, // approximation
            ActivationFunction::SotfStep => {
                if x <= 0.0 {
                    -1_000_000_f64
                } else {
                    -(1_f64 / (x + 1_f64)).ln()
                }
            }
            ActivationFunction::TanH => ((1_f64 + x) * (1_f64 - x)).ln() / 2.0,
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::ArcTan => 1_f64 / x.tan().powi(2) + 1_f64,
            ActivationFunction::Binary => 0_f64,
            ActivationFunction::ISRU => (1_f64 / (1_f64 + self.inverse(x).powi(2)).sqrt()).powi(3),
            ActivationFunction::LeakyReLU => {
                if x >= 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            ActivationFunction::Linear => 1_f64,
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1_f64
                } else {
                    0_f64
                }
            }
            ActivationFunction::SoftSign => 1_f64 / (1_f64 + self.inverse(x).powi(2)),
            ActivationFunction::SotfStep => x * (1_f64 - x),
            ActivationFunction::TanH => 1_f64 - x.powi(2),
        }
    }

    pub fn get_name(&self) -> String {
        match self {
            ActivationFunction::ArcTan => String::from("ARCTAN"),
            ActivationFunction::Binary => String::from("Binary"),
            ActivationFunction::ISRU => String::from("ISRU"),
            ActivationFunction::LeakyReLU => String::from("LeakyReLU"),
            ActivationFunction::Linear => String::from("Linear"),
            ActivationFunction::ReLU => String::from("ReLU"),
            ActivationFunction::SoftSign => String::from("SoftSign"),
            ActivationFunction::SotfStep => String::from("SoftStep"),
            ActivationFunction::TanH => String::from("TanH"),
        }
    }
}
