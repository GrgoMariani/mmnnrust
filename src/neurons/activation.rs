#[derive(Debug)]
pub enum ActivationFunction {
    ArcTan,
    Binary,
    ISRU,
    LeakyReLU,
    Linear,
    ReLU,
    ELU,
    GELU,
    Gaussian,
    SoftSign,
    SoftStep,
    TanH,
    Swish,
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
            "elu" => ActivationFunction::ELU,
            "gelu" => ActivationFunction::GELU,
            "gaussian" => ActivationFunction::Gaussian,
            "softsign" => ActivationFunction::SoftSign,
            "sigmoid" => ActivationFunction::SoftStep,
            "softstep" => ActivationFunction::SoftStep,
            "tanh" => ActivationFunction::TanH,
            "swish" => ActivationFunction::Swish,
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
            ActivationFunction::ELU => {
                if x >= 0.0 {
                    x
                } else {
                    0.1 * (std::f64::consts::E.powf(x) - 1.0)
                }
            }
            ActivationFunction::GELU => {
                0.5 * x
                    * (1.0
                        + ((2.0 / std::f64::consts::PI).sqrt() * (x.powi(3) * 0.044715 + x)).tanh())
            }
            ActivationFunction::Gaussian => std::f64::consts::E.powf(-x.powi(2)),
            ActivationFunction::SoftSign => x * (1_f64 + x.abs()),
            ActivationFunction::SoftStep => 1_f64 / (1_f64 + std::f64::consts::E.powf(-x)), //1/(1+e^-x)
            ActivationFunction::TanH => {
                let ex = std::f64::consts::E.powf(x);
                let exc = std::f64::consts::E.powf(-x);
                (ex - exc) / (ex + exc)
            }
            ActivationFunction::Swish => {
                let exc = std::f64::consts::E.powf(-x);
                x * (1.0 - exc)
            }
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::ArcTan => 1_f64 / x.tan().powi(2) + 1_f64,
            ActivationFunction::Binary => 0_f64,
            ActivationFunction::ISRU => {
                let inverse = x / (1.0 + x.powi(2)).sqrt();
                (1_f64 / (1_f64 + inverse.powi(2)).sqrt()).powi(3)
            }
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
            ActivationFunction::ELU => {
                if x >= 0.0 {
                    1.0
                } else {
                    self.activation(x) + 0.1
                }
            }
            ActivationFunction::GELU => {
                let cdf = 0.5
                    * ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))
                        + x / (2.0_f64).sqrt())
                    .tanh()
                    + 0.5;
                0.5 * (1.0 + cdf + x * (1.0 - cdf))
            }
            ActivationFunction::Gaussian => -2.0 * x * std::f64::consts::E.powf(-x.powi(2)),
            ActivationFunction::SoftSign => {
                let inverse = x.tan() * 2.0 / std::f64::consts::PI; // approximation
                1_f64 / (1_f64 + inverse.powi(2))
            }
            ActivationFunction::SoftStep => self.activation(x) * (1_f64 - self.activation(x)),
            ActivationFunction::TanH => {
                let ex = std::f64::consts::E.powf(x);
                let exc = std::f64::consts::E.powf(-x);
                4.0 / (ex + exc).powi(2)
            }
            ActivationFunction::Swish => {
                let ex = std::f64::consts::E.powf(x);
                let exc = std::f64::consts::E.powf(-x);
                exc / (-x + ex + 1.0)
            }
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
            ActivationFunction::ELU => String::from("ELU"),
            ActivationFunction::GELU => String::from("GELU"),
            ActivationFunction::Gaussian => String::from("Gaussian"),
            ActivationFunction::SoftSign => String::from("SoftSign"),
            ActivationFunction::SoftStep => String::from("SoftStep"),
            ActivationFunction::TanH => String::from("TanH"),
            ActivationFunction::Swish => String::from("Swish"),
        }
    }
}
