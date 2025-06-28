#[derive(Debug)]
pub enum ActivationFunction {
    Identity,
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
    Sinusoid,
    ELiSH,
}

impl ActivationFunction {
    pub fn new(name: &str) -> ActivationFunction {
        match name.to_lowercase().as_str() {
            "identity" => Self::Identity,
            "arctan" => Self::ArcTan,
            "binary" => Self::Binary,
            "isru" => Self::ISRU,
            "leakyrelu" => Self::LeakyReLU,
            "linear" => Self::Linear,
            "relu" => Self::ReLU,
            "elu" => Self::ELU,
            "gelu" => Self::GELU,
            "gaussian" => Self::Gaussian,
            "softsign" => Self::SoftSign,
            "sigmoid" => Self::SoftStep,
            "softstep" => Self::SoftStep,
            "tanh" => Self::TanH,
            "swish" => Self::Swish,
            "sinusoid" => Self::Sinusoid,
            "elish" => Self::ELiSH,
            _ => panic!("Unknown activation function '{}'", name),
        }
    }

    pub fn activation(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Identity => x,
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
            ActivationFunction::SoftSign => x / (1.0 + x.abs()),
            ActivationFunction::SoftStep => 1_f64 / (1_f64 + std::f64::consts::E.powf(-x)),
            ActivationFunction::TanH => {
                let ex = std::f64::consts::E.powf(x);
                let exc = std::f64::consts::E.powf(-x);
                (ex - exc) / (ex + exc)
            }
            ActivationFunction::Swish => {
                let exc = std::f64::consts::E.powf(-x);
                x * (1.0 - exc)
            }
            ActivationFunction::Sinusoid => x.sin(),
            ActivationFunction::ELiSH => {
                if x >= 0.0 {
                    (x) / (1.0 + std::f64::consts::E.powf(-x))
                } else {
                    (std::f64::consts::E.powf(x) - 1.0) / (1.0 + std::f64::consts::E.powf(-x))
                }
            }
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Identity => 1.0,
            ActivationFunction::ArcTan => 1.0 / (1.0 + x.powi(2)),
            ActivationFunction::Binary => 0_f64,
            ActivationFunction::ISRU => 1.0 / (1.0 + x.powi(2)).powf(1.5),
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
                    0.1 * std::f64::consts::E.powf(x)
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
            ActivationFunction::Gaussian => -2.0 * x * self.activation(x),
            ActivationFunction::SoftSign => 1.0 / (1.0 + x.abs()).powi(2),
            ActivationFunction::SoftStep => {
                let fx = self.activation(x);
                fx * (1.0 - fx)
            }
            ActivationFunction::TanH => 1.0 - self.activation(x).powi(2),
            ActivationFunction::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 + x * (1.0 - sigmoid))
            }
            ActivationFunction::Sinusoid => x.cos(),
            ActivationFunction::ELiSH => {
                if x >= 0.0 {
                    (x * std::f64::consts::E.powf(x)
                        + std::f64::consts::E.powf(2.0 * x)
                        + std::f64::consts::E.powf(x))
                        / (std::f64::consts::E.powf(2.0 * x) + 2.0 * std::f64::consts::E.powf(x) + 1.0)
                } else {
                    (2.0 * std::f64::consts::E.powf(2.0 * x) + std::f64::consts::E.powf(3.0 * x)
                        - std::f64::consts::E.powf(x))
                        / (std::f64::consts::E.powf(2.0 * x) + 2.0 * std::f64::consts::E.powf(x) + 1.0)
                }
            }
        }
    }

    pub fn get_name(&self) -> &'static str {
        match self {
            Self::Identity => "Identity",
            Self::ArcTan => "ARCTAN",
            Self::Binary => "Binary",
            Self::ISRU => "ISRU",
            Self::LeakyReLU => "LeakyReLU",
            Self::Linear => "Linear",
            Self::ReLU => "ReLU",
            Self::ELU => "ELU",
            Self::GELU => "GELU",
            Self::Gaussian => "Gaussian",
            Self::SoftSign => "SoftSign",
            Self::SoftStep => "SoftStep",
            Self::TanH => "TanH",
            Self::Swish => "Swish",
            Self::Sinusoid => "Sinusoid",
            Self::ELiSH => "ELiSH",
        }
    }
}
