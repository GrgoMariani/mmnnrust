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
            ActivationFunction::SoftSign => x / (1.0 + x.abs()),  // Changed formula
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
            ActivationFunction::ArcTan => 1.0 / (1.0 + x.powi(2)),  // Corrected
            ActivationFunction::Binary => 0_f64,
            ActivationFunction::ISRU => 1.0 / (1.0 + x.powi(2)).powf(1.5),  // Simplified
            ActivationFunction::LeakyReLU => {
                if x >= 0.0 { 1.0 } else { 0.01 }
            },
            ActivationFunction::Linear => 1_f64,
            ActivationFunction::ReLU => {
                if x > 0.0 { 1_f64 } else { 0_f64 }
            }
            ActivationFunction::ELU => {
                if x >= 0.0 { 1.0 } else { 0.1 * std::f64::consts::E.powf(x) }  // Corrected
            }
            ActivationFunction::GELU => {
                let cdf = 0.5
                    * ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))
                        + x / (2.0_f64).sqrt())
                    .tanh()
                    + 0.5;
                0.5 * (1.0 + cdf + x * (1.0 - cdf))
            }
            ActivationFunction::Gaussian => -2.0 * x * self.activation(x),  // Simplified using activation
            ActivationFunction::SoftSign => 1.0 / (1.0 + x.abs()).powi(2),  // Corrected
            ActivationFunction::SoftStep => {
                let fx = self.activation(x);
                fx * (1.0 - fx)  // More readable
            },
            ActivationFunction::TanH => 1.0 - self.activation(x).powi(2),  // Corrected and simplified
            ActivationFunction::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 + x * (1.0 - sigmoid))  // Corrected
            }
        }
    }

    pub fn get_name(&self) -> &'static str {
        match self {
            ActivationFunction::ArcTan => "ARCTAN",
            ActivationFunction::Binary => "Binary",
            ActivationFunction::ISRU => "ISRU",
            ActivationFunction::LeakyReLU => "LeakyReLU",
            ActivationFunction::Linear => "Linear",
            ActivationFunction::ReLU => "ReLU",
            ActivationFunction::ELU => "ELU",
            ActivationFunction::GELU => "GELU",
            ActivationFunction::Gaussian => "Gaussian",
            ActivationFunction::SoftSign => "SoftSign",
            ActivationFunction::SoftStep => "SoftStep",
            ActivationFunction::TanH => "TanH",
            ActivationFunction::Swish => "Swish",
        }
    }
}
