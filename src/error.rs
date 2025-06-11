use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum NeuralError {
    NetworkError(String),
    NeuronError(String),
    IoError(std::io::Error),
    ParseError(String),
}

impl fmt::Display for NeuralError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NeuralError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            NeuralError::NeuronError(msg) => write!(f, "Neuron error: {}", msg),
            NeuralError::IoError(e) => write!(f, "IO error: {}", e),
            NeuralError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl Error for NeuralError {}

impl From<std::io::Error> for NeuralError {
    fn from(err: std::io::Error) -> NeuralError {
        NeuralError::IoError(err)
    }
}
