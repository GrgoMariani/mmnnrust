use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "mmnn")]
#[command(about = "mmnn - Micro Managed Neural Network
A tool for neural network operations using JSON configurations.
Input layer values are read from stdin as space-separated numbers.
Networks are defined in JSON format with layers, neurons, and weights.
", long_about = None)]
#[command(arg_required_else_help = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(
        arg_required_else_help = true,
        about = "Forward propagate inputs through a neural network.
Reads space-separated input values from stdin and outputs the result of propagation.
Each line of input creates one line of output."
    )]
    Propagate {
        #[arg(help = "JSON file containing network structure, weights, and biases")]
        config_json_path: PathBuf,
    },
    #[command(
        arg_required_else_help = true,
        about = "Train the neural network using supervised learning.
Reads space-separated input values and expected outputs from stdin.
Format: <input values...> | <expected outputs...>
Training continues until EOF or SIGTERM signal."
    )]
    Learn {
        #[arg(help = "JSON file containing initial network structure and weights")]
        config_json_path: PathBuf,
        #[arg(help = "Output file to save the trained network configuration")]
        save_config_json_path: PathBuf,
        #[arg(
            long,
            default_value_t = 1.0,
            help = "Learning rate controlling step size during training (default: 1.0)"
        )]
        learning_rate: f64,
    },
}
