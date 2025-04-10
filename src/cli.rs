use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "mmnn")]
#[command(about = "mmnn - Micro Managed Neural Network
Define your neural network configuration as a JSON object and use this tool to
propagate the inputs or even to learn your network with expected values.
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
        about = "Take a neural network defined as JSON and propagate it"
    )]
    Propagate {
        #[arg(help = "Path to neural network JSON configuration")]
        config_json_path: PathBuf,
    },
    #[command(
        arg_required_else_help = true,
        about = "Take a neural network defined as JSON, learn it and save the output.
        Writing the configuration is done only once the stdin is empty or when SIGTERM
        has been caught"
    )]
    Learn {
        #[arg(help = "Path to neural network JSON configuration")]
        config_json_path: PathBuf,
        #[arg(help = "Path to store learnt neural network JSON configuration")]
        save_config_json_path: PathBuf,
        #[arg(
            long,
            default_value_t = 1.0,
            help = "floating point factor for learning"
        )]
        learning_rate: f64,
    },
}
