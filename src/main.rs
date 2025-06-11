mod cli;
mod error;
mod network;
mod neurons;

use cli::{Cli, Commands};
use network::NeuralNetwork;
use std::fs;
use std::io::{self, BufRead};
use std::sync::{Arc, Mutex};

use clap::Parser;

use crate::error::NeuralError;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    
    match args.command {
        Commands::Propagate { config_json_path } => {
            let mut nn = NeuralNetwork::new(config_json_path)?;
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let line = line?;
                let values: Result<Vec<f64>, _> = line
                    .trim()
                    .split_whitespace()
                    .map(|x| x.parse::<f64>())
                    .collect();
                
                let values = values.map_err(|e| NeuralError::ParseError(e.to_string()))?;
                
                if let Err(e) = nn.propagate(&values) {
                    eprintln!("Error: {}", e);
                    continue;
                }
                nn.print_outputs(false, true);
            }
        }
        Commands::Learn {
            config_json_path,
            save_config_json_path,
            learning_rate,
        } => {
            let mut nn = NeuralNetwork::new(config_json_path)?;
            let stdin = io::stdin();
            let mut propagate = true;

            let caught_sigterm: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
            let caught_sigterm_rc = Arc::clone(&caught_sigterm);

            ctrlc::set_handler(move || {
                if *caught_sigterm_rc.lock().unwrap() == true {
                    eprintln!("Goodbye!");
                    std::process::exit(1);
                }
                eprintln!("SIGTERM caught, exiting on next line iteration.");
                *caught_sigterm_rc.lock().unwrap() = true;
            })
            .expect("Error setting Ctrl-C handler");

            for line in stdin.lock().lines() {
                if *caught_sigterm.lock().unwrap() == true {
                    break;
                }
                propagate = match propagate {
                    true => {
                        let values: Vec<f64> = line
                            .unwrap()
                            .trim()
                            .split_whitespace()
                            .map(|x| x.to_string().trim().parse::<f64>().unwrap())
                            .collect();
                        match nn.propagate(&values) {
                            Ok(_) => {
                                nn.print_outputs(true, false);
                                false
                            },
                            Err(msg) => {
                                eprintln!("Propagation failed with message: '{}'", msg);
                                true
                            }
                        }
                    }
                    false => {
                        let values: Vec<f64> = line
                            .unwrap()
                            .trim()
                            .split_whitespace()
                            .map(|x| x.to_string().trim().parse::<f64>().unwrap())
                            .collect();
                        match nn.backpropagate(&values, learning_rate) {
                            Ok(_) => {
                                true
                            }
                            Err(msg) => {
                                eprintln!("Backpropagation failed with message: '{}'", msg);
                                false
                            }
                        }
                    }
                }
            }

            let data = nn.print_as_json();
            fs::write(save_config_json_path, data.as_str()).expect("Unable to write file");
        }
    }
    Ok(())
}
