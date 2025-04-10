mod cli;
mod network;
mod neurons;

use cli::{Cli, Commands};
use network::NeuralNetwork;
use std::fs;
use std::io::{self, BufRead};
use std::sync::{Arc, Mutex};

use clap::Parser;

fn main() {
    let args = Cli::parse();
    let mut nn: NeuralNetwork;

    match args.command {
        Commands::Propagate { config_json_path } => {
            nn = NeuralNetwork::new(config_json_path).unwrap();
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let values: Vec<f64> = line
                    .unwrap()
                    .trim()
                    .split_whitespace()
                    .map(|x| x.to_string().trim().parse::<f64>().unwrap())
                    .collect();
                nn.propagate(&values);
                nn.print_outputs(false, true);
            }
        }
        Commands::Learn {
            config_json_path,
            save_config_json_path,
            learning_rate,
        } => {
            nn = NeuralNetwork::new(config_json_path).unwrap();
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
                        nn.propagate(&values);
                        nn.print_outputs(true, false);
                        false
                    }
                    false => {
                        let values: Vec<f64> = line
                            .unwrap()
                            .trim()
                            .split_whitespace()
                            .map(|x| x.to_string().trim().parse::<f64>().unwrap())
                            .collect();
                        nn.backpropagate(&values, learning_rate);
                        true
                    }
                }
            }

            let data = nn.print_as_json();
            fs::write(save_config_json_path, data.as_str()).expect("Unable to write file");
        }
    }
}
