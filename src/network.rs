use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
pub mod loss_function;

use crate::error::NeuralError;
use crate::neurons::{ActivationFunction, Neuron, NeuronType};
use loss_function::LossFunction;
use serde::{Deserialize, Serialize};

fn default_neuron_activation() -> String {
    "Linear".to_string()
}

fn default_neuron_bias() -> f64 {
    0_f64
}

fn default_empty_synapses() -> HashMap<String, f64> {
    HashMap::new()
}

#[derive(Serialize, Deserialize, Debug)]
struct NeuronDefs {
    #[serde(default = "default_neuron_activation")]
    activation: String,
    #[serde(default = "default_neuron_bias")]
    bias: f64,
    #[serde(default = "default_empty_synapses")]
    synapses: HashMap<String, f64>,
}

#[derive(Serialize, Deserialize, Debug)]
struct ConfigJson {
    inputs: Vec<String>,
    outputs: Vec<String>,
    neurons: HashMap<String, NeuronDefs>,
}

#[derive(Debug)]
pub struct NeuralNetwork {
    inputs: Vec<Rc<RefCell<Neuron>>>,
    outputs: Vec<Rc<RefCell<Neuron>>>,
    neuron_map: HashMap<String, Rc<RefCell<Neuron>>>,
    sorted_neurons: Vec<Rc<RefCell<Neuron>>>,
    loss_function: LossFunction,
}

impl NeuralNetwork {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, NeuralError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let cfg: ConfigJson = serde_json::from_reader(reader)
            .map_err(|e| NeuralError::ParseError(e.to_string()))?;
        
        let mut nn = NeuralNetwork {
            inputs: vec![],
            outputs: vec![],
            neuron_map: HashMap::new(),
            sorted_neurons: vec![],
            loss_function: LossFunction::new(),
        };

        nn.create_inputs(&cfg.inputs);
        for (neuron_name, neuron_defs) in &cfg.neurons {
            let activation = ActivationFunction::new(neuron_defs.activation.as_str());
            nn.create_neuron(neuron_name, activation, neuron_defs.bias)?;
        }
        for (rneuron_name, neuron_defs) in &cfg.neurons {
            for (lneuron_name, &weight) in &neuron_defs.synapses {
                nn.connect_neurons(lneuron_name.as_str(), rneuron_name.as_str(), weight)?;
            }
        }
        nn.create_outputs(&cfg.outputs);
        nn.calculate_depths();
        nn.create_sorted_neuron_list();
        Ok(nn)
    }

    fn create_inputs(&mut self, input_names: &[String]) {
        for id in input_names {
            let neuron = Rc::new(RefCell::new(Neuron::new(
                id,
                NeuronType::Input,
                ActivationFunction::Linear,
                0_f64,
            )));
            self.neuron_map.insert(id.to_owned(), Rc::clone(&neuron));
            self.inputs.push(neuron);
        }
    }

    fn create_outputs(&mut self, output_names: &Vec<String>) {
        for id in output_names {
            let neuron = self
                .neuron_map
                .get(id)
                .expect(format!("Could not find neuron id {}", id).as_str());
            let neuron_copy = Rc::clone(&neuron);
            self.outputs.push(neuron_copy);
        }
    }

    fn create_neuron(&mut self, id: &str, activation: ActivationFunction, bias: f64) -> Result<(), NeuralError> {
        if self.neuron_map.contains_key(id) {
            return Err(NeuralError::NetworkError(
                format!("Neuron id '{}' already taken", id)
            ));
        }
        let neuron = Neuron::new(id, NeuronType::Normal, activation, bias);
        self.neuron_map.insert(id.to_owned(), Rc::new(RefCell::new(neuron)));
        Ok(())
    }

    fn connect_neurons(&self, lneuron_id: &str, rneuron_id: &str, weight: f64) -> Result<(), NeuralError> {
        let lneuron = self.neuron_map.get(lneuron_id)
            .ok_or_else(|| NeuralError::NetworkError(
                format!("Could not find neuron with id '{}'", lneuron_id)
            ))?;
        let rneuron = self.neuron_map.get(rneuron_id)
            .ok_or_else(|| NeuralError::NetworkError(
                format!("Could not find neuron with id '{}'", rneuron_id)
            ))?;
        
        let mut rneuron = rneuron.borrow_mut();
        rneuron.connect(Rc::clone(lneuron), weight)?;
        Ok(())
    }

    fn calculate_depths(&mut self) {
        for (neuron_id, neuron) in self.neuron_map.iter() {
            let mut current_neuron = neuron.borrow_mut();
            let _ = current_neuron.calculate_depth();
            if current_neuron.get_depth() == std::u32::MAX {
                panic!("Neuron id '{}': Could not calculate depth", neuron_id);
            }
        }
    }

    fn create_sorted_neuron_list(&mut self) {
        self.sorted_neurons = self
            .neuron_map
            .iter()
            .map(|(_, neuron)| Rc::clone(&neuron))
            .collect();
        self.sorted_neurons
            .sort_by(|a, b| a.borrow_mut().get_depth().cmp(&b.borrow_mut().get_depth()));
    }

    pub fn print_outputs(&self, print_names: bool, endline: bool) {
        for output in self.outputs.iter() {
            let output_neuron = output.borrow();
            if print_names {
                print!("{}:", output_neuron.get_id());
            }
            print!("{} ", output_neuron.get_activation_value());
        }
        if endline {
            println!();
        }
    }

    pub fn propagate(&mut self, input_values: &Vec<f64>) -> Result<(), String> {
        if input_values.len() != self.inputs.len() {
            return Err(format!(
                "Input sizes do not match. {} vs {}",
                input_values.len(),
                self.inputs.len()
            ));
        }
        for (input_value, neuron) in input_values.iter().zip(self.inputs.iter()) {
            let mut input_neuron = neuron.borrow_mut();
            input_neuron.set_activation_value(*input_value);
        }
        for neuron in self.sorted_neurons.iter() {
            let mut new_neuron = neuron.borrow_mut();
            if !new_neuron.is_input() {
                new_neuron.propagate();
            }
        }
        Ok(())
    }

    pub fn backpropagate(
        &mut self,
        expected_output_values: &Vec<f64>,
        learning_rate: f64,
    ) -> Result<(), String> {
        if expected_output_values.len() != self.outputs.len() {
            return Err(format!(
                "Output sizes do not match. {} vs {}",
                expected_output_values.len(),
                self.outputs.len()
            ));
        }
        let output_results: Vec<f64> = self
            .outputs
            .iter()
            .map(|x| x.borrow().get_activation_value())
            .collect();
        let total_error: f64 = self
            .loss_function
            .get_error(&output_results, &expected_output_values);
        println!("[Error: {}]", total_error);
        let mut error_map: HashMap<String, f64> = HashMap::new();

        for (out_neuron, expected) in self.outputs.iter().zip(expected_output_values.iter()) {
            let neuron = out_neuron.borrow_mut();
            let error = self
                .loss_function
                .get_derivative(neuron.get_activation_value(), *expected);
            error_map.insert(neuron.get_id().to_string(), error);
        }
        for item in self.sorted_neurons.iter().rev() {
            let mut neuron = item.borrow_mut();
            neuron.backpropagate(&mut error_map, learning_rate);
        }
        Ok(())
    }

    pub fn print_as_json(self) -> String {
        let mut final_object = ConfigJson {
            inputs: vec![],
            outputs: vec![],
            neurons: HashMap::new(),
        };
        for neuron in self.inputs.iter() {
            let neuron_name = neuron.borrow().get_id().to_string();
            final_object.inputs.push(neuron_name);
        }
        for neuron in self.outputs.iter() {
            let neuron_name = neuron.borrow().get_id().to_string();
            final_object.outputs.push(neuron_name);
        }
        for neuron in self.sorted_neurons.iter() {
            let neuron = neuron.borrow();
            if neuron.is_input() {
                continue;
            }
            let neuron_id = neuron.get_id().to_string();
            let activation = neuron.get_activation_name();
            let bias = neuron.get_bias();
            let synapses: HashMap<String, f64> = neuron.get_synapses_map();
            let neurondefs = NeuronDefs {
                activation,
                bias,
                synapses,
            };
            final_object.neurons.insert(neuron_id, neurondefs);
        }
        serde_json::to_string_pretty(&final_object).expect("Could not serialize the network")
    }

    #[allow(dead_code)]
    pub fn print_by_depth(&self) {
        let mut line_no = 0;
        print!("{}:  ", line_no);
        for item in self.sorted_neurons.iter() {
            let neuron = item.borrow_mut();
            if neuron.get_depth() != line_no {
                println!("");
                line_no = neuron.get_depth();
                print!("{}:  ", line_no);
            }
            print!("{}  ", neuron.get_id());
        }
        println!("");
    }
}
