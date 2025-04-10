use super::ActivationFunction;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, PartialEq, Eq)]
pub enum NeuronType {
    Input,
    Output,
    Normal,
}

#[derive(Debug)]
pub struct Neuron {
    id: String,
    ntype: NeuronType,
    synapses: Vec<(Rc<RefCell<Neuron>>, f64)>,
    activation: ActivationFunction,
    bias: f64,
    depth: u32,
    last_activation_value: f64,
    backup_activation_value: f64,
}

impl Neuron {
    pub fn new(id: &str, ntype: NeuronType, activation: ActivationFunction, bias: f64) -> Self {
        Neuron {
            id: String::from(id),
            ntype,
            synapses: vec![],
            activation,
            bias,
            depth: std::u32::MAX,
            last_activation_value: 0.0,
            backup_activation_value: 0.0,
        }
    }

    pub fn get_id(&self) -> &str {
        self.id.as_str()
    }

    pub fn get_depth(&self) -> u32 {
        self.depth
    }

    pub fn is_input(&self) -> bool {
        self.ntype == NeuronType::Input
    }

    pub fn get_activation_name(&self) -> String {
        self.activation.get_name()
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    pub fn get_activation_value(&self) -> f64 {
        self.last_activation_value
    }

    pub fn set_activation_value(&mut self, value: f64) {
        self.last_activation_value = value;
    }

    pub fn get_synapses_map(&self) -> HashMap<String, f64> {
        let mut result: HashMap<String, f64> = HashMap::new();
        for (lneuron, weight) in self.synapses.iter() {
            let neuron_id = match lneuron.try_borrow() {
                Ok(neuron) => neuron.get_id().to_string(),
                Err(_) => self.get_id().to_string(),
            };
            result.insert(neuron_id, *weight);
        }
        result
    }

    pub fn set_activation_bias(&mut self, activation: ActivationFunction, bias: f64) {
        self.activation = activation;
        self.bias = bias;
    }

    pub fn connect(&mut self, neuron: Rc<RefCell<Neuron>>, weight: f64) {
        self.synapses.push((neuron, weight));
    }

    pub fn calculate_depth(&mut self) {
        if self.depth != std::u32::MAX {
            return;
        }
        if self.synapses.is_empty() {
            self.depth = 0;
            return;
        }
        let result = self
            .synapses
            .iter()
            .map(|(lneuron, _)| match lneuron.try_borrow_mut() {
                Ok(mut neuron) => match neuron.depth {
                    x if x != std::u32::MAX => Some(x + 1),
                    _ => {
                        neuron.calculate_depth();
                        match neuron.depth {
                            std::u32::MAX => None,
                            _ => Some(neuron.depth + 1),
                        }
                    }
                },
                Err(_) => None,
            })
            .filter_map(|x| x)
            .max();
        match result {
            Some(depth) => self.depth = depth,
            None => {}
        };
    }

    pub fn propagate(&mut self) {
        let sum_activations: f64 = self
            .synapses
            .iter()
            .map(|(lneuron, weight)| match lneuron.try_borrow_mut() {
                Ok(neuron) => weight * neuron.last_activation_value,
                Err(_) => weight * self.last_activation_value,
            })
            .sum();
        // used for recursive cases backpropagation
        self.backup_activation_value = self.last_activation_value;
        self.last_activation_value = self.activation.activation(sum_activations + self.bias);
    }

    pub fn backpropagate(&mut self, error_map: &mut HashMap<String, f64>, learning_rate: f64) {
        let copy_self_id = self.get_id().to_string();
        let accumulated_error = *error_map.entry(self.get_id().to_string()).or_insert(0.0);
        let error = accumulated_error * self.activation.derivative(self.last_activation_value);
        let curr_depth = self.depth;
        for (rcneuron, ref mut weight) in self.synapses.iter_mut() {
            let self_id = copy_self_id.as_str().to_string();
            match rcneuron.try_borrow_mut() {
                Ok(lneuron) => {
                    let activation_value = if lneuron.depth <= curr_depth {
                        lneuron.last_activation_value
                    } else {
                        lneuron.backup_activation_value
                    };
                    let laccumulated = match error_map.get(&lneuron.get_id().to_string()) {
                        Some(value) => value + accumulated_error * (*weight),
                        None => accumulated_error * (*weight),
                    };
                    error_map.insert(lneuron.get_id().to_string(), laccumulated);
                    *weight -= accumulated_error * learning_rate * activation_value;
                }
                Err(_) => {
                    // recursive case
                    let laccumulated = match error_map.get(&self_id) {
                        Some(value) => value + accumulated_error * (*weight),
                        None => accumulated_error * (*weight),
                    };
                    error_map.insert(self_id, laccumulated);
                    let new_weight =
                        *weight - accumulated_error * learning_rate * self.backup_activation_value;
                    *weight = new_weight;
                }
            }
        }
        self.bias -= error * learning_rate;
    }
}
