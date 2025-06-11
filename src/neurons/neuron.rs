use super::ActivationFunction;
use crate::error::NeuralError;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, PartialEq, Eq)]
pub enum NeuronType {
    Input,
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
            id: id.to_owned(),
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
        self.activation.get_name().to_string()
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
        let mut result = HashMap::with_capacity(self.synapses.len());
        for (lneuron, weight) in &self.synapses {
            let neuron_id = match lneuron.try_borrow() {
                Ok(neuron) => neuron.get_id().to_owned(),
                Err(_) => self.get_id().to_owned(),
            };
            result.insert(neuron_id, *weight);
        }
        result
    }

    pub fn connect(&mut self, neuron: Rc<RefCell<Neuron>>, weight: f64) -> Result<(), NeuralError> {
        if self.is_input() {
            return Err(NeuralError::NeuronError(format!(
                "Cannot use input neuron '{}' as output to other neurons",
                self.get_id()
            )));
        }
        self.synapses.push((neuron, weight));
        Ok(())
    }

    pub fn calculate_depth(&mut self) -> Result<(), NeuralError> {
        if self.depth != std::u32::MAX {
            return Ok(());
        }
        if self.synapses.is_empty() {
            self.depth = 0;
            return Ok(());
        }
        let result = self
            .synapses
            .iter()
            .map(|(lneuron, _)| match lneuron.try_borrow_mut() {
                Ok(mut neuron) => match neuron.depth {
                    x if x != std::u32::MAX => Some(x + 1),
                    _ => {
                        neuron.calculate_depth().ok()?;
                        Some(neuron.depth + 1)
                    }
                },
                Err(_) => None,
            })
            .filter_map(|x| x)
            .max();

        match result {
            Some(depth) => {
                self.depth = depth;
                Ok(())
            }
            None => Err(NeuralError::NeuronError(format!(
                "Could not calculate depth for neuron '{}'",
                self.get_id()
            ))),
        }
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
        let self_id = self.get_id().to_owned();
        let accumulated_error = *error_map.entry(self_id.clone()).or_insert(0.0);
        let error = accumulated_error * self.activation.derivative(self.last_activation_value);
        let curr_depth = self.depth;

        // Create a vector to store weight updates
        let mut weight_updates = Vec::with_capacity(self.synapses.len());

        // First pass: Calculate all updates without modifying weights
        for (i, (rcneuron, weight)) in self.synapses.iter().enumerate() {
            match rcneuron.try_borrow_mut() {
                Ok(lneuron) => {
                    let activation_value = if lneuron.depth <= curr_depth {
                        lneuron.last_activation_value
                    } else {
                        lneuron.backup_activation_value
                    };
                    let neuron_id = lneuron.get_id().to_owned();
                    let laccumulated = match error_map.get(&neuron_id) {
                        Some(value) => value + accumulated_error * (*weight),
                        None => accumulated_error * (*weight),
                    };
                    error_map.insert(neuron_id, laccumulated);
                    weight_updates.push((i, accumulated_error * learning_rate * activation_value));
                }
                Err(_) => {
                    let laccumulated = match error_map.get(&self_id) {
                        Some(value) => value + accumulated_error * (*weight),
                        None => accumulated_error * (*weight),
                    };
                    error_map.insert(self_id.clone(), laccumulated);
                    weight_updates.push((i, accumulated_error * learning_rate * self.backup_activation_value));
                }
            }
        }

        // Second pass: Apply all weight updates
        for (index, update) in weight_updates {
            self.synapses[index].1 -= update;
        }

        self.bias -= error * learning_rate;
    }
}
