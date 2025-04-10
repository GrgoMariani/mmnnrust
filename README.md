# mmnn - Micro Managed Neural Network
rust based bash-cli for Neural Network propagation/backpropagation

# Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Features](#features)
   - [Basic Usage](#basic-usage)
      - [Propagate](#propagation)
      - [Learn](#learning)
   - [Advanced Usage](#advanced-usage)
3. [Examples](#examples)

## Introduction

A **bash-cli** Micro Managed Neural Network. Write your neural network configuration in JSON format and use this tool to run it.

**mmnn** takes input values from the stdin and outputs them to stdout so you can design your scripts to pipe the neuron input/output values to/from it.

The idea behind **mmnn** is that most of the frameworks for Neural Networks are connecting huge layers which makes it hard to analyze individual neuron connections. Additionally, direct neuron connections are only possible between consecutively connected layers. I wanted to change this and this is the result.

Take, for example, the following neural network, which is possible to be defined, propagated and trained with **mmnn**.

```mermaid
---
config:
  look: handDrawn
---
stateDiagram-v2
direction LR
classDef inputNeuron fill:#d7ece9,color:#000,font-weight:bold,stroke-width:2px,stroke:yellow
classDef outputNeuron fill:#cf388d,color:white,font-weight:bold,stroke-width:2px,stroke:yellow
state "INPUT1" as i1
state "INPUT2" as i2
state "INPUT3" as i3
state "INPUT4" as i4
state "A" as a
state "B" as b
state "C" as c
state "D" as d
state "OUTPUT1" as o1
state "OUTPUT2" as o2
    i1 --> a: 3.2
    i2 --> b: 2.4
    i2 --> c: -2.3
    i3 --> b: 1.2
    i4 --> c: -4.2
    a --> d: 4.3
    b --> d: 7.1
    c --> d: 1.55
    i1 --> d: 0.2
    b --> o1: -1.26
    d --> o1: 3.2
    c --> o2: 0.58
    b --> o2: -8.4
    i4 --> o1: 0.92

class i1 inputNeuron
class i2 inputNeuron
class i3 inputNeuron
class i4 inputNeuron
class o1 outputNeuron
class o2 outputNeuron

```

## Getting Started

### Installation

You can choose between two options:

<table>
<tr> <td> <b>Install cargo</b> </td>
<td> <b>Build Manually</b> </td>
</tr>
<tr>
<td>

```bash
$ cargo install mmnn
$ mmnn --help
```

</td>
<td>

```bash
$ git clone git@github.com:GrgoMariani/mmnn
$ cd mmnn
$ cargo run -- --help
```

</td>
</tr>
</table>

### Features

* JSON network configuration
* Forward propagation
* Backward propagation
* Recursive connections between neurons possible (more on that later)
* Different activations
  * ArcTan
  * Binary
  * ISRU
  * LeakyReLU
  * Linear
  * ReLU
  * SoftSign
  * SoftStep
  * TanH

### Basic Usage

#### Propagate

Take this network for example:

```mermaid
---
config:
  look: handDrawn
---
stateDiagram-v2
direction LR
state "INPUT" as i1
state "OUTPUT (ReLU)" as o1
state "bias" as b
    i1 --> o1: 3.2
    b --> o1: 0.21
```

This network consists of only one input neuron, one output neuron and a bias.

The equivalent configuration mmnn would use for this would be:

```json
{
    "inputs": ["INPUT"],
    "outputs": ["OUTPUT"],
    "neurons": {
        "OUTPUT": {
            "bias": 0.21,
            "activation": "ReLU",
            "synapses": {
                "INPUT": 3.2
            }
        }
    }
}
```

> INFO: Notice how the synapses are defined right-to-left. i.e. previous neuron results are arguments for the next neuron.

If we save this configuration as **config.json** we could propagate it like so:
```bash
$ mmnn propagate config.json
> 1
> 3.41
> 2
> 6.61
```

The propagation is done through the standard input where each line represents input values to the neurons.

Read the rest of this README for more configuration examples.

#### Learn

Learning is done similar to propagation

```bash
$ mmnn learn config.json saved_config.json --learning-rate 0.5
```

Note that each odd input line is propagation step while the propagation step is each even one. Learned configuration is saved once all stdin lines are read or when SIGTERM is caught.

The output is also somewhat different as it shows the neuron names and calculated error output.


### Advanced Usage

Here is a simple [Flip Flop](https://en.wikipedia.org/wiki/Flip-flop_(electronics))-like neural network which makes use of recursive neuron connections.
```mermaid
---
config:
  look: handDrawn
---
stateDiagram-v2
direction LR
classDef inputNeuron fill:#d7ece9,color:#000,font-weight:bold,stroke-width:2px,stroke:yellow
classDef outputNeuron fill:#cf388d,color:white,font-weight:bold,stroke-width:2px,stroke:yellow
state "INPUT 1" as i1
state "INPUT 2" as i2
state "OUTPUT Q
(ReLU)" as o1
state "OUTPUT !Q
(ReLU)" as o2
state "bias" as b
    i1 --> o1: -1
    i2 --> o2: -1
    o1 --> o2: -1
    o2 --> o1: -1
    b --> o1: 1
    b --> o2: 1
class i1 inputNeuron
class i2 inputNeuron
class o1 outputNeuron
class o2 outputNeuron
```

```bash
$ mmnn propagate <(cat <<-EOL
    {
        "inputs": ["i1", "i2"],
        "outputs": ["Q", "!Q"],
        "neurons": {
            "Q": {
                "bias": 1,
                "activation": "ReLU",
                "synapses": {
                    "!Q": -1,
                    "i1": -1
                }
            },
            "!Q": {
                "bias": 1,
                "activation": "ReLU",
                "synapses": {
                    "Q": -1,
                    "i2": -1
                }
            }
        }
    }
EOL
)
```

Playing around with the input values should showcase how memory is retained.

## Examples

By design this cargo package is a bash command line interface so bash can be utilized in full to create your propagation/training data.

For example:

```bash
#!/bin/bash

function create_training_data() {
    local iterations="${1}"
    local i1 i2 AND OR NOR
    for ((i=0; i<iterations; i++)); do
        i1=$((RANDOM%2))
        i2=$((RANDOM%2))
        AND=$((i1 & i2))
        OR=$((i1 | i2))
        NOR=$((!i1 & !i2))
        # odd lines for forward propagation
        echo "${i1} ${i2}"
        # even lines for backpropagation
        echo "${AND} ${OR} ${NOR}"
    done
}

TRAIN_DATA_FILE=$(mktemp)

echo "Creating the training data"
create_training_data 200000 > "${TRAIN_DATA_FILE}"

# train the model
echo "Training the model"
cat "${TRAIN_DATA_FILE}" | mmnn learn <(cat <<-EOL
    {
        "inputs": ["i1", "i2"],
        "outputs": ["AND", "OR", "NOR"],
        "neurons": {
            "AND": {
                "bias": 0.5,
                "activation": "relu",
                "synapses": {
                    "i1": 1,
                    "i2": 3
                }
            },
            "OR": {
                "bias": 2,
                "activation": "relu",
                "synapses": {
                    "i1": 3.2,
                    "i2": -1
                }
            },
            "NOR": {
                "bias": 1,
                "activation": "relu",
                "synapses": {
                    "i1": 2.2,
                    "i2": -1.1
                }
            }
        }
    }
EOL
) config_save.json --learning-rate 0.05
echo "Learning done!"
# try the saved model
mmnn propagate config_save.json
```

Have fun playing around with this tool!
