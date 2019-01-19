# Multi-task DPP-Net

## Introduction

Train a multi-task DPP-Net agent that can design multi-task models for multiple objectives.

## Usage

### Train

```
python main.py --train
```

Arguments:

 * `--type`: (default: `1`)
   * `1`: Train a multi-task DPP-Net agent for task *i* model.
 * `--data`: (default: `1`)
   * `1`: CIFAR-100
 * `--task`: Task ID (for type `1`) (default: None)
 * `--save`: A flag used to decide whether to save model or not.
 * `--load`: Load a pre-trained model before training. 
 * `--path`: Path (directory) that model and history are saved. (default: `'saved_models/default/'`)
 * `--verbose`: A flag used to decide whether to demonstrate verbose messages or not.

### Evaluate

```
python main.py --eval
```

Arguments:

 * `--type`: (default: `1`)
   * `1`: Evaluate a multi-task DPP-Net agent for task *i* model.
 * `--data`: (default: `1`)
   * `1`: CIFAR-100
 * `--task`: Task ID (for type `1`) (default: None)
 * `--save`: A flag used to decide whether to save evaluation models or not.
 * `--path`: Path (directory) that model and history are saved. (default: `'saved_models/default/'`)
