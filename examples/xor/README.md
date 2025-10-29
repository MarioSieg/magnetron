# 🧮 XOR Example

This example demonstrates how to train a small neural network with the Magnetron framework to learn the XOR function.

The model uses a minimal fully connected architecture and shows how to define tensors, layers, and optimization in pure Magnetron without external dependencies.

## 📚 Description

- Defines the XOR truth table using `Tensor.of()`
- Builds a simple feedforward neural network with `Linear` and `Tanh` layers
- Trains the model using **mean squared error (MSE)** loss and **stochastic gradient descent (SGD)** optimizer
- Prints predictions after training
- Plots the training loss over time

## 🚀 Usage

Run the example directly:

```bash
python main.py
```

No arguments are required.

## ⚙️ Requirements

```bash
uv pip install magnetron matplotlib
```