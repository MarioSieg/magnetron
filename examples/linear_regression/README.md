# 📈 Linear Regression Example

This example demonstrates a simple linear regression model implemented with the Magnetron framework.

The model learns a straight-line relationship between input `x` and target `y` using a single `Linear` layer and mean squared error (MSE) loss.

## 📚 Description

- Generates synthetic 1D data with added Gaussian noise
- Builds a single-layer linear regression model
- Trains the model using **stochastic gradient descent (SGD)**
- Plots the fitted line and the training loss curve

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