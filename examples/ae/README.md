# 🧠 Autoencoder Example

This example demonstrates a simple image autoencoder built with the Magnetron framework.

The model learns to compress and reconstruct an input image through a small fully connected encoder–decoder network.

## 📚 Description

- Loads an RGB image and resizes it using Magnetron’s built-in image loader (no external libraries required)
- Trains the model using mean squared error (MSE) loss
- Optimized with the **Adam** optimizer
- Reconstructs the image from a low-dimensional latent space
- Plots the reconstruction result and training loss curve

## 🚀 Usage

Run the example directly:

```bash
python main.py
```

All arguments have default values and can be customized if desired.  
Refer to the script source for the list of available options.

## ⚙️ Requirements

```bash
uv pip install magnetron matplotlib
```