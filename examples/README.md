# 🧪 Examples
Each example runs without PyTorch, TensorFlow, or NumPy.  
They highlight Magnetron’s native tensor engine, autograd system, neural layers, and optimizers.

### [GPT-2 Inference](examples/gpt2/)
Text generation using pretrained GPT-2 models.  
Shows transformer blocks, KV caching, streaming generation, and Hugging Face weight loading.

### [Autoencoder](examples/ae/)
Image reconstruction from a learned latent space.  
Shows built-in image loading, differentiable layers, and visualization.

### [Linear Regression](examples/linear/)
Fits a straight line to noisy 1D data.  
Shows gradient descent, loss tracking, and convergence.

### [XOR](examples/xor/)
Classic XOR problem learned by a small neural network.  
Shows tensors, nonlinear activations, and manual training loops.

## Requirements

Install the dependencies required for all examples:

```bash
uv pip install magnetron matplotlib tiktoken transformers rich
```

Each subdirectory includes its own `README.md` with usage details and parameters.