# Examples

Small, runnable demos for Magnetron. Some are pure Magnetron, others use external packages for tokenization, plotting, or downloading weights.

### [GPT-2 Inference](examples/gpt2/)
GPT-2 text generation in Magnetron (KV cache, streaming output). Uses `transformers` + `tiktoken`.

### [Qwen3 Inference](examples/qwen3/)
Qwen3 chat with a CLI and a local HTTP server. Loads weights from Magnetron `.mag` snapshots.

### [Autoencoder](examples/ae/)
Train an autoencoder on an image and visualize reconstruction.

### [Linear Regression](examples/linear_regression/)
Fit a line to noisy 1D data with `Linear` + SGD.

### [XOR](examples/xor/)
Train a tiny MLP to learn XOR.

## Install (optional deps)

If you want to run most examples:

```bash
uv pip install magnetron[examples]
```

Each folder has its own `README.md` with exact run commands and extra dependencies.