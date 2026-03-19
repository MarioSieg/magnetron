[![Stargazers][stars-shield]][stars-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
![GitHub Actions Workflow Status][ci-shield]

<br />
<div align="center">
  <a href="https://github.com/MarioSieg/magnetron">
    <img src="https://raw.githubusercontent.com/MarioSieg/magnetron/develop/media/logo.png" alt="Magnetron Logo" width="200" height="200">
  </a>

<h3 align="center">magnetron</h3>
  <p align="center">
    A compact machine learning runtime for developers who want to understand, control, and optimize the full stack.
    <br />
    Native C core, modern Python API, no runtime dependencies, no bloat.
    <br /><br />
    <a href="https://github.com/MarioSieg/magnetron/blob/master/docs/Magnetron-Cheatsheet.md"><strong>Documentation »</strong></a>
    <br /><br />
    <a href="https://github.com/MarioSieg/magnetron/blob/master/examples/qwen3">Qwen3 Inference Example</a>
    ·
    <a href="https://github.com/MarioSieg/magnetron/blob/master/examples/ae">Autoencoder Training Example</a>
    ·
    <a href="https://github.com/MarioSieg/magnetron/blob/master/examples/gpt2">GPT-2 Inference Example</a>
  </p>
</div>

---

## About

**Magnetron** is a machine learning runtime built from scratch in C, with a small modern Python interface for usability.  
It implements its own tensor system, operator set, autograd engine, and execution model - without relying on large external frameworks.

The goal is simple:

> Keep the stack small enough to understand and be hackable, but powerful enough to run real models.

This makes Magnetron useful in two situations:
- when you want **full control over execution and memory**
- when you want a **clean base for experimentation or new ideas**

---

## Why Magnetron?

Magnetron is not trying to compete with PyTorch on ecosystem or feature count.

Instead, it optimizes for a different axis:

| Magnetron | PyTorch |
|--------|------|
| Small, inspectable core | Large, layered system |
| Explicit execution | Implicit / abstracted |
| Minimal dependencies | Heavy runtime |
| Easy to modify kernels | Harder to reason about backend |
| Good for research & systems work | Good for production & scale |

If you want to:
- understand how your model actually runs
- experiment with kernels, memory layouts, or execution
- port ML workloads to unusual hardware

Magnetron gives you a much shorter path.

---

## Architecture Overview

Magnetron is built as a **single, cohesive runtime**, not a collection of loosely coupled libraries.

- **Tensor system**  
  Owns dtype, shape, strides, and memory – supports a full **view system with a view solver**, enabling complex slicing, reshaping, and broadcasting semantics similar to PyTorch while remaining explicit and predictable.

- **Execution model**  
  Eager execution with a dynamic autograd graph (reverse-mode), constructed per forward pass and traversed during backward.

- **Operator backend**  
  Central dispatch layer mapping high-level operations to architecture-specific kernel implementations.

- **CPU backend**  
  Multi-dispatch design with **compile-time optimized kernels** for a wide range of microarchitectures (Intel, AMD Zen1–Zen5, ARM).  
  At runtime, **CPUID-based detection** selects the most optimal kernel path automatically.  
  Supports multiple SIMD ISAs and extensions, including SSE (1–4), AVX, AVX2, FMA, AVX-512, AVX-512-BF16, AVX-512-FP16, F16C and ARM NEON, combined with multithreaded execution.

- **CUDA backend (in progress)**  
  Kernel layer is implemented - Memory management, execution pipeline, and integration are actively being completed.

- **Serialization**  
  Native `.mag` format designed for **zero-copy, memory-mapped loading**, enabling fast startup and efficient large model handling.  
  Conversion tools are provided to import weights from external formats.

- **Backend extensibility**  
  The architecture is intentionally **clean and modular**, making it straightforward to introduce new backends or target additional hardware platforms.

The system is intentionally kept **tight and explicit**, so each layer is understandable, controllable, and replaceable without hidden complexity.

---

## Highlights

- **Practical, not just educational**  
  Capable of running modern LLM inference (e.g. Qwen3 in BF16), not just toy models.

- **Small, controllable ML runtime**  
  Designed to stay inspectable end-to-end — no hidden execution layers or opaque backends.

- **True ownership of execution**  
  You can reason about memory layout, kernel dispatch, and graph behavior without abstraction barriers.

- **Hardware-aware by design**  
  Not a generic backend wrapper — kernels and execution are written with specific ISAs and microarchitectures in mind.

- **Zero-copy model loading**  
  Memory-mapped `.mag` format enables fast startup and efficient handling of large models.

- **Built for experimentation**  
  Easy to modify operators, add kernels, or prototype new execution strategies.

- **Minimal runtime surface**  
  Native extension with no required Python dependencies — easy to deploy and embed.

---

## Example Models

End-to-end demos live under `examples/`.

| Path | Description |
|-----|----------|
| [examples/qwen3/](examples/qwen3/) | Qwen3 transformer inference in bfloat16 with tokenizer integration, `.mag` weights, CLI chat, and HTTP/streaming API. |
| [examples/gpt2/](examples/gpt2/) | GPT-2 causal language model inference with KV cache, token streaming, and configurable generation. |
| [examples/ae/](examples/ae/) | Convolutional autoencoder with training loop and reconstruction visualization. |
| [examples/linear_regression/](examples/linear_regression/) | Simple 1D regression with SGD and loss tracking. |
| [examples/xor/](examples/xor/) | Minimal MLP demonstrating autograd and optimization. |

---

## Operator Cheat Sheet

Magnetron provides a compact but expressive operator set covering:

- elementwise operations (add, mul, div, ...)  
- reductions (sum, mean, ...)  
- tensor transformations (view, reshape, permute, ...)  
- neural building blocks (matmul, softmax, layernorm, ...)  
- type casting and memory views  

A full reference of operators, data types, and semantics is available here:

→ [Magnetron Cheat Sheet](docs/Magnetron-Cheatsheet.md)

---

## Installation
Magnetron is available on PyPI.

Make sure you are inside a Python virtual environment.

```bash
pip install magnetron
```
or with uv:
```bash
uv pip install magnetron
```

---

## Local Development

Clone the repository and install locally:

```bash
git clone --recursive https://github.com/MarioSieg/magnetron
cd magnetron
uv pip install . -v
```

For C/C++ development, open the project root (containing `CMakeLists.txt`) in an IDE such as CLion.

---

## Quick start

```python
from magnetron import Tensor, nn, optim

x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = Tensor([[0.0], [1.0], [1.0], [0.0]])

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.Tanh(),
    nn.Linear(2, 1),
    nn.Tanh(),
)

optimizer = optim.SGD(model.parameters(), lr=1e-1)
criterion = nn.MSELoss()

for epoch in range(2000):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss {loss.item():.6f}")

y_hat = model(x)
for i in range(x.shape[0]):
    print(f"Expected: {y[i].item():.1f}, Predicted: {y_hat[i].item():.4f}")
```

---

## Roadmap

- 🚧 **CUDA backend**  
  Finish memory model, execution pipeline, and stabilize for production use.

- 🚧 **Multi-GPU execution**  
  Introduce scalable execution across multiple devices.

- 🚧 **New CPU architectures**  
  Support for LoongArch and RISC-V.

- 🧪 **JIT compilation**  
  Custom SSA-based IR with register allocation and target-specific instruction emission.

---

## History

Magnetron started in 2024 as a personal project to understand how machine learning frameworks work internally: tensor storage, operator dispatch, autograd, and inference execution.
What began as a learning project gradually evolved into a full runtime with its own tensor engine, native snapshot format, SIMD-specialized CPU backend, and support for running modern models such as Qwen3 in BF16.
Today, Magnetron is developed both as a practical inference/runtime system and as a research platform for experimenting with new backends, execution strategies, and low-level ML systems ideas.

---

## License

(c) 2026 Mario Sieg - mario.sieg.64@gmail.com <br>
Distributed under the Apache 2 License. <br>
Developed in Berlin, Germany. <br>

---

## Similar Projects
- [PyTorch](https://github.com/pytorch/pytorch)
- [GGML](https://github.com/ggerganov/ggml)
- [tinygrad](https://github.com/tinygrad/tinygrad)
- [MLX](https://github.com/ml-explore/mlx)

[contributors-shield]: https://img.shields.io/github/contributors/MarioSieg/magnetron.svg?style=for-the-badge
[contributors-url]: https://github.com/MarioSieg/magnetron/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/MarioSieg/magnetron.svg?style=for-the-badge
[forks-url]: https://github.com/MarioSieg/magnetron/network/members
[stars-shield]: https://img.shields.io/github/stars/MarioSieg/magnetron.svg?style=for-the-badge
[stars-url]: https://github.com/MarioSieg/magnetron/stargazers
[issues-shield]: https://img.shields.io/github/issues/MarioSieg/magnetron.svg?style=for-the-badge
[issues-url]: https://github.com/MarioSieg/magnetron/issues
[license-shield]: https://img.shields.io/github/license/MarioSieg/magnetron.svg?style=for-the-badge
[license-url]: https://github.com/MarioSieg/magnetron/blob/master/LICENSE.txt
[ci-shield]: https://img.shields.io/github/actions/workflow/status/MarioSieg/magnetron/cmake-python-multi-platform.yml?style=for-the-badge
