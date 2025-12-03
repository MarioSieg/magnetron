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
    A compact, PyTorch-style machine learning framework written in pure C99.
    <br />
    Designed for speed, clarity, and portability - from desktop to embedded.
    <br /><br />
    <a href="https://github.com/MarioSieg/magnetron/blob/develop/docs/4.%20Operator%20Cheatsheet.md"><strong>Documentation »</strong></a>
    <br /><br />
    <a href="https://github.com/MarioSieg/magnetron/blob/master/examples/gpt2/gpt2.py">GPT-2 Example</a>
    ·
    <a href="https://github.com/MarioSieg/magnetron/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/MarioSieg/magnetron/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

---

## 📖 About

**Magnetron** is a lightweight, research-grade machine learning framework that mirrors the usability of PyTorch - but built entirely from scratch.  
Its C99 core, wrapped in a modern Python API, provides dynamic computation graphs, automatic differentiation, and high-performance operators with zero external dependencies.

Originally designed for constrained or experimental environments, Magnetron scales from small embedded systems to full desktop inference and training.  
A CUDA backend and mixed-precision support are currently in development.

---

### ⚡ Highlights

- **PyTorch-like API**  
  Familiar syntax for building and training models - easy to pick up, minimal to extend.

- **Dynamic autograd engine**  
  Eager execution with full gradient tracking on computation graphs.

- **Optimized C99 backend**  
  Custom tensor engine with SIMD acceleration (SSE, AVX2, AVX-512, NEON) and multithreaded execution.

- **Minimal dependencies**  
  No third-party math libraries; only **CFFI** is required for the Python interface.

- **Lightweight neural modules**  
  Includes `Linear`, `Sequential`, `ReLU`, `Tanh`, `Sigmoid`, `LayerNorm`, `Embedding`, and more.

- **Rich data types with many operators**  
  Supports `float16`, `float32`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`, and `boolean`.

- **Custom serialization format**  
  Fast, portable model saving and loading through Magnetron’s own binary tensor format.

- **Clean diagnostics**  
  Readable validation and error messages for faster debugging and experimentation.

---

## 🚀 Example Models

| Example                                          | Description |
|--------------------------------------------------|-------------|
| [GPT-2 Inference](examples/gpt2/)                | Transformer-based text generation using pretrained GPT-2 weights. |
| [Autoencoder](examples/ae/)                      | Image reconstruction using a small dense encoder–decoder network. |
| [Linear Regression](examples/linear_regression/) | Fits a linear model to noisy synthetic data. |
| [XOR](examples/xor/)                             | Trains a small neural network to learn the XOR logical function. |

---

## 📦 Installation

Make sure you are inside a **Python virtual environment** before installing.

**With uv**
```bash
uv pip install magnetron
```

**With pip**
```bash
pip install magnetron
```

## 🤝 Contributing
Contributions are welcome!  
Please open issues for ideas, or submit pull requests for new **features**.  
PRs that only fix typos or minor formatting will not be accepted.

## 📜 License
(c) 2025 Mario Sieg - mario.sieg.64@gmail.com<br>
Distributed under the Apache 2 License.
See `LICENSE` for more information.

## 🧩 Similar Projects

* [GGML](https://github.com/ggerganov/ggml)
* [TINYGRAD](https://github.com/tinygrad/tinygrad)
* [MICROGRAD](https://github.com/karpathy/micrograd)

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
