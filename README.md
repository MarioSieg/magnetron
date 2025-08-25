[![Stargazers][stars-shield]][stars-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
![GitHub Actions Workflow Status][ci-shield]

<br />
<div align="center">
  <a href="https://github.com/MarioSieg/magnetron">
    <img src="media/magnetron-logo.svg" alt="Logo" width="200" height="200">
  </a>

<h3 align="center">magnetron</h3>
  <p align="center">
    Super minimalistic machine-learning framework.
    <br />
    <a href="https://github.com/MarioSieg/magnetron/tree/master/docs"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/MarioSieg/magnetron/blob/master/examples/gpt2/gpt2.py">View GPT-2 Example</a>
    |
    <a href="https://github.com/MarioSieg/magnetron/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    |
    <a href="https://github.com/MarioSieg/magnetron/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

## 📖 About

Magnetron is a minimalistic, PyTorch-inspired machine-learning framework designed for IoT and other resource-limited environments.<br>
The tiny C core - wrapped in a modern Python API - gives you dynamic graphs, automatic differentiation and network building blocks without the bloat.<br>
A CUDA backend is also WIP.<br>

### Key features
* **PyTorch-close** Python API  
  - Seamless switch for PyTorch users with familiar syntax and behavior 
* **Automatic differentiation** on dynamic computation graphs  
  - Supports flexible model construction and training workflows  
* High-level **neural-net building blocks**  
  - Includes `nn.Module`, `Linear`, `Sequential`, and more out of the box
* **Highly optimized CPU kernels**
  - Runtime CPU detection, multithreading, and SIMD acceleration (SSE, AVX2, AVX512, NEON)
* Multiple **datatypes**: **float32**, **float16**, **int32**, and **boolean**  
  - Flexibility for both training and quantized inference  
* Custom **compressed tensor file formats**  
  - Fast serialization & model loading
* Clear **validation** and **error messages**  
  - Easier debugging and better developer experience
* No external **C or Python dependencies** (except **CFFI** for the Python wrapper)  
  - Lightweight and portable - great for embedded or restricted environments


## 📦 Installation

Make sure you are inside a **Python virtual environment** before installing.

**With pip**
```bash
pip install magnetron
```

**With uv**
```bash
uv pip install magnetron
```
> ⚠️ Currently, prebuilt wheels are available for **macOS (Apple Silicon)** and **Linux x86-64**.  
> On other platforms, the package must be built from source.

## 🧪 Examples

### GPT-2 inference

### Training a simple feedforward network

## 🤝 Contributing
Contributions are welcome!  
Please open issues for ideas, or submit pull requests for new **features**.  
PRs that only fix typos or minor formatting will not be accepted.

## 📜 License
(c) 2025 Mario Sieg. mario.sieg.64@gmail.com<br>
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
