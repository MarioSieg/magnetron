[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/MarioSieg/magnetron/cmake-multi-platform.yml?style=for-the-badge)


<br />
<div align="center">
  <a href="https://github.com/MarioSieg/magnetron">
    <img src="media/magnetron-logo.svg" alt="Logo" width="200" height="200">
  </a>

<h3 align="center">magnetron</h3>
  <p align="center">
    Minimalistic homemade PyTorch alternative, written in C99 and Python.
    <br />
    <a href="https://github.com/MarioSieg/magnetron/tree/master/python/examples/simple"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/MarioSieg/magnetron/blob/master/python/examples/xor.py">View Demo</a>
    |
    <a href="https://github.com/MarioSieg/magnetron/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    |
    <a href="https://github.com/MarioSieg/magnetron/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## News
- **[2025/01/14]** 🎉 CPU backend now uses multiple threads with dynamic scaling and thread pooling.
- **[2025/01/02]** 🎈 Magnetron released on GitHub.

## About

![ScreenShot](media/xor.png)
This project started as a learning experience and a way to understand the inner workings of PyTorch and other deep learning frameworks.<br>
The goal is to create a minimalistic but still powerful deep learning framework that can be used for research and production.<br>
The framework is written in C99 and Python and is designed to be easy to understand and modify.<br>

### Work in Progress
* The project is still in its early stages and many features are missing.
* Developed by a single person in their free time.
* The project is not yet fully optimized for performance.

## Getting Started

To get a local copy up and running follow these simple steps.<br>
Magnetron itself has **no** Python dependencies except for CFFI to call the C library from Python.<br>
Some examples use matplotlib and numpy for plotting and data generation, but these are not required to use the framework.

### Prerequisites
* Linux, MacOS or Windows
* A C99 compiler (gcc, clang, msvc)
* Python 3.6 or higher

### Installation
*A pip installable package will be provided, as soon as all core features are implemented.*
1. Clone the repo
2. `cd magnetron/python` (VENV recommended).
3. `pip install -r requirements.txt` Install dependencies for examples.
4. `cd magnetron_framework && bash install_wheel_local.sh && cd ../` Install the Magnetron wheel locally, a pip installable package will be provided in the future.
5. `python examples/xor.py` Run the XOR example.

## Usage
See the [Examples](python/examples) directory for examples on how to use the framework.
For usage in C and C++ see the [Unit Tests](test) directory in the root of the project.

## Features
* 6 Dimensional, linearized tensors
* Automatic Differentiation
* Multithreaded CPU Compute, SIMD optimized operators (SSE4, AVX2, AVX512, ARM NEON)
* Modern Python API (similar to PyTorch)
* Many operators with broadcasting support and in-place variants
* High level neural network building blocks
* Dynamic computation graph (eager evaluation)
* Modern PRNGs: Mersenne Twister and PCG
* Validation and friendly error messages
* Custom compressed tensor file formats

### Example
Code from the XOR example:
```python
def forward(self, x: Tensor) -> Tensor:
    return (self.weight @ x + self.bias).sigmoid()
```

### Operators
<table>
  <thead>
    <tr>
      <th>Operation</th>
      <th><div align="center">Description</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>clone(x)</td>
      <td><div align="center">Creates a copy of the tensor</div></td>
    </tr>
    <tr>
      <td>view(x)</td>
      <td><div align="center">Reshapes without changing data</div></td>
    </tr>
    <tr>
      <td>transpose(x)</td>
      <td><div align="center">Swaps tensor dimensions</div></td>
    </tr>
    <tr>
      <td>permute(x, d0, ...)</td>
      <td><div align="center">Reorders tensor dimensions</div></td>
    </tr>
    <tr>
      <td>mean(x)</td>
      <td><div align="center">Mean across dimensions</div></td>
    </tr>
    <tr>
      <td>min(x)</td>
      <td><div align="center">Minimum value of tensor</div></td>
    </tr>
    <tr>
      <td>max(x)</td>
      <td><div align="center">Maximum value of tensor</div></td>
    </tr>
    <tr>
      <td>sum(x)</td>
      <td><div align="center">Sum of elements</div></td>
    </tr>
    <tr>
      <td>abs(x)</td>
      <td><div align="center">Element-wise absolute value</div></td>
    </tr>
    <tr>
      <td>neg(x)</td>
      <td><div align="center">Element-wise negation</div></td>
    </tr>
    <tr>
      <td>log(x)</td>
      <td><div align="center">Element-wise natural logarithm</div></td>
    </tr>
    <tr>
      <td>sqr(x)</td>
      <td><div align="center">Element-wise square</div></td>
    </tr>
    <tr>
      <td>sqrt(x)</td>
      <td><div align="center">Element-wise square root</div></td>
    </tr>
    <tr>
      <td>sin(x)</td>
      <td><div align="center">Element-wise sine</div></td>
    </tr>
    <tr>
      <td>cos(x)</td>
      <td><div align="center">Element-wise cosine</div></td>
    </tr>
    <tr>
      <td>softmax(x)</td>
      <td><div align="center">Softmax along dimension</div></td>
    </tr>
    <tr>
      <td>sigmoid(x)</td>
      <td><div align="center">Element-wise sigmoid</div></td>
    </tr>
    <tr>
      <td>relu(x)</td>
      <td><div align="center">ReLU activation</div></td>
    </tr>
    <tr>
      <td>gelu(x)</td>
      <td><div align="center">GELU activation</div></td>
    </tr>
    <tr>
      <td>add(x, y)</td>
      <td><div align="center">Element-wise addition</div></td>
    </tr>
    <tr>
      <td>sub(x, y)</td>
      <td><div align="center">Element-wise subtraction</div></td>
    </tr>
    <tr>
      <td>mul(x, y)</td>
      <td><div align="center">Element-wise multiplication</div></td>
    </tr>
    <tr>
      <td>div(x, y)</td>
      <td><div align="center">Element-wise division</div></td>
    </tr>
    <tr>
      <td>matmul(A, B)</td>
      <td><div align="center">Matrix multiplication</div></td>
    </tr>
  </tbody>
</table>

## Roadmap

The goal is to implement training and inference for LLMs and other state of the art models, while providing a simple and small codebase that is easy to understand and modify.

- [ ] Compute on GPU (Cuda)
- [ ] Low-precision datatypes (f16, bf16, int8)
- [ ] Distributed Training and Inference
- [ ] CPU and GPU kernel JIT compilation
- [ ] Better examples with real world models (LLMs and state of the art models)

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

## License
Distributed under the Apache 2 License. See `LICENSE.txt` for more information.

## Similar Projects

* [GGML](https://github.com/ggerganov/ggml)
* [TINYGRAD](https://github.com/tinygrad/tinygrad)
* [MICROGRAD](https://github.com/karpathy/micrograd)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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
