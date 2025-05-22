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
    Super minimalistic machine-learning framework.
    <br />
    <a href="https://github.com/MarioSieg/magnetron/tree/master/python/examples/simple"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/MarioSieg/magnetron/blob/master/python/examples/xor.py">View Example</a>
    |
    <a href="https://github.com/MarioSieg/magnetron/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    |
    <a href="https://github.com/MarioSieg/magnetron/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

## About

Magnetron is a minimalistic, PyTorch-style machine-learning framework designed for IoT and other resource-limited environments.<br>
The tiny C99 core - wrapped in a modern Python API - gives you dynamic graphs, automatic differentiation and network building blocks without the bloat.<br>
A CUDA backend is also WIP.<br>

### Key features
* N-dimensional, flattened tensors
* Automatic differentiation on dynamic computation graphs
* CPU multithreading + SIMD (SSE4, AVX2/AVX512, ARM NEON)
* PyTorch-like Python API
* Broadcasting-aware operators with in-place variants
* High-level neural-net building blocks
* float32 and float16 datatypes
* Modern PRNGs (Mersenne Twister, PCG)
* Clear validation and error messages
* Custom compressed tensor file formats
* No C and Python dependencies (except CFFI for the Python wrapper)

## XOR Training Example
A simple XOR neuronal network (MLP) trained with Magnetron. Copy and paste the code below into a file called `xor.py` and run it with Python.
```python
import magnetron as mag
from magnetron import optim, nn
from matplotlib import pyplot as plt

EPOCHS: int = 2000

# Create the model, optimizer, and loss function
model = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1), nn.Tanh())
optimizer = optim.SGD(model.parameters(), lr=1e-1)
criterion = nn.MSELoss()
loss_values: list[float] = []

x = mag.Tensor.from_data([[0, 0], [0, 1], [1, 0], [1, 1]])
y = mag.Tensor.from_data([[0], [1], [1], [0]])

# Train the model
for epoch in range(EPOCHS):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_values.append(loss.item())

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Print the final predictions after the training
print('=== Final Predictions ===')

with mag.no_grad():
    y_hat = model(x)
    for i in range(x.shape[0]):
        print(f'Expected: {y[i]}, Predicted: {y_hat[i]}')

# Plot the loss

plt.figure()
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss over Time')
plt.grid(True)
plt.show()
```

This results in the following output:

![ScreenShot](media/xor.png)

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
5. `python examples/simple/xor.py` Run the XOR example.

## Usage
See the [Examples](python/examples) directory which contains various models and training examples.<br>
For usage in C or C++ see the [Unit Tests](test) directory.

### Operators
The following table lists all available operators and their properties.

|Mnemonic    |Desc                                |IN |OUT|Params |Flags|Inplace|Backward|Result        |Validation|CPU-Parallel|Type     |
|------------|------------------------------------|---|---|-------|-----|-------|--------|--------------|----------|------------|---------|
|NOP         |no-op                               |0  |0  |N/A    |N/A  |NO     |NO      |N/A           |NO        |NO          |NO-OP    |
|CLONE       |strided copy                        |1  |1  |N/A    |N/A  |NO     |YES     |ISOMORPH      |YES       |NO          |Morph    |
|VIEW        |memory view                         |1  |1  |N/A    |N/A  |NO     |YES     |ISOMORPH      |YES       |NO          |Morph    |
|TRANSPOSE   |𝑥ᵀ                                 |1  |1  |N/A    |N/A  |NO     |YES     |TRANSPOSED    |YES       |NO          |Morph    |
|PERMUTE     |swap axes by index                  |1  |1  |U64 [6]|N/A  |NO     |NO      |PERMUTED      |YES       |NO          |Morph    |
|MEAN        |(∑𝑥) ∕ 𝑛                          |1  |1  |N/A    |N/A  |NO     |YES     |SCALAR/REDUCED|YES       |NO          |Reduction|
|MIN         |min(𝑥)                             |1  |1  |N/A    |N/A  |NO     |NO      |SCALAR/REDUCED|YES       |NO          |Reduction|
|MAX         |max(𝑥)                             |1  |1  |N/A    |N/A  |NO     |NO      |SCALAR/REDUCED|YES       |NO          |Reduction|
|SUM         |∑𝑥                                 |1  |1  |N/A    |N/A  |NO     |YES     |SCALAR/REDUCED|YES       |NO          |Reduction|
|ABS         |&#124;𝑥&#124;                                |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|SGN         |𝑥⁄                                 |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|NEG         |−𝑥                                 |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|LOG         |log₁₀(𝑥)                           |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|SQR         |𝑥²                                 |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|SQRT        |√𝑥                                 |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|SIN         |sin(𝑥)                             |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|COS         |cos(𝑥)                             |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|STEP        |𝐻(𝑥)                              |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|EXP         |𝑒ˣ                                 |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|FLOOR       |⌊𝑥⌋                                |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|CEIL        |⌈𝑥⌉                                |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|ROUND       |⟦𝑥⟧                                |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|SOFTMAX     |𝑒ˣⁱ ∕ ∑𝑒ˣʲ                        |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|SOFTMAX_DV  |𝑑⁄𝑑𝑥 softmax(𝑥)                 |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|SIGMOID     |1 ∕ (1 + 𝑒⁻ˣ)                      |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|SIGMOID_DV  |𝑑⁄𝑑𝑥 sigmoid(𝑥)                 |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|HARD_SIGMOID|max(0, min(1, 0.2×𝑥 + 0.5))        |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|SILU        |𝑥 ∕ (1 + 𝑒⁻ˣ)                     |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|SILU_DV     |𝑑⁄𝑑𝑥 silu(𝑥)                    |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|TANH        |tanh(𝑥)                            |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|TANH_DV     |𝑑⁄𝑑𝑥 tanh(𝑥)                    |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|RELU        |max(0, 𝑥)                          |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|RELU_DV     |𝑑⁄𝑑𝑥 relu(𝑥)                    |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|GELU        |0.5×𝑥×(1 + erf(𝑥 ∕ √2))           |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|GELU_DV     |𝑑⁄𝑑𝑥 gelu(𝑥)                    |1  |1  |N/A    |N/A  |YES    |YES     |ISOMORPH      |YES       |YES         |Unary OP |
|ADD         |𝑥 + 𝑦                             |2  |1  |N/A    |N/A  |YES    |YES     |BROADCASTED   |YES       |YES         |Binary OP|
|SUB         |𝑥 − 𝑦                             |2  |1  |N/A    |N/A  |YES    |YES     |BROADCASTED   |YES       |YES         |Binary OP|
|MUL         |𝑥 ⊙ 𝑦                             |2  |1  |N/A    |N/A  |YES    |YES     |BROADCASTED   |YES       |YES         |Binary OP|
|DIV         |𝑥 ∕ 𝑦                             |2  |1  |N/A    |N/A  |YES    |YES     |BROADCASTED   |YES       |YES         |Binary OP|
|MATMUL      |𝑥𝑦                                |2  |1  |N/A    |N/A  |YES    |YES     |MATRIX        |YES       |YES         |Binary OP|
|REPEAT_BACK |gradient broadcast to repeated shape|2  |1  |N/A    |N/A  |YES    |YES     |BROADCASTED   |YES       |NO          |Binary OP|


## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

## License
(c) 2025 Mario "Neo" Sieg. mario.sieg.64@gmail.com<br>
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
