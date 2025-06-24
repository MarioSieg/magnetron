# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import math

from magnetron import Tensor
from magnetron.nn.module import Module, Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = Tensor.normal(out_features, in_features, mean=0.0, std=1.0)
        weight = weight / math.sqrt(in_features + out_features)
        self.weight = Parameter(weight)
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features, name='bias'))

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.weight.x.T
        if self.bias is not None:
            x = x + self.bias.x
        return x


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(Tensor.normal(num_embeddings, embedding_dim) / embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(Tensor.zeros(dim))

    def _norm(self, x: Tensor) -> Tensor:
        rms = ((x**2).mean(axis=-1, keepdim=True) + self.eps) ** 0.5
        return x / rms

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x)
        return output * self.weight


class LayerNorm(Module):
    """Layer Normalization over the last dimension with optional affine transform."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(Tensor.full(normalized_shape, fill_value=1.0), name='weight')
            self.bias = Parameter(Tensor.zeros(normalized_shape), name='bias')
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
        x_norm = (x - mean) / (var + self.eps) ** 0.5
        if self.elementwise_affine:
            x_norm = x_norm * self.weight.x + self.bias.x
        return x_norm
