# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations
from collections.abc import Iterator, Callable, MutableMapping, dict_values, dict_keys, dict_items
from typing import Any

from magnetron import Tensor


class Parameter:
    """A tensor that is a learnable parameter of a model."""

    def __init__(self, x: Tensor, name: str | None = None) -> None:
        x.requires_grad = True
        if name is not None:
            x.name = name
        self.x = x


class Module:
    """Base class for all neural network modules."""

    def parameters(self) -> list[Parameter]:
        """Return all unique and nested parameters of the module."""
        params: list[Parameter] = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, ModuleList):
                for m in v:
                    params.extend(m.parameters())
        # dedupe while preserving order
        unique: list[Parameter] = []
        seen = set()
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique

    def children(self) -> Iterator[Module]:
        """Yield immediate child modules."""
        for v in self.__dict__.values():
            if isinstance(v, Module):
                yield v
            elif isinstance(v, ModuleList):
                for m in v:
                    yield m

    def modules(self) -> Iterator[Module]:
        """Yield self and all submodules in pre-order."""
        yield self
        for child in self.children():
            yield from child.modules()

    def apply(self, fn: Callable[[Module], None]) -> Module:
        """
        Apply `fn` to self and all submodules.
        Example:
            model.apply(lambda m: init_fn(m))
        """
        for m in self.modules():
            fn(m)
        return self

    def eval(self) -> None:
        """Set module to evaluation mode (disable gradients)."""
        for p in self.parameters():
            p.x.requires_grad = False

    def train(self) -> None:
        """Set module to training mode (enable gradients)."""
        for p in self.parameters():
            p.x.requires_grad = True

    def forward(self, *args: Tensor, **kwargs: dict) -> Tensor:
        """Forward pass; must be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, *args: Tensor, **kwargs: dict) -> Tensor:
        return self.forward(*args, **kwargs)

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        """Register a persistent buffer (non-parameter tensor)."""
        buf = tensor.clone().detach() if isinstance(tensor, Tensor) else tensor
        setattr(self, name, buf)


class ModuleList(Module, list):
    """A list of modules that can be used as a single module."""

    def __init__(self, mods: list[Module] | None) -> None:
        super().__init__()
        if mods is not None:
            self += mods

    def append(self, mod: Module) -> None:
        super().append(mod)

    def extend(self, __iterable: list[Module]) -> None:
        super().extend(__iterable)

    def __iadd__(self, other: list[Module]) -> 'ModuleList':
        self.extend(other)
        return self

    def __setitem__(self, k: int, v: Module) -> None:
        super().__setitem__(k, v)

    def __getitem__(self, k: int) -> Module:
        return super().__getitem__(k)

    def parameters(self) -> list[Parameter]:
        """Returns all unique and nested parameters of the module."""
        params: list[Parameter] = []
        for mod in self:
            params += mod.parameters()
        return list(set(params))


class ModuleDict(Module, MutableMapping[str, Module]):
    """A dict of named submodules that behaves like a single Module."""

    def __init__(self, modules: dict[str, Module] | None = None) -> None:
        super().__init__()
        self._modules: dict[str, Module] = {}
        if modules is not None:
            for name, mod in modules.items():
                self[name] = mod

    def __setitem__(self, name: str, module: Module) -> None:
        if not isinstance(module, Module):
            raise ValueError(f'ModuleDict can only hold Module, got {type(module)}')
        # store in our internal dict
        self._modules[name] = module
        # also bind it as an attribute so Module.children()/modules() will see it
        setattr(self, name, module)

    def __getitem__(self, name: str) -> Module:
        return self._modules[name]

    def __delitem__(self, name: str) -> None:
        del self._modules[name]
        delattr(self, name)

    def __iter__(self) -> None:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)

    def keys(self) -> dict_keys[str, Module]:
        return self._modules.keys()

    def items(self) -> dict_items[str, Module]:
        return self._modules.items()

    def values(self) -> dict_values[str, Module]:
        return self._modules.values()

    def parameters(self) -> list[Parameter]:
        # flatten out all parameters from each module
        params: list[Parameter] = []
        for mod in self._modules.values():
            params.extend(mod.parameters())
        # dedupe
        unique: list[Parameter] = []
        seen = set()
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique


class Sequential(ModuleList):
    """
    A thin wrapper that chains several sub-modules together, feeding the output of one directly into the next.
    """

    def __init__(self, *modules: Module) -> None:
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = tuple(modules[0])
        super().__init__(list(modules))

    def forward(self, *args: Tensor, **kwargs: dict) -> Tensor:
        x: Tensor | tuple[Tensor, ...] = args[0] if len(args) == 1 else args
        for mod in self:
            if isinstance(x, tuple):
                x = mod(*x, **kwargs)
            else:
                x = mod(x, **kwargs)
            kwargs = {}  # Only applies to first call
        return x
