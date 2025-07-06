# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations
from collections.abc import Iterator, Callable, MutableMapping
from typing import Mapping, OrderedDict

from magnetron import Tensor


class Parameter:
    """A tensor that is a learnable parameter of a model."""

    def __init__(self, x: Tensor, name: str | None = None) -> None:
        x.requires_grad = True
        if name is not None:
            x.name = name
        self.x = x

    @property
    def data(self) -> Tensor:
        return self.x

    @data.setter
    def data(self, v: Tensor) -> None:
        self.x = v

    def __str__(self) -> str:
        return self.x.__str__()

    def __repr__(self) -> str:
        return self.x.__repr__()


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

    def state_dict(self) -> OrderedDict[str, Tensor]:
        dest = OrderedDict()

        def _recurse(m: Module | ModuleList, prefix: str = '') -> None:
            if isinstance(m, ModuleList):
                for i, sub in enumerate(m):
                    _recurse(sub, f'{prefix}{i}.')
                return

            for name, attr in m.__dict__.items():
                if isinstance(attr, Parameter):
                    dest[f'{prefix}{name}'] = attr.x.clone()
                elif isinstance(attr, Tensor):
                    dest[f'{prefix}{name}'] = attr.clone()
                elif isinstance(attr, Module):
                    _recurse(attr, f'{prefix}{name}.')
                elif isinstance(attr, ModuleList):
                    _recurse(attr, f'{prefix}{name}.')

        _recurse(self)
        return dest

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
    ) -> dict[str, list[str]]:
        missing, unexpected = [], []

        for full_key, tensor in state_dict.items():
            parts = full_key.split('.')
            target: 'Module | ModuleList' = self
            ok = True

            for p in parts[:-1]:
                if p.isdigit():
                    idx = int(p)
                    if not isinstance(target, (list, ModuleList)) or idx >= len(target):
                        ok = False
                        break
                    target = target[idx]
                else:
                    target = getattr(target, p, None)
                    if target is None:
                        ok = False
                        break

            if not ok:
                unexpected.append(full_key)
                continue

            leaf_name = parts[-1]
            leaf = target[int(leaf_name)] if leaf_name.isdigit() and isinstance(target, (list, ModuleList)) else getattr(target, leaf_name, None)

            if leaf is None:
                unexpected.append(full_key)
                continue

            if isinstance(leaf, Parameter):
                leaf.data = tensor.clone()
            elif isinstance(leaf, Tensor):
                setattr(target, leaf_name, tensor.clone())
            else:
                unexpected.append(full_key)

        def _find_missing(m: 'Module | ModuleList', prefix: str = '') -> None:
            if isinstance(m, ModuleList):
                for i, sub in enumerate(m):
                    _find_missing(sub, f'{prefix}{i}.')
                return
            for name, attr in m.__dict__.items():
                key = f'{prefix}{name}'
                if isinstance(attr, (Parameter, Tensor)):
                    if key not in state_dict:
                        missing.append(key)
                elif isinstance(attr, Module):
                    _find_missing(attr, f'{key}.')
                elif isinstance(attr, ModuleList):
                    _find_missing(attr, f'{key}.')

        _find_missing(self)

        if strict and (missing or unexpected):
            raise RuntimeError(f'Error(s) in loading state_dict:\n  Missing keys: {missing}\n  Unexpected keys: {unexpected}')

        return {'missing_keys': missing, 'unexpected_keys': unexpected}

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
            self.extend(mods)

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

    def _register(self, idx: int, mod: Module) -> None:
        if not isinstance(mod, Module):
            raise TypeError('ModuleList can only contain Module instances')
        super().append(mod)
        setattr(self, str(idx), mod)

    def append(self, mod: Module) -> None:
        self._register(len(self), mod)

    def extend(self, iterable: Iterator[Module]) -> None:
        for m in iterable:
            self.append(m)

    def __setitem__(self, idx: int, mod: Module) -> None:
        super().__setitem__(idx, mod)
        setattr(self, str(idx), mod)


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

    def keys(self) -> 'dict_keys[str, Module]':
        return self._modules.keys()

    def items(self) -> 'dict_items[str, Module]':
        return self._modules.items()

    def values(self) -> 'dict_values[str, Module]':
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
