# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations
from collections.abc import Iterator, Callable, MutableMapping
from typing import Mapping, OrderedDict

from magnetron import Tensor


class Parameter:
    """A tensor that is a learnable parameter of a model."""

    def __init__(self, x: Tensor) -> None:
        x.requires_grad = True
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

    def __init__(self) -> None:
        self._buffer_names = set()
        self._fwd_hooks: list[Callable[[Module, tuple, Tensor], None]] = []
        self._fwd_pre_hooks: list[Callable[[Module, tuple], None]] = []

    def _parameters(self, visited: set[int]) -> Iterator[Parameter]:
        for v in self.__dict__.values():
            if isinstance(v, Parameter):
                vid = id(v)
                if vid not in visited:
                    visited.add(vid)
                    yield v
            elif isinstance(v, Module):
                yield from v._parameters(visited)
            elif isinstance(v, ModuleList):
                for m in v:
                    yield from m._parameters(visited)

    def register_forward_hook(self, hook: Callable[[Module, tuple, Tensor], None]) -> Callable[[Module, tuple, Tensor], None]:
        self._fwd_hooks.append(hook)
        return hook

    def register_forward_pre_hook(self, hook: Callable[[Module, tuple], None]) -> Callable[[Module, tuple], None]:
        self._fwd_pre_hooks.append(hook)
        return hook

    def parameters(self) -> Iterator['Parameter']:
        """Yield all unique and nested parameters of the module."""
        visited: set[int] = set()
        yield from self._parameters(visited)

    def named_parameters(
        self,
        prefix: str = '',
    ) -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()
        for attr_name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    yield prefix + attr_name, value
            elif isinstance(value, Module):
                yield from value.named_parameters(prefix + attr_name + '.')
            elif isinstance(value, ModuleList):
                for idx, sub_mod in enumerate(value):
                    yield from sub_mod.named_parameters(f'{prefix}{attr_name}.{idx}.')

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

    def named_children(self) -> Iterator[tuple[str, Module]]:
        for attr_name, value in self.__dict__.items():
            if isinstance(value, Module):
                yield attr_name, value
            elif isinstance(value, ModuleList):
                for i, m in enumerate(value):
                    yield f'{attr_name}.{i}', m

    def named_modules(self, memo: set[int] | None = None, prefix: str = '') -> Iterator[tuple[str, Module]]:
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, child in self.named_children():
            next_prefix = f'{prefix}.{name}' if prefix else name
            yield from child.named_modules(memo, next_prefix)

    def _state_items(self, prefix: str = '') -> Iterator[tuple[str, Tensor]]:
        for name, attr in self.__dict__.items():
            if isinstance(attr, Parameter):
                yield f'{prefix}{name}', attr.x
            elif isinstance(attr, Tensor):
                yield f'{prefix}{name}', attr
            elif isinstance(attr, Module):
                yield from attr._state_items(f'{prefix}{name}.')
            elif isinstance(attr, ModuleList):
                for i, sub in enumerate(attr):
                    yield from sub._state_items(f'{prefix}{name}.{i}.')

    def state_items(self) -> Iterator[tuple[str, Tensor]]:
        yield from self._state_items()

    def state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(self.state_items())

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

    def eval(self) -> Module:
        """Set module to evaluation mode (disable gradients)."""
        for p in self.parameters():
            p.x.requires_grad = False
        return self

    def train(self) -> Module:
        """Set module to training mode (enable gradients)."""
        for p in self.parameters():
            p.x.requires_grad = True
        return self

    def forward(self, *args: Tensor, **kwargs: dict) -> Tensor:
        """Forward pass; must be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, *args: Tensor, **kwargs: dict) -> Tensor:
        for h in self._fwd_pre_hooks:
            h(self, args)
        out = self.forward(*args, **kwargs)
        for h in self._fwd_hooks:
            h(self, args, out)
        return out

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        buf = tensor.clone().detach() if isinstance(tensor, Tensor) else tensor
        setattr(self, name, buf)
        names = getattr(self, '_buffer_names', set())
        names.add(name)
        self._buffer_names = names

    def named_buffers(self, prefix: str = '') -> Iterator[tuple[str, Tensor]]:
        for name in getattr(self, '_buffer_names', set()):
            yield prefix + name, getattr(self, name)
        for attr_name, value in self.__dict__.items():
            if isinstance(value, Module):
                yield from value.named_buffers(prefix + attr_name + '.')
            elif isinstance(value, ModuleList):
                for i, m in enumerate(value):
                    yield from m.named_buffers(f'{prefix}{attr_name}.{i}.')

    def buffers(self) -> Iterator[Tensor]:
        for _, t in self.named_buffers():
            yield t


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

    def parameters(self) -> Iterator[Parameter]:
        seen: set[int] = set()
        for mod in self:
            yield from mod._parameters(seen)

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

    def parameters(self) -> Iterator[Parameter]:
        seen: set[int] = set()
        for mod in self._modules.values():
            yield from mod._parameters(seen)

    def named_parameters(self, prefix: str = '') -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()
        for name, mod in self._modules.items():
            for sub_name, p in mod.named_parameters(prefix + name + '.'):
                if id(p) not in seen:
                    seen.add(id(p))
                    yield sub_name, p


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
