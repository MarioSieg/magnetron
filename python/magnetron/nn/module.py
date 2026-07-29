# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations
from collections.abc import Iterator, Callable, MutableMapping
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any
from .. import Tensor, dtype


class Parameter(Tensor):
    """A tensor that is a learnable parameter of a model."""

    def __init__(self, x: Tensor) -> None:
        Tensor.__init__(self, x)
        self.requires_grad = True

    @property
    def data(self) -> Tensor:
        return self

    @data.setter
    def data(self, v: Tensor) -> None:
        self._replace(v)


class Buffer(Tensor):
    """A tensor that is registered as non-parameter module state."""

    def __init__(self, x: Tensor, persistent: bool = True) -> None:
        Tensor.__init__(self, x)
        self.persistent = persistent

    @property
    def data(self) -> Tensor:
        return self

    @data.setter
    def data(self, v: Tensor) -> None:
        self._replace(v)


class Module:
    def __init__(self) -> None:
        self.training = True
        self._buffers: dict[str, Buffer] = {}
        self._fwd_hooks: list[Callable[[Module, tuple[Any, ...], Tensor], None]] = []
        self._fwd_pre_hooks: list[Callable[[Module, tuple[Any, ...]], None]] = []

    def named_children(self) -> Iterator[tuple[str, Module]]:
        for name, value in self.__dict__.items():
            if name.startswith('_'):
                continue
            if isinstance(value, Module):
                yield name, value

    def children(self) -> Iterator[Module]:
        for _, child in self.named_children():
            yield child

    def named_modules(
        self,
        prefix: str = '',
        memo: set[int] | None = None,
    ) -> Iterator[tuple[str, Module]]:
        memo = set() if memo is None else memo
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, child in self.named_children():
            child_prefix = f'{prefix}.{name}' if prefix else name
            yield from child.named_modules(child_prefix, memo)

    def modules(self) -> Iterator[Module]:
        for _, module in self.named_modules():
            yield module

    def named_parameters(
        self,
        prefix: str = '',
        memo: set[int] | None = None,
    ) -> Iterator[tuple[str, Parameter]]:
        memo = set() if memo is None else memo
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter) and id(value) not in memo:
                memo.add(id(value))
                yield prefix + name, value
        for child_name, child in self.named_children():
            yield from child.named_parameters(prefix + child_name + '.', memo)

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def register_buffer(self, name: str, tensor: Tensor, persistent: bool = True) -> None:
        if not isinstance(tensor, Tensor):
            raise TypeError(f'buffer must be Tensor, got {type(tensor)}')
        buf = tensor if isinstance(tensor, Buffer) else Buffer(tensor, persistent)
        buf.persistent = persistent
        self._buffers[name] = buf
        setattr(self, name, buf)

    def named_buffers(
        self,
        prefix: str = '',
        memo: set[int] | None = None,
        persistent: bool | None = None,
    ) -> Iterator[tuple[str, Buffer]]:
        memo = set() if memo is None else memo
        for name, buf in self._buffers.items():
            if persistent is not None and buf.persistent != persistent:
                continue
            if id(buf) not in memo:
                memo.add(id(buf))
                yield prefix + name, buf
        for child_name, child in self.named_children():
            yield from child.named_buffers(prefix + child_name + '.', memo, persistent)

    def buffers(self) -> Iterator[Tensor]:
        for _, b in self.named_buffers():
            yield b

    def state_items(self) -> Iterator[tuple[str, Tensor]]:
        yield from self.named_parameters()
        yield from self.named_buffers(persistent=True)

    def state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict((k, v.clone()) for k, v in self.state_items())

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
    ) -> dict[str, list[str]]:
        own_state = dict(self.state_items())
        missing = [k for k in own_state if k not in state_dict]
        unexpected = [k for k in state_dict if k not in own_state]
        for key, tensor in state_dict.items():
            if key not in own_state:
                continue
            own_state[key]._replace(tensor.clone())
        if strict and (missing or unexpected):
            raise RuntimeError(f'Error(s) in loading state_dict:\n\tMissing keys: {missing}\n\tUnexpected keys: {unexpected}')
        return {'missing_keys': missing, 'unexpected_keys': unexpected}

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True) -> Module:
        for p in self.parameters():
            p.requires_grad = requires_grad
        return self

    def cast(self, dt: dtype.DType) -> Module:
        for p in self.parameters():
            req = p.requires_grad
            y = p.cast(dt)
            y.requires_grad = req
            p._replace(y)
        for name, buf in self.named_buffers():
            parent, leaf = self._resolve_parent(name)
            casted = Buffer(buf.cast(dt), persistent=buf.persistent)
            setattr(parent, leaf, casted)
            parent._buffers[leaf] = casted
        return self

    def _resolve_parent(self, key: str) -> tuple[Module, str]:
        parts = key.split('.')
        target: Module = self
        for p in parts[:-1]:
            if isinstance(target, ModuleDict):
                target = target[p]
            elif isinstance(target, ModuleList):
                target = target[int(p)]
            else:
                target = getattr(target, p)
        return target, parts[-1]

    def register_forward_hook(
        self,
        hook: Callable[[Module, tuple[Any, ...], Tensor], None],
    ) -> Callable[[Module, tuple[Any, ...], Tensor], None]:
        self._fwd_hooks.append(hook)
        return hook

    def register_forward_pre_hook(
        self,
        hook: Callable[[Module, tuple[Any, ...]], None],
    ) -> Callable[[Module, tuple[Any, ...]], None]:
        self._fwd_pre_hooks.append(hook)
        return hook

    def apply(self, fn: Callable[[Module], None]) -> Module:
        for mod in self.modules():
            fn(mod)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        for hook in self._fwd_pre_hooks:
            hook(self, args)
        out = self.forward(*args, **kwargs)
        for hook in self._fwd_hooks:
            hook(self, args, out)
        return out


class ModuleList(Module, list[Module]):
    def __init__(self, modules: list[Module] | None = None) -> None:
        Module.__init__(self)
        list.__init__(self)
        if modules:
            self.extend(modules)

    def append(self, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError('ModuleList can only contain Module instances')
        list.append(self, module)

    def extend(self, modules) -> None:
        for module in modules:
            self.append(module)

    def named_children(self) -> Iterator[tuple[str, Module]]:
        for i, module in enumerate(self):
            yield str(i), module


class ModuleDict(Module, MutableMapping[str, Module]):
    def __init__(self, modules: dict[str, Module] | None = None) -> None:
        super().__init__()
        self._modules: dict[str, Module] = {}
        if modules:
            for name, module in modules.items():
                self[name] = module

    def __setitem__(self, name: str, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(f'ModuleDict can only hold Module, got {type(module)}')
        self._modules[name] = module

    def __getitem__(self, name: str) -> Module:
        return self._modules[name]

    def __delitem__(self, name: str) -> None:
        del self._modules[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)

    def __getattr__(self, name: str) -> Module:
        if self._modules is not None and name in self._modules:
            return self._modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def values(self):
        return self._modules.values()

    def __contains__(self, name: object) -> bool:
        return name in self._modules

    def named_children(self) -> Iterator[tuple[str, Module]]:
        yield from self._modules.items()


class Sequential(ModuleList):
    def __init__(self, *modules: Module | list[Module] | tuple[Module, ...]) -> None:
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = tuple(modules[0])
        super().__init__(list(modules))

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        x: Any = args[0] if len(args) == 1 else args
        for module in self:
            x = module(*x, **kwargs) if isinstance(x, tuple) else module(x, **kwargs)
            kwargs = {}
        return x
