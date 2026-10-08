# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from typing import Sequence

from . import Tensor, dtype
from ._magnetron_bindings._distributed import Communicator, ReduceOp

__all__ = ['Communicator', 'ReduceOp', 'DistributedContext', 'DeviceMesh']

class DistributedContext:
    def __init__(
        self,
        world_rank: int,
        world_size: int,
        backend: str
    ) -> None:
        if world_size <= 0:
            raise ValueError('World size must be > 0')
        if world_rank < 0 or world_rank >= world_size:
            raise ValueError(f'World rank {world_rank} out of range for world size {world_size}')
        self._world_rank = world_rank
        self._world_size = world_size
        self._backend = backend
        self._comms: dict[tuple[int, ...], Communicator] = {}

    @property
    def world_rank(self) -> int:
        return self._world_rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def world(self) -> Communicator:
        return self.create_comm(range(self._world_size))

    def create_comm(self, ranks: Tensor | Sequence[int]) -> Communicator:
        if isinstance(ranks, Tensor):
            ranks = ranks.transfer('cpu').cast(dtype.uint32).tolist()
        ranks = tuple(int(r) for r in ranks)
        if not ranks:
            raise ValueError('Communicator ranks cannot be empty')
        if len(set(ranks)) != len(ranks):
            raise ValueError('Communicator ranks must be unique')
        for rank in ranks:
            if rank < 0 or rank >= self._world_size:
                raise ValueError(f'Rank {rank} is out of range for world size {self._world_size}')
        if self._world_rank not in ranks:
            raise ValueError(f'World rank {self._world_rank} is not a member of ranks {ranks}')
        key = ranks
        if key not in self._comms:
            self._comms[key] = Communicator(
                rank=ranks.index(self._world_rank),
                size=len(ranks),
                backend=self._backend,
            )
        return self._comms[key]

    def destroy(self) -> None:
        for comm in self._comms.values():
            comm.destroy()
        self._comms.clear()

    def __repr__(self) -> str:
        return (
            f'DistributedContext('
            f'world_rank={self.world_rank}, '
            f'world_size={self.world_size}, '
            f'backend={self.backend!r})'
        )

class DeviceMesh:
    def __init__(self, context: DistributedContext, device_type: str, mesh: Tensor | Sequence[int], *, mesh_dim_names: tuple[str, ...] | None = None) -> None:
        self._ctx = context
        self._device_type = device_type
        if not isinstance(mesh, Tensor):
            mesh = Tensor(mesh)
        if (mesh < 0).any():
            raise ValueError(f'Negative mesh ranks are disallowed')
        self._mesh: Tensor = mesh.transfer('cpu').cast(dtype.uint32).contiguous()
        self._mesh_dim_names: tuple[str, ...] | None = tuple(mesh_dim_names) if mesh_dim_names is not None else None
        if self._mesh_dim_names is not None:
            if len(self._mesh_dim_names) != self.mesh.rank:
                raise ValueError(f'Mesh dim names must have one name per dim, but have {len(self._mesh_dim_names)}, but requires {self.mesh.rank}')
            if len(set(self._mesh_dim_names)) != len(self._mesh_dim_names):
                raise ValueError('Mesh dim names must be unique')
        self._coords: Tensor | None = self._compute_coords()
        self._dim_comms: list[Communicator | None] = self._create_dim_comms()

    def _compute_coords(self) -> Tensor | None:
        coords = (self._mesh == self.world_rank).nonzero()
        return None if coords.numel == 0 else coords[0]

    def _resolve_dim(self, mesh_dim: int | str | None) -> int:
        if mesh_dim is None:
            if self.ndim != 1:
                raise RuntimeError('Mesh dim must be specified for a N-D mesh!')
            return 0
        if isinstance(mesh_dim, str):
            if self._mesh_dim_names is None:
                raise RuntimeError(f'DeviceMesh has no mesh dim names')
            try:
                return self._mesh_dim_names.index(mesh_dim)
            except ValueError:
                raise KeyError(f'Mesh dim {mesh_dim!r} does not exist')
        if mesh_dim < 0:
            mesh_dim += self.ndim
        if mesh_dim < 0 or mesh_dim >= self.ndim:
            raise IndexError(f'Mesh dim {mesh_dim} out of range for {self.ndim}D mesh')
        return mesh_dim

    def _get_dim_comm_ranks(self, mesh_dim: int) -> Tensor|None:
        if self._coords is None:
            return None
        index = []
        for dim in range(self.ndim):
            if dim == mesh_dim:
                index.append(slice(None))
            else:
                index.append(int(self._coords[dim]))
        return self.mesh[tuple(index)]

    def _create_dim_comms(self) -> list[Communicator | None]:
        comms = []
        for dim in range(self.ndim):
            ranks = self._get_dim_comm_ranks(dim)
            if ranks is None:
                comms.append(None)
                continue
            comms.append(self._ctx.create_comm(ranks))
        return comms

    def lookup_comm_opt(self, mesh_dim: int | str | None = None) -> Communicator | None:
        mesh_dim = self._resolve_dim(mesh_dim)
        return self._dim_comms[mesh_dim]


    def lookup_comm(self, mesh_dim: int | str | None = None) -> Communicator:
        comm = self.lookup_comm_opt(mesh_dim)
        if comm is None:
            raise RuntimeError(f'Rank {self.world_rank} is not part of the DeviceMesh')
        return comm

    def all_comms(self):
        return list(self._dim_comms)

    def local_rank(self, mesh_dim: int | str | None=None) -> int:
        dim = self._resolve_dim(mesh_dim)
        if self._coords is None:
            raise RuntimeError(f'Rank {self.world_rank} is not part of this DeviceMesh')
        return int(self._coords[dim])

    @property
    def context(self) -> DistributedContext:
        return self._ctx

    @property
    def device_type(self) -> str:
        return self._device_type

    @property
    def mesh(self) -> Tensor:
        return self._mesh

    @property
    def ndim(self) -> int:
        return self._mesh.rank

    @property
    def size(self) -> int:
        return self._mesh.numel

    @property
    def shape(self) -> tuple[int, ...]:
        return self._mesh.shape

    @property
    def world_rank(self) -> int:
        return self._ctx.world_rank

    @property
    def coords(self) -> Tensor | None:
        return self._coords

    @property
    def mesh_dim_names(self) -> tuple[str, ...] | None:
        return self._mesh_dim_names

    def __repr__(self) -> str:
        args = [
            repr(self.device_type),
            repr(self.mesh),
        ]
        if self.mesh_dim_names is not None:
            args.append(
                f'mesh_dim_names={self.mesh_dim_names!r}'
            )
        return f"DeviceMesh({', '.join(args)})"
