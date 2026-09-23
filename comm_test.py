from magnetron.distributed import DeviceMesh, DistributedContext

ctx = DistributedContext(
    world_rank=0,
    world_size=4,
    backend='nccl',
)

grid = DeviceMesh(
    context=ctx,
    device_type='cuda',
    mesh=[[0, 1],[2, 3]],
    mesh_dim_names=('ep', 'tp'),
)

print(ctx)
print(grid)
print(grid.ndim)
print(grid.size)
print(grid.coords)
print(grid.lookup_comm("ep"))
print(grid.lookup_comm("tp"))
