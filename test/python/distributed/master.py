# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import argparse
from magnetron import Tensor, distributed, dtype


def main() -> None:
    parser = argparse.ArgumentParser(description='Magnetron distributed master')
    parser.add_argument('--ip', default='0.0.0.0', help='Master IP address')
    parser.add_argument('--port', type=int, default=29500, help='Master TCP port')
    parser.add_argument('--world-size', type=int, default=2, help='Number of processes')
    parser.add_argument('--rank', type=int, default=0, help='Rank of this process')
    args = parser.parse_args()
    pg = distributed.ProcessGroup(
        master_addr=args.ip,
        master_port=args.port,
        rank=args.rank,
        world_size=args.world_size,
    )
    x = Tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype.bfloat16)
    print(f'Rank {pg.rank}/{pg.world_size}')
    print('Before:', x)
    pg.all_reduce_sum_(x)
    print('After: ', x)


if __name__ == '__main__':
    main()
