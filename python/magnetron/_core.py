# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from dataclasses import dataclass
from os import getenv

from ._dtype import DataType, float32


@dataclass
class Config:
    verbose: bool = getenv('MAGNETRON_VERBOSE', '0') == '1'
    default_dtype: DataType = float32
