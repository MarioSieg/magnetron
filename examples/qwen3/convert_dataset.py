# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import argparse
import glob
import json
import math
import os
import time
from dataclasses import dataclass

from magnetron import Tensor, dtype
from magnetron.snapshot import SnapshotWriter
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.table import Table
from safetensors import safe_open

import torch

console = Console()

# HF tensors that are recomputed by the Magnetron model instead of being stored
_SKIPPED_HF_KEYS: set[str] = {'rotary_emb.inv_freq'}


def _fmt_bytes(n: float) -> str:
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(n) < 1024.0 or unit == 'TiB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.2f} {unit}'
        n /= 1024.0
    raise AssertionError


def _mag_to_torch_dtype(mag_dtype: dtype.DType) -> torch.dtype:
    return {
        dtype.float16: torch.float16,
        dtype.bfloat16: torch.bfloat16,
        dtype.float32: torch.float32,
    }[mag_dtype]


def _mag_dtype_from_str(dtype_str: str) -> dtype.DType:
    return {
        'float16': dtype.float16,
        'bfloat16': dtype.bfloat16,
        'float32': dtype.float32,
    }[dtype_str]


def _iter_safetensor_shards(repo_dir: str) -> list[str]:
    index_path = os.path.join(repo_dir, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path, encoding='utf-8') as f:
            index = json.load(f)
        shards = sorted(set(index['weight_map'].values()))
        return [os.path.join(repo_dir, s) for s in shards]
    shards = sorted(glob.glob(os.path.join(repo_dir, 'model-*.safetensors')))
    if shards:
        return shards
    single = os.path.join(repo_dir, 'model.safetensors')
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError('No safetensors weights found in repo snapshot.')


def _mag_key_for(hf_key: str) -> str:
    return hf_key[len('model.') :] if hf_key.startswith('model.') else hf_key


@dataclass(frozen=True, slots=True)
class _TensorPlan:
    shard: str
    hf_key: str
    mag_key: str
    shape: tuple[int, ...]

    def numbytes(self, mag_dtype: dtype.DType) -> int:
        return math.prod(self.shape) * mag_dtype.size


def _plan_tensors(repo_dir: str) -> list[_TensorPlan]:
    plan: list[_TensorPlan] = []
    seen: dict[str, str] = {}
    for shard in _iter_safetensor_shards(repo_dir):
        with safe_open(shard, framework='pt') as f:
            for hf_key in sorted(f.keys()):
                if any(hf_key.endswith(skip) for skip in _SKIPPED_HF_KEYS):
                    continue
                mag_key = _mag_key_for(hf_key)
                if mag_key in seen:
                    raise KeyError(f'{mag_key} appears in both {os.path.basename(seen[mag_key])} and {os.path.basename(shard)}')
                seen[mag_key] = shard
                plan.append(
                    _TensorPlan(
                        shard=shard,
                        hf_key=hf_key,
                        mag_key=mag_key,
                        shape=tuple(f.get_slice(hf_key).get_shape()),
                    )
                )
    if not plan:
        raise RuntimeError('No convertible tensors found in the safetensors shards.')
    return plan


def _load_one(entry: _TensorPlan, torch_dtype: torch.dtype, mag_dtype: dtype.DType) -> Tensor:
    with safe_open(entry.shard, framework='pt') as f:
        src = f.get_tensor(entry.hf_key)
        src = src.to(torch_dtype).contiguous()  # No copy when the dtype already matches
        out = Tensor(src, dtype=mag_dtype)
    del src
    return out


def _load_hf_config(repo_dir: str) -> dict:
    config_path = os.path.join(repo_dir, 'config.json')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, encoding='utf-8') as f:
        return json.load(f)


def _write_model_card(
    path: str,
    *,
    repo: str,
    snap_file: str,
    mag_dtype: dtype.DType,
    cfg: dict,
    tensor_rows: list[tuple[str, tuple[int, ...], str]],
) -> None:
    tensor_rows = sorted(tensor_rows, key=lambda x: x[0])
    model_name = repo.split('/')[-1]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# {model_name} Magnetron Snapshot\n\n')
        f.write(f'This repository contains a Magnetron snapshot converted from the original Hugging Face model `{repo}`.\n\n')
        f.write('The snapshot is intended for inference with the Magnetron runtime. ')
        f.write(f'All convertible tensors are stored using `{mag_dtype.short_name}` where applicable.\n\n')
        f.write('## Model details\n\n')
        f.write(f'- **Source model:** `{repo}`\n')
        f.write(f'- **Snapshot file:** `{snap_file}`\n')
        f.write(f'- **Magnetron dtype mode:** `{mag_dtype.short_name}`\n')
        f.write(f'- **Tensor count:** `{len(tensor_rows)}`\n\n')
        f.write('## Qwen3 configuration\n\n')
        f.write('| Field | Value |\n')
        f.write('|---|---:|\n')
        for k, v in cfg.items():
            if isinstance(v, (dict, list)):
                continue
            f.write(f'| `{k}` | `{v}` |\n')
        f.write('\n')
        f.write('## Tensor manifest\n\n')
        f.write('| Name | Shape | DType |\n')
        f.write('|---|---:|---|\n')
        for name, shape, dt in tensor_rows:
            shape_s = 'x'.join(str(x) for x in shape)
            f.write(f'| `{name}` | `{shape_s}` | `{dt}` |\n')


def _print_stats(
    snap_file: str,
    *,
    repo: str,
    mag_dtype: dtype.DType,
    snap: SnapshotWriter,
    source_numbytes: int,
    elapsed: float,
) -> None:
    """Everything the file is made of, measured on the file itself, not estimated."""
    file_numbytes = os.path.getsize(snap_file)
    payload = snap.payload_numbytes
    blob = snap.blob_numbytes
    meta = snap.metadata_numbytes
    padding = blob-payload  # Inter tensor alignment
    container = file_numbytes-blob-meta  # Header plus the pad that puts the data section on a page
    table = Table(title=snap_file, title_style='bold', show_header=False, box=None, pad_edge=False)
    table.add_column(style='dim')
    table.add_column(justify='right')
    table.add_row('Source', repo)
    table.add_row('DType', mag_dtype.name)
    table.add_row('Tensors', f'{snap.tensor_count}')
    table.add_row('Payload', _fmt_bytes(payload))
    table.add_row('Alignment padding', f'{_fmt_bytes(padding)} ({padding/blob:.3%})')
    table.add_row('Data section', _fmt_bytes(blob))
    table.add_row('Metadata', _fmt_bytes(meta))
    table.add_row('Container overhead', _fmt_bytes(container))
    table.add_row('File size', _fmt_bytes(file_numbytes))
    table.add_row('Source shards', f'{_fmt_bytes(source_numbytes)} ({file_numbytes/source_numbytes:.2f}x)')
    table.add_row('Elapsed', f'{elapsed:.1f} s')
    table.add_row('Throughput', f'{_fmt_bytes(blob/elapsed)}/s')
    console.print()
    console.print(table)


def _convert_model(
    repo: str,
    torch_dtype: torch.dtype,
    mag_dtype: dtype.DType,
    *,
    write_model_card: bool = False,
    model_card_path: str = 'model_card.md',
) -> None:
    console.print(f'Downloading model {repo} from Hugging Face...', style='dim')
    repo_dir = snapshot_download(repo_id=repo)
    hf_config = _load_hf_config(repo_dir)

    plan = _plan_tensors(repo_dir)
    total_bytes = sum(entry.numbytes(mag_dtype) for entry in plan)
    source_numbytes = sum(os.path.getsize(shard) for shard in dict.fromkeys(entry.shard for entry in plan))
    snap_file: str = f'{repo.split("/")[-1].lower()}-{mag_dtype.short_name}.mag'
    console.print(f'Writing {len(plan)} tensors ({_fmt_bytes(total_bytes)} of {mag_dtype.short_name}) to {snap_file}', style='dim')

    metadata = {
        'source_repo': repo,
        'source_format': 'safetensors',
        'architecture': hf_config.get('model_type', 'qwen3'),
        'dtype': mag_dtype.name,
        'hf_config': hf_config,
    }
    start = time.perf_counter()
    with SnapshotWriter(snap_file, metadata) as snap:
        # Phase 1: the manifest. Shapes come from the headers, no weight data is touched.
        for entry in plan:
            snap.declare(entry.mag_key, entry.shape, mag_dtype)
        # Phase 2: the payload, one tensor at a time, in the order it was declared.
        with Progress(
            TextColumn('{task.fields[name]}', style='cyan'),
            BarColumn(),
            TaskProgressColumn(),
            DownloadColumn(binary_units=True),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task('convert', total=total_bytes, name='')
            for entry in plan:
                progress.update(task, name=f'{entry.mag_key[-34:]:<34}')
                snap.write(entry.mag_key, lambda entry=entry: _load_one(entry, torch_dtype, mag_dtype))
                progress.advance(task, entry.numbytes(mag_dtype))
    elapsed = time.perf_counter()-start

    if write_model_card:
        _write_model_card(
            model_card_path,
            repo=repo,
            snap_file=snap_file,
            mag_dtype=mag_dtype,
            cfg=hf_config,
            tensor_rows=[(entry.mag_key, entry.shape, mag_dtype.short_name) for entry in plan],
        )
        console.print(f'Model card saved to {model_card_path}', style='dim')
    _print_stats(
        snap_file,
        repo=repo,
        mag_dtype=mag_dtype,
        snap=snap,
        source_numbytes=source_numbytes,
        elapsed=elapsed,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description='Convert Hugging Face Qwen model to Magnetron file format')
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-4B-Instruct-2507',
        help='HF repo model name',
    )
    parser.add_argument(
        '--model-card',
        action='store_true',
        help='Write a Hugging Face-style model_card.md with tensor manifest',
    )
    parser.add_argument(
        '--model-card-path',
        type=str,
        default='model_card.md',
        help='Output path for the generated model card',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=['float16', 'bfloat16', 'float32'],
        help='Data type for Magnetron tensors',
    )
    args = parser.parse_args()
    mag_dtype = _mag_dtype_from_str(args.dtype)
    _convert_model(
        args.model,
        torch_dtype=_mag_to_torch_dtype(mag_dtype),
        mag_dtype=mag_dtype,
        write_model_card=args.model_card,
        model_card_path=args.model_card_path,
    )


if __name__ == '__main__':
    _main()
