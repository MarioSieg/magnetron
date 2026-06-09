# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# +---------------------------------------------------------------------+

import argparse
import json
import os
import glob
import gc

from magnetron import Snapshot, Tensor, dtype, context
from model import Qwen3Model, Qwen3HyperParams
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import torch


def _mag_to_torch_dtype(mag_dtype: dtype.DType) -> torch.dtype:
    return {
        dtype.float16: torch.float16,
        dtype.bfloat16: torch.bfloat16,
        dtype.float32: torch.float32,
        dtype.float8_e4m3fn: torch.float8_e4m3fn,
    }[mag_dtype]


def _mag_dtype_from_str(dtype_str: str) -> dtype.DType:
    return {
        'float16': dtype.float16,
        'bfloat16': dtype.bfloat16,
        'float32': dtype.float32,
        'mixed_fp8': dtype.float8_e4m3fn,
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


def _quantize(x: Tensor) -> tuple[Tensor, Tensor]:
    qtype = dtype.float8_e4m3fn
    highp = x.cast(dtype.float32)
    amax = float(highp.abs().max().item())
    fp8max = qtype.max() if callable(qtype.max) else qtype.max
    scale = 1.0 if amax < 1e-12 or not (amax == amax) else amax / fp8max
    inv_scale = 1.0 / scale if scale != 0.0 else 1.0
    q = (highp * inv_scale).clamp(-fp8max, fp8max).cast(qtype)
    return q, Tensor([scale], dtype=dtype.float32)


def _write_model_card(
    path: str,
    *,
    repo: str,
    snap_file: str,
    mag_dtype: dtype.DType,
    fp8w_mode: bool,
    cfg: Qwen3HyperParams,
    tensor_rows: list[tuple[str, tuple[int, ...], str]],
) -> None:
    tensor_rows = sorted(tensor_rows, key=lambda x: x[0])
    model_name = repo.split('/')[-1]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# {model_name} Magnetron Snapshot\n\n')
        f.write(f'This repository contains a Magnetron snapshot converted from the original Hugging Face model `{repo}`.\n\n')
        f.write('The snapshot is intended for inference with the Magnetron runtime. ')
        if fp8w_mode:
            f.write(
                'Weights are stored in mixed FP8 mode using `float8_e4m3fn` weight tensors '
                'and per-tensor `float32` scale tensors, while activations/default runtime '
                'dtype are kept in `bfloat16`.\n\n'
            )
        else:
            f.write(f'All convertible tensors are stored using `{mag_dtype.short_name}` where applicable.\n\n')
        f.write('## Model details\n\n')
        f.write(f'- **Source model:** `{repo}`\n')
        f.write(f'- **Snapshot file:** `{snap_file}`\n')
        f.write(f'- **Magnetron dtype mode:** `{mag_dtype.short_name}`\n')
        f.write(f'- **Mixed FP8 weights:** `{fp8w_mode}`\n')
        f.write(f'- **Tensor count:** `{len(tensor_rows)}`\n\n')
        f.write('## Qwen3 configuration\n\n')
        f.write('| Field | Value |\n')
        f.write('|---|---:|\n')
        for k, v in vars(cfg).items():
            f.write(f'| `{k}` | `{v}` |\n')
        f.write('\n')
        f.write('## Tensor manifest\n\n')
        f.write('| Name | Shape | DType |\n')
        f.write('|---|---:|---|\n')
        for name, shape, dt in tensor_rows:
            shape_s = 'x'.join(str(x) for x in shape)
            f.write(f'| `{name}` | `{shape_s}` | `{dt}` |\n')


def _convert_model(
    repo: str,
    torch_dtype: torch.dtype,
    mag_dtype: dtype.DType,
    *,
    write_model_card: bool = False,
    model_card_path: str = 'model_card.md',
) -> None:
    skip: set[str] = {'cos_cache', 'sin_cache'}
    print(f'Downloading model {repo} from Hugging Face...')
    repo_dir = snapshot_download(repo_id=repo)
    cfg = Qwen3HyperParams()
    fp8w_mode = mag_dtype == dtype.float8_e4m3fn
    print(f'fp8 mode: {fp8w_mode}')
    if fp8w_mode:
        context.set_default_dtype(dtype.bfloat16)
        activation_torch_dtype = torch.bfloat16
        cfg.quant_dtype = dtype.float8_e4m3fn
    else:
        context.set_default_dtype(mag_dtype)
        activation_torch_dtype = torch_dtype
        cfg.quant_dtype = None
    mag_model = Qwen3Model(cfg)
    if not fp8w_mode:
        mag_model = mag_model.cast(mag_dtype)
    sd_mag: dict[str, Tensor] = mag_model.state_dict()
    remaining = dict(sd_mag)
    for k in list(remaining.keys()):
        if k in skip:
            remaining.pop(k)

    def hf_key_for(mag_key: str) -> str:
        if mag_key == 'lm_head.weight' and getattr(cfg, 'tie_word_embeddings', False):
            return 'model.embed_tokens.weight'
        if mag_key.startswith('lm_head.'):
            return mag_key
        return 'model.' + mag_key

    scale_keys: set[str] = {k for k in remaining if k.endswith('.weight_scale')}
    snap_file: str = f'{repo.split("/")[1].lower()}-{mag_dtype.short_name}.mag'
    tensor_manifest: list[tuple[str, tuple[int, ...], str]] = []
    print(f'Writing snapshot to {snap_file}...')
    with Snapshot.write(snap_file) as snap:
        for shard_path in _iter_safetensor_shards(repo_dir):
            hf_state_dict: dict[str, torch.Tensor] = load_file(shard_path, device='cpu')
            processed_stack: list[str] = []
            for key, tensor in remaining.items():
                if fp8w_mode and key in scale_keys:
                    continue
                hf_key: str = hf_key_for(key)
                torch_tensor: torch.Tensor | None = hf_state_dict.get(hf_key)
                if torch_tensor is None:
                    continue
                scaled_quant: bool = key.endswith('.weight') and (f'{key}_scale' in scale_keys)
                if fp8w_mode and scaled_quant:
                    mag_tensor = Tensor(torch_tensor.to('cpu').contiguous()).cast(dtype.bfloat16)
                    quantized, scale = _quantize(mag_tensor)
                    print(f'Quantized {hf_key} -> {key}, Shape={tuple(torch_tensor.shape)}, Scale={scale.item()}')
                    snap.put_tensor(key, quantized)
                    tensor_manifest.append((key, tuple(quantized.shape), quantized.dtype.short_name))
                    scale_key: str = f'{key}_scale'
                    snap.put_tensor(scale_key, scale)
                    tensor_manifest.append((scale_key, tuple(scale.shape), scale.dtype.short_name))
                    processed_stack.append(key)
                    processed_stack.append(scale_key)
                else:
                    target_dtype = tensor.dtype
                    print(f'Converting {hf_key} -> {key} shape={tuple(torch_tensor.shape)} dtype={target_dtype.short_name}')
                    if target_dtype == dtype.float32:
                        tt = torch_tensor.to(torch.float32)
                    elif target_dtype == dtype.float16:
                        tt = torch_tensor.to(torch.float16)
                    elif target_dtype == dtype.bfloat16:
                        tt = torch_tensor.to(torch.bfloat16)
                    elif target_dtype == dtype.float8_e4m3fn:
                        tt = torch_tensor.to(torch.float8_e4m3fn)
                    else:
                        tt = torch_tensor.to(activation_torch_dtype)
                    tt = tt.to('cpu').contiguous()
                    out_tensor = Tensor(tt, dtype=target_dtype)
                    snap.put_tensor(key, out_tensor)
                    tensor_manifest.append((key, tuple(out_tensor.shape), out_tensor.dtype.short_name))
                    processed_stack.append(key)

            for k in processed_stack:
                remaining.pop(k)
            del hf_state_dict
            gc.collect()
            if not remaining:
                break
        if remaining:
            for key, tensor in list(remaining.items()):
                if key.endswith('.bias'):
                    print(f'Missing HF bias for {key}; writing zeros')
                    tensor.zeros_()
                    snap.put_tensor(key, tensor)
                    tensor_manifest.append((key, tuple(tensor.shape), tensor.dtype.short_name))
                    remaining.pop(key)
                elif fp8w_mode and key.endswith('.weight_scale'):
                    raise KeyError(f'Orphan weight_scale entry: {key}')
                else:
                    raise KeyError(f'Missing HF weight for magnetron key: {key}')
        snap.print_info()
    if write_model_card:
        _write_model_card(
            model_card_path,
            repo=repo,
            snap_file=snap_file,
            mag_dtype=mag_dtype,
            fp8w_mode=fp8w_mode,
            cfg=cfg,
            tensor_rows=tensor_manifest,
        )
        print(f'Model card saved to {model_card_path}')
    print(f'Converted model saved to {snap_file}')


def _main() -> None:
    parser = argparse.ArgumentParser(description='Convert Hugging Face Qwen model to Magnetron file format')
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-4B-Instruct-2507',
        help='HF repo model name',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='mixed_fp8',
        choices=['mixed_fp8', 'float16', 'bfloat16', 'float32'],
        help='Data type for Magnetron tensors',
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
