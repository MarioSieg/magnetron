# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import argparse
import json
import os
import glob

import torch
from magnetron import Snapshot, Tensor, dtype
from magnetron._bootstrap import _C, _FFI
import gc
from qwen25 import Qwen25Model, Qwen25HyperParams
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


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


def _copy_from_torch(dst: Tensor, src: torch.Tensor, torch_dtype: torch.dtype, mag_dtype: dtype.DataType) -> None:
    if src.dtype != torch_dtype:
        src = src.to(torch_dtype)
    if not src.is_contiguous():
        src = src.contiguous()
    assert dst.is_contiguous
    assert dst.dtype == mag_dtype
    assert tuple(dst.shape) == tuple(src.shape), f'Shape mismatch: {tuple(dst.shape)} != {tuple(src.shape)}'
    assert dst.numel == src.numel(), f'Numel mismatch: {dst.numel} != {src.numel()}'
    nb = src.numel() * src.element_size()
    _C.mag_copy_raw_(dst._ptr, _FFI.cast('const void*', src.data_ptr()), nb)


def _convert_model(model: str, torch_dtype: torch.dtype, mag_dtype: dtype.DataType) -> None:
    repo = f'Qwen/{model}'
    print(f'Downloading model {model} from Hugging Face...')
    repo_dir = snapshot_download(repo_id=repo)
    cfg = Qwen25HyperParams()
    mag_model = Qwen25Model(cfg).cast(mag_dtype)
    sd_mag: dict[str, Tensor] = mag_model.state_dict()
    remaining = dict(sd_mag)
    SKIP = {'cos_cache', 'sin_cache'}
    for k in list(remaining.keys()):
        if k in SKIP:
            remaining.pop(k)

    def hf_key_for(mag_key: str) -> str:
        if mag_key == 'lm_head.weight' and getattr(cfg, 'tie_word_embeddings', False):
            return 'model.embed_tokens.weight'
        if mag_key.startswith('lm_head.'):
            return mag_key
        return 'model.' + mag_key

    snap_file = f'{model.lower()}-{mag_dtype.name}.mag'
    print(f'Writing snapshot to {snap_file}...')
    with Snapshot.write(snap_file) as snap:
        for shard_path in _iter_safetensor_shards(repo_dir):
            sd_hf = load_file(shard_path, device='cpu')
            to_remove = []
            for mag_k, t_mag in remaining.items():
                hf_k = hf_key_for(mag_k)
                w = sd_hf.get(hf_k)
                if w is None:
                    continue
                print(f'Converting {hf_k} -> {mag_k}  shape={tuple(w.shape)}')
                _copy_from_torch(t_mag, w, torch_dtype=torch_dtype, mag_dtype=mag_dtype)
                snap.put_tensor(mag_k, t_mag)
                to_remove.append(mag_k)
            for k in to_remove:
                remaining.pop(k)
            del sd_hf
            gc.collect()
            if not remaining:
                break
        if remaining:
            for mag_k, t_mag in remaining.items():
                if mag_k.endswith('.bias'):
                    print(f'Missing HF bias for {mag_k}; writing zeros')
                    t_mag.zeros_()
                    snap.put_tensor(mag_k, t_mag)
                else:
                    raise KeyError(f'Missing HF weight for magnetron key: {mag_k}')

        snap.print_info()

    print(f'Converted model saved to {snap_file}')


def _main() -> None:
    args = argparse.ArgumentParser(description='Convert Hugging Face Qwen model to Magnetron file format')
    args.add_argument('--model', type=str, default='Qwen2.5-3B-Instruct', help='HF repo model name')
    args.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'], help='Data type for Magnetron tensors')
    args = args.parse_args()
    torch_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else (torch.float16 if args.dtype == 'float16' else torch.float32)
    mag_dtype = dtype.bfloat16 if args.dtype == 'bfloat16' else (dtype.float16 if args.dtype == 'float16' else dtype.float32)
    _convert_model(args.model, torch_dtype=torch_dtype, mag_dtype=mag_dtype)


if __name__ == '__main__':
    _main()
