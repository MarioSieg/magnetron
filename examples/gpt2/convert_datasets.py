# This script downloads a pretrained GPT2 model from huggingface and converts it into Magnetron's custom file format.

import magnetron as mag
import magnetron.io as io
import torch
from magnetron import FFI, C
from transformers import GPT2LMHeadModel

MODEL_TYPES: set[str] = {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
MODEL_LAYOUTS: dict[str, dict[str, object]] = {
    'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
    'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
    'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
    'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
}

with io.StorageArchive('gpt2-fp32.mag', 'r') as sto:
    print(sto.metadata())
    exit(0)


def convert_hf_to_mag_file(model_type: str) -> None:
    assert model_type in MODEL_TYPES
    print(f'Converting GPT-2 model variant {model_type}')
    cfg = MODEL_LAYOUTS[model_type]
    cfg['vocab_size'] = 50257
    cfg['block_size'] = 1024
    cfg['bias'] = True
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.eval()
    state_dict = model.state_dict()
    with io.StorageArchive(f'{model_type}-fp32.mag', 'w') as out:
        for (
            k,
            v,
        ) in cfg.items():
            out[k] = v
        for k, v in state_dict.items():
            print(f'Converting tensor {k}, Shape: {v.shape}, DType: {v.dtype}, Size: {v.numel() * v.element_size() / 1024.0**2.0} MiB')
            assert v.is_contiguous()
            assert v.device == torch.device('cpu')
            mag_tensor = mag.Tensor.empty(v.shape)
            nb: int = v.numel() * v.element_size()
            C.mag_tensor_fill_from_raw_bytes(mag_tensor.native_ptr, FFI.cast('void*', v.data_ptr()), nb)
            out[k] = mag_tensor

        print(out.metadata())
        print(out.tensor_keys())


# for model in MODEL_LAYOUTS:
# convert_hf_to_mag_file(model)

convert_hf_to_mag_file('gpt2')
