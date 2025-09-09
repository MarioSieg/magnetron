# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

from magnetron import io, Tensor
from transformers import GPT2LMHeadModel
import os
import torch

def test_read_write_gpt2_weights():
    cfg = dict(n_layer=12, n_head=12, n_embd=768)
    cfg['vocab_size'] = 50257
    cfg['block_size'] = 1024
    cfg['bias'] = True
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    state_dict = model.state_dict()
    with io.StorageArchive(f'gpt2-e8m23.mag', 'w') as dataset_out:
        for k, v in cfg.items():
            print(k, v)
            dataset_out[k] = v
        for k, v in state_dict.items():
            assert v.device == torch.device('cpu')
            mag_tensor = Tensor.of(v.tolist())
            dataset_out[k] = mag_tensor
    assert os.path.exists('gpt2-e8m23.mag')
    # Now compare:
    with io.StorageArchive(f'gpt2-e8m23.mag', 'w') as dataset_in:
        meta = dataset_in.metadata()
        assert meta['n_layer'] == cfg['n_layer']
        assert meta['n_head'] == cfg['n_head']
        assert meta['n_embd'] == cfg['n_embd']
        assert meta['vocab_size'] == cfg['vocab_size']
        assert meta['block_size'] == cfg['block_size']
        assert meta['bias'] == cfg['bias']
        for k, v in state_dict.items():
            mag_tensor = dataset_in[k]
            assert isinstance(mag_tensor, Tensor)
            torch_tensor = torch.tensor(mag_tensor.tolist())
            assert torch.equal(torch_tensor, v)
        os.remove('gpt2-e8m23.mag')
    assert not os.path.exists('gpt2-e8m23.mag')

