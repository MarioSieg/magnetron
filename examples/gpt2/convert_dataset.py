# This script downloads a pretrained GPT2 model from huggingface and converts it into Magnetron's custom file format.
# Only needed if you want a different GPT2 dataset, as the example downloads the already converted files automatically.

import magnetron as mag
import magnetron.io as io
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoConfig

OUTPUT_FILE: str = 'gpt2.mag'


def download_gpt2(model_name: str = 'gpt2') -> tuple[dict, 'PretrainedConfig', 'PreTrainedTokenizerBase']:
    print('Downloading GPT-2 model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    state_dict = model.state_dict()
    config = AutoConfig.from_pretrained(model_name)
    return state_dict, config, tokenizer


state_dict, config, tokenizer = download_gpt2()

with io.StorageStream() as storage:
    for key, val in state_dict.items():
        print(f'Converting {key} ({val.size()}) {val.dtype}')
        storage.put(key, mag.Tensor.of(val.tolist()))
    print(storage.tensor_keys())
    storage.serialize(OUTPUT_FILE)
    print(f'GPT-2 model converted and saved to {OUTPUT_FILE}')
