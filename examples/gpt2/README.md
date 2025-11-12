# 🧠 GPT-2 Inference Example

Run text generation with a GPT-2 model using the Magnetron framework.  
Supports streaming output and standard GPT-2 model variants.

## 📚 Description

- Loads a pretrained GPT-2 (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`) via `transformers`
- Uses `tiktoken` for tokenization with the GPT-2 vocabulary
- Implements causal self-attention, MLP blocks, and KV caching
- Provides both **streaming** and **non-streaming** generation modes
- Prints throughput (tokens/s) after generation

## 🚀 Usage

Basic run (streams tokens by default):

```bash
python main.py "What is the answer to life?"
```

Specify model and parameters:

```bash
python main.py "What is the answer to life?" --model gpt2-xl --max_tokens 128 --temp 0.7
```

Disable streaming:

```bash
python main.py "What is the answer to life?" --no-stream
```

### Arguments

- `prompt` (positional): prompt text to start generation
- `--model`: one of `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` (default: `gpt2`)
- `--max_tokens`: number of new tokens to generate (default: `128`)
- `--temp`: sampling temperature (default: `0.6`)
- `--no-stream`: disable incremental output

## ⚙️ Requirements

Install the minimal dependencies:

```bash
uv pip install magnetron tiktoken transformers rich
```

> Note: the first run will download model weights from Hugging Face.