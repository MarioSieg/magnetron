# GPT-2 inference

Runs GPT-2 text generation implemented in Magnetron. Includes KV caching and optional streaming output. Uses `transformers` for weights and `tiktoken` for tokenization.

## Install

From the repo root:

```bash
uv pip install -e .[examples]
```

## Run

```bash
python examples/gpt2/main.py "What is the answer to life?"
```

Pick a model and generation settings:

```bash
python examples/gpt2/main.py "Write a haiku about compilers" --model gpt2-xl --max_tokens 128 --temp 0.7
```

Disable streaming:

```bash
python examples/gpt2/main.py "Hello" --no-stream
```

## Notes

- First run downloads model weights from Hugging Face.