# 🧠 Qwen 2.5 Inference Example

Run text generation with a Qwen2.5 model using the Magnetron framework.

## Hardware Note!
The BF16 matmul is kernel is still not optimized, so we use float32 for now.<br>
Make sure to at least have 32GB of RAM :D

## Bash code to run locally
! Assumes git and uv installed.
```
git clone -b develop https://github.com/MarioSieg/magnetron
cd magnetron
uv venv && source .venv/bin/activate
uv pip install . huggingface_hub tokenizers rich
python examples/qwen25/main.py --repl
```
   