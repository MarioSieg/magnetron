# 🧠 Qwen 3 Inference Example

Run text generation with a Qwen3 model using the Magnetron framework.

## Bash code to run locally
! Assumes git and uv installed.
```
git clone -b develop https://github.com/MarioSieg/magnetron
cd magnetron
uv venv && source .venv/bin/activate
uv pip install . huggingface_hub tokenizers rich
python examples/qwen3/main.py --repl
```
   