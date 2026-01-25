# 🧠 Qwen 2.5 Inference Example

Run text generation with a Qwen2.5 model using the Magnetron framework.

## Hardware Note!
The BF16 matmul is kernel is still not optimized, so we use float32 for now.<br>
Make sure to at least have 32GB of RAM :D

## Bash code to run locally
! Assumes git and uv installed. torch huggingface_hub safetensors transformers are only required for the conversion step.
```
git clone -b develop https://github.com/MarioSieg/magnetron
cd magnetron
uv venv && source .venv/bin/activate
uv pip install . torch huggingface_hub safetensors rich transformers
python examples/qwen25/convert_dataset.py --dtype=float32
python examples/qwen25/main.py --repl
```
   