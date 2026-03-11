# Qwen3 inference example

Run Qwen3 chat with Magnetron. Three entrypoints:

- **`main.py`** — CLI: interactive REPL (`--repl`) or one-shot answer (`--prompt "..."`). Options: `--max_tokens`, `--temp`, `--system`, `--max_ctx`, etc.
- **`server.py`** — Local HTTP API. `POST /chat/stream` (SSE) or `POST /chat` (JSON) with `{"messages": [{"role":"user","content":"..."}]}`. Default: `http://127.0.0.1:8000`.
- **`inference.py`** — Library: `InferenceConfig` (dataclass with defaults or `from_args(namespace)` for CLI), `InferenceEngine(config)` (loads model, exposes `stream_chat` / `async_stream_chat` / `one_shot_answer`). Snapshot and tokenizer are downloaded on first use.

**Run (needs uv, magnetron deps):**

```bash
cd magnetron && uv venv && source .venv/bin/activate
uv pip install . huggingface_hub tokenizers rich
# REPL
python examples/qwen3/main.py --repl
# Server (add: fastapi uvicorn)
python examples/qwen3/server.py --port 8000
```
