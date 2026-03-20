# Qwen3 inference

Qwen3 chat running on Magnetron. Includes a CLI, a local HTTP server, and a small library wrapper. Weights are loaded from Magnetron `.mag` snapshots; the first run downloads the snapshot and tokenizer.

## Install

From the repo root:

```bash
uv pip install -e .[examples]
```

Server deps:

```bash
uv pip install fastapi uvicorn
```

## Run (CLI)

Interactive REPL:

```bash
python examples/qwen3/main.py --repl
```

One-shot prompt:

```bash
python examples/qwen3/main.py --prompt "Explain KV caching in one paragraph."
```

## Run (server)

```bash
python examples/qwen3/server.py --port 8000
```

Endpoints:

- `GET /health`
- `POST /chat`
- `POST /chat/stream` (SSE)
