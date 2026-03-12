# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

# Local inference server. Run: python server.py [--port 8000]
# Clients POST /chat/stream with {"messages": [{"role":"user","content":"..."}]} and receive SSE stream of tokens

import argparse
import json
import logging
import uvicorn

from inference import InferenceEngine, InferenceConfig
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("magnetron.inference")
_MAX_LOG = 300

engine = InferenceEngine(InferenceConfig())
app = FastAPI(title='Magnetron Qwen3 inference')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.get('/health')
async def health():
    return {'status': 'ok'}


def _last_user_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            raw = m.content or ""
            s = raw.strip().replace("\n", " ")[:_MAX_LOG]
            return s + "..." if len(raw) > _MAX_LOG else s
    return ""

@app.post('/chat/stream')
async def chat_stream(req: ChatRequest):
    history = [(m.role, m.content) for m in req.messages]
    last_user = _last_user_text(req.messages)
    logger.info("IN (%d msgs) user: %s", len(history), repr(last_user) if last_user else "(none)")

    async def generate():
        full: list[str] = []
        try:
            async for chunk in engine.async_stream_chat(history):
                full.append(chunk)
                yield f'event: token\ndata: {json.dumps({"token": chunk})}\n\n'
            out = "".join(full)
            logger.info("OUT %s", repr((out[: _MAX_LOG] + "..." if len(out) > _MAX_LOG else out)))
        except Exception as e:
            logger.exception("stream error: %s", e)
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'},
    )


@app.post('/chat')
async def chat(req: ChatRequest):
    history = [(m.role, m.content) for m in req.messages]
    last_user = _last_user_text(req.messages)
    logger.info("IN (%d msgs) user: %s", len(history), repr(last_user) if last_user else "(none)")
    parts = []
    async for chunk in engine.async_stream_chat(history):
        parts.append(chunk)
    out = ''.join(parts)
    logger.info("OUT %s", repr((out[:_MAX_LOG] + "..." if len(out) > _MAX_LOG else out)))
    return {'response': out}


def _main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description='Magnetron Qwen3 inference server')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind')
    args = parser.parse_args()
    logger.info("Starting inference server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    _main()
