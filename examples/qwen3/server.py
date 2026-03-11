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
import uvicorn

from inference import InferenceEngine, InferenceConfig
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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


@app.post('/chat/stream')
async def chat_stream(req: ChatRequest):
    history = [(m.role, m.content) for m in req.messages]

    async def generate():
        try:
            async for chunk in engine.async_stream_chat(history):
                yield f'event: token\ndata: {json.dumps({"token": chunk})}\n\n'
        except Exception as e:
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'},
    )


@app.post('/chat')
async def chat(req: ChatRequest):
    history = [(m.role, m.content) for m in req.messages]
    parts = []
    async for chunk in engine.async_stream_chat(history):
        parts.append(chunk)
    return {'response': ''.join(parts)}


def _main():
    parser = argparse.ArgumentParser(description='Magnetron Qwen3 inference server')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind')
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    _main()
