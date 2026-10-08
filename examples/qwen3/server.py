# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import argparse
import json
import time
import uuid
import uvicorn

from inference import InferenceConfig, InferenceEngine
from typing import Any, Literal
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = Field(default=None, alias='max_completion_tokens')
    temperature: float | None = None
    top_k: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    top_p: float | None = None
    stop: Any | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None


class OpenAIServer:
    def __init__(self, engine: InferenceEngine, model_name: str) -> None:
        self.engine = engine
        self.model_name = model_name
        self.app = FastAPI(title='Magnetron OpenAI-Compatible API')
        self._register_routes()

    def _register_routes(self) -> None:
        self.app.add_api_route('/v1/models', self.list_models, methods=['GET'])
        self.app.add_api_route('/v1/chat/completions', self.chat_completions, methods=['POST'])

    def list_models(self) -> dict[str, Any]:
        return {'object': 'list', 'data': [{'id': self.model_name, 'object': 'model', 'created': int(time.time()), 'owned_by': 'magnetron'}]}

    def chat_completions(self, req: ChatCompletionRequest, authorization: str | None = Header(default=None)) -> Any:
        prompt = self._format_prompt(req.messages, req.tools)
        if req.stream:
            return StreamingResponse(self._stream_chat(req, prompt), media_type='text/event-stream')

        return self._complete_chat(req, prompt)

    def _complete_chat(self, req: ChatCompletionRequest, prompt: str) -> dict[str, Any]:
        completion_id = self._completion_id()
        created = int(time.time())
        text = self.engine.gen_one_shot(prompt, max_tokens=req.max_tokens, temp=req.temperature, top_k=req.top_k)
        tool_calls = self._parse_tool_calls(text) if req.tools else None
        message: dict[str, Any] = {'role': 'assistant', 'content': None if tool_calls else text}
        finish_reason = 'stop'
        if tool_calls:
            message['tool_calls'] = tool_calls
            finish_reason = 'tool_calls'

        return {
            'id': completion_id,
            'object': 'chat.completion',
            'created': created,
            'model': req.model,
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
        }

    def _stream_chat(self, req: ChatCompletionRequest, prompt: str):
        completion_id = self._completion_id()
        created = int(time.time())
        yield self._sse(
            {
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': req.model,
                'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}],
            }
        )

        for chunk in self.engine.gen_stream(prompt, max_tokens=req.max_tokens, temp=req.temperature, top_k=req.top_k):
            if not chunk:
                continue
            yield self._sse(
                {
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': req.model,
                    'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}],
                }
            )
        yield self._sse(
            {
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': req.model,
                'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
            }
        )
        yield 'data: [DONE]\n\n'

    def _format_prompt(self, messages: list[ChatMessage], tools: list[dict[str, Any]] | None) -> str:
        parts: list[str] = []
        system_seen = False
        for msg in messages:
            if msg.role == 'system':
                system_seen = True
                parts.append(f'<|im_start|>system\n{msg.content or ""}{self._render_tools(tools)}<|im_end|>')
            elif msg.role == 'user':
                parts.append(f'<|im_start|>user\n{msg.content or ""}<|im_end|>')
            elif msg.role == 'assistant':
                content = msg.content or ''
                if msg.tool_calls:
                    content = json.dumps({'tool_calls': msg.tool_calls}, ensure_ascii=False)
                parts.append(f'<|im_start|>assistant\n{content}<|im_end|>')
            elif msg.role == 'tool':
                parts.append(f'<|im_start|>tool name={msg.name or ""} tool_call_id={msg.tool_call_id or ""}\n{msg.content or ""}<|im_end|>')
        if not system_seen and tools:
            parts.insert(0, f'<|im_start|>system\nYou are a helpful assistant.{self._render_tools(tools)}<|im_end|>')
        parts.append('<|im_start|>assistant\n')
        return '\n'.join(parts)

    def _render_tools(self, tools: list[dict[str, Any]] | None) -> str:
        if not tools:
            return ''

        return (
            '\n\nYou may call tools by replying with ONLY valid JSON:\n'
            '{"tool_calls":[{"name":"function_name","arguments":{...}}]}\n\n'
            'Available tools:\n'
            f'{json.dumps(tools, ensure_ascii=False, indent=2)}'
        )

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]] | None:
        s = text.strip()

        if s.startswith('```'):
            s = s.strip('`').strip()
            if s.startswith('json'):
                s = s[4:].strip()

        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None

        raw_calls = obj.get('tool_calls')
        if not isinstance(raw_calls, list):
            return None

        calls: list[dict[str, Any]] = []

        for raw in raw_calls:
            name = raw.get('name')
            arguments = raw.get('arguments', {})

            if not isinstance(name, str):
                continue

            calls.append(
                {
                    'id': f'call_{uuid.uuid4().hex[:24]}',
                    'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)},
                }
            )

        return calls or None

    @staticmethod
    def _completion_id() -> str:
        return f'chatcmpl-{uuid.uuid4().hex}'

    @staticmethod
    def _sse(obj: dict[str, Any]) -> str:
        return f'data: {json.dumps(obj, ensure_ascii=False)}\n\n'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model-name', default='magnetron-qwen3-4b')

    parser.add_argument('--system', default='You are a helpful assistant.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-ctx', type=int, default=4096)
    parser.add_argument('--reserve-gen', type=int, default=1024)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--temp', type=float, default=0.6)
    parser.add_argument('--top-k', type=int, default=200)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--snapshot', type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = InferenceConfig.from_args(args)
    engine = InferenceEngine(config)

    server = OpenAIServer(engine=engine, model_name=args.model_name)

    uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
