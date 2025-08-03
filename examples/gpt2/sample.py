import time

import magnetron as mag
import tiktoken
from model import GPTConfig, GPT

mag.active_context().stop_grad_recorder()

start = 'Hello, who are you?\n'
max_new_tokens = 64
temp = 0.9

model = GPT.from_pretrained('gpt2')
enc = tiktoken.get_encoding('gpt2')

encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
decode = lambda l: enc.decode(l)

x = mag.Tensor.of(encode(start), dtype=mag.int32)[None, ...]
start = time.perf_counter()
y = model.generate(x, max_new_tokens, temp=temp)
elapsed = time.perf_counter() - start
print(f"Generated in: {elapsed:.9f} seconds")
print(decode(y[0].tolist()))
