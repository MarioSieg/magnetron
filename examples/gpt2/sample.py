import magnetron as mag
import tiktoken
from model import GPTConfig, GPT

mag.active_context().stop_grad_recorder()

start = 'Hello, who are you?\n'
max_new_tokens = 32
temp = 0.8

model = GPT.from_pretrained('gpt2')
enc = tiktoken.get_encoding('gpt2')

encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
decode = lambda l: enc.decode(l)

x = mag.Tensor.of(encode(start), dtype=mag.int32)[None, ...]
print(x)
y = model.generate(x, max_new_tokens, temp=temp)
print(y)
print(decode(y[0].tolist()))
