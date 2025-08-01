import magnetron as mag
import tiktoken
from model import GPTConfig, GPT

mag.active_context().stop_grad_recorder()

start = 'Hello, who are you?\n'
num_samples = 10
max_new_tokens = 500
temp = 0.8
top_k = 200
seed = 1337

gpt = GPT(GPTConfig())

enc = tiktoken.get_encoding('gpt2')

encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
decode = lambda l: enc.decode(l)

x = mag.Tensor.of(encode(start), dtype=mag.int32)[None, ...]
print(x)
y = gpt.generate(x, 4, temp=1.0)
print(y)
print(decode(y[0].tolist()))
