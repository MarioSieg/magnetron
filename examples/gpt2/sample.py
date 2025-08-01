import magnetron as mag
from model import GPTConfig, GPT

mag.active_context().stop_grad_recorder()

start = '\n'
num_samples = 10
max_new_tokens = 500
temp = 0.8
top_k = 200
seed = 1337

gpt = GPT(GPTConfig())
