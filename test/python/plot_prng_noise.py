# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

noise = mag.Tensor.uniform(32, 32, 3)

plt.imshow(np.array(noise.permute(1, 0, 2).tolist()))
plt.axis('off')
plt.show()
