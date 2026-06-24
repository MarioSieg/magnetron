import matplotlib.pyplot as plt
from magnetron import Tensor
xs = Tensor.linspace(-5, 5, steps=100)
ys = Tensor.linspace(-5, 5, steps=100)
x, y = Tensor.meshgrid(xs, ys)
z = (x**2 + y**2).sqrt().sin()
ax = plt.axes(projection='3d')
ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
plt.savefig('meow.png')
