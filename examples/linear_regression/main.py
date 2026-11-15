from __future__ import annotations

from magnetron import Tensor, nn, optim, dtype
from matplotlib import pyplot as plt

EPOCHS: int = 100
LR: float = 1e-1

N = 100
x = -1.0 + Tensor.arange(0, N, 1, dtype=dtype.float32).reshape(N, 1) * (2.0 / (N - 1))
noise = Tensor.normal((N, 1), mean=0.0, std=0.2)
y = 3.0 * x + 0.5 + noise

# Create model, loss function, and optimizer
model = nn.Sequential(nn.Linear(1, 1))
criterion = nn.MSELoss()
opt = optim.SGD(model.parameters(), lr=LR)

# Train the model
losses = []
for epoch in range(EPOCHS):
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    losses.append(loss.item())

y_line = model(x)  # Fitted line

# Plot results
plt.subplot(1, 2, 1)
plt.scatter([x[i].item() for i in range(x.numel)], [y[i].item() for i in range(y.numel)], s=16, label='Data')
plt.plot([x[i].item() for i in range(x.numel)], [y_line[i].item() for i in range(y_line.numel)], c='r', label='Fit')
plt.legend()
plt.title('Linear Regression Fit')

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title('Loss')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()
