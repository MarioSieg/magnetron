from __future__ import annotations

from magnetron import nn, optim, Tensor
from matplotlib import pyplot as plt

EPOCHS: int = 2000
LR: float = 1e-1

# Define the XOR input and output data
x = Tensor.of([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = Tensor.of([[0.0], [1.0], [1.0], [0.0]])

# Create the model, loss function, and optimizer
model = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1), nn.Tanh())
optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# Train the model
losses: list[float] = []
for epoch in range(EPOCHS):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Print predictions
y_hat = model(x)
for i in range(x.shape[0]):
    print(f'Expected: {y[i].item()}, Predicted: {y_hat[i].item()}')

# Plot results
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss over Time')
plt.grid(True)
plt.show()
