# This example demonstrates how to implement a simple XOR neural network using Magnetron.
# The XOR problem is a classic example in machine learning, where the model learns to output 1 if the inputs are different and 0 if they are the same.
# The model consists of two linear layers with a tanh activation function.
# The model is trained using the Mean Squared Error (MSE) loss function and the Stochastic Gradient Descent (SGD) optimizer.

import magnetron as mag
from magnetron import optim, nn

EPOCHS: int = 2000

# Create the model, optimizer, and loss function
model = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1), nn.Tanh())
print(model.state_dict())
optimizer = optim.SGD(model.parameters(), lr=1e-1)
criterion = nn.MSELoss()
loss_values: list[float] = []

x = mag.Tensor.of([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = mag.Tensor.of([[0.0], [1.0], [1.0], [0.0]])

# Train the model
for epoch in range(EPOCHS):
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_values.append(loss.item())

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Print the final predictions after the training
print('=== Final Predictions ===')

with mag.no_grad():
    y_hat = model(x)
    for i in range(x.shape[0]):
        print(f'Expected: {y[i]}, Predicted: {y_hat[i]}')

# Plot the loss

try:
    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss over Time')
    plt.grid(True)
    plt.show()

except ImportError:
    print('matplotlib not installed; skipping loss plot.')
