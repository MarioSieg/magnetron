import magnetron as mag
from magnetron import optim, nn

# Create the model, optimizer, and loss function
model = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1), nn.Tanh())
optimizer = optim.SGD(model.parameters(), lr=1e-1)
criterion = nn.MSELoss()

# Data
x = mag.Tensor.from_data([[0, 0], [0, 1], [1, 0], [1, 1]])
y = mag.Tensor.from_data([[0], [1], [1], [0]])

# Train 2000 epochs
for epoch in range(2000):
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Print results
print([(int(y[i]), float(model(x)[i])) for i in range(4)])
