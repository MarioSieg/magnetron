from magnetron import Tensor, optim, nn, dtype
import matplotlib.pyplot as plt
from torchvision import datasets

BATCH_SIZE = 64
EPOCHS = 5

train_raw = datasets.MNIST(root='./data', train=True, download=True)
test_raw = datasets.MNIST(root='./data', train=False, download=True)

X_train = Tensor.of((train_raw.data.float() / 255.0).tolist())
y_train = Tensor.of(train_raw.targets.tolist()).cast(dtype.int64)
print(X_train.shape)
print(y_train.shape)

X_test = Tensor.of((test_raw.data.float() / 255.0).tolist())
y_test = Tensor.of(test_raw.targets.tolist())
print(X_test.shape)
print(y_test.shape)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

n_samples = len(X_train)
losses = []

for epoch in range(EPOCHS):
    indices = Tensor.rand_perm(n_samples)
    for i in range(0, n_samples, BATCH_SIZE):
        batch_idx = indices[i: i + BATCH_SIZE]
        batch_X = X_train[batch_idx]
        batch_y = y_train[batch_idx]

        output = model(batch_X)
        loss = criterion(output, batch_y)

        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()
        if i % 1000 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {loss.item():.4f}')

plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss over Time')
plt.grid(True)
plt.show()
