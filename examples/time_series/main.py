from __future__ import annotations

import math
import pandas as pd
from matplotlib import pyplot as plt
from magnetron import nn, optim, dtype, Tensor, no_grad
from magnetron.utils import data
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / 'airline_passengers.csv')
timeseries = Tensor(df[['Passengers']].values).cast(dtype.float32)
train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]


def mk_dataset(dataset: Tensor, window: int) -> tuple[Tensor, Tensor]:
    x, y = [], []
    for i in range(len(dataset) - window):
        x.append(dataset[i : i + window])
        y.append(dataset[i + 1 : i + window + 1])

    return Tensor.stack(x), Tensor.stack(y)


window = 4
x_train, y_train = mk_dataset(train, window=window)
x_test, y_test = mk_dataset(test, window=window)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.lstm(x)
        return self.linear(x)


model = Model()
optim = optim.Adam(model.parameters())
criterion = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=8)
epochs = 2000
for epoch in range(epochs):
    model.train()
    for xb, yb in loader:
        y_hat: Tensor = model(xb)
        loss: Tensor = criterion(y_hat, yb)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if epoch % 100 != 0:
            continue
        model.eval()  # validation
        with no_grad():
            y_hat = model(x_train)
            train_rmse = criterion(y_hat, y_train).sqrt()
            y_hat = model(x_test)
            test_rmse = criterion(y_hat, y_test).sqrt()
            print(f'Epoch {epoch}: Train RMSE {train_rmse.item()}, Test RMSE {test_rmse.item()}')

with no_grad():
    train_plot = Tensor.ones_like(timeseries) * math.nan
    y_hat = model(x_train)[:, -1, :]
    train_plot[window:train_size] = model(x_train)[:, -1, :]
    test_plot = Tensor.ones_like(timeseries) * math.nan
    test_plot[train_size + window : len(timeseries)] = model(x_test)[:, -1, :]

plt.plot(timeseries, c='b')
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()
