# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
from magnetron import optim, nn

def test_xor_network():
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

    assert round(model(x)[0]) == 0
    assert round(model(x)[1]) == 1
    assert round(model(x)[2]) == 1
    assert round(model(x)[3]) == 0
