from magnetron import nn, optim, Tensor

W, H = 32, 32
x = Tensor.normal(1, 1, W, H)


class AE(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(W * H, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, W * H),
        )

    def forward(self, x: Tensor) -> Tensor:
        y2 = self.decoder(self.encoder(x))
        return y2.view(x.shape[0], 1, W, H)


model = AE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):
    optimizer.zero_grad()
    y = model(x)
    loss = criterion(y, x)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f'Epoch [{step}/1000] Loss: {loss.item():.6f}')
