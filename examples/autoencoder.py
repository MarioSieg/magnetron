from magnetron import nn, optim, context, Tensor, no_grad
import matplotlib.pyplot as plt

context.manual_seed(42)

EPOCHS: int = 100
W, H = 64, 64
image = Tensor.load_image('media/logo.png', channels='RGB', resize_to=(W, H))[None, ...]

class AE(nn.Module):
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * W * H, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 3 * W * H),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        y2 = self.decoder(self.encoder(x))
        return y2.view(x.shape[0], 3, W, H)


model = AE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for step in range(EPOCHS):
    y_hat = model(image)
    loss = criterion(y_hat, image)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % 10 == 0:
        print(f'Epoch [{step}/{EPOCHS}] Loss: {loss.item():.6f}')

print('Training complete, showing results...')

with no_grad():
    reconstructed = model(image)

orig_hwc = image[0].permute(1, 2, 0).tolist()
recon_hwc = reconstructed[0].permute((1, 2, 0)).tolist()
plt.figure(dpi=300)
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(orig_hwc)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Reconstructed')
plt.imshow(recon_hwc)
plt.axis('off')
plt.tight_layout()
plt.savefig('autoencoder_result.png')
plt.show()
