from __future__ import annotations

import argparse

from magnetron import nn, optim, context, Tensor, no_grad, dtype
import matplotlib.pyplot as plt


class AE(nn.Module):
    def __init__(self, w: int, h: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.w = w
        self.h = h
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * w * h, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 3 * w * h),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        y2 = self.decoder(self.encoder(x))
        return y2.view(x.shape[0], 3, self.w, self.h)


def _main() -> None:
    args = argparse.ArgumentParser(description='Autoencoder Example')
    args.add_argument('--image', type=str, default='media/logo.png', help='Path to input image')
    args.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    args.add_argument('--latent', type=int, default=16, help='Dimensionality of the latent space')
    args.add_argument('--width', type=int, default=64, help='Resized image width')
    args.add_argument('--height', type=int, default=64, help='Resized image height')
    args.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility')
    args.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')

    args = args.parse_args()
    print(args)

    context.manual_seed(args.seed)

    # Load and preprocess image
    image = Tensor.load_image(args.image, channels='RGB', resize_to=(args.width, args.height))  # Load image into uint8 CxHxW tensor
    image = (image.cast(dtype.float32) / 255)[None, ...]  # Convert uint8 -> float tensor and normalize [0, 255) to [0, 1) and insert batch dim

    # Initialize model, loss function, and optimizer
    model = AE(w=args.width, h=args.height, latent_dim=args.latent)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the autoencoder
    model.train()
    losses: list[float] = []

    for step in range(args.epochs):
        y_hat = model(image)
        loss = criterion(y_hat, image)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f'Epoch [{step}/{args.epochs}] Loss: {loss.item():.6f}')

    print('Training complete')

    # Reconstruct image
    model.eval()
    with no_grad():
        reconstructed = model(image)

    # Plot original and reconstructed images
    original = image[0].permute(1, 2, 0).tolist()
    reconstructed = reconstructed[0].permute((1, 2, 0)).tolist()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(original)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Reconstructed')
    plt.imshow(reconstructed)
    plt.axis('off')
    plt.tight_layout()
    # plt.savefig('autoencoder_result.png', dpi=300)

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss over Time')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('autoencoder_loss.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    _main()
