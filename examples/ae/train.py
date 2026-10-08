# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from model import CORRUPTIONS, DenoisingAE, corrupt, preprocess

from magnetron import Tensor, context, dtype, nn, no_grad, optim
from magnetron.snapshot import deserialize


def _batches(images: Tensor, batch_size: int, shuffle: bool) -> list[Tensor]:
    order = list(range(images.shape[0]))
    if shuffle:
        random.shuffle(order)
    return [Tensor.stack([images[i] for i in order[start : start + batch_size]]) for start in range(0, len(order), batch_size)]


def _evaluate(model: nn.Module, images: Tensor, batch_size: int, kind: str, noise_std: float, hole_size: int) -> tuple[float, float]:
    criterion = nn.MSELoss()
    restored_loss = 0.0
    corrupted_loss = 0.0
    with no_grad():
        for x in _batches(images, batch_size, shuffle=False):
            clean = preprocess(x)
            damaged = corrupt(clean, kind, noise_std, hole_size)
            restored_loss += criterion(model(damaged), clean).item() * x.shape[0]
            corrupted_loss += criterion(damaged, clean).item() * x.shape[0]
    return restored_loss / images.shape[0], corrupted_loss / images.shape[0]


def _plot(step_losses: list[float], test_losses: list[float], baseline: float, steps_per_epoch: int, save: str | None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(step_losses, linewidth=0.6, alpha=0.5, label='Train step loss')
    epochs = range(1, len(test_losses) + 1)
    ax.plot([e * steps_per_epoch for e in epochs], test_losses, marker='o', label='Test loss')
    ax.axhline(baseline, color='gray', linestyle='--', label='Corrupted input (no model)')
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE to clean image')
    ax.set_yscale('log')
    ax.set_title('Reconstruction loss')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print(f'Saved training curves to {save}')
    else:
        plt.show()


def _main() -> None:
    parser = argparse.ArgumentParser(description='Train a denoising and inpainting convolutional autoencoder')
    parser.add_argument('--dataset', type=str, default='data/cats_dogs.mag', help='Snapshot written by examples/cats_dogs/prepare_dataset.py')
    parser.add_argument('--out', type=str, default='data/ae_model.mag', help='Where to write the trained weights')
    parser.add_argument('--corruption', type=str, default='both', choices=CORRUPTIONS, help='How training inputs are damaged')
    parser.add_argument('--noise-std', type=float, default=0.2, help='Standard deviation of the Gaussian noise, in [0, 1] pixel units')
    parser.add_argument('--hole-size', type=int, default=20, help='Side length of the square that is cut out of each image')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Peak AdamW learning rate, decays polynomially to zero')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW weight decay')
    parser.add_argument('--width', type=int, default=32, help='Channels of the first convolution')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--plot', type=str, default=None, help='Save the loss curves to this file instead of showing them')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting the loss curves')
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    context.manual_seed(args.seed)
    context.set_default_dtype(dtype.float32)
    if context.is_device_available(args.device):
        context.set_default_device(args.device)

    tensors, metadata = deserialize(args.dataset)
    train_images, test_images = tensors['train_images'], tensors['test_images']
    image_size = int(metadata['image_size'])
    print(f'Train: {train_images.shape[0]} images, Test: {test_images.shape[0]} images, Size: {image_size}x{image_size}')

    model = DenoisingAE(image_size=image_size, width=args.width)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (train_images.shape[0] + args.batch_size - 1) // args.batch_size
    scheduler = optim.PolynomialDecayLRScheduler(args.lr, args.epochs * steps_per_epoch)
    global_step = 0
    num_params = sum(p.numel for p in model.parameters())
    print(f'Parameters: {num_params:,}')

    best_loss = float('inf')
    step_losses: list[float] = []
    test_losses: list[float] = []
    baseline = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        total_loss = 0.0
        batches = _batches(train_images, args.batch_size, shuffle=True)
        for step, x in enumerate(batches):
            clean = preprocess(x)
            damaged = corrupt(clean, args.corruption, args.noise_std, args.hole_size)
            for group in optimizer.param_groups:
                group['lr'] = scheduler.step(global_step)
            global_step += 1
            loss = criterion(model(damaged), clean)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step_losses.append(loss.item())
            total_loss += step_losses[-1] * x.shape[0]
            if step % 25 == 0:
                print(f'  Epoch {epoch} Step [{step}/{len(batches)}] Loss: {loss.item():.5f}')
        model.eval()
        train_loss = total_loss / train_images.shape[0]
        test_loss, baseline = _evaluate(model, test_images, args.batch_size, args.corruption, args.noise_std, args.hole_size)
        test_losses.append(test_loss)
        elapsed = time.perf_counter() - start
        print(f'Epoch [{epoch}/{args.epochs}] Train: {train_loss:.5f} Test: {test_loss:.5f} Corrupted input: {baseline:.5f} ({elapsed:.1f}s)')
        if test_loss <= best_loss:
            best_loss = test_loss
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            model.save(
                args.out,
                {'epoch': epoch, 'test_loss': test_loss, 'corruption': args.corruption, 'noise_std': args.noise_std, 'hole_size': args.hole_size},
            )
    print(f'Training complete, best test loss {best_loss:.5f} (corrupted input {baseline:.5f}), weights saved to {args.out}')
    if not args.no_plot:
        _plot(step_losses, test_losses, baseline, steps_per_epoch, args.plot)


if __name__ == '__main__':
    _main()
