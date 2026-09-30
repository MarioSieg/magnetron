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
import math
import random
import time

from model import CORRUPTIONS, DenoisingAE, corrupt, preprocess

from magnetron import Tensor, context, dtype, no_grad
from magnetron.snapshot import deserialize


def _psnr(a: Tensor, b: Tensor) -> float:
    mse = (a - b).sqr().mean().item()
    return float('inf') if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def _show(rows: list[tuple[str, Tensor]], reference: Tensor | None, save: str | None) -> None:
    import matplotlib.pyplot as plt

    count = rows[0][1].shape[0]
    fig, axes = plt.subplots(len(rows), count, figsize=(1.6 * count, 1.8 * len(rows)), squeeze=False)
    for (label, batch), axis_row in zip(rows, axes, strict=True):
        for i, ax in enumerate(axis_row):
            ax.imshow(batch[i].permute(1, 2, 0).tolist())
            ax.set_xticks([])
            ax.set_yticks([])
            if reference is not None and label != 'Original':
                ax.set_title(f'{_psnr(batch[i], reference[i]):.1f} dB', fontsize=8)
        axis_row[0].set_ylabel(label, fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print(f'Saved figure to {save}')
    else:
        plt.show()


def _restore(model: DenoisingAE, damaged: Tensor) -> tuple[Tensor, float]:
    start = time.perf_counter()
    with no_grad():
        restored = model(damaged)
    return restored, time.perf_counter() - start


def _main() -> None:
    parser = argparse.ArgumentParser(description='Denoise and inpaint images with the trained autoencoder')
    parser.add_argument('images', type=str, nargs='*', help='Image files to restore, omit to sample the test split of the dataset')
    parser.add_argument('--model', type=str, default='data/ae_model.mag', help='Snapshot written by train.py')
    parser.add_argument('--dataset', type=str, default='data/cats_dogs.mag', help='Snapshot written by examples/cats_dogs/prepare_dataset.py')
    parser.add_argument('--count', type=int, default=8, help='Number of random test images to show')
    parser.add_argument(
        '--corruption',
        type=str,
        default=None,
        choices=(*CORRUPTIONS, 'none'),
        help='Damage applied before restoring, defaults to what the model was trained with',
    )
    parser.add_argument('--seed', type=int, default=None, help='Seed for sampling the test images and the damage')
    parser.add_argument('--save', type=str, default=None, help='Write the figure to this file instead of showing it')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    random.seed(seed)
    context.manual_seed(seed)
    context.set_default_dtype(dtype.float32)
    if context.is_device_available(args.device):
        context.set_default_device(args.device)

    model, metadata = DenoisingAE.load(args.model)
    model.eval()
    size = int(metadata['image_size'])
    kind = args.corruption or str(metadata['corruption'])
    print(f'Loaded {args.model} (epoch {metadata["epoch"]}, test loss {metadata["test_loss"]:.5f}, trained on {metadata["corruption"]})')

    if args.images:
        clean = preprocess(Tensor.stack([Tensor.load_image(path, channels='RGB', resize_to=(size, size)) for path in args.images]))
    else:
        images = deserialize(args.dataset)[0]['test_images']
        clean = preprocess(Tensor.stack([images[i] for i in random.sample(range(images.shape[0]), args.count)]))

    if kind == 'none':
        restored, elapsed = _restore(model, clean)
        print(f'{clean.shape[0]} image(s) restored in {elapsed * 1e3:.1f} ms')
        _show([('Input', clean), ('Restored', restored)], None, args.save)
        return

    damaged = corrupt(clean, kind, float(metadata['noise_std']), int(metadata['hole_size']))
    restored, elapsed = _restore(model, damaged)
    print(f'{clean.shape[0]} image(s) restored in {elapsed * 1e3:.1f} ms')
    print(f'PSNR corrupted {_psnr(damaged, clean):.2f} dB -> restored {_psnr(restored, clean):.2f} dB')
    _show([('Corrupted', damaged), ('Restored', restored), ('Original', clean)], clean, args.save)


if __name__ == '__main__':
    _main()
