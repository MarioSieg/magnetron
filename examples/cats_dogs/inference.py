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

from model import CatsDogsNet, preprocess

from magnetron import Tensor, context, dtype, no_grad
from magnetron.snapshot import deserialize


def _predict(model: CatsDogsNet, batch: Tensor) -> tuple[list[list[float]], float]:
    start = time.perf_counter()
    with no_grad():
        probs = model(preprocess(batch)).softmax(dim=-1)
    return probs.tolist(), time.perf_counter() - start


def _classify_files(model: CatsDogsNet, classes: list[str], size: int, paths: list[str]) -> None:
    batch = Tensor.stack([Tensor.load_image(path, channels='RGB', resize_to=(size, size)) for path in paths])
    rows, elapsed = _predict(model, batch)
    width = max(len(path) for path in paths)
    for path, row in zip(paths, rows, strict=True):
        best = max(range(len(classes)), key=lambda i: row[i])
        print(f'{path:<{width}}  {classes[best]:<4} {row[best]:6.1%}')
    print(f'{len(paths)} image(s) in {elapsed * 1e3:.1f} ms')


def _classify_dataset_grid(model: CatsDogsNet, classes: list[str], dataset: str, grid: int, seed: int, save: str | None) -> None:
    import matplotlib.pyplot as plt

    tensors, _ = deserialize(dataset)
    images, labels = tensors['test_images'], tensors['test_labels'].tolist()
    random.seed(seed)
    idx = random.sample(range(images.shape[0]), grid * grid)
    batch = Tensor.stack([images[i] for i in idx])
    rows, elapsed = _predict(model, batch)
    correct = 0
    fig, axes = plt.subplots(grid, grid, figsize=(1.6 * grid, 1.8 * grid))
    for ax, i, row in zip(axes.flat, idx, rows, strict=True):
        pred = max(range(len(classes)), key=lambda k: row[k])
        truth = labels[i]
        correct += int(pred == truth)
        ax.imshow(images[i].permute(1, 2, 0).tolist())
        ax.set_title(f'{classes[pred]} {row[pred]:.0%}', fontsize=9, color='green' if pred == truth else 'red')
        ax.axis('off')
    total = grid * grid
    fig.suptitle(f'{correct}/{total} correct ({correct / total:.1%}), {elapsed * 1e3:.1f} ms', fontsize=12)
    fig.tight_layout()
    print(f'{correct}/{total} correct ({correct / total:.1%}) in {elapsed * 1e3:.1f} ms')
    if save:
        fig.savefig(save, dpi=150)
        print(f'Saved grid to {save}')
    else:
        plt.show()


def _main() -> None:
    parser = argparse.ArgumentParser(description='Classify images as cat or dog')
    parser.add_argument('images', type=str, nargs='*', help='Image files to classify, omit to sample the test split of the dataset')
    parser.add_argument('--model', type=str, default='data/cats_dogs_model.mag', help='Snapshot written by train.py')
    parser.add_argument('--dataset', type=str, default='data/cats_dogs.mag', help='Snapshot written by prepare_dataset.py')
    parser.add_argument('--grid', type=int, default=8, help='Side length of the grid of random test images')
    parser.add_argument('--seed', type=int, default=None, help='Seed for picking the random test images')
    parser.add_argument('--save', type=str, default=None, help='Write the grid to this file instead of showing it')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    context.set_default_dtype(dtype.float32)
    if context.is_device_available(args.device):
        context.set_default_device(args.device)

    model, metadata = CatsDogsNet.load(args.model)
    model.eval()
    classes = list(metadata['classes'])
    size = int(metadata['image_size'])
    if 'test_accuracy' in metadata:
        print(f'Loaded {args.model} (epoch {metadata["epoch"]}, test accuracy {metadata["test_accuracy"]:.2%})')

    if args.images:
        _classify_files(model, classes, size, args.images)
    else:
        seed = args.seed if args.seed is not None else random.randrange(1 << 30)
        _classify_dataset_grid(model, classes, args.dataset, args.grid, seed, args.save)


if __name__ == '__main__':
    _main()
