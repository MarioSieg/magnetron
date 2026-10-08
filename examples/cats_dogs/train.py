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

from model import CLASSES, CatsDogsNet, preprocess

from magnetron import Tensor, context, dtype, nn, no_grad, optim
from magnetron.snapshot import deserialize


def _batches(images: Tensor, labels: list[int], batch_size: int, shuffle: bool) -> list[tuple[Tensor, Tensor]]:
    order = list(range(images.shape[0]))
    if shuffle:
        random.shuffle(order)
    out: list[tuple[Tensor, Tensor]] = []
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        out.append((Tensor.stack([images[i] for i in idx]), Tensor([labels[i] for i in idx], dtype=dtype.int64)))
    return out


def _augment(x: Tensor, pad: int = 4) -> Tensor:
    size = x.shape[-1]
    x = x.pad((pad, pad, pad, pad))
    x = x.view_slice(2, random.randint(0, 2 * pad), size, 1).view_slice(3, random.randint(0, 2 * pad), size, 1)
    if random.random() < 0.5:
        x = x.flip(3)
    return x.contiguous()


def _evaluate(model: nn.Module, images: Tensor, labels: list[int], batch_size: int) -> float:
    correct = 0
    with no_grad():
        for x, y in _batches(images, labels, batch_size, shuffle=False):
            pred = model(preprocess(x)).argmax(dim=-1)
            correct += int(pred.eq(y).cast(dtype.int64).sum().item())
    return correct / images.shape[0]


def _plot(step_losses: list[float], train_accs: list[float], test_accs: list[float], steps_per_epoch: int, save: str | None) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_loss.plot(step_losses, linewidth=0.6, alpha=0.5, label='Step loss')
    window = steps_per_epoch
    if len(step_losses) >= window:
        smooth = [sum(step_losses[i - window : i]) / window for i in range(window, len(step_losses) + 1)]
        ax_loss.plot(range(window - 1, len(step_losses)), smooth, linewidth=2, label=f'Moving average ({window} steps)')
    ax_loss.set_xlabel('Step')
    ax_loss.set_ylabel('Cross entropy')
    ax_loss.set_title('Training loss')
    ax_loss.grid(True)
    ax_loss.legend()
    epochs = range(1, len(train_accs) + 1)
    ax_acc.plot(epochs, [100 * a for a in train_accs], marker='o', label='Train')
    ax_acc.plot(epochs, [100 * a for a in test_accs], marker='o', label='Test')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('Accuracy per epoch')
    ax_acc.set_ylim(40, 100)
    ax_acc.set_xticks(list(epochs))
    ax_acc.grid(True)
    ax_acc.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print(f'Saved training curves to {save}')
    else:
        plt.show()


def _main() -> None:
    parser = argparse.ArgumentParser(description='Train a cats vs dogs classifier')
    parser.add_argument('--dataset', type=str, default='data/cats_dogs.mag', help='Snapshot written by prepare_dataset.py')
    parser.add_argument('--out', type=str, default='data/cats_dogs_model.mag', help='Where to write the trained weights')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Peak AdamW learning rate, decays polynomially to zero')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW weight decay')
    parser.add_argument('--width', type=int, default=32, help='Channels of the first convolution')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--plot', type=str, default=None, help='Save the loss and accuracy curves to this file instead of showing them')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting the training curves')
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    context.manual_seed(args.seed)
    context.set_default_dtype(dtype.float32)
    if context.is_device_available(args.device):
        context.set_default_device(args.device)

    tensors, metadata = deserialize(args.dataset)
    if list(metadata['classes']) != list(CLASSES):
        raise RuntimeError(f'Dataset classes {metadata["classes"]} do not match {CLASSES}')
    train_images, train_labels = tensors['train_images'], tensors['train_labels'].tolist()
    test_images, test_labels = tensors['test_images'], tensors['test_labels'].tolist()
    image_size = int(metadata['image_size'])
    print(f'Train: {train_images.shape[0]} images, Test: {test_images.shape[0]} images, Size: {image_size}x{image_size}')

    model = CatsDogsNet(image_size=image_size, width=args.width)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (train_images.shape[0] + args.batch_size - 1) // args.batch_size
    scheduler = optim.PolynomialDecayLRScheduler(args.lr, args.epochs * steps_per_epoch)
    global_step = 0
    num_params = sum(p.numel for p in model.parameters())
    print(f'Parameters: {num_params:,}')

    best_acc = 0.0
    step_losses: list[float] = []
    train_accs: list[float] = []
    test_accs: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        total_loss = 0.0
        correct = 0
        batches = _batches(train_images, train_labels, args.batch_size, shuffle=True)
        for step, (x, y) in enumerate(batches):
            x = _augment(preprocess(x))
            for group in optimizer.param_groups:
                group['lr'] = scheduler.step(global_step)
            global_step += 1
            logits = model(x)
            loss = criterion(logits, y.one_hot(len(CLASSES)).cast(logits.dtype))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step_losses.append(loss.item())
            total_loss += step_losses[-1] * x.shape[0]
            correct += int(logits.argmax(dim=-1).eq(y).cast(dtype.int64).sum().item())
            if step % 25 == 0:
                print(f'  Epoch {epoch} Step [{step}/{len(batches)}] Loss: {loss.item():.4f}')
        model.eval()
        train_loss = total_loss / train_images.shape[0]
        train_acc = correct / train_images.shape[0]
        test_acc = _evaluate(model, test_images, test_labels, args.batch_size)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        elapsed = time.perf_counter() - start
        print(f'Epoch [{epoch}/{args.epochs}] Loss: {train_loss:.4f} Train Acc: {train_acc:.2%} Test Acc: {test_acc:.2%} ({elapsed:.1f}s)')
        if test_acc >= best_acc:
            best_acc = test_acc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            model.save(args.out, {'epoch': epoch, 'test_accuracy': test_acc})
    print(f'Training complete, best test accuracy {best_acc:.2%}, weights saved to {args.out}')
    if not args.no_plot:
        _plot(step_losses, train_accs, test_accs, steps_per_epoch, args.plot)


if __name__ == '__main__':
    _main()
