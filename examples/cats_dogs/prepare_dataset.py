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
import os
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from model import CLASSES
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from magnetron import Tensor, dtype
from magnetron.snapshot import SnapshotWriter

REPO_ID: str = 'Bingsu/Cat_and_Dog'
SPLITS: dict[str, str] = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}

console = Console()


def _load_split(split: str, size: int, limit: int | None, export_dir: Path | None) -> tuple[Tensor, Tensor]:
    path = hf_hub_download(REPO_ID, SPLITS[split], repo_type='dataset')
    table = pq.read_table(path)
    if limit is not None and limit < table.num_rows:
        table = table.take(list(range(0, table.num_rows, table.num_rows // limit))[:limit])
    images = table.column('image').to_pylist()
    labels = table.column('labels').to_pylist()
    tensors: list[Tensor] = []
    counts = [0] * len(CLASSES)
    with (
        tempfile.TemporaryDirectory() as tmp,
        Progress(
            TextColumn('{task.description}', style='cyan'), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(), console=console
        ) as progress,
    ):
        task = progress.add_task(f'{split:<6}', total=len(images))
        scratch = os.path.join(tmp, 'image.jpg')
        for image, label in zip(images, labels, strict=True):
            if export_dir is not None:
                target = export_dir / split / CLASSES[label] / f'{counts[label]:05d}.jpg'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(image['bytes'])
                file = str(target)
            else:
                with open(scratch, 'wb') as f:
                    f.write(image['bytes'])
                file = scratch
            counts[label] += 1
            tensors.append(Tensor.load_image(file, channels='RGB', resize_to=(size, size)))
            progress.advance(task)
    return Tensor.stack(tensors), Tensor(labels, dtype=dtype.int64)


def _main() -> None:
    parser = argparse.ArgumentParser(description='Download the cats vs dogs dataset and pack it into a Magnetron snapshot')
    parser.add_argument('--out', type=str, default='data/cats_dogs.mag', help='Output snapshot path')
    parser.add_argument('--size', type=int, default=64, help='Side length images are resized to')
    parser.add_argument('--limit', type=int, default=None, help='Only take N evenly spaced images of each split')
    parser.add_argument('--export-images', type=str, default=None, help='Also write the raw JPEGs into this directory')
    args = parser.parse_args()
    print(args)

    export_dir = Path(args.export_images) if args.export_images else None
    start = time.perf_counter()
    splits = {split: _load_split(split, args.size, args.limit, export_dir) for split in SPLITS}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'source': REPO_ID, 'classes': list(CLASSES), 'image_size': args.size}
    with SnapshotWriter(out, metadata) as snap:
        for split, (images, labels) in splits.items():
            snap.declare(f'{split}_images', images.shape, images.dtype)
            snap.declare(f'{split}_labels', labels.shape, labels.dtype)
        for split, (images, labels) in splits.items():
            snap.write(f'{split}_images', images)
            snap.write(f'{split}_labels', labels)
    elapsed = time.perf_counter() - start
    for split, (images, _) in splits.items():
        console.print(f'{split}: {images.shape[0]} images of shape {tuple(images.shape[1:])}', style='green')
    console.print(f'Wrote {out} ({out.stat().st_size / (1 << 20):.1f} MiB) in {elapsed:.1f}s', style='bold green')


if __name__ == '__main__':
    _main()
