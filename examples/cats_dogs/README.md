# Cats vs Dogs

Trains a small convolutional classifier on the [Cat and Dog](https://huggingface.co/datasets/Bingsu/Cat_and_Dog) dataset (8000 train / 2000 test images) and runs it on your own pictures. Everything after the download is Magnetron: image decoding and resizing, the `.mag` dataset cache, the model, training, and inference.

## Install

From the repo root:

```bash
uv pip install -e .[examples]
```

## Prepare the dataset

Downloads the parquet shards from Hugging Face (about 230 MB), decodes and resizes every image with Magnetron's image loader and packs both splits into one memory-mapped snapshot:

```bash
python examples/cats_dogs/prepare_dataset.py
```

This writes `data/cats_dogs.mag` (uint8 images of shape `[N, 3, 64, 64]` plus int64 labels). Use `--size` for a different resolution, `--limit N` for a quick smoke test with N images per split and `--export-images DIR` to also keep the raw JPEGs on disk, which is handy for trying the inference script.

## Train

```bash
python examples/cats_dogs/train.py
```

Trains for 30 epochs with AdamW, a polynomially decaying learning rate, random crops and horizontal flips, evaluates on the test split after every epoch and saves the best weights to `data/cats_dogs_model.mag`. An epoch takes about 140 seconds on an Apple M3 Pro, so a full run takes a bit over an hour; `--epochs 10` gets you to about 76% in a sixth of the time. When training is done a matplotlib window shows the per-step loss and the per-epoch train and test accuracy; pass `--plot curves.png` to write it to a file instead or `--no-plot` to skip it. Other flags: `--batch-size`, `--lr`, `--weight-decay`, `--width` (channels of the first convolution, must be a multiple of 8), `--seed`, `--device`.

## Inference

Run without arguments to sample 64 random images from the test split and show them in an 8x8 grid, each labeled with the prediction and confidence (green when correct, red when wrong):

```bash
python examples/cats_dogs/inference.py
```

Use `--grid N` for a different grid size, `--seed` to get the same sample again and `--save grid.png` to write the figure to a file instead of opening a window.

Pass image files to classify your own pictures instead:

```bash
python examples/cats_dogs/inference.py my_cat.jpg my_dog.png
```

Prints the predicted class and confidence for every image. The model snapshot carries the input resolution and class names in its metadata, so no extra configuration is needed.

## Notes

- The network is seven 3x3 convolution blocks (convolution, group norm, ReLU) with four stride-2 stages, followed by a two layer MLP head, about 1.6M parameters at the default width.
- Expect about 83% test accuracy with the defaults. The same architecture and recipe trained in PyTorch on the same data reaches the same number, and the gradients of this model match PyTorch to a few parts per million, so the remaining errors are a limit of training a CNN from scratch on 8000 images at 64x64, not of the runtime. Close-up portraits, unusual framings and orange cats are the most common misses; real gains from here need transfer learning or far more data.
- Weights are saved through `SnapshotWriter` and loaded back with `deserialize`, the same format the Qwen3 example uses for its checkpoints.
