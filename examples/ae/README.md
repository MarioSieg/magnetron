# Autoencoder

Trains an image autoencoder in Magnetron and visualizes the reconstruction. Uses Magnetron’s image loader and standard `nn` modules + Adam.

## Install

From the repo root:

```bash
uv pip install -e .[examples]
```

## Run

```bash
python examples/ae/main.py
```

## Notes

- Defaults are set in the script; check `main.py` for available flags.