# Must be in venv already with .[dev] installed
pip install . --force-reinstall
# Use half of CPUs as the kernels themselves need cores too:
num_cores=$(($(nproc) / 2))
pytest -n "$num_cores" -s test/python/
