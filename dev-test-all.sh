# Must be in venv already with .[dev] installed
pip install . --force-reinstall
pytest -n auto -vv -s test/python/
