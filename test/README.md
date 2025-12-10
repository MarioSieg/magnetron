## Unit Test Suites

Magnetron has two test suites:
* Python tests, which test the Python API and all of Magnetron's operators against PyTorch and Numpy:`test/python`
* Internal C++ tests, which test low-level correctness and performance: `test/cpp`

### Running Python API Tests
From the root directory, run:
```bash
pytest -s test/python/
```

### Running C++ Tests
The C++ tests use CMake and google test and can be run as an executable target via the IDE or command line. 