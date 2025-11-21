# DirkTensor

A C++ tensor library for efficient inference.

## Testing

### Running Python Script Tests

The Python tokenization utilities have test coverage using pytest.

#### Prerequisites

Install pytest if you haven't already:

```bash
pip install pytest
```

The tokenization script also requires the `transformers` library:

```bash
pip install transformers
```

#### Running All Tests

To run all tests:

```bash
pytest tests/
```

To run tests with verbose output:

```bash
pytest tests/ -v
```

#### Running Specific Tests

To run tests for a specific file:

```bash
pytest tests/test_tokenize_text.py -v
```

To run a specific test function:

```bash
pytest tests/test_tokenize_text.py::test_tokenize_to_file -v
```

### Running C++ Tests

C++ tests are managed through CMake and CTest.

From the build directory:

```bash
# Run all C++ tests
ctest --output-on-failure

# Run a specific test
ctest --output-on-failure -R test_olmo_model

# Run tests from a specific build directory
ctest --test-dir cmake-build-relwithdebinfo --output-on-failure
```
