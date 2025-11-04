# Test Suite Documentation

This directory contains unit and integration tests for the ML census income prediction model.

## Test Structure

```
test/
├── conftest.py              # Shared pytest fixtures
├── sample_data.csv          # Sample data for testing
├── test_data.py             # Data processing tests
├── test_model.py            # Model training and inference tests
├── test_metrics.py          # Metrics computation tests
├── test_slice_metrics.py    # Slice analysis tests
├── test_integration.py      # End-to-end pipeline tests
└── test_api.py              # FastAPI endpoint tests
```

## Running Tests

### Option 1: Using the test script (recommended)

```bash
# Run all tests
./bin/run_tests.sh

# Run with verbose output
./bin/run_tests.sh -v

# Run specific test file
./bin/run_tests.sh test/test_model.py

# Run specific test function
./bin/run_tests.sh -k test_train_model_returns_classifier

# Run with coverage report
./bin/run_tests.sh --cov
```

### Option 2: Using pytest directly

```bash
# Run all tests
pytest test/

# Run with verbose output
pytest test/ -v

# Run specific test file
pytest test/test_model.py

# Run specific test class
pytest test/test_model.py::TestModel

# Run specific test function
pytest test/test_model.py::TestModel::test_train_model_returns_classifier
```

## Test Coverage

### Data Processing Tests (`test_data.py`)
- ✓ Training mode encoding and label binarization
- ✓ Inference mode with pre-fitted encoder
- ✓ Handling data without labels
- ✓ Preservation of continuous features

### Model Tests (`test_model.py`)
- ✓ Model training returns fitted classifier
- ✓ Inference returns correct prediction shape
- ✓ Single sample inference
- ✓ Model determinism with fixed random_state

### Metrics Tests (`test_metrics.py`)
- ✓ Perfect predictions (precision=1.0, recall=1.0, F1=1.0)
- ✓ Completely wrong predictions (all metrics=0.0)
- ✓ Mixed predictions
- ✓ Edge case: all negative class
- ✓ Metrics on trained model

### Slice Metrics Tests (`test_slice_metrics.py`)
- ✓ Slice analysis for education feature
- ✓ Slice analysis for protected attribute (sex)
- ✓ Invalid feature raises error
- ✓ Requires encoder and label binarizer
- ✓ Multiple slice features analysis

### Integration Tests (`test_integration.py`)
- ✓ Full pipeline: data → training → inference → metrics → slices
- ✓ Inference on new/unseen data

### API Tests (`test_api.py`)
- ✓ GET root endpoint
- ✓ POST with single tag
- ✓ POST with list of tags

## Fixtures

Shared fixtures are defined in `conftest.py`:

- **`sample_data`**: Loads sample census data from CSV
- **`categorical_features`**: List of categorical feature names
- **`trained_components`**: Pre-trained model, encoder, and label binarizer

## Sample Data

The `sample_data.csv` file contains 12 representative samples from the census dataset with:
- All 8 categorical features
- All 6 continuous features
- Binary salary labels (<=50K, >50K)
- Diverse demographic representation

## Requirements

```bash
pip install pytest
pip install pytest-cov  # Optional, for coverage reports
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: ./bin/run_tests.sh
```

## Notes

- Tests use small sample data for speed
- All tests are deterministic (fixed random_state)
- Protected attributes (race, sex) are included for bias testing
- Slice analysis tests verify fairness evaluation functionality
