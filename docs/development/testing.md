# Testing Guide

This guide covers testing practices and strategies for the Shifting Baseline library.

## Testing Philosophy

### Test-Driven Development

We follow test-driven development principles:

1. **Write tests first**: Before implementing new features
2. **Test behavior**: Focus on what the code does, not how
3. **Test edge cases**: Include boundary conditions and error cases
4. **Maintain test coverage**: Aim for high test coverage
5. **Keep tests simple**: Tests should be easy to understand and maintain

### Testing Levels

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test interactions between components
3. **System Tests**: Test complete workflows
4. **Performance Tests**: Test performance characteristics

## Test Structure

### Directory Organization

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and fixtures
├── data/                    # Test data files
│   ├── sample_climate.csv
│   ├── sample_historical.xlsx
│   └── test_regions.shp
├── test_data.py            # Data module tests
├── test_filters.py         # Filters module tests
├── test_compare.py         # Comparison module tests
├── test_calibration.py     # Calibration module tests
├── test_abm.py            # ABM module tests
├── test_process.py        # Process module tests
├── test_utils.py          # Utils module tests
└── integration/           # Integration tests
    ├── test_workflows.py
    └── test_performance.py
```

### Test File Naming

- `test_<module_name>.py`: Unit tests for specific modules
- `test_<feature_name>.py`: Tests for specific features
- `conftest.py`: Pytest configuration and shared fixtures

## Writing Tests

### Basic Test Structure

```python
import pytest
import pandas as pd
import numpy as np
from shifting_baseline.filters import classify_series

class TestClassifySeries:
    """Test cases for classify_series function."""

    def test_basic_classification(self):
        """Test basic classification functionality."""
        data = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = classify_series(data)
        expected = pd.Series([-2, -1, 0, 1, 2])
        pd.testing.assert_series_equal(result, expected)

    def test_custom_thresholds(self):
        """Test classification with custom thresholds."""
        data = pd.Series([-1.5, -0.5, 0.5, 1.5])
        thresholds = [-1.0, 0.0, 1.0]
        levels = [-1, 0, 1, 2]
        result = classify_series(data, thresholds=thresholds, levels=levels)
        expected = pd.Series([-1, 0, 1, 2])
        pd.testing.assert_series_equal(result, expected)

    def test_empty_series(self):
        """Test classification with empty series."""
        with pytest.raises(ValueError, match="Cannot classify empty series"):
            classify_series(pd.Series([]))

    def test_nan_handling(self):
        """Test classification with NaN values."""
        data = pd.Series([-1.0, float('nan'), 1.0])
        result = classify_series(data, handle_na="skip")
        assert result.isna().sum() == 1
        assert result.iloc[0] == -1
        assert result.iloc[2] == 1
```

### Using Fixtures

```python
import pytest
import pandas as pd
from shifting_baseline.data import HistoricalRecords

@pytest.fixture
def sample_climate_data():
    """Create sample climate data for testing."""
    return pd.Series(
        np.random.randn(100),
        index=pd.date_range('1000-01-01', periods=100, freq='Y')
    )

@pytest.fixture
def sample_historical_data():
    """Create sample historical data for testing."""
    data = pd.DataFrame({
        'station1': np.random.randint(1, 6, 100),
        'station2': np.random.randint(1, 6, 100),
        'station3': np.random.randint(1, 6, 100)
    }, index=pd.date_range('1000-01-01', periods=100, freq='Y'))
    return data

@pytest.fixture
def historical_records(sample_historical_data, tmp_path):
    """Create HistoricalRecords instance for testing."""
    # Create temporary files
    data_file = tmp_path / "test_data.xlsx"
    shp_file = tmp_path / "test_regions.shp"

    # Save sample data
    sample_historical_data.to_excel(data_file)

    # Create mock shapefile (simplified)
    # In real tests, you'd create actual shapefile

    return HistoricalRecords(
        shp_path=str(shp_file),
        data_path=str(data_file),
        region="华北地区"
    )

def test_historical_records_aggregation(historical_records):
    """Test historical records aggregation."""
    result = historical_records.aggregate("mean")
    assert isinstance(result, pd.Series)
    assert len(result) == 100
    assert not result.isna().all()
```

### Parametrized Tests

```python
import pytest
from shifting_baseline.filters import classify_single_value

@pytest.mark.parametrize("value,expected", [
    (-2.0, -2),
    (-1.0, -1),
    (0.0, 0),
    (1.0, 1),
    (2.0, 2),
    (-1.5, -2),
    (1.5, 2)
])
def test_classify_single_value(value, expected):
    """Test single value classification."""
    result = classify_single_value(value)
    assert result == expected

@pytest.mark.parametrize("method,expected_range", [
    ("pearson", (-1, 1)),
    ("kendall", (-1, 1)),
    ("spearman", (-1, 1))
])
def test_correlation_methods(method, expected_range):
    """Test different correlation methods."""
    from shifting_baseline.utils.calc import calc_corr

    data1 = pd.Series([1, 2, 3, 4, 5])
    data2 = pd.Series([2, 4, 6, 8, 10])

    r, p, n = calc_corr(data1, data2, how=method)

    assert expected_range[0] <= r <= expected_range[1]
    assert 0 <= p <= 1
    assert n == 5
```

### Testing with Mock Data

```python
import pytest
from unittest.mock import Mock, patch
from shifting_baseline.data import load_nat_data

@patch('shifting_baseline.data.get_files')
@patch('shifting_baseline.data.pd.read_csv')
def test_load_nat_data_mock(mock_read_csv, mock_get_files):
    """Test load_nat_data with mocked file operations."""
    # Setup mocks
    mock_get_files.return_value = ['file1.txt', 'file2.txt']

    # Mock CSV data
    mock_data1 = pd.DataFrame({
        'year': [1000, 1001, 1002],
        'value': [1.0, 2.0, 3.0]
    }).set_index('year')

    mock_data2 = pd.DataFrame({
        'year': [1000, 1001, 1002],
        'value': [2.0, 3.0, 4.0]
    }).set_index('year')

    mock_read_csv.side_effect = [mock_data1, mock_data2]

    # Test function
    datasets, uncertainties = load_nat_data(
        folder="test_folder",
        includes=["test"],
        start_year=1000
    )

    # Assertions
    assert len(datasets.columns) == 2
    assert len(uncertainties.columns) == 2
    assert datasets.index.min() == 1000
    assert datasets.index.max() == 1002
```

## Test Categories

### Unit Tests

Test individual functions and methods:

```python
def test_calc_std_deviation():
    """Test standard deviation calculation."""
    from shifting_baseline.filters import calc_std_deviation

    data = pd.Series([1, 2, 3, 4, 5])
    result = calc_std_deviation(data)

    # Last value is 5, mean is 3, std is sqrt(2)
    expected = (5 - 3) / np.sqrt(2)
    assert abs(result - expected) < 1e-10

def test_historical_records_initialization():
    """Test HistoricalRecords initialization."""
    from shifting_baseline.data import HistoricalRecords

    # Test with valid parameters
    history = HistoricalRecords(
        shp_path="test.shp",
        data_path="test.xlsx",
        region="华北地区"
    )

    assert history.region == "华北地区"
    assert hasattr(history, 'data')
```

### Integration Tests

Test interactions between components:

```python
def test_data_loading_and_processing():
    """Test complete data loading and processing workflow."""
    from shifting_baseline.data import HistoricalRecords, load_nat_data
    from shifting_baseline.filters import classify
    from shifting_baseline.compare import compare_corr

    # Load data
    history = HistoricalRecords(
        shp_path="data/regions.shp",
        data_path="data/historical_data.xlsx",
        region="华北地区"
    )

    datasets, uncertainties = load_nat_data(
        folder="data/tree_ring/",
        includes=["shi2018", "yellow2019"]
    )

    # Process data
    historical_series = history.aggregate("mean")
    climate_series = datasets.mean(axis=1)

    # Classify data
    hist_classified = classify(historical_series)
    clim_classified = classify(climate_series)

    # Compare data
    r, p, n = compare_corr(historical_series, climate_series)

    # Assertions
    assert isinstance(historical_series, pd.Series)
    assert isinstance(climate_series, pd.Series)
    assert len(hist_classified) == len(historical_series)
    assert len(clim_classified) == len(climate_series)
    assert -1 <= r <= 1
    assert 0 <= p <= 1
    assert n > 0
```

### Performance Tests

Test performance characteristics:

```python
import time
import pytest

def test_data_loading_performance():
    """Test data loading performance."""
    from shifting_baseline.data import load_nat_data

    start_time = time.time()

    datasets, uncertainties = load_nat_data(
        folder="data/tree_ring/",
        includes=["shi2018", "yellow2019"],
        start_year=1000
    )

    end_time = time.time()
    loading_time = end_time - start_time

    # Should load within 5 seconds
    assert loading_time < 5.0
    assert len(datasets) > 0
    assert len(uncertainties) > 0

@pytest.mark.slow
def test_large_dataset_processing():
    """Test processing of large datasets."""
    from shifting_baseline.filters import classify_series

    # Create large dataset
    large_data = pd.Series(np.random.randn(10000))

    start_time = time.time()
    result = classify_series(large_data)
    end_time = time.time()

    processing_time = end_time - start_time

    # Should process within 1 second
    assert processing_time < 1.0
    assert len(result) == len(large_data)
```

## Test Configuration

### Pytest Configuration

Create `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=shifting_baseline
    --cov-report=html
    --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### Conftest.py

```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_time_series():
    """Create sample time series for testing."""
    return pd.Series(
        np.random.randn(100),
        index=pd.date_range('1000-01-01', periods=100, freq='Y')
    )

@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'col1': np.random.randn(100),
        'col2': np.random.randn(100),
        'col3': np.random.randn(100)
    }, index=pd.date_range('1000-01-01', periods=100, freq='Y'))

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Configure pandas display options
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 10)

    yield

    # Cleanup after each test
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_filters.py

# Run specific test function
pytest tests/test_filters.py::TestClassifySeries::test_basic_classification

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=shifting_baseline --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run in parallel
pytest -n auto
```

### Continuous Integration

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest --cov=shifting_baseline --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Test Data Management

### Creating Test Data

```python
def create_test_climate_data(n_years=100, n_stations=5):
    """Create test climate data."""
    data = {}
    for i in range(n_stations):
        data[f'station_{i}'] = np.random.randn(n_years)

    return pd.DataFrame(
        data,
        index=pd.date_range('1000-01-01', periods=n_years, freq='Y')
    )

def create_test_historical_data(n_years=100, n_stations=3):
    """Create test historical data."""
    data = {}
    for i in range(n_stations):
        data[f'station_{i}'] = np.random.randint(1, 6, n_years)

    return pd.DataFrame(
        data,
        index=pd.date_range('1000-01-01', periods=n_years, freq='Y')
    )
```

### Test Data Files

Store test data in `tests/data/`:

```
tests/data/
├── sample_climate.csv
├── sample_historical.xlsx
├── test_regions.shp
├── test_regions.dbf
├── test_regions.prj
└── test_regions.shx
```

## Debugging Tests

### Debugging Failed Tests

```python
def test_debug_example():
    """Example of debugging a test."""
    data = pd.Series([1, 2, 3, 4, 5])
    result = classify_series(data)

    # Add debugging output
    print(f"Input data: {data}")
    print(f"Result: {result}")
    print(f"Expected: {pd.Series([-2, -1, 0, 1, 2])}")

    # Use pytest.set_trace() for interactive debugging
    # pytest.set_trace()

    expected = pd.Series([-2, -1, 0, 1, 2])
    pd.testing.assert_series_equal(result, expected)
```

### Using Pytest Debugging

```bash
# Run with debugging
pytest --pdb

# Run specific test with debugging
pytest tests/test_filters.py::test_debug_example --pdb

# Run with print statements
pytest -s

# Run with maximum verbosity
pytest -vv
```

## Best Practices

### Test Organization

1. **Group related tests**: Use test classes for related functionality
2. **Use descriptive names**: Test names should describe what is being tested
3. **Keep tests independent**: Tests should not depend on each other
4. **Use fixtures**: For common test data and setup
5. **Test edge cases**: Include boundary conditions and error cases

### Test Maintenance

1. **Update tests**: When changing functionality
2. **Remove obsolete tests**: When removing features
3. **Refactor tests**: Keep tests clean and maintainable
4. **Monitor coverage**: Ensure adequate test coverage
5. **Review tests**: Regular review of test quality

### Performance Considerations

1. **Use appropriate markers**: Mark slow tests
2. **Mock external dependencies**: Avoid slow I/O operations
3. **Use fixtures efficiently**: Avoid expensive setup
4. **Parallel execution**: Use pytest-xdist for parallel tests
5. **Profile tests**: Identify slow tests

## Troubleshooting

### Common Issues

1. **Import errors**: Check Python path and dependencies
2. **Fixture not found**: Check fixture scope and naming
3. **Test data issues**: Verify test data files exist
4. **Coverage issues**: Check coverage configuration
5. **Performance issues**: Profile and optimize slow tests

### Debugging Tips

1. **Use print statements**: For simple debugging
2. **Use pytest.set_trace()**: For interactive debugging
3. **Check test output**: Look for error messages and warnings
4. **Verify test data**: Ensure test data is correct
5. **Check dependencies**: Verify all dependencies are installed
