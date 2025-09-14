# Contributing to Past1000

Thank you for your interest in contributing to Past1000! This guide will help you get started with contributing to the project.

## Development Setup

### Prerequisites

- Python 3.11+
- Poetry (recommended) or pip
- Git
- A code editor (VS Code, PyCharm, etc.)

### Installation

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/past1000.git
   cd past1000
   ```

2. **Install dependencies:**
   ```bash
   # Using Poetry (recommended)
   poetry install --with dev

   # Or using pip
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Development Environment

```bash
# Activate virtual environment
poetry shell

# Or create virtual environment manually
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Code Style and Standards

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: isort
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public functions

### Code Formatting

We use several tools to maintain code quality:

```bash
# Format code
black past1000/
isort past1000/

# Check code style
flake8 past1000/
mypy past1000/

# Run all checks
pre-commit run --all-files
```

### Type Hints

All public functions must have type hints:

```python
from typing import Union, Optional, List, Dict, Any
import pandas as pd

def example_function(
    data: pd.Series,
    threshold: float = 0.5,
    method: str = "mean"
) -> Dict[str, Any]:
    """Example function with type hints."""
    pass
```

### Docstrings

Use Google style docstrings:

```python
def process_data(
    data: pd.DataFrame,
    method: str = "mean",
    **kwargs
) -> pd.Series:
    """Process data using specified method.

    Args:
        data: Input DataFrame to process
        method: Processing method to use
        **kwargs: Additional keyword arguments

    Returns:
        Processed data as Series

    Raises:
        ValueError: If method is not supported

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> result = process_data(df, method='mean')
        >>> print(result)
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data.py

# Run with coverage
pytest --cov=past1000 --cov-report=html

# Run with verbose output
pytest -v
```

### Writing Tests

Follow these guidelines for writing tests:

1. **Test file naming**: `test_<module_name>.py`
2. **Test function naming**: `test_<function_name>_<scenario>`
3. **Use fixtures**: For common test data
4. **Test edge cases**: Include boundary conditions
5. **Mock external dependencies**: Use `unittest.mock`

Example test:

```python
import pytest
import pandas as pd
from past1000.filters import classify_series

class TestClassifySeries:
    """Test cases for classify_series function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])

    def test_basic_classification(self, sample_data):
        """Test basic classification functionality."""
        result = classify_series(sample_data)
        expected = pd.Series([-2, -1, 0, 1, 2])
        pd.testing.assert_series_equal(result, expected)

    def test_custom_thresholds(self, sample_data):
        """Test classification with custom thresholds."""
        thresholds = [-1.0, 0.0, 1.0]
        levels = [-1, 0, 1, 2]
        result = classify_series(sample_data, thresholds=thresholds, levels=levels)
        expected = pd.Series([-1, -1, 0, 1, 2])
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
```

### Test Data

Place test data in the `tests/data/` directory:

```
tests/
├── data/
│   ├── sample_climate_data.csv
│   ├── sample_historical_data.xlsx
│   └── test_regions.shp
├── test_data.py
└── test_filters.py
```

## Documentation

### API Documentation

- Use docstrings for all public functions
- Include type hints
- Provide examples in docstrings
- Update API documentation when adding new functions

### User Documentation

- Update relevant sections in `docs/`
- Add examples for new features
- Update installation instructions if needed
- Add troubleshooting information

### Docstring Examples

```python
def advanced_correlation_analysis(
    data1: pd.Series,
    data2: pd.Series,
    method: str = "kendall",
    window_size: int = 30
) -> Dict[str, Any]:
    """Perform advanced correlation analysis between two time series.

    This function provides comprehensive correlation analysis including
    rolling window correlations, significance testing, and visualization.

    Args:
        data1: First time series for correlation analysis
        data2: Second time series for correlation analysis
        method: Correlation method to use. Options: 'pearson', 'kendall', 'spearman'
        window_size: Size of rolling window for analysis

    Returns:
        Dictionary containing:
            - 'correlation': Overall correlation coefficient
            - 'p_value': P-value for significance test
            - 'rolling_correlations': Series of rolling correlations
            - 'significance': Boolean indicating if correlation is significant

    Raises:
        ValueError: If method is not supported
        TypeError: If inputs are not pandas Series

    Example:
        >>> import pandas as pd
        >>> data1 = pd.Series([1, 2, 3, 4, 5])
        >>> data2 = pd.Series([2, 4, 6, 8, 10])
        >>> result = advanced_correlation_analysis(data1, data2)
        >>> print(result['correlation'])
        1.0
    """
    pass
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write code following the style guide
   - Add tests for new functionality
   - Update documentation
   - Ensure all tests pass

3. **Run quality checks:**
   ```bash
   # Format code
   black past1000/
   isort past1000/

   # Run tests
   pytest

   # Check code quality
   flake8 past1000/
   mypy past1000/
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

### Pull Request Guidelines

1. **Title**: Use descriptive title (e.g., "Add: New correlation analysis function")
2. **Description**: Explain what changes you made and why
3. **Tests**: Ensure all tests pass
4. **Documentation**: Update relevant documentation
5. **Examples**: Add usage examples if applicable

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] All existing tests still pass

## Documentation
- [ ] Docstrings updated
- [ ] API documentation updated
- [ ] User documentation updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] No merge conflicts
- [ ] Ready for review
```

## Code Review Process

### For Contributors

1. **Address feedback**: Respond to review comments
2. **Update code**: Make requested changes
3. **Test changes**: Ensure tests still pass
4. **Update documentation**: If requested

### For Reviewers

1. **Check functionality**: Ensure code works as intended
2. **Review style**: Check code follows guidelines
3. **Test coverage**: Ensure adequate test coverage
4. **Documentation**: Check documentation is complete

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Detailed steps to reproduce
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, etc.
6. **Code example**: Minimal code to reproduce

### Feature Requests

When requesting features, include:

1. **Description**: Clear description of the feature
2. **Use case**: Why this feature is needed
3. **Proposed solution**: How you think it should work
4. **Alternatives**: Other approaches considered

## Development Workflow

### Git Workflow

1. **Create feature branch** from `main`
2. **Make changes** and commit frequently
3. **Push branch** to your fork
4. **Create pull request** to `main`
5. **Address feedback** and update PR
6. **Merge** after approval

### Branch Naming

Use descriptive branch names:

- `feature/add-correlation-analysis`
- `bugfix/fix-data-loading-error`
- `docs/update-api-documentation`
- `refactor/improve-memory-usage`

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

Examples:
```
feat(filters): add new classification method
fix(data): resolve memory leak in data loading
docs(api): update correlation function documentation
```

## Performance Guidelines

### Memory Management

- Use generators for large datasets
- Clear variables when not needed
- Use chunked processing for large files
- Profile memory usage

### Optimization

- Use vectorized operations when possible
- Avoid unnecessary loops
- Use appropriate data structures
- Profile performance bottlenecks

### Example

```python
# Good: Vectorized operation
result = data.rolling(window=10).mean()

# Bad: Loop-based operation
result = []
for i in range(len(data)):
    window = data[max(0, i-9):i+1]
    result.append(window.mean())
```

## Release Process

### Version Bumping

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version in `pyproject.toml`
2. Update changelog
3. Run full test suite
4. Update documentation
5. Create release tag
6. Build and publish package

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: songshgeo@gmail.com

### Resources

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **API Reference**: [api/](api/)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project guidelines

### Unacceptable Behavior

- Harassment or discrimination
- Inappropriate language or behavior
- Spam or off-topic discussions
- Violation of project guidelines

## License

By contributing to Past1000, you agree that your contributions will be licensed under the MIT License.

## Thank You

Thank you for contributing to Past1000! Your contributions help make this project better for everyone.
