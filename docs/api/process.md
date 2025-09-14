# Process Module

The `shifting_baseline.process` module provides comprehensive data processing pipelines for historical climate reconstruction data.

## Overview

This module handles:
- Data loading and preprocessing
- Time axis conversion
- Memory management for large datasets
- Batch processing of reconstruction data
- Data export and formatting

## Core Classes

### ProcessRecon

A flexible data processing pipeline for historical reconstruction data.

```python
from shifting_baseline.process import ProcessRecon
```

#### Constructor

```python
ProcessRecon(name: str, path: str, processes: Dict[str, Dict[str, Any]])
```

**Parameters:**
- `name`: Name of the processing pipeline
- `path`: Path to the data file
- `processes`: Dictionary of processing functions and their parameters

**Example:**
```python
# Define processing pipeline
processes = {
    "open_dataarray": {"chunks": {"time": 100}},
    "convert_time_axis": {"begin_year": 1470}
}

# Create processor
processor = ProcessRecon("climate_data", "data/climate.nc", processes)
```

#### Key Methods

##### `process()`
Execute the processing pipeline.

**Returns:**
- `Any`: Processed data

**Example:**
```python
# Process data
processed_data = processor.process()
print(f"Processed data shape: {processed_data.shape}")
```

##### `export(data, path)`
Export processed data to file.

**Parameters:**
- `data`: Data to export
- `path`: Output file path

**Example:**
```python
# Export processed data
processor.export(processed_data, "outputs/processed_climate.nc")
```

## Utility Functions

### `convert_time_axis(ds, begin_year=1470)`

Convert time axis to actual years and preserve time attributes.

**Parameters:**
- `ds`: xarray Dataset with time axis
- `begin_year`: Starting year for conversion

**Returns:**
- `xr.Dataset`: Dataset with updated time axis

**Example:**
```python
from shifting_baseline.process import convert_time_axis

# Convert time axis
ds_converted = convert_time_axis(dataset, begin_year=1470)
print(f"Time range: {ds_converted.time.min().values} - {ds_converted.time.max().values}")
```

### `load_and_combine_datasets(extract_dir)`

Load and combine multiple NetCDF files efficiently.

**Parameters:**
- `extract_dir`: Directory containing NetCDF files

**Returns:**
- `xr.Dataset`: Combined dataset

**Example:**
```python
from shifting_baseline.process import load_and_combine_datasets

# Load and combine datasets
combined = load_and_combine_datasets("data/netcdf_files/")
print(f"Combined dataset: {combined}")
```

### `load_and_combine_datasets_chunked(extract_dir, chunk_size=5)`

Load NetCDF files in chunks to reduce memory usage.

**Parameters:**
- `extract_dir`: Directory containing NetCDF files
- `chunk_size`: Number of files to process per chunk

**Returns:**
- `xr.Dataset`: Combined dataset

**Example:**
```python
from shifting_baseline.process import load_and_combine_datasets_chunked

# Load with chunking
combined = load_and_combine_datasets_chunked("data/large_files/", chunk_size=3)
```

### `load_and_combine_datasets_lazy(extract_dir)`

Load NetCDF files using lazy loading with dask.

**Parameters:**
- `extract_dir`: Directory containing NetCDF files

**Returns:**
- `xr.Dataset`: Combined dataset with lazy loading

**Example:**
```python
from shifting_baseline.process import load_and_combine_datasets_lazy

# Lazy loading
combined = load_and_combine_datasets_lazy("data/netcdf_files/")
print(f"Dataset loaded with dask: {combined.chunks}")
```

### `extract_summer_precipitation(dataset, aggregation="sum", agg_months=None)`

Extract summer precipitation and aggregate by year.

**Parameters:**
- `dataset`: Input dataset
- `aggregation`: Aggregation method ("sum" or "mean")
- `agg_months`: Months to aggregate (default: [7, 8, 9])

**Returns:**
- `xr.DataArray`: Aggregated summer precipitation

**Example:**
```python
from shifting_baseline.process import extract_summer_precipitation

# Extract summer precipitation
summer_precip = extract_summer_precipitation(
    dataset,
    aggregation="sum",
    agg_months=[6, 7, 8, 9]  # Extended summer
)
print(f"Summer precipitation shape: {summer_precip.shape}")
```

## Memory Management

### `log_memory_usage(stage="")`

Log current memory usage.

**Parameters:**
- `stage`: Description of current processing stage

**Example:**
```python
from shifting_baseline.process import log_memory_usage

# Log memory usage
log_memory_usage("after loading data")
```

### `force_garbage_collection()`

Force garbage collection to free memory.

**Example:**
```python
from shifting_baseline.process import force_garbage_collection

# Force garbage collection
force_garbage_collection()
```

## Batch Processing

### `batch_process_recon_data(cfg)`

Batch process multiple reconstruction datasets.

**Parameters:**
- `cfg`: Configuration object with processing parameters

**Example:**
```python
from shifting_baseline.process import batch_process_recon_data
from omegaconf import DictConfig

# Define batch processing configuration
config = DictConfig({
    "how": {
        "recon": {
            "dataset1": {
                "path": "data/dataset1.nc",
                "out": "outputs/processed1.nc",
                "process": {
                    "convert_time_axis": {"begin_year": 1470},
                    "extract_summer_precipitation": {"aggregation": "sum"}
                }
            },
            "dataset2": {
                "path": "data/dataset2.nc",
                "out": "outputs/processed2.nc",
                "process": {
                    "convert_time_axis": {"begin_year": 1500}
                }
            }
        }
    }
})

# Run batch processing
batch_process_recon_data(config)
```

## Advanced Processing Patterns

### 1. Custom Processing Pipeline

```python
def create_custom_pipeline(input_path, output_path, processing_steps):
    """Create a custom data processing pipeline."""

    # Define processes
    processes = {}
    for step_name, step_params in processing_steps.items():
        processes[step_name] = step_params

    # Create processor
    processor = ProcessRecon("custom_pipeline", input_path, processes)

    # Process data
    processed_data = processor.process()

    # Export results
    processor.export(processed_data, output_path)

    return processed_data

# Define processing steps
steps = {
    "open_dataarray": {"chunks": {"time": 50}},
    "convert_time_axis": {"begin_year": 1470},
    "extract_summer_precipitation": {
        "aggregation": "sum",
        "agg_months": [6, 7, 8, 9]
    }
}

# Run custom pipeline
result = create_custom_pipeline(
    "data/input.nc",
    "outputs/processed.nc",
    steps
)
```

### 2. Memory-Efficient Processing

```python
def memory_efficient_processing(data_dir, output_dir, chunk_size=5):
    """Process large datasets with memory management."""

    import os
    from pathlib import Path

    # Get all NetCDF files
    nc_files = list(Path(data_dir).glob("*.nc"))

    # Process in chunks
    for i in range(0, len(nc_files), chunk_size):
        chunk_files = nc_files[i:i+chunk_size]

        print(f"Processing chunk {i//chunk_size + 1}")

        # Load chunk
        chunk_data = load_and_combine_datasets_chunked(
            str(Path(data_dir)),
            chunk_size=len(chunk_files)
        )

        # Process chunk
        processed_chunk = extract_summer_precipitation(chunk_data)

        # Save chunk
        output_file = Path(output_dir) / f"chunk_{i//chunk_size}.nc"
        processed_chunk.to_netcdf(output_file)

        # Clean up memory
        del chunk_data, processed_chunk
        force_garbage_collection()
        log_memory_usage(f"after chunk {i//chunk_size + 1}")

# Run memory-efficient processing
memory_efficient_processing("data/large/", "outputs/chunks/")
```

### 3. Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def parallel_process_files(file_paths, output_dir, processes):
    """Process multiple files in parallel."""

    def process_single_file(file_path):
        # Create processor for single file
        processor = ProcessRecon(
            file_path.stem,
            str(file_path),
            processes
        )

        # Process data
        processed_data = processor.process()

        # Export
        output_path = Path(output_dir) / f"{file_path.stem}_processed.nc"
        processor.export(processed_data, output_path)

        return output_path

    # Process files in parallel
    with Pool() as pool:
        results = pool.map(process_single_file, file_paths)

    return results

# Define processing parameters
processes = {
    "convert_time_axis": {"begin_year": 1470},
    "extract_summer_precipitation": {"aggregation": "sum"}
}

# Get file paths
file_paths = list(Path("data/").glob("*.nc"))

# Process in parallel
output_files = parallel_process_files(file_paths, "outputs/", processes)
```

## Data Validation

### Input Validation

```python
def validate_input_data(file_path, expected_vars=None):
    """Validate input data before processing."""

    try:
        # Open dataset
        ds = xr.open_dataset(file_path)

        # Check required variables
        if expected_vars:
            missing_vars = set(expected_vars) - set(ds.data_vars)
            if missing_vars:
                raise ValueError(f"Missing variables: {missing_vars}")

        # Check time dimension
        if "time" not in ds.dims:
            raise ValueError("Dataset missing time dimension")

        # Check data quality
        if ds.isnull().all().any():
            print("Warning: Some variables contain only NaN values")

        return True

    except Exception as e:
        print(f"Validation failed: {e}")
        return False

# Validate input
is_valid = validate_input_data("data/climate.nc", expected_vars=["precipitation"])
```

### Output Validation

```python
def validate_output_data(processed_data, original_data):
    """Validate processed data against original."""

    # Check data integrity
    assert not processed_data.isnull().all().any(), "Output contains all NaN values"

    # Check time range
    if hasattr(processed_data, 'time'):
        assert len(processed_data.time) > 0, "Empty time dimension"

    # Check data range
    if hasattr(processed_data, 'values'):
        assert not (processed_data.values == 0).all(), "All values are zero"

    print("Output validation passed")
    return True
```

## Error Handling

### Robust Processing

```python
def robust_process_file(file_path, processes, max_retries=3):
    """Process file with error handling and retries."""

    for attempt in range(max_retries):
        try:
            # Validate input
            if not validate_input_data(file_path):
                raise ValueError("Input validation failed")

            # Create processor
            processor = ProcessRecon(
                file_path.stem,
                str(file_path),
                processes
            )

            # Process data
            processed_data = processor.process()

            # Validate output
            validate_output_data(processed_data, None)

            return processed_data

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            continue

# Use robust processing
try:
    result = robust_process_file("data/problematic.nc", processes)
except Exception as e:
    print(f"Processing failed: {e}")
```

## Integration with Other Modules

### With Data Module

```python
from shifting_baseline.data import load_data
from shifting_baseline.process import extract_summer_precipitation

# Load data
datasets, uncertainties, history = load_data(config)

# Process climate data
processed_data = extract_summer_precipitation(datasets)
```

### With Comparison Module

```python
from shifting_baseline.compare import compare_corr
from shifting_baseline.process import convert_time_axis

# Process data
processed_ds = convert_time_axis(dataset, begin_year=1470)

# Extract time series
time_series = processed_ds.mean(dim=['lat', 'lon'])

# Compare with other data
r, p, n = compare_corr(time_series, other_series)
```

## Performance Optimization

### Chunked Processing

```python
def optimized_chunked_processing(input_dir, output_dir, chunk_size=10):
    """Optimized chunked processing with memory management."""

    # Get all files
    nc_files = sorted(Path(input_dir).glob("*.nc"))

    # Process in optimized chunks
    for i in range(0, len(nc_files), chunk_size):
        chunk_files = nc_files[i:i+chunk_size]

        # Load chunk with optimal parameters
        chunk_data = xr.open_mfdataset(
            chunk_files,
            chunks={"time": 100, "lat": 50, "lon": 50},
            parallel=True
        )

        # Process chunk
        processed = extract_summer_precipitation(chunk_data)

        # Save with compression
        output_file = Path(output_dir) / f"chunk_{i//chunk_size}.nc"
        processed.to_netcdf(
            output_file,
            encoding={var: {"zlib": True, "complevel": 6} for var in processed.data_vars}
        )

        # Clean up
        del chunk_data, processed
        force_garbage_collection()
```

### Lazy Loading

```python
def lazy_processing_pipeline(input_path, output_path):
    """Use lazy loading for memory efficiency."""

    # Open with lazy loading
    ds = xr.open_dataset(input_path, chunks={"time": 100})

    # Process lazily
    processed = ds.pipe(convert_time_axis, begin_year=1470)
    processed = processed.pipe(extract_summer_precipitation, aggregation="sum")

    # Compute and save
    processed.compute().to_netcdf(output_path)

    return processed
```
