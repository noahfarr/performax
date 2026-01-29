# Performax

A lightweight profiling framework for JAX that makes it easy to measure execution time of your functions.

## Features

- **Simple decorator API** - Just add `@track` to functions you want to profile
- **Minimal overhead** - Uses JAX's built-in Perfetto tracing
- **Detailed statistics** - Get total time, call counts, and averages
- **Multiple output formats** - Pretty tables, dictionaries, pandas DataFrames, and more
- **Flexible loggers** - Rich tables, JSON, CSV, Markdown, and plain text output
- **Thread-safe** - Prevents concurrent profiling conflicts

## Installation

```bash
pip install performax
```

Or install from source:

```bash
git clone https://github.com/noahfarr/performax.git
cd performax
pip install -e .
```

## Quick Start

```python
import jax.numpy as jnp
from performax import track, profile

@track
def matmul(a, b):
    return jnp.dot(a, b)

@track(name="forward")
def forward(x, weights):
    for w in weights:
        x = matmul(x, w)
    return x

def main():
    x = jnp.ones((1000, 512))
    weights = [jnp.ones((512, 512)) for _ in range(5)]
    return forward(x, weights)

result, stats = profile(main)
print(stats)
```

Output:
```
Function | Total (ms) | Calls | Avg (ms)
-----------------------------------------
forward  | 125.432    | 1     | 125.432
matmul   | 98.765     | 5     | 19.753
```

## API Reference

### `@track`

Decorator that marks a function for profiling.

```python
# Use function name
@track
def my_function():
    ...

# Use custom name
@track(name="custom_name")
def my_function():
    ...
```

### `profile(fn, *args, **kwargs)`

Profile a function and return timing statistics.

```python
result, stats = profile(main)
result, stats = profile(forward, x, weights)
result, stats = profile(compute, learning_rate=0.01)
```

**Parameters:**
- `fn` - The function to profile
- `*args` - Positional arguments to pass to `fn`
- `**kwargs` - Keyword arguments to pass to `fn`

**Returns:**
- `result` - The return value of `fn`
- `stats` - A `ProfileResult` containing timing statistics

**Raises:**
- `ProfilingError` - If profiling fails or another profile is already running

### `ProfileResult`

Container for profiling results with multiple output formats.

```python
# Pretty-print as table
print(stats)

# Convert to list of dictionaries
data = stats.to_dict()
# [{"name": "forward", "total_ms": 125.4, "calls": 1, "avg_ms": 125.4}, ...]

# Convert to pandas DataFrame (requires pandas)
df = stats.to_dataframe()

# Access individual function stats
for fn_stats in stats.stats:
    print(f"{fn_stats.name}: {fn_stats.total_duration_ms}ms")
```

### `FunctionStats`

Statistics for a single tracked function.

```python
fn_stats.name              # Function name
fn_stats.total_duration_ms # Total time across all calls
fn_stats.call_count        # Number of times called
fn_stats.avg_duration_ms   # Average time per call
```

### `ProfilingError`

Exception raised when profiling fails.

```python
from performax import ProfilingError

try:
    result, stats = profile(my_func)
except ProfilingError as e:
    print(f"Profiling failed: {e}")
```

## Loggers

Performax provides multiple loggers for outputting profile results in different formats.

### `PlainLogger`

Simple ASCII table output (default).

```python
from performax import PlainLogger

logger = PlainLogger()
print(logger.log(stats))
```

```
Function | Total (ms) | Calls | Avg (ms)
-----------------------------------------
forward  | 125.432    | 1     | 125.432
matmul   | 98.765     | 5     | 19.753
```

### `RichLogger`

Colored table output using the [rich](https://github.com/Textualize/rich) library.

```python
from performax import RichLogger

logger = RichLogger(title="My Profile Results")
print(logger.log(stats))
```

Customize styling:

```python
logger = RichLogger(
    title="Performance Report",
    header_style="bold white",
    function_style="cyan",
    total_style="red",
    calls_style="green",
    avg_style="yellow",
)
```

### `JsonLogger`

JSON output for machine-readable results.

```python
from performax import JsonLogger

logger = JsonLogger()
print(logger.log(stats))
# {"functions": [{"name": "forward", "total_ms": 125.4, ...}]}

# Compact output
logger = JsonLogger(indent=None)

# Include metadata (timestamp, totals)
logger = JsonLogger(include_metadata=True)
```

### `CSVLogger`

CSV output for spreadsheets and data analysis.

```python
from performax import CSVLogger

logger = CSVLogger()
print(logger.log(stats))
# function,total_ms,calls,avg_ms
# forward,125.432,1,125.432
# matmul,98.765,5,19.753

# Tab-separated values
logger = CSVLogger(delimiter="\t")

# Without header row
logger = CSVLogger(include_header=False)
```

### `MarkdownLogger`

Markdown table output for documentation and GitHub.

```python
from performax import MarkdownLogger

logger = MarkdownLogger()
print(logger.log(stats))
```

```markdown
| Function | Total (ms) | Calls | Avg (ms) |
|----------|------------|-------|----------|
| forward | 125.432 | 1 | 125.432 |
| matmul | 98.765 | 5 | 19.753 |
```

### `FileLogger`

Single-line output for log files.

```python
from performax import FileLogger

logger = FileLogger(prefix="[PERF]")
print(logger.log(stats))
# [PERF] forward=125.432ms(1x) | matmul=98.765ms(5x)
```

### Custom Loggers

Create your own logger by subclassing `Logger`:

```python
from performax import Logger, ProfileResult

class MyLogger(Logger):
    def log(self, result: ProfileResult) -> str:
        lines = [f"{s.name}: {s.total_duration_ms:.1f}ms" for s in result.stats]
        return "\n".join(lines)
```

## Advanced Usage

### Profiling JIT-compiled Functions

The `@track` decorator works with JIT-compiled functions. The annotation captures the time for the entire JIT execution:

```python
import jax

@track(name="jit_forward")
@jax.jit
def forward(x, w):
    return jnp.dot(x, w)
```

### Nested Function Profiling

Track nested function calls to understand where time is spent:

```python
@track
def inner_computation(x):
    return jnp.sum(x ** 2)

@track
def outer_computation(data):
    results = []
    for x in data:
        results.append(inner_computation(x))
    return results

result, stats = profile(outer_computation, my_data)
# Stats will show both inner_computation and outer_computation
```

### Using with pandas

For data analysis, convert results to a DataFrame:

```python
# pip install pandas
result, stats = profile(main)
df = stats.to_dataframe()

# Sort by average time
df_sorted = df.sort_values("Avg (ms)", ascending=False)

# Filter slow functions
slow_funcs = df[df["Total (ms)"] > 100]
```

### Handling Profiling Errors

JAX only supports one trace at a time. Performax is thread-safe and will raise an error if you try to profile concurrently:

```python
import threading
from performax import profile, ProfilingError

def worker():
    try:
        result, stats = profile(my_func)
    except ProfilingError as e:
        print(f"Could not profile: {e}")

# Only one thread can profile at a time
threads = [threading.Thread(target=worker) for _ in range(3)]
for t in threads:
    t.start()
```

## Requirements

- Python >= 3.12
- JAX >= 0.4.0
- rich (optional, for `RichLogger`)
- pandas (optional, for `to_dataframe()`)

## How It Works

Performax uses JAX's built-in Perfetto tracing to capture timing information:

1. The `@track` decorator wraps functions with `jax.profiler.TraceAnnotation`
2. The `profile()` function starts a JAX trace session
3. After execution, Performax parses the Perfetto trace file
4. Events with the `performax/` prefix are extracted and aggregated
5. Results are returned as a `ProfileResult` with timing statistics

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[development]"

# Run all tests
pytest

# Run with coverage
pytest --cov=performax

# Run specific test file
pytest tests/test_profiler.py

# Run tests excluding slow tests
pytest -m "not slow"
```

### Project Structure

```
performax/
├── performax/
│   ├── __init__.py      # Public API exports
│   ├── decorators.py    # @track decorator
│   ├── profiler.py      # profile() function
│   ├── parser.py        # Perfetto trace parsing
│   ├── result.py        # ProfileResult, FunctionStats
│   ├── logger.py        # Output loggers (Plain, Rich, JSON, etc.)
│   └── exceptions.py    # ProfilingError
├── tests/
│   ├── conftest.py      # Pytest fixtures
│   ├── test_decorators.py
│   ├── test_profiler.py
│   ├── test_parser.py
│   ├── test_result.py
│   ├── test_logger.py
│   └── test_exceptions.py
├── main.py              # Example usage
├── pyproject.toml
└── README.md
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
