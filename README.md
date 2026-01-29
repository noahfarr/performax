# ⚡ Performax

A lightweight profiling framework for JAX that makes it easy to measure execution time of your functions.

## ✨ Features

- 🎯 **Simple decorator API** - Just add `@track` to functions you want to profile
- 🚀 **Minimal overhead** - Uses JAX's built-in Perfetto tracing
- 📊 **Detailed statistics** - Get total time, call counts, and averages
- 📁 **Multiple output formats** - Pretty tables, dictionaries, pandas DataFrames, and more
- 🎨 **Flexible loggers** - Rich tables, JSON, CSV, Markdown, and plain text output
- 🔒 **Thread-safe** - Prevents concurrent profiling conflicts

## 📦 Installation

```bash
pip install performax
```

Or install from source:

```bash
git clone https://github.com/noahfarr/performax.git
cd performax
pip install -e .
```

## 🚀 Quick Start

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

## 🎨 Loggers

Performax provides multiple loggers for outputting profile results in different formats.

### PlainLogger

Simple ASCII table output (default).

```python
from performax import PlainLogger

logger = PlainLogger()
print(logger.log(stats))
```

### RichLogger

Colored table output using the [rich](https://github.com/Textualize/rich) library.

```python
from performax import RichLogger

logger = RichLogger(title="My Profile Results")
print(logger.log(stats))
```

### JsonLogger

JSON output for machine-readable results.

```python
from performax import JsonLogger

logger = JsonLogger()
print(logger.log(stats))
```

### CSVLogger

CSV output for spreadsheets and data analysis.

```python
from performax import CSVLogger

logger = CSVLogger()
print(logger.log(stats))
```

### MarkdownLogger

Markdown table output for documentation and GitHub.

```python
from performax import MarkdownLogger

logger = MarkdownLogger()
print(logger.log(stats))
```

### FileLogger

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

## 🔧 Advanced Usage

### Profiling JIT-compiled Functions

The `@track` decorator works with JIT-compiled functions:

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
```

### Using with pandas

For data analysis, convert results to a DataFrame:

```python
result, stats = profile(main)
df = stats.to_dataframe()

df_sorted = df.sort_values("Avg (ms)", ascending=False)
slow_funcs = df[df["Total (ms)"] > 100]
```

## 📋 Requirements

- Python >= 3.12
- JAX >= 0.4.0
- rich (optional, for `RichLogger`)
- pandas (optional, for `to_dataframe()`)

## 🛠️ Development

```bash
pip install -e ".[development]"

pytest
pytest --cov=performax
pytest -m "not slow"
```

## 📄 License

MIT

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
