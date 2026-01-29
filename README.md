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
