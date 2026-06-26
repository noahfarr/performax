# Performax

A lightweight profiling framework for JAX that makes it easy to measure execution time of your functions.

## Features

- **Simple decorator API** - Just add `@track` to functions you want to profile
- **Device-side profiling** - Measure steady-state GPU time per region *inside* a `jax.jit`-ed computation with `@scope`
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

# profile(fn) returns a wrapped callable; call it to run and collect stats
result, stats = profile(main)()
print(stats)
```

Output:
```
Function | Total (ms) | Calls | Avg (ms)
-----------------------------------------
forward  | 125.432    | 1     | 125.432
matmul   | 98.765     | 5     | 19.753
```

### Excluding JIT compilation time

The first call to a function traces and compiles it via XLA, and that
compilation cost lands in the profile. Pass `warmup=True` to run the function
once before tracing, so the reported time reflects steady-state execution:

```python
result, stats = profile(forward, warmup=True)(x, weights)
```

## Device-side profiling (under `jax.jit`)

`track` measures host-side time with `jax.profiler.TraceAnnotation`, which only
records while the Python wrapper body runs. That cannot measure a jitted
computation: on warm dispatch JAX runs the cached executable without re-running
the wrappers, so the annotations never fire.

To measure **steady-state device time per region under jit**, annotate regions
with `@scope` (which tags HLO ops via `jax.named_scope`) and profile with
`device_profile`, which reads the device timeline and attributes each GPU kernel
to its enclosing scope.

```python
import performax as px

# Must run before the first JAX op: disables CUDA command buffers so the
# per-op scope metadata survives into the device trace.
px.enable_device_profiling()

import jax
import jax.numpy as jnp

# Recommended: stop XLA from fusing across region boundaries.
px.enable_barriers()

@px.scope(name="layer")
def layer(x, w):
    return jax.nn.relu(jnp.dot(x, w))

@px.scope(name="forward")
def forward(x, weights):
    for w in weights:
        x = layer(x, w)
    return x

x = jnp.ones((512, 512))
weights = [jnp.ones((512, 512)) for _ in range(5)]

fn = jax.jit(forward)
result, stats = px.device_profile(fn, warmup=True)(x, weights)
print(stats)
```

Notes:

- **CUDA/GPU only** in this version.
- `@scope` must execute *inside* the jitted function (decorate before jitting).
- `enable_device_profiling()` reads `XLA_FLAGS` at backend init, so call it
  before the first JAX op. Equivalently, set
  `XLA_FLAGS=--xla_gpu_enable_command_buffer=` before launching.
- `call_count` in the result is the number of device kernels attributed to the
  scope, not the number of Python calls.

See [`examples/device_profiling.py`](examples/device_profiling.py) for a runnable example.

## Loggers

`profile` and `device_profile` return a `ProfileResult` that prints as a plain
table. For other formats, pass the result to a logger:

```python
from performax import ConsoleLogger, RichLogger, FileLogger

print(ConsoleLogger().log(stats))   # ASCII table
print(RichLogger().log(stats))      # colorful table (needs `rich`)
print(FileLogger().log(stats))      # single-line, log-friendly
```

You can also get the raw data with `stats.to_dict()` or `stats.to_dataframe()`
(needs `pandas`).

See the [`examples/`](examples/) directory for runnable scripts.

## Requirements

- Python >= 3.11
- JAX >= 0.4.0
- rich (optional, for `RichLogger`)
- pandas (optional, for `to_dataframe()`)

## Development

```bash
pip install -e ".[development]"

pytest
pytest --cov=performax
pytest -m "not slow"
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
