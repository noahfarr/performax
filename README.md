# Performax

A lightweight profiling framework for JAX that makes it easy to measure execution time of your functions.

## Features

- **Simple decorator API** - Just add `@track` to functions you want to profile
- **Host and device timelines** - One capture reports host dispatch time and device (GPU) kernel time per region
- **Minimal overhead** - Uses JAX's built-in Perfetto tracing
- **Detailed statistics** - Total time, call counts, and averages, printed as a plain table
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

result, stats = profile(main)()
print(stats.host)
```

`profile(fn)` returns a wrapped callable; call it to run `fn` and collect a
`Profile`. The `Profile` exposes two timelines from the same capture:
`stats.host` (host-side dispatch time) and `stats.device` (device kernel time).
Each is a `ProfileResult` that prints as a table:

```
Function | Total (ms) | Calls | Avg (ms)
-----------------------------------------
forward  | 125.432    | 1     | 125.432
matmul   | 98.765     | 5     | 19.753
```

`print(stats)` prints whichever timelines have data. For raw values use
`stats.host.to_dict()` or `stats.host.to_dataframe()` (the latter needs `pandas`).

### Excluding JIT compilation time

The first call to a function traces and compiles it via XLA, and that cost lands
in the profile. Pass `warmup=True` to run the function once before tracing, so
the reported time reflects steady-state execution:

```python
result, stats = profile(forward, warmup=True)(x, weights)
```

## Host vs device time

`@track` annotates a region two ways at once: a host-side `TraceAnnotation` and
a compiled-in `jax.named_scope`. After one `profile` capture:

- `stats.host` is the host-side `TraceAnnotation` durations (dispatch / Python
  time). Eager code populates this.
- `stats.device` is device kernel time attributed to each region. A warm
  `jax.jit`-ed computation populates this; its host annotations don't re-run on
  replay, so `stats.host` is empty for it.

### Profiling inside `jax.jit` (GPU)

To get per-region device time for a jitted computation:

```python
import performax as px

# Must run before the first JAX op: disables CUDA command buffers so the
# per-op scope metadata survives into the device trace.
px.enable_device_profiling()

import jax
import jax.numpy as jnp

# Recommended: stop XLA from fusing across region boundaries.
px.enable_barriers()

@px.track(name="layer")
def layer(x, w):
    return jax.nn.relu(jnp.dot(x, w))

@px.track(name="forward")
def forward(x, weights):
    for w in weights:
        x = layer(x, w)
    return x

x = jnp.ones((512, 512))
weights = [jnp.ones((512, 512)) for _ in range(5)]

fn = jax.jit(forward)
result, stats = px.profile(fn, warmup=True)(x, weights)
print(stats.device)
```

Notes:

- **CUDA/GPU only** in this version.
- `@track` regions must execute *inside* the jitted function.
- `enable_device_profiling()` reads `XLA_FLAGS` at backend init, so call it
  before the first JAX op. Equivalently, set
  `XLA_FLAGS=--xla_gpu_enable_command_buffer=` before launching.
- In `stats.device`, `call_count` is the number of device kernels attributed to
  the region, not the number of Python calls.

See the [`examples/`](examples/) directory for runnable scripts.

## Requirements

- Python >= 3.11
- JAX >= 0.4.0
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
