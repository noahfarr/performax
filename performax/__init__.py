"""Performax: A JAX profiling framework.

This package provides a simple way to profile JAX functions using the
@track decorator and profile() function.

Example:
    from performax import track, profile
    import jax.numpy as jnp

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
"""

from .decorators import track
from .exceptions import ProfilingError
from .profiler import profile
from .result import FunctionStats, ProfileResult

__all__ = [
    "track",
    "profile",
    "ProfileResult",
    "FunctionStats",
    "ProfilingError",
]
