"""Example usage of the performax profiling framework."""

import jax.numpy as jnp

from performax import profile, track


@track
def add(a, b):
    return a + b


@track
def matmul(a, b):
    return jnp.dot(a, b)


@track(name="forward")
def forward(x, weights):
    for w in weights:
        x = matmul(x, w)
        x = add(x, x)
    return x


def main():
    x = jnp.ones((1000, 512))
    weights = [jnp.ones((512, 512)) for _ in range(5)]
    return forward(x, weights)


if __name__ == "__main__":
    result, stats = profile(main)
    print(stats)
