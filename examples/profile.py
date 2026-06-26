"""Basic host-side profiling: decorate with @track, profile, print the result."""

import jax.numpy as jnp

from performax import profile, track


@track
def matmul(a, b):
    return jnp.dot(a, b)


@track()
def forward(x, weights):
    for w in weights:
        x = matmul(x, w)
    return x


def main():
    x = jnp.ones((1000, 512))
    weights = [jnp.ones((512, 512)) for _ in range(5)]

    result, stats = profile(forward)(x, weights)
    print(stats.host)


if __name__ == "__main__":
    main()
