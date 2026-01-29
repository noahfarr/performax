"""Example demonstrating the RichLogger for colorful table output.

Requires the 'rich' package: pip install rich
"""

import jax.numpy as jnp

from performax import RichLogger, profile, track


@track
def matmul(a, b):
    """Matrix multiplication."""
    return jnp.dot(a, b)


@track(name="forward_pass")
def forward(x, weights):
    """Forward pass through multiple layers."""
    for w in weights:
        x = matmul(x, w)
    return x


def main():
    x = jnp.ones((1000, 512))
    weights = [jnp.ones((512, 512)) for _ in range(5)]

    result, stats = profile(forward, x, weights)

    # Default styling
    logger = RichLogger()
    print(logger.log(stats))


if __name__ == "__main__":
    main()
