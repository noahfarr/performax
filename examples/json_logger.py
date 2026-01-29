"""Example demonstrating the JsonLogger for machine-readable output."""

import jax.numpy as jnp

from performax import JsonLogger, profile, track


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

    # Pretty-printed JSON
    logger = JsonLogger(indent=2)
    print("Pretty-printed JSON:")
    print(logger.log(stats))

    print("\n--- With metadata ---\n")

    # With metadata
    logger_with_meta = JsonLogger(indent=2, include_metadata=True)
    print(logger_with_meta.log(stats))

    print("\n--- Compact JSON ---\n")

    # Compact JSON (no indentation)
    compact_logger = JsonLogger(indent=None)
    print(compact_logger.log(stats))


if __name__ == "__main__":
    main()
