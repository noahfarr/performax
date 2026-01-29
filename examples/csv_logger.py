"""Example demonstrating the CSVLogger for spreadsheet-compatible output."""

import jax.numpy as jnp

from performax import CSVLogger, profile, track


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

    # Default CSV with header
    logger = CSVLogger()
    print("CSV with header:")
    print(logger.log(stats))

    print("\n--- Without header ---\n")

    # Without header
    no_header_logger = CSVLogger(include_header=False)
    print(no_header_logger.log(stats))

    print("\n--- Tab-separated (TSV) ---\n")

    # Tab-separated
    tsv_logger = CSVLogger(delimiter="\t")
    print(tsv_logger.log(stats))


if __name__ == "__main__":
    main()
