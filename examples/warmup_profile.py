"""Warmup profiling to exclude JIT compilation time.

When JAX runs a function for the first time, it traces and compiles it via XLA.
That compilation cost is captured in the profile. With warmup=True the function
runs once before tracing, so the profiled run reflects steady-state time.
"""

import jax
import jax.numpy as jnp

from performax import enable_barriers, profile, track


@track(name="attention", barrier=True)
def scaled_dot_product_attention(q, k, v):
    d_k = q.shape[-1]
    scores = jnp.dot(q, k.T) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.dot(weights, v)


@track(name="feed_forward", barrier=True)
def feed_forward(x, w1, w2):
    return jnp.dot(jax.nn.relu(jnp.dot(x, w1)), w2)


@track(name="layer_norm", barrier=True)
def layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


@track(name="transformer_block")
def transformer_block(x, q_w, k_w, v_w, ff_w1, ff_w2):
    q = jnp.dot(x, q_w)
    k = jnp.dot(x, k_w)
    v = jnp.dot(x, v_w)

    attn_out = scaled_dot_product_attention(q, k, v)
    x = layer_norm(x + attn_out)

    ff_out = feed_forward(x, ff_w1, ff_w2)
    x = layer_norm(x + ff_out)
    return x


def main():
    key = jax.random.PRNGKey(0)
    seq_len, d_model, d_ff = 128, 256, 512
    n_layers = 4

    keys = jax.random.split(key, 1 + n_layers * 5)
    x = jax.random.normal(keys[0], (seq_len, d_model))

    layer_weights = []
    for i in range(n_layers):
        base = 1 + i * 5
        layer_weights.append(
            (
                jax.random.normal(keys[base], (d_model, d_model)),
                jax.random.normal(keys[base + 1], (d_model, d_model)),
                jax.random.normal(keys[base + 2], (d_model, d_model)),
                jax.random.normal(keys[base + 3], (d_model, d_ff)),
                jax.random.normal(keys[base + 4], (d_ff, d_model)),
            )
        )

    @track(name="full_forward")
    def forward(x):
        for weights in layer_weights:
            x = transformer_block(x, *weights)
        return x

    enable_barriers()

    _, stats_cold = profile(forward)(x)
    print("=== Without warmup (includes JIT compilation) ===")
    print(stats_cold.host)

    print()

    _, stats_warm = profile(forward, warmup=True)(x)
    print("=== With warmup (steady-state) ===")
    print(stats_warm.host)


if __name__ == "__main__":
    main()
