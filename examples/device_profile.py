import performax as px

px.enable_device_profiling()

import jax
import jax.numpy as jnp

px.enable_barriers()


@px.track(name="attention")
def scaled_dot_product_attention(q, k, v):
    d_k = q.shape[-1]
    scores = jnp.dot(q, k.T) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.dot(weights, v)


@px.track(name="feed_forward")
def feed_forward(x, w1, w2):
    return jnp.dot(jax.nn.relu(jnp.dot(x, w1)), w2)


@px.track(name="transformer_block")
def transformer_block(x, q_w, k_w, v_w, ff_w1, ff_w2):
    q = jnp.dot(x, q_w)
    k = jnp.dot(x, k_w)
    v = jnp.dot(x, v_w)

    x = x + scaled_dot_product_attention(q, k, v)
    x = x + feed_forward(x, ff_w1, ff_w2)
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

    @px.track(name="full_forward")
    def forward(x):
        for weights in layer_weights:
            x = transformer_block(x, *weights)
        return x

    fn = jax.jit(forward)

    _, stats = px.profile(fn, warmup=True)(x)
    print(stats.device)


if __name__ == "__main__":
    main()
