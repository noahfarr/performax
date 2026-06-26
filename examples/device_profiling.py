"""Example demonstrating device-side profiling of a ``jax.jit``-ed computation.

The host-side ``track`` / ``profile`` path measures regions with
``jax.profiler.TraceAnnotation``, which only records while the Python wrapper
body runs. That cannot measure a jitted computation: on warm dispatch JAX runs
the cached executable without re-running the wrappers, so the annotations never
fire.

``scope`` + ``device_profile`` instead measure steady-state *device* time per
region: ``scope`` tags the HLO ops via ``jax.named_scope`` so the region name
survives into the compiled program, and ``device_profile`` reads the device
timeline and attributes each GPU kernel to its enclosing scope.

Requirements (CUDA/GPU only in this version):

* Call ``enable_device_profiling()`` before the first JAX op (it disables CUDA
  command buffers, which otherwise strip the per-op scope metadata). Equivalently
  set ``XLA_FLAGS=--xla_gpu_enable_command_buffer=`` before launching.
* Annotate regions with ``scope`` *inside* the jitted function.
* ``enable_barriers()`` is recommended so XLA does not fuse across region
  boundaries and blur the attribution.
"""

import performax as px

# Must run before the first JAX op so the scope metadata reaches the device trace.
px.enable_device_profiling()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# Recommended: keep XLA from fusing across region boundaries.
px.enable_barriers()


@px.scope(name="attention")
def scaled_dot_product_attention(q, k, v):
    """Scaled dot-product attention."""
    d_k = q.shape[-1]
    scores = jnp.dot(q, k.T) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.dot(weights, v)


@px.scope(name="feed_forward")
def feed_forward(x, w1, w2):
    """Two-layer feed-forward network with ReLU."""
    return jnp.dot(jax.nn.relu(jnp.dot(x, w1)), w2)


@px.scope(name="transformer_block")
def transformer_block(x, q_w, k_w, v_w, ff_w1, ff_w2):
    """Single transformer block: attention + feed-forward."""
    q = jnp.dot(x, q_w)
    k = jnp.dot(x, k_w)
    v = jnp.dot(x, v_w)

    x = x + scaled_dot_product_attention(q, k, v)
    x = x + feed_forward(x, ff_w1, ff_w2)
    return x


def main():
    if not any(d.platform == "gpu" for d in jax.devices()):
        print("device_profile requires a CUDA GPU; no GPU backend found.")
        return

    key = jax.random.PRNGKey(0)
    seq_len, d_model, d_ff = 128, 256, 512
    n_layers = 4

    keys = jax.random.split(key, 1 + n_layers * 5)
    x = jax.random.normal(keys[0], (seq_len, d_model))

    layer_weights = []
    for i in range(n_layers):
        base = 1 + i * 5
        layer_weights.append((
            jax.random.normal(keys[base], (d_model, d_model)),      # q_w
            jax.random.normal(keys[base + 1], (d_model, d_model)),  # k_w
            jax.random.normal(keys[base + 2], (d_model, d_model)),  # v_w
            jax.random.normal(keys[base + 3], (d_model, d_ff)),     # ff_w1
            jax.random.normal(keys[base + 4], (d_ff, d_model)),     # ff_w2
        ))

    @px.scope(name="full_forward")
    def forward(x):
        for weights in layer_weights:
            x = transformer_block(x, *weights)
        return x

    fn = jax.jit(forward)

    # warmup=True (the default) reports steady-state device time, excluding
    # compilation. call_count is the number of GPU kernels per scope.
    _, stats = px.device_profile(fn, warmup=True)(x)

    print(px.RichLogger(title="Device profile (per scope)").log(stats))


if __name__ == "__main__":
    main()
