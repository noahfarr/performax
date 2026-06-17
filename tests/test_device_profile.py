"""Integration test for performax.device_profile (requires a CUDA GPU).

device_profile needs CUDA command buffers disabled, which is controlled by an
XLA flag read at backend initialization. To guarantee that ordering regardless
of what other tests already initialized, the profiled run happens in a fresh
subprocess with XLA_FLAGS set.
"""

import os
import subprocess
import sys
import textwrap

import pytest


def _has_cuda() -> bool:
    try:
        import jax

        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


pytestmark = [pytest.mark.integration, pytest.mark.slow]


PROGRAM = textwrap.dedent(
    """
    import performax as px
    px.enable_device_profiling()
    import jax, jax.numpy as jnp
    px.enable_barriers()

    class M:
        def matmul(self, x):
            return (x @ x.T) + 1.0
        def train(self, x):
            for _ in range(8):
                x = self.matmul(x) / (jnp.linalg.norm(x) + 1.0)
            return x

    M.matmul = px.scope(name="matmul")(M.matmul)
    M.train = px.scope(name="train")(M.train)

    f = jax.jit(jax.vmap(M().train))
    x = jnp.ones((4, 64, 64), jnp.float32)
    _, res = px.device_profile(f, warmup=True)(x)
    names = {s.name: s.total_duration_ms for s in res.stats}
    assert "train" in names and names["train"] > 0.0, names
    assert "matmul" in names and names["matmul"] > 0.0, names
    print("DEVICE_PROFILE_OK", sorted(names))
    """
)


@pytest.mark.skipif(not _has_cuda(), reason="requires a CUDA GPU")
def test_device_profile_attributes_scopes_under_jit():
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
    ).strip()
    proc = subprocess.run(
        [sys.executable, "-c", PROGRAM],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "DEVICE_PROFILE_OK" in proc.stdout, proc.stdout
