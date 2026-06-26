import tempfile
import threading
from pathlib import Path
from typing import Callable, TypeVar

import jax

from .device import parse_device_trace
from .exceptions import ProfilingError
from .parser import parse_perfetto_trace
from .result import Profile, ProfileResult

T = TypeVar("T")

_profile_lock = threading.Lock()


def profile(
    fn: Callable[..., T], *, warmup: bool = False, inclusive: bool = True
) -> Callable[..., tuple[T, Profile]]:
    def wrapper(*args, **kwargs) -> tuple[T, Profile]:
        acquired = _profile_lock.acquire(blocking=False)
        if not acquired:
            raise ProfilingError(
                "Another profiling session is already in progress. "
                "JAX only supports one trace at a time."
            )

        try:
            if warmup:
                warmup_result = fn(*args, **kwargs)
                jax.block_until_ready(warmup_result)
                del warmup_result

            with tempfile.TemporaryDirectory() as tmpdir:
                trace_path = Path(tmpdir)

                with jax.profiler.trace(str(trace_path)):
                    result = fn(*args, **kwargs)
                    jax.block_until_ready(result)

                host = ProfileResult.from_events(parse_perfetto_trace(trace_path))
                device = ProfileResult.from_events(
                    parse_device_trace(trace_path, inclusive=inclusive)
                )

            return result, Profile(host=host, device=device)

        except ProfilingError:
            raise
        except Exception as e:
            raise ProfilingError(f"Profiling failed: {e}") from e
        finally:
            _profile_lock.release()

    return wrapper
