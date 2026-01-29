"""Main profiler functionality."""

import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Callable, TypeVar

import jax

from .exceptions import ProfilingError
from .parser import parse_perfetto_trace
from .result import FunctionStats, ProfileResult

T = TypeVar("T")

_profile_lock = threading.Lock()


def profile(fn: Callable[..., T], *args, **kwargs) -> tuple[T, ProfileResult]:
    """Profile a function and return timing statistics for @track decorated functions.

    This function executes the provided callable while capturing JAX profiler traces.
    Any functions decorated with @track will have their timing information collected
    and returned as a ProfileResult.

    Args:
        fn: The function to profile.
        *args: Positional arguments to pass to fn.
        **kwargs: Keyword arguments to pass to fn.

    Returns:
        A tuple of (result, ProfileResult) where result is the return value of fn
        and ProfileResult contains timing statistics for tracked functions.

    Raises:
        ProfilingError: If profiling fails or if another profile is already running.
    """
    acquired = _profile_lock.acquire(blocking=False)
    if not acquired:
        raise ProfilingError(
            "Another profiling session is already in progress. "
            "JAX only supports one trace at a time."
        )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir)

            with jax.profiler.trace(str(trace_path)):
                result = fn(*args, **kwargs)
                if hasattr(result, "block_until_ready"):
                    result.block_until_ready()
                elif isinstance(result, tuple):
                    for item in result:
                        if hasattr(item, "block_until_ready"):
                            item.block_until_ready()

            events = parse_perfetto_trace(trace_path)

            stats_by_name: dict[str, dict] = defaultdict(
                lambda: {"total_us": 0.0, "count": 0}
            )

            for event in events:
                stats_by_name[event.name]["total_us"] += event.duration_us
                stats_by_name[event.name]["count"] += 1

            function_stats = [
                FunctionStats(
                    name=name,
                    total_duration_ms=data["total_us"] / 1000.0,
                    call_count=data["count"],
                )
                for name, data in stats_by_name.items()
            ]

            return result, ProfileResult(function_stats)

    except ProfilingError:
        raise
    except Exception as e:
        raise ProfilingError(f"Profiling failed: {e}") from e
    finally:
        _profile_lock.release()
