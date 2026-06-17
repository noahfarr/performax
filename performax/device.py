"""Device-side per-region profiling for ``jax.jit``-ed computations.

The host-side :func:`performax.track` / :func:`performax.profile` path measures
regions with ``jax.profiler.TraceAnnotation``, which only records while the
Python wrapper body runs. That cannot measure a jitted computation: on warm
dispatch JAX runs the cached executable without re-running the wrappers, so the
annotations never fire.

This module measures **steady-state device time per region under jit**. It pairs
:func:`performax.scope` (which tags HLO ops with ``jax.named_scope``) with
:func:`device_profile`, which traces the device timeline and attributes each GPU
kernel to the scope recorded in its op metadata.

Requirements (CUDA/GPU only in this version):

* Annotate regions with :func:`performax.scope` *inside* the jitted function.
* Disable CUDA command buffers while profiling, otherwise XLA bundles ops into
  ``command_buffer`` nodes and strips the per-op scope metadata. Call
  :func:`enable_device_profiling` before the first JAX op, or set the env var
  ``XLA_FLAGS=--xla_gpu_enable_command_buffer=`` before launching.
* Recommended: :func:`performax.enable_barriers` so XLA does not fuse across
  region boundaries (which would blur the attribution).
"""

import gzip
import json
import os
import re
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, TypeVar

import jax

from .decorators import PERFORMAX_PREFIX
from .exceptions import ProfilingError
from .parser import TraceEvent, _find_trace_file
from .profiler import _profile_lock
from .result import FunctionStats, ProfileResult

T = TypeVar("T")

COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="

# Device-activity lanes are named like "/device:GPU:0" in the Perfetto trace.
_DEVICE_LANE_PREFIX = "/device:"


def enable_device_profiling() -> None:
    """Disable CUDA command buffers so scope metadata reaches the device trace.

    Appends ``--xla_gpu_enable_command_buffer=`` to ``XLA_FLAGS`` if not already
    present. XLA reads this flag at backend initialization, so this must run
    before the first JAX operation; a warning is emitted if the backend already
    looks initialized.
    """
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_enable_command_buffer" not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} {COMMAND_BUFFER_FLAG}".strip()

    try:  # best-effort: the check is on a private API and is only advisory
        if jax._src.xla_bridge.backends_are_initialized():
            warnings.warn(
                "enable_device_profiling() ran after JAX initialized its "
                "backend; XLA_FLAGS is read at init, so command buffers may "
                "still be enabled. Set XLA_FLAGS=--xla_gpu_enable_command_buffer= "
                "before importing jax / running any JAX op.",
                stacklevel=2,
            )
    except Exception:  # pragma: no cover - depends on JAX internals
        pass


def _scopes_in(op_name: str, prefix: str) -> list[str]:
    """Return the performax scopes enclosing a device op, outermost first.

    ``op_name`` is the HLO op metadata. It looks like
    ``jit(fn)/performax/train/performax/_update_step/dot`` but JAX transforms
    decorate it, e.g. ``jit(train)/vmap(performax/train)/dot_general`` — so the
    ``performax/<scope>`` markers can be nested inside ``transform(...)`` wrappers
    rather than being clean ``/``-delimited segments. Match them anywhere; a
    scope name runs until the next ``/`` or closing ``)``.
    """
    return _scope_pattern(prefix).findall(op_name)


def _scope_pattern(prefix: str) -> "re.Pattern[str]":
    marker = re.escape(prefix.rstrip("/"))
    return re.compile(rf"{marker}/([^/)]+)")


def parse_device_trace(
    trace_path: Path, *, prefix: str = PERFORMAX_PREFIX, inclusive: bool = True
) -> list[TraceEvent]:
    """Parse device kernel events from a Perfetto trace, keyed by scope.

    Only GPU-activity events (on a ``/device:*`` lane and carrying
    ``kernel_details``) are considered, so a kernel is counted exactly once. With
    ``inclusive`` (the default) a kernel's duration is credited to *every*
    enclosing scope, so a parent scope accrues the full total and a child accrues
    its share; with ``inclusive=False`` only the innermost scope is credited.
    """
    trace_file = _find_trace_file(trace_path)
    if trace_file is None:
        return []

    with gzip.open(trace_file, "rt", encoding="utf-8") as f:
        data = json.load(f)

    trace_events = data.get("traceEvents", [])

    device_pids = {
        event.get("pid")
        for event in trace_events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and str(event.get("args", {}).get("name", "")).startswith(_DEVICE_LANE_PREFIX)
    }

    events: list[TraceEvent] = []
    for event in trace_events:
        if event.get("ph") != "X" or event.get("pid") not in device_pids:
            continue
        args = event.get("args", {})
        if "kernel_details" not in args:
            continue

        scopes = _scopes_in(str(args.get("name", "")), prefix)
        if not scopes:
            continue
        if not inclusive:
            scopes = scopes[-1:]

        duration_us = event.get("dur", 0)
        for scope_name in scopes:
            events.append(TraceEvent(name=scope_name, duration_us=duration_us))

    return events


def _aggregate(events: list[TraceEvent]) -> ProfileResult:
    stats_by_name: dict[str, dict] = defaultdict(lambda: {"total_us": 0.0, "count": 0})
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
    return ProfileResult(function_stats)


def device_profile(
    fn: Callable[..., T],
    *,
    warmup: bool = True,
    inclusive: bool = True,
) -> Callable[..., tuple[T, ProfileResult]]:
    """Profile *device* time per :func:`performax.scope` region of a jitted fn.

    Mirrors :func:`performax.profile` but reads the device timeline. ``warmup``
    defaults to ``True`` so the reported time is steady-state (compilation
    excluded). For ``FunctionStats``, ``call_count`` is the number of device
    kernels attributed to the scope. Raises :class:`ProfilingError` if no scoped
    device kernels are found (see the module docstring for the requirements).
    """

    def wrapper(*args, **kwargs) -> tuple[T, ProfileResult]:
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

                events = parse_device_trace(trace_path, inclusive=inclusive)

            if not events:
                raise ProfilingError(
                    "No performax scopes found on the device timeline. Likely "
                    "causes: (1) CUDA command buffers are enabled — call "
                    "performax.enable_device_profiling() (or set "
                    "XLA_FLAGS=--xla_gpu_enable_command_buffer=) before the first "
                    "JAX op; (2) no performax.scope ran inside the jitted "
                    "function; (3) the computation did not run on a GPU backend."
                )

            return result, _aggregate(events)

        except ProfilingError:
            raise
        except Exception as e:
            raise ProfilingError(f"Device profiling failed: {e}") from e
        finally:
            _profile_lock.release()

    return wrapper
