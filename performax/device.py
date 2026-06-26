import gzip
import json
import os
import re
import warnings
from pathlib import Path

import jax

from .decorators import PERFORMAX_PREFIX
from .parser import TraceEvent, _find_trace_file

COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="

_DEVICE_LANE_PREFIX = "/device:"


def enable_device_profiling() -> None:
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_enable_command_buffer" not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} {COMMAND_BUFFER_FLAG}".strip()

    try:
        if jax._src.xla_bridge.backends_are_initialized():
            warnings.warn(
                "enable_device_profiling() ran after JAX initialized its "
                "backend; XLA_FLAGS is read at init, so command buffers may "
                "still be enabled. Set XLA_FLAGS=--xla_gpu_enable_command_buffer= "
                "before importing jax / running any JAX op.",
                stacklevel=2,
            )
    except Exception:
        pass


def _scopes_in(op_name: str, prefix: str) -> list[str]:
    return _scope_pattern(prefix).findall(op_name)


def _scope_pattern(prefix: str) -> "re.Pattern[str]":
    marker = re.escape(prefix.rstrip("/"))
    return re.compile(rf"{marker}/([^/)]+)")


def parse_device_trace(
    trace_path: Path, *, prefix: str = PERFORMAX_PREFIX, inclusive: bool = True
) -> list[TraceEvent]:
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
