import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from .decorators import PERFORMAX_PREFIX


@dataclass
class TraceEvent:
    name: str
    duration_us: float


def _find_trace_file(trace_path: Path) -> Path | None:
    for path in trace_path.rglob("*.trace.json.gz"):
        return path
    return None


def parse_perfetto_trace(trace_path: Path) -> list[TraceEvent]:
    trace_file = _find_trace_file(trace_path)

    if trace_file is None:
        return []

    with gzip.open(trace_file, "rt", encoding="utf-8") as f:
        data = json.load(f)

    events = []
    trace_events = data.get("traceEvents", [])

    for event in trace_events:
        if event.get("ph") != "X":
            continue

        name = event.get("name", "")
        if not name.startswith(PERFORMAX_PREFIX):
            continue

        func_name = name[len(PERFORMAX_PREFIX) :]
        duration_us = event.get("dur", 0)

        events.append(TraceEvent(name=func_name, duration_us=duration_us))

    return events
