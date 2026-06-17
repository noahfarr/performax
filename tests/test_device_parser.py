"""Tests for the device-trace parser (performax.device.parse_device_trace)."""

import gzip
import json
import tempfile
from pathlib import Path

from performax.decorators import PERFORMAX_PREFIX
from performax.device import _scopes_in, parse_device_trace

GPU_PID = 1
HOST_PID = 7


def _device_proc(pid: int = GPU_PID, name: str = "/device:GPU:0") -> dict:
    return {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": name}}


def _host_proc(pid: int = HOST_PID, name: str = "/host:CPU") -> dict:
    return {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": name}}


def _kernel(op_name: str, dur: float, *, pid: int = GPU_PID, kernel: bool = True) -> dict:
    args = {"name": op_name}
    if kernel:
        args["kernel_details"] = "regs:56 ..."
    return {"ph": "X", "pid": pid, "dur": dur, "args": args}


def _write_trace(tmpdir: Path, events: list[dict]) -> None:
    trace_dir = tmpdir / "plugins" / "profile" / "12345"
    trace_dir.mkdir(parents=True)
    trace_file = trace_dir / "hostname.trace.json.gz"
    with gzip.open(trace_file, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)


class TestScopesIn:
    """Tests for the op-metadata -> scope-name extraction."""

    def test_single_scope(self):
        name = "jit(f)/performax/train/dot_general"
        assert _scopes_in(name, PERFORMAX_PREFIX) == ["train"]

    def test_vmap_wrapped_scope(self):
        # JAX transforms decorate the metadata, e.g. vmap(performax/train).
        name = "jit(train)/vmap(performax/train)/dot_general"
        assert _scopes_in(name, PERFORMAX_PREFIX) == ["train"]

    def test_nested_scopes_outermost_first(self):
        name = "jit(f)/performax/train/performax/_update_step/dot"
        assert _scopes_in(name, PERFORMAX_PREFIX) == ["train", "_update_step"]

    def test_nested_scopes_inside_transform(self):
        name = "jit(train)/vmap(performax/train/performax/_update_step)/add"
        assert _scopes_in(name, PERFORMAX_PREFIX) == ["train", "_update_step"]

    def test_no_scope(self):
        assert _scopes_in("jit(f)/dot_general", PERFORMAX_PREFIX) == []


class TestParseDeviceTrace:
    """Tests for parse_device_trace."""

    def test_attributes_kernel_to_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_trace(
                Path(tmp),
                [
                    _device_proc(),
                    _kernel("jit(f)/vmap(performax/train)/dot_general", 1000),
                    _kernel("jit(f)/vmap(performax/train)/add", 500),
                ],
            )
            events = parse_device_trace(Path(tmp))
            assert sorted(e.name for e in events) == ["train", "train"]
            assert sum(e.duration_us for e in events) == 1500

    def test_inclusive_credits_every_enclosing_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_trace(
                Path(tmp),
                [
                    _device_proc(),
                    # one kernel nested under train -> _update_step
                    _kernel("performax/train/performax/_update_step/dot", 800),
                    # one kernel only under train
                    _kernel("performax/train/divide", 200),
                ],
            )
            events = parse_device_trace(Path(tmp), inclusive=True)
            by_scope: dict[str, float] = {}
            for e in events:
                by_scope[e.name] = by_scope.get(e.name, 0.0) + e.duration_us
            # train is inclusive: 800 (shared with child) + 200
            assert by_scope["train"] == 1000
            assert by_scope["_update_step"] == 800

    def test_exclusive_credits_only_innermost(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_trace(
                Path(tmp),
                [
                    _device_proc(),
                    _kernel("performax/train/performax/_update_step/dot", 800),
                    _kernel("performax/train/divide", 200),
                ],
            )
            events = parse_device_trace(Path(tmp), inclusive=False)
            by_scope: dict[str, float] = {}
            for e in events:
                by_scope[e.name] = by_scope.get(e.name, 0.0) + e.duration_us
            assert by_scope["_update_step"] == 800
            assert by_scope["train"] == 200  # only the kernel whose innermost is train

    def test_ignores_non_device_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_trace(
                Path(tmp),
                [
                    _device_proc(),
                    _host_proc(),
                    # a host-lane event carrying a performax op name must be skipped
                    _kernel("performax/train/op", 999, pid=HOST_PID, kernel=False),
                ],
            )
            assert parse_device_trace(Path(tmp)) == []

    def test_ignores_events_without_kernel_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_trace(
                Path(tmp),
                [
                    _device_proc(),
                    _kernel("performax/train/op", 999, kernel=False),
                ],
            )
            assert parse_device_trace(Path(tmp)) == []

    def test_ignores_unscoped_kernels(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_trace(
                Path(tmp),
                [
                    _device_proc(),
                    _kernel("jit(f)/some_fusion", 1000),
                ],
            )
            assert parse_device_trace(Path(tmp)) == []

    def test_missing_trace_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert parse_device_trace(Path(tmp)) == []
