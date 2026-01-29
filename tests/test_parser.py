"""Tests for the Perfetto trace parser."""

import gzip
import json
import tempfile
from pathlib import Path

import pytest

from performax.decorators import PERFORMAX_PREFIX
from performax.parser import TraceEvent, _find_trace_file, parse_perfetto_trace


class TestTraceEvent:
    """Tests for the TraceEvent dataclass."""

    def test_basic_creation(self):
        """Test creating a TraceEvent instance."""
        event = TraceEvent(name="test_func", duration_us=1000.0)
        assert event.name == "test_func"
        assert event.duration_us == 1000.0

    def test_zero_duration(self):
        """Test TraceEvent with zero duration."""
        event = TraceEvent(name="instant", duration_us=0.0)
        assert event.duration_us == 0.0

    def test_large_duration(self):
        """Test TraceEvent with large duration value."""
        event = TraceEvent(name="slow", duration_us=1_000_000_000.0)
        assert event.duration_us == 1_000_000_000.0


class TestFindTraceFile:
    """Tests for the _find_trace_file function."""

    def test_finds_trace_file(self):
        """Test finding trace file in nested structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "plugins" / "profile" / "12345"
            trace_dir.mkdir(parents=True)
            trace_file = trace_dir / "hostname.trace.json.gz"
            trace_file.write_bytes(b"")

            result = _find_trace_file(Path(tmpdir))
            assert result == trace_file

    def test_returns_none_when_not_found(self):
        """Test returns None when no trace file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _find_trace_file(Path(tmpdir))
            assert result is None

    def test_finds_first_trace_file(self):
        """Test finds trace file when multiple exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "plugins" / "profile" / "12345"
            trace_dir.mkdir(parents=True)
            trace_file = trace_dir / "host1.trace.json.gz"
            trace_file.write_bytes(b"")

            result = _find_trace_file(Path(tmpdir))
            assert result is not None
            assert result.suffix == ".gz"

    def test_handles_empty_directory(self):
        """Test handles empty directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _find_trace_file(Path(tmpdir))
            assert result is None


class TestParsePerfettoTrace:
    """Tests for the parse_perfetto_trace function."""

    def _create_trace_file(self, tmpdir: Path, events: list[dict]) -> Path:
        """Helper to create a mock trace file."""
        trace_dir = tmpdir / "plugins" / "profile" / "12345"
        trace_dir.mkdir(parents=True)
        trace_file = trace_dir / "hostname.trace.json.gz"

        trace_data = {"traceEvents": events}
        with gzip.open(trace_file, "wt", encoding="utf-8") as f:
            json.dump(trace_data, f)

        return trace_file

    def test_parses_performax_events(self):
        """Test parsing events with performax prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"ph": "X", "name": f"{PERFORMAX_PREFIX}my_function", "dur": 1000},
                {"ph": "X", "name": f"{PERFORMAX_PREFIX}other_func", "dur": 2000},
            ]
            self._create_trace_file(Path(tmpdir), events)

            result = parse_perfetto_trace(Path(tmpdir))

            assert len(result) == 2
            assert result[0].name == "my_function"
            assert result[0].duration_us == 1000
            assert result[1].name == "other_func"
            assert result[1].duration_us == 2000

    def test_ignores_non_performax_events(self):
        """Test that non-performax events are filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"ph": "X", "name": f"{PERFORMAX_PREFIX}tracked", "dur": 1000},
                {"ph": "X", "name": "untracked_function", "dur": 2000},
                {"ph": "X", "name": "jax/internal", "dur": 3000},
            ]
            self._create_trace_file(Path(tmpdir), events)

            result = parse_perfetto_trace(Path(tmpdir))

            assert len(result) == 1
            assert result[0].name == "tracked"

    def test_ignores_non_x_phase_events(self):
        """Test that non-complete (non-X phase) events are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"ph": "X", "name": f"{PERFORMAX_PREFIX}complete", "dur": 1000},
                {"ph": "B", "name": f"{PERFORMAX_PREFIX}begin", "dur": 2000},
                {"ph": "E", "name": f"{PERFORMAX_PREFIX}end", "dur": 3000},
                {"ph": "i", "name": f"{PERFORMAX_PREFIX}instant", "dur": 0},
            ]
            self._create_trace_file(Path(tmpdir), events)

            result = parse_perfetto_trace(Path(tmpdir))

            assert len(result) == 1
            assert result[0].name == "complete"

    def test_returns_empty_when_no_trace_file(self):
        """Test returns empty list when no trace file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = parse_perfetto_trace(Path(tmpdir))
            assert result == []

    def test_handles_empty_trace_events(self):
        """Test handles trace file with no events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_trace_file(Path(tmpdir), [])

            result = parse_perfetto_trace(Path(tmpdir))
            assert result == []

    def test_handles_missing_dur_field(self):
        """Test handles events without dur field (defaults to 0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"ph": "X", "name": f"{PERFORMAX_PREFIX}no_dur"},
            ]
            self._create_trace_file(Path(tmpdir), events)

            result = parse_perfetto_trace(Path(tmpdir))

            assert len(result) == 1
            assert result[0].duration_us == 0

    def test_handles_missing_name_field(self):
        """Test handles events without name field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"ph": "X", "dur": 1000},  # No name
                {"ph": "X", "name": f"{PERFORMAX_PREFIX}valid", "dur": 2000},
            ]
            self._create_trace_file(Path(tmpdir), events)

            result = parse_perfetto_trace(Path(tmpdir))

            assert len(result) == 1
            assert result[0].name == "valid"

    def test_multiple_calls_same_function(self):
        """Test parsing multiple calls to the same function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"ph": "X", "name": f"{PERFORMAX_PREFIX}repeated", "dur": 100},
                {"ph": "X", "name": f"{PERFORMAX_PREFIX}repeated", "dur": 200},
                {"ph": "X", "name": f"{PERFORMAX_PREFIX}repeated", "dur": 300},
            ]
            self._create_trace_file(Path(tmpdir), events)

            result = parse_perfetto_trace(Path(tmpdir))

            assert len(result) == 3
            assert all(e.name == "repeated" for e in result)
            assert [e.duration_us for e in result] == [100, 200, 300]

    def test_preserves_custom_names(self):
        """Test that custom names from @track(name='...') are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"ph": "X", "name": f"{PERFORMAX_PREFIX}custom_name", "dur": 1000},
                {"ph": "X", "name": f"{PERFORMAX_PREFIX}forward_pass", "dur": 2000},
            ]
            self._create_trace_file(Path(tmpdir), events)

            result = parse_perfetto_trace(Path(tmpdir))

            assert len(result) == 2
            assert result[0].name == "custom_name"
            assert result[1].name == "forward_pass"
