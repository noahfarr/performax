"""Tests for profile result loggers."""

import json

import pytest

from performax.logger import (
    FileLogger,
    Logger,
    ConsoleLogger,
    RichLogger,
)
from performax.result import FunctionStats, ProfileResult


@pytest.fixture
def sample_result():
    """Create a sample ProfileResult for testing."""
    stats = [
        FunctionStats(name="slow_func", total_duration_ms=500.0, call_count=10),
        FunctionStats(name="fast_func", total_duration_ms=50.0, call_count=100),
    ]
    return ProfileResult(stats)


@pytest.fixture
def empty_result():
    """Create an empty ProfileResult for testing."""
    return ProfileResult([])


class TestLoggerBase:
    """Tests for the Logger abstract base class."""

    def test_logger_is_abstract(self):
        """Test that Logger cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Logger()

    def test_logger_requires_log_method(self):
        """Test that subclasses must implement log method."""

        class IncompleteLogger(Logger):
            pass

        with pytest.raises(TypeError):
            IncompleteLogger()


class TestConsoleLogger:
    """Tests for ConsoleLogger."""

    def test_log_with_stats(self, sample_result):
        """Test logging results with statistics."""
        logger = ConsoleLogger()
        output = logger.log(sample_result)

        assert "Function" in output
        assert "Total (ms)" in output
        assert "Calls" in output
        assert "Avg (ms)" in output
        assert "slow_func" in output
        assert "fast_func" in output
        assert "500.000" in output

    def test_log_empty_result(self, empty_result):
        """Test logging empty results."""
        logger = PlainLogger()
        output = logger.log(empty_result)

        assert output == "No tracked functions were called."

    def test_log_table_alignment(self, sample_result):
        """Test that table columns are aligned."""
        logger = PlainLogger()
        output = logger.log(sample_result)
        lines = output.split("\n")

        # All data lines should have separators
        assert all("|" in line for line in lines if "-" not in line)


class TestRichLogger:
    """Tests for RichLogger."""

    def test_log_with_stats(self, sample_result):
        """Test logging results with rich formatting."""
        pytest.importorskip("rich")
        logger = RichLogger()
        output = logger.log(sample_result)

        assert "slow_func" in output
        assert "fast_func" in output
        assert "Profile Results" in output

    def test_log_empty_result(self, empty_result):
        """Test logging empty results."""
        pytest.importorskip("rich")
        logger = RichLogger()
        output = logger.log(empty_result)

        assert output == "No tracked functions were called."

    def test_custom_title(self, sample_result):
        """Test custom title option."""
        pytest.importorskip("rich")
        logger = RichLogger(title="Custom Title")
        output = logger.log(sample_result)

        assert "Custom Title" in output

    def test_raises_without_rich(self, sample_result, monkeypatch):
        """Test that ImportError is raised when rich is not available."""
        import sys

        monkeypatch.setitem(sys.modules, "rich", None)
        monkeypatch.setitem(sys.modules, "rich.console", None)
        monkeypatch.setitem(sys.modules, "rich.table", None)

        logger = RichLogger()
        with pytest.raises(ImportError, match="rich is required"):
            logger.log(sample_result)


class TestFileLogger:
    """Tests for FileLogger."""

    def test_log_with_stats(self, sample_result):
        """Test logging results in file-friendly format."""
        logger = FileLogger()
        output = logger.log(sample_result)

        assert output.startswith("[PROFILE]")
        assert "slow_func=" in output
        assert "fast_func=" in output
        assert "500.000ms" in output
        assert "(10x)" in output

    def test_log_empty_result(self, empty_result):
        """Test logging empty results."""
        logger = FileLogger()
        output = logger.log(empty_result)

        assert output == "[PROFILE] No tracked functions were called."

    def test_custom_prefix(self, sample_result):
        """Test custom prefix option."""
        logger = FileLogger(prefix="[PERF]")
        output = logger.log(sample_result)

        assert output.startswith("[PERF]")

    def test_custom_separator(self, sample_result):
        """Test custom separator option."""
        logger = FileLogger(separator=", ")
        output = logger.log(sample_result)

        assert ", " in output

    def test_single_line_output(self, sample_result):
        """Test that output is a single line."""
        logger = FileLogger()
        output = logger.log(sample_result)

        assert "\n" not in output

