"""Tests for profile result loggers."""

import json

import pytest

from performax.logger import (
    CSVLogger,
    FileLogger,
    JsonLogger,
    Logger,
    MarkdownLogger,
    PlainLogger,
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


class TestPlainLogger:
    """Tests for PlainLogger."""

    def test_log_with_stats(self, sample_result):
        """Test logging results with statistics."""
        logger = PlainLogger()
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


class TestJsonLogger:
    """Tests for JsonLogger."""

    def test_log_with_stats(self, sample_result):
        """Test logging results as JSON."""
        logger = JsonLogger()
        output = logger.log(sample_result)
        data = json.loads(output)

        assert "functions" in data
        assert len(data["functions"]) == 2
        assert data["functions"][0]["name"] == "slow_func"
        assert data["functions"][0]["total_ms"] == 500.0

    def test_log_empty_result(self, empty_result):
        """Test logging empty results as JSON."""
        logger = JsonLogger()
        output = logger.log(empty_result)
        data = json.loads(output)

        assert data["functions"] == []

    def test_compact_output(self, sample_result):
        """Test compact JSON output."""
        logger = JsonLogger(indent=None)
        output = logger.log(sample_result)

        assert "\n" not in output

    def test_with_metadata(self, sample_result):
        """Test JSON output with metadata."""
        logger = JsonLogger(include_metadata=True)
        output = logger.log(sample_result)
        data = json.loads(output)

        assert "metadata" in data
        assert "timestamp" in data["metadata"]
        assert "function_count" in data["metadata"]
        assert data["metadata"]["function_count"] == 2
        assert "total_time_ms" in data["metadata"]


class TestMarkdownLogger:
    """Tests for MarkdownLogger."""

    def test_log_with_stats(self, sample_result):
        """Test logging results as Markdown table."""
        logger = MarkdownLogger()
        output = logger.log(sample_result)

        assert "| Function | Total (ms) | Calls | Avg (ms) |" in output
        assert "|----------|------------|-------|----------|" in output
        assert "| slow_func |" in output
        assert "| fast_func |" in output

    def test_log_empty_result(self, empty_result):
        """Test logging empty results as Markdown."""
        logger = MarkdownLogger()
        output = logger.log(empty_result)

        assert output == "*No tracked functions were called.*"

    def test_valid_markdown_table(self, sample_result):
        """Test that output is valid Markdown table format."""
        logger = MarkdownLogger()
        output = logger.log(sample_result)
        lines = output.split("\n")

        # Header line
        assert lines[0].startswith("|") and lines[0].endswith("|")
        # Separator line
        assert all(c in "|-" for c in lines[1].replace(" ", ""))
        # Data lines
        for line in lines[2:]:
            assert line.startswith("|") and line.endswith("|")


class TestCSVLogger:
    """Tests for CSVLogger."""

    def test_log_with_stats(self, sample_result):
        """Test logging results as CSV."""
        logger = CSVLogger()
        output = logger.log(sample_result)
        lines = output.split("\n")

        assert lines[0] == "function,total_ms,calls,avg_ms"
        assert "slow_func,500.000,10,50.000" in lines[1]

    def test_log_empty_result(self, empty_result):
        """Test logging empty results as CSV."""
        logger = CSVLogger()
        output = logger.log(empty_result)

        assert output == "function,total_ms,calls,avg_ms"

    def test_without_header(self, sample_result):
        """Test CSV output without header."""
        logger = CSVLogger(include_header=False)
        output = logger.log(sample_result)

        assert "function,total_ms" not in output
        assert "slow_func" in output

    def test_custom_delimiter(self, sample_result):
        """Test CSV with custom delimiter."""
        logger = CSVLogger(delimiter="\t")
        output = logger.log(sample_result)

        assert "\t" in output
        assert "," not in output.replace("500.000", "")  # Exclude decimal

    def test_parseable_csv(self, sample_result):
        """Test that output is parseable as CSV."""
        import csv
        from io import StringIO

        logger = CSVLogger()
        output = logger.log(sample_result)

        reader = csv.DictReader(StringIO(output))
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["function"] == "slow_func"
        assert float(rows[0]["total_ms"]) == 500.0
