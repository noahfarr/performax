"""Tests for ProfileResult and FunctionStats classes."""

import pytest

from performax.result import FunctionStats, ProfileResult


class TestFunctionStats:
    """Tests for the FunctionStats dataclass."""

    def test_basic_creation(self):
        """Test creating a FunctionStats instance."""
        stats = FunctionStats(name="test_func", total_duration_ms=100.0, call_count=10)
        assert stats.name == "test_func"
        assert stats.total_duration_ms == 100.0
        assert stats.call_count == 10

    def test_avg_duration_calculation(self):
        """Test average duration calculation."""
        stats = FunctionStats(name="test_func", total_duration_ms=100.0, call_count=10)
        assert stats.avg_duration_ms == 10.0

    def test_avg_duration_with_single_call(self):
        """Test average duration with single call."""
        stats = FunctionStats(name="test_func", total_duration_ms=50.0, call_count=1)
        assert stats.avg_duration_ms == 50.0

    def test_avg_duration_zero_calls(self):
        """Test average duration with zero calls returns 0."""
        stats = FunctionStats(name="test_func", total_duration_ms=0.0, call_count=0)
        assert stats.avg_duration_ms == 0.0

    def test_avg_duration_fractional(self):
        """Test average duration with fractional result."""
        stats = FunctionStats(name="test_func", total_duration_ms=100.0, call_count=3)
        assert abs(stats.avg_duration_ms - 33.333333) < 0.001


class TestProfileResult:
    """Tests for the ProfileResult class."""

    def test_empty_result(self):
        """Test ProfileResult with no stats."""
        result = ProfileResult([])
        assert result.stats == []
        assert str(result) == "No tracked functions were called."
        assert repr(result) == "ProfileResult(0 functions)"

    def test_single_function(self):
        """Test ProfileResult with single function."""
        stats = [FunctionStats(name="func1", total_duration_ms=100.0, call_count=5)]
        result = ProfileResult(stats)
        assert len(result.stats) == 1
        assert result.stats[0].name == "func1"

    def test_sorting_by_total_duration(self):
        """Test that stats are sorted by total duration descending."""
        stats = [
            FunctionStats(name="fast", total_duration_ms=10.0, call_count=1),
            FunctionStats(name="slow", total_duration_ms=100.0, call_count=1),
            FunctionStats(name="medium", total_duration_ms=50.0, call_count=1),
        ]
        result = ProfileResult(stats)
        assert result.stats[0].name == "slow"
        assert result.stats[1].name == "medium"
        assert result.stats[2].name == "fast"

    def test_to_dict(self):
        """Test converting ProfileResult to dict."""
        stats = [FunctionStats(name="func1", total_duration_ms=100.0, call_count=10)]
        result = ProfileResult(stats)
        dict_result = result.to_dict()

        assert len(dict_result) == 1
        assert dict_result[0]["name"] == "func1"
        assert dict_result[0]["total_ms"] == 100.0
        assert dict_result[0]["calls"] == 10
        assert dict_result[0]["avg_ms"] == 10.0

    def test_to_dict_multiple_functions(self):
        """Test to_dict with multiple functions preserves order."""
        stats = [
            FunctionStats(name="fast", total_duration_ms=10.0, call_count=1),
            FunctionStats(name="slow", total_duration_ms=100.0, call_count=1),
        ]
        result = ProfileResult(stats)
        dict_result = result.to_dict()

        assert dict_result[0]["name"] == "slow"
        assert dict_result[1]["name"] == "fast"

    def test_to_dict_empty(self):
        """Test to_dict with empty stats."""
        result = ProfileResult([])
        assert result.to_dict() == []

    def test_str_formatting(self):
        """Test string representation formatting."""
        stats = [
            FunctionStats(name="function_a", total_duration_ms=123.456, call_count=10),
            FunctionStats(name="func_b", total_duration_ms=45.678, call_count=5),
        ]
        result = ProfileResult(stats)
        output = str(result)

        assert "Function" in output
        assert "Total (ms)" in output
        assert "Calls" in output
        assert "Avg (ms)" in output
        assert "function_a" in output
        assert "func_b" in output
        assert "123.456" in output
        assert "45.678" in output

    def test_str_table_alignment(self):
        """Test that table columns are properly aligned."""
        stats = [
            FunctionStats(name="a", total_duration_ms=1.0, call_count=1),
            FunctionStats(name="very_long_name", total_duration_ms=1000.0, call_count=100),
        ]
        result = ProfileResult(stats)
        output = str(result)
        lines = output.split("\n")

        # All lines should have the same pattern of separators
        assert all("|" in line for line in lines if line.strip() and "-" not in line)

    def test_repr(self):
        """Test repr shows function count."""
        stats = [
            FunctionStats(name="func1", total_duration_ms=100.0, call_count=1),
            FunctionStats(name="func2", total_duration_ms=50.0, call_count=1),
            FunctionStats(name="func3", total_duration_ms=25.0, call_count=1),
        ]
        result = ProfileResult(stats)
        assert repr(result) == "ProfileResult(3 functions)"

    def test_stats_property_returns_list(self):
        """Test that stats property returns the list."""
        stats = [FunctionStats(name="func1", total_duration_ms=100.0, call_count=1)]
        result = ProfileResult(stats)
        assert isinstance(result.stats, list)
        assert len(result.stats) == 1


class TestProfileResultToDataframe:
    """Tests for ProfileResult.to_dataframe() method."""

    def test_to_dataframe_without_pandas(self, monkeypatch):
        """Test to_dataframe raises ImportError when pandas not available."""
        import sys

        # Remove pandas from sys.modules if present
        monkeypatch.setitem(sys.modules, "pandas", None)

        stats = [FunctionStats(name="func1", total_duration_ms=100.0, call_count=1)]
        result = ProfileResult(stats)

        with pytest.raises(ImportError, match="pandas is required"):
            result.to_dataframe()

    def test_to_dataframe_with_pandas(self):
        """Test to_dataframe returns proper DataFrame when pandas available."""
        pytest.importorskip("pandas")
        import pandas as pd

        stats = [
            FunctionStats(name="func1", total_duration_ms=100.0, call_count=10),
            FunctionStats(name="func2", total_duration_ms=50.0, call_count=5),
        ]
        result = ProfileResult(stats)
        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Function", "Total (ms)", "Calls", "Avg (ms)"]
        assert len(df) == 2
        assert df.iloc[0]["Function"] == "func1"
        assert df.iloc[0]["Total (ms)"] == 100.0
        assert df.iloc[0]["Calls"] == 10
        assert df.iloc[0]["Avg (ms)"] == 10.0

    def test_to_dataframe_empty(self):
        """Test to_dataframe with empty stats."""
        pytest.importorskip("pandas")
        import pandas as pd

        result = ProfileResult([])
        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
