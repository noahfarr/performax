"""Result classes for profiling output."""

from dataclasses import dataclass


@dataclass
class FunctionStats:
    """Statistics for a single tracked function."""

    name: str
    total_duration_ms: float
    call_count: int

    @property
    def avg_duration_ms(self) -> float:
        """Average duration per call in milliseconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_duration_ms / self.call_count


class ProfileResult:
    """Container for profiling results with various output formats."""

    def __init__(self, stats: list[FunctionStats]):
        """Initialize with a list of function statistics.

        Args:
            stats: List of FunctionStats, sorted by total duration descending.
        """
        self._stats = sorted(stats, key=lambda s: s.total_duration_ms, reverse=True)

    @property
    def stats(self) -> list[FunctionStats]:
        """Get the list of function statistics."""
        return self._stats

    def to_dict(self) -> list[dict]:
        """Convert to a list of dictionaries.

        Returns:
            List of dicts with keys: name, total_ms, calls, avg_ms
        """
        return [
            {
                "name": s.name,
                "total_ms": s.total_duration_ms,
                "calls": s.call_count,
                "avg_ms": s.avg_duration_ms,
            }
            for s in self._stats
        ]

    def to_dataframe(self):
        """Convert to a pandas DataFrame.

        Returns:
            DataFrame with columns: Function, Total (ms), Calls, Avg (ms)

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        return pd.DataFrame(
            {
                "Function": [s.name for s in self._stats],
                "Total (ms)": [s.total_duration_ms for s in self._stats],
                "Calls": [s.call_count for s in self._stats],
                "Avg (ms)": [s.avg_duration_ms for s in self._stats],
            }
        )

    def __str__(self) -> str:
        """Pretty-print the profiling results as a table."""
        if not self._stats:
            return "No tracked functions were called."

        headers = ["Function", "Total (ms)", "Calls", "Avg (ms)"]

        rows = [
            [s.name, f"{s.total_duration_ms:.3f}", str(s.call_count), f"{s.avg_duration_ms:.3f}"]
            for s in self._stats
        ]

        col_widths = [
            max(len(headers[i]), max((len(row[i]) for row in rows), default=0))
            for i in range(len(headers))
        ]

        def format_row(items: list[str]) -> str:
            return " | ".join(item.ljust(col_widths[i]) for i, item in enumerate(items))

        separator = "-" * (sum(col_widths) + 3 * len(col_widths) - 1)

        lines = [format_row(headers), separator]
        lines.extend(format_row(row) for row in rows)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ProfileResult({len(self._stats)} functions)"
