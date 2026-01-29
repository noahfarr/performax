from dataclasses import dataclass


@dataclass
class FunctionStats:
    name: str
    total_duration_ms: float
    call_count: int

    @property
    def avg_duration_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_duration_ms / self.call_count


class ProfileResult:
    def __init__(self, stats: list[FunctionStats]):
        self._stats = sorted(stats, key=lambda s: s.total_duration_ms, reverse=True)

    @property
    def stats(self) -> list[FunctionStats]:
        return self._stats

    def to_dict(self) -> list[dict]:
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
