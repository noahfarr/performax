from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parser import TraceEvent


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

    @classmethod
    def from_events(cls, events: "list[TraceEvent]") -> "ProfileResult":
        stats_by_name: dict[str, dict] = defaultdict(
            lambda: {"total_us": 0.0, "count": 0}
        )
        for event in events:
            stats_by_name[event.name]["total_us"] += event.duration_us
            stats_by_name[event.name]["count"] += 1

        return cls(
            [
                FunctionStats(
                    name=name,
                    total_duration_ms=data["total_us"] / 1000.0,
                    call_count=data["count"],
                )
                for name, data in stats_by_name.items()
            ]
        )

    @property
    def stats(self) -> list[FunctionStats]:
        return self._stats

    def __bool__(self) -> bool:
        return bool(self._stats)

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
            [
                s.name,
                f"{s.total_duration_ms:.3f}",
                str(s.call_count),
                f"{s.avg_duration_ms:.3f}",
            ]
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


class Profile:
    def __init__(self, host: ProfileResult, device: ProfileResult):
        self.host = host
        self.device = device

    def to_dict(self) -> dict:
        return {"host": self.host.to_dict(), "device": self.device.to_dict()}

    def __str__(self) -> str:
        if not self.host and not self.device:
            return "No tracked functions were called."

        parts = []
        if self.host:
            parts.append("== Host (dispatch) ==\n" + str(self.host))
        if self.device:
            parts.append("== Device (kernels) ==\n" + str(self.device))
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"Profile(host={len(self.host.stats)} functions, "
            f"device={len(self.device.stats)} functions)"
        )
