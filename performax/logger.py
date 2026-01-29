from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .result import ProfileResult


class Logger(ABC):
    @abstractmethod
    def log(self, result: "ProfileResult") -> str:
        pass


class ConsoleLogger(Logger):
    def log(self, result: "ProfileResult") -> str:
        if not result.stats:
            return "No tracked functions were called."

        headers = ["Function", "Total (ms)", "Calls", "Avg (ms)"]

        rows = [
            [
                s.name,
                f"{s.total_duration_ms:.3f}",
                str(s.call_count),
                f"{s.avg_duration_ms:.3f}",
            ]
            for s in result.stats
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


class RichLogger(Logger):
    def __init__(
        self,
        *,
        title: str = "Profile Results",
        show_header: bool = True,
        header_style: str = "bold cyan",
        function_style: str = "green",
        total_style: str = "yellow",
        calls_style: str = "blue",
        avg_style: str = "magenta",
    ):
        self.title = title
        self.show_header = show_header
        self.header_style = header_style
        self.function_style = function_style
        self.total_style = total_style
        self.calls_style = calls_style
        self.avg_style = avg_style

    def log(self, result: "ProfileResult") -> str:
        if not result.stats:
            return "No tracked functions were called."

        try:
            from io import StringIO

            from rich.console import Console
            from rich.table import Table
        except ImportError:
            raise ImportError(
                "rich is required for RichLogger. Install with: pip install rich"
            )

        table = Table(
            title=self.title,
            show_header=self.show_header,
            header_style=self.header_style,
        )
        table.add_column("Function", style=self.function_style)
        table.add_column("Total (ms)", justify="right", style=self.total_style)
        table.add_column("Calls", justify="right", style=self.calls_style)
        table.add_column("Avg (ms)", justify="right", style=self.avg_style)

        for s in result.stats:
            table.add_row(
                s.name,
                f"{s.total_duration_ms:.3f}",
                str(s.call_count),
                f"{s.avg_duration_ms:.3f}",
            )

        console = Console(file=StringIO(), force_terminal=True)
        console.print(table)
        return console.file.getvalue()


class FileLogger(Logger):
    def __init__(self, *, separator: str = " | ", prefix: str = "[PROFILE]"):
        self.separator = separator
        self.prefix = prefix

    def log(self, result: "ProfileResult") -> str:
        if not result.stats:
            return f"{self.prefix} No tracked functions were called."

        entries = [
            f"{s.name}={s.total_duration_ms:.3f}ms({s.call_count}x)"
            for s in result.stats
        ]
        return f"{self.prefix} {self.separator.join(entries)}"
