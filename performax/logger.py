"""Loggers for displaying profiling results in various formats."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .result import ProfileResult


class Logger(ABC):
    """Abstract base class for profile result loggers."""

    @abstractmethod
    def log(self, result: "ProfileResult") -> str:
        """Log the profile result to a string.

        Args:
            result: The ProfileResult to log.

        Returns:
            A string representation of the results.
        """
        pass


class PlainLogger(Logger):
    """Logs profile results as a plain text ASCII table."""

    def log(self, result: "ProfileResult") -> str:
        """Log results as a plain text table.

        Args:
            result: The ProfileResult to log.

        Returns:
            A plain text table string.
        """
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
    """Logs profile results as a rich formatted table with colors.

    Requires the 'rich' package to be installed.
    """

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
        """Initialize the rich logger with styling options.

        Args:
            title: Table title.
            show_header: Whether to show the header row.
            header_style: Style for the header row.
            function_style: Style for the function name column.
            total_style: Style for the total duration column.
            calls_style: Style for the call count column.
            avg_style: Style for the average duration column.
        """
        self.title = title
        self.show_header = show_header
        self.header_style = header_style
        self.function_style = function_style
        self.total_style = total_style
        self.calls_style = calls_style
        self.avg_style = avg_style

    def log(self, result: "ProfileResult") -> str:
        """Log results as a rich formatted table.

        Args:
            result: The ProfileResult to log.

        Returns:
            A rich formatted table string with ANSI colors.

        Raises:
            ImportError: If rich is not installed.
        """
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
    """Logs profile results in a log-friendly single-line format.

    Useful for logging to files or structured logging systems.
    """

    def __init__(self, *, separator: str = " | ", prefix: str = "[PROFILE]"):
        """Initialize the file logger.

        Args:
            separator: Separator between function entries.
            prefix: Prefix for the log line.
        """
        self.separator = separator
        self.prefix = prefix

    def log(self, result: "ProfileResult") -> str:
        """Log results as a log-friendly string.

        Args:
            result: The ProfileResult to log.

        Returns:
            A single-line log string.
        """
        if not result.stats:
            return f"{self.prefix} No tracked functions were called."

        entries = [
            f"{s.name}={s.total_duration_ms:.3f}ms({s.call_count}x)"
            for s in result.stats
        ]
        return f"{self.prefix} {self.separator.join(entries)}"


class JsonLogger(Logger):
    """Logs profile results as a JSON string.

    Useful for machine-readable output or API responses.
    """

    def __init__(self, *, indent: int | None = 2, include_metadata: bool = False):
        """Initialize the JSON logger.

        Args:
            indent: JSON indentation level. None for compact output.
            include_metadata: Whether to include metadata like timestamp.
        """
        self.indent = indent
        self.include_metadata = include_metadata

    def log(self, result: "ProfileResult") -> str:
        """Log results as a JSON string.

        Args:
            result: The ProfileResult to log.

        Returns:
            A JSON string.
        """
        import json
        from datetime import datetime, timezone

        data: dict = {"functions": result.to_dict()}

        if self.include_metadata:
            data["metadata"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "function_count": len(result.stats),
                "total_time_ms": sum(s.total_duration_ms for s in result.stats),
            }

        return json.dumps(data, indent=self.indent)


class MarkdownLogger(Logger):
    """Logs profile results as a Markdown table.

    Useful for documentation, GitHub issues, or notebooks.
    """

    def log(self, result: "ProfileResult") -> str:
        """Log results as a Markdown table.

        Args:
            result: The ProfileResult to log.

        Returns:
            A Markdown formatted table string.
        """
        if not result.stats:
            return "*No tracked functions were called.*"

        lines = [
            "| Function | Total (ms) | Calls | Avg (ms) |",
            "|----------|------------|-------|----------|",
        ]

        for s in result.stats:
            lines.append(
                f"| {s.name} | {s.total_duration_ms:.3f} | {s.call_count} | {s.avg_duration_ms:.3f} |"
            )

        return "\n".join(lines)


class CSVLogger(Logger):
    """Logs profile results as CSV.

    Useful for importing into spreadsheets or data analysis tools.
    """

    def __init__(self, *, include_header: bool = True, delimiter: str = ","):
        """Initialize the CSV logger.

        Args:
            include_header: Whether to include a header row.
            delimiter: Field delimiter character.
        """
        self.include_header = include_header
        self.delimiter = delimiter

    def log(self, result: "ProfileResult") -> str:
        """Log results as CSV.

        Args:
            result: The ProfileResult to log.

        Returns:
            A CSV formatted string.
        """
        lines = []

        if self.include_header:
            lines.append(
                self.delimiter.join(["function", "total_ms", "calls", "avg_ms"])
            )

        for s in result.stats:
            lines.append(
                self.delimiter.join(
                    [
                        s.name,
                        f"{s.total_duration_ms:.3f}",
                        str(s.call_count),
                        f"{s.avg_duration_ms:.3f}",
                    ]
                )
            )

        return "\n".join(lines)
