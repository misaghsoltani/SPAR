"""Rich-based logging utilities for console output."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import TYPE_CHECKING, Literal, TypedDict

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

if TYPE_CHECKING:
    from logging import Logger
    from types import TracebackType
    from typing import TypeAlias, Unpack


_ExcInfo: TypeAlias = (
    bool | BaseException | tuple[type[BaseException], BaseException, "TracebackType | None"] | tuple[None, None, None]
)
_LogLevel: TypeAlias = Literal["debug", "info", "warning", "error"]


class _ForwardLogKwargs(TypedDict, total=False):
    exc_info: _ExcInfo | None
    stack_info: bool
    stacklevel: int


class _LogKwargs(_ForwardLogKwargs, total=False):
    extra: Mapping[str, object]


class RichLogger:
    """Rich-based logger for the SPAR Data Dashboard."""

    def __init__(self, name: str = __name__) -> None:
        """Initialize the RichLogger.

        Args:
            name: Logger name, typically __name__ from the calling module.
        """
        self.logger: Logger = logging.getLogger(name)
        self.console: Console = Console()

        # Set up Rich handler if not already configured
        if not any(isinstance(handler, RichHandler) for handler in self.logger.handlers):
            rich_handler = RichHandler(
                console=self.console, show_path=False, rich_tracebacks=True, tracebacks_show_locals=False
            )
            rich_handler.setFormatter(logging.Formatter("{message}", style="{"))
            self.logger.addHandler(rich_handler)
            self.logger.setLevel(logging.DEBUG)

    def _log_with_markup(
        self,
        level: _LogLevel,
        prefix: str,
        message: str,
        *args: object,
        exception: bool = False,
        **kwargs: Unpack[_LogKwargs],
    ) -> None:
        """Log while preserving any caller-supplied ``extra`` metadata and Rich markup."""
        extra = kwargs.pop("extra", None)
        merged_extra: dict[str, object] = dict(extra) if isinstance(extra, Mapping) else {}
        merged_extra.setdefault("markup", True)
        forwarded: _ForwardLogKwargs = {}
        if "exc_info" in kwargs:
            forwarded["exc_info"] = kwargs["exc_info"]
        if "stack_info" in kwargs:
            forwarded["stack_info"] = kwargs["stack_info"]
        if "stacklevel" in kwargs:
            forwarded["stacklevel"] = kwargs["stacklevel"]
        formatted: str = f"{prefix} {message}"
        if exception:
            self.logger.error(formatted, *args, extra=merged_extra, **forwarded)
        elif level == "debug":
            self.logger.debug(formatted, *args, extra=merged_extra, **forwarded)
        elif level == "info":
            self.logger.info(formatted, *args, extra=merged_extra, **forwarded)
        elif level == "warning":
            self.logger.warning(formatted, *args, extra=merged_extra, **forwarded)
        else:
            self.logger.error(formatted, *args, extra=merged_extra, **forwarded)

    def info(self, message: str, *args: object, **kwargs: Unpack[_LogKwargs]) -> None:
        """Log an info message with styling."""
        self._log_with_markup("info", "[blue]i[/blue]", message, *args, **kwargs)

    def debug(self, message: str, *args: object, **kwargs: Unpack[_LogKwargs]) -> None:
        """Log a debug message with styling."""
        self._log_with_markup("debug", "[dim]🔍[/dim]", message, *args, **kwargs)

    def warning(self, message: str, *args: object, **kwargs: Unpack[_LogKwargs]) -> None:
        """Log a warning message with styling."""
        self._log_with_markup("warning", "[yellow]⚠[/yellow]", message, *args, **kwargs)

    def error(self, message: str, *args: object, **kwargs: Unpack[_LogKwargs]) -> None:
        """Log an error message with styling."""
        self._log_with_markup("error", "[red]✗[/red]", message, *args, **kwargs)

    def exception(self, message: str, *args: object, **kwargs: Unpack[_LogKwargs]) -> None:
        """Log an exception with styling."""
        self._log_with_markup("error", "[red bold]💥[/red bold]", message, *args, exception=True, **kwargs)

    def success(self, message: str, *args: object, **kwargs: Unpack[_LogKwargs]) -> None:
        """Log a success message with styling."""
        self._log_with_markup("info", "[green]✓[/green]", message, *args, **kwargs)

    def panel(self, content: str, title: str | None = None, border_style: str = "blue") -> None:
        """Display content in a panel."""
        self.console.print(Panel(content, title=title, border_style=border_style, padding=(1, 2), expand=False))

    def cache_hit(self, key: str) -> None:
        """Log a cache hit with appropriate styling."""
        self.debug(f"Cache [green]HIT[/green]: [cyan]{key}[/cyan]")

    def cache_miss(self, key: str) -> None:
        """Log a cache miss with appropriate styling."""
        self.debug(f"Cache [yellow]MISS[/yellow]: [cyan]{key}[/cyan]")

    def cache_init(self, cache_type: str) -> None:
        """Log cache initialization with appropriate styling."""
        self.success(f"Cache initialized: [cyan]{cache_type}[/cyan]")

    def render_start(self, variations: str) -> None:
        """Log render operation start."""
        self.debug(f"Rendering with variations: [dim]{variations}[/dim]")

    def render_error(self, message: str, retry: bool = False) -> None:
        """Log render error with context."""
        prefix = "Retrying render" if retry else "Render failed"
        self.error(f"{prefix}: {message}")

    def effect_not_found(self, name: str) -> None:
        """Log when an effect is not found."""
        self.warning(f"Effect '[cyan]{name}[/cyan]' not found")

    def effect_creation_failed(self, name: str, error: str) -> None:
        """Log when effect creation fails."""
        self.error(f"Failed to create effect '[cyan]{name}[/cyan]': {error}")

    def startup_banner(self, host: str, port: int, debug: bool) -> None:
        """Display startup banner."""
        title: str = "[bold blue]SPAR Data Dashboard[/bold blue]"
        url_line: str = f"[green]URL:[/green] http://{host}:{port}/"
        debug_line: str = f"[yellow]Debug Mode:[/yellow] {'[green]Enabled[/green]' if debug else '[red]Disabled[/red]'}"

        content_lines: list[str] = [title, "", url_line, debug_line, "", "[dim]Press Ctrl+C to quit[/dim]"]

        self.console.print(
            Panel("\n".join(content_lines), title="Dashboard ready", border_style="blue", padding=(1, 2), expand=False)
        )


# Global instances for each module
_loggers: dict[str, RichLogger] = {}


def get_rich_logger(name: str = __name__) -> RichLogger:
    """Get or create a RichLogger instance for the given name.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        RichLogger instance.
    """
    if name not in _loggers:
        _loggers[name] = RichLogger(name)
    return _loggers[name]
