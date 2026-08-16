from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from rich.console import Console
from rich.logging import RichHandler
from rich.protocol import is_renderable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from datetime import datetime
    from logging import LogRecord
    from types import ModuleType
    from typing import IO

    from rich.highlighter import Highlighter
    from rich.text import Text
    from typing_extensions import Unpack


class RichHandlerKwargs(TypedDict, total=False):
    """Keyword arguments forwarded to RichHandler."""

    level: int | str
    show_time: bool
    omit_repeated_times: bool
    show_level: bool
    show_path: bool
    enable_link_path: bool
    highlighter: Highlighter | None
    markup: bool
    rich_tracebacks: bool
    tracebacks_width: int | None
    tracebacks_code_width: int | None
    tracebacks_extra_lines: int
    tracebacks_theme: str | None
    tracebacks_word_wrap: bool
    tracebacks_show_locals: bool
    tracebacks_suppress: Iterable[str | ModuleType]
    tracebacks_max_frames: int
    locals_max_length: int
    locals_max_string: int
    log_time_format: str | Callable[[datetime], Text]
    keywords: list[str] | None


terminal_console: Console = Console(soft_wrap=True, width=200)


class CustomRichHandler(RichHandler):
    """Custom Rich logging handler that supports file output and direct Rich renderable objects.

    This handler extends the Rich RichHandler to provide additional functionality:
    - Optional file output for logs
    - Direct handling of Rich renderable objects without additional formatting
    - Automatic directory creation for log files
    - Proper resource cleanup

    Args:
        filename (str | Path | None): Optional path to write logs to a file. If None,
            logs to terminal console.
        mode (str): File opening mode when filename is provided. Defaults to 'w'.
        encoding (str): File encoding when filename is provided. Defaults to 'utf-8'.
        **kwargs: Additional keyword arguments passed to the parent RichHandler.
    """

    def __init__(
        self,
        filename: str | Path | None = None,
        mode: str = "w",
        encoding: str = "utf-8",
        width: int | None = None,
        **kwargs: Unpack[RichHandlerKwargs],
    ) -> None:
        self.file_handle: IO[str] | None = None
        self._exit_stack: contextlib.ExitStack | None = None
        console: Console | None = None
        if filename is not None:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._exit_stack = contextlib.ExitStack()
            self.file_handle = self._exit_stack.enter_context(path.open(mode, encoding=encoding))
            console = Console(file=self.file_handle, soft_wrap=True, width=width)
        else:
            console = terminal_console

        super().__init__(console=console, **kwargs)

    def close(self) -> None:
        """Close the file handle if it exists."""
        if hasattr(self, "_exit_stack") and self._exit_stack is not None:
            self._exit_stack.close()
            self._exit_stack = None
            self.file_handle = None
        elif hasattr(self, "file_handle") and self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
        super().close()

    def __del__(self) -> None:
        """Close the file handle during garbage collection."""
        with contextlib.suppress(Exception):
            self.close()

    def emit(self, record: LogRecord) -> None:
        """Process and emit a log record.

        Rich-renderable values go directly to the console. Other records use
        :class:`RichHandler` formatting.

        Args:
            record: Log record to emit.
        """
        try:
            # Inspect the original message before Rich applies formatting.
            msg = record.msg
            if is_renderable(msg):
                # Log with timestamp/level as console.log does
                self.console.print(msg)
            else:
                # Fallback to normal RichHandler behavior
                super().emit(record)
        except Exception:
            self.handleError(record)
