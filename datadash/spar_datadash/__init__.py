"""React-based SPAR data dashboard package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from .react_api import run_react_dashboard as _run_react_dashboard

__all__: list[str] = ["run_react_dashboard"]
__version__ = "0.1.0"
__author__ = "Misagh Soltani"


def run_react_dashboard(
    host: str = "127.0.0.1", port: int = 8060, debug: bool = False, frontend_dist: Path | None = None
) -> None:
    """Wrapper around the React dashboard API/static host."""
    _run_react_dashboard(host=host, port=port, debug=debug, frontend_dist=frontend_dist)
