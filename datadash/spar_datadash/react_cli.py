"""CLI for serving the React-based SPAR data dashboard."""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
from typing import TYPE_CHECKING

from .react_api import run_react_dashboard

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace

DEFAULT_BIND_HOST: str = str(ipaddress.IPv4Address(0))
DEFAULT_PORT: int = 8060


def build_parser() -> ArgumentParser:
    """Build the command-line parser for the React dashboard server.

    Returns:
        argparse.ArgumentParser: Configured parser for dashboard CLI options.
    """
    parser = argparse.ArgumentParser(description="Run the React SPAR data dashboard API and static host.")
    parser.add_argument(
        "--host", default=DEFAULT_BIND_HOST, help=f"Host interface to bind (default: {DEFAULT_BIND_HOST})"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument(
        "--frontend-dist", type=Path, default=None, help="Optional path to frontend dist folder containing index.html"
    )
    return parser


def main() -> None:
    """Run the React dashboard CLI entrypoint."""
    parser: ArgumentParser = build_parser()
    args: Namespace = parser.parse_args()
    run_react_dashboard(host=args.host, port=args.port, debug=args.debug, frontend_dist=args.frontend_dist)


if __name__ == "__main__":
    main()
