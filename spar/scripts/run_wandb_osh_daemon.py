"""Run a head-node wandb-osh daemon for syncing offline W&B runs."""

from __future__ import annotations

import argparse
from logging import getLogger
from pathlib import Path
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace
    from logging import Logger

logger: Logger = getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> Namespace:
    """Parse CLI arguments for head-node wandb-osh daemon execution."""
    parser = argparse.ArgumentParser(description="Run wandb-osh daemon for offline W&B sync")
    parser.add_argument("--command-dir", type=Path, required=True, help="Shared command directory for wandb-osh")
    parser.add_argument("--wait", type=int, default=10, help="Polling interval in seconds")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds passed to wandb-osh")
    parser.add_argument("--sync-all", action="store_true", help="Pass --sync-all to underlying wandb sync invocations")
    parser.add_argument("--verbose", action="store_true", help="Print full command before execution")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the daemon process and return the `wandb-osh` exit code."""
    args = parse_args(argv)

    if shutil.which("wandb-osh") is None:
        print("`wandb-osh` CLI not found. Install with `pixi add --pypi wandb-osh`.", file=sys.stderr)
        return 1

    args.command_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "wandb-osh",
        "--command-dir",
        str(args.command_dir),
        "--wait",
        str(args.wait),
        "--timeout",
        str(args.timeout),
    ]

    passthrough: list[str] = []
    if args.sync_all:
        passthrough.append("--sync-all")

    if passthrough:
        cmd.append("--")
        cmd.extend(passthrough)

    if args.verbose:
        print("Executing:", " ".join(cmd))

    try:
        return subprocess.run(cmd, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
