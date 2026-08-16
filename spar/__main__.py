"""Main entry point for the SPAR framework when run as a module."""

from __future__ import annotations

import sys

from .pipeline import run


def main() -> int:
    """Run the SPAR pipeline.

    Returns:
        int: Process exit code (0 on success). Exceptions raised by
        :func:`spar.pipeline.run` will propagate, resulting in a non-zero
        termination from the interpreter.
    """
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
