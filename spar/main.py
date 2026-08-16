"""Main entry point for the SPAR CLI."""

from __future__ import annotations

from .pipeline import run


def main() -> None:
    """Run the SPAR pipeline."""
    run()


if __name__ == "__main__":
    main()
