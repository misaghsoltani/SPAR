"""Cleanup generated artifacts for the React SPAR data dashboard."""

from __future__ import annotations

from pathlib import Path
import shutil


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """Remove generated artifacts for the React SPAR data dashboard."""
    root: Path = _repo_root()
    webapp_root: Path = root / "datadash"
    frontend_root: Path = webapp_root / "spar-datadash-react"

    for path in (
        frontend_root / "node_modules",
        frontend_root / "dist",
        frontend_root / ".parcel-cache",
        webapp_root / "spar_datadash" / "_frontend",
    ):
        shutil.rmtree(path, ignore_errors=True)

    (frontend_root / "bundle.html").unlink(missing_ok=True)

    for path in webapp_root.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    for suffix in ("*.pyc", "*.pyo"):
        for path in webapp_root.rglob(suffix):
            if path.is_file():
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
