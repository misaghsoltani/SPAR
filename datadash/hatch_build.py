"""Build the React frontend for Hatch distributions.

The dashboard serves a pre-built single-page app from ``spar_datadash/_frontend``.
This hook compiles that bundle from ``spar-datadash-react`` when Node.js and pnpm
are available. It skips compilation when the toolchain is missing or the bundle
already exists.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

if TYPE_CHECKING:
    from typing import Any, ClassVar


class FrontendBuildHook(BuildHookInterface):
    """Compile the React frontend bundle before a build target is assembled."""

    PLUGIN_NAME: ClassVar[str] = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Build the frontend bundle when a toolchain is available.

        Args:
            version: The version being built (unused).
            build_data: Mutable build metadata shared with the builder (unused).
        """
        del version, build_data

        root: Path = Path(self.root)
        frontend_source: Path = root / "spar-datadash-react"
        bundle_dir: Path = root / "spar_datadash" / "_frontend"
        bundle_index: Path = bundle_dir / "index.html"

        # Use the bundle already present in the source tree or wheel.
        if bundle_index.is_file():
            return

        # Without frontend sources, use the package data as provided.
        if not (frontend_source / "package.json").is_file():
            return

        pnpm: str | None = shutil.which("pnpm")
        if pnpm is None:
            self.app.display_warning(
                "spar-datadash: pnpm not found. Skipping frontend build. "
                "The wheel will not contain a UI bundle unless one is present."
            )
            return

        env: dict[str, str] = dict(os.environ)
        try:
            subprocess.run([pnpm, "install", "--frozen-lockfile"], cwd=frontend_source, check=True, env=env)
            subprocess.run([pnpm, "build"], cwd=frontend_source, check=True, env=env)
        except (OSError, subprocess.CalledProcessError) as exc:
            self.app.display_warning(f"spar-datadash: frontend build failed ({exc}). Shipping without a UI bundle.")
