"""Greedy Best-First Search stage entry point."""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from .qstar import run_search as run_search_shared

if TYPE_CHECKING:
    from logging import Logger

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.utils.config_utils.config_schema import SearchGBFSSPARConfig

    from .qstar import QStarResults


logger: Logger = getLogger(__name__)

__all__: list[str] = ["run_search"]


def run_search(env: ABCEnvironment[ABCState], cfg: SearchGBFSSPARConfig) -> QStarResults | None:
    """Run the ``search_gbfs`` pipeline stage.

    Args:
        env: The environment instance.
        cfg: The validated ``search_gbfs`` stage configuration.

    Returns:
        Aggregated search results, or ``None`` when no result is produced.
    """
    logger.info("Starting GBFS search")
    return run_search_shared(env, cfg, algorithm="gbfs")
