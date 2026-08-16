"""Optuna storage, sampler, and pruner builders for SPAR."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, PercentilePruner, SuccessiveHalvingPruner
from optuna.samplers import CmaEsSampler, GridSampler, NSGAIISampler, RandomSampler, TPESampler
from optuna.storages import RDBStorage
from optuna.storages.journal import JournalFileBackend, JournalStorage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from optuna.pruners import BasePruner
    from optuna.samplers import BaseSampler
    from optuna.storages import BaseStorage
    from optuna.trial import FrozenTrial

    from spar.utils.config_utils.config_schema import PrunerConfig, SamplerConfig, StorageConfig

    from .types import SampledValue


def constraint_values(trial: FrozenTrial) -> Sequence[float]:
    """Return stored constraint values for Optuna samplers that support them."""
    values: Sequence[float] = trial.user_attrs.get("constraints_vector", [])
    if isinstance(values, list):
        return list(values)

    return []


def build_storage(storage_cfg: StorageConfig, *, study_dir: Path, study_name: str) -> BaseStorage | str:
    """Build the configured Optuna storage backend."""
    if storage_cfg.kind == "journal":
        journal_path: Path = Path(storage_cfg.path) if storage_cfg.path else study_dir / f"{study_name}.journal"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        return JournalStorage(JournalFileBackend(str(journal_path)))

    if storage_cfg.kind == "sqlite":
        sqlite_path: Path = Path(storage_cfg.path) if storage_cfg.path else study_dir / f"{study_name}.sqlite3"
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{sqlite_path}"

    if storage_cfg.kind == "rdb":
        if storage_cfg.url is None:
            raise ValueError("storage.url must be provided when optuna.storage.kind='rdb'")

        return RDBStorage(
            url=storage_cfg.url,
            heartbeat_interval=storage_cfg.heartbeat_interval,
            grace_period=storage_cfg.grace_period,
            engine_kwargs=dict(storage_cfg.engine_kwargs),
        )

    raise ValueError(f"Unsupported Optuna storage kind: {storage_cfg.kind}")


def build_sampler(
    sampler_cfg: SamplerConfig,
    *,
    constraints_enabled: bool,
    grid_search_space: Mapping[str, Sequence[SampledValue]] | None = None,
) -> BaseSampler:
    """Build the configured Optuna sampler.

    Args:
        sampler_cfg: Sampler configuration.
        constraints_enabled: Whether trial constraints are configured.
        grid_search_space: Finite parameter values for grid sampling.

    Returns:
        The configured Optuna sampler.

    Raises:
        ValueError: If grid sampling is selected without a finite search space.
    """
    constraints_func: Callable[..., Sequence[float]] | None = constraint_values if constraints_enabled else None

    if sampler_cfg.kind == "tpe":
        return TPESampler(
            seed=sampler_cfg.seed,
            n_startup_trials=sampler_cfg.n_startup_trials,
            multivariate=sampler_cfg.multivariate,
            group=sampler_cfg.group,
            constant_liar=sampler_cfg.constant_liar,
            constraints_func=constraints_func,
        )
    if sampler_cfg.kind == "random":
        return RandomSampler(seed=sampler_cfg.seed)

    if sampler_cfg.kind == "cmaes":
        return CmaEsSampler(seed=sampler_cfg.seed)

    if sampler_cfg.kind == "nsga2":
        return NSGAIISampler(seed=sampler_cfg.seed, constraints_func=constraints_func)

    if sampler_cfg.kind == "grid":
        if not grid_search_space:
            raise ValueError("Grid sampler requires at least one finite non-fixed parameter")
        return GridSampler(grid_search_space, seed=sampler_cfg.seed)

    raise ValueError(f"Unsupported Optuna sampler kind: {sampler_cfg.kind}")


def build_pruner(pruner_cfg: PrunerConfig) -> BasePruner:
    """Build the configured Optuna pruner."""
    if pruner_cfg.kind == "none":
        return NopPruner()

    if pruner_cfg.kind == "median":
        return MedianPruner(
            n_startup_trials=pruner_cfg.n_startup_trials,
            n_warmup_steps=pruner_cfg.n_warmup_steps,
            interval_steps=pruner_cfg.interval_steps,
            n_min_trials=pruner_cfg.n_min_trials,
        )
    if pruner_cfg.kind == "successive_halving":
        return SuccessiveHalvingPruner(
            min_resource=pruner_cfg.min_resource,
            reduction_factor=pruner_cfg.reduction_factor,
            min_early_stopping_rate=pruner_cfg.n_warmup_steps,
        )
    if pruner_cfg.kind == "hyperband":
        if not isinstance(pruner_cfg.min_resource, int):
            raise TypeError("Hyperband pruner requires an integer min_resource")
        return HyperbandPruner(min_resource=pruner_cfg.min_resource, reduction_factor=pruner_cfg.reduction_factor)

    if pruner_cfg.kind == "percentile":
        return PercentilePruner(
            percentile=pruner_cfg.percentile,
            n_startup_trials=pruner_cfg.n_startup_trials,
            n_warmup_steps=pruner_cfg.n_warmup_steps,
            interval_steps=pruner_cfg.interval_steps,
            n_min_trials=pruner_cfg.n_min_trials,
        )
    raise ValueError(f"Unsupported Optuna pruner kind: {pruner_cfg.kind}")
