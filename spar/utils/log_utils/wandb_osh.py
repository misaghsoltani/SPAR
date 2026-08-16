"""wandb-osh utilities for offline HPC sync workflows."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from logging import getLogger
from pathlib import Path
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger
    from types import ModuleType

    from wandb_osh.hooks import TriggerWandbSyncHook

    from spar.utils.config_utils.config_schema import WandbConfig, WandbOshConfig


logger: Logger = getLogger(__name__)


@dataclass(slots=True)
class WandbOshSyncTrigger:
    """Compute-node trigger for requesting offline run synchronization."""

    enabled: bool
    command_dir: Path
    trigger_every_seconds: int
    timeout_seconds: int
    trigger_on_phase_end: bool
    _trigger_fn: Callable[[], None] | None = None
    _last_trigger_ts: float | None = None

    def maybe_trigger(self, *, force: bool = False) -> bool:
        """Trigger sync request when cadence/force conditions are met."""
        if not self.enabled or self._trigger_fn is None:
            return False

        now: float = time.monotonic()
        should_trigger: bool = force or self.trigger_every_seconds <= 0
        # The first opportunity always syncs (no prior trigger recorded). The
        # sentinel must be explicit: `time.monotonic()` has a platform-defined
        # origin, so comparing against 0.0 is not a reliable "never" marker.
        if not should_trigger and (
            self._last_trigger_ts is None or now - self._last_trigger_ts >= self.trigger_every_seconds
        ):
            should_trigger = True

        if not should_trigger:
            return False

        try:
            self._trigger_fn()
        except Exception:
            logger.exception("Failed to trigger wandb-osh sync request")
        else:
            self._last_trigger_ts = now
            return True
        return False


_DEF_TRIGGER_SECONDS: int = 300
_DEF_TIMEOUT_SECONDS: int = 300


def build_wandb_osh_trigger(wb_cfg: WandbConfig) -> WandbOshSyncTrigger | None:
    """Build optional wandb-osh sync trigger from W&B config."""
    osh_cfg: WandbOshConfig = wb_cfg.wandb_osh
    if not osh_cfg.enabled:
        return None
    if wb_cfg.mode != "offline":
        logger.warning(f"wandb_osh is enabled but wandb.mode={wb_cfg.mode} (expected offline). Trigger disabled")
        return None

    if not osh_cfg.command_dir:
        logger.warning("wandb_osh.enabled=true but command_dir is unset. wandb-osh trigger disabled")
        return None

    command_dir = Path(osh_cfg.command_dir)
    command_dir.mkdir(parents=True, exist_ok=True)

    try:
        hooks_module: ModuleType = importlib.import_module("wandb_osh.hooks")
        hook_ctor: type[TriggerWandbSyncHook] = hooks_module.TriggerWandbSyncHook
    except Exception:
        logger.exception("wandb-osh hook import failed. Install wandb-osh to enable offline syncing")
        return None

    hook: TriggerWandbSyncHook = hook_ctor(communication_dir=command_dir)
    return WandbOshSyncTrigger(
        enabled=True,
        command_dir=command_dir,
        trigger_every_seconds=max(0, osh_cfg.trigger_every_seconds or _DEF_TRIGGER_SECONDS),
        timeout_seconds=max(1, osh_cfg.timeout_seconds or _DEF_TIMEOUT_SECONDS),
        trigger_on_phase_end=osh_cfg.trigger_on_phase_end,
        _trigger_fn=hook,
    )
