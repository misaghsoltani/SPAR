"""World Model Training Runner for SPAR."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, TypeVar

import h5py
import numpy as np
from rich.align import Align
from rich.panel import Panel
import torch

from spar.models.factory import ModelFactory
from spar.training.world_model_trainer.continuous_model_trainer import ContinuousWorldModelTrainer
from spar.training.world_model_trainer.discrete_model_trainer import DiscreteWorldModelTrainer
from spar.utils.log_utils.console_logger import terminal_console as console
from spar.utils.log_utils.wandb_logger import get_active_tracking_session, log_metrics, log_model_artifact
from spar.utils.log_utils.wandb_osh import build_wandb_osh_trigger
from spar.utils.pytorch_utils.nnet_utils import load_model

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from logging import Logger
    from typing import TypeAlias

    from numpy.typing import NDArray
    from torch import Tensor, nn
    from torch.nn import Parameter
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer
    from torch.utils.data import DataLoader

    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.training.world_model_trainer.base_trainer import WorldModelTrainerBase
    from spar.utils.config_utils.config_schema import (
        DataLoaderConfig,
        ModelConfig,
        TrainConfig,
        TrainDataPathConfig,
        TrainEnvModelSPARConfig,
        TrainPhaseConfig,
        TrainSavePathConfig,
    )
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession

    StateEpisode: TypeAlias = NDArray[np.float32]
    ActionEpisode: TypeAlias = list[int]
    Episodes: TypeAlias = tuple[list[StateEpisode], list[ActionEpisode]]
    Metrics: TypeAlias = dict[str, list[float] | int]
    TrainReturn: TypeAlias = dict[str, dict[str, nn.Module] | dict[int, dict[str, list[float] | int]]]

logger: Logger = logging.getLogger(__name__)
NumpyScalar = TypeVar("NumpyScalar", bound=np.generic)
ReporterScalar: TypeAlias = float | int | bool | str | None
ReporterPayload: TypeAlias = dict[str, ReporterScalar | dict[str, ReporterScalar]]


def _to_str_list(raw: str | bytes | Iterable[str | bytes] | None) -> list[str]:
    """Convert a raw value to a list of strings, handling bytes and None.

    Args:
        raw: The raw value to convert.

    Returns:
        A list of strings, or an empty list if raw is None or cannot be converted.
    """
    if raw is None:
        return []

    if isinstance(raw, (str, bytes)):
        raw = [raw]

    try:
        iterable: Iterable[str | bytes] = raw

    except TypeError:
        return []

    return [b.decode() if isinstance(b, bytes) else b for b in iterable]


def _decode_str(raw: str | bytes | None, *, default: str = "unknown") -> str:
    """Decode a raw value to a string, handling bytes and None.

    Args:
        raw: The raw value to decode.
        default: Default value to return if raw is None.

    Returns:
        The decoded string or the default value if raw is None.
    """
    if raw is None:
        return default
    return raw.decode() if isinstance(raw, bytes) else raw


def _read_dataset(group: h5py.Group, path: str, *, dtype: type[NumpyScalar]) -> NDArray[NumpyScalar]:
    """Return the numpy array stored at path inside group.

    Args:
        group: The HDF5 group containing the dataset.
        path: The path to the dataset within the group.
        dtype: Target numpy scalar dtype used for deterministic typing of the returned array.

    Returns:
        The numpy array read from the dataset.

    Raises:
        TypeError: If the object at path is not an HDF5 dataset.
    """
    obj: h5py.Group | h5py.Dataset | h5py.Datatype = group[path]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"Expected an HDF5 dataset at '{path}', got {type(obj).__name__}")
    return np.asarray(obj[...], dtype=dtype)


def _resolve_variant_group(
    group: h5py.Group, *, episode_idx: int, data_file: Path, variant_candidates: Iterable[str]
) -> str:
    """Return the name of a variant group containing states for an episode."""
    obj: h5py.Group | h5py.Dataset | h5py.Datatype
    for candidate in variant_candidates:
        if candidate in group:
            obj = group[candidate]
            if isinstance(obj, h5py.Group) and "states" in obj:
                return candidate

    for key in group:
        key_str = str(key)
        if key_str == "actions":
            continue
        obj = group[key_str]
        if isinstance(obj, h5py.Group) and "states" in obj:
            return key_str

    raise KeyError(f"No state variation group found for episode_{episode_idx:07d} in {data_file}.")


def load_test_data(data_file: str | Path, *, base_only: bool = False) -> Episodes:
    """Load test data stored by SPAR in a HDF-5 file.

    Args:
        data_file: Path to the HDF-5 file containing the test data.
        base_only: If True, load only the ``base`` variation for each episode.

    Returns:
        Tuple of lists containing state episodes and action episodes.

    Raises:
        KeyError: If ``base_only`` is True but the base variation is unavailable in the dataset.
    """
    start: float = time.perf_counter()
    data_file = Path(data_file)

    with h5py.File(data_file, "r", rdcc_nbytes=128 * 1024**2, rdcc_nslots=10_007, rdcc_w0=0.25) as h5:
        num_episodes = int(h5.attrs.get("num_episodes", 0))
        variant_names: list[str] = _to_str_list(h5.attrs.get("variant_names"))
        variant_type: str = _decode_str(h5.attrs.get("variant_type"))

        logger.info(
            f"Loading {data_file} - {num_episodes} episode(s), variant_type={variant_type}, "
            f"variants={variant_names}, base_only={base_only}"
        )

        states: list[StateEpisode] = []
        actions: list[ActionEpisode] = []

        episodes_obj = h5.get("episodes")
        if not isinstance(episodes_obj, h5py.Group):
            raise TypeError(f"Expected 'episodes' to be a group in {data_file}, got {type(episodes_obj).__name__}")
        episodes_grp: h5py.Group = episodes_obj
        prefer_base: bool = base_only or ("base" in variant_names)
        variant_candidates: list[str] = [name for name in variant_names if name != "base"]

        ep: h5py.Group
        for idx in range(num_episodes):
            ep_obj = episodes_grp.get(f"episode_{idx:07d}")
            if not isinstance(ep_obj, h5py.Group):
                raise KeyError(f"Missing episode group episode_{idx:07d} in {data_file}")
            ep = ep_obj

            # actions
            actions_arr: NDArray[np.int64] = _read_dataset(ep, "actions", dtype=np.int64)
            actions.append([int(action) for action in actions_arr.tolist()])

            has_base_group: bool = "base" in ep

            if base_only:
                if not has_base_group:
                    raise KeyError(
                        f"Requested base_only data but 'base' group is missing for episode_{idx:07d} in {data_file}."
                    )
                states.append(_read_dataset(ep, "base/states", dtype=np.float32))
                continue

            if has_base_group and (prefer_base or not variant_candidates):
                states.append(_read_dataset(ep, "base/states", dtype=np.float32))
                continue

            variant: str = _resolve_variant_group(
                ep, episode_idx=idx, data_file=data_file, variant_candidates=variant_candidates
            )
            states.append(_read_dataset(ep, f"{variant}/states", dtype=np.float32))

    elapsed: float = time.perf_counter() - start
    logger.info(f"Loaded {len(states)} episode(s) in {elapsed:.2f}s")
    logger.info(f"State array shape: {np.array(states).shape} | Action array shape: {np.array(actions).shape}")
    return states, actions


def train(
    env: ABCEnvironment[ABCState],
    cfg: TrainEnvModelSPARConfig,
    tracking: WandbTrackingSession | None = None,
    reporter: Callable[[ReporterPayload], None] | None = None,
) -> TrainReturn:
    """Train discrete or continuous environment model and return models and metrics.

    Args:
        env: The environment to train the model on.
        cfg: Configuration for training the environment model.
        tracking: Optional explicit W&B tracking session passed from the CLI lifecycle.
        reporter: Optional sparse progress callback invoked once per training phase.

    Returns:
        TrainReturn: A dictionary containing the trained models and training metrics.
    """
    # Extract config
    train_cfg: TrainConfig = cfg.train
    phases: list[TrainPhaseConfig] | None = train_cfg.phases
    if not phases:
        raise ValueError("train_cfg.phases must contain at least one phase")

    first_phase: TrainPhaseConfig = phases[0]

    data_paths_cfg: TrainDataPathConfig = cfg.data_paths
    save_paths_cfg: TrainSavePathConfig = cfg.save_paths

    # Device handling
    device: str = train_cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.info("[bold orange]WARNING:[/bold orange] CUDA is not available, switching to CPU.")
        device = "cpu"

    # Model factories
    is_continuous: bool = cfg.world_model_type == "continuous"
    get_encoder: Callable[[ModelConfig], nn.Module] = env.get_encoder_cont if is_continuous else env.get_encoder_disc
    get_decoder: Callable[[ModelConfig], nn.Module] = env.get_decoder_cont if is_continuous else env.get_decoder_disc
    get_env_model: Callable[[ModelConfig], nn.Module] = (
        env.get_env_model_cont if is_continuous else env.get_env_model_disc
    )

    # Create params list to collect parameters from models
    params_list: list[Parameter] = []

    # Load models
    encoder: nn.Module = load_model(
        model=get_encoder(cfg.model),
        device=device,
        freeze=False,
        params_list=params_list,
        compile_cfg=train_cfg.compile,
    )
    transition_model: nn.Module = load_model(
        model=get_env_model(cfg.model),
        device=device,
        freeze=False,
        params_list=params_list,
        compile_cfg=train_cfg.compile,
    )
    decoder: nn.Module = load_model(
        model=get_decoder(cfg.model),
        device=device,
        freeze=False,
        params_list=params_list,
        compile_cfg=train_cfg.compile,
    )
    models: dict[str, nn.Module] = {"encoder": encoder, "transition_model": transition_model, "decoder": decoder}
    optimizer: Optimizer = ModelFactory.build_optimizer(
        optimizer_name=train_cfg.optimizer, params=params_list, lr=first_phase.lr
    )
    scheduler: LRScheduler = ModelFactory.build_scheduler(optimizer=optimizer, cfg=train_cfg.scheduler)

    # Data
    data_loader_cfg: DataLoaderConfig = train_cfg.dataloader
    with console.status("Loading training data...", spinner="dots"):
        train_dataloader: DataLoader[dict[str, Tensor]] = ModelFactory.build_dataloader(
            data_paths_cfg.train_data, data_loader_cfg
        )
    with console.status("Loading validation data...", spinner="dots"):
        val_dataloader: DataLoader[dict[str, Tensor]] = ModelFactory.build_dataloader(
            data_paths_cfg.val_data, data_loader_cfg
        )

    test_state_episodes: list[StateEpisode]
    test_action_episodes: list[ActionEpisode]
    with console.status("Loading test data...", spinner="dots"):
        test_state_episodes, test_action_episodes = load_test_data(
            data_paths_cfg.test_data, base_only=data_loader_cfg.base_only
        )

    # Trainer selection
    checkpoint_path: str = str(Path(save_paths_cfg.model_dir) / "world_model_checkpoint.pth")
    Path(save_paths_cfg.model_dir).mkdir(exist_ok=True, parents=True)

    trainer_cls: type[WorldModelTrainerBase] = (
        ContinuousWorldModelTrainer if is_continuous else DiscreteWorldModelTrainer
    )
    tracking_session: WandbTrackingSession | None = tracking or get_active_tracking_session()
    wandb_osh_trigger = build_wandb_osh_trigger(cfg.wandb)

    trainer: WorldModelTrainerBase = trainer_cls(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_state_episodes=test_state_episodes,
        test_action_episodes=test_action_episodes,
        encoder=encoder,
        transition_model=transition_model,
        decoder=decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        env_coeff=None if cfg.end_to_end else first_phase.env_coeff,
        device=device,
        use_wandb=cfg.wandb.mode in {"online", "offline"},
        tracking_session=tracking_session,
        wandb_osh_trigger=wandb_osh_trigger,
        checkpoint_path=checkpoint_path,
        end_to_end=cfg.end_to_end,
    )

    # Training loop
    batch_size: int = data_loader_cfg.batch_size
    metrics_all: dict[int, Metrics] = {}
    current_itr: int = 0

    for phase_idx, phase in enumerate(phases):
        max_itr: int = phase.max_itrs

        logger.info(
            Panel(
                Align.center(
                    f"Phase {phase_idx + 1}/{len(phases)} | "
                    f"lr={phase.lr:.2E} | "
                    f"{'' if cfg.end_to_end else f'env_coeff={phase.env_coeff} | '}Itrs=[{current_itr}, {max_itr}]"
                ),
                title="[bold yellow]Training Phase[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
                width=120,
            )
        )

        trainer.test_model(batch_size, num_steps=train_cfg.test_steps, message="Testing before training")

        metrics: Metrics = trainer.train(
            phase_name=f"{phase_idx + 1}/{len(phases)}",
            starting_iteration=current_itr,
            ending_iteration=max_itr,
            initial_lr=phase.lr,
            env_coeff=None if cfg.end_to_end else phase.env_coeff,
        )
        current_itr = max_itr
        metrics["current_itr"] = current_itr
        metrics_all[phase_idx] = metrics

        trainer.test_model(batch_size, num_steps=train_cfg.test_steps, message="Testing after training")

        # WandB logging
        if cfg.wandb.mode in {"online", "offline"} and tracking_session is not None:
            summary: dict[str, float | int | None] = {
                "phase": phase_idx,
                "lr": phase.lr,
                "env_coeff": None if cfg.end_to_end else phase.env_coeff,
                "global_step": current_itr,
            }

            for key, val in metrics.items():
                if isinstance(val, list) and val:
                    summary[f"final_{key}"] = val[-1]
                elif isinstance(val, (int, float)):
                    summary[f"final_{key}"] = val

            log_metrics(tracking_session, summary, step=current_itr)

            if phase_idx == len(phases) - 1:
                model_name: str = f"{env.get_env_name()}_{getattr(cfg, 'world_model_type', 'unknown_type')}_model_final"
                train_loss: list[float] | int | None = metrics.get("train_loss")
                val_loss: list[float] | int | None = metrics.get("val_loss")
                base_meta: dict[str, float | int | None] = {
                    "current_itr": current_itr,
                    "final_train_loss": train_loss[-1] if isinstance(train_loss, list) and train_loss else None,
                    "final_val_loss": val_loss[-1] if isinstance(val_loss, list) and val_loss else None,
                }
                for comp_name, model in models.items():
                    log_model_artifact(
                        tracking_session,
                        model,
                        name=f"{model_name}_{comp_name}",
                        artifact_type="model",
                        metadata={**base_meta, "component": comp_name},
                    )

        if reporter is not None:
            val_loss_raw: list[float] | int | None = metrics.get("val_loss")
            train_loss_raw: list[float] | int | None = metrics.get("train_loss")
            report_val_loss: float | None = (
                val_loss_raw[-1]
                if isinstance(val_loss_raw, list) and val_loss_raw
                else float(val_loss_raw)
                if isinstance(val_loss_raw, int | float)
                else None
            )
            report_train_loss: float | None = (
                train_loss_raw[-1]
                if isinstance(train_loss_raw, list) and train_loss_raw
                else float(train_loss_raw)
                if isinstance(train_loss_raw, int | float)
                else None
            )
            reporter({
                "phase_index": phase_idx,
                "iteration": current_itr,
                "primary": report_val_loss,
                "metrics": {"val_loss": report_val_loss, "train_loss": report_train_loss, "lr": phase.lr},
            })

    return {"models": models, "metrics": metrics_all}
