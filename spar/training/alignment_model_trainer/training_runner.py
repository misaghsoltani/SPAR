"""Alignment Model Trainer for SPAR."""

from __future__ import annotations

from logging import getLogger
import pathlib
from typing import TYPE_CHECKING

from rich.align import Align
from rich.panel import Panel
import torch

from spar.data.alignment_dataset import AlignmentDataset, create_dataloader
from spar.models.factory import ModelFactory
from spar.utils.log_utils.console_logger import terminal_console as console
from spar.utils.log_utils.wandb_logger import get_active_tracking_session, log_metrics, log_model_artifact
from spar.utils.log_utils.wandb_osh import build_wandb_osh_trigger
from spar.utils.pytorch_utils.nnet_utils import load_model

from .continuous_model_trainer import ContinuousAlignmentModelTrainer
from .discrete_model_trainer import DiscreteAlignmentModelTrainer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from logging import Logger
    from typing import TypeAlias

    from torch import Tensor, nn
    from torch.nn import Parameter
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader, Dataset

    from spar.data.alignment_dataset import AlignmentDatasetInfo
    from spar.environments.abstracts import ABCEnvironment, ABCState
    from spar.utils.config_utils.config_schema import (
        CompileConfig,
        DataLoaderConfig,
        ModelConfig,
        PretrainedModelPathConfig,
        TrainAlignmentModelSPARConfig,
        TrainConfig,
        TrainDataPathConfig,
        TrainPhaseConfig,
        TrainSavePathConfig,
    )
    from spar.utils.log_utils.wandb_logger import WandbTrackingSession

    from .base_trainer import AlignmentModelTrainerBase, MetricHistory, MetricHistoryValue

logger: Logger = getLogger(__name__)
ReporterScalar: TypeAlias = float | int | bool | str | None
ReporterPayload: TypeAlias = dict[str, ReporterScalar | dict[str, ReporterScalar]]
RunnerMetricSeries: TypeAlias = list["MetricHistoryValue"]
RunnerMetricValue: TypeAlias = RunnerMetricSeries | int | float
RunnerMetrics: TypeAlias = dict[str, RunnerMetricValue]


def _last_numeric_metric(value: RunnerMetricValue | None) -> float | int | None:
    if isinstance(value, list):
        if not value:
            return None
        last_value = value[-1]
        return last_value if isinstance(last_value, (int, float)) else None
    return value


def summarize_metrics(metrics: Mapping[str, RunnerMetricValue]) -> dict[str, float | int | None]:
    """Return a scalar WandB summary from trainer metrics."""
    out: dict[str, float | int | None] = {}
    for key, val in metrics.items():
        out[f"final_{key}"] = _last_numeric_metric(val)

    return out


def log_final_models(
    tracking_session: WandbTrackingSession,
    models: dict[str, nn.Module],
    metrics: Mapping[str, RunnerMetricValue],
    env: ABCEnvironment[ABCState],
    cfg: TrainAlignmentModelSPARConfig,
    current_itr: int,
) -> None:
    """Log final model artifacts to WandB with basic metadata."""
    env_name: str = env.get_env_name()
    alignment_type: str = getattr(cfg, "alignment_model_type", "unknown_type")
    model_name: str = f"{env_name}_{alignment_type}_alignment_model_final"

    train_loss_val: float | int | None = _last_numeric_metric(metrics.get("train_loss"))
    val_loss_val: float | int | None = _last_numeric_metric(metrics.get("val_loss"))
    final_train_loss: float | None = float(train_loss_val) if train_loss_val is not None else None
    final_val_loss: float | None = float(val_loss_val) if val_loss_val is not None else None

    base_metadata: dict[str, float | None] = {
        "current_itr": current_itr,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }

    for component_name, model_component in models.items():
        log_model_artifact(
            tracking_session,
            model_component,
            name=f"{model_name}_{component_name}",
            artifact_type="model",
            metadata={**base_metadata, "component": component_name},
        )


def train(
    env: ABCEnvironment[ABCState],
    cfg: TrainAlignmentModelSPARConfig,
    tracking: WandbTrackingSession | None = None,
    reporter: Callable[[ReporterPayload], None] | None = None,
) -> dict[str, dict[str, nn.Module] | dict[int, RunnerMetrics]]:
    """Train alignment model.

    Args:
        env: The environment to use.
        cfg: The configuration object.
        tracking: Optional explicit W&B tracking session passed from the CLI lifecycle.
        reporter: Optional sparse progress callback invoked once per training phase.

    Returns:
        Dictionary containing the final alignment model and metrics.
    """
    train_cfg: TrainConfig = cfg.train
    pretrained_paths_cfg: PretrainedModelPathConfig = cfg.pretrained_model_paths
    data_paths_cfg: TrainDataPathConfig = cfg.data_paths
    save_paths_cfg: TrainSavePathConfig = cfg.save_paths
    end_to_end: bool = cfg.end_to_end
    freeze_pretrained_models: bool = cfg.freeze_pretrained_models

    device: str = train_cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.info("[bold orange]WARNING:[/ bold orange] CUDA is not available, switching to CPU.")
        device = "cpu"

    is_continuous: bool = cfg.alignment_model_type == "continuous"

    # Create alignment model
    alignment_model: nn.Module = env.get_alignment_model(cfg.model).to(device)
    get_encoder: Callable[[ModelConfig], nn.Module] = env.get_encoder_cont if is_continuous else env.get_encoder_disc
    get_decoder: Callable[[ModelConfig], nn.Module] = env.get_decoder_cont if is_continuous else env.get_decoder_disc
    get_env_model: Callable[[ModelConfig], nn.Module] = (
        env.get_env_model_cont if is_continuous else env.get_env_model_disc
    )
    encoder: nn.Module | None = get_encoder(cfg.model) if not end_to_end else None
    decoder: nn.Module | None = get_decoder(cfg.model) if end_to_end else None
    transition_model: nn.Module | None = get_env_model(cfg.model) if end_to_end else None

    pretrained_encoder_path: str | None = pretrained_paths_cfg.encoder_path
    assert end_to_end or pretrained_encoder_path is not None, (
        "Pretrained encoder path must be provided when not in end-to-end mode."
    )

    pretrained_decoder_path: str | None = pretrained_paths_cfg.decoder_path
    assert not end_to_end or not freeze_pretrained_models or pretrained_decoder_path is not None, (
        "Pretrained decoder path must be provided when in end-to-end mode with freeze_pretrained_models=True."
    )

    pretrained_transition_model_path: str | None = pretrained_paths_cfg.transition_model_path
    assert not end_to_end or not freeze_pretrained_models or pretrained_transition_model_path is not None, (
        "Pretrained transition model path must be provided when in end-to-end mode with freeze_pretrained_models=True."
    )

    # Collect parameters for optimization (explicit list of nn.Parameter)
    params_list: list[Parameter] = list(alignment_model.parameters())

    compile_cfg: CompileConfig | None = train_cfg.compile if not train_cfg.compile.disable else None

    # Configure alignment model (no pretrained weights, always trainable)
    alignment_model = load_model(
        model=alignment_model,
        device=device,
        pretrained_path=None,
        freeze=False,
        compile_cfg=compile_cfg,
        params_list=None,  # Already added to params_list above
    )

    # Configure encoder
    if encoder is not None:
        encoder = load_model(
            model=encoder,
            device=device,
            pretrained_path=pretrained_encoder_path,
            freeze=True,  # Encoder is always frozen
            compile_cfg=compile_cfg,
            params_list=None,
        )

    # Configure transition model
    if transition_model is not None:
        # In end-to-end mode: respect freeze_pretrained_models if pretrained path exists, otherwise train from scratch
        # Not in end-to-end mode: freeze if pretrained path exists
        # (standard alignment training doesn't train world model)
        freeze_transition: bool = bool(pretrained_transition_model_path) and (
            not end_to_end or freeze_pretrained_models
        )

        transition_model = load_model(
            model=transition_model,
            device=device,
            pretrained_path=pretrained_transition_model_path,
            freeze=freeze_transition,
            compile_cfg=compile_cfg,
            params_list=params_list if not freeze_transition else None,
        )

    # Configure decoder
    if decoder is not None:
        # In end-to-end mode: respect freeze_pretrained_models if pretrained path exists, otherwise train from scratch
        # Not in end-to-end mode: freeze if pretrained path exists
        # (standard alignment training doesn't use decoder)

        freeze_decoder: bool = bool(pretrained_decoder_path) and (not end_to_end or freeze_pretrained_models)
        decoder = load_model(
            model=decoder,
            device=device,
            pretrained_path=pretrained_decoder_path,
            freeze=freeze_decoder,
            compile_cfg=compile_cfg,
            params_list=params_list if not freeze_decoder else None,
        )

    # Store model components for logging and return
    models: dict[str, nn.Module] = {"alignment_model": alignment_model}
    if encoder is not None:
        models["pretrained_encoder"] = encoder
    if transition_model is not None:
        models["transition_model"] = transition_model
    if decoder is not None:
        models["decoder"] = decoder

    # Training requires at least one configured phase.
    assert train_cfg.phases is not None, "Train phases must be specified"
    first_phase: TrainPhaseConfig = train_cfg.phases[0]
    optimizer: Optimizer = ModelFactory.build_optimizer(
        optimizer_name=train_cfg.optimizer, params=params_list, lr=first_phase.lr
    )
    scheduler: LRScheduler = ModelFactory.build_scheduler(optimizer=optimizer, cfg=train_cfg.scheduler)
    use_wandb: bool = cfg.wandb.mode in {"online", "offline"}

    data_loader_cfg: DataLoaderConfig = train_cfg.dataloader

    # Load training and validation data
    # Don't precompute targets in end-to-end mode since raw base images are used
    precompute_targets: bool = cfg.precompute_targets and not end_to_end

    # Validate that encoder is provided when online encoding is needed
    if not precompute_targets and not end_to_end and encoder is None:
        raise ValueError(
            "Online target encoding requires a pretrained encoder. Either:\n"
            "1. Set precompute_targets=True in config to use precomputed targets, or\n"
            "2. Provide a valid encoder for online encoding."
        )

    with console.status("Loading training data...\n", spinner="dots"):
        train_dataloader: DataLoader[dict[str, Tensor]] = create_dataloader(
            file_path=data_paths_cfg.train_data,
            batch_size=data_loader_cfg.batch_size,
            num_batches_per_epoch=data_loader_cfg.num_batches_per_epoch,
            replacement=data_loader_cfg.replacement,
            dtype=torch.float32,
            infinite=data_loader_cfg.infinite,
            num_workers=data_loader_cfg.num_workers,
            pin_memory=data_loader_cfg.pin_memory,
            persistent_workers=data_loader_cfg.persistent_workers,
            pin_memory_device=data_loader_cfg.pin_memory_device,
            prefetch_factor=data_loader_cfg.prefetch_factor,
            variations_to_use=data_loader_cfg.variations_to_use,
            variations_to_ignore=data_loader_cfg.variations_to_ignore,
            encoder=encoder if precompute_targets and not end_to_end else None,
            precompute_targets=precompute_targets and not end_to_end,
            device=device,
            use_next_state_targets=end_to_end,
        )

        # Validate that actions are available for end-to-end training
        if end_to_end:
            ds: Dataset[dict[str, Tensor]] = train_dataloader.dataset
            if not isinstance(ds, AlignmentDataset):
                raise ValueError("End-to-end training requires an AlignmentDataset instance for training data")

            dataset_info: AlignmentDatasetInfo = ds.get_info()

            if "actions_shape" not in dataset_info:
                raise ValueError(
                    "End-to-end training requires actions in the dataset, but no actions were found. "
                    "The training data must include action information."
                )

    with console.status("Loading validation data...\n", spinner="dots"):
        val_dataloader: DataLoader[dict[str, Tensor]] = create_dataloader(
            file_path=data_paths_cfg.val_data,
            batch_size=data_loader_cfg.batch_size,
            num_batches_per_epoch=data_loader_cfg.num_batches_per_epoch,
            replacement=data_loader_cfg.replacement,
            dtype=torch.float32,
            infinite=data_loader_cfg.infinite,
            num_workers=data_loader_cfg.num_workers,
            pin_memory=data_loader_cfg.pin_memory,
            persistent_workers=data_loader_cfg.persistent_workers,
            pin_memory_device=data_loader_cfg.pin_memory_device,
            prefetch_factor=data_loader_cfg.prefetch_factor,
            variations_to_use=data_loader_cfg.variations_to_use,
            variations_to_ignore=data_loader_cfg.variations_to_ignore,
            encoder=encoder if precompute_targets and not end_to_end else None,
            precompute_targets=precompute_targets and not end_to_end,
            device=device,
            use_next_state_targets=end_to_end,
        )

    # Create save directory
    pathlib.Path(save_paths_cfg.model_dir).mkdir(exist_ok=True, parents=True)
    checkpoint_path: str = str(pathlib.Path(save_paths_cfg.model_dir) / "alignment_model_checkpoint.pth")

    # Track which models are actually frozen for proper mode management
    is_transition_frozen: bool = transition_model is not None and (
        (end_to_end and pretrained_transition_model_path is not None and freeze_pretrained_models)
        or (not end_to_end and pretrained_transition_model_path is not None)
    )
    is_decoder_frozen: bool = decoder is not None and (
        (end_to_end and pretrained_decoder_path is not None and freeze_pretrained_models)
        or (not end_to_end and pretrained_decoder_path is not None)
    )

    trainer_cls: type[AlignmentModelTrainerBase] = (
        ContinuousAlignmentModelTrainer if is_continuous else DiscreteAlignmentModelTrainer
    )

    # Construct the selected trainer with the shared inputs.
    tracking_session: WandbTrackingSession | None = tracking or get_active_tracking_session()
    wandb_osh_trigger = build_wandb_osh_trigger(cfg.wandb)

    trainer: AlignmentModelTrainerBase = trainer_cls(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        pretrained_encoder=encoder,
        alignment_model=alignment_model,
        transition_model=transition_model,
        decoder=decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_wandb=use_wandb,
        tracking_session=tracking_session,
        wandb_osh_trigger=wandb_osh_trigger,
        checkpoint_path=checkpoint_path,
        end_to_end=end_to_end,
        freeze_pretrained_models=freeze_pretrained_models,
        is_transition_frozen=is_transition_frozen,
        is_decoder_frozen=is_decoder_frozen,
        has_transition_path=pretrained_transition_model_path is not None,
        has_decoder_path=pretrained_decoder_path is not None,
    )

    # Training loop over phases
    phases: list[TrainPhaseConfig] = train_cfg.phases
    metrics_all: dict[int, RunnerMetrics] = {}
    current_itr: int = 0

    for phase_idx, phase in enumerate(phases):
        max_itr: int = phase.max_itrs

        logger.info(
            Panel(
                Align.center(
                    f"Phase {phase_idx + 1}/{len(train_cfg.phases)} | "
                    f"lr={phase.lr:.2E} | "
                    f"Itrs=[{current_itr}, {max_itr}]"
                ),
                title="[bold yellow]Alignment Training Phase[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
                width=120,
            )
        )

        # Train for this phase
        metrics_raw: MetricHistory = trainer.train(
            phase_name=f"{phase_idx + 1}/{len(train_cfg.phases)}",
            starting_iteration=current_itr,
            ending_iteration=max_itr,
            initial_lr=phase.lr,
        )

        metrics: RunnerMetrics = dict(metrics_raw)
        current_itr = max_itr
        metrics["current_itr"] = current_itr
        metrics_all[phase_idx] = metrics

        # WandB logging
        if use_wandb and tracking_session is not None:
            # Create summary dictionary
            summary: dict[str, float | int | None] = {"phase": phase_idx, "lr": phase.lr, "global_step": current_itr}
            summary.update(summarize_metrics(metrics))

            log_metrics(tracking_session, summary, step=current_itr)

            if phase_idx == len(train_cfg.phases) - 1:
                log_final_models(
                    tracking_session=tracking_session,
                    models=models,
                    metrics=metrics,
                    env=env,
                    cfg=cfg,
                    current_itr=current_itr,
                )

        if reporter is not None:
            val_loss_raw: float | int | None = _last_numeric_metric(metrics.get("val_loss"))
            train_loss_raw: float | int | None = _last_numeric_metric(metrics.get("train_loss"))
            val_loss: float | None = float(val_loss_raw) if val_loss_raw is not None else None
            train_loss: float | None = float(train_loss_raw) if train_loss_raw is not None else None
            reporter({
                "phase_index": phase_idx,
                "iteration": current_itr,
                "primary": val_loss,
                "metrics": {"val_loss": val_loss, "train_loss": train_loss, "lr": phase.lr},
            })

    return {"models": models, "metrics": metrics_all}
