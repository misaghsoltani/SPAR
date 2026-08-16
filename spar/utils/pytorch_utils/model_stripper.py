from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
import re
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from logging import Logger
    from typing import TypeAlias

    from numpy.typing import NDArray


logger: Logger = getLogger(__name__)

_OPTIMIZER_KEYS: frozenset[str] = frozenset([
    "optimizer",
    "optimizer_state_dict",
    "optim_state_dict",
    "optimizer_state",
    "optim_state",
    "optimizer_config",
    "param_groups",
    "lr",
    "learning_rate",
    "momentum",
    "weight_decay",
    "eps",
    "betas",
    "amsgrad",
    "maximize",
    "foreach",
    "differentiable",
    "fused",
    "adam_w_mode",
    "capturable",
    "lr_decay",
])

_SCHEDULER_KEYS: frozenset[str] = frozenset([
    "scheduler",
    "scheduler_state_dict",
    "lr_scheduler",
    "lr_scheduler_state_dict",
    "scheduler_state",
    "lr_schedule",
    "step_lr",
    "gamma",
    "last_epoch",
    "verbose",
    "T_max",
    "eta_min",
    "milestones",
    "step_size",
    "patience",
    "factor",
    "threshold",
    "cooldown",
    "min_lr",
    "mode",
    "T_0",
    "T_mult",
    "warmup_epochs",
    "warmup_lr",
])

_TRAINING_KEYS: frozenset[str] = frozenset([
    "epoch",
    "epochs",
    "step",
    "steps",
    "iteration",
    "iterations",
    "global_step",
    "global_steps",
    "current_epoch",
    "current_step",
    "batch_idx",
    "batch_size",
    "num_batches",
    "total_steps",
    "total_epochs",
    "completed_epochs",
    "resume_epoch",
    "loss",
    "train_loss",
    "val_loss",
    "test_loss",
    "best_loss",
    "loss_history",
    "train_losses",
    "val_losses",
    "losses",
    "accuracy",
    "val_accuracy",
    "best_accuracy",
    "metrics",
    "train_metrics",
    "val_metrics",
    "best_metrics",
    "score",
    "best_score",
    "f1_score",
    "precision",
    "recall",
    "auc",
    "top1",
    "top5",
    "mse",
    "mae",
    "rmse",
    "psnr",
    "ssim",
    "bleu",
    "rouge",
    "perplexity",
])

_SYSTEM_KEYS: frozenset[str] = frozenset([
    "amp_scaler",
    "scaler",
    "grad_scaler",
    "gradient_scaler",
    "rng_state",
    "cuda_rng_state",
    "numpy_rng_state",
    "random_state",
    "torch_rng_state",
    "python_rng_state",
    "device",
    "cuda_device",
    "gpu_id",
    "local_rank",
    "world_size",
    "distributed",
    "ddp",
    "fsdp",
    "data_parallel",
    "model_parallel",
    "training_time",
    "elapsed_time",
    "wall_time",
    "timestamp",
    "config",
    "args",
    "hyperparameters",
    "hparams",
    "kwargs",
    "model_config",
    "train_config",
    "data_config",
    "optimizer_config",
    "scheduler_config",
    "checkpoint_config",
    "experiment_config",
    "pytorch_version",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "python_version",
    "git_hash",
    "git_commit",
    "build_info",
    "version",
    "framework_version",
    "library_version",
    "torchvision_version",
    "transformers_version",
    "build_date",
    "commit_hash",
    "branch",
    "tag",
    "version_info",
    "metadata",
    "checkpoint_metadata",
    "distributed_metadata",
    "rank_metadata",
    "shard_metadata",
    "storage_metadata",
    "fsdp_metadata",
    "ddp_metadata",
    "mp_metadata",
    "torch_distributed_metadata",
    "checkpoint_version",
    "compile_info",
    "compilation_metadata",
    "inductor_config",
])

# Suffix patterns for metadata removal
_SUFFIX_PATTERNS: frozenset[str] = frozenset([
    "_history",
    "_log",
    "_stats",
    "_counter",
    "_buffer",
    "_cache",
    "_tracker",
    "_monitor",
    "_callback",
    "_handler",
    "_hook",
    "_checkpoint",
    "_backup",
    "_temp",
    "_tmp",
    "_debug",
    "_info",
    "_meta",
    "_config",
    "_settings",
    "_params",
    "_options",
    "_flags",
    "_state",
    "_version",
    "_build",
    "_time",
    "_date",
    "_timestamp",
    "_seed",
    "_uuid",
    "_id",
])

NumpyRNGState: TypeAlias = tuple[str, "NDArray[np.uint32]", int, int, float]

CheckpointValue = (
    torch.Tensor
    | str
    | bytes
    | int
    | float
    | bool
    | NumpyRNGState
    | dict[str, "CheckpointValue"]
    | Sequence["CheckpointValue"]
    | dict[str, float | int | list[float]]
    | None
)


class StripOptions(TypedDict, total=False):
    """Options passed to batch_strip_checkpoints -> strip_pytorch_redundant_strings.

    All keys are optional and match the named parameters of
    :func:`strip_pytorch_redundant_strings`.
    """

    strip_optimizer: bool
    strip_scheduler: bool
    strip_training: bool
    strip_system: bool
    strip_dataparallel: bool
    strip_ddp: bool
    strip_compiled: bool
    strip_fsdp: bool
    strip_metadata_patterns: bool
    preserve_keys: set[str]
    strict_mode: bool
    in_place: bool


def strip_pytorch_redundant_strings(
    model_data: dict[str, CheckpointValue] | nn.Module | str,
    *,
    strip_optimizer: bool = False,
    strip_scheduler: bool = False,
    strip_training: bool = False,
    strip_system: bool = False,
    strip_dataparallel: bool = False,
    strip_ddp: bool = False,
    strip_compiled: bool = True,
    strip_fsdp: bool = False,
    strip_metadata_patterns: bool = False,
    preserve_keys: set[str] | None = None,
    strict_mode: bool = False,
    in_place: bool = False,
) -> dict[str, CheckpointValue]:
    """Strip PyTorch models' redundant strings.

    Args:
        model_data: Checkpoint dict, model, or file path
        strip_optimizer: Remove optimizer states and configs
        strip_scheduler: Remove scheduler states and configs
        strip_training: Remove training metadata (epochs, losses, metrics)
        strip_system: Remove system info (versions, devices, RNG states)
        strip_dataparallel: Remove DataParallel 'module.' prefixes
        strip_ddp: Remove DDP 'module.' prefixes
        strip_compiled: Remove torch.compile '_orig_mod.' prefixes
        strip_fsdp: Remove FSDP prefixes
        strip_metadata_patterns: Remove keys matching metadata patterns
        preserve_keys: Set of keys to always preserve
        strict_mode: Keep only state_dict/model keys
        in_place: Modify input dict directly

    Returns:
        Cleaned checkpoint dict
    """
    loaded: CheckpointValue
    if isinstance(model_data, str):
        loaded = torch.load(model_data, map_location="cpu", weights_only=False)
    elif isinstance(model_data, nn.Module):
        module_state: dict[str, CheckpointValue] = {}
        for key, value in model_data.state_dict().items():
            module_state[str(key)] = value
        module_checkpoint: dict[str, CheckpointValue] = {"state_dict": module_state}
        loaded = module_checkpoint
    else:
        loaded = model_data if in_place else dict(model_data)

    # Copy checkpoint keys into a mutable string-keyed mapping.
    if isinstance(loaded, dict):
        checkpoint: dict[str, CheckpointValue] = {str(k): v for k, v in loaded.items()}
    else:
        # Treat a non-mapping checkpoint payload as its state dictionary.
        checkpoint = {"state_dict": loaded}

    if not checkpoint:
        return {}

    preserve_keys = preserve_keys or set()

    if strict_mode:
        # Strict mode: keep only essential keys
        essential_keys: set[str] = {"state_dict", "model"} | preserve_keys
        return {k: v for k, v in checkpoint.items() if k in essential_keys}

    # Build removal set dynamically based on flags
    keys_to_remove: set[str] = set()

    if strip_optimizer:
        keys_to_remove.update(_OPTIMIZER_KEYS)
    if strip_scheduler:
        keys_to_remove.update(_SCHEDULER_KEYS)
    if strip_training:
        keys_to_remove.update(_TRAINING_KEYS)
    if strip_system:
        keys_to_remove.update(_SYSTEM_KEYS)

    # Remove metadata patterns
    if strip_metadata_patterns:
        for key in list(checkpoint.keys()):
            if key not in preserve_keys and (  # Check suffix patterns / private attributes
                any(key.endswith(suffix) for suffix in _SUFFIX_PATTERNS) or (key.startswith("_") and key != "_metadata")
            ):
                keys_to_remove.add(key)

    # Remove keys in batch
    for key in keys_to_remove:
        checkpoint.pop(key, None)

    # Clean state_dict with prefix removal
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        typed_state: dict[str, CheckpointValue] = dict(checkpoint["state_dict"])
        checkpoint["state_dict"] = _clean_state_dict(
            state_dict=typed_state,
            strip_dataparallel=strip_dataparallel,
            strip_ddp=strip_ddp,
            strip_compiled=strip_compiled,
            strip_fsdp=strip_fsdp,
        )

    # Handle 'model' key
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        typed_model: dict[str, CheckpointValue] = dict(checkpoint["model"])
        checkpoint["model"] = _clean_state_dict(
            state_dict=typed_model,
            strip_dataparallel=strip_dataparallel,
            strip_ddp=strip_ddp,
            strip_compiled=strip_compiled,
            strip_fsdp=strip_fsdp,
        )

    return checkpoint


def _clean_state_dict(
    state_dict: dict[str, CheckpointValue],
    strip_dataparallel: bool = True,
    strip_ddp: bool = True,
    strip_compiled: bool = True,
    strip_fsdp: bool = True,
) -> dict[str, CheckpointValue]:
    """Clean state_dict."""
    # If empty, return an empty dict (preserve dict type).
    if not state_dict:
        return {}

    # Check if any cleaning is needed
    needs_cleaning = False
    # Display a small sample of keys for inspection.
    sample_keys: list[str] = list(state_dict.keys())[: min(5, len(state_dict))]

    for key in sample_keys:
        if "module." in key or "_orig_mod." in key or "_module." in key or "_fsdp_" in key or "flat_param_" in key:
            needs_cleaning = True
            break

    if not needs_cleaning:
        return state_dict

    # Build active patterns based on flags
    active_patterns: list[tuple[str, int]] = []

    if strip_dataparallel or strip_ddp:
        # Both use 'module.' prefix
        active_patterns.append(("module.", 7))

    if strip_compiled:
        active_patterns.extend([("_orig_mod.", 10), ("_module.", 8), ("_wrapped_model.", 15)])

    if strip_fsdp:
        active_patterns.extend([("_fsdp_wrapped_module.", 22), ("_fpw_module.", 12), ("_flat_param.", 12)])

    if not active_patterns:
        return dict(state_dict)

    # Single-pass cleaning
    cleaned_dict: dict[str, CheckpointValue] = OrderedDict()

    for old_key, value in state_dict.items():
        new_key: str = old_key

        # Prefix matching
        for prefix, prefix_len in active_patterns:
            if new_key.startswith(prefix):
                new_key = new_key[prefix_len:]
                break  # Only remove first matching prefix

        # Handle special FSDP flat_param patterns
        if strip_fsdp and "flat_param_" in new_key:
            # Remove flat_param_N. pattern

            new_key = re.sub(r"^flat_param_\d+\.", "", new_key)

        # Reject rewrites that produce an empty key.
        if new_key:
            cleaned_dict[new_key] = value

    return cleaned_dict


def _estimate_size(obj: CheckpointValue | NDArray[np.uint32]) -> int:
    """Size estimation."""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_estimate_size(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_size(item) for item in obj)
    return len(str(obj)) * 2  # Rough estimate


def get_stripped_model_info(
    original_checkpoint: dict[str, CheckpointValue], cleaned_checkpoint: dict[str, CheckpointValue]
) -> dict[str, CheckpointValue]:
    """Compute stripping statistics."""
    original_size: int = _estimate_size(original_checkpoint) if original_checkpoint else 0
    cleaned_size: int = _estimate_size(cleaned_checkpoint) if cleaned_checkpoint else 0

    reduction: int = original_size - cleaned_size
    reduction_percent: float = (reduction / original_size) * 100 if original_size > 0 else 0

    # Count state_dict changes
    orig_state_keys: int = 0
    cleaned_state_keys: int = 0

    if "state_dict" in original_checkpoint and isinstance(original_checkpoint["state_dict"], dict):
        orig_state_keys = len(original_checkpoint["state_dict"])
    if "state_dict" in cleaned_checkpoint and isinstance(cleaned_checkpoint["state_dict"], dict):
        cleaned_state_keys = len(cleaned_checkpoint["state_dict"])

    return {
        "original_size_bytes": original_size,
        "cleaned_size_bytes": cleaned_size,
        "reduction_bytes": reduction,
        "reduction_percent": reduction_percent,
        "original_keys": len(original_checkpoint),
        "cleaned_keys": len(cleaned_checkpoint),
        "keys_removed": len(original_checkpoint) - len(cleaned_checkpoint),
        "original_state_dict_keys": orig_state_keys,
        "cleaned_state_dict_keys": cleaned_state_keys,
        "state_dict_keys_cleaned": orig_state_keys - cleaned_state_keys,
    }


def batch_strip_checkpoints(
    checkpoint_paths: list[str] | tuple[str, ...], output_dir: str, strip_options: StripOptions | None = None
) -> None:
    """Process multiple checkpoints."""
    output_path: Path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    def _strip_single(checkpoint_path: str) -> None:
        try:
            # Load and strip
            cleaned: dict[str, CheckpointValue] = strip_pytorch_redundant_strings(
                checkpoint_path, **(strip_options or {})
            )

            # Save with same name
            output_file = output_path / Path(checkpoint_path).name
            torch.save(cleaned, output_file)

            logger.info(f"✓ Cleaned: {checkpoint_path} -> {output_file}")

        except Exception:
            logger.exception(f"✗ Failed: {checkpoint_path}")

    for checkpoint_path in checkpoint_paths:
        _strip_single(checkpoint_path)


# Utility functions for common use cases
def strip_for_inference(checkpoint: dict[str, CheckpointValue]) -> dict[str, CheckpointValue]:
    """Prepare an inference-only model."""
    return strip_pytorch_redundant_strings(
        model_data=checkpoint,
        strip_optimizer=True,
        strip_scheduler=True,
        strip_training=True,
        strip_system=True,
        strip_dataparallel=True,
        strip_ddp=True,
        strip_compiled=True,
        strip_fsdp=True,
        strict_mode=True,
        in_place=False,
    )


def strip_distributed_prefixes_only(checkpoint: dict[str, CheckpointValue]) -> dict[str, CheckpointValue]:
    """Remove distributed prefix only."""
    return strip_pytorch_redundant_strings(
        model_data=checkpoint,
        strip_optimizer=False,
        strip_scheduler=False,
        strip_training=False,
        strip_system=False,
        strip_dataparallel=True,
        strip_ddp=True,
        strip_compiled=True,
        strip_fsdp=True,
        strip_metadata_patterns=False,
        strict_mode=False,
        in_place=False,
    )


def strip_compiled_prefixes(
    checkpoint: dict[str, CheckpointValue], in_place: bool = False
) -> dict[str, CheckpointValue]:
    """Remove only torch.compile prefixes."""
    return strip_pytorch_redundant_strings(
        checkpoint,
        strip_optimizer=False,
        strip_scheduler=False,
        strip_training=False,
        strip_system=False,
        strip_dataparallel=False,
        strip_ddp=False,
        strip_compiled=True,
        strip_fsdp=False,
        strip_metadata_patterns=False,
        strict_mode=False,
        in_place=in_place,
    )
