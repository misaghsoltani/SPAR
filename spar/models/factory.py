from __future__ import annotations

from typing import TYPE_CHECKING, overload

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
import torch
from torch import nn, optim
import torch.nn.functional as F

from spar.data.dataset import create_dataloader
from spar.utils.config_utils.config_schema import ModelArchitectureConfig, ModelComponentConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal, TypeAlias, TypeGuard

    from torch import Tensor
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import ParamsT
    from torch.utils.data import DataLoader

    from spar.utils.config_utils.config_schema import (
        AlignmentModelConfig,
        ArchitectureSpec,
        DataLoaderConfig,
        DecoderConfig,
        EncoderConfig,
        EnvModelConfig,
        HydraValue,
        ModuleSpec,
        SchedulerConfig,
        SchedulerParams,
    )


ConfigMapping: TypeAlias = dict[str, "HydraValue"]
ArchitectureInput: TypeAlias = "ArchitectureSpec | ListConfig"


def _is_architecture_spec(value: HydraValue) -> TypeGuard[ArchitectureSpec]:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _extract_architecture(config: DictConfig | ConfigMapping) -> ArchitectureInput | None:
    if isinstance(config, DictConfig):
        raw_specs = config.get("architecture")
        if raw_specs is None or isinstance(raw_specs, (ListConfig, list)):
            return raw_specs
        raise TypeError("Config 'architecture' field must be a list of module specs")

    raw_specs = config.get("architecture")
    if raw_specs is None:
        return None
    if _is_architecture_spec(raw_specs):
        return raw_specs
    raise TypeError("Config 'architecture' field must be a list of module specs")


def _has_architecture(specs: ArchitectureInput | None) -> TypeGuard[ArchitectureInput]:
    return specs is not None and len(specs) > 0


def _scheduler_int(params: SchedulerParams, key: str) -> int:
    value = params[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Scheduler parameter '{key}' must be an int")
    return value


def _scheduler_float(params: SchedulerParams, key: str) -> float:
    value = params[key]
    if isinstance(value, bool):
        raise TypeError(f"Scheduler parameter '{key}' must be a float")
    return float(value)


class ModelFactory:
    """Factory to create PyTorch modules from Hydra configs."""

    # Overloads to provide precise return types based on the input config shape
    @overload
    @staticmethod
    def build_model(config: ModelComponentConfig) -> dict[str, nn.Sequential]: ...

    @overload
    @staticmethod
    def build_model(config: ConfigMapping) -> dict[str, nn.Sequential]: ...

    @overload
    @staticmethod
    def build_model(config: DictConfig) -> dict[str, nn.Sequential]: ...

    @overload
    @staticmethod
    def build_model(
        config: ModelArchitectureConfig | AlignmentModelConfig | EncoderConfig | DecoderConfig | EnvModelConfig,
    ) -> nn.Sequential: ...

    @overload
    @staticmethod
    def build_model(config: ListConfig | list[ModuleSpec]) -> nn.Sequential: ...

    @staticmethod
    def build_model(
        config: (
            ModelComponentConfig
            | ModelArchitectureConfig
            | AlignmentModelConfig
            | EncoderConfig
            | DecoderConfig
            | EnvModelConfig
            | DictConfig
            | ListConfig
            | list[ModuleSpec]
            | ConfigMapping
        ),
    ) -> nn.Sequential | dict[str, nn.Sequential]:
        """Create model(s) based on a Hydra/OmegaConf config.

        Args:
            config: Can be one of:
                - A raw Python list of spec-dicts (or ListConfig)
                - A ModelArchitectureConfig or its subclasses (AlignmentModelConfig,
                  EncoderConfig, etc.) with 'architecture' field
                - A DictConfig with 'architecture' field mapping to a list
                - A Python dict mapping names to sub-config DictConfigs

        Returns:
            - 'nn.Sequential' for raw list or single config
            - 'Dict[str, nn.Sequential]' for dict of sub-configs
        """
        # Handle ModelComponentConfig specifically (dataclass with sub-configs)
        models: dict[str, nn.Sequential] = {}
        if isinstance(config, ModelComponentConfig):
            component_configs: dict[str, ModelArchitectureConfig | None] = {
                "encoder": config.encoder,
                "env_model": config.env_model,
                "decoder": config.decoder,
                "alignment_model": config.alignment_model,
            }
            for name_str, component_subcfg in component_configs.items():
                # Skip None or empty configs
                if component_subcfg is None:
                    models[name_str] = nn.Sequential()
                    continue

                component_arch: ArchitectureInput | None = component_subcfg.architecture
                if not _has_architecture(component_arch):
                    # Create empty Sequential for configs without architecture
                    models[name_str] = nn.Sequential()
                    continue

                models[name_str] = ModelFactory.build_model(component_arch)
            return models

        # Dict of named sub-configs (no top-level 'architecture')
        if (isinstance(config, DictConfig) or type(config) is dict) and "architecture" not in config:
            for key, dict_subcfg in config.items():
                name_str = str(key)

                # Skip None or empty configs
                if dict_subcfg is None or (type(dict_subcfg) is dict and not dict_subcfg):
                    continue

                if not (isinstance(dict_subcfg, DictConfig) or type(dict_subcfg) is dict):
                    continue

                # Get architecture - might be None for empty configs
                dict_arch = _extract_architecture(dict_subcfg)
                if not _has_architecture(dict_arch):
                    # Create empty Sequential for configs without architecture
                    models[name_str] = nn.Sequential()
                    continue

                models[name_str] = ModelFactory.build_model(dict_arch)
            return models

        # ModelArchitectureConfig and its subclasses expose architecture directly
        specs: ArchitectureInput | None
        if isinstance(config, ModelArchitectureConfig):
            specs = config.architecture
        # Raw list or single DictConfig with 'architecture'
        elif isinstance(config, (ListConfig, list)):
            specs = config
        else:
            specs = _extract_architecture(config)

        if not _has_architecture(specs):
            # Return empty Sequential for None or empty architecture
            return nn.Sequential()

        modules = instantiate(specs)
        if isinstance(modules, ListConfig):
            # Convert ListConfig to Python list
            modules = list(modules)
        elif not isinstance(modules, list):
            raise TypeError("Hydra instantiate did not return a list or ListConfig of modules")

        # Wrap into nn.Sequential and return
        return nn.Sequential(*modules)

    @staticmethod
    def build_optimizer(optimizer_name: str, params: ParamsT, lr: float | Tensor) -> optim.Optimizer:
        """Create an optimizer instance based on the optimizer name."""
        if optimizer_name == "adam":
            return optim.Adam(params=params, lr=lr)
        if optimizer_name == "sgd":
            return optim.SGD(params=params, lr=lr)
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    @staticmethod
    def build_scheduler(optimizer: optim.Optimizer, cfg: SchedulerConfig) -> LRScheduler:
        """Create a learning rate scheduler based on the provided config."""
        if cfg.type == "StepLR":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=_scheduler_int(cfg.params, "step_size"),
                gamma=_scheduler_float(cfg.params, "gamma"),
            )
        if cfg.type == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=_scheduler_float(cfg.params, "gamma"))
        if cfg.type == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=_scheduler_int(cfg.params, "T_max"), eta_min=_scheduler_float(cfg.params, "eta_min")
            )
        raise ValueError(f"Unsupported scheduler type: {cfg.type}")

    # Overloads for build_loss_function: when return_module is False -> Callable[..., Tensor], True -> nn.Module
    @overload
    @staticmethod
    def build_loss_function(loss_name: str, return_module: Literal[False] = ...) -> Callable[..., Tensor]: ...

    @overload
    @staticmethod
    def build_loss_function(loss_name: str, return_module: Literal[True]) -> nn.Module: ...

    @staticmethod
    def build_loss_function(loss_name: str, return_module: bool = False) -> Callable[..., Tensor] | nn.Module:
        """Return a loss function or nn.Module based on the loss name."""
        if loss_name == "cross_entropy":
            return F.cross_entropy if not return_module else nn.CrossEntropyLoss()
        if loss_name == "mse":
            return F.mse_loss if not return_module else nn.MSELoss()
        if loss_name == "bce":
            return F.binary_cross_entropy if not return_module else nn.BCELoss()
        raise ValueError(f"Unsupported loss function: {loss_name}")

    @staticmethod
    def build_dataloader(file_path: str, cfg: DataLoaderConfig) -> DataLoader[dict[str, Tensor]]:
        """Create a DataLoader instance based on the provided config."""
        dtype_mapping: dict[str, torch.dtype] = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "int8": torch.int8,
            "float16": torch.float16,
            "long": torch.long,
        }
        return create_dataloader(
            file_path=file_path,
            batch_size=cfg.batch_size,
            num_batches_per_epoch=cfg.num_batches_per_epoch,
            replacement=cfg.replacement,
            transform=cfg.transform,
            dtype=dtype_mapping[cfg.dtype],
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.prefetch_factor,
            pin_memory_device=cfg.pin_memory_device,
            infinite=cfg.infinite,
            base_only=cfg.base_only,
            variations_to_use=cfg.variations_to_use,
            variations_to_ignore=cfg.variations_to_ignore,
        )
