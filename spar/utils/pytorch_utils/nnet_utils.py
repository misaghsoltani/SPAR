"""Utility functions for neural networks and models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from spar.utils.pytorch_utils.model_stripper import CheckpointValue, strip_pytorch_redundant_strings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.nn import Parameter
    from torch.types import Device

    from spar.utils.config_utils.config_schema import CompileConfig

# import torch.distributed as dist
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
# from torch.distributed.fsdp.wrap import ModuleWrapPolicy
# from spar.utils.config_utils.config_schema import FSDPConfig

# def create_fsdp_model(
#     model: nn.Module, fsdp_cfg: FSDPConfig, device_id: int | torch.device | None = None
# ) -> FSDP | nn.Module:
#     """Create an FSDP-wrapped model with sharded parameters.

#     Args:
#         model: The model to wrap
#         fsdp_cfg: FSDP configuration object
#         device_id: The device ID to use for the model. If None, the model will be placed on the current CUDA device.

#     Returns:
#         FSDP-wrapped model or original model if not distributed
#     """
#     if not dist.is_initialized():
#         return model

#     param_dtype = getattr(torch, fsdp_cfg.param_dtype, torch.float16)
#     reduce_dtype = getattr(torch, fsdp_cfg.reduce_dtype, torch.float16)
#     buffer_dtype = getattr(torch, fsdp_cfg.buffer_dtype, torch.float16)

#     # Configure the selected mixed-precision policy.
#     mixed_precision_policy: MixedPrecision | None = (
#         MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
#         if fsdp_cfg.mixed_precision and torch.cuda.is_available()
#         else None
#     )

#     wrap_policy = ModuleWrapPolicy(fsdp_cfg.wrap_policy_module_classes)

#     fsdp_model = FSDP(
#         module=model,
#         auto_wrap_policy=wrap_policy,
#         mixed_precision=mixed_precision_policy,
#         device_id=device_id if device_id is not None else torch.cuda.current_device(),
#         sync_module_states=fsdp_cfg.sync_module_states,
#         cpu_offload=fsdp_cfg.cpu_offload,
#         sharding_strategy=fsdp_cfg.sharding_strategy,
#         backward_prefetch=fsdp_cfg.backward_prefetch,
#         ignored_modules=fsdp_cfg.ignored_modules,
#         param_init_fn=fsdp_cfg.param_init_fn,
#         forward_prefetch=fsdp_cfg.forward_prefetch,
#         limit_all_gathers=fsdp_cfg.limit_all_gathers,
#         use_orig_params=fsdp_cfg.use_orig_params,
#         ignored_states=fsdp_cfg.ignored_states,
#     )

#     return fsdp_model


def load_model(
    model: nn.Module,
    device: Device = "cpu",
    pretrained_path: str | None = None,
    strip_compiled_prefixes: bool = False,
    strip_ddp_prefixes: bool = False,
    strip_dataparallel_prefixes: bool = False,
    freeze: bool = False,
    compile_cfg: CompileConfig | None = None,
    params_list: Sequence[torch.Tensor] | None = None,
) -> nn.Module:
    """Load and configure a model with optional pretrained weights, freezing, and compilation.

    Args:
        model: The model to configure.
        device: Device to load the model on.
        pretrained_path: Path to pretrained weights. If None, model is not loaded from checkpoint.
        strip_compiled_prefixes: Whether to strip distributed prefixes from state_dict.
        strip_ddp_prefixes: Whether to strip DDP prefixes from state_dict.
        strip_dataparallel_prefixes: Whether to strip DataParallel prefixes from state_dict.
        freeze: Whether to freeze model parameters (only applies if pretrained_path is provided).
        compile_cfg: Compilation configuration for torch.compile. If None, model is not compiled.
        params_list: List to add model parameters to (only if not frozen and pretrained_path is None).

    Returns:
        The configured model.
    """
    # Move model to device
    model = model.to(device)

    # Load pretrained weights if provided
    if pretrained_path is not None:
        state_dict: dict[str, CheckpointValue] = torch.load(
            pretrained_path, map_location=str(device), weights_only=True
        )
        if strip_compiled_prefixes or strip_ddp_prefixes or strip_dataparallel_prefixes:
            # Strip redundant prefixes from state_dict
            state_dict = strip_pytorch_redundant_strings(
                model_data=state_dict,
                strip_dataparallel=strip_dataparallel_prefixes,
                strip_ddp=strip_ddp_prefixes,
                strip_compiled=strip_compiled_prefixes,
                in_place=True,
            )
        model.load_state_dict(state_dict)

        params: list[Parameter]
        if freeze:
            # Freeze parameters
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        else:
            model.train()
            if params_list is not None:
                params = list(model.parameters())
                if isinstance(params_list, list):
                    params_list.extend(params)
    else:
        # No pretrained weights - model should be trainable
        model.train()
        if params_list is not None:
            params = list(model.parameters())
            if isinstance(params_list, list):
                params_list.extend(params)

    # Apply compilation if configured
    if compile_cfg is not None:
        model.compile(
            fullgraph=compile_cfg.fullgraph,
            dynamic=compile_cfg.dynamic,
            backend=compile_cfg.backend,
            mode=compile_cfg.mode,
            disable=compile_cfg.disable,
        )

    return model
