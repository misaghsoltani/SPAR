from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn.modules.batchnorm import _BatchNorm as BatchNormType
    from torch.types import Device


class FrozenAffineNd(nn.Module):
    """Drop-in replacement for eval-mode BatchNorm{1,2,3}d or SyncBatchNorm.

    Computes y = x * scale + bias with per-channel parameters.

    Attributes:
        scale (Tensor): Per-channel scale buffer, shaped for broadcasting.
        bias (Tensor): Per-channel bias buffer, shaped for broadcasting.
    """

    scale: Tensor
    bias: Tensor

    def __init__(self, num_features: int, dim: int, dtype: torch.dtype | None = None, device: Device = None) -> None:
        super().__init__()
        assert dim in {1, 2, 3}
        shape: list[int] = [1, num_features] + [1] * dim  # N C (D) (H) (W)
        self.scale: Tensor
        self.bias: Tensor
        self.register_buffer("scale", torch.ones(shape, dtype=dtype, device=device))
        self.register_buffer("bias", torch.zeros(shape, dtype=dtype, device=device))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the frozen affine transform to the input.

        Args:
            x (Tensor): Input tensor. Can be channels-first or channels-last.

        Returns:
            Tensor: Transformed tensor (x * scale + bias).
        """
        return x * self.scale + self.bias

    @staticmethod
    def from_batchnorm(bn: BatchNormType) -> FrozenAffineNd:
        """Create a FrozenAffineNd from a BatchNorm module.

        Args:
            bn (nn.modules.batchnorm._BatchNorm): Instance of BatchNorm1d,
                BatchNorm2d, BatchNorm3d or SyncBatchNorm. The function reads
                `num_features`, `eps`, optional affine parameters (`weight`,
                `bias`) and running statistics (`running_mean`, `running_var`) to
                construct an equivalent frozen affine transform that matches
                eval-mode BatchNorm behaviour.

        Returns:
            FrozenAffineNd: A module performing the same affine transform as
                the provided batchnorm in evaluation mode.
        """
        # bn is BatchNorm1d/2d/3d or SyncBatchNorm (all have same attributes)
        assert hasattr(bn, "num_features")
        assert hasattr(bn, "eps")
        dim: int = (
            1
            if isinstance(bn, nn.modules.batchnorm.BatchNorm1d)
            else 2
            if isinstance(bn, nn.modules.batchnorm.BatchNorm2d)
            else 3
            if isinstance(bn, nn.modules.batchnorm.BatchNorm3d)
            else 2
        )  # best-effort
        dtype: torch.dtype | None = None
        device: Device = None

        # Prefer parameter dtype/device (if affine) else buffers.
        weight: nn.Parameter = bn.weight
        running_mean_buf: Tensor | None = bn.running_mean
        if bn.affine:
            # weight is a Parameter when present
            assert isinstance(weight, nn.Parameter)
            dtype = weight.dtype
            device = weight.device
        elif running_mean_buf is not None:
            assert isinstance(running_mean_buf, torch.Tensor)
            dtype = running_mean_buf.dtype
            device = running_mean_buf.device

        num_features_int: int = bn.num_features
        out: FrozenAffineNd = FrozenAffineNd(num_features_int, dim, dtype=dtype, device=device)

        # Handle affine flag and defaults
        gamma: Tensor
        beta: Tensor
        bias_param: Tensor | None = bn.bias
        if bn.affine:
            assert isinstance(weight, nn.Parameter)
            assert isinstance(bias_param, nn.Parameter)
            gamma = weight.detach()
            beta = bias_param.detach()
        else:
            gamma = torch.ones(num_features_int, dtype=dtype, device=device)
            beta = torch.zeros(num_features_int, dtype=dtype, device=device)

        # BatchNorm running statistics are tensor buffers with defaults of 0 and 1.
        running_mean_buf2: Tensor | None = bn.running_mean
        running_var_buf2: Tensor | None = bn.running_var
        assert isinstance(running_mean_buf2, torch.Tensor)
        assert isinstance(running_var_buf2, torch.Tensor)
        running_mean: Tensor = running_mean_buf2.detach()
        running_var: Tensor = running_var_buf2.detach()

        # Compute scale and bias in the BatchNorm parameter dtype.
        denom: Tensor = torch.sqrt(running_var + bn.eps)
        scale_1d: Tensor = gamma / denom
        bias_1d: Tensor = beta - running_mean * scale_1d

        # Reshape to NCHW/NDH... broadcastable form
        view_shape: list[int] = [1, num_features_int] + [1] * dim
        out.scale.copy_(scale_1d.view(*view_shape))
        out.bias.copy_(bias_1d.view(*view_shape))
        return out


def _convert_bn_module(module: nn.Module) -> nn.Module:
    """Recursively replace BatchNorm modules with FrozenAffineNd.

    The function traverses the module tree and replaces any BatchNorm/Sync
    BatchNorm instances with an equivalent FrozenAffineNd computed from the
    batchnorm's parameters and running statistics. Replacement is performed
    on the parent module using ``add_module`` when a child is replaced.

    Args:
        module (nn.Module): Module to transform.

    Returns:
        nn.Module: The transformed module. If `module` itself is a BatchNorm
        instance, a new FrozenAffineNd is returned. Otherwise, `module` is
        returned after transforming its children in-place.
    """
    bn_types: tuple[type[BatchNormType], ...] = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    # Replace the module itself if it is a BN
    if isinstance(module, bn_types):
        return FrozenAffineNd.from_batchnorm(module)

    # Otherwise, recurse into children and replace in-place
    new_child: nn.Module
    for name, child in list(module.named_children()):
        new_child = _convert_bn_module(child)  # recurse
        if new_child is not child:
            module.add_module(name, new_child)
    return module


def make_half_compatible(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Replace BatchNorm modules with frozen affine transforms.

    Stored BatchNorm statistics and affine parameters are folded into
    :class:`FrozenAffineNd` before float16 or bfloat16 inference. Other layer
    types are left unchanged.

    Args:
        model (nn.Module): Source model. Should be a pretrained module.
        inplace (bool): If True, transform `model` in-place. If False (the default),
            a shallow deep-copy of the model is created and transformed.

    Returns:
        nn.Module: The transformed module, ready for inference in reduced precision.
            If `inplace` is False this is a deep-copied instance.
    """
    if not inplace:
        # shallow copy of state + structure via torch.nn.Module deepcopy
        model = copy.deepcopy(model)

    model.eval()  # Transformation is only correct for eval BN semantics
    _convert_bn_module(model)
    return model
