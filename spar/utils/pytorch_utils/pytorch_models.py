# from collections.abc import Callable

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from torch import Tensor
    from torch.autograd.function import FunctionCtx


PaddingMode: TypeAlias = Literal["zeros", "reflect", "replicate", "circular"]


def _weight_norm_init(weight: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Initialize weight normalization parameters from an existing weight tensor."""
    with torch.no_grad():
        weight_v: Tensor = weight.detach().clone()
    weight_moved: Tensor = weight_v.movedim(dim, 0)
    flat: Tensor = weight_moved.reshape(weight_moved.shape[0], -1)
    weight_g: Tensor = flat.norm(dim=1)
    return weight_v, weight_g


def _weight_norm_weight(weight_v: Tensor, weight_g: Tensor, dim: int) -> Tensor:
    """Compute the weight tensor from weight normalization parameters."""
    eps: float = torch.finfo(weight_v.dtype).eps
    weight_moved: Tensor = weight_v.movedim(dim, 0)
    flat: Tensor = weight_moved.reshape(weight_moved.shape[0], -1)
    denom: Tensor = flat.norm(dim=1).clamp_min(eps)
    scale: Tensor = (weight_g / denom).view(-1, *([1] * (weight_moved.dim() - 1)))
    normalized: Tensor = weight_moved * scale
    return normalized.movedim(0, dim)


class WeightNormalizedLinear(nn.Module):
    """Linear layer that performs weight normalization inside the compiled graph."""

    bias: Tensor | None
    weight_v: Tensor
    weight_g: Tensor

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True) -> None:
        super().__init__()
        base: nn.Linear = nn.Linear(in_features, out_features, bias=bias)
        weight_v, weight_g = _weight_norm_init(base.weight, dim=0)
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.register_parameter("weight_v", nn.Parameter(weight_v))
        self.register_parameter("weight_g", nn.Parameter(weight_g))
        if bias is False:
            self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", nn.Parameter(base.bias.detach().clone()))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the linear transformation with weight normalization."""
        return F.linear(x, self.weight, self.bias)

    @property
    def weight(self) -> Tensor:
        """Compute the weight tensor from weight normalization parameters."""
        return _weight_norm_weight(self.weight_v, self.weight_g, dim=0)


class WeightNormalizedConv2d(nn.Conv2d):
    """nn.Conv2d variant with weight normalization kept purely in tensor ops."""

    weight_v: Tensor
    weight_g: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "zeros",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        # initialize from the underlying Parameter created by super().__init__
        # use the attribute (Tensor) rather than a method call
        weight_v, weight_g = _weight_norm_init(self.weight, dim=0)
        del self._parameters["weight"]
        self.__dict__.pop("weight", None)
        self.register_parameter("weight_v", nn.Parameter(weight_v))
        self.register_parameter("weight_g", nn.Parameter(weight_g))

    def forward(self, input: Tensor) -> Tensor:
        """Perform the convolution using weight normalization."""
        return self._conv_forward(input, self.weight, self.bias)

    @property
    def weight(self) -> Tensor:
        """Compute the weight tensor from weight normalization parameters.

        Exposed as a property to match the base class's attribute shape/type.
        """
        return _weight_norm_weight(self.weight_v, self.weight_g, dim=0)

    @weight.setter
    def weight(self, val: Tensor) -> None:
        """Set the layer weight by updating internal weight_v and weight_g.

        This attempts an in-place copy to preserve Parameter objects. If the
        provided tensor has a different shape, the parameters are re-registered.
        """
        weight_v_new, weight_g_new = _weight_norm_init(val, dim=0)
        try:
            # try in-place copy to keep Parameter identity
            with torch.no_grad():
                self.weight_v.copy_(weight_v_new)
                self.weight_g.copy_(weight_g_new)
        except Exception:
            # fallback: replace Parameters entirely
            if "weight_v" in self._parameters:
                del self._parameters["weight_v"]
            if "weight_g" in self._parameters:
                del self._parameters["weight_g"]
            self.register_parameter("weight_v", nn.Parameter(weight_v_new))
            self.register_parameter("weight_g", nn.Parameter(weight_g_new))


def _to_list(v: int | tuple[int, ...]) -> list[int]:
    """Convert an int or tuple of ints to a list of ints."""
    return [v] if isinstance(v, int) else list(v)


class WeightNormalizedConvTranspose2d(nn.ConvTranspose2d):
    """nn.ConvTranspose2d variant with weight normalization inline."""

    weight_v: Tensor
    weight_g: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
        padding_mode: PaddingMode = "zeros",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        # initialize from the underlying Parameter created by super().__init__
        # use the attribute (Tensor) rather than a method call
        weight_v, weight_g = _weight_norm_init(self.weight, dim=0)
        del self._parameters["weight"]
        self.__dict__.pop("weight", None)
        self.register_parameter("weight_v", nn.Parameter(weight_v))
        self.register_parameter("weight_g", nn.Parameter(weight_g))

    def forward(self, input: Tensor, output_size: Sequence[int] | None = None) -> Tensor:
        """Perform the convolution using weight normalization."""
        output_padding: tuple[int, ...]
        if output_size is None:
            output_padding = self.output_padding
        else:
            # Convert string padding to zero and normalize spatial arguments to lists.
            pad_arg: tuple[int, ...] | Literal[0] = 0 if isinstance(self.padding, str) else self.padding

            num_spatial_dims: int = len(self.kernel_size)
            output_padding_list: list[int] = self._output_padding(
                input=input,
                output_size=list(output_size),
                stride=_to_list(self.stride),
                padding=_to_list(pad_arg),
                kernel_size=_to_list(self.kernel_size),
                num_spatial_dims=num_spatial_dims,
                dilation=_to_list(self.dilation),
            )
            # conv_transpose2d requires a tuple for output_padding.
            output_padding = tuple(output_padding_list)
        pad_for_call: tuple[int, ...] | Literal[0] = 0 if isinstance(self.padding, str) else self.padding
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, pad_for_call, output_padding, self.groups, self.dilation
        )

    @property
    def weight(self) -> Tensor:
        """Compute the weight tensor from weight normalization parameters.

        Exposed as a property to match the base class's attribute shape/type.
        """
        return _weight_norm_weight(self.weight_v, self.weight_g, dim=0)

    @weight.setter
    def weight(self, val: Tensor) -> None:
        """Set the layer weight by updating internal weight_v and weight_g.

        This attempts an in-place copy to preserve Parameter objects. If the
        provided tensor has a different shape, the parameters are re-registered.
        """
        weight_v_new, weight_g_new = _weight_norm_init(val, dim=0)
        try:
            with torch.no_grad():
                self.weight_v.copy_(weight_v_new)
                self.weight_g.copy_(weight_g_new)
        except Exception:
            if "weight_v" in self._parameters:
                del self._parameters["weight_v"]
            if "weight_g" in self._parameters:
                del self._parameters["weight_g"]
            self.register_parameter("weight_v", nn.Parameter(weight_v_new))
            self.register_parameter("weight_g", nn.Parameter(weight_g_new))


def ste_threshold(x: Tensor, threshold: float = 0.5) -> Tensor:
    """Straight-through estimator binarization that stays fully in graph."""
    discrete: Tensor = torch.as_tensor(x.gt(threshold), dtype=x.dtype)
    return x + (discrete - x).detach()


class RoundSTEFn(torch.autograd.Function):
    """Shared helper for Straight-Through Estimator (STE) rounding."""

    @staticmethod
    def forward(_ctx: FunctionCtx, x: Tensor) -> Tensor:
        """Forward pass that rounds the input tensor.

        Forward: same value as x + (x.round() - x).detach()
        because x + (round(x) - x) = round(x)
        """
        return x.round()

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: Tensor | None) -> tuple[Tensor | None, ...]:
        """Backward pass that uses identity for gradient (STE)."""
        del ctx
        return grad_outputs


class STEThresh(nn.Module):
    """Layer that applies Straight-Through Estimator (STE) binarization.

    This layer binarizes the input tensor using a threshold, and uses the STE trick for
    backpropagation to allow gradient flow.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize STEThresh layer.

        This layer applies the Straight-Through Estimator (STE) for discretization.

        Args:
            threshold (float, optional): Threshold for discretization. Defaults to 0.5.
        """
        super().__init__()
        self.threshold: float = threshold
        self.RoundSTEFn: type[RoundSTEFn] = RoundSTEFn

    def forward(self, x: Tensor) -> Tensor:
        """Apply STE binarization using the shared helper."""
        rounded = self.RoundSTEFn.apply(x)
        if not isinstance(rounded, torch.Tensor):
            raise TypeError("RoundSTEFn.apply returned a non-tensor result")
        return rounded


class SPLASH(nn.Module):
    """Spline-based activation with learnable hinge points."""

    def __init__(self, num_hinges: int = 5, init: str = "RELU") -> None:
        super().__init__()
        msg: str
        if num_hinges <= 0:
            msg = f"Number of hinges must be > 0 (got {num_hinges})"
            raise ValueError(msg)
        if ((num_hinges + 1) % 2) != 0:
            msg = f"Number of hinges must be odd (got {num_hinges})"
            raise ValueError(msg)
        init = init.upper()

        # how many per side
        self.num_each_side: int = (num_hinges + 1) // 2

        # these hinge locations never change, so register them as a buffer
        hinge_vals: Tensor = torch.linspace(0.0, 2.5, self.num_each_side)
        self.hinges: Tensor
        self.register_buffer("hinges", hinge_vals)

        # bias
        self.output_bias: nn.Parameter = nn.Parameter(torch.zeros(1), requires_grad=True)

        # initialize coeffs on right and left
        # right: [1, 0, 0, ...], left: either [0, ...] or [-1, 0, ...]
        ones: Tensor = torch.ones(1)
        zeros: Tensor = torch.zeros(self.num_each_side - 1)

        self.coeffs_right: nn.Parameter = nn.Parameter(torch.cat([ones, zeros]), requires_grad=True)
        self.coeffs_left: nn.Parameter
        if init == "RELU":
            self.coeffs_left = nn.Parameter(torch.zeros(self.num_each_side), requires_grad=True)
        elif init == "LINEAR":
            self.coeffs_left = nn.Parameter(torch.cat([-ones, zeros]), requires_grad=True)
        else:
            msg = f"Unknown init {init}"
            raise ValueError(msg)

    def forward(self, x: Tensor) -> Tensor:
        """Compute activation using hinge-based piecewise linear function."""
        # reshape hinges and coeffs so they broadcast against x
        view_shape: Sequence[int] = [self.num_each_side] + [1] * x.dim()
        h: Tensor = self.hinges.to(dtype=x.dtype, device=x.device).reshape(view_shape)  # shape [H, 1,1,...]
        cr: Tensor = self.coeffs_right.reshape(view_shape)  # same shape
        cl: Tensor = self.coeffs_left.reshape(view_shape)

        # expand x to shape [H, *x.shape]
        x_expanded: Tensor = x.unsqueeze(0)

        # pos = sum_i coeffs_right[i] * relu(x - hinge[i])
        pos: Tensor = (cr * F.relu(x_expanded - h)).sum(dim=0)

        # neg = sum_i coeffs_left[i] * relu(-x - hinge[i])
        neg: Tensor = (cl * F.relu(-x_expanded - h)).sum(dim=0)

        return pos + neg + self.output_bias


# Map activation function names to their classes
_ACT_FN_MAP: dict[str, type[nn.Module]] = {
    "RELU": nn.ReLU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "SOFTMAX": nn.Softmax,
    "SIGMOID": nn.Sigmoid,
    "TANH": nn.Tanh,
    "LRELU": nn.LeakyReLU,
    "LINEAR": nn.Identity,
    "SPLASH": SPLASH,
    "STE_THRESH": STEThresh,
}


def get_act_fn(act: str) -> nn.Module:
    """Return the activation function based on the given string.

    Args:
        act (str): Activation function name.

    Returns:
        nn.Module: Corresponding activation function.

    Raises:
        ValueError: If the activation type is undefined.

    """
    act = act.upper()
    if act in _ACT_FN_MAP:
        return _ACT_FN_MAP[act]()

    raise ValueError(f"Undefined activation type {act}")


class FullyConnectedModel(nn.Module):
    """Feed-forward fully connected network."""

    def __init__(
        self,
        input_dim: int,
        layer_dims: Sequence[int],
        layer_batch_norms: Sequence[bool],
        layer_acts: Sequence[str],
        *,
        weight_norms: Sequence[bool] | None = None,
        layer_norms: Sequence[bool] | None = None,
        dropouts: Sequence[float] | None = None,
        use_bias_with_norm: bool = True,
    ) -> None:
        super().__init__()

        dims: Sequence[int] = list(layer_dims)
        batch_norms: list[bool] = list(layer_batch_norms)
        acts: list[str] = [act.upper() for act in layer_acts]
        num_layers: int = len(dims)

        if not (len(batch_norms) == len(acts) == num_layers):
            msg = "layer_dims, layer_batch_norms, and layer_acts must share the same length"
            raise ValueError(msg)

        wn_flags: list[bool] = list(weight_norms) if weight_norms is not None else [False] * num_layers
        ln_flags: list[bool] = list(layer_norms) if layer_norms is not None else [False] * num_layers
        dropout_rates: list[float] = list(dropouts) if dropouts is not None else [0.0] * num_layers

        if len(wn_flags) != num_layers:
            msg = "weight_norms must match layer_dims length"
            raise ValueError(msg)
        if len(ln_flags) != num_layers:
            msg = "layer_norms must match layer_dims length"
            raise ValueError(msg)
        if len(dropout_rates) != num_layers:
            msg = "dropouts must match layer_dims length"
            raise ValueError(msg)

        blocks: list[nn.Sequential] = []
        prev_dim: int = input_dim

        for dim, bn, act, wn, ln, do in zip(dims, batch_norms, acts, wn_flags, ln_flags, dropout_rates, strict=True):
            seq: nn.ModuleList = nn.ModuleList()

            # Linear
            use_bias: bool = use_bias_with_norm or (not bn and not ln)
            linear: nn.Module
            if wn:
                linear = WeightNormalizedLinear(prev_dim, dim, bias=use_bias)
            else:
                linear = nn.Linear(prev_dim, dim, bias=use_bias)
            seq.append(linear)

            # Optional LayerNorm or BatchNorm
            if ln:
                seq.append(nn.LayerNorm(dim))
            if bn:
                seq.append(nn.BatchNorm1d(dim))

            # Activation
            seq.append(get_act_fn(act))

            # In-place Dropout
            if do > 0.0:
                seq.append(nn.Dropout(do))

            blocks.append(nn.Sequential(*seq))
            prev_dim = dim

        self.layers: nn.ModuleList = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for fully connected layers."""
        for block in self.layers:
            x = block(x)
        return x


class ResidualFCBlock(nn.Module):
    """Single residual block.

    Two FC layers with skip-connection + activation.
    """

    def __init__(self, dim: int, batch_norm: bool, act: str, use_bias_with_norm: bool = True) -> None:
        super().__init__()
        self.net: nn.Module = FullyConnectedModel(
            input_dim=dim,
            layer_dims=[dim, dim],
            layer_batch_norms=[batch_norm, batch_norm],
            layer_acts=[act, "LINEAR"],
            use_bias_with_norm=use_bias_with_norm,
        )
        self.act: nn.Module = get_act_fn(act)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for a residual fully connected block."""
        res: Tensor = x
        out: Tensor = self.net(x)
        return self.act(out + res)


class ResnetModel(nn.Module):
    """Residual fully connected network."""

    def __init__(
        self,
        resnet_dim: int,
        num_resnet_blocks: int,
        out_dim: int,
        batch_norm: bool,
        *,
        act: str,
        use_bias_with_norm: bool = True,
    ) -> None:
        super().__init__()

        self.blocks: nn.Sequential = nn.Sequential(*[
            ResidualFCBlock(resnet_dim, batch_norm, act, use_bias_with_norm) for _ in range(num_resnet_blocks)
        ])
        self.fc_out: nn.Module = nn.Linear(resnet_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward input through residual blocks and final linear layer."""
        x = self.blocks(x)
        return self.fc_out(x)


class Conv2dModel(nn.Module):
    """Configurable 2D convolutional neural network model."""

    def __init__(
        self,
        chan_in: int,
        channel_sizes: Sequence[int],
        kernel_sizes: Sequence[int | tuple[int, int]],
        paddings: Sequence[int | tuple[int, int] | str],
        *,
        layer_batch_norms: Sequence[bool],
        layer_acts: Sequence[str],
        strides: Sequence[int | tuple[int, int]] | None = None,
        transpose: bool = False,
        weight_norms: Sequence[bool] | None = None,
        poolings: Sequence[str | None] | None = None,
        dropouts: Sequence[float] | None = None,
        padding_modes: Sequence[str] | None = None,
        padding_values: Sequence[int | float] | None = None,
        group_norms: Sequence[int] | None = None,
        use_bias_with_norm: bool = True,
    ) -> None:
        super().__init__()

        channels: Sequence[int] = list(channel_sizes)
        kernels: list[int | tuple[int, int]] = list(kernel_sizes)
        pads: list[int | tuple[int, int] | str] = list(paddings)
        batch_norms: list[bool] = list(layer_batch_norms)
        acts: list[str] = [act.upper() for act in layer_acts]
        num_layers: int = len(channels)

        if not (len(kernels) == len(pads) == len(batch_norms) == len(acts) == num_layers):
            msg = "channel_sizes, kernel_sizes, paddings, layer_batch_norms, and layer_acts must match in length"
            raise ValueError(msg)

        stride_list: list[int | tuple[int, int]] = list(strides) if strides is not None else [1] * num_layers
        weight_norm_list: list[bool] = list(weight_norms) if weight_norms is not None else [False] * num_layers
        dropout_rates: list[float] = list(dropouts) if dropouts is not None else [0.0] * num_layers
        group_norm_list: Sequence[int] = list(group_norms) if group_norms is not None else [0] * num_layers
        padding_mode_list: list[str] = list(padding_modes) if padding_modes is not None else ["zeros"] * num_layers
        padding_value_list: list[int | float] = (
            list(padding_values) if padding_values is not None else [0.0] * num_layers
        )
        pooling_list: list[str | None] = list(poolings) if poolings is not None else [None] * num_layers

        if not (
            len(stride_list)
            == len(weight_norm_list)
            == len(dropout_rates)
            == len(group_norm_list)
            == len(padding_mode_list)
            == len(padding_value_list)
            == len(pooling_list)
            == num_layers
        ):
            msg = "All per-layer configuration lists must align with channel_sizes length"
            raise ValueError(msg)

        layers: list[nn.Sequential] = []
        ci: int = chan_in

        for co, k, p, bn, act, st, wn, do, pm, pv, gn, pool in zip(
            channels,
            kernels,
            pads,
            batch_norms,
            acts,
            stride_list,
            weight_norm_list,
            dropout_rates,
            padding_mode_list,
            padding_value_list,
            group_norm_list,
            pooling_list,
            strict=True,
        ):
            block: nn.ModuleList = nn.ModuleList()

            pad_mode: PaddingMode
            pad: int | tuple[int, int]
            const_pad: int | tuple[int, int, int, int]
            if pm == "constant":
                # Support int or tuple padding, ConstantPad2d expects (left, right, top, bottom)
                if isinstance(p, int):
                    const_pad = p
                else:
                    # p is Tuple[H, W] (same convention as conv2d), convert to (left, right, top, bottom)
                    ph: int
                    pw: int
                    ph, pw = p if isinstance(p, tuple) else tuple(p)
                    const_pad = (pw, pw, ph, ph)
                block.append(nn.ConstantPad2d(const_pad, pv))
                pad, pad_mode = 0, "zeros"
            else:
                # Normalize "none" -> "zeros" and validate against allowed modes
                pm_norm: str = "zeros" if pm == "none" else pm
                if pm_norm not in {"zeros", "reflect", "replicate", "circular"}:
                    raise ValueError(
                        f"Invalid padding_mode '{pm_norm}'. Expected one of: "
                        "zeros, reflect, replicate, circular, or use 'constant'/'none'."
                    )
                if pm_norm == "reflect":
                    pad_mode = "reflect"
                elif pm_norm == "replicate":
                    pad_mode = "replicate"
                elif pm_norm == "circular":
                    pad_mode = "circular"
                else:
                    pad_mode = "zeros"
                pad = p

            # Conv or conv-transpose
            use_bias: bool = use_bias_with_norm or (not bn and gn == 0)
            conv: nn.Module
            if transpose:
                if wn:
                    conv = WeightNormalizedConvTranspose2d(
                        ci, co, k, stride=st, padding=pad, padding_mode=pad_mode, bias=use_bias
                    )
                else:
                    conv = nn.ConvTranspose2d(ci, co, k, stride=st, padding=pad, padding_mode=pad_mode, bias=use_bias)
            elif wn:
                conv = WeightNormalizedConv2d(ci, co, k, stride=st, padding=pad, padding_mode=pad_mode, bias=use_bias)
            else:
                conv = nn.Conv2d(ci, co, k, stride=st, padding=pad, padding_mode=pad_mode, bias=use_bias)

            block.append(conv)

            # Normalization
            if bn:
                block.append(nn.BatchNorm2d(co))
            elif gn > 0:
                assert co % gn == 0, f"{co} channels not divisible by {gn} groups"
                block.append(nn.GroupNorm(gn, co))

            # activation
            block.append(get_act_fn(act))

            # dropout
            if do > 0.0:
                block.append(nn.Dropout(do))

            # optional pooling
            if not transpose and pool == "avg":
                block.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
            elif transpose and pool == "max":
                block.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))

            layers.append(nn.Sequential(*block))
            ci = co

        self.layers: nn.ModuleList = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        """Pass input through the convolutional layers."""
        for block in self.layers:
            x = block(x)
        return x


class ResnetConv2dModel(nn.Module):
    """Residual 2D convolutional network with optional channel- and shape-matching."""

    def __init__(
        self,
        in_channels: int,
        resnet_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int | str,
        num_resnet_blocks: int,
        batch_norm: bool,
        act: str,
        group_norm: int | None = 0,
        use_bias_with_norm: bool = True,
    ) -> None:
        super().__init__()

        self.act_fn: nn.Module = get_act_fn(act)
        self.blocks: nn.ModuleList = nn.ModuleList()

        self.needs_shape_match: bool = in_channels != resnet_channels
        self.needs_downsample: bool = resnet_channels != out_channels

        # Convert None group_norm to 0 for consistency
        gn: int = group_norm if group_norm is not None else 0

        # first-layer to match channel dims if needed
        self.first_layer: nn.Module | None
        if self.needs_shape_match:
            self.first_layer = Conv2dModel(
                chan_in=in_channels,
                channel_sizes=[resnet_channels],
                kernel_sizes=[1],
                paddings=[0],
                layer_batch_norms=[False],
                layer_acts=["RELU"],
                group_norms=[gn],
                use_bias_with_norm=use_bias_with_norm,
            )
        else:
            self.first_layer = None

        # determine padding per conv in block
        pads: list[int | tuple[int, int] | str] = [1, 0] if kernel_size == 2 and padding == "same" else [padding] * 2

        # build residual blocks
        block: nn.Module
        for _ in range(num_resnet_blocks):
            block = Conv2dModel(
                chan_in=resnet_channels,
                channel_sizes=[resnet_channels, resnet_channels],
                kernel_sizes=[kernel_size, kernel_size],
                paddings=pads,
                layer_batch_norms=[batch_norm, batch_norm],
                layer_acts=[act, "LINEAR"],
                group_norms=[gn, gn],
                use_bias_with_norm=use_bias_with_norm,
            )
            self.blocks.append(block)

        # downsample at end if channel dims differ
        self.downsample: nn.Module | None
        if self.needs_downsample:
            self.downsample = Conv2dModel(
                chan_in=resnet_channels,
                channel_sizes=[out_channels],
                kernel_sizes=[1],
                paddings=[0],
                layer_batch_norms=[False],
                layer_acts=["RELU"],
                group_norms=[gn],
                use_bias_with_norm=use_bias_with_norm,
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through residual convolutional blocks."""
        if self.first_layer is not None:
            x = self.first_layer(x)

        res: Tensor
        for block in self.blocks:
            res = x
            x = block(x)
            # elementwise add + activation
            x = self.act_fn(x + res)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class UnflattenWrapper(nn.Module):
    """Wrapper for torch.nn.Unflatten that handles list to tuple conversion for Hydra compatibility."""

    def __init__(self, dim: int, unflattened_size: Sequence[int] | tuple[int, ...]) -> None:
        super().__init__()
        # Convert ListConfig, list, or tuple to tuple
        self.unflatten: nn.Unflatten = nn.Unflatten(dim, tuple(unflattened_size))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through unflatten layer."""
        return self.unflatten(x)
