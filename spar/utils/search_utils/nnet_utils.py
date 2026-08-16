from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from logging import getLogger
import os
from typing import TYPE_CHECKING

import numpy as np
from numpy import float32
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger
    from types import FunctionType
    from typing import TypeAlias

    from numpy.typing import NDArray
    from torch import Tensor, device as TorchDevice
    from torch.cuda.streams import Stream

    CallableModelType: TypeAlias = Callable[[NDArray[float32], NDArray[float32]], NDArray[float32]] | FunctionType


def apply_cuda_tf32_settings(
    allow_tf32: bool, float32_matmul_precision: str | None, logger: Logger | None = None
) -> None:
    """Apply YAML-backed CUDA TF32 settings, leaving PyTorch defaults unchanged by default."""
    if not torch.cuda.is_available():
        return

    log: Logger = logger if logger is not None else getLogger(__name__)
    if allow_tf32:
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            cudnn_conv_backend = getattr(torch.backends.cudnn, "conv", None)
            if cudnn_conv_backend is not None:
                cudnn_conv_backend.fp32_precision = "tf32"
            else:
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                log.warning("Failed to apply TF32 settings. Continuing...")

    if float32_matmul_precision is not None:
        try:
            torch.set_float32_matmul_precision(float32_matmul_precision)
        except Exception:
            log.warning("Failed to set float32 matmul precision. Continuing...")


def get_device() -> tuple[TorchDevice, list[int], bool]:
    """Return the available compute device, logical device IDs, and a boolean for GPU/accelerator use.

    Order of preference:
      1) CUDA/ROCm (device type 'cuda')
      2) Intel XPU (device type 'xpu')
      3) Apple Metal (device type 'mps')
      4) CPU

    Returns:
        (device, devices, on_gpu):
            device: TorchDevice to place new tensors/models on.
            devices: logical device indices for the selected backend (e.g., [0,1,2,...]).
            on_gpu: True if using any GPU/accelerator backend (CUDA/XPU/MPS), else False.
    """
    device: TorchDevice = torch.device("cpu")
    devices: list[int] = []
    on_gpu: bool = False

    # CUDA / ROCm (device type is still "cuda")
    if torch.cuda.is_available():
        num_visible: int = torch.cuda.device_count()
        if num_visible > 0:
            # Respect torchrun / DDP conventions
            local_rank_str: str | None = os.environ.get("LOCAL_RANK")
            try:
                local_rank: int = int(local_rank_str) if local_rank_str is not None else 0
            except ValueError:
                local_rank = 0
            if not (0 <= local_rank < num_visible):
                local_rank = 0

            # Pin current process to its device
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")

            devices = list(range(num_visible))  # CUDA_VISIBLE_DEVICES may remap them
            on_gpu = True

            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                logger: Logger = getLogger(__name__)
                logger.warning("Failed to enable cuDNN benchmark. Continuing...")

            return device, devices, on_gpu

    # Intel XPU
    # Use guard for older versions of PyTorch
    if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        num_visible = torch.xpu.device_count()
        if num_visible > 0:
            local_rank_str = os.environ.get("LOCAL_RANK")
            try:
                local_rank = int(local_rank_str) if local_rank_str is not None else 0
            except ValueError:
                local_rank = 0
            if not (0 <= local_rank < num_visible):
                local_rank = 0

            torch.xpu.set_device(local_rank)
            device = torch.device(f"xpu:{local_rank}")
            devices = list(range(num_visible))
            on_gpu = True
            return device, devices, on_gpu

    # Apple Metal (MPS)
    # Single logical device today - no device_count API
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        devices = [0]
        on_gpu = True
        return device, devices, on_gpu

    # CPU fallback
    return device, devices, on_gpu


def load_nnet(model_file: str, nnet: nn.Module, device: TorchDevice | None = None) -> nn.Module:
    """Loads a neural network from a file.

    Args:
        model_file (str): Path to the model file.
        nnet (nn.Module): The neural network module to load the state dict into.
        device (torch.device | None): The device to map the model to.

    Returns:
        nn.Module: The loaded neural network.
    """
    if device is None:
        device = torch.device("cpu")

    state_dict: dict[str, Tensor] = torch.load(model_file, map_location=device)

    # Remove common Distributed/DataParallel 'module.' prefix if present
    new_state_dict: OrderedDict[str, Tensor] = OrderedDict({
        (k.removeprefix("module.")): v for k, v in state_dict.items()
    })

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()
    nnet.to(device)

    return nnet


@dataclass(slots=True)
class HeuristicFnWrapper:
    """Callable wrapper around a heuristic nn.Module avoiding nested functions.

    Uses from_numpy + non_blocking transfers to minimize CPU<->GPU overhead.
    """

    nnet: nn.Module
    device: TorchDevice
    clip_zero: bool = False
    batch_size: int | None = None

    @torch.inference_mode()
    def __call__(self, states_np: NDArray[float32], states_goal_np: NDArray[float32]) -> NDArray[float32]:
        """Compute the heuristic cost-to-go from states to goal states with async overlap when on CUDA."""
        self.nnet.eval()
        num_states: int = states_np.shape[0]
        bs: int = self.batch_size or num_states
        use_cuda: bool = self.device.type == "cuda"
        if not use_cuda:
            # Use the non-distributed path on CPU.
            out_list: list[NDArray[float32]] = []
            start: int = 0
            while start < num_states:
                end: int = min(start + bs, num_states)
                sb: Tensor = torch.from_numpy(states_np[start:end]).float().to(self.device)
                gb: Tensor = torch.from_numpy(states_goal_np[start:end]).float().to(self.device)
                out: Tensor = self.nnet(sb, gb)
                out_list.append(out.detach().to("cpu").numpy())
                start = end
            cost_to_go: NDArray[float32] = np.concatenate(out_list, axis=0)
            if self.clip_zero:
                cost_to_go = np.maximum(cost_to_go, 0.0)
            return cost_to_go

        # CUDA path with stream overlap and async D2H copies into pinned CPU
        streams: list[torch.cuda.Stream]
        try:
            streams = [torch.cuda.Stream(device=self.device), torch.cuda.Stream(device=self.device)]
        except Exception:
            streams = [torch.cuda.current_stream(self.device)]

        n_chunks: int = max(1, (num_states + bs - 1) // bs)
        cpu_outputs: list[Tensor] = [torch.empty(0)] * n_chunks

        # Launch H2D + compute + async D2H per chunk on alternating streams
        start = 0
        chunk_idx = 0
        while start < num_states:
            end = min(start + bs, num_states)
            sb_np: NDArray[float32] = states_np[start:end]
            gb_np: NDArray[float32] = states_goal_np[start:end]
            # Host tensors from pinned memory for non_blocking copies
            sb_t: Tensor = torch.from_numpy(sb_np)
            gb_t: Tensor = torch.from_numpy(gb_np)
            try:
                sb_t = sb_t.pin_memory()
                gb_t = gb_t.pin_memory()
            except Exception:
                pass
            stream: torch.cuda.Stream = streams[chunk_idx % len(streams)]
            with torch.cuda.stream(stream):
                sb = sb_t.to(self.device, non_blocking=True).float()
                gb = gb_t.to(self.device, non_blocking=True).float()
                out = self.nnet(sb, gb)
                # Schedule async D2H copy to pinned CPU tensor
                cpu_outputs[chunk_idx] = out.detach().to("cpu", non_blocking=True)
            start = end
            chunk_idx += 1

        # Synchronize before returning timing-sensitive results.
        for s in streams:
            s.synchronize()
        torch.cuda.synchronize(self.device)

        # Concatenate CPU tensors and convert to numpy
        out_cpu: Tensor = torch.cat([t.contiguous() for t in cpu_outputs if t.numel() > 0], dim=0)
        cost_to_go = out_cpu.numpy()
        if self.clip_zero:
            cost_to_go = np.maximum(cost_to_go, 0.0)
        return cost_to_go


@dataclass(slots=True)
class ModelFnWrapper:
    """Callable wrapper around a transition nn.Module avoiding nested functions.

    Uses from_numpy + non_blocking transfers to minimize CPU<->GPU overhead.
    """

    nnet: nn.Module
    device: TorchDevice
    batch_size: int | None = None

    @torch.inference_mode()
    def __call__(self, states_np: NDArray[float32], actions_np: NDArray[float32]) -> NDArray[float32]:
        """Compute next states with async overlap and device-side rounding when on CUDA."""
        self.nnet.eval()
        num_states: int = int(states_np.shape[0])
        bs: int = self.batch_size or num_states
        use_cuda: bool = self.device.type == "cuda"
        start: int
        if not use_cuda:
            out_list: list[NDArray[float32]] = []
            start = 0
            while start < num_states:
                end: int = min(start + bs, num_states)
                sb: Tensor = torch.from_numpy(states_np[start:end]).float().to(self.device)
                ab: Tensor = torch.from_numpy(actions_np[start:end]).float().to(self.device)
                out: Tensor = self.nnet(sb, ab)
                out_np: NDArray[float32] = out.detach().to("cpu").numpy().round().astype(float32, copy=False)
                out_list.append(out_np)
                start = end
            return np.concatenate(out_list, axis=0)

        streams: list[Stream]
        try:
            streams = [torch.cuda.Stream(device=self.device), torch.cuda.Stream(device=self.device)]
        except Exception:
            streams = [torch.cuda.current_stream(self.device)]

        n_chunks: int = max(1, (num_states + bs - 1) // bs)
        cpu_outputs: list[Tensor] = [torch.empty(0)] * n_chunks
        start = 0
        chunk_idx = 0
        while start < num_states:
            end = min(start + bs, num_states)
            sb_np: NDArray[float32] = states_np[start:end]
            ab_np: NDArray[float32] = actions_np[start:end]
            sb_t: Tensor = torch.from_numpy(sb_np)
            ab_t: Tensor = torch.from_numpy(ab_np)
            try:
                sb_t = sb_t.pin_memory()
                ab_t = ab_t.pin_memory()
            except Exception:
                pass
            stream: Stream = streams[chunk_idx % len(streams)]
            with torch.cuda.stream(stream):
                sb = sb_t.to(self.device, non_blocking=True).float()
                ab = ab_t.to(self.device, non_blocking=True).float()
                out = self.nnet(sb, ab)
                out = torch.round(out).to(dtype=torch.float32)
                # Async D2H into pinned CPU tensor for overlap
                cpu_outputs[chunk_idx] = out.detach().to("cpu", non_blocking=True)
            start = end
            chunk_idx += 1

        # Synchronize streams before assembling
        for s in streams:
            s.synchronize()
        torch.cuda.synchronize(self.device)

        out_cpu: Tensor = torch.cat([t.contiguous() for t in cpu_outputs if t.numel() > 0], dim=0)
        return out_cpu.numpy().astype(np.float32, copy=False)


@torch.inference_mode()
def load_heuristic_fn(
    nnet_dir: str,
    device: TorchDevice,
    on_gpu: bool,
    nnet: nn.Module,
    clip_zero: bool = False,
    gpu_num: int = -1,
    batch_size: int | None = None,
) -> CallableModelType:
    """Loads a heuristic function from a neural network.

    Args:
        nnet_dir (str): Directory containing the neural network model.
        device (torch.device): The device to run the computation on.
        on_gpu (bool): Whether to use GPU.
        nnet (nn.Module): The neural network module.
        clip_zero (bool, optional): Whether to clip the cost to zero. Defaults to False.
        gpu_num (int, optional): The GPU number to use. Defaults to -1.
        batch_size (Optional[int], optional): The batch size for processing. Defaults to None.

    Returns:
        Callable[[NDArray, NDArray], NDArray]: The heuristic function.
    """
    if (gpu_num >= 0) and on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    model_file: str = f"{nnet_dir}"  # /model_state_dict.pt"

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    # Enable DataParallel only when multiple CUDA devices are present
    if on_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        nnet = nn.DataParallel(nnet)

    heuristic_fn: CallableModelType = HeuristicFnWrapper(
        nnet=nnet, device=device, clip_zero=clip_zero, batch_size=batch_size
    )

    return heuristic_fn


@torch.inference_mode()
def load_model_fn(
    model_file: str,
    device: TorchDevice,
    on_gpu: bool,
    nnet: nn.Module,
    gpu_num: int = -1,
    batch_size: int | None = None,
) -> CallableModelType:
    """Loads a model function from a neural network.

    Args:
        model_file (str): Path to the model file.
        device (torch.device): The device to run the computation on.
        on_gpu (bool): Whether to use GPU.
        nnet (nn.Module): The neural network module.
        gpu_num (int, optional): The GPU number to use. Defaults to -1.
        batch_size (Optional[int], optional): The batch size for processing. Defaults to None.

    Returns:
        Callable[[NDArray, NDArray], NDArray]: The model function.
    """
    if (gpu_num >= 0) and on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    # Enable DataParallel only when multiple CUDA devices are present
    if on_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        nnet = nn.DataParallel(nnet)

    return ModelFnWrapper(nnet=nnet, device=device, batch_size=batch_size)
