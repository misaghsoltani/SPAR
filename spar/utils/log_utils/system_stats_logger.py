from __future__ import annotations

import contextlib
from contextlib import ContextDecorator
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
from logging import INFO, getLogger
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import distro
import psutil
from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger
    from types import ModuleType, TracebackType

    from GPUtil import GPU
    from psutil._ntuples import pmem, scpufreq, sdiskusage, svmem
    from typing_extensions import Self


md: ModuleType | None
try:
    md = importlib.import_module("importlib.metadata")
except Exception:
    md = None

md_legacy: ModuleType | None
try:
    md_legacy = importlib.import_module("importlib_metadata")
except Exception:
    md_legacy = None

logger: Logger = getLogger(__name__)


def _count_sched_affinity(sched: Callable[[int], set[int]], default: int) -> int:
    try:
        return sum(1 for _ in sched(0))
    except Exception:
        return default


def _count_process_cpu_affinity(default: int) -> int:
    try:
        proc = psutil.Process()
    except Exception:
        return default

    try:
        cpu_aff: Callable[[], list[int]] | None = getattr(proc, "cpu_affinity", None)
    except Exception:
        return default

    if not callable(cpu_aff):
        return default

    try:
        return len(cpu_aff())
    except Exception:
        return default


def _get_available_cpu_count(default: int) -> int:
    sched: Callable[[int], set[int]] | None = getattr(os, "sched_getaffinity", None)
    if callable(sched):
        return _count_sched_affinity(sched, default)

    return _count_process_cpu_affinity(default)


@dataclass(frozen=True)
class OSInfo:
    """Operating system information."""

    system: str
    release: str
    version: str
    distro_name: str | None
    distro_version: str | None
    architecture: str


@dataclass(frozen=True)
class CPUInfo:
    """CPU information."""

    brand: str | None
    architecture: str
    physical_cores: int
    logical_cores: int
    available_cores: int
    max_frequency_mhz: float | None
    utilization_percent: float


@dataclass(frozen=True)
class MemoryInfo:
    """Memory (RAM) information."""

    total_bytes: int
    available_bytes: int
    used_bytes: int
    percent: float


@dataclass(frozen=True)
class DiskInfo:
    """Disk usage information."""

    mount_point: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent: float


@dataclass(frozen=True)
class GPUInfo:
    """GPU information."""

    id: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_util_percent: float
    load_percent: float
    temperature: float | None


@dataclass(frozen=True)
class PythonInfo:
    """Python interpreter information."""

    implementation: str
    version: str
    compiler: str


@dataclass(frozen=True)
class PackageInfo:
    """Python package information."""

    name: str
    version: str


@dataclass(frozen=True)
class ProcessInfo:
    """Process-specific information."""

    pid: int
    memory_rss_mb: float
    memory_vms_mb: float
    memory_percent: float
    cpu_percent: float
    num_threads: int
    open_files: int


@dataclass(frozen=True)
class JobResourceInfo:
    """Job-specific resource allocation and usage."""

    allocated_cpus: int | None
    allocated_memory_gb: float | None
    allocated_gpus: list[int] | None
    process: ProcessInfo


@dataclass(frozen=True)
class RuntimeStats:
    """Runtime statistics focused on job execution."""

    timestamp: float
    elapsed_seconds: float | None
    job_resources: JobResourceInfo
    gpu_usage: list[GPUInfo]  # Only allocated GPUs
    system_summary: str  # Brief system info
    python_info: PythonInfo
    extended_info: ExtendedSystemInfo | None


@dataclass(frozen=True)
class GitInfo:
    """Git repository information."""

    remote: str | None
    commit: str | None
    branch: str | None


@dataclass(frozen=True)
class HostInfo:
    """Host and environment information."""

    hostname: str | None
    user: str | None
    executable: str | None
    working_dir: str | None


@dataclass(frozen=True)
class SlurmInfo:
    """SLURM job information."""

    job_id: str | None
    job_name: str | None
    partition: str | None
    node_list: str | None
    cpus_per_task: str | None
    mem_per_node: str | None
    gpus_on_node: str | None


@dataclass(frozen=True)
class ExtendedSystemInfo:
    """Extended system information including host, git, and SLURM details."""

    host: HostInfo
    git: GitInfo | None
    slurm: SlurmInfo | None
    disk_usage: DiskInfo | None
    total_memory_gb: float | None
    total_cpu_count: int | None
    cuda_version: str | None


def safe_escape(text: str | None) -> str:
    """Safely escape text for Rich markup, handling None and edge cases.

    Args:
        text: The text to escape, can be None.

    Returns:
        str: Safely escaped text, or empty string if input was None.
    """
    if text is None:
        return ""

    try:
        # Convert to string and strip any existing problematic characters
        text_str: str = text.strip()
        # Use Rich's escape function to handle markup characters

        return escape(text_str)

    except Exception:
        # If escaping fails, return a safe fallback

        return text.replace("[", "\\[").replace("]", "\\]")


def get_os_info() -> OSInfo:
    """Gather operating system information.

    Attempts to retrieve system, release, version, distribution name and version,
    and architecture details using the platform and (optionally) distro modules.

    Returns:
        OSInfo: Dataclass containing OS details.
    """
    uname: platform.uname_result = platform.uname()
    d_name: str = distro.name(pretty=False)
    d_ver: str = distro.version(pretty=False)

    return OSInfo(
        system=uname.system,
        release=uname.release,
        version=uname.version,
        distro_name=d_name,
        distro_version=d_ver,
        architecture=platform.machine(),
    )


def get_cpu_info() -> CPUInfo:
    """Gather CPU information including brand, architecture, core counts, frequency, and utilization.

    Returns:
        CPUInfo: CPU details such as brand, architecture, number of physical/logical/available cores,
            maximum frequency in MHz, and current utilization percentage.
    """
    # brand if available
    brand: str | None = None
    try:
        cpuinfo: ModuleType = importlib.import_module("cpuinfo")
        info: dict[str, str] = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw")

    except Exception:
        brand = None

    phys: int = psutil.cpu_count(logical=False) or 1
    logi: int = psutil.cpu_count(logical=True) or phys
    # detect actual allocated cores (HPC/container)
    # sched_getaffinity is not available on all platforms (macOS), so guard it.
    avail: int = _get_available_cpu_count(logi)

    util: float = psutil.cpu_percent(interval=0.0)
    freq: scpufreq = psutil.cpu_freq()
    max_mhz: float | None = freq.max if freq and freq.max else None

    return CPUInfo(
        brand=brand,
        architecture=platform.processor() or platform.machine(),
        physical_cores=phys,
        logical_cores=logi,
        available_cores=avail,
        max_frequency_mhz=max_mhz,
        utilization_percent=util,
    )


def get_memory_info() -> MemoryInfo:
    """Gather memory (RAM) usage statistics.

    Returns:
        MemoryInfo: Dataclass containing total, available, used memory in bytes and percent used.
    """
    m: svmem = psutil.virtual_memory()

    return MemoryInfo(total_bytes=m.total, available_bytes=m.available, used_bytes=m.used, percent=m.percent)


def get_disk_info() -> DiskInfo:
    """Gather disk usage statistics for the root filesystem.

    Returns:
        DiskInfo: Dataclass containing mount point, total, used, free bytes and percent used.
    """
    root: str = str(pathlib.Path(os.sep).resolve())
    d: sdiskusage = psutil.disk_usage(root)

    return DiskInfo(mount_point=root, total_bytes=d.total, used_bytes=d.used, free_bytes=d.free, percent=d.percent)


def get_gpu_info() -> list[GPUInfo]:
    """Gather NVIDIA GPU statistics if available.

    Returns:
        list[GPUInfo]: List of GPUInfo dataclasses for each detected GPU, or empty list if none found.
    """
    try:
        gputil: ModuleType = importlib.import_module("GPUtil")
        gpus = gputil.getGPUs()
    except Exception:
        return []

    out: list[GPUInfo] = []
    for g in gpus:
        temp: int | None = getattr(g, "temperature", None)
        out.append(
            GPUInfo(
                id=g.id,
                name=g.name,
                memory_total_mb=g.memoryTotal,
                memory_used_mb=g.memoryUsed,
                memory_util_percent=round(g.memoryUtil * 100, 1),
                load_percent=round(g.load * 100, 1),
                temperature=temp,
            )
        )

    return out


def get_python_info() -> PythonInfo:
    """Gather Python interpreter information.

    Returns:
        PythonInfo: Dataclass containing implementation, version, and compiler details.
    """
    return PythonInfo(
        implementation=platform.python_implementation(),
        version=platform.python_version(),
        compiler=platform.python_compiler(),
    )


def get_package_info(names: list[str]) -> list[PackageInfo]:
    """Get version information for a list of Python packages.

    Args:
        names (list[str]): List of package names to query.

    Returns:
        list[PackageInfo]: List of PackageInfo dataclasses with name and version for each package.
    """
    pkg_list: list[PackageInfo] = []
    for nm in names:
        ver: str = "unknown"
        try:
            if md is not None:
                ver = md.version(nm)
            elif md_legacy is not None:
                ver = md_legacy.version(nm)
        except Exception:
            ver = "unknown"
        pkg_list.append(PackageInfo(name=nm, version=ver))

    return pkg_list


def _fallback_process_info() -> ProcessInfo:
    return ProcessInfo(
        pid=os.getpid(),
        memory_rss_mb=0.0,
        memory_vms_mb=0.0,
        memory_percent=0.0,
        cpu_percent=0.0,
        num_threads=1,
        open_files=0,
    )


def _count_open_files(process: psutil.Process) -> int:
    try:
        open_files = process.open_files()
    except Exception:
        return 0

    try:
        return len(open_files)
    except Exception:
        return 0


def get_process_info() -> ProcessInfo:
    """Get information about the current process."""
    try:
        process: psutil.Process = psutil.Process()
        mem_info: pmem = process.memory_info()
    except Exception:
        return _fallback_process_info()

    open_files_count: int = _count_open_files(process)
    try:
        return ProcessInfo(
            pid=process.pid,
            memory_rss_mb=mem_info.rss / (1024 * 1024),
            memory_vms_mb=mem_info.vms / (1024 * 1024),
            memory_percent=process.memory_percent(),
            cpu_percent=process.cpu_percent(),
            num_threads=process.num_threads(),
            open_files=open_files_count,
        )
    except Exception:
        return _fallback_process_info()


def _allocated_cpus_from_sched_affinity() -> tuple[int | None, bool]:
    sched: Callable[[int], set[int]] | None = getattr(os, "sched_getaffinity", None)
    if not callable(sched):
        return None, False

    try:
        aff: set[int] = sched(0)
    except Exception:
        return None, False

    try:
        return sum(1 for _ in aff), True
    except Exception:
        return None, True


def _allocated_cpus_from_process_affinity() -> int | None:
    try:
        proc = psutil.Process()
    except Exception:
        return None

    try:
        cpu_aff: Callable[[], set[int]] | None = getattr(proc, "cpu_affinity", None)
    except Exception:
        return None

    if not callable(cpu_aff):
        return None

    try:
        return sum(1 for _ in cpu_aff())
    except Exception:
        return None


def get_allocated_cpus() -> int | None:
    """Get the number of CPUs allocated to this job/process."""
    # SLURM environment variables (prefer explicit allocations)
    slurm_val: str | None = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_val:
        try:
            return int(slurm_val)

        except Exception:
            pass

    # sched_getaffinity when available (not on macOS)
    cpu_count: int | None
    handled_sched: bool
    cpu_count, handled_sched = _allocated_cpus_from_sched_affinity()
    if cpu_count is not None or handled_sched:
        return cpu_count

    # Fallback to psutil Process.cpu_affinity if present
    if (cpu_count := _allocated_cpus_from_process_affinity()) is not None:
        return cpu_count

    return None


def get_allocated_memory() -> float | None:
    """Get the amount of memory allocated to this job in GB."""
    try:
        # Try SLURM allocation
        if "SLURM_MEM_PER_NODE" in os.environ:
            return int(os.environ["SLURM_MEM_PER_NODE"]) / 1024  # Convert MB to GB

        if "SLURM_MEM_PER_CPU" in os.environ:
            mem_per_cpu = int(os.environ["SLURM_MEM_PER_CPU"])
            cpus = get_allocated_cpus() or 1

            return (mem_per_cpu * cpus) / 1024  # Convert MB to GB

        # Could add support for other job schedulers here
    except Exception:
        return None

    else:
        return None


def _parse_allocated_gpu_ids(gpu_str: str | None) -> list[int] | None:
    if gpu_str and gpu_str != "-1":
        return [int(x.strip()) for x in gpu_str.split(",") if x.strip().isdigit()]

    return None


def _allocated_gpus_from_slurm_count(gpu_count: str | None) -> list[int] | None:
    if not gpu_count:
        return None

    try:
        num_gpus = int(gpu_count)
    except Exception:
        return None

    return list(range(num_gpus))


def get_allocated_gpus() -> list[int] | None:
    """Get the list of GPU IDs allocated to this job."""
    # Try CUDA_VISIBLE_DEVICES first
    cuda_gpus: list[int] | None = _parse_allocated_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if cuda_gpus is not None:
        return cuda_gpus

    # Try SLURM GPU allocation
    slurm_gpus: list[int] | None = _parse_allocated_gpu_ids(os.environ.get("SLURM_GPUS"))
    if slurm_gpus is not None:
        return slurm_gpus

    try:
        return _allocated_gpus_from_slurm_count(os.environ.get("SLURM_GPUS_ON_NODE"))
    except Exception:
        return None


def _get_gputil_gpus() -> list[GPU] | None:
    try:
        gputil: ModuleType = importlib.import_module("GPUtil")
        return gputil.getGPUs()
    except Exception:
        return None


def _gpu_info_from_gputil(gpu: GPU) -> GPUInfo:
    return GPUInfo(
        id=gpu.id,
        name=gpu.name,
        memory_total_mb=gpu.memoryTotal,
        memory_used_mb=gpu.memoryUsed,
        memory_util_percent=round(gpu.memoryUtil * 100, 1),
        load_percent=round(gpu.load * 100, 1),
        temperature=getattr(gpu, "temperature", None),
    )


def get_job_resources() -> JobResourceInfo:
    """Get job-specific resource allocation and usage."""
    return JobResourceInfo(
        allocated_cpus=get_allocated_cpus(),
        allocated_memory_gb=get_allocated_memory(),
        allocated_gpus=get_allocated_gpus(),
        process=get_process_info(),
    )


def get_relevant_gpu_info(allocated_gpus: list[int] | None) -> list[GPUInfo]:
    """Get GPU info only for allocated GPUs."""
    all_gpus: list[GPU] | None = _get_gputil_gpus()
    if not all_gpus:
        return []

    try:
        if allocated_gpus is None:
            # If no allocation info, show all GPUs but limit to reasonable number
            return [_gpu_info_from_gputil(g) for g in all_gpus[:4]]  # Limit to first 4 GPUs

        # Only show allocated GPUs
        relevant_gpus: list[GPUInfo] = [_gpu_info_from_gputil(g) for g in all_gpus if g.id in allocated_gpus]
    except Exception:
        return []

    else:
        return relevant_gpus


def get_system_summary() -> str:
    """Get a brief, one-line system summary."""
    try:
        os_info: OSInfo = get_os_info()
        cpu_info: CPUInfo = get_cpu_info()
        python_info: PythonInfo = get_python_info()
    except Exception:
        return "System Info Unavailable"

    else:
        return f"{os_info.system} | {cpu_info.brand or 'CPU'} | Python {python_info.version}"


def collect_runtime_stats(elapsed: float | None = None) -> RuntimeStats:
    """Collect runtime statistics focused on job execution."""
    job_resources: JobResourceInfo = get_job_resources()

    return RuntimeStats(
        timestamp=time.time(),
        elapsed_seconds=elapsed,
        job_resources=job_resources,
        gpu_usage=get_relevant_gpu_info(job_resources.allocated_gpus),
        system_summary=get_system_summary(),
        python_info=get_python_info(),
        extended_info=get_extended_system_info(),
    )


def format_timestamp(timestamp: float) -> str:
    """Format timestamp as 'YYYY-MM-DD | HH:MM:SS:MS'.

    Args:
        timestamp: Unix timestamp as float

    Returns:
        str: Formatted timestamp string
    """
    dt: datetime = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    milliseconds: int = int((timestamp % 1) * 1000)

    return f"{dt.strftime('%Y-%m-%d')} | {dt.strftime('%H:%M:%S')}:{milliseconds:03d}"


class SystemStatsLogger(ContextDecorator):
    """Logs runtime statistics focused on job execution and resource usage.

    Args:
        description (str): Label for the code block (e.g., "Training Loop").
        include_system_summary (bool): If True, include brief system info. Defaults to True.
        include_process_stats (bool): If True, include process-specific stats. Defaults to True.
        include_resource_allocation (bool): If True, include job resource allocation. Defaults to True.
        include_gpu_stats (bool): If True, include allocated GPU stats. Defaults to True.
        extra_packages (list[str] or None): List of PyPI packages to report versions for. Defaults to None.

    Attributes:
        description (str): Label for the code block.
        include_system_summary (bool): Whether to include system summary.
        include_process_stats (bool): Whether to include process stats.
        include_resource_allocation (bool): Whether to include resource allocation.
        include_gpu_stats (bool): Whether to include GPU stats.
        extra_packages (list[str]): List of extra PyPI packages to report.
        start_stats (RuntimeStats or None): Snapshot of runtime stats at entry.

    Example:
        ```python
        with SystemStatsLogger("Training Loop"):
            train_model()
        ```
    """

    __slots__ = (
        "description",
        "extra_packages",
        "include_gpu_stats",
        "include_process_stats",
        "include_resource_allocation",
        "include_system_summary",
        "start_stats",
    )

    def __init__(
        self,
        description: str = "Code Block",
        *,
        include_system_summary: bool = True,
        include_process_stats: bool = True,
        include_resource_allocation: bool = True,
        include_gpu_stats: bool = True,
        extra_packages: list[str] | None = None,
    ) -> None:
        """Initialize the SystemStatsLogger context manager.

        Args:
            description (str): Label for the code block (e.g., "Training Loop").
            include_system_summary (bool): If True, include brief system info. Defaults to True.
            include_process_stats (bool): If True, include process-specific stats. Defaults to True.
            include_resource_allocation (bool): If True, include job resource allocation. Defaults to True.
            include_gpu_stats (bool): If True, include allocated GPU stats. Defaults to True.
            extra_packages (list[str] or None): List of PyPI packages to report versions for. Defaults to None.
        """
        self.description: str = description
        self.include_system_summary: bool = include_system_summary
        self.include_process_stats: bool = include_process_stats
        self.include_resource_allocation: bool = include_resource_allocation
        self.include_gpu_stats: bool = include_gpu_stats
        self.extra_packages: list[str] = extra_packages or []
        self.start_stats: RuntimeStats | None = None

    def __enter__(self) -> Self:
        """Enter the context manager, logging runtime stats at the start of the block.

        Returns:
            SystemStatsLogger: The context manager instance.
        """
        if not logger.isEnabledFor(INFO):
            return self

        self.start_stats = collect_runtime_stats()
        self.log(self.start_stats, event="START")

        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: TracebackType | None
    ) -> bool | None:
        """Exit the context manager, logging runtime stats at the end of the block and any exception info.

        Args:
            exc_type (type | None): Exception type, if any.
            exc_value (BaseException | None): Exception value, if any.
            tb (object | None): Traceback object, if any.

        Returns:
            bool: False (never suppresses exceptions).
        """
        if logger.isEnabledFor(INFO):
            elapsed: float = time.time() - self.start_stats.timestamp if self.start_stats else 0.0
            end_stats: RuntimeStats = collect_runtime_stats(elapsed)
            self.log(end_stats, event="END", elapsed=elapsed)
            if exc_type:
                logger.error(f"Exception in {self.description}", exc_info=sys.exc_info())

        return None  # never suppress exceptions

    def _runtime_title(self, s: RuntimeStats, event: str, elapsed: float | None) -> str:
        title_parts: list[str] = [f"{event} {safe_escape(self.description)}"]
        title_parts.append(f"@ {format_timestamp(s.timestamp)}")
        if elapsed is not None:
            title_parts.append(f"| elapsed={elapsed:.2f}s")

        return " ".join(title_parts)

    def _add_system_summary(self, main_table: Table, s: RuntimeStats) -> None:
        system_table = Table.grid(padding=(0, 2))
        system_table.add_column(style="dim white", justify="left")
        system_table.add_column(style="dim white", justify="left", no_wrap=True)
        system_table.add_column(style="dim white", justify="left")

        summary_parts: list[str] = s.system_summary.split(" | ")
        if len(summary_parts) >= 3:
            system_table.add_row("[bright_blue]OS[/bright_blue]", "[bright_blue]CPU[/bright_blue]")
            cpu_name: str = safe_escape(summary_parts[1])
            if len(cpu_name) > 50:
                cpu_name = f"{cpu_name[:47]}..."
            system_table.add_row(f"[dim]{safe_escape(summary_parts[0])}[/dim]", f"[dim]{cpu_name}[/dim]")
        else:
            system_table.add_row("[bright_blue]Info[/bright_blue]")
            system_table.add_row(f"[dim]{s.system_summary}[/dim]")

        main_table.add_row("System", system_table)

    def _build_machine_table(self) -> Table:
        cpu_info: CPUInfo = get_cpu_info()
        machine_memory: MemoryInfo = get_memory_info()
        all_gpus: list[GPUInfo] = get_gpu_info()

        machine_table = Table.grid(padding=(0, 2))
        machine_table.add_column(style="dim white", justify="left")
        machine_table.add_column(style="dim white", justify="left")
        machine_table.add_column(style="dim white", justify="left")

        machine_table.add_row(
            "[bright_green]CPUs[/bright_green]",
            "[bright_green]Memory[/bright_green]",
            "[bright_green]GPUs[/bright_green]",
        )

        total_cpus: int = cpu_info.logical_cores
        total_memory_gb: float = machine_memory.total_bytes / (1024**3)
        total_gpus: int = len(all_gpus)
        machine_table.add_row(
            f"[dim]{total_cpus}[/dim]", f"[dim]{total_memory_gb:.1f}GB[/dim]", f"[dim]{total_gpus}[/dim]"
        )

        return machine_table

    def _add_resource_allocation(self, main_table: Table, s: RuntimeStats) -> None:
        try:
            machine_table: Table = self._build_machine_table()
        except Exception:
            pass
        else:
            main_table.add_row("Machine", machine_table)

        jr: JobResourceInfo = s.job_resources
        allocation_table: Table = Table.grid(padding=(0, 2))
        allocation_table.add_column(style="dim white", justify="left")
        allocation_table.add_column(style="dim white", justify="left")
        allocation_table.add_column(style="dim white", justify="left")

        allocation_table.add_row(
            "[bright_green]CPUs[/bright_green]",
            "[bright_green]Memory[/bright_green]",
            "[bright_green]GPUs[/bright_green]",
        )

        cpu_str: str = str(jr.allocated_cpus) if jr.allocated_cpus is not None else "N/A"
        memory_str: str = f"{jr.allocated_memory_gb:.1f}GB" if jr.allocated_memory_gb is not None else "N/A"
        gpu_str: str = "N/A" if jr.allocated_gpus is None else f"{len(jr.allocated_gpus)} (IDs: {jr.allocated_gpus})"

        allocation_table.add_row(f"[dim]{cpu_str}[/dim]", f"[dim]{memory_str}[/dim]", f"[dim]{gpu_str}[/dim]")
        main_table.add_row("Allocated", allocation_table)

    def _add_process_stats(self, main_table: Table, s: RuntimeStats) -> None:
        proc: ProcessInfo = s.job_resources.process

        process_table: Table = Table.grid(padding=(0, 2))
        process_table.add_column(style="dim white", justify="left")
        process_table.add_column(style="dim white", justify="left")
        process_table.add_column(style="dim white", justify="left")
        process_table.add_column(style="dim white", justify="left")

        process_table.add_row(
            "[bright_yellow]PID[/bright_yellow]",
            "[bright_yellow]RAM[/bright_yellow]",
            "[bright_yellow]CPU[/bright_yellow]",
            "[bright_yellow]Threads[/bright_yellow]",
        )
        process_table.add_row(
            f"[dim]{proc.pid}[/dim]",
            f"[dim]{proc.memory_rss_mb:.1f}MB ({proc.memory_percent:.1f}%)[/dim]",
            f"[dim]{proc.cpu_percent:.1f}%[/dim]",
            f"[dim]{proc.num_threads}[/dim]",
        )

        main_table.add_row("Process", process_table)

        python_table: Table = Table.grid(padding=(0, 2))
        python_table.add_column(style="dim white", justify="left")
        python_table.add_row("[bright_yellow]Version[/bright_yellow]")
        python_table.add_row(f"[dim]{safe_escape(s.python_info.version)}[/dim]")
        main_table.add_row("Python", python_table)

    def _add_gpu_stats(self, main_table: Table, s: RuntimeStats) -> None:
        gpu_table: Table = Table.grid(padding=(0, 2))
        gpu_table.add_column(style="dim cyan", justify="left")
        gpu_table.add_column(style="dim white", justify="left")
        gpu_table.add_column(style="dim white", justify="left")
        gpu_table.add_column(style="dim white", justify="left")
        gpu_table.add_column(style="dim white", justify="left")

        gpu_table.add_row(
            "[bright_magenta]ID[/bright_magenta]",
            "[bright_magenta]Name[/bright_magenta]",
            "[bright_magenta]Memory[/bright_magenta]",
            "[bright_magenta]Load[/bright_magenta]",
            "[bright_magenta]Temp[/bright_magenta]",
        )

        for gpu in s.gpu_usage:
            temp_str: str = f"{gpu.temperature:.0f}°C" if gpu.temperature is not None else "N/A"
            memory_str = f"{gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f}MB ({gpu.memory_util_percent:.1f}%)"
            gpu_table.add_row(
                f"[dim]{gpu.id}[/dim]",
                f"[dim]{safe_escape(gpu.name)}[/dim]",
                f"[dim]{memory_str}[/dim]",
                f"[dim]{gpu.load_percent:.1f}%[/dim]",
                f"[dim]{temp_str}[/dim]",
            )

        main_table.add_row("GPUs", gpu_table)

    def _add_package_versions(self, main_table: Table) -> None:
        packages: list[PackageInfo] = get_package_info(self.extra_packages)
        if not packages:
            return

        package_table: Table = Table.grid(padding=(0, 2))
        package_table.add_column(style="dim white", justify="left")
        package_table.add_column(style="dim white", justify="left")

        package_table.add_row("[bright_cyan]Package[/bright_cyan]", "[bright_cyan]Version[/bright_cyan]")
        for pkg in packages:
            package_table.add_row(f"[dim]{pkg.name}[/dim]", f"[dim]{pkg.version}[/dim]")

        main_table.add_row("Packages", package_table)

    def _add_extended_info(self, main_table: Table, ext: ExtendedSystemInfo) -> None:
        host_table = Table.grid(padding=(0, 2))
        host_table.add_column(style="dim white", justify="left")
        host_table.add_column(style="dim white", justify="left")
        host_table.add_column(style="dim white", justify="left")

        host_table.add_row(
            "[bright_cyan]Host[/bright_cyan]", "[bright_cyan]User[/bright_cyan]", "[bright_cyan]CUDA[/bright_cyan]"
        )
        hostname: str = ext.host.hostname or "N/A"
        user: str = ext.host.user or "N/A"
        cuda_ver: str = ext.cuda_version or "N/A"
        host_table.add_row(
            f"[dim]{safe_escape(hostname)}[/dim]", f"[dim]{safe_escape(user)}[/dim]", f"[dim]{cuda_ver}[/dim]"
        )

        main_table.add_row("Host", host_table)
        self._add_git_info(main_table, ext)
        self._add_slurm_info(main_table, ext)
        self._add_disk_info(main_table, ext)

    def _add_git_info(self, main_table: Table, ext: ExtendedSystemInfo) -> None:
        if not ext.git:
            return

        git_table: Table = Table.grid(padding=(0, 2))
        git_table.add_column(style="dim white", justify="left")
        git_table.add_column(style="dim white", justify="left")

        git_table.add_row("[bright_cyan]Branch[/bright_cyan]", "[bright_cyan]Commit[/bright_cyan]")
        branch: str = ext.git.branch or "N/A"
        commit: str = f"{ext.git.commit[:8]}..." if ext.git.commit else "N/A"
        git_table.add_row(f"[dim]{safe_escape(branch)}[/dim]", f"[dim]{commit}[/dim]")
        main_table.add_row("Git", git_table)

    def _add_slurm_info(self, main_table: Table, ext: ExtendedSystemInfo) -> None:
        if not ext.slurm:
            return

        slurm_table: Table = Table.grid(padding=(0, 2))
        slurm_table.add_column(style="dim white", justify="left")
        slurm_table.add_column(style="dim white", justify="left")
        slurm_table.add_column(style="dim white", justify="left")

        slurm_table.add_row(
            "[bright_cyan]Job ID[/bright_cyan]",
            "[bright_cyan]Partition[/bright_cyan]",
            "[bright_cyan]Nodes[/bright_cyan]",
        )
        job_id: str = ext.slurm.job_id or "N/A"
        partition: str = ext.slurm.partition or "N/A"
        nodes: str = ext.slurm.node_list or "N/A"
        slurm_table.add_row(
            f"[dim]{job_id}[/dim]", f"[dim]{safe_escape(partition)}[/dim]", f"[dim]{safe_escape(nodes)}[/dim]"
        )
        main_table.add_row("SLURM", slurm_table)

    def _add_disk_info(self, main_table: Table, ext: ExtendedSystemInfo) -> None:
        if not ext.disk_usage:
            return

        disk_table: Table = Table.grid(padding=(0, 2))
        disk_table.add_column(style="dim white", justify="left")
        disk_table.add_column(style="dim white", justify="left")

        disk_table.add_row("[bright_cyan]Usage[/bright_cyan]", "[bright_cyan]Free[/bright_cyan]")
        used_gb: float = ext.disk_usage.used_bytes / (1024**3)
        free_gb: float = ext.disk_usage.free_bytes / (1024**3)
        disk_table.add_row(f"[dim]{used_gb:.1f}GB ({ext.disk_usage.percent:.1f}%)[/dim]", f"[dim]{free_gb:.1f}GB[/dim]")
        main_table.add_row("Disk", disk_table)

    def _build_panel(self, s: RuntimeStats, *, event: str, elapsed: float | None) -> Panel:
        title: str = self._runtime_title(s, event, elapsed)
        main_table: Table = Table.grid(padding=(0, 2))
        main_table.add_column(style="dim cyan", justify="left")
        main_table.add_column(style="dim white", justify="left")

        if self.include_system_summary:
            self._add_system_summary(main_table, s)
        if self.include_resource_allocation:
            self._add_resource_allocation(main_table, s)
        if self.include_process_stats:
            self._add_process_stats(main_table, s)
        if self.include_gpu_stats and s.gpu_usage:
            self._add_gpu_stats(main_table, s)
        if event == "START" and self.extra_packages:
            self._add_package_versions(main_table)
        if event in {"START", "INFO"} and s.extended_info:
            self._add_extended_info(main_table, s.extended_info)

        return Panel(
            main_table,
            title=f"[bold]{title}[/bold]",
            box=box.ROUNDED,
            width=120,
            border_style="dim blue",
            padding=(1, 2),
        )

    def _log_fallback(self, s: RuntimeStats, *, event: str, elapsed: float | None) -> None:
        logger.info(
            f"[dim]Runtime Stats [{event}] @ {format_timestamp(s.timestamp)}"
            f"{f' | elapsed={elapsed:.2f}s' if elapsed is not None else ''}[/dim]"
        )

        if self.include_process_stats:
            proc = s.job_resources.process
            logger.info(
                f"[dim]Process: PID={proc.pid:d}, RAM={proc.memory_rss_mb:.1f}MB, CPU={proc.cpu_percent:.1f}%[/dim]"
            )

        if self.include_gpu_stats and s.gpu_usage:
            for gpu in s.gpu_usage:
                logger.info(
                    f"[dim]GPU {gpu.id:d}: {gpu.name}, "
                    f"Memory={gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f}MB[/dim]"
                )

    def log(self, s: RuntimeStats, *, event: str, elapsed: float | None = None) -> None:
        """Log runtime statistics in a clean, focused format using tables.

        Args:
            s (RuntimeStats): The runtime statistics to log.
            event (str): Event label (e.g., 'START', 'END').
            elapsed (float | None): Elapsed time in seconds, if applicable.
        """
        try:
            panel: Panel = self._build_panel(s, event=event, elapsed=elapsed)
            logger.info(panel)
        except Exception:
            logger.exception("Failed to render runtime stats")
            try:
                self._log_fallback(s, event=event, elapsed=elapsed)
            except Exception:
                logger.exception("Even fallback logging failed")


def log_system_stats(
    description: str = "System Info",
    *,
    include_system_summary: bool = True,
    include_process_stats: bool = True,
    include_resource_allocation: bool = True,
    include_gpu_stats: bool = True,
    extra_packages: list[str] | None = None,
) -> None:
    """Log system statistics in a single call.

    Args:
        description (str): Label for the statistics (e.g., "Training Start").
        include_system_summary (bool): If True, include brief system info. Defaults to True.
        include_process_stats (bool): If True, include process-specific stats. Defaults to True.
        include_resource_allocation (bool): If True, include job resource allocation. Defaults to True.
        include_gpu_stats (bool): If True, include allocated GPU stats. Defaults to True.
        extra_packages (list[str] or None): List of PyPI packages to report versions for. Defaults to None.
    """
    if not logger.isEnabledFor(INFO):
        return

    # Create a temporary logger instance to use its log method
    temp_logger = SystemStatsLogger(
        description=description,
        include_system_summary=include_system_summary,
        include_process_stats=include_process_stats,
        include_resource_allocation=include_resource_allocation,
        include_gpu_stats=include_gpu_stats,
        extra_packages=extra_packages,
    )

    # Collect current stats and log them
    stats: RuntimeStats = collect_runtime_stats()
    temp_logger.log(stats, event="INFO")


def _run_git_command(git_executable: str, *args: str) -> str | None:
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            [git_executable, *args], capture_output=True, text=True, timeout=5, check=False
        )
    except Exception:
        return None

    return result.stdout.strip() if result.returncode == 0 else None


def get_git_info() -> GitInfo | None:
    """Get git repository information if available."""
    git_executable: str | None = shutil.which("git")
    if git_executable is None:
        return None

    if _run_git_command(git_executable, "rev-parse", "--is-inside-work-tree") is None:
        return None

    remote: str | None = _run_git_command(git_executable, "remote", "get-url", "origin")
    commit: str | None = _run_git_command(git_executable, "rev-parse", "HEAD")
    branch: str | None = _run_git_command(git_executable, "branch", "--show-current")
    if remote or commit or branch:
        return GitInfo(remote=remote, commit=commit, branch=branch)

    return None


def get_host_info() -> HostInfo:
    """Get host and environment information."""
    hostname: str | None = None
    try:
        hostname = platform.node()
    except Exception:
        hostname = os.environ.get("HOSTNAME")

    user: str | None = None
    try:
        user = os.environ.get("USER") or os.environ.get("USERNAME")
    except Exception:
        user = None

    executable: str | None = None
    try:
        executable = sys.executable
    except Exception:
        executable = None

    working_dir: str | None = None
    try:
        working_dir = str(pathlib.Path.cwd())
    except Exception:
        working_dir = None

    return HostInfo(hostname=hostname, user=user, executable=executable, working_dir=working_dir)


def get_slurm_info() -> SlurmInfo | None:
    """Get SLURM job information if available."""
    try:
        # Check if we're in a SLURM environment
        if "SLURM_JOB_ID" not in os.environ:
            return None

        return SlurmInfo(
            job_id=os.environ.get("SLURM_JOB_ID"),
            job_name=os.environ.get("SLURM_JOB_NAME"),
            partition=os.environ.get("SLURM_JOB_PARTITION"),
            node_list=os.environ.get("SLURM_JOB_NODELIST"),
            cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"),
            mem_per_node=os.environ.get("SLURM_MEM_PER_NODE"),
            gpus_on_node=os.environ.get("SLURM_GPUS_ON_NODE"),
        )
    except Exception:
        return None


def get_cuda_version() -> str | None:
    """Get CUDA version if available."""
    nvcc_executable: str | None = shutil.which("nvcc")
    if nvcc_executable is None:
        return None

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            [nvcc_executable, "--version"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            # Parse version from output
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    # Extract version number
                    match: re.Match[str] | None = re.search(r"release (\d+\.\d+)", line)
                    if match:
                        return match.group(1)

    except Exception:
        return None

    else:
        return None


def get_extended_system_info() -> ExtendedSystemInfo:
    """Get extended system information."""
    # Get total system memory
    total_memory_gb: float | None = None
    try:
        mem_info: MemoryInfo = get_memory_info()
        total_memory_gb = mem_info.total_bytes / (1024**3)
    except Exception:
        pass

    # Get total CPU count
    total_cpu_count: int | None = None
    with contextlib.suppress(Exception):
        total_cpu_count = psutil.cpu_count(logical=True)

    return ExtendedSystemInfo(
        host=get_host_info(),
        git=get_git_info(),
        slurm=get_slurm_info(),
        disk_usage=get_disk_info(),
        total_memory_gb=total_memory_gb,
        total_cpu_count=total_cpu_count,
        cuda_version=get_cuda_version(),
    )
