from __future__ import annotations

from collections.abc import Callable
from logging import getLogger
import multiprocessing
import os
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger
    from subprocess import CompletedProcess
    from typing import TypeAlias


logger: Logger = getLogger(__name__)

SchedGetAffinityT: TypeAlias = Callable[[int], set[int]]


def _read_text(path: str) -> str | None:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _cpu_count_from_cgroup_v1() -> int | None:
    cgroup_text: str | None = _read_text("/proc/self/cgroup")
    if cgroup_text is None or "cpu" not in cgroup_text:
        return None

    quota_raw: str | None = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_raw: str | None = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_raw is None or period_raw is None:
        return None

    try:
        quota_us: int = int(quota_raw.strip())
        period_us: int = int(period_raw.strip())
    except ValueError:
        return None

    if quota_us > 0 and period_us > 0:
        return max(1, quota_us // period_us)

    return None


def get_cpu_count() -> int:
    """Return the CPU limit visible to the current process.

    Priority order:
    1. SLURM job allocation variables
    2. Cgroup limits (containerized environments)
    3. Process affinity masks
    4. System CPU count
    5. Hardware thread count

    Returns:
        int: Number of CPUs available.
    """
    # SLURM environment detection (HPC clusters)
    slurm_vars: tuple[str, ...] = ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_NPROCS", "SLURM_NTASKS")
    for var in slurm_vars:
        if (value := os.environ.get(var)) and value.isdigit():
            return int(value)

    # Container/cgroup limits (Docker, Kubernetes, etc.)
    try:
        # Check cgroup v2 first (newer systems)
        quota, period = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip().split()
        if quota != "max" and period.isdigit() and int(period) > 0:
            return max(1, int(quota) // int(period))
    except (OSError, ValueError, IndexError):
        pass

    if (cgroup_v1_count := _cpu_count_from_cgroup_v1()) is not None:
        return cgroup_v1_count

    # Process affinity (actual available CPUs to this process)
    try:
        if sys.platform.startswith("linux"):
            sched: SchedGetAffinityT | None = getattr(os, "sched_getaffinity", None)
            if sched is not None:
                return len(sched(0))
    except OSError:
        # Calling sched_getaffinity may raise OSError on some platforms/contexts.
        pass

    # System-level CPU detection with cached results
    cpu_count: int = get_system_cpu_count()
    return cpu_count if cpu_count > 0 else 1


def get_system_cpu_count() -> int:
    """Return the online logical CPU count using platform fallbacks."""
    # Start with the standard library count.
    if (count := os.cpu_count()) is not None:
        return count

    # Windows fallback
    if sys.platform == "win32":
        if (count := os.environ.get("NUMBER_OF_PROCESSORS")) and count.isdigit():
            return int(count)

    # Unix-like systems: direct sysconf call
    try:
        # Prefer the standard library os.sysconf where available.
        if hasattr(os, "sysconf"):
            value: int = os.sysconf("SC_NPROCESSORS_ONLN")
            if value > 0:
                return value
    except (AttributeError, OSError, ValueError):
        pass

    # Linux: parse /proc/cpuinfo (more accurate than /proc/stat)
    try:
        return pathlib.Path("/proc/cpuinfo").read_bytes().count(b"processor\t:")
    except OSError:
        pass

    # macOS: sysctl hardware query
    if sys.platform == "darwin":
        try:
            result: CompletedProcess[str] = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.ncpu"], check=False, capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return int(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            pass

    # Fall back to the count reported by multiprocessing.
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        pass

    return 0  # Indicates failure, handled by caller


if __name__ == "__main__":
    logger.info(f"Available CPUs: {get_cpu_count()}")
