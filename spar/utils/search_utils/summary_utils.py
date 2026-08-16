from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

from rich.align import Align
from rich.panel import Panel

if TYPE_CHECKING:
    from typing import TypeAlias


SummaryValue: TypeAlias = str | int | float | bool | None
SolveCategory: TypeAlias = Literal["both", "search_only", "none"]


class SummaryVariantRow(TypedDict):
    start_variant: str
    goal_variant: str
    solve_category: SolveCategory
    solved_by_search: bool
    solved_by_env: bool
    elapsed_sec: float
    num_moves: SummaryValue
    path_cost: SummaryValue


class SummaryIndexBucket(TypedDict):
    total: int
    both: int
    search_only: int
    none: int
    variants: list[SummaryVariantRow]


class SummaryCategoryCounts(TypedDict):
    total: int
    both: int
    search_only: int
    none: int


class QStarLogEntry(TypedDict):
    """Per-instance Q* search log entry."""

    index: int
    sequence_index: int
    input_type: Literal["hdf5", "images"]
    start_variant: str
    goal_variant: str
    solved_by_search: bool
    env_validation_attempted: bool
    solved_by_env: bool | None
    solve_category: SolveCategory
    path_cost: float
    num_moves: int
    logged_moves: list[int]
    num_nodes_generated: int
    num_iterations: int
    elapsed_sec: float
    nodes_per_sec: float
    timings_sec: dict[str, float]
    visuals_emitted: bool


class QStarSummaryOverall(TypedDict):
    """Top-level aggregate metrics for a Q* search run."""

    entries_total: int
    solved_by_both: int
    solved_by_search_only: int
    unsolved_by_both: int
    search_success_rate: float
    avg_moves_solved_any: float
    avg_iterations_solved_any: float
    avg_nodes_generated: float
    avg_time_sec: float
    avg_nodes_per_sec: float


class QStarOutputSummary(TypedDict):
    """JSON-serializable summary payload for Q* search results."""

    overall: QStarSummaryOverall
    by_index: dict[str, SummaryIndexBucket]
    by_start_variant: dict[str, SummaryCategoryCounts]
    by_goal_variant: dict[str, SummaryCategoryCounts]


def _float_or_default(value: SummaryValue, default: float) -> float:
    if not value:
        return default

    return float(value)


def _numeric_value(value: SummaryValue) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)

    return None


def _increment_category(bucket: SummaryIndexBucket | SummaryCategoryCounts, category: SolveCategory) -> None:
    if category == "both":
        bucket["both"] += 1
    elif category == "search_only":
        bucket["search_only"] += 1
    else:
        bucket["none"] += 1


__all__: list[str] = [
    "QStarLogEntry",
    "QStarOutputSummary",
    "QStarSummaryStats",
    "QStarSummaryTracker",
    "build_summary_panel",
    "format_summary_lines",
]


@dataclass(frozen=True, slots=True)
class QStarSummaryStats:
    """Aggregated metrics for Q* search runs."""

    entries_total: int
    solved_by_both: int
    solved_by_search_only: int
    unsolved_by_both: int
    search_success_rate: float
    avg_moves_solved_any: float
    avg_iterations_solved_any: float
    avg_nodes_generated: float
    avg_time_sec: float
    avg_nodes_per_sec: float
    optimal: str
    per_index: dict[int, SummaryIndexBucket]
    by_start_variant: dict[str, SummaryCategoryCounts]
    by_goal_variant: dict[str, SummaryCategoryCounts]


class QStarSummaryTracker:
    """Incrementally tracks summary statistics for Q* search results."""

    __slots__: tuple[str, ...] = (
        "_itrs_count",
        "_itrs_sum",
        "_moves_count",
        "_moves_sum",
        "_nodes_sum",
        "_nps_count",
        "_nps_sum",
        "_time_sum",
        "by_goal_variant",
        "by_start_variant",
        "entries_total",
        "per_index",
        "solved_by_both",
        "solved_by_search_only",
        "unsolved_by_both",
    )

    def __init__(self) -> None:
        self.entries_total: int = 0
        self.solved_by_both: int = 0
        self.solved_by_search_only: int = 0
        self.unsolved_by_both: int = 0
        self._time_sum: float = 0.0
        self._nodes_sum: float = 0.0
        self._nps_sum: float = 0.0
        self._nps_count: int = 0
        self._moves_sum: float = 0.0
        self._moves_count: int = 0
        self._itrs_sum: float = 0.0
        self._itrs_count: int = 0
        self.per_index: dict[int, SummaryIndexBucket] = {}
        self.by_start_variant: dict[str, SummaryCategoryCounts] = {}
        self.by_goal_variant: dict[str, SummaryCategoryCounts] = {}

    def update(self, entry: QStarLogEntry) -> None:
        """Consume a per-instance result entry and update aggregates."""
        self.entries_total += 1
        elapsed: float = _float_or_default(entry.get("elapsed_sec"), 0.0)
        nodes_generated: float = _float_or_default(entry.get("num_nodes_generated"), 0.0)
        self._time_sum += elapsed
        self._nodes_sum += nodes_generated
        # Prefer the recorded node rate and derive it when absent.
        nps_val: float | None = _numeric_value(entry.get("nodes_per_sec"))
        if nps_val is None and elapsed > 0:
            nps_val = nodes_generated / elapsed
        if nps_val is not None:
            self._nps_sum += nps_val
            self._nps_count += 1

        solved_by_search: bool = entry.get("solved_by_search", False)
        solved_by_env_val: SummaryValue = entry.get("solved_by_env")
        solved_by_env: bool = bool(solved_by_env_val) if solved_by_env_val is not None else False
        solved_any: bool = solved_by_search or solved_by_env

        category: SolveCategory = entry["solve_category"]
        if category == "both":
            self.solved_by_both += 1
        elif category == "search_only":
            self.solved_by_search_only += 1
        else:
            self.unsolved_by_both += 1

        if solved_any:
            self._moves_sum += float(entry["num_moves"])
            self._moves_count += 1
            self._itrs_sum += float(entry["num_iterations"])
            self._itrs_count += 1

        start_variant: str = entry["start_variant"]
        goal_variant: str = entry["goal_variant"]

        idx: int = entry["index"]
        idx_bucket: SummaryIndexBucket = self.per_index.setdefault(
            idx, {"total": 0, "both": 0, "search_only": 0, "none": 0, "variants": []}
        )
        idx_bucket["total"] += 1
        _increment_category(idx_bucket, category)
        idx_bucket["variants"].append({
            "start_variant": start_variant,
            "goal_variant": goal_variant,
            "solve_category": category,
            "solved_by_search": solved_by_search,
            "solved_by_env": solved_by_env,
            "elapsed_sec": elapsed,
            "num_moves": entry.get("num_moves"),
            "path_cost": entry.get("path_cost"),
        })

        start_bucket: SummaryCategoryCounts = self.by_start_variant.setdefault(
            start_variant, {"total": 0, "both": 0, "search_only": 0, "none": 0}
        )
        start_bucket["total"] += 1
        _increment_category(start_bucket, category)

        goal_bucket: SummaryCategoryCounts = self.by_goal_variant.setdefault(
            goal_variant, {"total": 0, "both": 0, "search_only": 0, "none": 0}
        )
        goal_bucket["total"] += 1
        _increment_category(goal_bucket, category)

    def stats(self) -> QStarSummaryStats:
        """Return a snapshot of the current summary statistics."""
        entries: int = self.entries_total
        avg_time: float = (self._time_sum / entries) if entries else 0.0
        avg_nodes: float = (self._nodes_sum / entries) if entries else 0.0
        avg_moves: float = (self._moves_sum / self._moves_count) if self._moves_count else 0.0
        avg_iters: float = (self._itrs_sum / self._itrs_count) if self._itrs_count else 0.0
        solved_total: int = self.solved_by_both + self.solved_by_search_only
        success_rate: float = (solved_total / entries * 100.0) if entries else 0.0
        avg_nodes_per_sec: float = (self._nps_sum / self._nps_count) if self._nps_count else 0.0

        return QStarSummaryStats(
            entries_total=entries,
            solved_by_both=self.solved_by_both,
            solved_by_search_only=self.solved_by_search_only,
            unsolved_by_both=self.unsolved_by_both,
            search_success_rate=success_rate,
            avg_moves_solved_any=avg_moves,
            avg_iterations_solved_any=avg_iters,
            avg_nodes_generated=avg_nodes,
            avg_time_sec=avg_time,
            avg_nodes_per_sec=avg_nodes_per_sec,
            optimal="N/A",
            per_index=self.per_index,
            by_start_variant=self.by_start_variant,
            by_goal_variant=self.by_goal_variant,
        )


def format_summary_lines(stats: QStarSummaryStats) -> list[str]:
    """Return rich-formatted summary lines for display."""
    return [
        f"[bold]Entries[/bold]: {stats.entries_total}",
        f"[bold]Solved by both[/bold]: {stats.solved_by_both}",
        f"[bold]Solved by search only[/bold]: {stats.solved_by_search_only}",
        f"[bold]Unsolved by both[/bold]: {stats.unsolved_by_both}",
        f"[bold]Search success rate[/bold]: {stats.search_success_rate:.2f}%",
        f"[bold]Avg Number of Moves (solved-any)[/bold]: {stats.avg_moves_solved_any:.2f}",
        f"[bold]Optimal[/bold]: {stats.optimal}",
        f"[bold]Avg Itrs (solved-any)[/bold]: {stats.avg_iterations_solved_any:.2f}",
        f"[bold]Avg Number of Nodes Gen[/bold]: {stats.avg_nodes_generated:.2f}",
        f"[bold]Avg Time[/bold]: {stats.avg_time_sec:.2f}s",
        f"[bold]Avg Nodes/Sec[/bold]: {stats.avg_nodes_per_sec:.2E}",
    ]


def build_summary_panel(stats: QStarSummaryStats, *, title: str, border_style: str = "blue") -> Panel:
    """Construct a Rich Panel rendering of the summary statistics."""
    lines: list[str] = format_summary_lines(stats)
    return Panel(
        Align.left("\n".join(lines)),
        title=title,
        title_align="left",
        border_style=border_style,
        padding=(1, 2),
        width=120,
    )
