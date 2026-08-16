"""Utilities for percentage formatting and rounding in visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def round_percentages(arr: NDArray[np.floating] | Sequence[float]) -> NDArray[np.floating]:
    """Round percentages to 2 decimal places, but truncate (don't round up) values that would round to exactly 100.00%.

    Args:
        arr: numpy array or array-like of percentage values

    Returns:
        numpy array of rounded percentages
    """
    arr = np.asarray(arr)

    # Check which values would round to 100.00
    rounded: NDArray[np.floating] = np.round(arr, 2)
    would_round_to_100 = (rounded >= 100.0) & (arr < 100.0)

    # Truncate those that would round to 100.00 but aren't actually 100.00
    return np.where(would_round_to_100, np.floor(arr * 100) / 100, rounded)


def format_percentage(value: float | np.floating | str | None, use_special_rounding: bool = True) -> str:
    """Format a single percentage value with special rounding behavior.

    Args:
        value: Single percentage value to format
        use_special_rounding: Whether to use the special rounding behavior

    Returns:
        Formatted percentage string
    """
    if value is None or (isinstance(value, str) and value == "N/A"):
        return "N/A"

    # Convert supported scalar types to float.
    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return "N/A"

    rounded_value: float = round_percentages([numeric_value])[0] if use_special_rounding else np.round(numeric_value, 2)

    return f"{rounded_value:.2f}%"


def format_percentages(values: NDArray[np.floating], use_special_rounding: bool = True) -> list[str]:
    """Format multiple percentage values with special rounding behavior.

    Args:
        values: Array-like of percentage values to format
        use_special_rounding: Whether to use the special rounding behavior

    Returns:
        List of formatted percentage strings
    """
    rounded_values: NDArray[np.floating] = round_percentages(values) if use_special_rounding else np.round(values, 2)

    return [f"{val:.2f}%" for val in rounded_values]
