"""Visualization utilities for SPAR package."""

from __future__ import annotations

from os import environ as os_environ
from typing import TYPE_CHECKING

from spar.utils.import_utils.lazy_importer import LazyImporter

if TYPE_CHECKING:
    from spar.utils.import_utils.lazy_importer import AttributeGetter, DirectoryLister, LazyAttribute

    from .image_grid import FigureStyle, FrameStyle, GridStyle, TextStyle, create_custom_style, create_image_grid

# Lazy imports mapping
IMPORTS: dict[str, tuple[str, str]] = {
    "FrameStyle": ("spar.utils.viz_utils.image_grid", "FrameStyle"),
    "TextStyle": ("spar.utils.viz_utils.image_grid", "TextStyle"),
    "FigureStyle": ("spar.utils.viz_utils.image_grid", "FigureStyle"),
    "GridStyle": ("spar.utils.viz_utils.image_grid", "GridStyle"),
    "create_custom_style": ("spar.utils.viz_utils.image_grid", "create_custom_style"),
    "create_image_grid": ("spar.utils.viz_utils.image_grid", "create_image_grid"),
    "MSEPlotter": ("spar.utils.viz_utils.mse_plotter", "MSEPlotter"),
    "PlotConfig": ("spar.utils.viz_utils.mse_plotter", "PlotConfig"),
    "StyleName": ("spar.utils.viz_utils.mse_plotter", "StyleName"),
    "StyleSpec": ("spar.utils.viz_utils.mse_plotter", "StyleSpec"),
    "SmoothingMethod": ("spar.utils.viz_utils.mse_plotter", "SmoothingMethod"),
    "UncertaintyMethod": ("spar.utils.viz_utils.mse_plotter", "UncertaintyMethod"),
    "plot_mse_with_defaults": ("spar.utils.viz_utils.mse_plotter", "plot_mse_with_defaults"),
    "compare_mse_series": ("spar.utils.viz_utils.mse_plotter", "compare_mse_series"),
}

# Create lazy importer
lazy_importer: LazyImporter[LazyAttribute] = LazyImporter(
    imports=IMPORTS,
    module_name=__name__,
    # type_checking_imports=[
    #     "from .image_grid import FrameStyle,TextStyle, FigureStyle, GridStyle, create_custom_style, create_image_grid"
    # ],
    cache_enabled=True,
    thread_safe=False,
    debug_mode=(os_environ.get("SPAR_DEBUG_LAZY_IMPORTS", "").lower() in {"true", "1", "yes"}),
    validate_imports=True,
)

# Export the interface
__all__: list[str] = ["FigureStyle", "FrameStyle", "GridStyle", "TextStyle", "create_custom_style", "create_image_grid"]
__getattr__: AttributeGetter = lazy_importer.get_attr
__dir__: DirectoryLister = lazy_importer.get_dir


# Development utilities (only available in debug mode)
if lazy_importer.debug_mode:

    def generate_viz_type_checking_block() -> str:
        """Generate TYPE_CHECKING block for debugging.

        Returns:
            String containing the TYPE_CHECKING block
        """
        return lazy_importer.generate_type_checking_block()

    def get_viz_type_stubs() -> dict[str, str]:
        """Get type stub information for debugging.

        Returns:
            Dictionary mapping attribute names to type stub strings
        """
        return lazy_importer.get_type_stubs()

    # Add debug utilities to module in debug mode
    globals().update({
        "generate_viz_type_checking_block": generate_viz_type_checking_block,
        "get_viz_type_stubs": get_viz_type_stubs,
    })
