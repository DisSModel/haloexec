"""
Visualization and checkpoint components for haloexec and dissmodel.

Provides CheckpointRasterMap, which draws and saves frames (PNG) only at
chosen simulation steps or years (avoiding the overhead on intermediate
steps).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

try:
    from dissmodel.visualization.raster_map import RasterMap
    HAS_RASTERMAP = True
except ImportError:
    HAS_RASTERMAP = False


if HAS_RASTERMAP:
    class CheckpointRasterMap(RasterMap):
        """
        RasterMap extension that filters which simulation steps or years are
        drawn and exported to PNG.

        Skips matplotlib rendering and memmap page reads on intermediate steps,
        making long simulations on large grids practical.

        Parameters
        ----------
        save_steps : Iterable[int] | None
            List or set of steps (e.g. [1, 5, 10, 20]) at which the frame is
            rendered and saved. If None, behaves like the standard RasterMap
            (following the `step` parameter of the base Model class).
        **kwargs
            All other arguments are passed on to RasterMap (backend, band,
            color_map, cmap, save_frames, etc.).
        """

        def setup(  # type: ignore[override]
            self,
            *args: Any,
            save_steps: Iterable[int] | None = None,
            **kwargs: Any,
        ) -> None:
            self.save_steps: set[int] | None = set(save_steps) if save_steps is not None else None
            super().setup(*args, **kwargs)

        def execute(self) -> None:
            step = int(self.env.now())
            if self.save_steps is None or step in self.save_steps:
                super().execute()
else:
    class CheckpointRasterMap:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "RasterMap requires dissmodel installed with the viz extra: "
                "pip install 'dissmodel[viz]'"
            )
