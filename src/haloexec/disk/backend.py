"""
RasterBackend adapter for MemmapRasterWorkspace.

Lets components of the dissmodel ecosystem (such as RasterMap viewers,
exporters or collectors) access the arrays of the workspace's current
read slot without materializing or duplicating the whole grid in RAM.
"""

from __future__ import annotations

import numpy as np

from .workspace import MemmapRasterWorkspace


class WorkspaceRasterBackend:
    """
    Lightweight adapter that exposes a MemmapRasterWorkspace through the
    RasterBackend interface (shape and a dictionary of arrays).

    The exposed arrays are direct references (np.memmap) to the
    workspace's current read slot, honouring double-buffering and
    swap_buffers.

    Parameters
    ----------
    workspace : MemmapRasterWorkspace
        On-disk workspace to adapt.
    stride : int, default=1
        Spatial subsampling (decimation) factor applied when accessing
        arrays. Useful for large-scale visualization (e.g. 30M+ cells),
        greatly reducing matplotlib's rendering time and memory use
        without changing the data on disk.
    nodata_value : float | None, default=None
        Optional nodata value, for RasterMap's extent masks.
    """

    def __init__(
        self,
        workspace: MemmapRasterWorkspace,
        stride: int = 1,
        nodata_value: float | None = None,
    ) -> None:
        self.workspace = workspace
        self.stride = max(1, int(stride))
        self.nodata_value = nodata_value
        h, w = workspace.shape
        self.shape: tuple[int, int] = (h // self.stride, w // self.stride)

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """Dictionary with the arrays of the current read slot."""
        slot = self.workspace.checkpoint_data["read_slot"]
        memmaps = self.workspace._slots[slot]
        if self.stride == 1:
            return memmaps
        return {name: mm[::self.stride, ::self.stride] for name, mm in memmaps.items()}

    def get(self, name: str) -> np.ndarray:
        """Return the array with the given name."""
        return self.arrays[name]

    def snapshot(self) -> dict[str, np.ndarray]:
        """Return a copy of the arrays of the current read slot."""
        return {k: np.asarray(v).copy() for k, v in self.arrays.items()}

    def __repr__(self) -> str:
        names = list(self.workspace.metadata["arrays"].keys())
        return (
            f"WorkspaceRasterBackend(shape={self.shape}, arrays={names}, "
            f"stride={self.stride}, slot={self.workspace.checkpoint_data['read_slot']})"
        )
