"""
dissmodel integration: HaloChunkedRasterCellularAutomaton.

Extends dissmodel.geo.raster.cellular_automaton.RasterCellularAutomaton
to run rule() in blocks with a halo, instead of over the whole grid at
once.

Central design point: it keeps the SAME rule() contract as the base
class (`rule(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]`).
Any RasterCellularAutomaton already written for dissmodel therefore
runs in blocks+halo just by swapping the base class — no change to the
rule's logic is needed. This property is what makes moving an existing
model onto this engine a structural swap, not a rewrite.

How it works
------------
1. Take a snapshot of the global grid (equivalent to `self.backend.past`).
2. Pad each array with the global halo (`np.pad`).
3. For each block, build a temporary RasterBackend holding only that
   block's sub-grid + halo, and point `self.backend` at it. This is
   what makes calls inside the rule such as
   `self.backend.focal_sum_mask(...)` operate on the correct local
   shape (RasterBackend.focal_sum_mask uses the active backend's
   `self.shape`) instead of the global one.
4. Call `self.rule(block_backend.snapshot())` — the usual signature.
5. Crop the halo from the result (keep only the "core" region) and
   write it at the matching position of the new global grid.
6. Restore `self.backend` to the real global backend and apply the
   updates.

Theoretical basis: Kjolstad & Snir (2010), Ghost Cell Pattern
(ParaPLoP); Xia et al. (2025), ISPRS IJGI 14(3):109 — see README.md.
"""

from __future__ import annotations

import numpy as np
from dissmodel.geo.raster.backend import RasterBackend
from dissmodel.geo.raster.cellular_automaton import RasterCellularAutomaton

from ..engine import Block, make_blocks, resolve_boundary_value


class HaloChunkedRasterCellularAutomaton(RasterCellularAutomaton):
    """
    RasterCellularAutomaton that processes the grid in blocks with a
    halo, instead of all at once.

    Usage: any existing RasterCellularAutomaton subclass can switch its
    base class to this one and gain domain decomposition without
    changing `rule()`.

    Examples
    --------
    >>> class GameOfLife(HaloChunkedRasterCellularAutomaton):
    ...     def rule(self, arrays):
    ...         state = arrays["state"]
    ...         neighbors = self.backend.focal_sum_mask(state == 1)
    ...         born = (state == 0) & (neighbors == 3)
    ...         survive = (state == 1) & np.isin(neighbors, [2, 3])
    ...         return {"state": np.where(born | survive, 1, 0)}
    >>> b = RasterBackend(shape=(50, 50))
    >>> b.set("state", np.random.randint(0, 2, (50, 50)))
    >>> env = Environment(start_time=1, end_time=100)
    >>> GameOfLife(backend=b, block_h=10, block_w=10, halo=1)
    >>> env.run()
    """

    def setup(  # type: ignore[override]
        self,
        backend: RasterBackend,
        block_h: int,
        block_w: int,
        halo: int = 1,
        boundary_value: float = 0,
        state_attr: str = "state",
    ) -> None:
        """
        Parameters
        ----------
        backend : RasterBackend
            Shared global backend (same semantics as the base class).
        block_h, block_w : int
            Processing block size.
        halo : int, optional
            The rule's neighbourhood radius. Must be >= the maximum
            spatial dependency reach of one time step. Default 1
            (immediate Moore/Von Neumann neighbours).
        boundary_value : float, optional
            Fill value of the global halo at the grid's outer edges
            (outside the simulated domain). Default 0.
        state_attr : str, optional
            See the base class.
        """
        super().setup(backend=backend, state_attr=state_attr)
        self.block_h = block_h
        self.block_w = block_w
        self.halo = halo
        self.boundary_value = boundary_value

    def _block_backend(self, padded: dict[str, np.ndarray], block: Block) -> RasterBackend:
        """Build a temporary RasterBackend with the block's sub-grid + halo."""
        h = self.halo
        block_shape = (block.r1 - block.r0 + 2 * h, block.c1 - block.c0 + 2 * h)
        temp = RasterBackend(shape=block_shape)
        for name, arr in padded.items():
            sub = arr[block.r0: block.r1 + 2 * h, block.c0: block.c1 + 2 * h]
            temp.set(name, sub)
        return temp

    def execute(self) -> None:
        """
        Run one time step, processing the grid in blocks+halo.

        Replaces the base class's execute() (which calls rule() once over
        the whole grid) with a block loop, keeping the same rule()
        contract for whoever writes the rule.
        """
        real_backend = self.backend
        height, width = real_backend.shape
        h = self.halo

        global_snapshot = real_backend.snapshot()
        padded = {
            name: np.pad(arr, h, mode="constant",
                         constant_values=resolve_boundary_value(self.boundary_value, name))
            for name, arr in global_snapshot.items()
            if arr.ndim == 2  # temporal (time, y, x) arrays are not supported here
        }

        new_arrays: dict[str, np.ndarray] = {
            name: np.zeros_like(arr) for name, arr in global_snapshot.items() if arr.ndim == 2
        }

        for block in make_blocks(height, width, self.block_h, self.block_w):
            block_backend = self._block_backend(padded, block)

            # Temporary swap: makes calls such as
            # self.backend.focal_sum_mask(...) inside rule() operate on
            # the block's local shape, not the global one.
            self.backend = block_backend
            try:
                updates = self.rule(block_backend.snapshot())
            finally:
                self.backend = real_backend

            for name, block_result in updates.items():
                core = block_result[h:-h, h:-h] if h > 0 else block_result
                if name not in new_arrays:
                    new_arrays[name] = np.zeros((height, width), dtype=core.dtype)
                new_arrays[name][block.r0:block.r1, block.c0:block.c1] = core

        for name, arr in new_arrays.items():
            real_backend.arrays[name] = arr
