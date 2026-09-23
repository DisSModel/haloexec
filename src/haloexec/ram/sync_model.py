"""
dissmodel integration: HaloChunkedSyncRasterModel.

Unlike RasterCellularAutomaton (which exposes a dedicated rule(arrays)
hook), models based on SyncRasterModel/RasterModel — such as a flood
model — implement their scientific logic directly in execute(), reading
and writing named arrays on the backend (self.backend.arrays["alt"],
self.backend.get("uso_past"), etc.) and using self.shape/self.shift/
self.dirs inherited from RasterModel.

This module extends the same chunking+halo strategy to that pattern,
through cooperative multiple inheritance (a mixin):
HaloChunkedSyncRasterModel intercepts execute() and setup() and
delegates the real logic to the concrete subclass through
super().execute(), with self.backend/self.shape temporarily swapped for
a per-block local sub-grid.

This means NOT A SINGLE LINE of the model (FloodModel or any other
SyncRasterModel) has to change — only the inheritance order in the
class declaration:

    class FloodModelHalo(HaloChunkedSyncRasterModel, FloodModel):
        pass

The order matters (MRO): the mixin must come first, so its
execute()/setup() runs first, with super() delegating to the real
FloodModel.execute()/setup().

Known limitation: SyncRasterModel's pre_execute()/post_execute() (which
take the "<name>_past" snapshot) are NOT intercepted by this mixin —
they keep operating on the real global backend, outside the block loop.
This is intentional: synchronizing "_past" is a plain whole-array copy
with no neighbourhood dependency, so it needs no domain decomposition.
The halo is only needed inside execute(), where neighbours are read
through self.shift.
"""

from __future__ import annotations

import numpy as np
from dissmodel.geo.raster.backend import RasterBackend

from ..engine import make_blocks, resolve_boundary_value


class HaloChunkedSyncRasterModel:
    """
    Mixin that runs a RasterModel/SyncRasterModel's execute() in blocks
    with a halo, delegating the scientific logic to the next class in
    the MRO through super().

    Parameters (setup, on top of those the concrete subclass accepts)
    -----------------------------------------------------------------
    block_h, block_w : int
        Processing block size.
    halo : int, optional
        Neighbourhood radius used by the rule (default 1).

        WARNING — halo is NOT always equal to the nominal shift radius
        the rule uses. If the rule computes a quantity DERIVED from
        neighbours (e.g. a "flow" that depends on how many neighbours
        satisfy a condition) and then reads that derived quantity FROM A
        NEIGHBOUR (not its own raw value), the real dependency is 2 hops,
        not 1 — halo=1 is then subtly wrong near internal block
        boundaries (not at the domain edges, which boundary_value already
        handles). Documented in the README ("the correct halo depth is
        the dependency chain's depth"): the BR-MANGUE FloodModel needs
        halo=2 for exactly this reason (its neighbour flow depends on the
        neighbour's count of lower cells, which depends on the
        neighbour's neighbours). When adapting a new rule, if the
        equivalence tests pass on simple synthetic data but fail on
        real/irregular data, suspect a 2+ hop dependency before anything
        else.
    boundary_value : float, optional
        Fill value of the global halo at the grid's outer edges.
        Default 0.
    """

    def setup(self, backend: RasterBackend, block_h: int, block_w: int,
              halo: int = 1, boundary_value: float = 0, **kwargs) -> None:
        self.block_h = block_h
        self.block_w = block_w
        self.halo = halo
        self.boundary_value = boundary_value
        super().setup(backend=backend, **kwargs)  # delegate to the real subclass

    def execute(self) -> None:
        real_backend = self.backend
        real_shape = self.shape
        height, width = real_backend.shape
        h = self.halo

        # Every static (2-D) array of the global backend, whatever its
        # name — generic enough for any concrete model.
        static_names = [n for n, a in real_backend.arrays.items() if a.ndim == 2]
        padded = {
            n: np.pad(real_backend.arrays[n], h, mode="constant",
                      constant_values=resolve_boundary_value(self.boundary_value, n))
            for n in static_names
        }

        new_arrays: dict[str, np.ndarray] = {}

        for block in make_blocks(height, width, self.block_h, self.block_w):
            block_shape = (block.r1 - block.r0 + 2 * h, block.c1 - block.c0 + 2 * h)
            block_backend = RasterBackend(shape=block_shape)
            for name, arr in padded.items():
                sub = arr[block.r0: block.r1 + 2 * h, block.c0: block.c1 + 2 * h]
                block_backend.set(name, sub)

            # Temporary swap: makes self.shape and self.backend (used
            # directly inside the real subclass's execute(), e.g.
            # `rows, cols = self.shape`) reflect the block's local shape,
            # not the global one.
            self.backend = block_backend
            self.shape = block_backend.shape
            try:
                super().execute()  # the real logic (e.g. FloodModel.execute)
            finally:
                self.backend = real_backend
                self.shape = real_shape

            # Reconcile: crop the halo and write into the new global grid.
            # "<name>_past" arrays are skipped here — they are managed by
            # the global synchronize() in pre_execute()/post_execute() and
            # must not be overwritten with local slices that carry a halo.
            for name, arr in block_backend.arrays.items():
                if name.endswith("_past"):
                    continue
                core = arr[h:-h, h:-h] if h > 0 else arr
                if name not in new_arrays:
                    new_arrays[name] = np.zeros((height, width), dtype=core.dtype)
                new_arrays[name][block.r0:block.r1, block.c0:block.c1] = core

        for name, arr in new_arrays.items():
            real_backend.arrays[name] = arr
