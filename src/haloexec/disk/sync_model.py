"""
Disk + halo integration for SyncRasterModel (FloodModel, MangroveModel,
and any other dissmodel model following the same pattern), without
modifying the installed dissmodel package.

Why reusing HaloChunkedSyncRasterModel (in-memory) is not enough
----------------------------------------------------------------
SyncRasterModel.pre_execute()/post_execute() call synchronize(), which
does `self.backend.get(name).copy()` on the WHOLE array — if
`self.backend` were a wrapper over an np.memmap, that `.copy()` would
materialize the whole grid in RAM just to take the "<name>_past"
snapshot, defeating the purpose of using disk.

This module REPLICATES SyncRasterModel's "_past" synchronization logic
locally (it neither imports nor modifies it in the installed package),
adapted to copy block by block between memmaps through
MemmapRasterWorkspace. When the migration to the dissmodel core
happens, this is the part to reconcile with
dissmodel.geo.raster.sync_model.SyncRasterModel.synchronize() — for
now the two live side by side, with no coupling.

Double-buffer semantics for "_past"
-----------------------------------
Unlike the "current" arrays (e.g. "uso", "alt"), which only become
ready in the OTHER slot after a complete step (classic ping-pong), the
"_past" arrays must be available in the SAME slot that execute() will
READ in that step — which is why they use write_block_to_read_slot(),
not write_block_core(). See the MemmapRasterWorkspace docstring.

Usage (array and parameter names are those of the BR-MANGUE FloodModel)
-----------------------------------------------------------------------
    class FloodModelDiskHalo(DiskChunkedSyncRasterModel, FloodModel):
        pass

    arrays = workspace_arrays_for_sync_model(
        base={"uso": np.int16, "alt": np.float32},
        land_use_types=["uso", "alt"],
    )
    ws = MemmapRasterWorkspace.create(root=..., shape=..., arrays=arrays,
                                       block_h=.., block_w=.., halo=1)
    ws.fill("uso", uso_inicial)
    ws.fill("alt", alt_inicial)

    env = Environment(start_time=1, end_time=n)
    FloodModelDiskHalo(workspace=ws, taxa_elevacao=0.05)
    env.run()
"""

from __future__ import annotations

import numpy as np
from dissmodel.geo.raster.backend import RasterBackend

from .workspace import MemmapRasterWorkspace


def workspace_arrays_for_sync_model(
    base: dict[str, np.dtype],
    land_use_types: list[str],
) -> dict[str, np.dtype]:
    """Build the dict of arrays to declare in MemmapRasterWorkspace.create(),
    adding "<name>_past" automatically for each name in land_use_types
    (same dtype as the base array)."""
    arrays = dict(base)
    for name in land_use_types:
        arrays[f"{name}_past"] = np.dtype(base[name])
    return arrays


class DiskChunkedSyncRasterModel:
    """
    Mixin that runs a SyncRasterModel (e.g. FloodModel) in blocks read
    from a MemmapRasterWorkspace, including the "_past" synchronization
    done block by block — never materializing the whole grid in RAM.

    Inheritance order (MRO): this mixin must come first, e.g.
    `class FloodModelDiskHalo(DiskChunkedSyncRasterModel, FloodModel)`.
    """

    def setup(self, workspace: MemmapRasterWorkspace, halo: int | None = None,
              boundary_value: float = 0, **kwargs) -> None:
        self.workspace = workspace
        self.halo = workspace.halo if halo is None else halo
        self.boundary_value = boundary_value
        self._synced_before_first_execute = False

        # Lightweight placeholder: RasterBackend(shape=...) allocates no
        # arrays; it only satisfies RasterModel.setup()'s contract
        # (self.backend = backend; self.shape = backend.shape). The real
        # per-block backend is created inside execute().
        placeholder = RasterBackend(shape=workspace.shape)
        super().setup(backend=placeholder, **kwargs)  # delegate to the real subclass

    def _synchronize_via_workspace(self) -> None:
        """Block-by-block equivalent of SyncRasterModel.synchronize():
        copies "<name>" -> "<name>_past" within the SAME current read
        slot (see the module docstring)."""
        for name in getattr(self, "land_use_types", []):
            for block in self.workspace.blocks():
                values = self.workspace.read_block_core(block, name)
                self.workspace.write_block_to_read_slot(block, f"{name}_past", values)

    def pre_execute(self) -> None:
        if not self._synced_before_first_execute:
            self._synchronize_via_workspace()
            self._synced_before_first_execute = True

    def post_execute(self) -> None:
        self._synchronize_via_workspace()

    def execute(self) -> None:
        ws = self.workspace
        real_backend = self.backend
        real_shape = self.shape
        h = self.halo

        for block in ws.blocks():
            window = ws.read_block_with_halo(block, boundary_value=self.boundary_value)
            block_backend = RasterBackend(shape=next(iter(window.values())).shape)
            for name, arr in window.items():
                block_backend.set(name, arr)

            self.backend = block_backend
            self.shape = block_backend.shape
            try:
                super().execute()  # the real logic (e.g. FloodModel.execute)
            finally:
                self.backend = real_backend
                self.shape = real_shape

            updates = {}
            for name, arr in block_backend.arrays.items():
                # IMPORTANT: do not skip "_past" here. If this model does
                # not manage a given "_past" (e.g. FloodModel does not
                # manage "solo_past", only MangroveModel does), it still
                # has to be carried over unchanged across the swap —
                # otherwise it is orphaned in the new slot (never written,
                # left with the zeroed/stale value of the memmap's initial
                # allocation). The "_past" that THIS model manages is
                # correctly overwritten by _synchronize_via_workspace in
                # post_execute(), already in the post-swap slot.
                core = arr[h:-h, h:-h] if h > 0 else arr
                updates[name] = core
            ws.write_block_core(block, updates)

        ws.swap_buffers()
