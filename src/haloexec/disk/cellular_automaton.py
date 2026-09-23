"""
dissmodel integration: DiskChunkedRasterCellularAutomaton.

Disk counterpart of ram/cellular_automaton.py::HaloChunkedRasterCellularAutomaton
— same rule() contract (`rule(arrays) -> dict`), but reading/writing
through MemmapRasterWorkspace instead of materializing the whole grid
in RAM.

Complements disk/sync_model.py::DiskChunkedSyncRasterModel (the disk
adapter for SyncRasterModel models) for RasterCellularAutomaton models:
any CA written with rule() (such as
dissmodel_ca.models.game_of_life_raster.GameOfLife) can run from disk.

Usage: `class GameOfLifeHalo(DiskChunkedRasterCellularAutomaton, GameOfLife): pass`
reuses the original class's real rule() without rewriting anything —
the same cooperative-composition (MRO) principle used in
disk/sync_model.py with FloodModel/MangroveModel.
"""

from __future__ import annotations

from dissmodel.geo.raster.backend import RasterBackend

from .workspace import MemmapRasterWorkspace


class DiskChunkedRasterCellularAutomaton:
    """
    Mixin that runs a RasterCellularAutomaton's rule() in blocks read
    from a MemmapRasterWorkspace, never materializing the whole grid in
    RAM.

    Inheritance order (MRO): this mixin must come first, e.g.
    `class GameOfLifeHalo(DiskChunkedRasterCellularAutomaton, GameOfLife)`.
    """

    def setup(
        self,
        workspace: MemmapRasterWorkspace,
        halo: int | None = None,
        boundary_value: dict | float = 0,
        state_attr: str = "state",
        **kwargs,
    ) -> None:
        self.workspace = workspace
        self.halo = workspace.halo if halo is None else halo
        self.boundary_value = boundary_value

        # Lightweight placeholder: RasterBackend(shape=...) allocates no
        # arrays; it only satisfies RasterModel.setup()'s contract
        # (self.backend = backend; self.shape = backend.shape). The real
        # per-block backend is created inside execute().
        placeholder = RasterBackend(shape=workspace.shape)
        super().setup(backend=placeholder, state_attr=state_attr, **kwargs)

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
                updates = self.rule(block_backend.snapshot())  # the usual contract
            finally:
                self.backend = real_backend
                self.shape = real_shape

            core_updates = {}
            for name, arr in updates.items():
                core = arr[h:-h, h:-h] if h > 0 else arr
                core_updates[name] = core
            ws.write_block_core(block, core_updates)

        ws.swap_buffers()
