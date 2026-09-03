"""
Integração com dissmodel: DiskChunkedRasterCellularAutomaton.

Equivalente em disco de ram/cellular_automaton.py::HaloChunkedRasterCellularAutomaton
— mesmo contrato de rule() (`rule(arrays) -> dict`), mas lendo/escrevendo
via MemmapRasterWorkspace em vez de materializar a grade inteira em RAM.

Preenche uma lacuna real: existia adaptador de disco pra modelos
SyncRasterModel (disk/sync_model.py::DiskChunkedSyncRasterModel), mas
não para modelos RasterCellularAutomaton — qualquer AC escrito com
rule() (como dissmodel_ca.models.game_of_life_raster.GameOfLife) só
rodava em RAM até este módulo existir.

Uso: `class GameOfLifeHalo(DiskChunkedRasterCellularAutomaton, GameOfLife): pass`
reusa o rule() real da classe original sem reescrever nada — mesmo
princípio de composição cooperativa (MRO) já usado em
disk/sync_model.py com FloodModel/MangroveModel.
"""

from __future__ import annotations

import numpy as np

from dissmodel.geo.raster.backend import RasterBackend

from .workspace import MemmapRasterWorkspace
from ..engine import resolve_boundary_value


class DiskChunkedRasterCellularAutomaton:
    """
    Mixin que processa rule() de um RasterCellularAutomaton em blocos
    lidos de um MemmapRasterWorkspace, sem nunca materializar a grade
    inteira em RAM.

    Ordem de herança (MRO): este mixin deve vir primeiro, ex.:
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

        # Placeholder leve: RasterBackend(shape=...) não aloca arrays,
        # só existe para satisfazer o contrato de RasterModel.setup()
        # (self.backend = backend; self.shape = backend.shape). O
        # backend real por bloco é criado dentro de execute().
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
                updates = self.rule(block_backend.snapshot())  # mesmo contrato de sempre
            finally:
                self.backend = real_backend
                self.shape = real_shape

            core_updates = {}
            for name, arr in updates.items():
                core = arr[h:-h, h:-h] if h > 0 else arr
                core_updates[name] = core
            ws.write_block_core(block, core_updates)

        ws.swap_buffers()
