"""
Game of Life com padrões clássicos + haloexec + RasterMap
============================================================
Posiciona padrões conhecidos (glider, blinker, beacon, toad, block,
pulsar) DELIBERADAMENTE sobre as fronteiras de bloco, para que seja
fácil verificar visualmente se o halo está sincronizando corretamente
(um padrão que atravessa a borda de um bloco e continua se comportando
como deveria é a evidência visual mais direta).

Requisitos
----------
    pip install dissmodel
    pip install -e /caminho/para/dissmodel-ca
    pip install -e /caminho/para/haloexec

Uso
---
    python gol_patterns_haloexec.py

Sem display interativo, os PNGs caem em ./raster_map_frames/.
"""
from __future__ import annotations

import numpy as np

from dissmodel.core import Environment
from dissmodel.geo import raster_grid
from dissmodel.visualization.raster_map import RasterMap

from haloexec import HaloChunkedRasterCellularAutomaton
from dissmodel_ca.models.game_of_life import PATTERNS


# ---------------------------------------------------------------------------
# Modelo: GameOfLife com halo
# ---------------------------------------------------------------------------
class GameOfLifeHalo(HaloChunkedRasterCellularAutomaton):
    def setup(
        self,
        backend,
        block_h: int = 10,
        block_w: int = 10,
        halo: int = 1,
        state_attr: str = "state",
    ) -> None:
        super().setup(
            backend=backend,
            block_h=block_h,
            block_w=block_w,
            halo=halo,
            state_attr=state_attr,
        )

    def rule(self, arrays: dict) -> dict:
        state = arrays[self.state_attr]
        neighbors = self.backend.focal_sum_mask(state == 1)
        survive = (state == 1) & np.isin(neighbors, [2, 3])
        born = (state == 0) & (neighbors == 3)
        return {self.state_attr: np.where(survive | born, 1, 0).astype(np.int8)}


def place(grid: np.ndarray, pattern: list[list[int]], top: int, left: int) -> None:
    """Escreve um padrão na grade a partir de (top, left)."""
    arr = np.array(pattern)
    h, w = arr.shape
    grid[top:top + h, left:left + w] = arr


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROWS, COLS = 40, 40
BLOCK_H, BLOCK_W = 10, 10
HALO = 1
GENERATIONS = 16

grid = np.zeros((ROWS, COLS), dtype=np.int8)

# Posicionados DE PROPÓSITO sobre fronteiras de bloco (múltiplos de 10)
# — o pior caso para testar a sincronização do halo.
# CORRIGIDO: beacon movido de col 25 -> col 33 (na versão original,
# beacon em (9,25) 4x4 colidia com pulsar em (4,20) 13x13 -- a área
# do beacon [9:13, 25:29] cai inteira dentro da área do pulsar
# [4:17, 20:33]. place() sobrescreve por atribuição direta, então os
# dois padrões ficariam corrompidos silenciosamente, sem erro nenhum.
place(grid, PATTERNS["glider"], 8, 8)     # atravessa o cruzamento (10,10) na diagonal
place(grid, PATTERNS["blinker"], 20, 5)   # atravessa a borda horizontal em r=20
place(grid, PATTERNS["beacon"], 9, 33)    # atravessa a borda horizontal em r=10
place(grid, PATTERNS["toad"], 29, 15)     # atravessa a borda horizontal em r=30
place(grid, PATTERNS["block"], 19, 19)    # atravessa o cruzamento de 4 blocos (10,20)+(10,20)
place(grid, PATTERNS["pulsar"], 4, 20)    # oscilador maior, período 3

backend = raster_grid(rows=ROWS, cols=COLS, attrs={"state": grid})

env = Environment(start_time=0, end_time=GENERATIONS)

gol = GameOfLifeHalo(
    backend=backend,
    block_h=BLOCK_H,
    block_w=BLOCK_W,
    halo=HALO,
)

# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------
RasterMap(
    backend=backend,
    band="state",
    color_map={0: "#ffffff", 1: "#2f8f6e"},
    labels={0: "morta", 1: "viva"},
    title=f"Padrões clássicos sobre fronteiras de bloco (halo={HALO})",
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
env.run()
