"""
Game of Life with classic patterns + haloexec + RasterMap
=========================================================
Places well-known patterns (glider, blinker, beacon, toad, block,
pulsar) DELIBERATELY over block boundaries, so it is easy to check
visually whether the halo synchronizes correctly (a pattern that
crosses a block edge and keeps behaving as it should is the most direct
visual evidence).

Requirements
------------
    pip install dissmodel
    pip install -e /path/to/dissmodel-ca
    pip install -e /path/to/haloexec

Usage
-----
    python examples/gol/gol_patterns_haloexec.py

Without an interactive display, the PNGs go to ./raster_map_frames/.
"""
from __future__ import annotations

import numpy as np
from dissmodel.core import Environment
from dissmodel.geo import raster_grid
from dissmodel.visualization.raster_map import RasterMap
from dissmodel_ca.models.game_of_life import PATTERNS

from haloexec import HaloChunkedRasterCellularAutomaton


# ---------------------------------------------------------------------------
# Model: GameOfLife with a halo
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
    """Write a pattern into the grid starting at (top, left)."""
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

# Placed ON PURPOSE over block boundaries (multiples of 10)
# — the worst case for testing halo synchronization.
# FIXED: beacon moved from col 25 -> col 33 (in the original version,
# the 4x4 beacon at (9,25) collided with the 13x13 pulsar at (4,20) --
# the beacon's area [9:13, 25:29] falls entirely inside the pulsar's
# area [4:17, 20:33]. place() overwrites by direct assignment, so both
# patterns would have been silently corrupted, with no error at all.
place(grid, PATTERNS["glider"], 8, 8)     # crosses the (10,10) corner diagonally
place(grid, PATTERNS["blinker"], 20, 5)   # crosses the horizontal edge at r=20
place(grid, PATTERNS["beacon"], 9, 33)    # crosses the horizontal edge at r=10
place(grid, PATTERNS["toad"], 29, 15)     # crosses the horizontal edge at r=30
place(grid, PATTERNS["block"], 19, 19)    # sits on the corner where 4 blocks meet (20,20)
place(grid, PATTERNS["pulsar"], 4, 20)    # larger oscillator, period 3

backend = raster_grid(rows=ROWS, cols=COLS, attrs={"state": grid})

env = Environment(start_time=0, end_time=GENERATIONS)

gol = GameOfLifeHalo(
    backend=backend,
    block_h=BLOCK_H,
    block_w=BLOCK_W,
    halo=HALO,
)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
RasterMap(
    backend=backend,
    band="state",
    color_map={0: "#ffffff", 1: "#2f8f6e"},
    labels={0: "dead", 1: "alive"},
    title=f"Classic patterns over block boundaries (halo={HALO})",
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
env.run()
