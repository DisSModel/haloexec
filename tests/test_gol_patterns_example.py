"""
Equivalence proof for examples/gol/gol_patterns_haloexec.py: classic
Game of Life patterns (glider, blinker, beacon, toad, block, pulsar),
deliberately placed over block boundaries, must produce an IDENTICAL
result between the monolithic run (plain RasterCellularAutomaton) and
the blocks+halo run (HaloChunkedRasterCellularAutomaton).

This is what actually backs the example's claim ("a pattern that crosses
a block edge and keeps behaving as it should is the most direct visual
evidence that the halo synchronizes correctly") -- running without
error does not prove it; only the cell-by-cell comparison does.

Found while writing this test: the example's original coordinates put
the beacon (9,25, 4x4) entirely inside the pulsar's area (4,20, 13x13)
-- a silent overlap, since place() overwrites by direct assignment.
Fixed by moving the beacon to column 33.
"""

import numpy as np
import pytest

pytest.importorskip("dissmodel_ca")

from dissmodel.core import Environment
from dissmodel.geo import raster_grid
from dissmodel.geo.raster.cellular_automaton import RasterCellularAutomaton
from dissmodel_ca.models.game_of_life import PATTERNS

from haloexec import HaloChunkedRasterCellularAutomaton

ROWS, COLS = 40, 40
GENERATIONS = 16

# same positions as the example, already fixed (see the docstring above)
POSITIONS = {
    "glider":  (8, 8),
    "blinker": (20, 5),
    "beacon":  (9, 33),
    "toad":    (29, 15),
    "block":   (19, 19),
    "pulsar":  (4, 20),
}


def _place(grid: np.ndarray, pattern: list[list[int]], top: int, left: int) -> None:
    arr = np.array(pattern)
    h, w = arr.shape
    grid[top:top + h, left:left + w] = arr


def _initial_grid() -> np.ndarray:
    grid = np.zeros((ROWS, COLS), dtype=np.int8)
    for name, (top, left) in POSITIONS.items():
        _place(grid, PATTERNS[name], top, left)
    return grid


def test_no_pattern_overlaps():
    """Confirm the positions do not collide -- if they did, place() would
    silently write one pattern over another, with no error at all
    (exactly the bug found with the original beacon at (9,25), before
    the fix)."""
    occupancy = np.zeros((ROWS, COLS), dtype=int)
    for name, (top, left) in POSITIONS.items():
        arr = np.array(PATTERNS[name])
        h, w = arr.shape
        region = occupancy[top:top + h, left:left + w]
        assert region.sum() == 0, f"{name} at ({top},{left}) collides with a pattern already placed"
        occupancy[top:top + h, left:left + w] += 1


class _GameOfLifeRuleMixin:
    def rule(self, arrays: dict) -> dict:
        state = arrays["state"]
        neighbors = self.backend.focal_sum_mask(state == 1)
        survive = (state == 1) & np.isin(neighbors, [2, 3])
        born = (state == 0) & (neighbors == 3)
        return {"state": np.where(survive | born, 1, 0).astype(np.int8)}


class _GoLMono(_GameOfLifeRuleMixin, RasterCellularAutomaton):
    pass


class _GoLHalo(_GameOfLifeRuleMixin, HaloChunkedRasterCellularAutomaton):
    pass


@pytest.mark.parametrize(
    "block_h, block_w, halo, label",
    [
        (10, 10, 1, "block_10x10_same_as_example"),
        (7, 13, 1, "irregular_block_not_aligned_to_patterns"),
        (5, 5, 2, "small_block_larger_halo"),
    ],
)
def test_classic_patterns_equivalence(block_h, block_w, halo, label):
    grid0 = _initial_grid()

    backend_mono = raster_grid(rows=ROWS, cols=COLS, attrs={"state": grid0.copy()})
    env_mono = Environment(start_time=0, end_time=GENERATIONS)
    _GoLMono(backend=backend_mono, state_attr="state")
    env_mono.run()
    golden = backend_mono.arrays["state"].copy()

    backend_halo = raster_grid(rows=ROWS, cols=COLS, attrs={"state": grid0.copy()})
    env_halo = Environment(start_time=0, end_time=GENERATIONS)
    _GoLHalo(backend=backend_halo, block_h=block_h, block_w=block_w, halo=halo, state_attr="state")
    env_halo.run()
    result = backend_halo.arrays["state"].copy()

    n_diff = int(np.sum(golden != result))
    assert n_diff == 0, f"[{label}] {n_diff}/{ROWS*COLS} cells differ"
