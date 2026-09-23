"""
Equivalence proof for sweep_until_convergence, using exactly the use
case that motivated the primitive: connectivity propagation
(binary_propagation) through blocks+halo+sweeps, compared with a
monolithic binary_propagation over the whole domain at once.

It confirms that the block version converges to exactly the same result
as the monolithic one.
"""

import numpy as np
import pytest

scipy_ndimage = pytest.importorskip("scipy.ndimage")
from scipy.ndimage import binary_propagation

from haloexec import MemmapRasterWorkspace, sweep_until_convergence


def _connectivity_rule(window: dict[str, np.ndarray], halo: int = 1) -> dict[str, np.ndarray]:
    """Dilate 'connected' through 'permeable' inside the halo window,
    and return only the core."""
    connected = window["connected"].astype(bool)
    permeable = window["permeable"].astype(bool)
    propagated = binary_propagation(connected, mask=permeable)
    core = propagated[halo:-halo, halo:-halo]
    return {"connected": core.astype(np.uint8)}


def _run_monolithic(seeds: np.ndarray, permeable: np.ndarray) -> np.ndarray:
    return binary_propagation(seeds.astype(bool), mask=permeable.astype(bool)).astype(np.uint8)


def _run_chunked(tmp_path, seeds: np.ndarray, permeable: np.ndarray,
                  block_h: int, block_w: int, halo: int = 1) -> tuple[np.ndarray, dict]:
    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=seeds.shape,
        arrays={"connected": np.uint8, "permeable": np.uint8},
        block_h=block_h, block_w=block_w, halo=halo,
    )
    ws.fill("connected", seeds.astype(np.uint8))
    ws.fill("permeable", permeable.astype(np.uint8))

    info = sweep_until_convergence(
        ws, lambda w: _connectivity_rule(w, halo), boundary_value=0,
    )
    ws.flush()
    return ws.snapshot("connected"), info


def _labyrinth_scenario(height: int, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a permeability labyrinth that forces connectivity to wind
    across several block boundaries before converging — it really tests
    propagation through many blocks, not just the immediate
    neighbourhood of a central source."""
    rng = np.random.default_rng(seed)
    permeable = rng.random((height, width)) < 0.65  # most cells are permeable
    seeds = np.zeros((height, width), dtype=bool)
    seeds[0, 0] = True  # single source, in the corner -- forces a long propagation
    permeable[0, 0] = True
    return seeds, permeable


@pytest.mark.parametrize(
    "height, width, block_h, block_w, seed, label",
    [
        (40, 40, 10, 10, 42, "grid_divides_exactly"),
        (37, 53, 8, 12, 7, "grid_with_remainder_irregular_blocks"),
        (30, 30, 6, 6, 123, "small_blocks_many_boundaries"),
        (20, 20, 100, 100, 99, "block_larger_than_grid"),
    ],
)
def test_sweep_until_convergence_equivalence(tmp_path, height, width, block_h, block_w, seed, label):
    seeds, permeable = _labyrinth_scenario(height, width, seed)

    golden = _run_monolithic(seeds, permeable)
    chunked, info = _run_chunked(tmp_path, seeds, permeable, block_h, block_w)

    n_diff = int(np.sum(golden != chunked))
    assert n_diff == 0, f"[{label}] {n_diff}/{height*width} cells differ (info={info})"
    assert info["converged"]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_sweep_until_convergence_stress_random_seeds(tmp_path, seed):
    seeds, permeable = _labyrinth_scenario(35, 35, seed)

    golden = _run_monolithic(seeds, permeable)
    chunked, _info = _run_chunked(tmp_path, seeds, permeable, block_h=7, block_w=7)

    assert np.array_equal(golden, chunked)


def test_sweep_until_convergence_raises_if_never_converges(tmp_path):
    """A rule that always 'changes' something (never settles) must raise
    RuntimeError, not hang in a silent loop."""
    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(10, 10),
        arrays={"counter": np.uint8}, block_h=5, block_w=5, halo=1,
    )
    ws.fill("counter", np.zeros((10, 10), dtype=np.uint8))

    def unstable_rule(window):
        # always increments -- never converges
        core = window["counter"][1:-1, 1:-1]
        return {"counter": (core + 1) % 250}

    with pytest.raises(RuntimeError, match="did not converge"):
        sweep_until_convergence(ws, unstable_rule, max_sweeps=3)


def test_sweep_until_convergence_reports_sweep_count(tmp_path):
    """A single isolated source cell (no permeable neighbour) converges
    in the first sweep -- a trivial case, a sanity check of the sweep
    counter."""
    seeds = np.zeros((10, 10), dtype=bool)
    seeds[5, 5] = True
    permeable = np.zeros((10, 10), dtype=bool)
    permeable[5, 5] = True  # isolated -- nowhere to propagate

    _, info = _run_chunked(tmp_path, seeds, permeable, block_h=5, block_w=5)
    assert info["sweeps"] == 1
