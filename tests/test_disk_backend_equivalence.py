"""
Prova de equivalência do MemmapRasterWorkspace: execução em blocos+halo
lidos do DISCO (memmap, double-buffer, sem materializar a grade
inteira em memória) deve ser idêntica à execução monolítica em RAM.

Este teste é deliberadamente independente de dissmodel — o workspace
em disco é genérico o suficiente para ser usado por qualquer framework
ou script solto, não só modelos dissmodel.
"""

from pathlib import Path

import numpy as np
import pytest

from haloexec.disk_backend import MemmapRasterWorkspace


def _game_of_life_rule(padded: dict[str, np.ndarray], halo: int = 1) -> dict[str, np.ndarray]:
    state = padded["state"]
    core = state[halo:-halo, halo:-halo]
    neighbor_count = (
        state[0:-2, 0:-2] + state[0:-2, 1:-1] + state[0:-2, 2:]
        + state[1:-1, 0:-2] + state[1:-1, 2:]
        + state[2:, 0:-2] + state[2:, 1:-1] + state[2:, 2:]
    )
    born = (core == 0) & (neighbor_count == 3)
    survive = (core == 1) & ((neighbor_count == 2) | (neighbor_count == 3))
    return {"state": (born | survive).astype(np.uint8)}


def _run_monolithic(initial: np.ndarray, generations: int, halo: int = 1) -> np.ndarray:
    state = initial.copy()
    for _ in range(generations):
        padded = np.pad(state, halo, mode="constant", constant_values=0)
        state = _game_of_life_rule({"state": padded}, halo)["state"]
    return state


def _run_disk_backed(tmp_path: Path, initial: np.ndarray, generations: int,
                      block_h: int, block_w: int, halo: int = 1) -> np.ndarray:
    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace",
        shape=initial.shape,
        arrays={"state": np.uint8},
        block_h=block_h, block_w=block_w, halo=halo,
    )
    ws.fill("state", initial)
    ws.fill("state", initial, slot="b")  # ambos os slots começam iguais

    for step in range(generations):
        for block in ws.blocks():
            window = ws.read_block_with_halo(block, boundary_value=0)
            result = _game_of_life_rule(window, halo)
            ws.write_block_core(block, result)
        ws.swap_buffers()
        ws.checkpoint(step)
    ws.flush()
    return ws.snapshot("state")


def _random_grid(height, width, density, seed):
    rng = np.random.default_rng(seed)
    return (rng.random((height, width)) < density).astype(np.uint8)


@pytest.mark.parametrize(
    "height, width, block_h, block_w, generations, seed, label",
    [
        (40, 40, 10, 10, 20, 42, "grade_divisivel_exatamente"),
        (37, 53, 8, 12, 15, 7, "grade_com_resto_blocos_irregulares"),
        (30, 30, 1, 30, 10, 123, "blocos_de_1_linha_estresse_borda"),
        (20, 20, 100, 100, 10, 99, "bloco_maior_que_grade"),
    ],
)
def test_disk_backend_equivalence(tmp_path, height, width, block_h, block_w, generations, seed, label):
    initial = _random_grid(height, width, density=0.35, seed=seed)

    golden = _run_monolithic(initial.copy(), generations)
    disk = _run_disk_backed(tmp_path, initial.copy(), generations, block_h, block_w)

    assert np.array_equal(golden, disk), (
        f"[{label}] divergencia: {int(np.sum(golden != disk))}/{height*width} celulas"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_disk_backend_stress_random_seeds(tmp_path, seed):
    initial = _random_grid(25, 25, density=0.45, seed=seed)

    golden = _run_monolithic(initial.copy(), generations=30)
    disk = _run_disk_backed(tmp_path, initial.copy(), generations=30, block_h=6, block_w=6)

    assert np.array_equal(golden, disk)
