"""
Prova de equivalência matemática usando a maquinaria REAL do dissmodel
(Environment, RasterBackend, RasterCellularAutomaton) — não um harness
isolado.

A mesma regra de Game of Life é escrita uma única vez (via mixin) e
executada por duas classes base diferentes:

  - GameOfLifeMono   (dissmodel.RasterCellularAutomaton)      — monolítica
  - GameOfLifeHalo   (haloexec.HaloChunkedRasterCellularAutomaton) — em blocos

O resultado final deve ser IDÊNTICO célula a célula após N passos de
tempo, para qualquer decomposição de domínio válida. Isso prova que
trocar a classe base (a mudança real que o BR-MANGUE fará na migração)
não altera o resultado científico do modelo.
"""

import numpy as np
import pytest

from dissmodel.core import Environment
from dissmodel.geo.raster.backend import RasterBackend
from dissmodel.geo.raster.cellular_automaton import RasterCellularAutomaton

from haloexec import HaloChunkedRasterCellularAutomaton


class GameOfLifeRuleMixin:
    """Regra escrita uma única vez, reutilizada pelas duas classes base."""

    def rule(self, arrays):
        state = arrays["state"]
        neighbors = self.backend.focal_sum_mask(state == 1)
        born = (state == 0) & (neighbors == 3)
        survive = (state == 1) & np.isin(neighbors, [2, 3])
        return {"state": np.where(born | survive, 1, 0).astype(np.uint8)}


class GameOfLifeMono(GameOfLifeRuleMixin, RasterCellularAutomaton):
    pass


class GameOfLifeHalo(GameOfLifeRuleMixin, HaloChunkedRasterCellularAutomaton):
    pass


def _random_state(height, width, density, seed):
    rng = np.random.default_rng(seed)
    return (rng.random((height, width)) < density).astype(np.uint8)


def _run_mono(initial_state, generations):
    backend = RasterBackend(shape=initial_state.shape)
    backend.set("state", initial_state)
    env = Environment(start_time=1, end_time=generations)
    GameOfLifeMono(backend=backend)
    env.run()
    return backend.get("state").copy()


def _run_halo(initial_state, generations, block_h, block_w):
    backend = RasterBackend(shape=initial_state.shape)
    backend.set("state", initial_state)
    env = Environment(start_time=1, end_time=generations)
    GameOfLifeHalo(backend=backend, block_h=block_h, block_w=block_w, halo=1)
    env.run()
    return backend.get("state").copy()


@pytest.mark.parametrize(
    "height, width, block_h, block_w, generations, seed, label",
    [
        (40, 40, 10, 10, 20, 42, "grade_divisivel_exatamente"),
        (37, 53, 8, 12, 15, 7, "grade_com_resto_blocos_irregulares"),
        (30, 30, 1, 30, 10, 123, "blocos_de_1_linha_estresse_borda"),
        (20, 20, 100, 100, 10, 99, "bloco_maior_que_grade"),
    ],
)
def test_equivalence_dissmodel_real(height, width, block_h, block_w, generations, seed, label):
    initial = _random_state(height, width, density=0.35, seed=seed)

    golden = _run_mono(initial.copy(), generations)
    blocked = _run_halo(initial.copy(), generations, block_h, block_w)

    assert np.array_equal(golden, blocked), (
        f"[{label}] divergencia: {int(np.sum(golden != blocked))}/"
        f"{height * width} celulas diferentes"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_equivalence_stress_random_seeds(seed):
    initial = _random_state(25, 25, density=0.45, seed=seed)

    golden = _run_mono(initial.copy(), generations=30)
    blocked = _run_halo(initial.copy(), generations=30, block_h=6, block_w=6)

    assert np.array_equal(golden, blocked)
