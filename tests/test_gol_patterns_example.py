"""
Prova de equivalência para examples/gol_patterns/gol_patterns_haloexec.py:
padrões clássicos de Game of Life (glider, blinker, beacon, toad,
block, pulsar), posicionados deliberadamente sobre fronteiras de
bloco, devem produzir resultado IDÊNTICO entre a execução monolítica
(RasterCellularAutomaton puro) e em blocos+halo
(HaloChunkedRasterCellularAutomaton).

Isso é o que de fato sustenta a alegação do exemplo ("um padrão que
atravessa a borda de um bloco e continua se comportando como deveria
é a evidência visual mais direta de que o halo sincroniza
corretamente") -- rodar sem erro não prova isso; só a comparação
célula a célula prova.

Achado ao escrever este teste: as coordenadas originais do exemplo
tinham beacon (9,25, 4x4) caindo inteiro dentro da área do pulsar
(4,20, 13x13) -- sobreposição silenciosa, já que a função place()
sobrescreve por atribuição direta. Corrigido movendo beacon para a
coluna 33.
"""

import numpy as np
import pytest

pytest.importorskip("dissmodel_ca")

from dissmodel.core import Environment
from dissmodel.geo import raster_grid
from dissmodel.geo.raster.cellular_automaton import RasterCellularAutomaton

from haloexec import HaloChunkedRasterCellularAutomaton
from dissmodel_ca.models.game_of_life import PATTERNS

ROWS, COLS = 40, 40
GENERATIONS = 16

# mesmas posições do exemplo, já corrigidas (ver docstring acima)
POSICOES = {
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


def _grade_inicial() -> np.ndarray:
    grid = np.zeros((ROWS, COLS), dtype=np.int8)
    for nome, (top, left) in POSICOES.items():
        _place(grid, PATTERNS[nome], top, left)
    return grid


def test_nenhum_padrao_se_sobrepoe():
    """Confirma que as posições não colidem entre si -- se colidissem,
    place() sobrescreveria um padrão sobre o outro silenciosamente,
    sem erro nenhum (foi exatamente o bug encontrado com o beacon
    original em (9,25), antes da correção)."""
    ocupacao = np.zeros((ROWS, COLS), dtype=int)
    for nome, (top, left) in POSICOES.items():
        arr = np.array(PATTERNS[nome])
        h, w = arr.shape
        regiao = ocupacao[top:top + h, left:left + w]
        assert regiao.sum() == 0, f"{nome} em ({top},{left}) colide com outro padrão já posicionado"
        ocupacao[top:top + h, left:left + w] += 1


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
        (10, 10, 1, "bloco_10x10_mesmo_do_exemplo"),
        (7, 13, 1, "bloco_irregular_nao_alinhado_aos_padroes"),
        (5, 5, 2, "bloco_pequeno_halo_maior"),
    ],
)
def test_padroes_classicos_equivalencia(block_h, block_w, halo, label):
    grid0 = _grade_inicial()

    backend_mono = raster_grid(rows=ROWS, cols=COLS, attrs={"state": grid0.copy()})
    env_mono = Environment(start_time=0, end_time=GENERATIONS)
    _GoLMono(backend=backend_mono, state_attr="state")
    env_mono.run()
    golden = backend_mono.arrays["state"].copy()

    backend_halo = raster_grid(rows=ROWS, cols=COLS, attrs={"state": grid0.copy()})
    env_halo = Environment(start_time=0, end_time=GENERATIONS)
    _GoLHalo(backend=backend_halo, block_h=block_h, block_w=block_w, halo=halo, state_attr="state")
    env_halo.run()
    resultado = backend_halo.arrays["state"].copy()

    n_diff = int(np.sum(golden != resultado))
    assert n_diff == 0, f"[{label}] {n_diff}/{ROWS*COLS} células divergentes"
