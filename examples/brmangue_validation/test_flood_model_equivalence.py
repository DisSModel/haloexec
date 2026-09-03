"""
Prova de equivalência matemática usando o FloodModel REAL do BR-MANGUE
(brmangue-dissmodel, sem nenhuma modificação de código-fonte).

FloodModelHalo herda de HaloChunkedSyncRasterModel + FloodModel via
herança múltipla cooperativa — nenhuma linha do FloodModel original
é alterada. Se este teste passar, a decomposição de domínio com halo
preserva exatamente o resultado do modelo hidrológico do BR-MANGUE,
célula a célula, independentemente do tamanho de bloco escolhido.

Nota de escopo: este teste verifica equivalência bloco-vs-monolítico
do comportamento ATUAL do FloodModel, não sua correção científica
(há pendências conhecidas de validação contra o golden TerraME,
registradas separadamente). Equivalência e correção são propriedades
independentes — testamos aqui apenas que "rodar em blocos" não muda
o que quer que o modelo já compute monoliticamente.
"""

import numpy as np
import pytest

from dissmodel.core import Environment
from dissmodel.geo.raster.backend import RasterBackend

from brmangue.models.raster.flood_model import FloodModel
from brmangue.common.constants import (
    MANGUE, VEGETACAO_TERRESTRE, AREA_ANTROPIZADA, SOLO_DESCOBERTO, MAR,
)

from haloexec import HaloChunkedSyncRasterModel


class FloodModelHalo(HaloChunkedSyncRasterModel, FloodModel):
    """Mesma lógica do FloodModel original, agora em blocos+halo.
    Nenhuma linha de FloodModel foi tocada."""
    pass


def _synthetic_grid(height: int, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Gera uso/alt sintéticos plausíveis: uma faixa de MAR na borda
    esquerda (fonte de inundação) e uso/altitude aleatórios no resto,
    para que a regra de propagação tenha algo real para propagar."""
    rng = np.random.default_rng(seed)

    usos_secos = [MANGUE, VEGETACAO_TERRESTRE, AREA_ANTROPIZADA, SOLO_DESCOBERTO]
    uso = rng.choice(usos_secos, size=(height, width)).astype(np.int16)
    uso[:, 0] = MAR  # fonte de inundação na borda esquerda

    alt = rng.uniform(-0.5, 2.0, size=(height, width)).astype(np.float32)
    alt[:, 0] = -1.0  # mar sempre abaixo do nível de referência

    return uso, alt


def _run(uso: np.ndarray, alt: np.ndarray, generations: int, model_cls, **model_kwargs) -> dict:
    backend = RasterBackend(shape=uso.shape)
    backend.set("uso", uso.copy())
    backend.set("alt", alt.copy())

    env = Environment(start_time=1, end_time=generations)
    model_cls(backend=backend, taxa_elevacao=0.05, **model_kwargs)
    env.run()

    return {
        "uso": backend.get("uso").copy(),
        "alt": backend.get("alt").copy(),
    }


@pytest.mark.parametrize(
    "height, width, block_h, block_w, generations, seed, label",
    [
        (30, 30, 10, 10, 15, 42, "grade_divisivel_exatamente"),
        (27, 41, 7, 11, 10, 7, "grade_com_resto_blocos_irregulares"),
        (25, 25, 1, 25, 8, 123, "blocos_de_1_linha_estresse_borda"),
        (18, 18, 100, 100, 8, 99, "bloco_maior_que_grade"),
    ],
)
def test_flood_model_equivalence(height, width, block_h, block_w, generations, seed, label):
    uso, alt = _synthetic_grid(height, width, seed)

    golden = _run(uso, alt, generations, FloodModel)
    blocked = _run(uso, alt, generations, FloodModelHalo, block_h=block_h, block_w=block_w, halo=1)

    n_diff_uso = int(np.sum(golden["uso"] != blocked["uso"]))
    n_diff_alt = int(np.sum(~np.isclose(golden["alt"], blocked["alt"], atol=1e-9)))

    assert n_diff_uso == 0, f"[{label}] 'uso' divergente em {n_diff_uso}/{height*width} celulas"
    assert n_diff_alt == 0, f"[{label}] 'alt' divergente em {n_diff_alt}/{height*width} celulas"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_flood_model_stress_random_seeds(seed):
    uso, alt = _synthetic_grid(22, 22, seed)

    golden = _run(uso, alt, generations=20, model_cls=FloodModel)
    blocked = _run(uso, alt, generations=20, model_cls=FloodModelHalo, block_h=5, block_w=5, halo=1)

    assert np.array_equal(golden["uso"], blocked["uso"])
    assert np.allclose(golden["alt"], blocked["alt"], atol=1e-9)
