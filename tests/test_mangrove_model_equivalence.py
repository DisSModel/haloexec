"""
Prova de equivalência matemática usando o MangroveModel REAL do
BR-MANGUE (brmangue-dissmodel, sem nenhuma modificação de código-fonte).

Mesma estratégia validada com FloodModel: MangroveModelHalo herda de
HaloChunkedSyncRasterModel + MangroveModel via herança múltipla
cooperativa — nenhuma linha do MangroveModel original é alterada.

Nota de escopo: equivalência bloco-vs-monolítico, não correção
científica contra o golden TerraME (fora do escopo deste pacote).
"""

import numpy as np
import pytest

from dissmodel.core import Environment
from dissmodel.geo.raster.backend import RasterBackend

from brmangue.models.raster.mangrove_model import MangroveModel
from brmangue.common.constants import (
    MANGUE, MANGUE_MIGRADO, VEGETACAO_TERRESTRE, SOLO_DESCOBERTO,
    SOLO_MANGUE, SOLO_MANGUE_MIGRADO, SOLO_CANAL_FLUVIAL, SOLO_OUTROS,
)

from haloexec import HaloChunkedSyncRasterModel


class MangroveModelHalo(HaloChunkedSyncRasterModel, MangroveModel):
    """Mesma lógica do MangroveModel original, agora em blocos+halo.
    Nenhuma linha de MangroveModel foi tocada."""
    pass


def _synthetic_grid(height: int, width: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uso/solo com fontes de migração (MANGUE/SOLO_MANGUE) e alvos
    disponíveis (VEGETACAO_TERRESTRE/SOLO_OUTROS), alt em faixa que
    cruza o zi (altura_mare + nivel_mar) para gerar transições reais."""
    rng = np.random.default_rng(seed)

    usos_alvo = [VEGETACAO_TERRESTRE, SOLO_DESCOBERTO]
    uso = rng.choice(usos_alvo, size=(height, width)).astype(np.int16)
    uso[:, 0] = MANGUE  # fonte de migração de uso na borda esquerda

    solos_alvo = [SOLO_OUTROS]
    solo = rng.choice(solos_alvo, size=(height, width)).astype(np.int16)
    solo[:, 0] = SOLO_MANGUE  # fonte de migração de solo na borda esquerda

    alt = rng.uniform(0.0, 8.0, size=(height, width)).astype(np.float32)

    return uso, alt, solo


def _run(uso, alt, solo, generations: int, model_cls, boundary_value=None, **kwargs) -> dict:
    backend = RasterBackend(shape=uso.shape)
    backend.set("uso", uso.copy())
    backend.set("alt", alt.copy())
    backend.set("solo", solo.copy())

    env = Environment(start_time=1, end_time=generations)
    if boundary_value is not None:
        kwargs["boundary_value"] = boundary_value
    model_cls(backend=backend, taxa_elevacao=0.05, **kwargs)
    env.run()

    return {
        "uso": backend.get("uso").copy(),
        "alt": backend.get("alt").copy(),
        "solo": backend.get("solo").copy(),
    }


# boundary_value alinhado ao nodata real de cada array (TIFF_BANDS):
# uso=0, alt=-9999.0, solo=-1 — 0 NÃO é seguro para "solo", pois
# SOLO_CANAL_FLUVIAL=0 é um código válido (ver achado documentado em
# engine.resolve_boundary_value). Só se aplica à variante Halo — o
# monolítico não faz padding, não recebe esse parâmetro.
BOUNDARY_VALUE = {"uso": 0, "alt": -9999.0, "solo": -1}


@pytest.mark.parametrize(
    "height, width, block_h, block_w, generations, seed, label",
    [
        (30, 30, 10, 10, 15, 42, "grade_divisivel_exatamente"),
        (27, 41, 7, 11, 10, 7, "grade_com_resto_blocos_irregulares"),
        (25, 25, 1, 25, 8, 123, "blocos_de_1_linha_estresse_borda"),
        (18, 18, 100, 100, 8, 99, "bloco_maior_que_grade"),
    ],
)
def test_mangrove_model_equivalence(height, width, block_h, block_w, generations, seed, label):
    uso, alt, solo = _synthetic_grid(height, width, seed)

    golden = _run(uso, alt, solo, generations, MangroveModel)
    blocked = _run(uso, alt, solo, generations, MangroveModelHalo,
                    block_h=block_h, block_w=block_w, halo=1, boundary_value=BOUNDARY_VALUE)

    n_diff_uso = int(np.sum(golden["uso"] != blocked["uso"]))
    n_diff_solo = int(np.sum(golden["solo"] != blocked["solo"]))
    n_diff_alt = int(np.sum(~np.isclose(golden["alt"], blocked["alt"], atol=1e-9)))

    assert n_diff_uso == 0, f"[{label}] 'uso' divergente em {n_diff_uso}/{height*width} celulas"
    assert n_diff_solo == 0, f"[{label}] 'solo' divergente em {n_diff_solo}/{height*width} celulas"
    assert n_diff_alt == 0, f"[{label}] 'alt' divergente em {n_diff_alt}/{height*width} celulas"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_mangrove_model_stress_random_seeds(seed):
    uso, alt, solo = _synthetic_grid(22, 22, seed)

    golden = _run(uso, alt, solo, generations=20, model_cls=MangroveModel)
    blocked = _run(uso, alt, solo, generations=20, model_cls=MangroveModelHalo,
                    block_h=5, block_w=5, halo=1, boundary_value=BOUNDARY_VALUE)

    assert np.array_equal(golden["uso"], blocked["uso"])
    assert np.array_equal(golden["solo"], blocked["solo"])
    assert np.allclose(golden["alt"], blocked["alt"], atol=1e-9)


@pytest.mark.parametrize("seed", [10, 11])
def test_mangrove_model_equivalence_com_acrecao(seed):
    """Caso com acrecao_ativa=True — exercita o ramo que lê alt_past
    e escreve alt, não coberto pelos casos default (acrecao_ativa=False)."""
    uso, alt, solo = _synthetic_grid(20, 20, seed)

    golden = _run(uso, alt, solo, generations=15, model_cls=MangroveModel,
                  acrecao_ativa=True)
    blocked = _run(uso, alt, solo, generations=15, model_cls=MangroveModelHalo,
                    block_h=6, block_w=6, halo=1, acrecao_ativa=True, boundary_value=BOUNDARY_VALUE)

    assert np.array_equal(golden["uso"], blocked["uso"])
    assert np.array_equal(golden["solo"], blocked["solo"])
    assert np.allclose(golden["alt"], blocked["alt"], atol=1e-6)
