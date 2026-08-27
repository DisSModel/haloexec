"""
Regressão: halo=1 é INSUFICIENTE para FloodModel — precisa de halo=2.

Achado: `FloodModel.execute()` computa, para cada célula, `viz_baixos`
(quantos vizinhos têm elevação <= a própria) e `fluxo` (derivado de
`viz_baixos`). O update de uma célula usa `fluxo_viz` — o fluxo do
VIZINHO — que por sua vez depende dos vizinhos DO VIZINHO. Essa é uma
dependência de 2 saltos, não 1: com halo=1, o `viz_baixos` calculado
no próprio anel de halo já está errado (seus vizinhos de 2 saltos
foram zero-padded pelo shift2d local), e esse erro contamina o núcleo
do bloco via `fluxo_viz`.

`MangroveModel`, em contraste, só verifica associação direta a um
conjunto (`eh_fonte_solo`/`eh_fonte_uso`) — dependência de 1 salto —
por isso halo=1 é suficiente para ele (confirmado nos testes de
test_mangrove_model_equivalence.py).

Por que os testes sintéticos anteriores (test_flood_model_equivalence.py)
não pegaram isso: usavam uma fonte de inundação (MAR) preenchendo uma
coluna INTEIRA na BORDA do domínio — o artefato de zero-padding do
shift2d já existe ali em ambos os lados (monolítico E halo, já que
ambos preenchem a borda VERDADEIRA do domínio com boundary_value),
então não há divergência. O bug só aparece quando uma célula-fonte
está perto de uma fronteira INTERNA de bloco, longe da borda do
domínio — o que só aconteceu ao validar contra o dataset real da Ilha
do Maranhão (ver examples/brmangue_validation/validate_against_terrame.py).

Esta fixture é um recorte 30x30 real desse dataset (elevacao_pol.zip),
em torno de uma célula que efetivamente diverge com halo=1 — não é
dado sintético, é uma prova recortada, determinística e rápida.
"""

from pathlib import Path

import numpy as np
import pytest

from dissmodel.core import Environment
from dissmodel.geo.raster.backend import RasterBackend

from brmangue.models.raster.flood_model import FloodModel
from brmangue.models.raster.mangrove_model import MangroveModel

from haloexec import HaloChunkedSyncRasterModel

FIXTURE = Path(__file__).parent / "fixtures" / "flood_model_halo_bug.npz"
BOUNDARY_VALUE = {"uso": 0, "alt": -9999.0, "solo": -1, "mask": 0}


class FloodModelHalo(HaloChunkedSyncRasterModel, FloodModel):
    pass


class MangroveModelHalo(HaloChunkedSyncRasterModel, MangroveModel):
    pass


def _load_fixture():
    data = np.load(FIXTURE)
    return data["uso"], data["alt"], data["solo"], data["mask"]


def _run(flood_cls, mangue_cls, uso, alt, solo, mask, generations, **kwargs):
    backend = RasterBackend(shape=uso.shape)
    backend.set("uso", uso.copy())
    backend.set("alt", alt.copy())
    backend.set("solo", solo.copy())
    backend.set("mask", mask.copy())

    env = Environment(start_time=1, end_time=generations)
    flood_cls(backend=backend, taxa_elevacao=0.05, **kwargs)
    mangue_cls(backend=backend, taxa_elevacao=0.05, altura_mare=6.0, **kwargs)
    env.run()

    return backend.get("alt").copy()


def test_halo_1_is_insufficient_for_flood_model():
    """Documenta o bug: halo=1 DEVE divergir nesta fixture. Se este
    teste passar a falhar (ou seja, halo=1 parar de divergir), a
    fixture pode ter perdido a propriedade que a torna útil como
    regressão — investigar antes de simplesmente apagar o teste."""
    uso, alt, solo, mask = _load_fixture()

    alt_mono = _run(FloodModel, MangroveModel, uso, alt, solo, mask, generations=5)
    alt_halo1 = _run(FloodModelHalo, MangroveModelHalo, uso, alt, solo, mask,
                      generations=5, block_h=10, block_w=10, halo=1,
                      boundary_value=BOUNDARY_VALUE)

    n_diff = int(np.sum(~np.isclose(alt_mono, alt_halo1, atol=1e-9)))
    assert n_diff > 0, (
        "halo=1 deixou de divergir nesta fixture — o bug de dependência "
        "de 2 saltos pode ter sido corrigido de outra forma, ou a fixture "
        "perdeu a propriedade que a tornava um caso de teste útil."
    )


@pytest.mark.parametrize("block_h,block_w", [(10, 10), (6, 6), (5, 25)])
def test_halo_2_fixes_flood_model(block_h, block_w):
    """halo=2 deve ser suficiente e produzir equivalência exata."""
    uso, alt, solo, mask = _load_fixture()

    alt_mono = _run(FloodModel, MangroveModel, uso, alt, solo, mask, generations=5)
    alt_halo2 = _run(FloodModelHalo, MangroveModelHalo, uso, alt, solo, mask,
                      generations=5, block_h=block_h, block_w=block_w, halo=2,
                      boundary_value=BOUNDARY_VALUE)

    assert np.allclose(alt_mono, alt_halo2, atol=1e-9)
