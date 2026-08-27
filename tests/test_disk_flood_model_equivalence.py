"""
Prova de equivalência final: FloodModel REAL do BR-MANGUE, rodando
inteiramente via disco (MemmapRasterWorkspace + halo + double-buffer,
incluindo sincronização "_past" bloco a bloco), comparado ao
monolítico em RAM. Nenhuma linha de FloodModel é modificada.

Este é o teste mais forte do pacote: junta as três camadas
(decomposição de domínio, halo, e armazenamento em disco) sobre o
modelo científico real, não um exemplo didático como Game of Life.
"""

import numpy as np
import pytest

from dissmodel.core import Environment
from dissmodel.geo.raster.backend import RasterBackend

from brmangue.models.raster.flood_model import FloodModel
from brmangue.common.constants import (
    MANGUE, VEGETACAO_TERRESTRE, AREA_ANTROPIZADA, SOLO_DESCOBERTO, MAR,
)

from haloexec import DiskChunkedSyncRasterModel, MemmapRasterWorkspace, workspace_arrays_for_sync_model


class FloodModelDiskHalo(DiskChunkedSyncRasterModel, FloodModel):
    """Mesma lógica do FloodModel original, agora em blocos+halo+disco.
    Nenhuma linha de FloodModel foi tocada."""
    pass


def _synthetic_grid(height: int, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    usos_secos = [MANGUE, VEGETACAO_TERRESTRE, AREA_ANTROPIZADA, SOLO_DESCOBERTO]
    uso = rng.choice(usos_secos, size=(height, width)).astype(np.int16)
    uso[:, 0] = MAR
    alt = rng.uniform(-0.5, 2.0, size=(height, width)).astype(np.float32)
    alt[:, 0] = -1.0
    return uso, alt


def _run_mono(uso: np.ndarray, alt: np.ndarray, generations: int) -> dict:
    backend = RasterBackend(shape=uso.shape)
    backend.set("uso", uso.copy())
    backend.set("alt", alt.copy())
    env = Environment(start_time=1, end_time=generations)
    FloodModel(backend=backend, taxa_elevacao=0.05)
    env.run()
    return {"uso": backend.get("uso").copy(), "alt": backend.get("alt").copy()}


def _run_disk_halo(tmp_path, uso: np.ndarray, alt: np.ndarray, generations: int,
                    block_h: int, block_w: int, halo: int = 1) -> dict:
    arrays = workspace_arrays_for_sync_model(
        base={"uso": np.int16, "alt": np.float32},
        land_use_types=["uso", "alt"],
    )
    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "flood_workspace",
        shape=uso.shape, arrays=arrays,
        block_h=block_h, block_w=block_w, halo=halo,
    )
    ws.fill("uso", uso)
    ws.fill("alt", alt)

    env = Environment(start_time=1, end_time=generations)
    FloodModelDiskHalo(workspace=ws, taxa_elevacao=0.05)
    env.run()
    ws.flush()

    return {"uso": ws.snapshot("uso"), "alt": ws.snapshot("alt")}


@pytest.mark.parametrize(
    "height, width, block_h, block_w, generations, seed, label",
    [
        (30, 30, 10, 10, 15, 42, "grade_divisivel_exatamente"),
        (27, 41, 7, 11, 10, 7, "grade_com_resto_blocos_irregulares"),
        (25, 25, 1, 25, 8, 123, "blocos_de_1_linha_estresse_borda"),
        (18, 18, 100, 100, 8, 99, "bloco_maior_que_grade"),
    ],
)
def test_flood_model_disk_halo_equivalence(tmp_path, height, width, block_h, block_w,
                                             generations, seed, label):
    uso, alt = _synthetic_grid(height, width, seed)

    golden = _run_mono(uso, alt, generations)
    disk = _run_disk_halo(tmp_path, uso, alt, generations, block_h, block_w)

    n_diff_uso = int(np.sum(golden["uso"] != disk["uso"]))
    n_diff_alt = int(np.sum(~np.isclose(golden["alt"], disk["alt"], atol=1e-9)))

    assert n_diff_uso == 0, f"[{label}] 'uso' divergente em {n_diff_uso}/{height*width} celulas"
    assert n_diff_alt == 0, f"[{label}] 'alt' divergente em {n_diff_alt}/{height*width} celulas"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_flood_model_disk_halo_stress_random_seeds(tmp_path, seed):
    uso, alt = _synthetic_grid(22, 22, seed)

    golden = _run_mono(uso, alt, generations=20)
    disk = _run_disk_halo(tmp_path, uso, alt, generations=20, block_h=5, block_w=5)

    assert np.array_equal(golden["uso"], disk["uso"])
    assert np.allclose(golden["alt"], disk["alt"], atol=1e-9)
