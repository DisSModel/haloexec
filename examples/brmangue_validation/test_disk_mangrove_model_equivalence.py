"""
Prova de equivalência: MangroveModel REAL do BR-MANGUE, via disco
(MemmapRasterWorkspace + halo + double-buffer + "_past" bloco a
bloco), comparado ao monolítico. Nenhuma linha de MangroveModel é
modificada.

boundary_value por-array é OBRIGATÓRIO aqui — ver a nota em
haloexec.engine.resolve_boundary_value: 0 não é seguro como valor de
contorno para "solo" (SOLO_CANAL_FLUVIAL=0 é um código válido).
"""

import numpy as np
import pytest

from dissmodel.core import Environment

from brmangue.models.raster.mangrove_model import MangroveModel
from brmangue.common.constants import (
    MANGUE, VEGETACAO_TERRESTRE, SOLO_DESCOBERTO, SOLO_MANGUE, SOLO_OUTROS,
)

from haloexec import DiskChunkedSyncRasterModel, MemmapRasterWorkspace, workspace_arrays_for_sync_model
from dissmodel.geo.raster.backend import RasterBackend


class MangroveModelDiskHalo(DiskChunkedSyncRasterModel, MangroveModel):
    """Mesma lógica do MangroveModel original, agora em blocos+halo+disco.
    Nenhuma linha de MangroveModel foi tocada."""
    pass


BOUNDARY_VALUE = {"uso": 0, "alt": -9999.0, "solo": -1}


def _synthetic_grid(height, width, seed):
    rng = np.random.default_rng(seed)
    usos_alvo = [VEGETACAO_TERRESTRE, SOLO_DESCOBERTO]
    uso = rng.choice(usos_alvo, size=(height, width)).astype(np.int16)
    uso[:, 0] = MANGUE
    solo = rng.choice([SOLO_OUTROS], size=(height, width)).astype(np.int16)
    solo[:, 0] = SOLO_MANGUE
    alt = rng.uniform(0.0, 8.0, size=(height, width)).astype(np.float32)
    return uso, alt, solo


def _run_mono(uso, alt, solo, generations):
    backend = RasterBackend(shape=uso.shape)
    backend.set("uso", uso.copy())
    backend.set("alt", alt.copy())
    backend.set("solo", solo.copy())
    env = Environment(start_time=1, end_time=generations)
    MangroveModel(backend=backend, taxa_elevacao=0.05)
    env.run()
    return {"uso": backend.get("uso").copy(), "alt": backend.get("alt").copy(),
            "solo": backend.get("solo").copy()}


def _run_disk_halo(tmp_path, uso, alt, solo, generations, block_h, block_w, halo=1):
    arrays = workspace_arrays_for_sync_model(
        base={"uso": np.int16, "alt": np.float32, "solo": np.int16},
        land_use_types=["uso", "alt", "solo"],
    )
    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "mangrove_workspace",
        shape=uso.shape, arrays=arrays,
        block_h=block_h, block_w=block_w, halo=halo,
    )
    ws.fill("uso", uso)
    ws.fill("alt", alt)
    ws.fill("solo", solo)

    env = Environment(start_time=1, end_time=generations)
    MangroveModelDiskHalo(workspace=ws, taxa_elevacao=0.05, boundary_value=BOUNDARY_VALUE)
    env.run()
    ws.flush()

    return {"uso": ws.snapshot("uso"), "alt": ws.snapshot("alt"), "solo": ws.snapshot("solo")}


@pytest.mark.parametrize(
    "height, width, block_h, block_w, generations, seed, label",
    [
        (30, 30, 10, 10, 15, 42, "grade_divisivel_exatamente"),
        (27, 41, 7, 11, 10, 7, "grade_com_resto_blocos_irregulares"),
        (25, 25, 1, 25, 8, 123, "blocos_de_1_linha_estresse_borda"),
        (18, 18, 100, 100, 8, 99, "bloco_maior_que_grade"),
    ],
)
def test_mangrove_model_disk_halo_equivalence(tmp_path, height, width, block_h, block_w,
                                                generations, seed, label):
    uso, alt, solo = _synthetic_grid(height, width, seed)

    golden = _run_mono(uso, alt, solo, generations)
    disk = _run_disk_halo(tmp_path, uso, alt, solo, generations, block_h, block_w)

    n_diff_uso = int(np.sum(golden["uso"] != disk["uso"]))
    n_diff_solo = int(np.sum(golden["solo"] != disk["solo"]))
    n_diff_alt = int(np.sum(~np.isclose(golden["alt"], disk["alt"], atol=1e-9)))

    assert n_diff_uso == 0, f"[{label}] 'uso' divergente em {n_diff_uso}/{height*width} celulas"
    assert n_diff_solo == 0, f"[{label}] 'solo' divergente em {n_diff_solo}/{height*width} celulas"
    assert n_diff_alt == 0, f"[{label}] 'alt' divergente em {n_diff_alt}/{height*width} celulas"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_mangrove_model_disk_halo_stress_random_seeds(tmp_path, seed):
    uso, alt, solo = _synthetic_grid(22, 22, seed)

    golden = _run_mono(uso, alt, solo, generations=20)
    disk = _run_disk_halo(tmp_path, uso, alt, solo, generations=20, block_h=5, block_w=5)

    assert np.array_equal(golden["uso"], disk["uso"])
    assert np.array_equal(golden["solo"], disk["solo"])
    assert np.allclose(golden["alt"], disk["alt"], atol=1e-9)
