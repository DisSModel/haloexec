"""
Regressão: FloodModel + MangroveModel combinados no MESMO
MemmapRasterWorkspace não podem deixar "_past" órfão.

Achado: FloodModel só gerencia land_use_types=["uso","alt"] — nunca
sincroniza "solo_past". MangroveModel gerencia ["uso","alt","solo"].
Quando os dois modelos compartilham um workspace em disco, cada
execute() termina com seu próprio ws.swap_buffers(). Se a reconciliação
de blocos EXCLUI arrays "_past" da escrita padrão (por assumir que
"cada modelo cuida do seu"), qualquer "_past" que o modelo ATUAL não
gerencia (aqui, "solo_past" durante a execução do Flood) fica órfão no
slot novo após o swap — nunca é levado adiante, permanece com o valor
zerado da alocação inicial do memmap. O próximo modelo (Mangrove) lê
esse "solo_past" órfão e produz resultado errado.

Isolado, cada modelo (testado sozinho em seu próprio workspace) SEMPRE
passava — por isso os testes anteriores (test_disk_flood_model_*.py,
test_disk_mangrove_model_*.py) nunca pegaram isso. Só apareceu ao
combinar os dois modelos no mesmo workspace, que é justamente como o
BR-MANGUE real funciona (Flood + Mangrove sempre rodam juntos).

Fix: a reconciliação de blocos em DiskChunkedSyncRasterModel.execute()
NÃO exclui mais "_past" da escrita — todo array presente na janela do
bloco é levado adiante através do swap, e o "_past" que cada modelo
gerencia é corretamente sobrescrito depois, em post_execute().
"""

import numpy as np
import pytest

from dissmodel.core import Environment
from dissmodel.geo.raster.backend import RasterBackend

from brmangue.models.raster.flood_model import FloodModel
from brmangue.models.raster.mangrove_model import MangroveModel
from brmangue.common.constants import (
    MANGUE, VEGETACAO_TERRESTRE, SOLO_DESCOBERTO, SOLO_MANGUE, SOLO_OUTROS, MAR,
)

from haloexec import (
    HaloChunkedSyncRasterModel, DiskChunkedSyncRasterModel,
    MemmapRasterWorkspace, workspace_arrays_for_sync_model,
)


class FloodModelHalo(HaloChunkedSyncRasterModel, FloodModel):
    pass


class MangroveModelHalo(HaloChunkedSyncRasterModel, MangroveModel):
    pass


class FloodModelDiskHalo(DiskChunkedSyncRasterModel, FloodModel):
    pass


class MangroveModelDiskHalo(DiskChunkedSyncRasterModel, MangroveModel):
    pass


BOUNDARY_VALUE = {"uso": 0, "alt": -9999.0, "solo": -1}


def _synthetic_grid(height, width, seed):
    rng = np.random.default_rng(seed)
    uso = rng.choice([VEGETACAO_TERRESTRE, SOLO_DESCOBERTO], size=(height, width)).astype(np.int16)
    uso[:, 0] = MAR
    solo = rng.choice([SOLO_OUTROS], size=(height, width)).astype(np.int16)
    solo[height // 2 - 2:height // 2 + 2, width // 2 - 2:width // 2 + 2] = SOLO_MANGUE
    alt = rng.uniform(-1.0, 3.0, size=(height, width)).astype(np.float32)
    alt[:, 0] = -1.0
    return uso, alt, solo


def _run_ram_combo(uso, alt, solo, generations, block_h, block_w, halo=2):
    backend = RasterBackend(shape=uso.shape)
    backend.set("uso", uso.copy())
    backend.set("alt", alt.copy())
    backend.set("solo", solo.copy())
    env = Environment(start_time=1, end_time=generations)
    FloodModelHalo(backend=backend, taxa_elevacao=0.05, block_h=block_h, block_w=block_w,
                    halo=halo, boundary_value=BOUNDARY_VALUE)
    MangroveModelHalo(backend=backend, taxa_elevacao=0.05, altura_mare=6.0,
                       block_h=block_h, block_w=block_w, halo=halo, boundary_value=BOUNDARY_VALUE)
    env.run()
    return {"uso": backend.get("uso").copy(), "alt": backend.get("alt").copy(),
            "solo": backend.get("solo").copy()}


def _run_disk_combo(tmp_path, uso, alt, solo, generations, block_h, block_w, halo=2):
    arrays = workspace_arrays_for_sync_model(
        base={"uso": np.int16, "alt": np.float32, "solo": np.int16},
        land_use_types=["uso", "alt", "solo"],
    )
    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "combo_workspace", shape=uso.shape, arrays=arrays,
        block_h=block_h, block_w=block_w, halo=halo,
    )
    ws.fill("uso", uso)
    ws.fill("alt", alt)
    ws.fill("solo", solo)

    env = Environment(start_time=1, end_time=generations)
    FloodModelDiskHalo(workspace=ws, taxa_elevacao=0.05, boundary_value=BOUNDARY_VALUE)
    MangroveModelDiskHalo(workspace=ws, taxa_elevacao=0.05, altura_mare=6.0,
                           boundary_value=BOUNDARY_VALUE)
    env.run()
    ws.flush()

    return {"uso": ws.snapshot("uso"), "alt": ws.snapshot("alt"), "solo": ws.snapshot("solo")}


@pytest.mark.parametrize(
    "height, width, block_h, block_w, generations, seed, label",
    [
        (20, 20, 6, 6, 10, 7, "combo_padrao"),
        (25, 25, 5, 5, 8, 3, "blocos_pequenos"),
        (18, 18, 100, 100, 6, 11, "bloco_maior_que_grade"),
    ],
)
def test_flood_and_mangrove_combined_on_disk_no_orphan_past(
    tmp_path, height, width, block_h, block_w, generations, seed, label
):
    uso, alt, solo = _synthetic_grid(height, width, seed)

    ram = _run_ram_combo(uso, alt, solo, generations, block_h, block_w)
    disk = _run_disk_combo(tmp_path, uso, alt, solo, generations, block_h, block_w)

    n_diff_uso = int(np.sum(ram["uso"] != disk["uso"]))
    n_diff_solo = int(np.sum(ram["solo"] != disk["solo"]))
    n_diff_alt = int(np.sum(~np.isclose(ram["alt"], disk["alt"], atol=1e-9)))

    assert n_diff_uso == 0, f"[{label}] 'uso' divergente em {n_diff_uso}/{height*width} celulas"
    assert n_diff_solo == 0, (
        f"[{label}] 'solo' divergente em {n_diff_solo}/{height*width} celulas — "
        f"possível regressão do bug de '_past' órfão (ver docstring do módulo)"
    )
    assert n_diff_alt == 0, f"[{label}] 'alt' divergente em {n_diff_alt}/{height*width} celulas"
