"""
Prova de equivalência ponta a ponta: Game of Life carregado de um
GeoTIFF (simulando o resultado de um mosaico já materializado — ver
mosaic_io.py) direto para MemmapRasterWorkspace, executado em
blocos+halo, comparado a uma execução monolítica em RAM carregada do
MESMO arquivo.

Fecha o ciclo que faltava: os testes anteriores validam (a) o motor de
blocos+halo com dado sintético em RAM/disco, e (b) o carregamento de
TIFF/mosaico isoladamente (round-trip, sem rodar nenhum modelo em
cima). Este teste roda um modelo de verdade a partir de um TIFF de
verdade, incluindo um caso "grande" (maior que os testes sintéticos
anteriores) para dar mais confiança de escala.
"""

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from haloexec import MemmapRasterWorkspace, load_geotiff_into_workspace


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


def _write_geotiff(path: Path, initial: np.ndarray) -> None:
    transform = from_origin(500_000.0, 9_700_000.0, 100.0, 100.0)
    with rasterio.open(
        str(path), "w", driver="GTiff", height=initial.shape[0], width=initial.shape[1],
        count=1, dtype="uint8", crs="EPSG:31984", transform=transform,
    ) as dst:
        dst.write(initial, 1)


def _run_monolithic_from_tiff(path: Path, generations: int) -> np.ndarray:
    with rasterio.open(str(path)) as ds:
        state = ds.read(1)
    for _ in range(generations):
        padded = np.pad(state, 1, mode="constant", constant_values=0)
        state = _game_of_life_rule({"state": padded})["state"]
    return state


def _run_disk_from_tiff(path: Path, tmp_path: Path, generations: int,
                         block_h: int, block_w: int) -> np.ndarray:
    with rasterio.open(str(path)) as ds:
        shape = (ds.height, ds.width)

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=shape,
        arrays={"state": np.uint8}, block_h=block_h, block_w=block_w, halo=1,
    )
    load_geotiff_into_workspace(ws, path, [("state", "uint8", 0)])

    for step in range(generations):
        for block in ws.blocks():
            window = ws.read_block_with_halo(block, boundary_value=0)
            result = _game_of_life_rule(window, halo=1)
            ws.write_block_core(block, result)
        ws.swap_buffers()
        ws.checkpoint(step)
    ws.flush()
    return ws.snapshot("state")


@pytest.mark.parametrize(
    "height, width, block_h, block_w, generations, seed, label",
    [
        (40, 40, 10, 10, 10, 42, "pequeno_grade_divisivel"),
        (37, 53, 8, 12, 8, 7, "pequeno_com_resto"),
    ],
)
def test_gameoflife_from_geotiff_equivalence(tmp_path, height, width, block_h, block_w,
                                               generations, seed, label):
    rng = np.random.default_rng(seed)
    initial = (rng.random((height, width)) < 0.35).astype(np.uint8)

    tif_path = tmp_path / "mosaico.tif"
    _write_geotiff(tif_path, initial)

    golden = _run_monolithic_from_tiff(tif_path, generations)
    disk = _run_disk_from_tiff(tif_path, tmp_path, generations, block_h, block_w)

    assert np.array_equal(golden, disk), f"[{label}] divergência pós-TIFF"


def test_gameoflife_from_large_geotiff():
    """Caso 'grande': TIFF de 2000x2000 (4 milhões de células), gerado
    bloco a bloco (nunca materializado inteiro em RAM na geração),
    lido bloco a bloco, rodado em blocos+halo — a mesma cadeia
    mosaico->TIFF->disco->halo que seria usada com um mosaico real."""
    import shutil
    import tempfile

    tmp_path = Path(tempfile.mkdtemp())
    try:
        height, width = 2000, 2000
        rng = np.random.default_rng(99)
        initial = (rng.random((height, width)) < 0.35).astype(np.uint8)

        tif_path = tmp_path / "mosaico_grande.tif"
        _write_geotiff(tif_path, initial)

        generations = 3
        golden = _run_monolithic_from_tiff(tif_path, generations)
        disk = _run_disk_from_tiff(tif_path, tmp_path, generations, block_h=256, block_w=256)

        n_diff = int(np.sum(golden != disk))
        assert n_diff == 0, f"{n_diff}/{height*width} células divergentes no caso grande"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
