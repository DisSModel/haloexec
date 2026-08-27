"""
Teste de integração haloexec <-> geomosaic (pacotes separados, sem
dependência de runtime entre si).

Prova que load_geotiff_into_workspace (haloexec) lê corretamente,
bloco a bloco, um mosaico produzido pelo geomosaic — inclusive para
blocos que cruzam a fronteira entre tiles. geomosaic é usado aqui só
como dependência de TESTE (extra "geomosaic"), nunca importado pelo
código de runtime do haloexec.
"""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
geomosaic = pytest.importorskip("geomosaic")
from rasterio.transform import from_origin

from haloexec import MemmapRasterWorkspace, load_geotiff_into_workspace


def _write_tile(path, data, origin_x, origin_y, px_size=100.0, crs="EPSG:31984"):
    transform = from_origin(origin_x, origin_y, px_size, px_size)
    with rasterio.open(
        str(path), "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype=str(data.dtype), crs=crs, transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_haloexec_reads_geomosaic_vrt_across_tile_boundary(tmp_path):
    rng = np.random.default_rng(7)
    ref = rng.integers(1, 100, size=(20, 20)).astype("int16")

    origin_x, origin_y, px = 500_000.0, 9_700_000.0, 100.0
    tiles = {
        "top_left": (ref[0:10, 0:10], origin_x, origin_y),
        "top_right": (ref[0:10, 10:20], origin_x + 10 * px, origin_y),
        "bottom_left": (ref[10:20, 0:10], origin_x, origin_y - 10 * px),
        "bottom_right": (ref[10:20, 10:20], origin_x + 10 * px, origin_y - 10 * px),
    }
    paths = []
    for name, (data, ox, oy) in tiles.items():
        p = tmp_path / f"{name}.tif"
        _write_tile(p, data, ox, oy)
        paths.append(p)

    contract = geomosaic.build_mosaic_contract(paths)
    vrt_path = geomosaic.write_vrt(contract, tmp_path / "mosaico.vrt")

    # blocos de 6x6 cruzam a fronteira do mosaico (coluna/linha 10)
    # várias vezes -- nenhum código especial de mosaico no haloexec.
    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(20, 20),
        arrays={"estado": np.int16}, block_h=6, block_w=6, halo=1,
    )
    load_geotiff_into_workspace(ws, vrt_path, [("estado", "int16", 0)])

    assert np.array_equal(ws.snapshot("estado"), ref)
