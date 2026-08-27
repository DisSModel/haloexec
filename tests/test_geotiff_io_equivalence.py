"""
Prova de equivalência: carregar um GeoTIFF bloco a bloco direto para
MemmapRasterWorkspace deve produzir exatamente os mesmos valores que
carregar o arquivo inteiro em memória (via rasterio puro, como
referência) — para cada array declarado, em toda a extensão do
domínio, incluindo blocos de borda com resto.
"""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from haloexec import MemmapRasterWorkspace, load_geotiff_into_workspace


BAND_SPEC = [
    ("uso", "int16", 0),
    ("alt", "float32", -9999.0),
    ("solo", "int16", -1),
]


def _write_test_geotiff(path, height, width, seed):
    rng = np.random.default_rng(seed)
    uso = rng.integers(1, 9, size=(height, width)).astype(np.int16)
    alt = rng.uniform(-2.0, 8.0, size=(height, width)).astype(np.float32)
    solo = rng.integers(0, 5, size=(height, width)).astype(np.int16)

    transform = from_origin(500_000.0, 9_700_000.0, 100.0, 100.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=3,
        dtype="float64", crs="EPSG:31984", transform=transform,
    ) as dst:
        dst.write(uso.astype("float64"), 1)
        dst.write(alt.astype("float64"), 2)
        dst.write(solo.astype("float64"), 3)

    return uso, alt, solo


@pytest.mark.parametrize(
    "height, width, block_h, block_w, seed, label",
    [
        (40, 40, 10, 10, 42, "grade_divisivel_exatamente"),
        (37, 53, 8, 12, 7, "grade_com_resto_blocos_irregulares"),
        (25, 25, 100, 100, 99, "bloco_maior_que_grade"),
    ],
)
def test_load_geotiff_into_workspace_equivalence(tmp_path, height, width, block_h, block_w, seed, label):
    tif_path = tmp_path / "test.tif"
    uso, alt, solo = _write_test_geotiff(tif_path, height, width, seed)

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace",
        shape=(height, width),
        arrays={"uso": np.int16, "alt": np.float32, "solo": np.int16},
        block_h=block_h, block_w=block_w, halo=1,
    )
    load_geotiff_into_workspace(ws, tif_path, BAND_SPEC)

    assert np.array_equal(ws.snapshot("uso"), uso), f"[{label}] 'uso' divergente"
    assert np.array_equal(ws.snapshot("solo"), solo), f"[{label}] 'solo' divergente"
    assert np.allclose(ws.snapshot("alt"), alt, atol=1e-5), f"[{label}] 'alt' divergente"


def test_load_geotiff_into_workspace_shape_mismatch_raises(tmp_path):
    tif_path = tmp_path / "test.tif"
    _write_test_geotiff(tif_path, 20, 20, seed=1)

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace",
        shape=(30, 30),  # shape errado de propósito
        arrays={"uso": np.int16, "alt": np.float32, "solo": np.int16},
        block_h=10, block_w=10, halo=1,
    )
    with pytest.raises(ValueError, match="não bate"):
        load_geotiff_into_workspace(ws, tif_path, BAND_SPEC)


def test_load_geotiff_into_workspace_partial_bands(tmp_path):
    """Só declarar 'uso' e 'alt' no workspace deve ignorar 'solo' sem erro."""
    tif_path = tmp_path / "test.tif"
    uso, alt, _solo = _write_test_geotiff(tif_path, 15, 15, seed=5)

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace",
        shape=(15, 15),
        arrays={"uso": np.int16, "alt": np.float32},  # sem "solo" de propósito
        block_h=5, block_w=5, halo=1,
    )
    load_geotiff_into_workspace(ws, tif_path, BAND_SPEC)

    assert np.array_equal(ws.snapshot("uso"), uso)
    assert np.allclose(ws.snapshot("alt"), alt, atol=1e-5)
