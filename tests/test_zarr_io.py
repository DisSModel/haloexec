"""
Testes de zarr_io.py: carregamento de Zarr (grupo multi-variável,
array único, e com dimensão temporal — padrão disscube) direto para
MemmapRasterWorkspace, bloco a bloco.
"""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from haloexec import MemmapRasterWorkspace, load_zarr_into_workspace


def test_load_zarr_group_multi_variable(tmp_path):
    """Simula o layout do disscube: um grupo com várias variáveis
    (ex.: 'uso', 'alt'), cada uma acessada por nome."""
    rng = np.random.default_rng(1)
    uso = rng.integers(1, 9, size=(20, 20)).astype("int16")
    alt = rng.uniform(-2.0, 8.0, size=(20, 20)).astype("float32")

    store_path = str(tmp_path / "grupo.zarr")
    root = zarr.open_group(store_path, mode="w")
    root.create_array("uso", shape=(20, 20), dtype="int16")
    root["uso"][:] = uso
    root.create_array("alt", shape=(20, 20), dtype="float32")
    root["alt"][:] = alt

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(20, 20),
        arrays={"uso": np.int16, "alt": np.float32}, block_h=6, block_w=6, halo=1,
    )
    load_zarr_into_workspace(ws, store_path)

    assert np.array_equal(ws.snapshot("uso"), uso)
    assert np.allclose(ws.snapshot("alt"), alt, atol=1e-6)


def test_load_zarr_group_with_variable_map(tmp_path):
    """Nome do array no workspace difere do nome da variável no zarr."""
    rng = np.random.default_rng(2)
    dado = rng.integers(0, 100, size=(15, 15)).astype("int32")

    store_path = str(tmp_path / "grupo.zarr")
    root = zarr.open_group(store_path, mode="w")
    root.create_array("dist_sedes", shape=(15, 15), dtype="int32")
    root["dist_sedes"][:] = dado

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(15, 15),
        arrays={"distancia": np.int32}, block_h=5, block_w=5, halo=1,
    )
    load_zarr_into_workspace(ws, store_path, variable_map={"distancia": "dist_sedes"})

    assert np.array_equal(ws.snapshot("distancia"), dado)


def test_load_zarr_single_array(tmp_path):
    """Store é um único array (sem grupo)."""
    rng = np.random.default_rng(3)
    dado = rng.integers(0, 10, size=(12, 12)).astype("uint8")

    store_path = str(tmp_path / "unico.zarr")
    za = zarr.open_array(store_path, mode="w", shape=(12, 12), dtype="uint8")
    za[:] = dado

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(12, 12),
        arrays={"estado": np.uint8}, block_h=4, block_w=4, halo=1,
    )
    load_zarr_into_workspace(ws, store_path, variable_map={"estado": None})

    assert np.array_equal(ws.snapshot("estado"), dado)


def test_load_zarr_temporal_variable(tmp_path):
    """Variável 3D (time, y, x) — padrão do 'Temporal Backend' do
    disscube para produtos derivados com janela de validade."""
    rng = np.random.default_rng(4)
    serie = rng.integers(0, 5, size=(3, 10, 10)).astype("int16")  # 3 anos

    store_path = str(tmp_path / "temporal.zarr")
    root = zarr.open_group(store_path, mode="w")
    root.create_array("mangue", shape=(3, 10, 10), dtype="int16")
    root["mangue"][:] = serie

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(10, 10),
        arrays={"mangue": np.int16}, block_h=4, block_w=4, halo=1,
    )
    load_zarr_into_workspace(ws, store_path, time_index=1)

    assert np.array_equal(ws.snapshot("mangue"), serie[1])


def test_load_zarr_temporal_without_time_index_raises(tmp_path):
    store_path = str(tmp_path / "temporal.zarr")
    root = zarr.open_group(store_path, mode="w")
    root.create_array("mangue", shape=(3, 10, 10), dtype="int16")
    root["mangue"][:] = np.zeros((3, 10, 10), dtype="int16")

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(10, 10),
        arrays={"mangue": np.int16}, block_h=4, block_w=4, halo=1,
    )
    with pytest.raises(ValueError, match="time_index"):
        load_zarr_into_workspace(ws, store_path)


def test_load_zarr_shape_mismatch_raises(tmp_path):
    store_path = str(tmp_path / "grupo.zarr")
    root = zarr.open_group(store_path, mode="w")
    root.create_array("uso", shape=(30, 30), dtype="int16")
    root["uso"][:] = np.zeros((30, 30), dtype="int16")

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(20, 20),  # shape errado de proposito
        arrays={"uso": np.int16}, block_h=5, block_w=5, halo=1,
    )
    with pytest.raises(ValueError, match="não bate"):
        load_zarr_into_workspace(ws, store_path)
