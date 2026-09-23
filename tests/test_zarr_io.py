"""
Tests for disk/io/zarr.py: loading Zarr (multi-variable group, single
array, and with a time dimension — the disscube layout) straight into a
MemmapRasterWorkspace, block by block.
"""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from haloexec import MemmapRasterWorkspace, load_zarr_into_workspace


def test_load_zarr_group_multi_variable(tmp_path):
    """Mimic the disscube layout: a group with several variables
    (e.g. 'uso', 'alt'), each accessed by name."""
    rng = np.random.default_rng(1)
    uso = rng.integers(1, 9, size=(20, 20)).astype("int16")
    alt = rng.uniform(-2.0, 8.0, size=(20, 20)).astype("float32")

    store_path = str(tmp_path / "group.zarr")
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
    """The workspace array name differs from the variable name in the store."""
    rng = np.random.default_rng(2)
    data = rng.integers(0, 100, size=(15, 15)).astype("int32")

    store_path = str(tmp_path / "group.zarr")
    root = zarr.open_group(store_path, mode="w")
    root.create_array("dist_to_towns", shape=(15, 15), dtype="int32")
    root["dist_to_towns"][:] = data

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(15, 15),
        arrays={"distance": np.int32}, block_h=5, block_w=5, halo=1,
    )
    load_zarr_into_workspace(ws, store_path, variable_map={"distance": "dist_to_towns"})

    assert np.array_equal(ws.snapshot("distance"), data)


def test_load_zarr_single_array(tmp_path):
    """The store is a single array (no group)."""
    rng = np.random.default_rng(3)
    data = rng.integers(0, 10, size=(12, 12)).astype("uint8")

    store_path = str(tmp_path / "single.zarr")
    za = zarr.open_array(store_path, mode="w", shape=(12, 12), dtype="uint8")
    za[:] = data

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(12, 12),
        arrays={"state": np.uint8}, block_h=4, block_w=4, halo=1,
    )
    load_zarr_into_workspace(ws, store_path, variable_map={"state": None})

    assert np.array_equal(ws.snapshot("state"), data)


def test_load_zarr_temporal_variable(tmp_path):
    """3-D variable (time, y, x) — the layout of disscube's 'Temporal
    Backend' for derived products with a validity window."""
    rng = np.random.default_rng(4)
    series = rng.integers(0, 5, size=(3, 10, 10)).astype("int16")  # 3 years

    store_path = str(tmp_path / "temporal.zarr")
    root = zarr.open_group(store_path, mode="w")
    root.create_array("mangue", shape=(3, 10, 10), dtype="int16")
    root["mangue"][:] = series

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(10, 10),
        arrays={"mangue": np.int16}, block_h=4, block_w=4, halo=1,
    )
    load_zarr_into_workspace(ws, store_path, time_index=1)

    assert np.array_equal(ws.snapshot("mangue"), series[1])


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
    store_path = str(tmp_path / "group.zarr")
    root = zarr.open_group(store_path, mode="w")
    root.create_array("uso", shape=(30, 30), dtype="int16")
    root["uso"][:] = np.zeros((30, 30), dtype="int16")

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(20, 20),  # wrong shape on purpose
        arrays={"uso": np.int16}, block_h=5, block_w=5, halo=1,
    )
    with pytest.raises(ValueError, match="does not match"):
        load_zarr_into_workspace(ws, store_path)
