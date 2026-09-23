"""
Regression: load_zarr_into_workspace must correctly read a Zarr store
whose on-disk axis order is NOT (y, x) -- a real scenario, not a
hypothetical one.

disscube (VariableWriter) writes through
`da.to_dataset(name=var_name).to_zarr(...)`, and disscube's own
CubeClient.load() calls `.transpose("y", "x")` defensively before using
any loaded array -- evidence that the on-disk axis order is NOT
guaranteed to be (y, x).

The danger: a SQUARE array with swapped (x, y) axes instead of (y, x)
has the SAME shape either way -- a shape check alone does not detect
the swap. Without the fix, rows/columns are silently corrupted, with no
error at all.

Reproduced here with real xarray, writing exactly as disscube's
VariableWriter does (to_dataset().to_zarr()), not with a direct
zarr.create_array() -- to capture the real dimension_names metadata
xarray writes (the native Zarr v3 field), the same information the fix
in disk/io/zarr.py uses to normalize.
"""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
xr = pytest.importorskip("xarray")

from haloexec import MemmapRasterWorkspace, load_zarr_into_workspace


def _write_like_disscube(path, data_yx: np.ndarray, dims: tuple[str, ...], var_name: str):
    """Write exactly as disscube's VariableWriter does:
    da.to_dataset(name=...).to_zarr(..., mode="w", consolidated=False).
    `dims` sets the axis order written to disk -- ("y","x") is the
    "correct"/expected case, ("x","y") the real dangerous one."""
    if dims == ("y", "x"):
        raw = data_yx
    elif dims == ("x", "y"):
        raw = data_yx.T
    else:
        raise ValueError(dims)
    da = xr.DataArray(raw, dims=dims, name=var_name)
    da.to_dataset(name=var_name).to_zarr(str(path), mode="w", consolidated=False)


def test_load_zarr_handles_yx_axis_order(tmp_path):
    """The 'correct' (y, x) case -- must keep working as always."""
    n = 6
    data = np.arange(n * n).reshape(n, n).astype("int16")
    store = tmp_path / "correct.zarr"
    _write_like_disscube(store, data, ("y", "x"), "uso")

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(n, n),
        arrays={"uso": np.int16}, block_h=3, block_w=3, halo=1,
    )
    load_zarr_into_workspace(ws, str(store), variable_map={"uso": "uso"})

    assert np.array_equal(ws.snapshot("uso"), data)


def test_load_zarr_handles_xy_axis_order_square_array(tmp_path):
    """DANGEROUS case: a SQUARE array written with (x, y) axes -- same
    shape as the correct case, but the data is physically transposed on
    disk. Without the dimension_names fix, this would pass the shape
    check and silently corrupt rows/columns."""
    n = 6
    # distinct values per row AND column, so a wrong transposition
    # produces an array MEASURABLY different from the original
    data = np.arange(n * n).reshape(n, n).astype("int16")
    store = tmp_path / "swapped.zarr"
    _write_like_disscube(store, data, ("x", "y"), "uso")

    # confirm the on-disk shape EQUALS the expected one (exactly what
    # makes the bug silent without the axis fix)
    root = zarr.open(str(store), mode="r")
    assert root["uso"].shape == data.shape, "test precondition: shapes must match"

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(n, n),
        arrays={"uso": np.int16}, block_h=3, block_w=3, halo=1,
    )
    load_zarr_into_workspace(ws, str(store), variable_map={"uso": "uso"})

    result = ws.snapshot("uso")
    assert np.array_equal(result, data), (
        "load_zarr_into_workspace read the array with rows/columns swapped -- "
        "regression of the (x,y) vs (y,x) axis-order bug"
    )


def test_load_zarr_handles_txy_axis_order_temporal(tmp_path):
    """Temporal (3-D) variable with a non-canonical axis order: (x, y, time)
    instead of (time, y, x)."""
    n = 5
    n_time = 3
    # (time, y, x) -- original values
    series = np.arange(n_time * n * n).reshape(n_time, n, n).astype("int16")
    # physically written in (x, y, time) order
    raw_on_disk = np.transpose(series, (2, 1, 0))  # from (time,y,x) to (x,y,time)

    da = xr.DataArray(raw_on_disk, dims=("x", "y", "time"), name="mangue")
    store = tmp_path / "temporal.zarr"
    da.to_dataset(name="mangue").to_zarr(str(store), mode="w", consolidated=False)

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(n, n),
        arrays={"mangue": np.int16}, block_h=2, block_w=2, halo=1,
    )
    load_zarr_into_workspace(ws, str(store), variable_map={"mangue": "mangue"}, time_index=1)

    assert np.array_equal(ws.snapshot("mangue"), series[1])


def test_load_zarr_handles_xy_axis_order_zarr_v2_format(tmp_path):
    """A store in Zarr v2 FORMAT (e.g. written by older xarray/disscube,
    or with zarr_format=2): there is no metadata.dimension_names; xarray
    keeps the names in the `_ARRAY_DIMENSIONS` attribute. Even when read
    with zarr-python 3, without the fallback to that attribute the square
    (x, y) array was loaded TRANSPOSED, silently."""
    n = 6
    data = np.arange(n * n).reshape(n, n).astype("int16")
    store = tmp_path / "v2.zarr"
    da = xr.DataArray(data.T, dims=("x", "y"), name="uso")
    da.to_dataset(name="uso").to_zarr(str(store), mode="w", consolidated=False, zarr_format=2)

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(n, n),
        arrays={"uso": np.int16}, block_h=3, block_w=3, halo=1,
    )
    load_zarr_into_workspace(ws, str(store), variable_map={"uso": "uso"})

    assert np.array_equal(ws.snapshot("uso"), data), (
        "Zarr v2-format store with (x, y) axes loaded transposed -- "
        "missing fallback to _ARRAY_DIMENSIONS"
    )
