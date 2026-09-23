"""
haloexec <-> disscube integration test (separate packages, no runtime
dependency between them).

Proves that `load_zarr_tiles_into_workspace` correctly assembles a
mosaic of N Zarr stores placed side by side — including blocks whose
halo CROSSES the boundary between two files, and holes in the tiling.

It is the Zarr counterpart of what `test_geomosaic_integration.py`
proves for GeoTIFF/VRT. The difference matters: with a VRT, GDAL does
the stitching before haloexec comes in, so a positioning error would
already show when reading the VRT. Here the stitching is done by this
module, tile by tile — a wrong offset, or a missing tile, produces wrong
data at one boundary and nowhere else. That is why the tests aim
precisely at the boundaries.

The layout format (keys and semantics) is the contract with disscube's
`CubeClient.tile_layout()`; `test_layout_shape_matches_disscube` pins it
from both sides when disscube is installed.
"""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from haloexec import (
    Block,
    MemmapRasterWorkspace,
    load_zarr_tiles_into_workspace,
)

T = 8   # side of each tile — small, so the boundaries stay inspectable


def _write_tile(path, data):
    """Write a 2-D array as a Zarr store with dimension_names (y, x),
    the same way xarray.to_zarr writes it — which is how disscube writes."""
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_array(
        "v", shape=data.shape, dtype=str(data.dtype), dimension_names=("y", "x")
    )
    arr[:] = data
    return str(path)


def _ws(tmp_path, shape, block=4, halo=1, dtype="float64"):
    return MemmapRasterWorkspace.create(
        tmp_path / "ws", shape=shape, arrays={"v": dtype},
        block_h=block, block_w=block, halo=halo,
    )


def _quadrants(tmp_path, values):
    """Four TxT tiles in a 2Tx2T workspace. `values` gives the constant
    value of each quadrant: (NW, NE, SW, SE). None = missing tile (hole)."""
    pos = [(0, 0), (0, T), (T, 0), (T, T)]
    tiles = []
    for i, ((r, c), val) in enumerate(zip(pos, values)):
        if val is None:
            continue
        url = _write_tile(tmp_path / f"t{i}.zarr", np.full((T, T), val, dtype="float64"))
        tiles.append({
            "tile_id": f"t{i}", "variable": "v", "url": url,
            "row_off": r, "col_off": c, "height": T, "width": T,
        })
    return tiles


# ── basic assembly ───────────────────────────────────────────────────────────

def test_four_tiles_land_in_their_own_quadrants(tmp_path):
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(ws, _quadrants(tmp_path, (1.0, 2.0, 3.0, 4.0)))

    reopened = MemmapRasterWorkspace(tmp_path / "ws")
    full = np.empty((2 * T, 2 * T))
    for b in reopened.blocks():
        full[b.r0:b.r1, b.c0:b.c1] = reopened.read_block_core(b, "v")

    assert np.all(full[:T, :T] == 1.0), "NW quadrant"
    assert np.all(full[:T, T:] == 2.0), "NE quadrant"
    assert np.all(full[T:, :T] == 3.0), "SW quadrant"
    assert np.all(full[T:, T:] == 4.0), "SE quadrant"


def test_row_and_column_offsets_are_not_swapped(tmp_path):
    """Swapping row_off and col_off would transpose the mosaic — and with
    square tiles the shape would still be right, so only the content
    gives it away."""
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(ws, _quadrants(tmp_path, (1.0, 2.0, 3.0, 4.0)))
    reopened = MemmapRasterWorkspace(tmp_path / "ws")
    ne = reopened.read_block_core(Block(r0=0, r1=4, c0=T, c1=T + 4), "v")
    so = reopened.read_block_core(Block(r0=T, r1=T + 4, c0=0, c1=4), "v")
    assert np.all(ne == 2.0) and np.all(so == 3.0)


# ── boundaries: what this module can break on its own ───────────────────────

def test_halo_crosses_boundary_between_two_zarr_files(tmp_path):
    """The halo window must bring the value of the NEIGHBOURING tile —
    which came from another file — not the tile's own value nor nodata."""
    ws = _ws(tmp_path, (2 * T, 2 * T), block=4, halo=1)
    load_zarr_tiles_into_workspace(ws, _quadrants(tmp_path, (1.0, 2.0, 3.0, 4.0)))
    reopened = MemmapRasterWorkspace(tmp_path / "ws")

    # block touching the vertical boundary: its right halo falls in the NE tile
    blk = Block(r0=0, r1=4, c0=T - 4, c1=T)
    win = reopened.read_block_with_halo(blk, boundary_value=np.nan)["v"]
    assert np.all(win[1:-1, 1:-1] == 1.0), "the core belongs to the NW tile"
    assert np.all(win[1:-1, -1] == 2.0), "the right halo must come from the NE tile"


def test_halo_crosses_horizontal_boundary(tmp_path):
    ws = _ws(tmp_path, (2 * T, 2 * T), block=4, halo=1)
    load_zarr_tiles_into_workspace(ws, _quadrants(tmp_path, (1.0, 2.0, 3.0, 4.0)))
    reopened = MemmapRasterWorkspace(tmp_path / "ws")
    blk = Block(r0=T - 4, r1=T, c0=0, c1=4)
    win = reopened.read_block_with_halo(blk, boundary_value=np.nan)["v"]
    assert np.all(win[1:-1, 1:-1] == 1.0)
    assert np.all(win[-1, 1:-1] == 3.0), "the bottom halo must come from the SW tile"


def test_block_straddling_a_boundary_is_assembled_from_both_files(tmp_path):
    """A block that falls half in one tile and half in the other needs
    both halves — the case that breaks if assembly is done tile by tile."""
    ws = _ws(tmp_path, (2 * T, 2 * T), block=4, halo=1)
    load_zarr_tiles_into_workspace(ws, _quadrants(tmp_path, (1.0, 2.0, 3.0, 4.0)))
    reopened = MemmapRasterWorkspace(tmp_path / "ws")
    core_values = reopened.read_block_core(Block(r0=0, r1=4, c0=T - 2, c1=T + 2), "v")
    assert np.all(core_values[:, :2] == 1.0) and np.all(core_values[:, 2:] == 2.0)


# ── holes in the tiling ──────────────────────────────────────────────────────

def test_missing_tile_becomes_fill_not_garbage(tmp_path):
    """A real hole in the tiling (MapBiomas exports have several) becomes nodata."""
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(ws, _quadrants(tmp_path, (1.0, 2.0, None, 4.0)))
    reopened = MemmapRasterWorkspace(tmp_path / "ws")
    hole = reopened.read_block_core(Block(r0=T, r1=T + 4, c0=0, c1=4), "v")
    assert np.all(np.isnan(hole))


def test_halo_over_a_hole_is_fill_while_core_keeps_data(tmp_path):
    ws = _ws(tmp_path, (2 * T, 2 * T), block=4, halo=1)
    load_zarr_tiles_into_workspace(ws, _quadrants(tmp_path, (1.0, 2.0, None, 4.0)))
    reopened = MemmapRasterWorkspace(tmp_path / "ws")
    blk = Block(r0=T - 4, r1=T, c0=0, c1=4)          # last block of the NW tile
    win = reopened.read_block_with_halo(blk, boundary_value=0.0)["v"]
    assert np.all(win[1:-1, 1:-1] == 1.0), "the core keeps the data"
    assert np.all(np.isnan(win[-1, 1:-1])), "the halo falls in the hole -> fill"


def test_explicit_fill_value_is_used(tmp_path):
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(
        ws, _quadrants(tmp_path, (1.0, 2.0, None, 4.0)), fill=-9999.0
    )
    reopened = MemmapRasterWorkspace(tmp_path / "ws")
    assert np.all(reopened.read_block_core(Block(r0=T, r1=T + 4, c0=0, c1=4), "v") == -9999.0)


def test_skip_empty_blocks_leaves_them_zero(tmp_path):
    """Disk-saving mode: an empty block is not written, so it reads zero
    (not `fill`) — the trade-off documented in the README."""
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(
        ws, _quadrants(tmp_path, (1.0, 2.0, None, 4.0)), skip_empty_blocks=True
    )
    reopened = MemmapRasterWorkspace(tmp_path / "ws")
    assert np.all(reopened.read_block_core(Block(r0=T, r1=T + 4, c0=0, c1=4), "v") == 0.0)


# ── contract and errors ──────────────────────────────────────────────────────

def test_single_tile_covering_the_grid_also_works(tmp_path):
    """A global variable (no tiles) arrives as a one-item layout."""
    ws = _ws(tmp_path, (T, T))
    url = _write_tile(tmp_path / "g.zarr", np.arange(T * T, dtype="float64").reshape(T, T))
    load_zarr_tiles_into_workspace(ws, [{
        "tile_id": None, "variable": "v", "url": url,
        "row_off": 0, "col_off": 0, "height": T, "width": T,
    }])
    reopened = MemmapRasterWorkspace(tmp_path / "ws")
    assert reopened.read_block_core(Block(r0=0, r1=2, c0=0, c1=2), "v")[0, 0] == 0.0


def test_array_name_can_differ_from_variable_name(tmp_path):
    ws = MemmapRasterWorkspace.create(
        tmp_path / "ws", shape=(T, T), arrays={"uso": "float64"},
        block_h=4, block_w=4, halo=1,
    )
    url = _write_tile(tmp_path / "g.zarr", np.ones((T, T)))
    load_zarr_tiles_into_workspace(ws, [{
        "tile_id": None, "variable": "v", "url": url,
        "row_off": 0, "col_off": 0, "height": T, "width": T,
    }], array="uso")
    assert np.all(MemmapRasterWorkspace(tmp_path / "ws")
                  .read_block_core(Block(r0=0, r1=4, c0=0, c1=4), "uso") == 1.0)


def test_empty_tile_list_raises(tmp_path):
    ws = _ws(tmp_path, (T, T))
    with pytest.raises(ValueError, match="empty"):
        load_zarr_tiles_into_workspace(ws, [])


def test_missing_key_names_what_is_missing(tmp_path):
    ws = _ws(tmp_path, (T, T))
    with pytest.raises(ValueError, match="row_off"):
        load_zarr_tiles_into_workspace(ws, [{"url": "x", "variable": "v"}])


def test_undeclared_array_raises(tmp_path):
    ws = _ws(tmp_path, (T, T))
    url = _write_tile(tmp_path / "g.zarr", np.ones((T, T)))
    with pytest.raises(ValueError, match="does not declare"):
        load_zarr_tiles_into_workspace(ws, [{
            "tile_id": None, "variable": "nonexistent", "url": url,
            "row_off": 0, "col_off": 0, "height": T, "width": T,
        }])


def test_tile_outside_workspace_raises(tmp_path):
    """A tile out of bounds means a wrong grid — failing loudly avoids a
    truncated mosaic that would go unnoticed."""
    ws = _ws(tmp_path, (T, T))
    url = _write_tile(tmp_path / "g.zarr", np.ones((T, T)))
    with pytest.raises(ValueError, match="does not fit"):
        load_zarr_tiles_into_workspace(ws, [{
            "tile_id": "outside", "variable": "v", "url": url,
            "row_off": T, "col_off": 0, "height": T, "width": T,
        }])


# ── the contract with disscube, when it is available ────────────────────────

def test_layout_shape_matches_disscube(tmp_path):
    """Pins that the keys this loader requires are the ones
    CubeClient.tile_layout() produces. Skipped without disscube."""
    pytest.importorskip("disscube")
    from disscube.client import CubeClient
    from disscube.models import DerivedVariable, GridSpec, SpatialSource

    cube = CubeClient(catalog=str(tmp_path / "c.db"), store=str(tmp_path / "s"))
    cube.register_grid(GridSpec(
        id="G", type="local", crs="EPSG:31982", resolution=10.0,
        bbox=[0.0, 0.0, 100.0, 100.0],
    ))
    cube.register_spatial_source(SpatialSource(
        id="G_T1", name="T1", format="raster", asset_url="planned",
        crs="EPSG:31982", bbox=[0.0, 50.0, 50.0, 100.0],
    ))
    cube.catalog.save_derived(DerivedVariable(
        id="v_T1", name="v", grid_id="G", role="test", times=[], dtype="float64",
        derivation_id="d", spec_hash="h", tile_id="T1", asset_url="x.zarr",
    ))

    required = {"url", "variable", "row_off", "col_off", "height", "width"}
    assert required <= set(cube.tile_layout("v", "G")[0])


# ── overlapping positions ────────────────────────────────────────────────────
# Defence in depth: even if the layout comes out wrong from any source, two
# pieces at the same position must not be accepted. The real case behind
# this: a temporal variable whose layout mixed the slices of several years,
# all at the same positions — loading it would let the last year win, with
# no error at all.

def test_two_tiles_at_the_same_position_raise(tmp_path):
    ws = _ws(tmp_path, (T, T))
    a = _write_tile(tmp_path / "a.zarr", np.ones((T, T)))
    b = _write_tile(tmp_path / "b.zarr", np.full((T, T), 2.0))
    base = {"variable": "v", "row_off": 0, "col_off": 0, "height": T, "width": T}
    with pytest.raises(ValueError, match="occupy position"):
        load_zarr_tiles_into_workspace(ws, [
            {**base, "tile_id": "1985", "url": a},
            {**base, "tile_id": "1995", "url": b},
        ])


def test_overlap_error_names_both_tiles(tmp_path):
    ws = _ws(tmp_path, (T, T))
    a = _write_tile(tmp_path / "a.zarr", np.ones((T, T)))
    base = {"variable": "v", "row_off": 0, "col_off": 0, "height": T, "width": T}
    with pytest.raises(ValueError) as exc:
        load_zarr_tiles_into_workspace(ws, [
            {**base, "tile_id": "first", "url": a},
            {**base, "tile_id": "second", "url": a},
        ])
    assert "first" in str(exc.value) and "second" in str(exc.value)


def test_distinct_positions_still_accepted(tmp_path):
    """Guard: the check must not reject a legitimate mosaic."""
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(ws, _quadrants(tmp_path, (1.0, 2.0, 3.0, 4.0)))
