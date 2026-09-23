"""
Load arrays from a Zarr store directly into a MemmapRasterWorkspace,
block by block — the second data-input path, next to geotiff.py
(GeoTIFF/VRT).

Motivation: disscube (DisSModel/disscube) stores derived variables
natively in Zarr (`data/derived/{grid_id}/{tile_id}/{spec_hash}/
{variable_name}.zarr`), already aligned to the master grid by its
GridAligner (per-operator resampling and fine alignment for categorical
data — see its README). This module consumes that data directly,
without materializing it to GeoTIFF first.

Why a module separate from geotiff.py
-------------------------------------
Zarr is natively chunked — it needs neither rasterio.windows.Window
nor a VRT for partial reads; a zarr.Array supports direct slicing
(`arr[r0:r1, c0:c1]`) that reads only the chunks it needs from the
store. The public API mirrors geotiff.py (same way of declaring which
workspace arrays come from which source variable), so the two input
paths (GeoTIFF/VRT and Zarr) are interchangeable for code that uses
MemmapRasterWorkspace — swap the loader, and the rest of the pipeline
(halo, disk, models) stays the same.

Requires the optional "zarr" extra (zarr>=3). xarray/rioxarray are NOT
needed here — the raw zarr.Array is read by variable name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from ..workspace import MemmapRasterWorkspace


def _resolve_axis_order(arr, expected_names: tuple[str, ...]) -> tuple[int, ...] | None:
    """Use arr.metadata.dimension_names (the native Zarr v3 field, the
    same one xarray writes when saving a DataArray with to_zarr) to find
    the actual axis order of the array on disk, and return the transpose
    indices that bring it to expected_names.

    Why this is needed: xarray/disscube do NOT guarantee that the axis
    order written to disk is (y, x) — disscube's own CubeClient.load()
    calls `.transpose("y", "x")` defensively before using any array,
    precisely because the order may differ. A SQUARE array with swapped
    axes has the SAME shape either way — a shape check alone does not
    catch the swap; it is silent, nothing fails.

    Zarr v2-format arrays have no dimension_names: xarray stores the
    names in the `_ARRAY_DIMENSIONS` attribute, which is read as a
    fallback.

    Returns None if neither is available (a Zarr store without dimension
    metadata — there is nothing to check against, so the order is taken
    as it is).
    """
    dims = getattr(getattr(arr, "metadata", None), "dimension_names", None)
    if not dims:
        attrs = getattr(arr, "attrs", None)
        dims = attrs.get("_ARRAY_DIMENSIONS") if attrs is not None else None
    if not dims:
        return None
    dims = tuple(dims)
    if set(dims) != set(expected_names):
        return None  # unexpected dimension names -- do not risk reordering
    return tuple(dims.index(name) for name in expected_names)


def _open_variable(store: str, variable_name: str | None):
    """Open a Zarr store, which may be a group (several variables,
    accessed by name) or a single array (the variable itself)."""
    opened = zarr.open(store, mode="r")
    if hasattr(opened, "arrays") or hasattr(opened, "array_keys"):
        # A group (zarr.Group) — needs the name of the variable inside it.
        if variable_name is None:
            raise ValueError(
                f"'{store}' is a Zarr group with several variables — "
                f"give the variable name in variable_map."
            )
        return opened[variable_name]
    # A single array — variable_name is ignored (or used only as a label).
    return opened


def load_zarr_into_workspace(
    workspace: MemmapRasterWorkspace,
    store: str | Path,
    variable_map: dict[str, str] | None = None,
    time_index: int | None = None,
) -> None:
    """
    Fill a MemmapRasterWorkspace block by block from a Zarr store,
    reading only one block's slice at a time.

    Parameters
    ----------
    store : str | Path
        Path to the Zarr store — either a group (several variables,
        accessed by name) or a single array.
    variable_map : dict[workspace_array_name, zarr_variable_name], optional
        Maps workspace array names to variable names inside the Zarr
        group. If None, the names are assumed to match (same name in the
        workspace and in the store), and `store` is taken as a single
        array (not a group) when no variable is named.
    time_index : int, optional
        If the variable has a leading time dimension (shape
        (time, y, x), the layout of disscube's "Temporal Backend" for
        derived products with a validity window), which time index to
        load. Required when the variable is 3-D.

    Raises
    ------
    ValueError
        If the variable's (y, x) shape does not match workspace.shape,
        or if a 3-D variable is given without time_index.
    """
    if not HAS_ZARR:
        raise ImportError("zarr is required — pip install -e '.[zarr]'")

    store = str(store)
    declared = set(workspace.metadata["arrays"])
    var_map = variable_map or {name: name for name in declared}

    for array_name, zarr_var_name in var_map.items():
        if array_name not in declared:
            continue

        arr = _open_variable(store, zarr_var_name)

        if arr.ndim == 3:
            if time_index is None:
                raise ValueError(
                    f"Variable '{zarr_var_name}' has 3 dimensions (probably "
                    f"a time dimension) — give time_index."
                )
            expected_dims = ("time", "y", "x")
        elif arr.ndim == 2:
            expected_dims = ("y", "x")
        else:
            raise ValueError(
                f"Variable '{zarr_var_name}' has {arr.ndim} dimensions; expected 2 or 3."
            )

        # Normalize the axis order to (y, x) or (time, y, x) using the
        # store's dimension metadata, when available. Needed because the
        # written axis order is NOT guaranteed (see the docstring of
        # _resolve_axis_order) — a square array with swapped axes has the
        # same shape either way, so a shape check alone misses the swap.
        axis_order = _resolve_axis_order(arr, expected_dims)

        shape2d = arr.shape[1:] if arr.ndim == 3 else arr.shape
        if axis_order is not None:
            reordered_shape = tuple(arr.shape[i] for i in axis_order)
            shape2d = reordered_shape[1:] if arr.ndim == 3 else reordered_shape

        if shape2d != tuple(workspace.shape):
            raise ValueError(
                f"Shape of '{zarr_var_name}' {shape2d} does not match the "
                f"workspace shape {tuple(workspace.shape)}."
            )

        for block in workspace.blocks():
            disk_index: list = [slice(None)] * arr.ndim
            y_disk_axis = x_disk_axis = None
            time_disk_axis = None

            for canonical_pos, name in enumerate(expected_dims):
                disk_axis = axis_order[canonical_pos] if axis_order is not None else canonical_pos
                if name == "time":
                    disk_index[disk_axis] = time_index
                    time_disk_axis = disk_axis
                elif name == "y":
                    disk_index[disk_axis] = slice(block.r0, block.r1)
                    y_disk_axis = disk_axis
                elif name == "x":
                    disk_index[disk_axis] = slice(block.c0, block.c1)
                    x_disk_axis = disk_axis

            raw = np.asarray(arr[tuple(disk_index)])

            if axis_order is not None and raw.ndim == 2:
                # raw is still in the on-disk relative order (minus the
                # time axis, already dropped by the integer indexing
                # above) -- transpose to canonical (y, x).
                def _shift(pos, time_disk_axis=time_disk_axis):
                    return pos - 1 if (time_disk_axis is not None and time_disk_axis < pos) else pos
                data = np.transpose(raw, (_shift(y_disk_axis), _shift(x_disk_axis)))
            else:
                data = raw

            workspace.write_block_to_read_slot(block, array_name, data)

    workspace.flush()


def load_zarr_tiles_into_workspace(
    workspace: MemmapRasterWorkspace,
    tiles: list[dict],
    array: str | None = None,
    fill: float | None = None,
    skip_empty_blocks: bool = False,
) -> None:
    """
    Fill ONE workspace array from N Zarr stores placed side by side —
    the multi-tile case, which `load_zarr_into_workspace` does not cover
    (it takes one store and requires it to have the whole workspace
    shape).

    It is to Zarr what geomosaic's VRT is to GeoTIFF: it joins pieces
    into one continuous grid. The difference is that Zarr has no mosaic
    format, so the stitching happens here, at read time.

    Parameters
    ----------
    tiles : list[dict]
        One dictionary per piece, with the keys:

        - ``url``      — path of the Zarr store
        - ``variable`` — name of the variable inside the store
        - ``row_off``  — row, in pixels, where the piece starts in the workspace
        - ``col_off``  — column, in pixels
        - ``height``   — piece height, in pixels
        - ``width``    — piece width, in pixels

        This is exactly the format returned by disscube's
        ``CubeClient.tile_layout()``, but nothing here depends on
        disscube: any source that can give a path and a position works.
        Extra keys are ignored.
    array : str, optional
        Name of the WORKSPACE array to fill. If None, the ``variable`` of
        the first tile is used — handy when the names match.
    fill : float, optional
        Value for cells no tile covers (holes in the tiling, corners
        outside the study area). If None, NaN for floating-point arrays
        and 0 for integer ones.
    skip_empty_blocks : bool
        If True, blocks that no tile touches are not written. Since the
        workspace `.dat` files start sparse, those blocks then take no
        disk space — but they READ AS ZERO, not as `fill`. Use only when
        0 is not a valid value in the domain (see "Sparsity and disk
        cost" in the README). Default False, which writes `fill` and
        keeps the distinction at the cost of disk space.

    Raises
    ------
    ValueError
        If `tiles` is empty, if the array is not declared in the
        workspace, if a tile lacks a required key, or if a tile falls
        outside the workspace bounds — every case in which carrying on
        would produce a silently wrong mosaic.
    ImportError
        If the "zarr" extra is not installed.
    """
    if not HAS_ZARR:
        raise ImportError("zarr is required — pip install -e '.[zarr]'")
    if not tiles:
        raise ValueError("The list of tiles is empty — nothing to load.")

    required = {"url", "variable", "row_off", "col_off", "height", "width"}
    for i, t in enumerate(tiles):
        missing = required - set(t)
        if missing:
            raise ValueError(
                f"tiles[{i}] is missing the keys {sorted(missing)}; "
                f"every tile needs {sorted(required)}."
            )

    # Two pieces at the same position are not an ambiguity to resolve by
    # order: one would silently overwrite the other. It really happens when
    # the layout mixes time slices of the same variable — each year repeats
    # the same positions.
    occupied: dict[tuple[int, int], str] = {}
    for t in tiles:
        key = (t["row_off"], t["col_off"])
        if key in occupied:
            raise ValueError(
                f"Two tiles occupy position ({key[0]}, {key[1]}): "
                f"{occupied[key]!r} and {t.get('tile_id')!r}. One would overwrite "
                f"the other. If the layout mixes time slices, pick one "
                f"before loading."
            )
        occupied[key] = t.get("tile_id")

    array_name = array or tiles[0]["variable"]
    declared = set(workspace.metadata["arrays"])
    if array_name not in declared:
        raise ValueError(
            f"The workspace does not declare the array {array_name!r} "
            f"(declared: {sorted(declared)})."
        )

    height, width = workspace.shape
    for t in tiles:
        if (t["row_off"] < 0 or t["col_off"] < 0
                or t["row_off"] + t["height"] > height
                or t["col_off"] + t["width"] > width):
            raise ValueError(
                f"tile {t.get('tile_id')!r} at "
                f"({t['row_off']},{t['col_off']}) {t['height']}x{t['width']} "
                f"does not fit in the workspace {height}x{width}."
            )

    dtype = np.dtype(workspace.metadata["arrays"][array_name])
    if fill is None:
        fill = np.nan if np.issubdtype(dtype, np.floating) else 0

    opened = {}
    try:
        for t in tiles:
            key = (t["url"], t["variable"])
            if key not in opened:
                opened[key] = _open_variable(t["url"], t["variable"])

        # Iterate over the workspace BLOCKS, not the tiles: a block may span
        # two neighbouring tiles, or a hole in the tiling. Assembling it from
        # everything that covers it is what makes the stitching correct —
        # writing tile by tile would make the edges depend on the order.
        for block in workspace.blocks():
            buf = None
            for t in tiles:
                r0 = max(block.r0, t["row_off"])
                r1 = min(block.r1, t["row_off"] + t["height"])
                c0 = max(block.c0, t["col_off"])
                c1 = min(block.c1, t["col_off"] + t["width"])
                if r0 >= r1 or c0 >= c1:
                    continue

                if buf is None:
                    buf = np.full(
                        (block.r1 - block.r0, block.c1 - block.c0), fill, dtype=dtype
                    )

                arr = opened[(t["url"], t["variable"])]
                order = _resolve_axis_order(arr, ("y", "x"))
                piece = np.asarray(arr[
                    r0 - t["row_off"]:r1 - t["row_off"],
                    c0 - t["col_off"]:c1 - t["col_off"],
                ]) if order in (None, (0, 1)) else np.asarray(arr[
                    c0 - t["col_off"]:c1 - t["col_off"],
                    r0 - t["row_off"]:r1 - t["row_off"],
                ]).T

                buf[r0 - block.r0:r1 - block.r0, c0 - block.c0:c1 - block.c0] = piece

            if buf is None:
                if skip_empty_blocks:
                    continue
                buf = np.full(
                    (block.r1 - block.r0, block.c1 - block.c0), fill, dtype=dtype
                )

            workspace.write_block_to_read_slot(block, array_name, buf)
    finally:
        opened.clear()

    workspace.flush()
