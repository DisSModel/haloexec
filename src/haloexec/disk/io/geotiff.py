"""
Load a GeoTIFF directly into a MemmapRasterWorkspace, block by block,
through rasterio.windows.Window — a whole band is never materialized in
RAM.

Bands are described with the `band_spec` convention already used by
`dissmodel.io.raster.load_geotiff` — a list of (name, dtype, nodata) —
instead of hard-coded band names, so any band layout works.

Why not use dissmodel.io.raster.load_geotiff directly
-----------------------------------------------------
That function (from the dissmodel package) reads each whole band at
once (`ds.read(i)`) into an in-RAM RasterBackend — right for the
HaloChunkedSyncRasterModel path (which materializes the global grid
anyway), but unsuitable for MemmapRasterWorkspace, whose whole purpose
is never to materialize the entire grid.

Requires rasterio (optional "geotiff" extra: pip install -e ".[geotiff]").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from ..workspace import MemmapRasterWorkspace


def load_geotiff_into_workspace(
    workspace: MemmapRasterWorkspace,
    path: str | Path,
    band_spec: list[tuple[str, str, float]],
) -> None:
    """
    Fill a MemmapRasterWorkspace block by block from a single GeoTIFF.
    Convenience shortcut over load_geotiffs_into_workspace (several
    files) for the common one-file case.

    Parameters
    ----------
    workspace : MemmapRasterWorkspace
        Already created with the GeoTIFF's shape (workspace.shape must
        match the file's (height, width)) and with the `band_spec` arrays
        declared in `arrays=` of `.create()`.
    path : str | Path
        Path of the local GeoTIFF.
    band_spec : list[(name, dtype, nodata)]
        Same format as dissmodel.io.raster.load_geotiff — file band 1
        maps to band_spec[0], band 2 to band_spec[1], and so on. Arrays
        whose name is not declared in the workspace are skipped (so a
        subset of the bands can be loaded).

    Note on nodata: this loader does NOT filter or replace nodata
    values — it copies the file's raw values. Use each band_spec entry's
    `nodata` to set the matching `boundary_value` for the halo layers
    (see the README finding on per-array boundary_value).
    """
    load_geotiffs_into_workspace(workspace, [(path, band_spec)])


def load_geotiffs_into_workspace(
    workspace: MemmapRasterWorkspace,
    sources: list[tuple[str | Path, list[tuple[str, str, float]]]],
) -> None:
    """
    Fill a MemmapRasterWorkspace block by block from SEVERAL GeoTIFF
    files, reading the SAME block window from each file at a time.

    Generalizes load_geotiff_into_workspace (a single file) to inputs
    split across several rasters that share one grid (e.g. a domain mask
    and a base layer read together, with consistent shape/CRS checked
    before processing). The block+halo engine (Block/make_blocks/
    halo_window) has always been agnostic to how many files feed the
    grid — a block is just a position (r0:r1, c0:c1); this function lets
    the loader match that generality.

    Parameters
    ----------
    workspace : MemmapRasterWorkspace
        Already created with a shape matching ALL input files.
    sources : list[(path, band_spec)]
        Each file contributes the arrays declared in its own band_spec
        (same format as load_geotiff_into_workspace). An array may come
        from any of the files — band_specs of different files do not
        need overlapping names.

    Raises
    ------
    ValueError
        If the files do not share the same shape, do not match the
        workspace shape, or have different CRSs (when a CRS is defined in
        more than one file).
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required — pip install -e '.[geotiff]'")

    declared = set(workspace.metadata["arrays"])
    datasets = [rasterio.open(str(path)) for path, _ in sources]

    try:
        ref_shape = (datasets[0].height, datasets[0].width)
        ref_crs = datasets[0].crs
        for ds, (path, _) in zip(datasets, sources):
            shape = (ds.height, ds.width)
            if shape != ref_shape:
                raise ValueError(
                    f"Inconsistent shape between files: {path} has {shape}, "
                    f"expected {ref_shape} (from the first file in the list)."
                )
            if ds.crs is not None and ref_crs is not None and ds.crs != ref_crs:
                raise ValueError(
                    f"Inconsistent CRS between files: {path} has {ds.crs}, "
                    f"expected {ref_crs} (from the first file in the list)."
                )

        if ref_shape != tuple(workspace.shape):
            raise ValueError(
                f"Shape of the GeoTIFFs {ref_shape} does not match the "
                f"workspace shape {tuple(workspace.shape)}."
            )

        for block in workspace.blocks():
            window = Window(block.c0, block.r0, block.c1 - block.c0, block.r1 - block.r0)
            for ds, (path, band_spec) in zip(datasets, sources):
                for band_index, (name, dtype, _nodata) in enumerate(band_spec, start=1):
                    if name not in declared or band_index > ds.count:
                        continue
                    arr = ds.read(band_index, window=window).astype(dtype)
                    workspace.write_block_to_read_slot(block, name, arr)
    finally:
        for ds in datasets:
            ds.close()

    workspace.flush()


def save_workspace_to_geotiff(
    workspace: MemmapRasterWorkspace,
    path: str | Path,
    bands: list[str] | list[tuple[str, str, float]],
    transform: Any = None,
    crs: Any = "EPSG:31984",
    compress: str = "lzw",
) -> None:
    """
    Write MemmapRasterWorkspace arrays to a GeoTIFF in windows (block
    by block), never materializing the whole grid in RAM.

    Parameters
    ----------
    workspace : MemmapRasterWorkspace
        Workspace whose current read slot is written.
    path : str | Path
        Path of the output GeoTIFF.
    bands : list[str] or list[(name, dtype, nodata)]
        Names of the arrays to write as bands (1, 2, ...).
    transform : Affine, optional
        rasterio geotransform. If None, a default one is used (with a
        warning).
    crs : CRS or str, default="EPSG:31984"
        Coordinate reference system.
    compress : str, default="lzw"
        GeoTIFF compression.

    Note
    ----
    GeoTIFF/GDAL requires a single dtype and a single nodata for ALL
    bands of a file (a limitation of the format, not of this code —
    tested empirically: GDAL collapses per-band nodata to one value,
    even when distinct values are passed). If `bands` mixes dtypes or
    nodata values, this function converts everything to the common dtype
    (the smallest type that holds them all, via `np.result_type`) and
    uses the FIRST band's nodata for the whole file — and emits a
    warning (`warnings.warn`) whenever that discards information, rather
    than doing it silently. If each array must keep its own
    dtype/nodata, write one GeoTIFF per array instead of a multi-band
    file.
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required — pip install -e '.[geotiff]'")

    import warnings

    from rasterio.transform import from_origin

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parsed_bands: list[tuple[str, str, float | None]] = []
    for item in bands:
        if isinstance(item, tuple):
            parsed_bands.append((item[0], item[1], item[2]))
        else:
            name = str(item)
            dtype_str = str(workspace.metadata["arrays"][name])
            parsed_bands.append((name, dtype_str, None))

    height, width = workspace.shape
    if transform is None:
        transform = from_origin(500_000.0, 9_700_000.0, 30.0, 30.0)
        warnings.warn(
            "save_workspace_to_geotiff: no `transform` was given — "
            "using an arbitrary default origin (500000, 9700000, 30x30 m). "
            "The resulting GeoTIFF will NOT be correctly georeferenced "
            "unless that origin matches the real domain. Pass "
            "`transform=` explicitly for real data.",
            stacklevel=2,
        )

    distinct_dtypes = {dtype for _, dtype, _ in parsed_bands}
    common_dtype = np.result_type(*(np.dtype(d) for d in distinct_dtypes))
    if len(distinct_dtypes) > 1:
        warnings.warn(
            f"save_workspace_to_geotiff: bands with different dtypes "
            f"{sorted(distinct_dtypes)} — GeoTIFF requires a single dtype per "
            f"file, converting everything to {common_dtype} (may increase "
            f"the file size and/or change the meaning of categorical "
            f"arrays). To keep the dtypes, write a separate GeoTIFF "
            f"per array.",
            stacklevel=2,
        )

    distinct_nodatas = {n for _, _, n in parsed_bands if n is not None}
    first_nodata = parsed_bands[0][2]
    if len(distinct_nodatas) > 1:
        warnings.warn(
            f"save_workspace_to_geotiff: bands with different nodata values "
            f"{sorted(distinct_nodatas)} — GeoTIFF supports only one nodata "
            f"per file (a GDAL limitation, tested empirically: per-band "
            f"values are silently collapsed). Using the first band's "
            f"nodata ({first_nodata!r}) for the whole file; the other "
            f"bands will not have the correct nodata.",
            stacklevel=2,
        )

    tiled_kwargs = {}
    if width >= 16 and height >= 16:
        bx = (min(workspace.block_w, width) // 16) * 16 or 16
        by = (min(workspace.block_h, height) // 16) * 16 or 16
        tiled_kwargs = {"tiled": True, "blockxsize": bx, "blockysize": by}
    else:
        tiled_kwargs = {"tiled": False}

    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(parsed_bands),
        dtype=str(common_dtype),
        crs=crs,
        transform=transform,
        nodata=first_nodata,
        compress=compress,
        **tiled_kwargs,
    ) as dst:
        for block in workspace.blocks():
            window = Window(block.c0, block.r0, block.c1 - block.c0, block.r1 - block.r0)
            for band_idx, (name, _dtype, _nodata) in enumerate(parsed_bands, start=1):
                core = workspace.read_block_core(block, name).astype(common_dtype)
                dst.write(core, band_idx, window=window)


