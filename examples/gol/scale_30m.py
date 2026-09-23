"""
Scale test: ~30 million pixels, writing the TIFF in windows (never
materializing the whole grid in RAM), loading it block by block, and
running dissmodel_ca's real GameOfLife through
DiskChunkedRasterCellularAutomaton (haloexec) -- the same scale
challenge a real coastal model faces.

Compares against a monolithic reference AT THE SAME SCALE (30M uint8
cells fit in RAM as a single array, ~30 MB -- what must not happen is
materializing while WRITING/LOADING the large file, which is what this
script actually tests).
"""

import time
from pathlib import Path

import numpy as np
import rasterio
from dissmodel.core import Environment
from dissmodel.geo import raster_grid
from dissmodel_ca.models.game_of_life_raster import GameOfLife
from rasterio.transform import from_origin
from rasterio.windows import Window

from haloexec import DiskChunkedRasterCellularAutomaton, MemmapRasterWorkspace, load_geotiff_into_workspace


class GameOfLifeHalo(DiskChunkedRasterCellularAutomaton, GameOfLife):
    pass


def memory_mb() -> dict:
    """RssAnon (real heap) from /proc/self/status -- the metric that
    proves materialization, not VmRSS/ru_maxrss (which include the page
    cache of mapped files, always high with memmap without meaning a
    problem)."""
    values = {}
    with open("/proc/self/status") as f:
        for line in f:
            for key in ("VmRSS", "RssAnon", "RssFile"):
                if line.startswith(key + ":"):
                    values[key] = int(line.split()[1]) / 1024
    return values


def write_tiff_in_windows(path: Path, height: int, width: int, density: float,
                          seed: int, block: int = 512) -> None:
    """Write the GeoTIFF block by block through rasterio.windows.Window
    -- never allocates the whole (height, width) grid in RAM at once.
    Deterministic RNG per block position (reproducible)."""
    transform = from_origin(500_000.0, 9_700_000.0, 30.0, 30.0)
    master_rng = np.random.default_rng(seed)

    with rasterio.open(
        str(path), "w", driver="GTiff", height=height, width=width,
        count=1, dtype="uint8", crs="EPSG:31984", transform=transform,
        tiled=True, blockxsize=block, blockysize=block, compress="lzw",
    ) as dst:
        for r0 in range(0, height, block):
            r1 = min(r0 + block, height)
            for c0 in range(0, width, block):
                c1 = min(c0 + block, width)
                block_seed = int(master_rng.integers(0, 2**31 - 1)) ^ (r0 * 92821 + c0)
                rng = np.random.default_rng(block_seed & 0xFFFFFFFF)
                data = (rng.random((r1 - r0, c1 - c0)) < density).astype("uint8")
                dst.write(data, 1, window=Window(c0, r0, c1 - c0, r1 - r0))


def main():
    # ~30 million pixels
    HEIGHT, WIDTH = 5480, 5480  # 30,030,400 cells
    DENSITY = 0.35
    SEED = 42
    GENERATIONS = 5
    BLOCK = 256
    HALO = 1

    tmp = Path("/tmp/haloexec_scale_30m")
    tmp.mkdir(exist_ok=True)
    tif_path = tmp / "initial_state.tif"

    print(f"Grid: {HEIGHT}x{WIDTH} = {HEIGHT*WIDTH:,} cells "
          f"(~{HEIGHT*WIDTH/1024**2:.1f} MB per uint8 array)")

    # ── 1. write the TIFF in windows ────────────────────────────────
    rss_before = memory_mb()
    t0 = time.time()
    write_tiff_in_windows(tif_path, HEIGHT, WIDTH, DENSITY, SEED, block=BLOCK)
    t_write = time.time() - t0
    rss_after_write = memory_mb()
    print(f"\n[1/3] TIFF written in {t_write:.1f}s -- "
          f"RssAnon={rss_after_write['RssAnon']:.1f}MB "
          f"(delta since start: {rss_after_write['RssAnon']-rss_before['RssAnon']:.1f}MB)")

    # ── 2. load block by block into the workspace ───────────────────
    t0 = time.time()
    ws = MemmapRasterWorkspace.create(
        root=tmp / "workspace", shape=(HEIGHT, WIDTH),
        arrays={"state": np.uint8}, block_h=BLOCK, block_w=BLOCK, halo=HALO,
    )
    load_geotiff_into_workspace(ws, tif_path, [("state", "uint8", 0)])
    t_load = time.time() - t0
    rss_after_load = memory_mb()
    print(f"[2/3] Loaded in {t_load:.1f}s -- "
          f"RssAnon={rss_after_load['RssAnon']:.1f}MB "
          f"(delta since writing: {rss_after_load['RssAnon']-rss_after_write['RssAnon']:.1f}MB)")

    # ── 3. run GameOfLife on disk+halo, through dissmodel ───────────
    t0 = time.time()
    env = Environment(start_time=1, end_time=GENERATIONS)
    GameOfLifeHalo(workspace=ws, halo=HALO, boundary_value=0)
    env.run()
    ws.flush()
    t_run = time.time() - t0
    rss_after_run = memory_mb()
    print(f"[3/3] {GENERATIONS} generations in {t_run:.1f}s "
          f"({t_run/GENERATIONS*1000:.0f}ms/generation) -- "
          f"RssAnon={rss_after_run['RssAnon']:.1f}MB "
          f"(delta since loading: {rss_after_run['RssAnon']-rss_after_load['RssAnon']:.1f}MB)")

    disk_result = ws.snapshot("state")

    print("\n=== memory summary ===")
    print(f"size of one full array: {HEIGHT*WIDTH/1024**2:.1f} MB")
    print(f"final RssAnon: {rss_after_run['RssAnon']:.1f} MB "
          f"(ratio to 1 array: {rss_after_run['RssAnon']/(HEIGHT*WIDTH/1024**2):.2f}x)")
    print(f"final RssFile: {rss_after_run['RssFile']:.1f} MB (page cache, not materialization)")

    # ── 4. equivalence against a monolithic reference AT THE SAME SCALE ──
    # 30M uint8 cells fit in RAM as a SINGLE array (~30 MB) -- what must
    # not happen is materializing while WRITING and LOADING the file,
    # which was shown above through RssAnon.
    print("\n[extra] building a monolithic reference at the same scale for the equivalence proof...")
    with rasterio.open(str(tif_path)) as ds:
        state0 = ds.read(1)  # here we DO materialize, on purpose, only for the golden reference
    backend_mono = raster_grid(rows=HEIGHT, cols=WIDTH, attrs={"state": state0.copy()})
    env_mono = Environment(start_time=1, end_time=GENERATIONS)
    GameOfLife(backend=backend_mono)
    t0 = time.time()
    env_mono.run()
    t_mono = time.time() - t0
    golden = backend_mono.arrays["state"].copy()

    n_diff = int(np.sum(golden != disk_result))
    print(f"monolithic run took {t_mono:.1f}s ({t_mono/GENERATIONS*1000:.0f}ms/generation)")
    print(f"\ndisk-vs-monolithic differences: {n_diff} of {HEIGHT*WIDTH:,} cells")
    print(f"IDENTICAL? {n_diff == 0}")


if __name__ == "__main__":
    main()
