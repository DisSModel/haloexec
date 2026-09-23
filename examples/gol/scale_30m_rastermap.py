"""
Scale test with controlled visualization through RasterMap:
30 million pixels on disk + halo, saving PNG frames only at selected
generations/years so neither memory nor CPU time is overloaded.

Usage:
    python examples/gol/scale_30m_rastermap.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
from dissmodel.core import Environment
from dissmodel_ca.models.game_of_life_raster import GameOfLife
from rasterio.transform import from_origin
from rasterio.windows import Window

from haloexec import (
    DiskChunkedRasterCellularAutomaton,
    MemmapRasterWorkspace,
    WorkspaceRasterBackend,
    load_geotiff_into_workspace,
)
from haloexec.visualization import CheckpointRasterMap


class GameOfLifeHalo(DiskChunkedRasterCellularAutomaton, GameOfLife):
    pass


def memory_mb() -> dict[str, float]:
    """RssAnon (real heap) from /proc/self/status."""
    values = {}
    with open("/proc/self/status") as f:
        for line in f:
            for key in ("VmRSS", "RssAnon", "RssFile"):
                if line.startswith(key + ":"):
                    values[key] = int(line.split()[1]) / 1024
    return values


def write_tiff_in_windows(
    path: Path, height: int, width: int, density: float, seed: int, block: int = 512
) -> None:
    """Write a GeoTIFF block by block through Window (never allocates the whole grid in RAM)."""
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


def main() -> None:
    HEIGHT, WIDTH = 5480, 5480  # ~30,030,400 cells
    DENSITY = 0.35
    SEED = 42
    GENERATIONS = 5
    BLOCK = 256
    HALO = 1
    # Years / steps to save in the visualization
    STEPS_TO_SAVE = [1, 3, 5]

    tmp = Path("/tmp/haloexec_scale_rastermap")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    tif_path = tmp / "initial_state.tif"

    print("=== Scale test with checkpoint visualization ===")
    print(f"Grid: {HEIGHT}x{WIDTH} = {HEIGHT*WIDTH:,} cells (~{HEIGHT*WIDTH/1024**2:.1f} MB per uint8 array)")
    print(f"Saving PNG frames only at steps: {STEPS_TO_SAVE}\n")

    # 1. Write the TIFF in windows
    t0 = time.time()
    write_tiff_in_windows(tif_path, HEIGHT, WIDTH, DENSITY, SEED, block=BLOCK)
    print(f"[1/4] Initial TIFF written in windows ({time.time() - t0:.1f}s)")

    # 2. Create the workspace and load
    t0 = time.time()
    ws = MemmapRasterWorkspace.create(
        root=tmp / "workspace",
        shape=(HEIGHT, WIDTH),
        arrays={"state": np.uint8},
        block_h=BLOCK,
        block_w=BLOCK,
        halo=HALO,
    )
    load_geotiff_into_workspace(ws, tif_path, [("state", "uint8", 0)])
    print(f"[2/4] Loaded into the on-disk workspace ({time.time() - t0:.1f}s)")

    # 3. Set up the adapter and the viewer
    # stride=4 decimates the 5480x5480 grid to 1370x1370 for fast plotting
    backend_adapter = WorkspaceRasterBackend(ws, stride=4)

    env = Environment(start_time=1, end_time=GENERATIONS)
    GameOfLifeHalo(workspace=ws, halo=HALO, boundary_value=0)

    # CheckpointRasterMap only runs at the given steps
    CheckpointRasterMap(
        backend=backend_adapter,
        band="state",
        color_map={0: "#ffffff", 1: "#2f8f6e"},
        labels={0: "dead", 1: "alive"},
        title="Game of Life 30M (Disk+Halo)",
        save_frames=True,
        save_steps=STEPS_TO_SAVE,
        auto_mask=False,
    )

    # 4. Run the simulation
    rss_before = memory_mb()
    t0 = time.time()
    env.run()
    ws.flush()
    t_exec = time.time() - t0
    rss_after = memory_mb()

    print(f"\n[3/4] Simulation finished in {t_exec:.1f}s ({t_exec/GENERATIONS*1000:.0f}ms/generation)")
    print(f"final RssAnon: {rss_after['RssAnon']:.1f} MB (delta: {rss_after['RssAnon'] - rss_before['RssAnon']:.1f} MB)")

    out_dir = Path("raster_map_frames")
    pngs = sorted(out_dir.glob("state_step_*.png")) if out_dir.exists() else []
    print(f"\n[4/4] Frames written to {out_dir}/:")
    for p in pngs:
        print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
