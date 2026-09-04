# Tutorials and Practical Recipes

This guide provides end-to-end recipes for implementing and executing models with `haloexec`.

---

## Table of Contents
1. [Recipe 1: In-Memory Chunked Cellular Automaton](#recipe-1-in-memory-chunked-cellular-automaton)
2. [Recipe 2: Out-of-Core Simulation on Massive Grids](#recipe-2-out-of-core-simulation-on-massive-grids)
3. [Recipe 3: Out-of-Core Ecological Model with Historical Arrays (`_past`)](#recipe-3-out-of-core-ecological-model-with-historical-arrays-_past)
4. [Recipe 4: Streaming Real-World GeoTIFFs In and Out](#recipe-4-streaming-real-world-geotiffs-in-and-out)
5. [Recipe 5: Ingesting Zarr Cubes and Multi-Tile Layouts](#recipe-5-ingesting-zarr-cubes-and-multi-tile-layouts)
6. [Recipe 6: Unbounded Hydrological Connectivity with Iterative Sweeps](#recipe-6-unbounded-hydrological-connectivity-with-iterative-sweeps)
7. [Recipe 7: Visualizing Giant Rasters Efficiently](#recipe-7-visualizing-giant-rasters-efficiently)
8. [Recipe 8: Profiling Real Memory Footprint (`RssAnon` vs `RssFile`)](#recipe-8-profiling-real-memory-footprint-rssanon-vs-rssfile)

---

## Recipe 1: In-Memory Chunked Cellular Automaton

Use this recipe when grids fit comfortably in RAM, but you want to evaluate block-wise behavior or prepare models for scaling.

```python
import numpy as np
from dissmodel.core import Environment
from dissmodel.geo.raster.backend import RasterBackend
from haloexec import HaloChunkedRasterCellularAutomaton

class ForestFire(HaloChunkedRasterCellularAutomaton):
    """
    States:
      0 = Empty
      1 = Tree
      2 = Burning
    """
    def rule(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        state = arrays["state"]
        # Query active block backend for focal operations
        burning_neighbors = self.backend.focal_sum_mask(state == 2)
        
        # Transitions
        ignite = (state == 1) & (burning_neighbors > 0)
        extinguish = (state == 2)
        
        new_state = state.copy()
        new_state[ignite] = 2
        new_state[extinguish] = 0
        return {"state": new_state}

# Setup domain: 1000 x 1000
backend = RasterBackend(shape=(1000, 1000))
initial_forest = np.random.choice([0, 1], size=(1000, 1000), p=[0.2, 0.8]).astype(np.uint8)
initial_forest[500, 500] = 2  # Ignite spark at center
backend.set("state", initial_forest)

env = Environment(start_time=1, end_time=50)
# Process in 250 x 250 blocks with a 1-cell halo
ForestFire(backend=backend, block_h=250, block_w=250, halo=1, boundary_value=0)
env.run()
```

---

## Recipe 2: Out-of-Core Simulation on Massive Grids

Use this recipe when grid dimensions exceed physical RAM (e.g., $50,000 \times 50,000 = 2.5\text{ billion cells}$).

```python
from pathlib import Path
import numpy as np
from dissmodel.core import Environment
from haloexec import MemmapRasterWorkspace, DiskChunkedRasterCellularAutomaton

class MassiveGameOfLife(DiskChunkedRasterCellularAutomaton):
    def rule(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        state = arrays["state"]
        alive_neighbors = self.backend.focal_sum_mask(state == 1)
        born = (state == 0) & (alive_neighbors == 3)
        survive = (state == 1) & np.isin(alive_neighbors, [2, 3])
        return {"state": np.where(born | survive, 1, 0).astype(np.uint8)}

# 1. Initialize disk workspace
ws_path = Path("/tmp/massive_gol_workspace")
ws = MemmapRasterWorkspace.create(
    root=ws_path,
    shape=(50_000, 50_000),
    arrays={"state": np.uint8},
    block_h=1024,
    block_w=1024,
    halo=1,
)

# 2. Seed initial state block-by-block (never materialize full grid)
for block in ws.blocks():
    # Deterministic local seeding
    local_data = (np.random.rand(*block.shape) < 0.3).astype(np.uint8)
    ws.write_block_to_read_slot(block, "state", local_data)
ws.flush()

# 3. Execute out-of-core
env = Environment(start_time=1, end_time=10)
MassiveGameOfLife(workspace=ws, halo=1, boundary_value=0)
env.run()
ws.flush()
```

---

## Recipe 3: Out-of-Core Ecological Model with Historical Arrays (`_past`)

This recipe shows how to run models that inherit from `SyncRasterModel` and depend on historical state tracking (`uso_past`, `alt_past`).

```python
from pathlib import Path
import numpy as np
from dissmodel.core import Environment
from dissmodel.geo.raster.sync_model import SyncRasterModel
from haloexec import (
    MemmapRasterWorkspace,
    DiskChunkedSyncRasterModel,
    workspace_arrays_for_sync_model,
)

class MangroveAccretion(SyncRasterModel):
    """Scientific model implementing coastal mangrove sediment accretion."""
    def setup(self, accretion_rate: float = 0.05, **kwargs):
        super().setup(**kwargs)
        self.accretion_rate = accretion_rate
        self.land_use_types = ["uso", "alt"]

    def execute(self):
        alt_past = self.backend.get("alt_past")
        uso = self.backend.get("uso")
        
        # Elevate cells where mangrove exists (uso == 2)
        new_alt = alt_past.copy()
        new_alt[uso == 2] += self.accretion_rate
        self.backend.set("alt", new_alt)

# Mixin inheritance: DiskChunkedSyncRasterModel MUST come first
class MangroveAccretionDisk(DiskChunkedSyncRasterModel, MangroveAccretion):
    pass

# Setup workspace with automatic '<name>_past' allocation
base_arrays = {"uso": np.int16, "alt": np.float32}
declared_arrays = workspace_arrays_for_sync_model(base_arrays, land_use_types=["uso", "alt"])

ws = MemmapRasterWorkspace.create(
    root=Path("/tmp/mangrove_workspace"),
    shape=(2000, 2000),
    arrays=declared_arrays,
    block_h=500,
    block_w=500,
    halo=1,
)

env = Environment(start_time=1, end_time=5)
MangroveAccretionDisk(
    workspace=ws,
    accretion_rate=0.08,
    boundary_value={"uso": 0, "alt": -9999.0},
)
env.run()
ws.flush()
```

---

## Recipe 4: Streaming Real-World GeoTIFFs In and Out

Stream multi-gigabyte raster files into a workspace without loading them into memory, run a simulation, and export to Cloud-Optimized GeoTIFF.

```python
from pathlib import Path
from haloexec import (
    MemmapRasterWorkspace,
    load_geotiff_into_workspace,
    save_workspace_to_geotiff,
)

# 1. Inspect raster metadata without loading pixels
import rasterio
with rasterio.open("study_area.tif") as src:
    height, width = src.height, src.width
    crs = src.crs
    transform = src.transform

# 2. Create workspace matching GeoTIFF dimensions
ws = MemmapRasterWorkspace.create(
    root=Path("/tmp/geotiff_workspace"),
    shape=(height, width),
    arrays={"uso": "int16", "alt": "float32"},
    block_h=512,
    block_w=512,
    halo=2,
)

# 3. Stream data block-by-block (band 1 -> uso, band 2 -> alt)
load_geotiff_into_workspace(
    ws,
    path="study_area.tif",
    band_spec=[("uso", "int16", 0), ("alt", "float32", -9999.0)],
)

# 4. ... Run simulation models on workspace ...

# 5. Export updated state to GeoTIFF
save_workspace_to_geotiff(
    ws,
    path="output_step_10.tif",
    bands=["uso", "alt"],
    transform=transform,
    crs=crs,
    compress="lzw",
)
```

---

## Recipe 5: Ingesting Zarr Cubes and Multi-Tile Layouts

Seamlessly assemble non-contiguous satellite tiles into a single unified workspace.

```python
from pathlib import Path
from haloexec import MemmapRasterWorkspace, load_zarr_tiles_into_workspace

# Define multi-tile layout (e.g. from disscube CubeClient.tile_layout())
tile_layout = [
    {
        "url": "data/tiles/tile_001.zarr",
        "variable": "land_cover",
        "row_off": 0,
        "col_off": 0,
        "height": 5000,
        "width": 5000,
    },
    {
        "url": "data/tiles/tile_002.zarr",
        "variable": "land_cover",
        "row_off": 0,
        "col_off": 5000,
        "height": 5000,
        "width": 5000,
    },
]

ws = MemmapRasterWorkspace.create(
    root=Path("/tmp/zarr_mosaic_workspace"),
    shape=(5000, 10000),
    arrays={"land_cover": "uint8"},
    block_h=1024,
    block_w=1024,
    halo=1,
)

# Stitch tiles directly into the workspace
load_zarr_tiles_into_workspace(
    ws,
    tiles=tile_layout,
    array="land_cover",
    fill=0,
    skip_empty_blocks=False,
)
```

---

## Recipe 6: Unbounded Hydrological Connectivity with Iterative Sweeps

Solve tidal flooding or basin routing where water spreads across unlimited block boundaries until reaching an equilibrium point.

```python
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_propagation
from haloexec import MemmapRasterWorkspace, sweep_until_convergence

def tidal_connectivity_rule(window: dict[str, np.ndarray], halo: int = 1) -> dict[str, np.ndarray]:
    """Expands water connectivity through permeable lowlands."""
    water = window["water"].astype(bool)
    permeable = window["permeable"].astype(bool)
    
    # Propagate water connectivity within the halo window
    flooded = binary_propagation(water, mask=permeable)
    
    # Trim halo and return updated core
    return {"water": flooded[halo:-halo, halo:-halo].astype(np.uint8)}

ws = MemmapRasterWorkspace.create(
    root=Path("/tmp/tidal_ws"),
    shape=(4000, 4000),
    arrays={"water": np.uint8, "permeable": np.uint8},
    block_h=500,
    block_w=500,
    halo=1,
)

# Execute Gauss-Seidel sweeps until entire domain reaches fixed point
info = sweep_until_convergence(
    ws,
    rule=tidal_connectivity_rule,
    boundary_value={"water": 0, "permeable": 0},
)
print(f"Converged in {info['sweeps']} sweeps across {info['blocks_changed_total']} block updates.")
```

---

## Recipe 7: Visualizing Giant Rasters Efficiently

Render and save checkpoints on massive rasters without CPU/RAM bottlenecks.

```python
from dissmodel.core import Environment
from haloexec import (
    MemmapRasterWorkspace,
    CheckpointRasterMap,
)

ws = MemmapRasterWorkspace(Path("/tmp/my_large_workspace"))

# 1. Create decimated backend adapter (downsample by 4x for rendering)
downsampled_backend = ws.as_backend(stride=4, nodata_value=-9999.0)

# 2. Configure CheckpointRasterMap to save PNGs ONLY on milestone years
env = Environment(start_time=2000, end_time=2050)
CheckpointRasterMap(
    backend=downsampled_backend,
    band="uso",
    color_map={0: "#ffffff", 1: "#2f8f6e", 2: "#0000ff"},
    labels={0: "Empty", 1: "Forest", 2: "Water"},
    title="Simulation Decimated View",
    save_frames=True,
    save_steps=[2000, 2010, 2025, 2050],  # Render only these years!
    auto_mask=False,
)
env.run()
```

---

## Recipe 8: Profiling Real Memory Footprint (`RssAnon` vs `RssFile`)

Accurately monitor heap usage versus kernel file page cache in memory-mapped simulations.

```python
import os

def report_memory_breakdown() -> dict[str, float]:
    """Parses Linux /proc/self/status for anonymous vs file-backed RSS."""
    metrics = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                metrics["RssAnon_MB"] = int(line.split()[1]) / 1024.0
            elif line.startswith("RssFile:"):
                metrics["RssFile_MB"] = int(line.split()[1]) / 1024.0
            elif line.startswith("VmRSS:"):
                metrics["VmRSS_MB"] = int(line.split()[1]) / 1024.0
    return metrics

mem = report_memory_breakdown()
print(f"True Heap Memory (RssAnon): {mem['RssAnon_MB']:.2f} MB")
print(f"OS File Page Cache (RssFile): {mem['RssFile_MB']:.2f} MB")
print(f"Total Combined RSS (VmRSS): {mem['VmRSS_MB']:.2f} MB")
```
