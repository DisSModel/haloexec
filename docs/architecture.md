# Software Architecture and Design Patterns

This document details the architectural design, component interactions, and design patterns utilized throughout `haloexec`.

---

## 1. Design Philosophy

`haloexec` is structured around four architectural tenets:

1. **Zero-Invasive Scientific Code (Contract Parity)**:  
   Scientists, hydrologists, and domain specialists write complex transition rules involving neighbourhood evaluations, directional shifts, and environmental equations. `haloexec` guarantees that existing models written for [dissmodel](https://pypi.org/project/dissmodel/) (e.g., `RasterCellularAutomaton` and `SyncRasterModel`) run on chunked grids **without modifying a single line of their scientific transition code**.

2. **Decoupled Core Primitives**:  
   The underlying domain decomposition engine, disk workspace, memory-mapped I/O, and convergence algorithms are completely self-contained. They possess **zero runtime dependencies on `dissmodel`** and can be reused in any Python scientific computing framework.

3. **Layered Separation of Concerns**:  
   Ingestion, spatial partitioning, storage management, model execution, and visualization are strictly separated into discrete architectural tiers.

4. **Cooperative Extensibility via Python MRO**:  
   Rather than wrapping models in complex proxy classes or modifying base class implementations, `haloexec` employs cooperative multiple inheritance (mixins) governed by Python's Method Resolution Order (MRO).

---

## 2. Layered Architecture

The system is organized into five distinct layers:

```mermaid
graph TD
    subgraph Layer 4: Visualization & Monitoring
        CRMAP[CheckpointRasterMap]
        WRB[WorkspaceRasterBackend]
    end

    subgraph Layer 3: Model Execution Adapters (dissmodel optional)
        HCRCA[HaloChunkedRasterCellularAutomaton]
        HCSM[HaloChunkedSyncRasterModel]
        DCRCA[DiskChunkedRasterCellularAutomaton]
        DCSM[DiskChunkedSyncRasterModel]
        CONV[sweep_until_convergence]
    end

    subgraph Layer 2: Spatial Ingestion & Egress
        GTIFF_IO[geotiff.py: Loaders / Savers]
        ZARR_IO[zarr.py: Multi-tile / Axis Normalizer]
    end

    subgraph Layer 1: Disk Storage & Workspace Management
        MRW[MemmapRasterWorkspace]
        HWIN[halo_window & HaloWindow]
    end

    subgraph Layer 0: Core Spatial Primitives (Zero Dependencies)
        BLK[Block]
        MBLK[make_blocks]
        RBV[resolve_boundary_value]
    end

    Layer 4 --> Layer 1
    Layer 3 --> Layer 1
    Layer 3 --> Layer 0
    Layer 2 --> Layer 1
    Layer 1 --> Layer 0
```

---

## 3. Component Deep Dive by Layer

### Layer 0: Pure Core Primitives (`haloexec.engine`)
- **[Block](../src/haloexec/engine.py#L21)**: An immutable, hashable dataclass representing a rectangular domain slice $[r_0:r_1, c_0:c_1)$. Provides properties `.shape` and `.core` (ready-to-use tuple of slice objects).
- **[make_blocks](../src/haloexec/engine.py#L41)**: Pure generator/function that subdivides arbitrary 2D dimensions $(H, W)$ into regular blocks of target size $(b_h, b_w)$, naturally accommodating edge remainders.
- **[resolve_boundary_value](../src/haloexec/engine.py#L55)**: Sentinel resolution utility mapping array names to boundary nodata values, with automatic fallback for `<name>_past` temporal layers.

### Layer 1: Storage & Workspace Management (`haloexec.disk.workspace`)
- **[MemmapRasterWorkspace](../src/haloexec/disk/workspace.py#L81)**: Manages binary disk arrays on the filesystem.
  - **Directory Hierarchy**:
    ```text
    workspace_root/
    ├── metadata.json       # Shape, block dimensions, halo depth, array dtypes
    ├── checkpoint.json     # Current simulation step and active read_slot ("a" or "b")
    ├── a/                  # Physical double-buffer Slot A
    │   ├── uso.dat
    │   └── alt.dat
    └── b/                  # Physical double-buffer Slot B
        ├── uso.dat
        └── alt.dat
    ```
  - **Methods**:
    - `create()`: Allocates sparse memory-mapped files without dense zero-initialization.
    - `read_block_with_halo(block, boundary_value)`: Extracts the halo window, reading existing cells from disk and applying boundary sentinels to external borders.
    - `write_block_core(block, values)`: Writes updated core results into the **write slot** (preserving ping-pong synchrony).
    - `write_block_to_read_slot(block, name, values)`: Writes directly into the active **read slot** (used exclusively for `<name>_past` temporal snapshots).
    - `write_block_core_in_place(block, values)`: Writes directly into the active read slot for iterative Gauss-Seidel convergence.
    - `swap_buffers()`: Inverts the active read/write slots.
    - `checkpoint(step)`: Atomically persists execution state.

### Layer 2: Spatial Ingestion & Egress (`haloexec.disk.io`)
- **[geotiff.py](../src/haloexec/disk/io/geotiff.py)**:
  - Streams GeoTIFF and VRT rasters into a `MemmapRasterWorkspace` block-by-block using `rasterio.windows.Window`.
  - Validates coordinate reference systems (CRS) and dimension consistency across multi-file inputs.
  - Exports workspace slots to Cloud-Optimized GeoTIFFs (COG) with block-aligned tiling.
- **[zarr.py](../src/haloexec/disk/io/zarr.py)**:
  - Streams directly from Zarr groups or arrays (compatible with `disscube` data cubes).
  - Normalizes coordinate axis orders using Zarr v3 dimension metadata (`arr.metadata.dimension_names`) to prevent silent transpose bugs.
  - Assembles multi-tile footprints (`load_zarr_tiles_into_workspace`) into continuous seamless rasters.

### Layer 3: Model Execution Adapters (`haloexec.ram`, `haloexec.disk`)
Contains the execution harnesses that connect simulation rules to either RAM buffers or disk workspaces.

---

## 4. Key Design Patterns

### 4.1 Cooperative Multiple Inheritance via Python MRO
In models inheriting from `SyncRasterModel` (such as `FloodModel` or `MangroveModel`), the scientific logic is contained within `execute()`. To add chunked execution without altering the source model, `haloexec` utilizes mixin inheritance:

```python
class FloodModelDiskHalo(DiskChunkedSyncRasterModel, FloodModel):
    pass
```

#### Method Resolution Order (MRO) Mechanics
Python resolves method calls using the C3 Linearization algorithm. Placing `DiskChunkedSyncRasterModel` first in the class declaration produces the following MRO chain:

```mermaid
graph LR
    Subclass[FloodModelDiskHalo] --> Mixin[DiskChunkedSyncRasterModel]
    Mixin --> BaseScientific[FloodModel]
    BaseScientific --> SyncBase[SyncRasterModel]
    SyncBase --> RasterModelBase[RasterModel]
    RasterModelBase --> CoreModel[dissmodel.core.Model]
```

When the simulation environment invokes `model.execute()`:
1. `DiskChunkedSyncRasterModel.execute()` intercepts the call first.
2. The mixin iterates over the workspace blocks:
   - For each block, it populates a local `block_backend`.
   - It substitutes `self.backend = block_backend` and `self.shape = block_backend.shape`.
   - It invokes `super().execute()`.
3. `super().execute()` evaluates `FloodModel.execute()`, which computes equations against the **local block**, unaware that it is operating on a sub-domain.
4. The mixin recovers the result, trims the halo, writes the core to the destination buffer, and restores `self.backend` in a `finally` block.

### 4.2 The Runtime Backend-Swapping Pattern
Scientific models frequently query context and spatial helper methods directly from their attached backend:
- `self.backend.focal_sum_mask(...)`
- `rows, cols = self.shape`
- `self.shift(array, direction)`

If the model operated on the global backend while processing a block, operations like `focal_sum_mask()` would allocate arrays of global shape $(H, W)$, defeating the purpose of domain decomposition.

The **Backend-Swapping Pattern** dynamically contextualizes the model:

```python
# haloexec/ram/cellular_automaton.py:L142-L148
self.backend = block_backend
try:
    updates = self.rule(block_backend.snapshot())
finally:
    self.backend = real_backend
```

By ensuring that `self.backend` and `self.shape` point to `block_backend` during the inner evaluation, all spatial kernels and memory buffers allocate locally with dimensions $(b_h + 2h, b_w + 2h)$.

### 4.3 Decimation Adapter Pattern (`WorkspaceRasterBackend`)
Visualizing multi-million-cell rasters with `matplotlib` or interactive dashboards causes severe performance degradation and memory spikes.

Rather than copying and downsampling the entire raster in RAM, [WorkspaceRasterBackend](../src/haloexec/disk/backend.py#L17) implements an on-the-fly strided view:

```python
# haloexec/disk/backend.py:L57
return {name: mm[::self.stride, ::self.stride] for name, mm in memmaps.items()}
```

Through NumPy striding, slicing a memory-mapped array with `::stride` creates a view without copying data. The kernel loads only the specific pages containing sampled pixels, rendering massive rasters instantaneously.
