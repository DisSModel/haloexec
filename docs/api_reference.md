# API Reference

This document provides a comprehensive, exhaustive reference for all public classes, functions, and interfaces in the `haloexec` package.

---

## Table of Contents

- [haloexec.engine](#haloexecengine)
  - [Block](#block)
  - [make_blocks](#make_blocks)
  - [resolve_boundary_value](#resolve_boundary_value)
- [haloexec.disk.workspace](#haloexecdiskworkspace)
  - [HaloWindow](#halowindow)
  - [halo_window](#halo_window)
  - [MemmapRasterWorkspace](#memmaprasterworkspace)
- [haloexec.disk.backend](#haloexecdiskbackend)
  - [WorkspaceRasterBackend](#workspacerasterbackend)
- [haloexec.disk.convergence](#haloexecdiskconvergence)
  - [sweep_until_convergence](#sweep_until_convergence)
- [haloexec.disk.sync_model](#haloexecdisksync_model)
  - [workspace_arrays_for_sync_model](#workspace_arrays_for_sync_model)
  - [DiskChunkedSyncRasterModel](#diskchunkedsyncrastermodel)
- [haloexec.disk.cellular_automaton](#haloexecdiskcellular_automaton)
  - [DiskChunkedRasterCellularAutomaton](#diskchunkedrastercellularautomaton)
- [haloexec.disk.io.geotiff](#haloexecdiskio_geotiff)
  - [load_geotiff_into_workspace](#load_geotiff_into_workspace)
  - [load_geotiffs_into_workspace](#load_geotiffs_into_workspace)
  - [save_workspace_to_geotiff](#save_workspace_to_geotiff)
- [haloexec.disk.io.zarr](#haloexecdiskio_zarr)
  - [load_zarr_into_workspace](#load_zarr_into_workspace)
  - [load_zarr_tiles_into_workspace](#load_zarr_tiles_into_workspace)
- [haloexec.ram.cellular_automaton](#haloexecramcellular_automaton)
  - [HaloChunkedRasterCellularAutomaton](#halochunkedrastercellularautomaton)
- [haloexec.ram.sync_model](#haloexecramsync_model)
  - [HaloChunkedSyncRasterModel](#halochunkedsyncrastermodel)
- [haloexec.visualization](#haloexecvisualization)
  - [CheckpointRasterMap](#checkpointrastermap)

---

## `haloexec.engine`

Primitivas de decomposição de domínio sem dependências externas.

### `Block`
```python
@dataclass(frozen=True)
class Block:
    r0: int
    r1: int
    c0: int
    c1: int
```
Represents an immutable 2D rectangular sub-domain slice $[r_0:r_1, c_0:c_1)$ in global grid coordinates.

#### Properties
- **`shape -> tuple[int, int]`**: Returns `(r1 - r0, c1 - c0)`.
- **`core -> tuple[slice, slice]`**: Returns `(slice(r0, r1), slice(c0, c1))`, ready for direct NumPy indexing into global arrays or memory-mapped files.

---

### `make_blocks`
```python
def make_blocks(height: int, width: int, block_h: int, block_w: int) -> list[Block]
```
Partitions a 2D global grid of dimensions `(height, width)` into regular sub-domains of maximum size `(block_h, block_w)`.

#### Parameters
- **`height`** (*int*): Global domain row count.
- **`width`** (*int*): Global domain column count.
- **`block_h`** (*int*): Target block height.
- **`block_w`** (*int*): Target block width.

#### Returns
- **`list[Block]`**: Sequential list of blocks covering the entire domain. Edge blocks along the south and east boundaries contain residual dimensions when dimensions are not evenly divisible.

---

### `resolve_boundary_value`
```python
def resolve_boundary_value(boundary_value: dict[str, float] | float, name: str) -> float
```
Resolves the external ghost cell fill value for a given array.

#### Parameters
- **`boundary_value`** (*dict | float*): Scalar constant or dictionary mapping array names to fill values.
- **`name`** (*str*): Name of the array to resolve.

#### Returns
- **`float`**: Resolved boundary sentinel. If `name` ends with `"_past"` and does not have an explicit key in a `boundary_value` dict, automatically falls back to the base name without `"_past"`. If not found, defaults to `0`.

---

## `haloexec.disk.workspace`

Gerenciador de arrays bidimensionais em disco baseados em `np.memmap` com suporte a double-buffering e checkpoints.

### `HaloWindow`
```python
@dataclass(frozen=True)
class HaloWindow:
    global_slices: tuple[slice, slice]
    core_offset: tuple[int, int]
```
Describes a disk read window clipped at domain boundaries.
- **`global_slices`**: Slice coordinates to read from the disk array.
- **`core_offset`**: `(row_offset, col_offset)` of the core block within the read window.

---

### `halo_window`
```python
def halo_window(block: Block, shape: tuple[int, int], halo: int) -> HaloWindow
```
Computes the clipped read window coordinates for a block with halo $h$ across a domain of shape $(H, W)$.

---

### `MemmapRasterWorkspace`
```python
class MemmapRasterWorkspace:
    METADATA = "metadata.json"
    CHECKPOINT = "checkpoint.json"
```

#### Class Methods

##### `create`
```python
@classmethod
def create(
    cls,
    root: Path,
    shape: tuple[int, int],
    arrays: dict[str, np.dtype],
    block_h: int,
    block_w: int,
    halo: int = 1,
) -> MemmapRasterWorkspace
```
Initializes a new on-disk workspace directory structure containing double-buffered `.dat` files for slots `"a"` and `"b"`.

- Files are allocated as POSIX sparse files (unwritten blocks consume 0 disk space).
- Raises `FileExistsError` if `root` exists and is non-empty.

#### Constructors

##### `__init__`
```python
def __init__(self, root: Path) -> None
```
Opens an existing initialized workspace from disk. Loads `metadata.json` and `checkpoint.json`.

#### Instance Methods

##### `blocks`
```python
def blocks(self) -> list[Block]
```
Returns all domain blocks partitioned according to the workspace's metadata.

##### `fill`
```python
def fill(self, name: str, array: np.ndarray, slot: str | None = None) -> None
```
Populates an entire array layer in the specified slot (defaults to active `read_slot`). Intended for initial state setup.

##### `read_block_with_halo`
```python
def read_block_with_halo(self, block: Block, boundary_value: dict | float = 0) -> dict[str, np.ndarray]
```
Extracts the halo window for all arrays from the active `read_slot`. Populates boundary ghost cells with `boundary_value`.

##### `write_block_core`
```python
def write_block_core(self, block: Block, values: dict[str, np.ndarray]) -> None
```
Writes the updated core regions (excluding halo) into the **write slot** (the inactive ping-pong slot).

##### `read_block_core`
```python
def read_block_core(self, block: Block, name: str) -> np.ndarray
```
Reads strictly the core region of an array from the active `read_slot`.

##### `write_block_to_read_slot`
```python
def write_block_to_read_slot(self, block: Block, name: str, values: np.ndarray) -> None
```
Writes core block data directly into the active **read slot**. Reserved for synchronizing `<name>_past` arrays.

##### `write_block_core_in_place`
```python
def write_block_core_in_place(self, block: Block, values: dict[str, np.ndarray]) -> None
```
Writes core block data directly into the active **read slot**, making changes immediately visible to subsequent blocks in the same pass (used for Gauss-Seidel convergence).

##### `swap_buffers`
```python
def swap_buffers(self) -> None
```
Logically inverts the active `read_slot` (`"a"` $\leftrightarrow$ `"b"`).

##### `checkpoint`
```python
def checkpoint(self, step: int) -> None
```
Persists the current simulation step and active slot atomically to `checkpoint.json`.

##### `snapshot`
```python
def snapshot(self, name: str) -> np.ndarray
```
Materializes a full copy of an array from the active read slot in RAM (use only for validation or small grids).

##### `flush`
```python
def flush(self) -> None
```
Forces an explicit sync of memory-mapped buffers to physical storage.

##### `as_backend`
```python
def as_backend(self, stride: int = 1, nodata_value: float | int | None = None) -> WorkspaceRasterBackend
```
Creates a lightweight `RasterBackend` adapter for visualization and inspection.

---

## `haloexec.disk.backend`

### `WorkspaceRasterBackend`
```python
class WorkspaceRasterBackend:
    def __init__(
        self,
        workspace: MemmapRasterWorkspace,
        stride: int = 1,
        nodata_value: float | int | None = None,
    ) -> None
```
Exposes a `MemmapRasterWorkspace` through the `RasterBackend` duck-type interface without duplicating memory.

#### Parameters
- **`workspace`** (*MemmapRasterWorkspace*): The underlying disk workspace.
- **`stride`** (*int*): Spatial subsampling decimation factor ($1 = \text{full resolution}$, $2 = 50\%$, etc.).
- **`nodata_value`** (*float | int | None*): Optional sentinel for extent mask resolution.

#### Properties
- **`arrays -> dict[str, np.ndarray]`**: Dictionary of memory-mapped views onto the active read slot, sliced with `::stride`.
- **`shape -> tuple[int, int]`**: Scaled grid shape `(H // stride, W // stride)`.

---

## `haloexec.disk.convergence`

### `sweep_until_convergence`
```python
def sweep_until_convergence(
    workspace: MemmapRasterWorkspace,
    rule: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]],
    boundary_value: dict | float = 0,
    max_sweeps: int | None = None,
) -> dict[str, Any]
```
Executes iterative block sweeps until an entire global pass produces zero modified blocks.

#### Parameters
- **`workspace`** (*MemmapRasterWorkspace*): Target workspace.
- **`rule`** (*Callable*): Transition function receiving a halo window dictionary and returning updated core arrays.
- **`boundary_value`**: Fill sentinel for external boundary ghost cells.
- **`max_sweeps`** (*int, optional*): Safety limit. Defaults to $\text{number of blocks} + 1$.

#### Returns
- **`dict`**: Summary containing `{"sweeps": int, "blocks_changed_total": int, "converged": True}`.

#### Raises
- **`RuntimeError`**: If convergence is not achieved within `max_sweeps`.

---

## `haloexec.disk.sync_model`

### `workspace_arrays_for_sync_model`
```python
def workspace_arrays_for_sync_model(
    base: dict[str, np.dtype],
    land_use_types: list[str],
) -> dict[str, np.dtype]
```
Constructs the array declaration dictionary for a workspace, automatically adding `<name>_past` entries for all variables in `land_use_types`.

---

### `DiskChunkedSyncRasterModel`
```python
class DiskChunkedSyncRasterModel:
    def setup(
        self,
        workspace: MemmapRasterWorkspace,
        halo: int | None = None,
        boundary_value: float | dict = 0,
        **kwargs,
    ) -> None
```
Cooperative mixin for executing `SyncRasterModel` subclasses (e.g., `FloodModel`) out-of-core on a `MemmapRasterWorkspace`.

- Must precede the scientific model in class inheritance order:
  `class FloodModelDisk(DiskChunkedSyncRasterModel, FloodModel): pass`
- Handles block-by-block `_past` synchronization in `pre_execute()` and `post_execute()`.

---

## `haloexec.disk.cellular_automaton`

### `DiskChunkedRasterCellularAutomaton`
```python
class DiskChunkedRasterCellularAutomaton:
    def setup(
        self,
        workspace: MemmapRasterWorkspace,
        halo: int | None = None,
        boundary_value: dict | float = 0,
        state_attr: str = "state",
        **kwargs,
    ) -> None
```
Cooperative mixin for executing `RasterCellularAutomaton` subclasses (defined with `rule(arrays) -> dict`) out-of-core on a `MemmapRasterWorkspace`.

---

## `haloexec.disk.io.geotiff`

### `load_geotiff_into_workspace`
```python
def load_geotiff_into_workspace(
    workspace: MemmapRasterWorkspace,
    path: str | Path,
    band_spec: list[tuple[str, str, float]],
) -> None
```
Loads a single GeoTIFF file into a workspace block-by-block using windowed streaming.

### `load_geotiffs_into_workspace`
```python
def load_geotiffs_into_workspace(
    workspace: MemmapRasterWorkspace,
    sources: list[tuple[str | Path, list[tuple[str, str, float]]]],
) -> None
```
Loads multiple GeoTIFF files into a workspace simultaneously across blocks, validating shape and CRS alignment.

### `save_workspace_to_geotiff`
```python
def save_workspace_to_geotiff(
    workspace: MemmapRasterWorkspace,
    path: str | Path,
    bands: list[str] | list[tuple[str, str, float]],
    transform: Any = None,
    crs: Any = "EPSG:31984",
    compress: str = "lzw",
) -> None
```
Exports workspace arrays from the active read slot directly into a tiled Cloud-Optimized GeoTIFF.

---

## `haloexec.disk.io.zarr`

### `load_zarr_into_workspace`
```python
def load_zarr_into_workspace(
    workspace: MemmapRasterWorkspace,
    store: str | Path,
    variable_map: dict[str, str] | None = None,
    time_index: int | None = None,
) -> None
```
Streams a Zarr store into a workspace block-by-block, normalizing dimension axis orders.

### `load_zarr_tiles_into_workspace`
```python
def load_zarr_tiles_into_workspace(
    workspace: MemmapRasterWorkspace,
    tiles: list[dict[str, Any]],
    array: str | None = None,
    fill: float | None = None,
    skip_empty_blocks: bool = False,
) -> None
```
Assembles multiple distinct Zarr tiles into a unified continuous workspace array.

---

## `haloexec.ram.cellular_automaton`

### `HaloChunkedRasterCellularAutomaton`
```python
class HaloChunkedRasterCellularAutomaton(RasterCellularAutomaton):
    def setup(
        self,
        backend: RasterBackend,
        block_h: int,
        block_w: int,
        halo: int = 1,
        boundary_value: float | dict = 0,
        state_attr: str = "state",
    ) -> None
```
In-memory domain decomposition execution engine for `RasterCellularAutomaton`. Preserves the `rule(arrays: dict) -> dict` interface.

---

## `haloexec.ram.sync_model`

### `HaloChunkedSyncRasterModel`
```python
class HaloChunkedSyncRasterModel:
    def setup(
        self,
        backend: RasterBackend,
        block_h: int,
        block_w: int,
        halo: int = 1,
        boundary_value: float | dict = 0,
        **kwargs,
    ) -> None
```
In-memory cooperative mixin for executing `SyncRasterModel` subclasses in blocks with halo.

---

## `haloexec.visualization`

### `CheckpointRasterMap`
```python
class CheckpointRasterMap(RasterMap):
    def setup(
        self,
        *args: Any,
        save_steps: Iterable[int] | None = None,
        **kwargs: Any,
    ) -> None
```
Subclass of `RasterMap` that restricts map rendering and PNG export to an arbitrary set of simulation steps (`save_steps`).
