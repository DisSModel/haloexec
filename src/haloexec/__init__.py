from .disk.backend import WorkspaceRasterBackend
from .disk.convergence import sweep_until_convergence
from .disk.io.geotiff import (
    load_geotiff_into_workspace,
    load_geotiffs_into_workspace,
    save_workspace_to_geotiff,
)
from .disk.io.zarr import load_zarr_into_workspace, load_zarr_tiles_into_workspace
from .disk.workspace import MemmapRasterWorkspace
from .engine import Block, make_blocks, resolve_boundary_value

__all__ = [
    "Block",
    "MemmapRasterWorkspace",
    "WorkspaceRasterBackend",
    "load_geotiff_into_workspace",
    "load_geotiffs_into_workspace",
    "load_zarr_into_workspace",
    "load_zarr_tiles_into_workspace",
    "make_blocks",
    "resolve_boundary_value",
    "save_workspace_to_geotiff",
    "sweep_until_convergence",
]

# The dissmodel adapters (HaloChunkedRasterCellularAutomaton,
# HaloChunkedSyncRasterModel, DiskChunkedSyncRasterModel,
# DiskChunkedRasterCellularAutomaton) are optional -- the modules above
# work without dissmodel installed. They are only available when the
# "dissmodel" extra is installed (pip install "haloexec[dissmodel]").
try:
    from .disk.cellular_automaton import DiskChunkedRasterCellularAutomaton
    from .disk.sync_model import DiskChunkedSyncRasterModel, workspace_arrays_for_sync_model
    from .ram.cellular_automaton import HaloChunkedRasterCellularAutomaton
    from .ram.sync_model import HaloChunkedSyncRasterModel
    __all__ += [
        "DiskChunkedRasterCellularAutomaton",
        "DiskChunkedSyncRasterModel",
        "HaloChunkedRasterCellularAutomaton",
        "HaloChunkedSyncRasterModel",
        "workspace_arrays_for_sync_model",
    ]
except ImportError:
    pass

__version__ = "2.0.0"
