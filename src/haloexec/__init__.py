from .engine import Block, make_blocks, resolve_boundary_value
from .disk.workspace import MemmapRasterWorkspace
from .disk.backend import WorkspaceRasterBackend
from .disk.io.geotiff import (
    load_geotiff_into_workspace,
    load_geotiffs_into_workspace,
    save_workspace_to_geotiff,
)
from .disk.io.zarr import load_zarr_into_workspace, load_zarr_tiles_into_workspace
from .disk.convergence import sweep_until_convergence

__all__ = [
    "Block",
    "make_blocks",
    "resolve_boundary_value",
    "MemmapRasterWorkspace",
    "WorkspaceRasterBackend",
    "load_geotiff_into_workspace",
    "load_geotiffs_into_workspace",
    "save_workspace_to_geotiff",
    "load_zarr_into_workspace",
    "load_zarr_tiles_into_workspace",
    "sweep_until_convergence",
]

# Os adaptadores dissmodel (HaloChunkedRasterCellularAutomaton,
# HaloChunkedSyncRasterModel, DiskChunkedSyncRasterModel) são opcionais
# -- os módulos acima funcionam sem dissmodel instalado. Só ficam
# disponíveis se o extra "dissmodel" estiver instalado
# (pip install "haloexec[dissmodel]"). Mesmo padrão usado em
# pymangue/__init__.py para CMMAModel.
try:
    from .ram.cellular_automaton import HaloChunkedRasterCellularAutomaton
    from .ram.sync_model import HaloChunkedSyncRasterModel
    from .disk.sync_model import DiskChunkedSyncRasterModel, workspace_arrays_for_sync_model
    from .disk.cellular_automaton import DiskChunkedRasterCellularAutomaton
    __all__ += [
        "HaloChunkedRasterCellularAutomaton",
        "HaloChunkedSyncRasterModel",
        "DiskChunkedSyncRasterModel",
        "workspace_arrays_for_sync_model",
        "DiskChunkedRasterCellularAutomaton",
    ]
except ImportError:
    pass

__version__ = "2.0.0"
