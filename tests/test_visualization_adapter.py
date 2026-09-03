"""
Testes para WorkspaceRasterBackend e CheckpointRasterMap.
"""

from pathlib import Path
import numpy as np
import pytest

from haloexec import MemmapRasterWorkspace, WorkspaceRasterBackend

pytest.importorskip("dissmodel")
from dissmodel.core import Environment
from dissmodel_ca.models.game_of_life_raster import GameOfLife
from haloexec import DiskChunkedRasterCellularAutomaton
from haloexec.visualization import CheckpointRasterMap


class _GoLHalo(DiskChunkedRasterCellularAutomaton, GameOfLife):
    pass


def test_workspace_raster_backend_basic(tmp_path: Path):
    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "ws",
        shape=(100, 100),
        arrays={"state": np.uint8, "elevation": np.float32},
        block_h=20,
        block_w=20,
        halo=1,
    )
    arr0 = np.arange(10000, dtype=np.uint8).reshape(100, 100)
    ws.fill("state", arr0)

    adapter = WorkspaceRasterBackend(ws)
    assert adapter.shape == (100, 100)
    assert "state" in adapter.arrays
    assert np.array_equal(adapter.arrays["state"], arr0)
    assert np.array_equal(adapter.get("state"), arr0)


def test_workspace_raster_backend_stride(tmp_path: Path):
    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "ws_stride",
        shape=(100, 100),
        arrays={"state": np.uint8},
        block_h=20,
        block_w=20,
        halo=1,
    )
    arr0 = np.arange(10000, dtype=np.uint8).reshape(100, 100)
    ws.fill("state", arr0)

    adapter_stride = WorkspaceRasterBackend(ws, stride=4)
    assert adapter_stride.shape == (25, 25)
    assert adapter_stride.arrays["state"].shape == (25, 25)
    assert np.array_equal(adapter_stride.arrays["state"], arr0[::4, ::4])


def test_checkpoint_raster_map_filtering(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "ws_sim",
        shape=(40, 40),
        arrays={"state": np.uint8},
        block_h=10,
        block_w=10,
        halo=1,
    )
    rng = np.random.default_rng(42)
    grid0 = (rng.random((40, 40)) < 0.3).astype(np.uint8)
    ws.fill("state", grid0)

    adapter = WorkspaceRasterBackend(ws)

    env = Environment(start_time=1, end_time=5)
    _GoLHalo(workspace=ws, halo=1, boundary_value=0)

    # Só salvar os passos 1 e 4
    save_steps = [1, 4]
    CheckpointRasterMap(
        backend=adapter,
        band="state",
        color_map={0: "#ffffff", 1: "#2f8f6e"},
        save_frames=True,
        save_steps=save_steps,
        auto_mask=False,
    )

    env.run()

    out_dir = tmp_path / "raster_map_frames"
    assert out_dir.exists()
    saved = sorted(f.name for f in out_dir.glob("*.png"))
    assert saved == ["state_step_001.png", "state_step_004.png"]


def test_save_workspace_to_geotiff_roundtrip(tmp_path: Path):
    from haloexec import save_workspace_to_geotiff, load_geotiff_into_workspace
    import rasterio

    shape = (60, 80)
    ws1 = MemmapRasterWorkspace.create(
        root=tmp_path / "ws1",
        shape=shape,
        arrays={"uso": np.int16, "alt": np.float32},
        block_h=20,
        block_w=20,
        halo=1,
    )
    uso_data = np.random.randint(0, 10, shape, dtype=np.int16)
    alt_data = np.random.randn(*shape).astype(np.float32)
    ws1.fill("uso", uso_data)
    ws1.fill("alt", alt_data)

    tif_path = tmp_path / "output.tif"
    save_workspace_to_geotiff(ws1, tif_path, ["uso", "alt"])

    with rasterio.open(str(tif_path)) as ds:
        assert ds.count == 2
        assert ds.shape == shape
        assert np.array_equal(ds.read(1), uso_data)
        assert np.allclose(ds.read(2), alt_data)

    # Verifica também o método as_backend do workspace
    backend = ws1.as_backend(stride=2)
    assert backend.shape == (30, 40)
    assert np.array_equal(backend.arrays["uso"], uso_data[::2, ::2])

