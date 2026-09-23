"""
Sparse allocation of MemmapRasterWorkspace.

`create()` sizes the `.dat` files without pre-writing zeros, so they
start sparse. These tests pin BOTH halves of the contract:

  * the semantics are unchanged — reading a region never written still
    returns zero, which is what pre-writing gave;
  * the disk cost changed — only what is written takes up blocks.

The second depends on the filesystem supporting sparse files (ext4, xfs
and btrfs do). Where it does not, the test is skipped instead of
failing: the behaviour is still correct, it just saves nothing there.
"""

import numpy as np
import pytest

from haloexec import MemmapRasterWorkspace

SHAPE = (2048, 2048)
ARRAYS = {"uso": "float64"}


def _real_bytes(path):
    return path.stat().st_blocks * 512


def _fs_supports_sparse(tmp_path) -> bool:
    p = tmp_path / "probe.bin"
    mm = np.memmap(p, dtype="float64", mode="w+", shape=(1024, 1024))
    mm.flush()
    del mm
    return _real_bytes(p) < p.stat().st_size


def _ws(tmp_path, **kw):
    return MemmapRasterWorkspace.create(
        tmp_path / "ws", shape=SHAPE, arrays=ARRAYS,
        block_h=256, block_w=256, halo=1, **kw
    )


# ── semantics: unchanged ─────────────────────────────────────────────────────

def test_region_never_written_reads_zero(tmp_path):
    """Old contract kept: a new workspace reads as zeros."""
    ws = _ws(tmp_path)
    for block in ws.blocks()[:5]:
        assert np.all(ws.read_block_core(block, "uso") == 0)


def test_halo_of_new_workspace_reads_zero(tmp_path):
    ws = _ws(tmp_path)
    window = ws.read_block_with_halo(ws.blocks()[10], boundary_value=0)["uso"]
    assert np.all(window == 0)


def test_write_and_read_still_work(tmp_path):
    ws = _ws(tmp_path)
    blk = ws.blocks()[3]
    values = np.full((blk.r1 - blk.r0, blk.c1 - blk.c0), 7.5)
    ws.write_block_to_read_slot(blk, "uso", values)
    ws.flush()
    assert np.array_equal(ws.read_block_core(blk, "uso"), values)


def test_neighbour_of_a_written_block_stays_zero(tmp_path):
    """Writing one block must not materialize values in another."""
    ws = _ws(tmp_path)
    blocks = ws.blocks()
    ws.write_block_to_read_slot(
        blocks[0], "uso",
        np.ones((blocks[0].r1 - blocks[0].r0, blocks[0].c1 - blocks[0].c0)),
    )
    ws.flush()
    assert np.all(ws.read_block_core(blocks[1], "uso") == 0)


# ── disk cost: this is the gain ──────────────────────────────────────────────

def test_new_workspace_takes_no_disk(tmp_path):
    if not _fs_supports_sparse(tmp_path):
        pytest.skip("filesystem does not support sparse files")
    _ws(tmp_path)
    dat = tmp_path / "ws" / "a" / "uso.dat"
    assert dat.stat().st_size == SHAPE[0] * SHAPE[1] * 8   # full apparent size
    assert _real_bytes(dat) < dat.stat().st_size // 10     # but almost nothing real


def test_only_what_was_written_takes_disk(tmp_path):
    if not _fs_supports_sparse(tmp_path):
        pytest.skip("filesystem does not support sparse files")
    ws = _ws(tmp_path)
    dat = tmp_path / "ws" / "a" / "uso.dat"
    blocks = ws.blocks()
    for blk in blocks[:4]:
        ws.write_block_to_read_slot(
            blk, "uso", np.ones((blk.r1 - blk.r0, blk.c1 - blk.c0))
        )
    ws.flush()
    used = _real_bytes(dat)
    assert used > 0                                        # what was written counts
    assert used < dat.stat().st_size // 2                  # the rest does not
