"""
On-disk array workspace (np.memmap) with block+halo decomposition,
double-buffering and checkpointing — for grids too large to fit in RAM
as a whole.

Generic: it knows nothing about any model's state variables or domain —
it only stores named 2-D arrays with a declared dtype.

Why it is separate from the rest of haloexec
--------------------------------------------
`HaloChunkedRasterCellularAutomaton` and `HaloChunkedSyncRasterModel`
(ram/) build the halo with `np.pad` over the WHOLE grid in memory —
fine while the grid fits in RAM. When it does not (e.g. a stretch of
coastline at fine resolution), the whole grid must never be
materialized: only the block+halo window is read from disk, clipped at
the edges when the block touches the domain boundary.

Theoretical basis: the same as the rest of haloexec (Kjolstad & Snir
2010; Xia et al. 2025), here in the variant where the halo is read
directly from the file on disk, with no in-memory padding of the global
grid.

Double-buffering
----------------
Each array has two physical slots ("a" and "b"). A time step reads from
the current slot and writes to the other one — so a cell never reads a
neighbour's value already updated in the same step (the classic hazard
of synchronous cellular automata). At the end of the step the slots
swap roles (a logical swap, no data copied).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..engine import Block, make_blocks, resolve_boundary_value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class HaloWindow:
    """Read window with halo, clipped at the domain edges.

    global_slices : where to read in the global array (may be smaller
        than block+2*halo near the edges — there is no padding on disk).
    core_offset : (row, col) offset of the start of the block's "core"
        inside the window read, since the window may start closer to the
        core when the halo was clipped at an edge.
    """
    global_slices: tuple[slice, slice]
    core_offset: tuple[int, int]


def halo_window(block: Block, shape: tuple[int, int], halo: int) -> HaloWindow:
    """Compute a block's halo window, clipped at the domain edges."""
    height, width = shape
    r0 = max(0, block.r0 - halo)
    c0 = max(0, block.c0 - halo)
    r1 = min(height, block.r1 + halo)
    c1 = min(width, block.c1 + halo)
    return HaloWindow(
        global_slices=(slice(r0, r1), slice(c0, c1)),
        core_offset=(block.r0 - r0, block.c0 - c0),
    )


class MemmapRasterWorkspace:
    """
    Stores named 2-D arrays on disk (np.memmap), with block+halo
    decomposition, double-buffering and progress checkpoints.

    Generic: it imposes no variable names or domain. Any model (dissmodel
    or not) that operates on named 2-D arrays with a local neighbourhood
    rule can use this workspace.

    Typical use
    -----------
    >>> ws = MemmapRasterWorkspace.create(
    ...     root=Path("/tmp/my_workspace"),
    ...     shape=(10000, 10000),
    ...     arrays={"state": np.uint8},
    ...     block_h=512, block_w=512, halo=1,
    ... )
    >>> ws.fill("state", initial_array)
    >>> for step in range(n_steps):
    ...     for block in ws.blocks():
    ...         window = ws.read_block_with_halo(block)   # dict[name, np.ndarray]
    ...         result = my_rule(window)                   # dict[name, np.ndarray]
    ...         ws.write_block_core(block, result)
    ...     ws.swap_buffers()
    ...     ws.checkpoint(step)
    >>> ws.flush()
    """

    METADATA = "metadata.json"
    CHECKPOINT = "checkpoint.json"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        metadata_path = self.root / self.METADATA
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Workspace not initialized: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.shape = tuple(int(v) for v in self.metadata["shape"])
        self.block_h = int(self.metadata["block_h"])
        self.block_w = int(self.metadata["block_w"])
        self.halo = int(self.metadata["halo"])
        self._dtypes = {n: np.dtype(d) for n, d in self.metadata["arrays"].items()}
        self._slots: dict[str, dict[str, np.memmap]] = {
            "a": self._open_slot("a"),
            "b": self._open_slot("b"),
        }
        checkpoint_path = self.root / self.CHECKPOINT
        self.checkpoint_data = (
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint_path.is_file()
            else {"step": 0, "read_slot": "a"}
        )

    def _slot_path(self, slot: str, name: str) -> Path:
        return self.root / slot / f"{name}.dat"

    def _open_slot(self, slot: str) -> dict[str, np.memmap]:
        return {
            name: np.memmap(self._slot_path(slot, name), dtype=dtype,
                            mode="r+", shape=self.shape)
            for name, dtype in self._dtypes.items()
        }

    @classmethod
    def create(
        cls,
        root: Path,
        shape: tuple[int, int],
        arrays: dict[str, np.dtype],
        block_h: int,
        block_w: int,
        halo: int = 1,
    ) -> MemmapRasterWorkspace:
        """Create a new workspace, with both double-buffer slots.

        The `.dat` files start SPARSE: only regions actually written take
        up blocks on disk. Reading a region never written returns zero —
        guaranteed by the POSIX filesystem itself, identical to what
        pre-writing zeros would give. The semantics are therefore
        unchanged; only the disk cost differs (measured: a 4000x4000
        float64 array goes from 122 MB on disk to 0 MB until something
        is written).

        Practical consequence, and the limit of this saving: it only
        shows if the loader LEAVES blocks unwritten. A loader that fills
        every block — including the empty ones, with a sentinel such as
        NaN — makes the file dense again. And leaving a block unwritten
        means that region is ZERO, not "missing": in domains where 0 is a
        valid value (a class code, an elevation at sea level) the two
        cases become indistinguishable.

        Telling "missing" apart from "valid zero" would need an extra
        channel, which this format does not have — see "Sparsity and disk
        cost" in the README for the proposed design (an index of present
        blocks + nodata declared per array, the same mechanism GeoTIFF
        uses with TileOffsets == 0).

        None of this affects RAM cost, which the memmap itself solves:
        the kernel brings pages in on demand, so walking the whole grid
        never materializes it at once.
        """
        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"The workspace directory must be new and empty: {root}")
        root.mkdir(parents=True, exist_ok=True)

        dtypes = {n: np.dtype(d) for n, d in arrays.items()}
        for slot in ("a", "b"):
            for name, dtype in dtypes.items():
                path = root / slot / f"{name}.dat"
                path.parent.mkdir(parents=True, exist_ok=True)
                # mode="w+" sizes the file without touching its bytes: it stays
                # sparse. Do NOT pre-write zeros here — that would allocate
                # everything physically without changing anything read later.
                mm = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
                mm.flush()

        _write_json_atomic(root / cls.METADATA, {
            "shape": [int(shape[0]), int(shape[1])],
            "block_h": int(block_h),
            "block_w": int(block_w),
            "halo": int(halo),
            "arrays": {n: str(d) for n, d in dtypes.items()},
        })
        _write_json_atomic(root / cls.CHECKPOINT, {"step": 0, "read_slot": "a"})
        return cls(root)

    # ── block access ─────────────────────────────────────────────────

    def blocks(self) -> list[Block]:
        return make_blocks(self.shape[0], self.shape[1], self.block_h, self.block_w)

    def fill(self, name: str, array: np.ndarray, slot: str | None = None) -> None:
        """Fill a whole array (one-off use, e.g. the initial state). For
        large arrays, prefer writing block by block with write_block_core."""
        slot = slot or self.checkpoint_data["read_slot"]
        mm = self._slots[slot][name]
        mm[:] = np.asarray(array, dtype=mm.dtype)
        mm.flush()

    def read_block_with_halo(self, block: Block, boundary_value=0) -> dict[str, np.ndarray]:
        """Read the halo window of every array for a block, from the
        current read slot. Only the part of the halo that falls outside
        the domain (the grid's outer edges) is filled with boundary_value
        — the rest is read straight from disk, without materializing the
        whole grid.

        boundary_value takes a scalar (same value for every array) or a
        dict {name: value} — needed whenever 0 is not a safe sentinel for
        some array (e.g. a valid class code in the domain). See
        engine.resolve_boundary_value."""
        read_slot = self._slots[self.checkpoint_data["read_slot"]]
        window = halo_window(block, self.shape, self.halo)
        full_h = block.r1 - block.r0 + 2 * self.halo
        full_w = block.c1 - block.c0 + 2 * self.halo

        result = {}
        for name, mm in read_slot.items():
            name_boundary = resolve_boundary_value(boundary_value, name)
            raw = np.asarray(mm[window.global_slices])
            if raw.shape == (full_h, full_w):
                result[name] = raw.copy()
            else:
                # edge block: the window was clipped; fill the missing
                # halo with that array's boundary value.
                padded = np.full((full_h, full_w), name_boundary, dtype=raw.dtype)
                dest_r0 = self.halo - window.core_offset[0]
                dest_c0 = self.halo - window.core_offset[1]
                padded[dest_r0:dest_r0 + raw.shape[0], dest_c0:dest_c0 + raw.shape[1]] = raw
                result[name] = padded
        return result

    def write_block_core(self, block: Block, values: dict[str, np.ndarray]) -> None:
        """Write a block's core region (no halo) to the WRITE slot (the
        other slot, not the read one) — keeps the double buffer intact:
        the current step never writes to the array other blocks of the
        same step are still reading."""
        write_slot_name = "b" if self.checkpoint_data["read_slot"] == "a" else "a"
        write_slot = self._slots[write_slot_name]
        for name, core in values.items():
            write_slot[name][block.core] = core

    def read_block_core(self, block: Block, name: str) -> np.ndarray:
        """Read only the core region (no halo) of one array, from the
        current read slot. Used for "_past" synchronization — no
        neighbourhood needed, just a direct copy."""
        read_slot = self._slots[self.checkpoint_data["read_slot"]]
        return np.asarray(read_slot[name][block.core]).copy()

    def write_block_to_read_slot(self, block: Block, name: str, values: np.ndarray) -> None:
        """Write to the SAME slot currently being read (not to the
        ping-pong write slot). Only for filling "<name>_past" arrays:
        they must be available in the read slot BEFORE execute() runs in
        that same step — unlike the "current" arrays, which only become
        ready in the next step, after the swap."""
        read_slot = self._slots[self.checkpoint_data["read_slot"]]
        read_slot[name][block.core] = values

    def write_block_core_in_place(self, block: Block, values: dict[str, np.ndarray]) -> None:
        """Write several arrays to the SAME read slot, immediately
        visible to any block processed later (within the same sweep). No
        ping-pong: deliberately different from write_block_core() (which
        writes to the opposite slot, visible only after swap_buffers()).

        Use: iterative convergence algorithms (see sweep_until_convergence
        in convergence.py), where there is no "state frozen at the start
        of the step" — it is successive refinement of the SAME state up
        to a fixed point, and seeing the update of the neighbouring block
        processed moments earlier speeds up convergence (Gauss-Seidel
        iteration, not Jacobi)."""
        read_slot = self._slots[self.checkpoint_data["read_slot"]]
        for name, core in values.items():
            read_slot[name][block.core] = core

    def swap_buffers(self) -> None:
        """Swap read/write slots at the end of a complete time step."""
        self.checkpoint_data["read_slot"] = (
            "b" if self.checkpoint_data["read_slot"] == "a" else "a"
        )

    def checkpoint(self, step: int) -> None:
        self.checkpoint_data["step"] = int(step)
        _write_json_atomic(self.root / self.CHECKPOINT, self.checkpoint_data)

    def snapshot(self, name: str) -> np.ndarray:
        """Read the whole array from the current read slot (materializes
        it in RAM — for inspection/tests only, never inside the block
        loop)."""
        return np.asarray(self._slots[self.checkpoint_data["read_slot"]][name]).copy()

    def flush(self) -> None:
        for slot in self._slots.values():
            for mm in slot.values():
                mm.flush()

    def as_backend(self, stride: int = 1, nodata_value: float | None = None):
        """Return a WorkspaceRasterBackend adapter for use with RasterMap/dissmodel."""
        from .backend import WorkspaceRasterBackend
        return WorkspaceRasterBackend(self, stride=stride, nodata_value=nodata_value)

