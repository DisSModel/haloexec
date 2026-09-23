"""
Generate a large synthetic grid STRAIGHT ON DISK (block by block, never
materialized whole in RAM) and run Game of Life through
MemmapRasterWorkspace, measuring run time and peak RAM RSS.

Goal: show empirically that the memory footprint stays bounded by the
block size (+halo), not by the grid size — however large the grid on
disk is.

Usage
-----
    python scripts/generate_and_benchmark.py --shape 20000 20000 \\
        --block 512 512 --halo 1 --generations 5 --density 0.35 \\
        --root /tmp/haloexec_bench

    python scripts/generate_and_benchmark.py --shape 5000 5000 \\
        --block 128 128 --generations 20 --root /tmp/bench_small

Swap the rule (_game_of_life_rule) for any other
`dict[str, np.ndarray] -> dict[str, np.ndarray]` function to test
another simple model — the generation/benchmark mechanics stay the same.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from haloexec.disk.workspace import MemmapRasterWorkspace


def _game_of_life_rule(padded: dict[str, np.ndarray], halo: int = 1) -> dict[str, np.ndarray]:
    state = padded["state"]
    core = state[halo:-halo, halo:-halo]
    neighbor_count = (
        state[0:-2, 0:-2] + state[0:-2, 1:-1] + state[0:-2, 2:]
        + state[1:-1, 0:-2] + state[1:-1, 2:]
        + state[2:, 0:-2] + state[2:, 1:-1] + state[2:, 2:]
    )
    born = (core == 0) & (neighbor_count == 3)
    survive = (core == 1) & ((neighbor_count == 2) | (neighbor_count == 3))
    return {"state": (born | survive).astype(np.uint8)}


def generate_synthetic_on_disk(
    ws: MemmapRasterWorkspace, name: str, density: float, seed: int
) -> None:
    """Fill one workspace array block by block, with a deterministic RNG
    per block — never allocates the whole grid in RAM at once."""
    master_rng = np.random.default_rng(seed)
    for block in ws.blocks():
        block_seed = int(master_rng.integers(0, 2**31 - 1)) ^ (block.r0 * 92821 + block.c0)
        rng = np.random.default_rng(block_seed & 0xFFFFFFFF)
        h, w = block.r1 - block.r0, block.c1 - block.c0
        data = (rng.random((h, w)) < density).astype(np.uint8)
        ws.write_block_to_read_slot(block, name, data)


def memory_breakdown_mb() -> dict[str, float]:
    """RSS breakdown from /proc/self/status: RssAnon is what the process
    actually allocated on the heap (Python/numpy arrays kept alive);
    RssFile is the page cache of touched memory-mapped files —
    reclaimable by the kernel under memory pressure, NOT the same as
    "the whole grid is materialized in the process". ru_maxrss/VmRSS
    add the two, which is misleading for mmap-based workflows: RssFile
    grows with the volume of data TOUCHED over time (cumulative), not
    with how much is held at once."""
    values = {}
    with open("/proc/self/status") as f:
        for line in f:
            for key in ("VmRSS", "RssAnon", "RssFile", "RssShmem"):
                if line.startswith(key + ":"):
                    kb = int(line.split()[1])
                    values[key] = kb / 1024
    return values


def run_benchmark(
    root: Path,
    shape: tuple[int, int],
    block_h: int,
    block_w: int,
    halo: int,
    generations: int,
    density: float,
    seed: int,
    keep: bool,
) -> None:
    if root.exists():
        shutil.rmtree(root)

    grid_bytes = shape[0] * shape[1]  # uint8: 1 byte/cell
    print(f"Grid: {shape[0]}x{shape[1]} = {shape[0]*shape[1]:,} cells "
          f"(~{grid_bytes / 1024**2:.1f} MB per array, x2 slots x2 disks "
          f"= ~{grid_bytes * 4 / 1024**2:.1f} MB on disk)")
    print(f"Block: {block_h}x{block_w}, halo={halo}, generations={generations}")

    rss_before = memory_breakdown_mb()

    t0 = time.time()
    ws = MemmapRasterWorkspace.create(
        root=root, shape=shape, arrays={"state": np.uint8},
        block_h=block_h, block_w=block_w, halo=halo,
    )
    generate_synthetic_on_disk(ws, "state", density=density, seed=seed)
    t_write = time.time() - t0
    m = memory_breakdown_mb()
    print(f"Synthetic generation on disk: {t_write:.2f}s | "
          f"RssAnon={m['RssAnon']:.1f}MB RssFile={m['RssFile']:.1f}MB "
          f"VmRSS={m['VmRSS']:.1f}MB")

    t0 = time.time()
    for step in range(generations):
        for block in ws.blocks():
            window = ws.read_block_with_halo(block, boundary_value=0)
            result = _game_of_life_rule(window, halo)
            ws.write_block_core(block, result)
        ws.swap_buffers()
        ws.checkpoint(step)
    ws.flush()
    t_run = time.time() - t0

    m_after = memory_breakdown_mb()
    print(f"Run ({generations} generations): {t_run:.2f}s "
          f"({t_run/generations*1000:.1f} ms/generation)")
    print(f"RssAnon (the process's real heap): {m_after['RssAnon']:.1f} MB "
          f"(delta since start: {m_after['RssAnon'] - rss_before['RssAnon']:.1f} MB)")
    print(f"RssFile (mmap page cache, reclaimable): {m_after['RssFile']:.1f} MB")
    print(f"total VmRSS (sum of the two, what ru_maxrss would measure): {m_after['VmRSS']:.1f} MB")
    print(f"Size of one full array on disk: {grid_bytes / 1024**2:.1f} MB")
    print("→ RssAnon is the right metric for 'how much the process actually "
          "materialized'; RssFile grows with the accumulated volume TOUCHED "
          "(cache), not with what is held at once.")

    if not keep:
        shutil.rmtree(root)
        print(f"Workspace removed ({root}). Use --keep to keep it.")
    else:
        print(f"Workspace kept at {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=2, default=[20000, 20000],
                        metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--block", type=int, nargs=2, default=[512, 512],
                        metavar=("BLOCK_H", "BLOCK_W"))
    parser.add_argument("--halo", type=int, default=1)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--density", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=Path, default=Path("/tmp/haloexec_bench"))
    parser.add_argument("--keep", action="store_true",
                        help="do not delete the workspace at the end")
    args = parser.parse_args()

    run_benchmark(
        root=args.root,
        shape=tuple(args.shape),
        block_h=args.block[0], block_w=args.block[1],
        halo=args.halo, generations=args.generations,
        density=args.density, seed=args.seed, keep=args.keep,
    )
