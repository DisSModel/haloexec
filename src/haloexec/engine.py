"""
Generic Domain Decomposition primitives with Halo zones (Ghost Cell
Pattern) — no dependency on dissmodel.

Theoretical basis: Kjolstad & Snir (2010), "Ghost Cell Pattern",
ParaPLoP; application to geospatial CA-LULC: Xia et al. (2025), ISPRS
IJGI 14(3):109. See README.md.

This module holds only the grid-partitioning logic (Block, make_blocks).
The execution itself — calling the transition rule per block and
reconciling the result — belongs to whoever consumes these primitives.
The concrete dissmodel integration is in `ram/cellular_automaton.py`
(HaloChunkedRasterCellularAutomaton).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    """A rectangular sub-domain of the global grid (no halo)."""

    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.r1 - self.r0, self.c1 - self.c0)

    @property
    def core(self) -> tuple[slice, slice]:
        """Slices ready to index this block's region in a global array
        (used by the disk layer, e.g. write_block_core)."""
        return (slice(self.r0, self.r1), slice(self.c0, self.c1))


def make_blocks(height: int, width: int, block_h: int, block_w: int) -> list[Block]:
    """Split a (height, width) grid into regular blocks of size
    (block_h, block_w). Blocks on the right/bottom edge may be smaller
    (the remainder), as in regular domain decomposition
    (Xia et al. 2025, Section 2.1)."""
    blocks = []
    for r0 in range(0, height, block_h):
        r1 = min(r0 + block_h, height)
        for c0 in range(0, width, block_w):
            c1 = min(c0 + block_w, width)
            blocks.append(Block(r0, r1, c0, c1))
    return blocks


def resolve_boundary_value(boundary_value, name: str) -> float:
    """Resolve the outer-halo fill value for one array. Accepts a
    scalar (same value for every array) or a dict {name: value}.

    If `name` ends in "_past" and has no entry of its own in the dict,
    it falls back to the base name's value (without "_past") — so
    whoever sets {"solo": -1} does not have to remember to repeat it for
    "solo_past". Without this fallback, "_past" would silently get the
    default 0, bringing back the very problem this mechanism exists to
    prevent.

    Important: 0 is not a safe sentinel for every domain — in BR-MANGUE,
    for example, `solo=0` is SOLO_CANAL_FLUVIAL, a VALID soil code (not
    "no data"). Using 0 as boundary_value for that array creates phantom
    migration sources at the domain's outer edge, diverging from the
    monolithic result. Prefer aligning boundary_value with each array's
    real nodata (e.g. the domain's TIFF_BANDS), passing a dict instead of
    a single scalar.
    """
    if not isinstance(boundary_value, dict):
        return boundary_value
    if name in boundary_value:
        return boundary_value[name]
    if name.endswith("_past"):
        base_name = name[: -len("_past")]
        if base_name in boundary_value:
            return boundary_value[base_name]
    return 0
