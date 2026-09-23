"""
Repeated sweeps until convergence — for problems with UNBOUNDED SPATIAL
DEPENDENCY (connectivity, flow routing, watershed delineation), where no
fixed-size halo is enough on its own: a cell's value may, in principle,
depend on the whole domain.

Only the orchestration pattern (small halo + repetition, instead of one
large halo) lives here; the actual rule (e.g. a connectivity propagation
with scipy.ndimage.binary_propagation per block) is supplied by the
caller. Any `rule` that operates on a window with a halo and returns an
updated core works.

Why this is the right pattern for unbounded dependency
------------------------------------------------------
A fixed-size halo lets information cross only ONE block boundary per
call. Repeating the sweep N times (N = number of blocks, in the worst
case) lets the propagation "front" advance one block per round — after
N rounds it has necessarily reached every reachable cell in the whole
domain, whatever its size. It is the same principle as a distributed
BFS in bounded rounds.

Gauss-Seidel, not Jacobi
------------------------
Each block writes its result IMMEDIATELY back to the same read slot
(through write_block_core_in_place, no ping-pong) — a block processed
later in the SAME sweep already sees the update of a block processed
earlier. This speeds up convergence (fewer sweeps) without changing the
final result, as long as the rule is a monotone operator (e.g.
connectivity only grows, never shrinks) — then the final fixed point
does not depend on block order, only the number of sweeps to reach it
does. For non-monotone rules the final result MAY depend on the order —
it is up to the rule's author to ensure monotonicity if that behaviour
must be well defined.
"""

from __future__ import annotations

import numpy as np

from .workspace import Block, MemmapRasterWorkspace


def sweep_until_convergence(
    workspace: MemmapRasterWorkspace,
    rule,
    boundary_value=0,
    max_sweeps: int | None = None,
) -> dict:
    """
    Repeat block+halo sweeps until a whole sweep changes no block.

    Parameters
    ----------
    rule : callable
        `rule(window: dict[str, np.ndarray]) -> dict[str, np.ndarray]`.
        Receives the halo window of each workspace array (same format as
        MemmapRasterWorkspace.read_block_with_halo) and returns the
        updated arrays, sized to the block CORE (no halo) — the same
        contract as the rules used in
        HaloChunkedSyncRasterModel/DiskChunkedSyncRasterModel.
    boundary_value : scalar or dict, optional
        Same mechanism as haloexec.resolve_boundary_value.
    max_sweeps : int, optional
        Safety limit. Default: number of blocks + 1 (each complete sweep
        advances the propagation front by at least one block, so this
        limit is enough for any domain).

    Returns
    -------
    dict with "sweeps" (how many sweeps ran until convergence),
    "blocks_changed_total" (number of blocks changed in each sweep,
    summed over all sweeps) and "converged".

    Raises
    ------
    RuntimeError
        If it does not converge within max_sweeps — fails loudly instead
        of truncating silently.
    """
    blocks: list[Block] = workspace.blocks()
    if max_sweeps is None:
        max_sweeps = len(blocks) + 1

    total_changed_blocks = 0
    for sweep in range(1, max_sweeps + 1):
        changed_this_sweep = 0
        for block in blocks:
            window = workspace.read_block_with_halo(block, boundary_value=boundary_value)
            updates = rule(window)

            block_changed = False
            for name, new_core in updates.items():
                old_core = workspace.read_block_core(block, name)
                if not np.array_equal(old_core, new_core):
                    block_changed = True

            if block_changed:
                workspace.write_block_core_in_place(block, updates)
                changed_this_sweep += 1

        total_changed_blocks += changed_this_sweep
        if changed_this_sweep == 0:
            return {"sweeps": sweep, "blocks_changed_total": total_changed_blocks, "converged": True}

    raise RuntimeError(
        f"sweep_until_convergence did not converge in {max_sweeps} sweeps "
        f"(limit = number of blocks + 1). Check that the rule is monotone "
        f"and terminates, or raise max_sweeps explicitly."
    )
