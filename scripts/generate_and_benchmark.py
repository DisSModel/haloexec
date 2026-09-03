"""
Gera uma grade sintética grande DIRETO EM DISCO (bloco a bloco, nunca
materializada inteira em RAM) e roda Game of Life via
MemmapRasterWorkspace, medindo tempo de execução e pico de RSS de RAM.

Objetivo: provar empiricamente que o footprint de memória fica
limitado ao tamanho do bloco (+halo), não ao tamanho da grade —
independentemente de quão grande for a grade em disco.

Uso
---
    python scripts/generate_and_benchmark.py --shape 20000 20000 \\
        --block 512 512 --halo 1 --generations 5 --density 0.35 \\
        --root /tmp/haloexec_bench

    python scripts/generate_and_benchmark.py --shape 5000 5000 \\
        --block 128 128 --generations 20 --root /tmp/bench_pequeno

Troque a regra (_game_of_life_rule) por qualquer outra função
`dict[str, np.ndarray] -> dict[str, np.ndarray]` para testar outro
modelo simples — a mecânica de geração/benchmark não muda.
"""

from __future__ import annotations

import argparse
import resource
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
    """Popula um array do workspace bloco a bloco, com RNG determinística
    por bloco — nunca aloca a grade inteira em RAM de uma vez."""
    master_rng = np.random.default_rng(seed)
    for block in ws.blocks():
        block_seed = int(master_rng.integers(0, 2**31 - 1)) ^ (block.r0 * 92821 + block.c0)
        rng = np.random.default_rng(block_seed & 0xFFFFFFFF)
        h, w = block.r1 - block.r0, block.c1 - block.c0
        data = (rng.random((h, w)) < density).astype(np.uint8)
        ws.write_block_to_read_slot(block, name, data)


def memory_breakdown_mb() -> dict[str, float]:
    """Quebra de RSS via /proc/self/status: RssAnon é o que o processo
    de fato alocou no heap (arrays Python/numpy mantidos vivos);
    RssFile é cache de páginas de arquivos mapeados (mmap) tocadas —
    reclamável pelo kernel sob pressão de memória, NÃO é o mesmo que
    "a grade inteira está materializada no processo". ru_maxrss/VmRSS
    somam os dois, o que é enganoso para workflows baseados em mmap:
    RssFile cresce com o volume de dados TOCADO ao longo do tempo
    (cumulativo), não com o quanto está retido de uma vez."""
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

    grid_bytes = shape[0] * shape[1]  # uint8: 1 byte/célula
    print(f"Grade: {shape[0]}x{shape[1]} = {shape[0]*shape[1]:,} células "
          f"(~{grid_bytes / 1024**2:.1f} MB por array, x2 slots x2 discos "
          f"= ~{grid_bytes * 4 / 1024**2:.1f} MB em disco)")
    print(f"Bloco: {block_h}x{block_w}, halo={halo}, gerações={generations}")

    rss_before = memory_breakdown_mb()

    t0 = time.time()
    ws = MemmapRasterWorkspace.create(
        root=root, shape=shape, arrays={"state": np.uint8},
        block_h=block_h, block_w=block_w, halo=halo,
    )
    generate_synthetic_on_disk(ws, "state", density=density, seed=seed)
    t_geracao = time.time() - t0
    m = memory_breakdown_mb()
    print(f"Geração sintética em disco: {t_geracao:.2f}s | "
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
    t_execucao = time.time() - t0

    m_depois = memory_breakdown_mb()
    print(f"Execução ({generations} gerações): {t_execucao:.2f}s "
          f"({t_execucao/generations*1000:.1f} ms/geração)")
    print(f"RssAnon (heap real do processo): {m_depois['RssAnon']:.1f} MB "
          f"(delta desde o início: {m_depois['RssAnon'] - rss_before['RssAnon']:.1f} MB)")
    print(f"RssFile (cache de páginas mmap, reclamável): {m_depois['RssFile']:.1f} MB")
    print(f"VmRSS total (soma dos dois, é o que ru_maxrss mediria): {m_depois['VmRSS']:.1f} MB")
    print(f"Tamanho de um array completo em disco: {grid_bytes / 1024**2:.1f} MB")
    print(f"→ RssAnon é a métrica correta para 'quanto o processo materializou "
          f"de fato'; RssFile cresce com o volume TOCADO acumulado (cache), "
          f"não com o que está retido de uma vez.")

    if not keep:
        shutil.rmtree(root)
        print(f"Workspace removido ({root}). Use --keep para preservar.")
    else:
        print(f"Workspace preservado em {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=2, default=[20000, 20000],
                        metavar=("ALTURA", "LARGURA"))
    parser.add_argument("--block", type=int, nargs=2, default=[512, 512],
                        metavar=("BLOCO_H", "BLOCO_W"))
    parser.add_argument("--halo", type=int, default=1)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--density", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=Path, default=Path("/tmp/haloexec_bench"))
    parser.add_argument("--keep", action="store_true",
                        help="não apagar o workspace ao final")
    args = parser.parse_args()

    run_benchmark(
        root=args.root,
        shape=tuple(args.shape),
        block_h=args.block[0], block_w=args.block[1],
        halo=args.halo, generations=args.generations,
        density=args.density, seed=args.seed, keep=args.keep,
    )
