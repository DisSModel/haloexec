"""
Teste de escala com visualização controlada via RasterMap:
30 milhões de pixels em disco + halo, salvando quadros PNG apenas em
gerações/anos selecionados para não sobrecarregar a memória nem o tempo de CPU.

Uso:
    python examples/gol_patterns/teste_escala_com_rastermap.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin

from dissmodel.core import Environment
from dissmodel_ca.models.game_of_life_raster import GameOfLife

from haloexec import (
    MemmapRasterWorkspace,
    DiskChunkedRasterCellularAutomaton,
    load_geotiff_into_workspace,
    WorkspaceRasterBackend,
)
from haloexec.visualization import CheckpointRasterMap


class GameOfLifeHalo(DiskChunkedRasterCellularAutomaton, GameOfLife):
    pass


def memoria_mb() -> dict[str, float]:
    """RssAnon (heap real) via /proc/self/status."""
    valores = {}
    with open("/proc/self/status") as f:
        for line in f:
            for chave in ("VmRSS", "RssAnon", "RssFile"):
                if line.startswith(chave + ":"):
                    valores[chave] = int(line.split()[1]) / 1024
    return valores


def gerar_tiff_em_janelas(
    path: Path, height: int, width: int, density: float, seed: int, block: int = 512
) -> None:
    """Gera GeoTIFF escrevendo bloco a bloco via Window (nunca aloca grade inteira em RAM)."""
    transform = from_origin(500_000.0, 9_700_000.0, 30.0, 30.0)
    master_rng = np.random.default_rng(seed)

    with rasterio.open(
        str(path), "w", driver="GTiff", height=height, width=width,
        count=1, dtype="uint8", crs="EPSG:31984", transform=transform,
        tiled=True, blockxsize=block, blockysize=block, compress="lzw",
    ) as dst:
        for r0 in range(0, height, block):
            r1 = min(r0 + block, height)
            for c0 in range(0, width, block):
                c1 = min(c0 + block, width)
                block_seed = int(master_rng.integers(0, 2**31 - 1)) ^ (r0 * 92821 + c0)
                rng = np.random.default_rng(block_seed & 0xFFFFFFFF)
                dado = (rng.random((r1 - r0, c1 - c0)) < density).astype("uint8")
                dst.write(dado, 1, window=Window(c0, r0, c1 - c0, r1 - r0))


def main() -> None:
    HEIGHT, WIDTH = 5480, 5480  # ~30.030.400 células
    DENSITY = 0.35
    SEED = 42
    GENERATIONS = 5
    BLOCK = 256
    HALO = 1
    # Anos / passos a salvar na visualização
    ANOS_PARA_SALVAR = [1, 3, 5]

    tmp = Path("/tmp/teste_escala_rastermap")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    tif_path = tmp / "estado_inicial.tif"

    print(f"=== Teste de Escala com Visualização por Checkpoints ===")
    print(f"Grade: {HEIGHT}x{WIDTH} = {HEIGHT*WIDTH:,} células (~{HEIGHT*WIDTH/1024**2:.1f} MB por array uint8)")
    print(f"Salvando quadros PNG apenas nos passos: {ANOS_PARA_SALVAR}\n")

    # 1. Gerar TIFF em janelas
    t0 = time.time()
    gerar_tiff_em_janelas(tif_path, HEIGHT, WIDTH, DENSITY, SEED, block=BLOCK)
    print(f"[1/4] TIFF inicial gerado em janelas ({time.time() - t0:.1f}s)")

    # 2. Criar workspace e carregar
    t0 = time.time()
    ws = MemmapRasterWorkspace.create(
        root=tmp / "workspace",
        shape=(HEIGHT, WIDTH),
        arrays={"state": np.uint8},
        block_h=BLOCK,
        block_w=BLOCK,
        halo=HALO,
    )
    load_geotiff_into_workspace(ws, tif_path, [("state", "uint8", 0)])
    print(f"[2/4] Carregado no workspace em disco ({time.time() - t0:.1f}s)")

    # 3. Configurar adapter e visualizador
    # stride=4 decima a grade de 5480x5480 para 1370x1370 para plotagem super rápida
    backend_adapter = WorkspaceRasterBackend(ws, stride=4)

    env = Environment(start_time=1, end_time=GENERATIONS)
    GameOfLifeHalo(workspace=ws, halo=HALO, boundary_value=0)

    # CheckpointRasterMap só executa nos passos indicados
    CheckpointRasterMap(
        backend=backend_adapter,
        band="state",
        color_map={0: "#ffffff", 1: "#2f8f6e"},
        labels={0: "morta", 1: "viva"},
        title="Game of Life 30M (Disco+Halo)",
        save_frames=True,
        save_steps=ANOS_PARA_SALVAR,
        auto_mask=False,
    )

    # 4. Rodar simulação
    rss_antes = memoria_mb()
    t0 = time.time()
    env.run()
    ws.flush()
    t_exec = time.time() - t0
    rss_depois = memoria_mb()

    print(f"\n[3/4] Simulação finalizada em {t_exec:.1f}s ({t_exec/GENERATIONS*1000:.0f}ms/geração)")
    print(f"RssAnon final: {rss_depois['RssAnon']:.1f} MB (delta: {rss_depois['RssAnon'] - rss_antes['RssAnon']:.1f} MB)")

    out_dir = Path("raster_map_frames")
    pngs = sorted(out_dir.glob("state_step_*.png")) if out_dir.exists() else []
    print(f"\n[4/4] Quadros gerados em {out_dir}/:")
    for p in pngs:
        print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
