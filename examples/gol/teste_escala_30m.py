"""
Teste de escala: ~30 milhões de pixels, gerando o TIFF em janelas
(nunca materializa a grade inteira em RAM), carregando bloco a bloco,
e rodando o GameOfLife real do dissmodel_ca via
DiskChunkedRasterCellularAutomaton (haloexec) -- o mesmo desafio de
escala que o modelo real do manguezal vai enfrentar.

Compara contra uma referência monolítica NA MESMA ESCALA (30M células
uint8 cabem em RAM como um único array, ~30MB -- o que não cabe é o
processo de GERAR/CARREGAR via arquivo grande sem materializar, que é
o que este script realmente testa).
"""

import time
import resource
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin

from dissmodel.core import Environment
from dissmodel.geo import raster_grid
from dissmodel_ca.models.game_of_life_raster import GameOfLife

from haloexec import MemmapRasterWorkspace, DiskChunkedRasterCellularAutomaton, load_geotiff_into_workspace


class GameOfLifeHalo(DiskChunkedRasterCellularAutomaton, GameOfLife):
    pass


def memoria_mb() -> dict:
    """RssAnon (heap real) via /proc/self/status -- a métrica que
    prova materialização, não VmRSS/ru_maxrss (inclui cache de página
    de arquivo mapeado, sempre alto em memmap sem indicar problema)."""
    valores = {}
    with open("/proc/self/status") as f:
        for line in f:
            for chave in ("VmRSS", "RssAnon", "RssFile"):
                if line.startswith(chave + ":"):
                    valores[chave] = int(line.split()[1]) / 1024
    return valores


def gerar_tiff_em_janelas(path: Path, height: int, width: int, density: float,
                           seed: int, block: int = 512) -> None:
    """Gera o GeoTIFF escrevendo bloco a bloco via rasterio.windows.Window
    -- nunca aloca a grade (height, width) inteira em RAM de uma vez.
    RNG determinística por posição de bloco (reprodutível)."""
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


def main():
    # ~30 milhões de pixels
    HEIGHT, WIDTH = 5480, 5480  # 30.030.400 células
    DENSITY = 0.35
    SEED = 42
    GENERATIONS = 5
    BLOCK = 256
    HALO = 1

    tmp = Path("/tmp/teste_escala_30m")
    tmp.mkdir(exist_ok=True)
    tif_path = tmp / "estado_inicial.tif"

    print(f"Grade: {HEIGHT}x{WIDTH} = {HEIGHT*WIDTH:,} células "
          f"(~{HEIGHT*WIDTH/1024**2:.1f} MB por array uint8)")

    # ── 1. gerar o TIFF em janelas ──────────────────────────────────
    rss_antes = memoria_mb()
    t0 = time.time()
    gerar_tiff_em_janelas(tif_path, HEIGHT, WIDTH, DENSITY, SEED, block=BLOCK)
    t_geracao = time.time() - t0
    rss_pos_geracao = memoria_mb()
    print(f"\n[1/3] TIFF gerado em {t_geracao:.1f}s -- "
          f"RssAnon={rss_pos_geracao['RssAnon']:.1f}MB "
          f"(delta desde inicio: {rss_pos_geracao['RssAnon']-rss_antes['RssAnon']:.1f}MB)")

    # ── 2. carregar bloco a bloco no workspace ──────────────────────
    t0 = time.time()
    ws = MemmapRasterWorkspace.create(
        root=tmp / "workspace", shape=(HEIGHT, WIDTH),
        arrays={"state": np.uint8}, block_h=BLOCK, block_w=BLOCK, halo=HALO,
    )
    load_geotiff_into_workspace(ws, tif_path, [("state", "uint8", 0)])
    t_carga = time.time() - t0
    rss_pos_carga = memoria_mb()
    print(f"[2/3] Carregado em {t_carga:.1f}s -- "
          f"RssAnon={rss_pos_carga['RssAnon']:.1f}MB "
          f"(delta desde geracao: {rss_pos_carga['RssAnon']-rss_pos_geracao['RssAnon']:.1f}MB)")

    # ── 3. rodar GameOfLife em disco+halo, via dissmodel ────────────
    t0 = time.time()
    env = Environment(start_time=1, end_time=GENERATIONS)
    GameOfLifeHalo(workspace=ws, halo=HALO, boundary_value=0)
    env.run()
    ws.flush()
    t_execucao = time.time() - t0
    rss_pos_execucao = memoria_mb()
    print(f"[3/3] {GENERATIONS} gerações em {t_execucao:.1f}s "
          f"({t_execucao/GENERATIONS*1000:.0f}ms/geração) -- "
          f"RssAnon={rss_pos_execucao['RssAnon']:.1f}MB "
          f"(delta desde carga: {rss_pos_execucao['RssAnon']-rss_pos_carga['RssAnon']:.1f}MB)")

    resultado_disco = ws.snapshot("state")

    print(f"\n=== resumo de memoria ===")
    print(f"tamanho de um array completo: {HEIGHT*WIDTH/1024**2:.1f} MB")
    print(f"RssAnon final: {rss_pos_execucao['RssAnon']:.1f} MB "
          f"(razao sobre 1 array: {rss_pos_execucao['RssAnon']/(HEIGHT*WIDTH/1024**2):.2f}x)")
    print(f"RssFile final: {rss_pos_execucao['RssFile']:.1f} MB (cache de pagina, nao e materializacao)")

    # ── 4. equivalencia contra referencia monolitica NA MESMA ESCALA ──
    # 30M celulas uint8 cabem em RAM como array UNICO (~30MB) -- o que
    # nao cabe/nao deveria ser feito e materializar durante GERACAO e
    # CARGA do arquivo, que ja foi provado acima via RssAnon.
    print(f"\n[extra] gerando referencia monolitica na mesma escala para prova de equivalencia...")
    with rasterio.open(str(tif_path)) as ds:
        estado0 = ds.read(1)  # aqui SIM materializamos, de proposito, so para a referencia golden
    backend_mono = raster_grid(rows=HEIGHT, cols=WIDTH, attrs={"state": estado0.copy()})
    env_mono = Environment(start_time=1, end_time=GENERATIONS)
    GameOfLife(backend=backend_mono)
    t0 = time.time()
    env_mono.run()
    t_mono = time.time() - t0
    golden = backend_mono.arrays["state"].copy()

    n_diff = int(np.sum(golden != resultado_disco))
    print(f"monolitico rodou em {t_mono:.1f}s ({t_mono/GENERATIONS*1000:.0f}ms/geracao)")
    print(f"\ndivergencias disco-vs-monolitico: {n_diff} de {HEIGHT*WIDTH:,} celulas")
    print(f"IDENTICO? {n_diff == 0}")


if __name__ == "__main__":
    main()
