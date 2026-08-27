"""
Validação end-to-end contra o TerraME — monolítico vs. blocos+halo.

Roda FloodModel + MangroveModel (brmangue-dissmodel, código real e
inalterado) sobre o dataset real da Ilha do Maranhão (elevacao_pol.zip,
50.496 células, grade 323x349), comparando duas execuções contra os
CSVs golden do TerraME (Bezerra 2014):

  1. Monolítico  — RasterFlood + RasterMangue "puros" (baseline)
  2. Halo         — mesmas classes, envolvidas por HaloChunkedSyncRasterModel
                    (blocos+halo, em memória)

Isso fecha o ciclo de validação que faltava: os testes anteriores
provaram equivalência bloco-vs-monolítico com dado sintético; este
script prova que a equivalência se mantém no dataset real, e que
ambas as execuções batem com a referência científica externa
(TerraME), não só uma com a outra.

Reaproveita os helpers privados de
brmangue.executors.validation_executor (_build_raster, _metrics,
CheckpointModel, BANDS, GOLDEN_STEP_OFFSET) em vez de reimplementá-los
— evita duas fontes de verdade divergentes para a lógica de comparação.

Uso
---
    python examples/brmangue_validation/validate_against_terrame.py \\
        --end-time 19 --checkpoints 1 5 10 15 19 \\
        --block-h 64 --block-w 64
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pandas as pd

from dissmodel.core import Environment
from dissmodel.io import load_dataset

from brmangue.models.raster.flood_model import FloodModel
from brmangue.models.raster.mangrove_model import MangroveModel
from brmangue.executors.validation_executor import (
    BANDS, GOLDEN_STEP_OFFSET, CheckpointModel, _build_raster, _metrics,
)

from haloexec import HaloChunkedSyncRasterModel

HERE = pathlib.Path(__file__).parent

# boundary_value alinhado ao nodata real de cada array — ver o achado
# documentado no README do haloexec (0 não é seguro para "solo",
# SOLO_CANAL_FLUVIAL=0 é um código válido).
BOUNDARY_VALUE = {"uso": 0, "alt": -9999.0, "solo": -1, "mask": 0}


class FloodModelHalo(HaloChunkedSyncRasterModel, FloodModel):
    pass


class MangroveModelHalo(HaloChunkedSyncRasterModel, MangroveModel):
    pass


def load_golden(golden_dir: pathlib.Path, checkpoints: list[int]) -> dict[int, pd.DataFrame]:
    golden_map = {}
    for step in checkpoints:
        golden_step = step + GOLDEN_STEP_OFFSET
        path = golden_dir / f"step_{golden_step:02d}.csv"
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values(["row", "col"]).reset_index(drop=True)
        golden_map[step] = df
    return golden_map


def run_variant(label: str, backend_source, flood_cls, mangue_cls, end_time: int,
                 checkpoints: list[int], taxa_elevacao: float, altura_mare: float,
                 **model_kwargs) -> tuple[dict, float]:
    backend, rows_idx, cols_idx = _build_raster(backend_source)

    env = Environment(start_time=1, end_time=end_time)
    flood_cls(backend=backend, taxa_elevacao=taxa_elevacao, **model_kwargs)
    mangue_cls(backend=backend, taxa_elevacao=taxa_elevacao, altura_mare=altura_mare,
               **model_kwargs)
    checkpointer = CheckpointModel(backend=backend, bands=list(BANDS), checkpoints=checkpoints)

    t0 = time.perf_counter()
    env.run()
    elapsed_ms = (time.perf_counter() - t0) * 1000 / end_time

    return checkpointer.snapshots, elapsed_ms, rows_idx, cols_idx


def compare_to_golden(label: str, snapshots: dict, rows_idx, cols_idx,
                       golden_map: dict[int, pd.DataFrame], alt_atol: float) -> None:
    print(f"\n=== {label} ===")
    for step in sorted(snapshots):
        snap = snapshots[step]
        golden = golden_map[step]
        for band, strategy in BANDS.items():
            if band not in golden.columns:
                continue
            ras_vals = snap[band][rows_idx, cols_idx].astype(float)
            gold_vals = golden[band].values.astype(float)
            tol = alt_atol if strategy == "approx" else 0.0
            m = _metrics(ras_vals, gold_vals, tol)
            print(f"  step={step:02d}  {band}: match={m['match_pct']:.1f}%  "
                  f"MAE={m['mae']:.6f}  max_err={m['max_err']:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=int, default=19)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1, 5, 10, 15, 19])
    parser.add_argument("--block-h", type=int, default=64)
    parser.add_argument("--block-w", type=int, default=64)
    parser.add_argument("--halo", type=int, default=2,
                        help="halo=1 é insuficiente para FloodModel (dependência de "
                             "2 saltos via fluxo/viz_baixos) — ver README/regressão "
                             "em tests/test_flood_model_halo_depth_regression.py")
    parser.add_argument("--taxa-elevacao", type=float, default=0.05)
    parser.add_argument("--altura-mare", type=float, default=6.0)
    parser.add_argument("--alt-atol", type=float, default=1e-3)
    args = parser.parse_args()

    input_path = HERE / "elevacao_pol.zip"
    golden_dir = HERE / "golden"

    gdf, _ = load_dataset(str(input_path), fmt="vector")
    gdf.columns = [c.lower() for c in gdf.columns]
    gdf = gdf.sort_values(["row", "col"]).reset_index(drop=True)
    print(f"Domínio carregado: {len(gdf):,} células")

    golden_map = load_golden(golden_dir, args.checkpoints)

    # ── monolítico (baseline) ────────────────────────────────────────
    snaps_mono, ms_mono, rows_idx, cols_idx = run_variant(
        "monolitico", gdf, FloodModel, MangroveModel,
        args.end_time, args.checkpoints, args.taxa_elevacao, args.altura_mare,
    )
    compare_to_golden(f"MONOLÍTICO ({ms_mono:.1f} ms/passo)", snaps_mono,
                       rows_idx, cols_idx, golden_map, args.alt_atol)

    # ── halo (blocos, em memória) ────────────────────────────────────
    snaps_halo, ms_halo, rows_idx2, cols_idx2 = run_variant(
        "halo", gdf, FloodModelHalo, MangroveModelHalo,
        args.end_time, args.checkpoints, args.taxa_elevacao, args.altura_mare,
        block_h=args.block_h, block_w=args.block_w, halo=args.halo,
        boundary_value=BOUNDARY_VALUE,
    )
    compare_to_golden(f"HALO bloco={args.block_h}x{args.block_w} halo={args.halo} "
                       f"({ms_halo:.1f} ms/passo)",
                       snaps_halo, rows_idx2, cols_idx2, golden_map, args.alt_atol)

    # ── monolítico vs. halo, direto (deve ser bit-exato) ─────────────
    print("\n=== MONOLÍTICO vs. HALO (equivalência direta) ===")
    for step in sorted(snaps_mono):
        for band in BANDS:
            a = snaps_mono[step][band]
            b = snaps_halo[step][band]
            if band == "alt":
                n_diff = int(np.sum(~np.isclose(a, b, atol=1e-9)))
            else:
                n_diff = int(np.sum(a != b))
            status = "OK" if n_diff == 0 else "DIVERGENTE"
            print(f"  step={step:02d}  {band}: {status} ({n_diff} células diferentes)")


if __name__ == "__main__":
    main()
