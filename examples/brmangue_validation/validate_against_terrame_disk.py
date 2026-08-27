"""
Validação end-to-end contra o TerraME — caminho DISCO (não RAM).

Mesmo dataset e golden de validate_against_terrame.py, mas carregando
o domínio de um GeoTIFF real via load_geotiff_into_workspace (bloco a
bloco, nunca materializa a grade inteira em RAM) e rodando
FloodModel+MangroveModel combinados via DiskChunkedSyncRasterModel,
compartilhando o mesmo MemmapRasterWorkspace.

Este é o cenário que expôs o bug de "_past" órfão (ver README): quando
dois modelos com land_use_types diferentes (FloodModel não gerencia
"solo_past") compartilham um workspace com double-buffer, um array
"_past" que só um dos modelos gerencia pode ficar órfão através do
swap do outro modelo, se a reconciliação de blocos excluir "_past" da
escrita padrão. Corrigido em disk_sync_model.py.

Requer o TIFF de entrada com 4 bandas: uso, alt, solo, mask (nessa
ordem) — sem a banda mask, células fora do domínio real (fora do
polígono da costa) são tratadas como válidas, o que muda o resultado
sutilmente (confirmado empiricamente: 99.7% -> 100% de match ao
incluir mask).

Uso
---
    python examples/brmangue_validation/validate_against_terrame_disk.py \\
        --end-time 19 --checkpoints 1 5 10 15 19 \\
        --block-h 64 --block-w 64 --halo 2
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pandas as pd
import rasterio

from dissmodel.core import Environment

from brmangue.models.raster.flood_model import FloodModel
from brmangue.models.raster.mangrove_model import MangroveModel
from brmangue.executors.validation_executor import BANDS, GOLDEN_STEP_OFFSET, _metrics

from haloexec import (
    DiskChunkedSyncRasterModel, MemmapRasterWorkspace,
    workspace_arrays_for_sync_model, load_geotiff_into_workspace,
)

HERE = pathlib.Path(__file__).parent

# uso: 0=nodata, alt: -9999.0=nodata, solo: -1=nodata (fora do domínio
# válido — ver achado sobre boundary_value no README), mask: 0=inválido.
BAND_SPEC = [
    ("uso", "int16", 0),
    ("alt", "float32", -9999.0),
    ("solo", "int16", -1),
    ("mask", "uint8", 0),
]
BOUNDARY_VALUE = {"uso": 0, "alt": -9999.0, "solo": -1, "mask": 0}


class FloodModelDiskHalo(DiskChunkedSyncRasterModel, FloodModel):
    pass


class MangroveModelDiskHalo(DiskChunkedSyncRasterModel, MangroveModel):
    pass


def load_golden(golden_dir: pathlib.Path, checkpoints: list[int]) -> dict[int, pd.DataFrame]:
    golden_map = {}
    for step in checkpoints:
        golden_step = step + GOLDEN_STEP_OFFSET
        df = pd.read_csv(golden_dir / f"step_{golden_step:02d}.csv")
        df.columns = [c.lower() for c in df.columns]
        golden_map[step] = df.sort_values(["row", "col"]).reset_index(drop=True)
    return golden_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-time", type=int, default=19)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1, 5, 10, 15, 19])
    parser.add_argument("--block-h", type=int, default=64)
    parser.add_argument("--block-w", type=int, default=64)
    parser.add_argument("--halo", type=int, default=2)
    parser.add_argument("--taxa-elevacao", type=float, default=0.05)
    parser.add_argument("--altura-mare", type=float, default=6.0)
    parser.add_argument("--alt-atol", type=float, default=1e-3)
    parser.add_argument("--tif", type=pathlib.Path, default=HERE / "ilha_maranhao.tif")
    args = parser.parse_args()

    with rasterio.open(str(args.tif)) as ds:
        shape = (ds.height, ds.width)
    print(f"Domínio: {shape[0]}x{shape[1]} = {shape[0]*shape[1]:,} células "
          f"(carregado do disco, nunca materializado inteiro em RAM)")

    golden_map = load_golden(HERE / "golden", args.checkpoints)

    arrays = workspace_arrays_for_sync_model(
        base={"uso": np.int16, "alt": np.float32, "solo": np.int16, "mask": np.uint8},
        land_use_types=["uso", "alt", "solo"],
    )
    ws = MemmapRasterWorkspace.create(
        root=pathlib.Path("/tmp/haloexec_terrame_disk_validation"),
        shape=shape, arrays=arrays,
        block_h=args.block_h, block_w=args.block_w, halo=args.halo,
    )
    load_geotiff_into_workspace(ws, args.tif, BAND_SPEC)

    env = Environment(start_time=1, end_time=args.end_time)
    FloodModelDiskHalo(workspace=ws, taxa_elevacao=args.taxa_elevacao,
                        boundary_value=BOUNDARY_VALUE)
    MangroveModelDiskHalo(workspace=ws, taxa_elevacao=args.taxa_elevacao,
                           altura_mare=args.altura_mare, boundary_value=BOUNDARY_VALUE)

    t0 = time.perf_counter()
    env.run()
    ws.flush()
    ms_per_step = (time.perf_counter() - t0) * 1000 / args.end_time
    print(f"Execução: {ms_per_step:.1f} ms/passo (disco, bloco={args.block_h}x{args.block_w}, "
          f"halo={args.halo})")

    uso_final = ws.snapshot("uso")
    solo_final = ws.snapshot("solo")
    alt_final = ws.snapshot("alt")
    snap = {"uso": uso_final, "solo": solo_final, "alt": alt_final}

    print(f"\n=== DISCO vs TerraME (checkpoint {args.checkpoints[-1]}) ===")
    golden = golden_map[args.checkpoints[-1]]
    rows = golden["row"].astype(int).values - golden["row"].astype(int).min()
    cols = golden["col"].astype(int).values - golden["col"].astype(int).min()
    for band, strategy in BANDS.items():
        if band not in golden.columns:
            continue
        ras_vals = snap[band][rows, cols].astype(float)
        gold_vals = golden[band].values.astype(float)
        tol = args.alt_atol if strategy == "approx" else 0.0
        m = _metrics(ras_vals, gold_vals, tol)
        print(f"  {band}: match={m['match_pct']:.1f}%  MAE={m['mae']:.6f}  "
              f"max_err={m['max_err']:.6f}")

    import shutil
    shutil.rmtree("/tmp/haloexec_terrame_disk_validation", ignore_errors=True)


if __name__ == "__main__":
    main()
