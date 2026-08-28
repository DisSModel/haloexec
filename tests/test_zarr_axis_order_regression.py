"""
Regressão: load_zarr_into_workspace deve ler corretamente um Zarr cuja
ordem de eixos em disco NÃO é (y, x) -- cenário real, não hipotético.

O disscube (VariableWriter) grava via
`da.to_dataset(name=var_name).to_zarr(...)`, e o próprio CubeClient.load()
do disscube faz `.transpose("y", "x")` defensivamente antes de usar
qualquer array carregado -- evidência de que a ordem de eixos em disco
NÃO é garantida como (y, x).

O perigo: um array QUADRADO com eixos trocados (x, y) em vez de (y, x)
tem o MESMO shape nos dois casos -- a checagem de shape sozinha não
detecta a inversão. Sem correção, isso corrompe silenciosamente
linha/coluna, sem erro nenhum.

Reproduzido aqui com xarray real, gravando exatamente como o
VariableWriter do disscube grava (to_dataset().to_zarr()), não com
zarr.create_array() direto -- para capturar o metadado real de
dimension_names que o xarray grava (campo nativo do Zarr v3), a
mesma informação que a correção em zarr_io.py usa para normalizar.
"""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
xr = pytest.importorskip("xarray")

from haloexec import MemmapRasterWorkspace, load_zarr_into_workspace


def _write_like_disscube(path, data_yx: np.ndarray, dims: tuple[str, ...], var_name: str):
    """Grava exatamente como VariableWriter do disscube:
    da.to_dataset(name=...).to_zarr(..., mode="w", consolidated=False).
    `dims` controla a ordem de eixos gravada em disco -- ("y","x") é o
    caso "correto"/esperado, ("x","y") é o caso perigoso real."""
    if dims == ("y", "x"):
        raw = data_yx
    elif dims == ("x", "y"):
        raw = data_yx.T
    else:
        raise ValueError(dims)
    da = xr.DataArray(raw, dims=dims, name=var_name)
    da.to_dataset(name=var_name).to_zarr(str(path), mode="w", consolidated=False)


def test_load_zarr_handles_yx_axis_order(tmp_path):
    """Caso 'correto' (y, x) -- deve continuar funcionando como sempre."""
    n = 6
    data = np.arange(n * n).reshape(n, n).astype("int16")
    store = tmp_path / "correto.zarr"
    _write_like_disscube(store, data, ("y", "x"), "uso")

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(n, n),
        arrays={"uso": np.int16}, block_h=3, block_w=3, halo=1,
    )
    load_zarr_into_workspace(ws, str(store), variable_map={"uso": "uso"})

    assert np.array_equal(ws.snapshot("uso"), data)


def test_load_zarr_handles_xy_axis_order_square_array(tmp_path):
    """Caso PERIGOSO: array QUADRADO gravado com eixos (x, y) --
    mesmo shape do caso correto, mas dado fisicamente transposto em
    disco. Sem a correção de dimension_names, isso passaria a checagem
    de shape e corromperia silenciosamente linha/coluna."""
    n = 6
    # valores distintos por linha E coluna, para que uma transposição
    # incorreta produza um array MENSURAVELMENTE diferente do original
    data = np.arange(n * n).reshape(n, n).astype("int16")
    store = tmp_path / "perigoso.zarr"
    _write_like_disscube(store, data, ("x", "y"), "uso")

    # confirma que o shape em disco é IGUAL ao esperado (é exatamente
    # isso que torna o bug silencioso sem a correção de eixo)
    root = zarr.open(str(store), mode="r")
    assert root["uso"].shape == data.shape, "pré-condição do teste: shapes devem coincidir"

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(n, n),
        arrays={"uso": np.int16}, block_h=3, block_w=3, halo=1,
    )
    load_zarr_into_workspace(ws, str(store), variable_map={"uso": "uso"})

    resultado = ws.snapshot("uso")
    assert np.array_equal(resultado, data), (
        "load_zarr_into_workspace leu o array com linha/coluna trocadas -- "
        "regressão do bug de ordem de eixos (x,y) vs (y,x)"
    )


def test_load_zarr_handles_txy_axis_order_temporal(tmp_path):
    """Variável temporal (3D) com ordem de eixos não-canônica: (x, y, time)
    em vez de (time, y, x)."""
    n = 5
    n_time = 3
    # (time, y, x) -- valores originais
    serie = np.arange(n_time * n * n).reshape(n_time, n, n).astype("int16")
    # grava fisicamente em ordem (x, y, time)
    raw_disco = np.transpose(serie, (2, 1, 0))  # de (time,y,x) para (x,y,time)

    da = xr.DataArray(raw_disco, dims=("x", "y", "time"), name="mangue")
    store = tmp_path / "temporal.zarr"
    da.to_dataset(name="mangue").to_zarr(str(store), mode="w", consolidated=False)

    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(n, n),
        arrays={"mangue": np.int16}, block_h=2, block_w=2, halo=1,
    )
    load_zarr_into_workspace(ws, str(store), variable_map={"mangue": "mangue"}, time_index=1)

    assert np.array_equal(ws.snapshot("mangue"), serie[1])
