"""
Carrega arrays de um Zarr store direto para MemmapRasterWorkspace,
bloco a bloco — segunda opção de entrada de dados, ao lado de
geotiff_io.py (TIFF/VRT).

Motivação: o disscube (DisSModel/disscube) armazena variáveis
derivadas nativamente em Zarr (`data/derived/{grid_id}/{tile_id}/
{spec_hash}/{variable_name}.zarr`), já alinhadas à grade mestra pelo
seu GridAligner (com resampling por-operador e alinhamento fino para
categóricas — ver README). Este módulo permite consumir esse dado
direto, sem precisar materializar para GeoTIFF primeiro.

Por que um módulo separado de geotiff_io.py
---------------------------------------------
Zarr já é nativamente chunked — não precisa de rasterio.windows.Window
nem de VRT para leitura parcial; um zarr.Array suporta slicing direto
(`arr[r0:r1, c0:c1]`) que só lê os chunks necessários do store. A API
pública espelha geotiff_io.py (mesma forma de declarar quais arrays do
workspace vêm de qual variável de origem), para que os dois caminhos
de entrada (GeoTIFF/VRT e Zarr) sejam intercambiáveis do ponto de
vista de quem usa MemmapRasterWorkspace — troca-se o loader, o resto
do pipeline (halo, disco, modelos) não muda.

Requer o extra opcional "zarr" (zarr>=2.16). xarray/rioxarray NÃO são
necessários aqui — lê-se o zarr.Array bruto por nome de variável.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from .disk_backend import MemmapRasterWorkspace


def _resolve_axis_order(arr, expected_names: tuple[str, ...]) -> tuple[int, ...] | None:
    """Usa arr.metadata.dimension_names (campo nativo do Zarr v3, o
    mesmo que xarray grava ao salvar um DataArray via to_zarr) para
    determinar a ordem real dos eixos no array em disco, e devolve os
    índices de transposição necessários para chegar em expected_names.

    Por que isso é necessário: xarray/disscube NÃO garantem que a
    ordem de eixos gravada em disco seja (y, x) — o próprio
    CubeClient.load() do disscube faz `.transpose("y", "x")`
    defensivamente antes de usar qualquer array, precisamente porque
    a ordem pode vir diferente. Um array QUADRADO com eixos trocados
    tem o MESMO shape nos dois casos — a checagem de shape sozinha
    não detecta a troca; é silenciosa, não trava.

    Retorna None se dimension_names não estiver disponível (Zarr sem
    metadado de dimensão — não há como verificar, assume-se a ordem
    como está, mesmo comportamento de antes desta correção).
    """
    dims = getattr(getattr(arr, "metadata", None), "dimension_names", None)
    if not dims:
        return None
    dims = tuple(dims)
    if set(dims) != set(expected_names):
        return None  # nomes de dimensão inesperados -- não arrisca reordenar
    return tuple(dims.index(name) for name in expected_names)


def _open_variable(store: str, variable_name: str | None):
    """Abre um zarr store, que pode ser um grupo (com várias variáveis,
    acessadas por nome) ou um array único (variável direta)."""
    opened = zarr.open(store, mode="r")
    if hasattr(opened, "arrays") or hasattr(opened, "array_keys"):
        # É um grupo (zarr.Group) — precisa do nome da variável dentro dele.
        if variable_name is None:
            raise ValueError(
                f"'{store}' é um grupo Zarr com múltiplas variáveis — "
                f"informe o nome da variável em variable_map."
            )
        return opened[variable_name]
    # É um array único — variable_name é ignorado (ou usado só como rótulo).
    return opened


def load_zarr_into_workspace(
    workspace: MemmapRasterWorkspace,
    store: str | Path,
    variable_map: dict[str, str] | None = None,
    time_index: int | None = None,
) -> None:
    """
    Popula um MemmapRasterWorkspace bloco a bloco a partir de um Zarr
    store, lendo apenas a fatia de cada bloco por vez.

    Parameters
    ----------
    store : str | Path
        Caminho do Zarr store — pode ser um grupo (múltiplas variáveis,
        acessadas por nome) ou um array único.
    variable_map : dict[nome_array_workspace, nome_variavel_zarr], optional
        Mapeia nomes de array do workspace para nomes de variável
        dentro do grupo zarr. Se None, assume que os nomes já batem
        (mesmo nome no workspace e no zarr), e que `store` é um único
        array (não um grupo) se nenhuma variável for nomeada.
    time_index : int, optional
        Se a variável tiver uma dimensão temporal inicial (shape
        (time, y, x), padrão do "Temporal Backend" do disscube para
        produtos derivados com janela de validade), qual índice de
        tempo carregar. Obrigatório se a variável for 3D.

    Raises
    ------
    ValueError
        Se o shape (y, x) da variável não bater com workspace.shape,
        ou se uma variável 3D for informada sem time_index.
    """
    if not HAS_ZARR:
        raise ImportError("zarr é necessário — pip install -e '.[zarr]'")

    store = str(store)
    declared = set(workspace.metadata["arrays"])
    var_map = variable_map or {name: name for name in declared}

    for array_name, zarr_var_name in var_map.items():
        if array_name not in declared:
            continue

        arr = _open_variable(store, zarr_var_name)

        if arr.ndim == 3:
            if time_index is None:
                raise ValueError(
                    f"Variável '{zarr_var_name}' tem 3 dimensões (provável "
                    f"dimensão temporal) — informe time_index."
                )
            expected_dims = ("time", "y", "x")
        elif arr.ndim == 2:
            expected_dims = ("y", "x")
        else:
            raise ValueError(
                f"Variável '{zarr_var_name}' tem {arr.ndim} dimensões; esperado 2 ou 3."
            )

        # Normaliza a ordem de eixos para (y, x) ou (time, y, x) usando
        # o metadado nativo de dimensão do Zarr v3, quando disponível.
        # Necessário porque a ordem de eixos gravada NÃO é garantida
        # (ver docstring de _resolve_axis_order) — um array quadrado
        # com eixos trocados tem o mesmo shape nos dois casos, então a
        # checagem de shape sozinha não pega a inversão.
        axis_order = _resolve_axis_order(arr, expected_dims)

        shape2d = arr.shape[1:] if arr.ndim == 3 else arr.shape
        if axis_order is not None:
            reordered_shape = tuple(arr.shape[i] for i in axis_order)
            shape2d = reordered_shape[1:] if arr.ndim == 3 else reordered_shape

        if shape2d != tuple(workspace.shape):
            raise ValueError(
                f"Shape de '{zarr_var_name}' {shape2d} não bate com o "
                f"shape do workspace {tuple(workspace.shape)}."
            )

        for block in workspace.blocks():
            disk_index: list = [slice(None)] * arr.ndim
            y_disk_axis = x_disk_axis = None
            time_disk_axis = None

            for canonical_pos, name in enumerate(expected_dims):
                disk_axis = axis_order[canonical_pos] if axis_order is not None else canonical_pos
                if name == "time":
                    disk_index[disk_axis] = time_index
                    time_disk_axis = disk_axis
                elif name == "y":
                    disk_index[disk_axis] = slice(block.r0, block.r1)
                    y_disk_axis = disk_axis
                elif name == "x":
                    disk_index[disk_axis] = slice(block.c0, block.c1)
                    x_disk_axis = disk_axis

            raw = np.asarray(arr[tuple(disk_index)])

            if axis_order is not None and raw.ndim == 2:
                # raw ainda está na ordem relativa em disco (menos o
                # eixo de tempo, já reduzido pela indexação inteira
                # acima) -- transpõe para (y, x) canônico.
                def _shift(pos):
                    return pos - 1 if (time_disk_axis is not None and time_disk_axis < pos) else pos
                data = np.transpose(raw, (_shift(y_disk_axis), _shift(x_disk_axis)))
            else:
                data = raw

            workspace.write_block_to_read_slot(block, array_name, data)

    workspace.flush()
