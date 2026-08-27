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
            shape2d = arr.shape[1:]
        elif arr.ndim == 2:
            shape2d = arr.shape
        else:
            raise ValueError(
                f"Variável '{zarr_var_name}' tem {arr.ndim} dimensões; esperado 2 ou 3."
            )

        if shape2d != tuple(workspace.shape):
            raise ValueError(
                f"Shape de '{zarr_var_name}' {shape2d} não bate com o "
                f"shape do workspace {tuple(workspace.shape)}."
            )

        for block in workspace.blocks():
            if arr.ndim == 3:
                data = arr[time_index, block.r0:block.r1, block.c0:block.c1]
            else:
                data = arr[block.r0:block.r1, block.c0:block.c1]
            workspace.write_block_to_read_slot(block, array_name, np.asarray(data))

    workspace.flush()
