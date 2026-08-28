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


def load_zarr_tiles_into_workspace(
    workspace: MemmapRasterWorkspace,
    tiles: list[dict],
    array: str | None = None,
    fill: float | None = None,
    skip_empty_blocks: bool = False,
) -> None:
    """
    Popula UM array do workspace a partir de N stores Zarr posicionados
    lado a lado — o caso multi-tile, que `load_zarr_into_workspace` não
    cobre (ela recebe um store e exige que ele tenha o shape do workspace
    inteiro).

    Está para o Zarr como o VRT do geomosaic está para o GeoTIFF: junta
    pedaços numa grade contínua. A diferença é que não existe formato de
    mosaico para Zarr, então a costura acontece aqui, na leitura.

    Parameters
    ----------
    tiles : list[dict]
        Um dicionário por pedaço, com as chaves:

        - ``url``      — caminho do store Zarr
        - ``variable`` — nome da variável dentro do store
        - ``row_off``  — linha, em pixel, onde o pedaço começa no workspace
        - ``col_off``  — coluna, em pixel
        - ``height``   — altura do pedaço, em pixel
        - ``width``    — largura do pedaço, em pixel

        É exatamente o formato que ``CubeClient.tile_layout()`` do
        disscube devolve, mas nada aqui depende do disscube: qualquer
        origem que saiba dizer caminho e posição serve. Chaves extras são
        ignoradas.
    array : str, optional
        Nome do array NO WORKSPACE a preencher. Se None, usa o
        ``variable`` do primeiro tile — útil quando os nomes coincidem.
    fill : float, optional
        Valor para as células que nenhum tile cobre (buracos da malha,
        cantos fora da área de estudo). Se None, usa NaN para arrays de
        ponto flutuante e 0 para inteiros.
    skip_empty_blocks : bool
        Se True, blocos que nenhum tile toca não são escritos. Como os
        `.dat` do workspace nascem esparsos, isso deixa esses blocos sem
        ocupar disco — mas eles passam a LER COMO ZERO, não como `fill`.
        Só use quando 0 não for um valor válido do domínio (ver a nota
        "Esparsidade e custo de disco" no README). Default False, que
        escreve `fill` e mantém a distinção ao custo do disco.

    Raises
    ------
    ValueError
        Se `tiles` estiver vazio, se o array não for declarado no
        workspace, se algum tile faltar chave obrigatória, ou se um tile
        cair fora dos limites do workspace — todos casos em que seguir
        adiante produziria um mosaico silenciosamente errado.
    ImportError
        Se o extra "zarr" não estiver instalado.
    """
    if not HAS_ZARR:
        raise ImportError("zarr é necessário — pip install -e '.[zarr]'")
    if not tiles:
        raise ValueError("A lista de tiles está vazia — nada a carregar.")

    obrigatorias = {"url", "variable", "row_off", "col_off", "height", "width"}
    for i, t in enumerate(tiles):
        faltando = obrigatorias - set(t)
        if faltando:
            raise ValueError(
                f"tiles[{i}] não tem as chaves {sorted(faltando)}; "
                f"cada tile precisa de {sorted(obrigatorias)}."
            )

    array_name = array or tiles[0]["variable"]
    declarados = set(workspace.metadata["arrays"])
    if array_name not in declarados:
        raise ValueError(
            f"O workspace não declara o array {array_name!r} "
            f"(declarados: {sorted(declarados)})."
        )

    altura, largura = workspace.shape
    for t in tiles:
        if (t["row_off"] < 0 or t["col_off"] < 0
                or t["row_off"] + t["height"] > altura
                or t["col_off"] + t["width"] > largura):
            raise ValueError(
                f"tile {t.get('tile_id')!r} em "
                f"({t['row_off']},{t['col_off']}) {t['height']}x{t['width']} "
                f"não cabe no workspace {altura}x{largura}."
            )

    dtype = np.dtype(workspace.metadata["arrays"][array_name])
    if fill is None:
        fill = np.nan if np.issubdtype(dtype, np.floating) else 0

    abertos = {}
    try:
        for t in tiles:
            chave = (t["url"], t["variable"])
            if chave not in abertos:
                abertos[chave] = _open_variable(t["url"], t["variable"])

        # Percorre os BLOCOS do workspace, não os tiles: um bloco pode cair
        # sobre dois tiles vizinhos, ou sobre um buraco da malha. Montá-lo a
        # partir de tudo que o cobre é o que faz a costura ficar correta —
        # escrever tile a tile deixaria as bordas dependendo da ordem.
        for block in workspace.blocks():
            buf = None
            for t in tiles:
                r0 = max(block.r0, t["row_off"])
                r1 = min(block.r1, t["row_off"] + t["height"])
                c0 = max(block.c0, t["col_off"])
                c1 = min(block.c1, t["col_off"] + t["width"])
                if r0 >= r1 or c0 >= c1:
                    continue

                if buf is None:
                    buf = np.full(
                        (block.r1 - block.r0, block.c1 - block.c0), fill, dtype=dtype
                    )

                arr = abertos[(t["url"], t["variable"])]
                ordem = _resolve_axis_order(arr, ("y", "x"))
                trecho = np.asarray(arr[
                    r0 - t["row_off"]:r1 - t["row_off"],
                    c0 - t["col_off"]:c1 - t["col_off"],
                ]) if ordem in (None, (0, 1)) else np.asarray(arr[
                    c0 - t["col_off"]:c1 - t["col_off"],
                    r0 - t["row_off"]:r1 - t["row_off"],
                ]).T

                buf[r0 - block.r0:r1 - block.r0, c0 - block.c0:c1 - block.c0] = trecho

            if buf is None:
                if skip_empty_blocks:
                    continue
                buf = np.full(
                    (block.r1 - block.r0, block.c1 - block.c0), fill, dtype=dtype
                )

            workspace.write_block_to_read_slot(block, array_name, buf)
    finally:
        abertos.clear()

    workspace.flush()
