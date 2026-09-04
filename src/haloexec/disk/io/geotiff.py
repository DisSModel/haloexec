"""
Carrega um GeoTIFF diretamente para um MemmapRasterWorkspace, bloco a
bloco, via rasterio.windows.Window — nunca materializa uma banda
inteira em RAM.

Extraído e generalizado do mesmo padrão usado em um protótipo de aluno
(chunked_engine.py, br_mangue_preprocess), que já lia rasters reais
bloco a bloco com `rasterio.open(...).read(banda, window=Window(...))`.
Aqui a extração usa a convenção `band_spec` já estabelecida em
`dissmodel.io.raster.load_geotiff` — lista de (nome, dtype, nodata) —
em vez de nomes de banda hardcoded ("papel", "elevação"), então serve
tanto para o TIFF_BANDS do BR-MANGUE quanto para qualquer outro layout.

Por que não usar dissmodel.io.raster.load_geotiff diretamente
-----------------------------------------------------------------
Essa função (real, do pacote dissmodel) lê cada banda inteira de uma
vez (`ds.read(i)`) para dentro de um RasterBackend em RAM — correta
para o caminho HaloChunkedSyncRasterModel (que já materializa a grade
global de qualquer forma), mas inadequada para MemmapRasterWorkspace,
cujo propósito é justamente nunca materializar a grade inteira.

Requer rasterio (extra opcional "geotiff": pip install -e ".[geotiff]").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from ..workspace import MemmapRasterWorkspace


def load_geotiff_into_workspace(
    workspace: MemmapRasterWorkspace,
    path: str | Path,
    band_spec: list[tuple[str, str, float]],
) -> None:
    """
    Popula um MemmapRasterWorkspace bloco a bloco a partir de um único
    GeoTIFF. Atalho de conveniência sobre load_geotiffs_into_workspace
    (múltiplos arquivos) para o caso comum de um arquivo só.

    Parameters
    ----------
    workspace : MemmapRasterWorkspace
        Já criado com o mesmo shape do GeoTIFF (workspace.shape deve
        bater com (altura, largura) do arquivo) e com os arrays de
        `band_spec` já declarados em `arrays=` no `.create()`.
    path : str | Path
        Caminho do GeoTIFF local.
    band_spec : list[(nome, dtype, nodata)]
        Mesmo formato de dissmodel.io.raster.load_geotiff — banda 1
        do arquivo mapeia para band_spec[0], banda 2 para band_spec[1],
        etc. Arrays cujo nome não foi declarado no workspace são
        ignorados (permite carregar só um subconjunto das bandas).

    Nota sobre nodata: este loader NÃO filtra nem substitui valores de
    nodata — copia os valores brutos do arquivo. Use o `nodata` de
    cada entrada de band_spec para configurar o `boundary_value`
    correspondente nas camadas de halo (ver achado documentado no
    README sobre boundary_value por-array).
    """
    load_geotiffs_into_workspace(workspace, [(path, band_spec)])


def load_geotiffs_into_workspace(
    workspace: MemmapRasterWorkspace,
    sources: list[tuple[str | Path, list[tuple[str, str, float]]]],
) -> None:
    """
    Popula um MemmapRasterWorkspace bloco a bloco a partir de
    MÚLTIPLOS arquivos GeoTIFF, lendo a MESMA janela de bloco de cada
    arquivo por vez.

    Generaliza load_geotiff_into_workspace (um único arquivo) para o
    padrão de múltiplos rasters usado em um protótipo de aluno
    (chunked_engine.py: `dominio_path` + `base_path` lidos juntos,
    validando shape/CRS consistentes antes de processar). O motor de
    blocos+halo (Block/make_blocks/halo_window) sempre foi agnóstico a
    quantos arquivos alimentam a grade — um bloco é só uma posição
    (r0:r1, c0:c1); esta função é o que estava faltando para o
    carregador acompanhar essa generalidade.

    Parameters
    ----------
    workspace : MemmapRasterWorkspace
        Já criado com shape batendo com TODOS os arquivos de entrada.
    sources : list[(caminho, band_spec)]
        Cada arquivo contribui com os arrays declarados em seu próprio
        band_spec (mesmo formato de load_geotiff_into_workspace). Um
        array pode vir de qualquer um dos arquivos — não precisa haver
        sobreposição de nomes entre band_specs de arquivos diferentes.

    Raises
    ------
    ValueError
        Se os arquivos não tiverem o mesmo shape entre si, ou não
        baterem com o shape do workspace, ou tiverem CRS diferentes
        entre si (quando CRS está definido em mais de um arquivo).
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio é necessário — pip install -e '.[geotiff]'")

    declared = set(workspace.metadata["arrays"])
    datasets = [rasterio.open(str(path)) for path, _ in sources]

    try:
        ref_shape = (datasets[0].height, datasets[0].width)
        ref_crs = datasets[0].crs
        for ds, (path, _) in zip(datasets, sources):
            shape = (ds.height, ds.width)
            if shape != ref_shape:
                raise ValueError(
                    f"Shape inconsistente entre arquivos: {path} tem {shape}, "
                    f"esperado {ref_shape} (do primeiro arquivo da lista)."
                )
            if ds.crs is not None and ref_crs is not None and ds.crs != ref_crs:
                raise ValueError(
                    f"CRS inconsistente entre arquivos: {path} tem {ds.crs}, "
                    f"esperado {ref_crs} (do primeiro arquivo da lista)."
                )

        if ref_shape != tuple(workspace.shape):
            raise ValueError(
                f"Shape dos GeoTIFFs {ref_shape} não bate com o shape "
                f"do workspace {tuple(workspace.shape)}."
            )

        for block in workspace.blocks():
            window = Window(block.c0, block.r0, block.c1 - block.c0, block.r1 - block.r0)
            for ds, (path, band_spec) in zip(datasets, sources):
                for band_index, (name, dtype, _nodata) in enumerate(band_spec, start=1):
                    if name not in declared or band_index > ds.count:
                        continue
                    arr = ds.read(band_index, window=window).astype(dtype)
                    workspace.write_block_to_read_slot(block, name, arr)
    finally:
        for ds in datasets:
            ds.close()

    workspace.flush()


def save_workspace_to_geotiff(
    workspace: MemmapRasterWorkspace,
    path: str | Path,
    bands: list[str] | list[tuple[str, str, float]],
    transform: Any = None,
    crs: Any = "EPSG:31984",
    compress: str = "lzw",
) -> None:
    """
    Grava arrays do MemmapRasterWorkspace em um GeoTIFF em janelas
    (bloco a bloco), sem nunca materializar a grade inteira em RAM.

    Parameters
    ----------
    workspace : MemmapRasterWorkspace
        Workspace de onde ler os dados do slot de leitura atual.
    path : str | Path
        Caminho do arquivo GeoTIFF de saída.
    bands : list[str] ou list[(nome, dtype, nodata)]
        Nomes dos arrays a gravar como bandas (1, 2, ...).
    transform : Affine, optional
        Matriz de geotransformação do rasterio. Se None, cria uma padrão.
    crs : CRS ou str, default="EPSG:31984"
        Sistema de referência de coordenadas.
    compress : str, default="lzw"
        Compressão do GeoTIFF.
    Note
    ----
    GeoTIFF/GDAL exige um único dtype e um único nodata para TODAS as
    bandas de um arquivo (limitação do formato, não deste código —
    testado empiricamente: GDAL colapsa nodata por banda para um valor
    só, mesmo passando valores distintos). Se `bands` misturar dtypes
    ou nodata diferentes, esta função converte tudo para o dtype comum
    (menor tipo que comporta todos, via `np.result_type`) e usa o
    nodata da PRIMEIRA banda para o arquivo inteiro — e emite um aviso
    (`warnings.warn`) sempre que isso descartar informação, em vez de
    fazer silenciosamente. Se cada array precisa manter seu próprio
    dtype/nodata, grave um GeoTIFF por array em vez de multibanda.
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio é necessário — pip install -e '.[geotiff]'")

    import warnings
    from rasterio.transform import from_origin

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parsed_bands: list[tuple[str, str, float | None]] = []
    for item in bands:
        if isinstance(item, tuple):
            parsed_bands.append((item[0], item[1], item[2]))
        else:
            name = str(item)
            dtype_str = str(workspace.metadata["arrays"][name])
            parsed_bands.append((name, dtype_str, None))

    height, width = workspace.shape
    if transform is None:
        transform = from_origin(500_000.0, 9_700_000.0, 30.0, 30.0)
        warnings.warn(
            "save_workspace_to_geotiff: nenhum `transform` foi passado — "
            "usando origem padrão arbitrária (500000, 9700000, 30x30m). "
            "O GeoTIFF resultante NÃO estará georreferenciado corretamente "
            "a menos que essa origem coincida com o domínio real. Passe "
            "`transform=` explicitamente para dados reais.",
            stacklevel=2,
        )

    distinct_dtypes = {dtype for _, dtype, _ in parsed_bands}
    common_dtype = np.result_type(*(np.dtype(d) for d in distinct_dtypes))
    if len(distinct_dtypes) > 1:
        warnings.warn(
            f"save_workspace_to_geotiff: bandas com dtypes distintos "
            f"{sorted(distinct_dtypes)} — GeoTIFF exige um único dtype por "
            f"arquivo, convertendo tudo para {common_dtype} (pode aumentar "
            f"o tamanho do arquivo e/ou alterar a semântica de arrays "
            f"categóricos). Para preservar dtypes, grave um GeoTIFF "
            f"separado por array.",
            stacklevel=2,
        )

    distinct_nodatas = {n for _, _, n in parsed_bands if n is not None}
    first_nodata = parsed_bands[0][2]
    if len(distinct_nodatas) > 1:
        warnings.warn(
            f"save_workspace_to_geotiff: bandas com nodata distintos "
            f"{sorted(distinct_nodatas)} — GeoTIFF só suporta um nodata "
            f"por arquivo (limitação do GDAL, testado empiricamente: "
            f"valores por banda são silenciosamente colapsados). Usando "
            f"nodata da primeira banda ({first_nodata!r}) para o arquivo "
            f"inteiro; as demais bandas ficam sem nodata correto.",
            stacklevel=2,
        )

    tiled_kwargs = {}
    if width >= 16 and height >= 16:
        bx = (min(workspace.block_w, width) // 16) * 16 or 16
        by = (min(workspace.block_h, height) // 16) * 16 or 16
        tiled_kwargs = {"tiled": True, "blockxsize": bx, "blockysize": by}
    else:
        tiled_kwargs = {"tiled": False}

    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(parsed_bands),
        dtype=str(common_dtype),
        crs=crs,
        transform=transform,
        nodata=first_nodata,
        compress=compress,
        **tiled_kwargs,
    ) as dst:
        for block in workspace.blocks():
            window = Window(block.c0, block.r0, block.c1 - block.c0, block.r1 - block.r0)
            for band_idx, (name, _dtype, _nodata) in enumerate(parsed_bands, start=1):
                core = workspace.read_block_core(block, name).astype(common_dtype)
                dst.write(core, band_idx, window=window)


