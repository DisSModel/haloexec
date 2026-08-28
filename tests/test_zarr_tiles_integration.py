"""
Teste de integração haloexec <-> disscube (pacotes separados, sem
dependência de runtime entre si).

Prova que `load_zarr_tiles_into_workspace` monta corretamente um mosaico
de N stores Zarr posicionados lado a lado — inclusive para blocos cujo
halo CRUZA a fronteira entre dois arquivos, e para buracos da malha.

É o análogo, para Zarr, do que `test_geomosaic_integration.py` prova para
GeoTIFF/VRT. A diferença importa: no caso VRT o GDAL costura antes do
haloexec entrar em cena, então um erro de posicionamento apareceria já na
leitura do VRT. Aqui a costura é feita por este módulo, tile a tile — um
offset errado, ou um tile faltando, produz dado errado numa fronteira e
em mais lugar nenhum. Por isso os testes miram exatamente as fronteiras.

O formato do layout (chaves e semântica) é o contrato com
`CubeClient.tile_layout()` do disscube; `test_layout_shape_matches_disscube`
o fixa dos dois lados quando o disscube está instalado.
"""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from haloexec import (  # noqa: E402
    Block,
    MemmapRasterWorkspace,
    load_zarr_tiles_into_workspace,
)

T = 8   # lado de cada tile — pequeno para as fronteiras ficarem inspecionáveis


def _write_tile(path, data):
    """Grava um array 2D como store Zarr com dimension_names (y, x),
    do mesmo jeito que xarray.to_zarr grava — que é como o disscube grava."""
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_array(
        "v", shape=data.shape, dtype=str(data.dtype), dimension_names=("y", "x")
    )
    arr[:] = data
    return str(path)


def _ws(tmp_path, shape, block=4, halo=1, dtype="float64"):
    return MemmapRasterWorkspace.create(
        tmp_path / "ws", shape=shape, arrays={"v": dtype},
        block_h=block, block_w=block, halo=halo,
    )


def _quadrantes(tmp_path, valores):
    """Quatro tiles TxT num workspace 2Tx2T. `valores` dá o valor constante
    de cada quadrante: (NO, NE, SO, SE). None = tile ausente (buraco)."""
    pos = [(0, 0), (0, T), (T, 0), (T, T)]
    tiles = []
    for i, ((r, c), val) in enumerate(zip(pos, valores)):
        if val is None:
            continue
        url = _write_tile(tmp_path / f"t{i}.zarr", np.full((T, T), val, dtype="float64"))
        tiles.append({
            "tile_id": f"t{i}", "variable": "v", "url": url,
            "row_off": r, "col_off": c, "height": T, "width": T,
        })
    return tiles


# ── montagem básica ──────────────────────────────────────────────────────────

def test_four_tiles_land_in_their_own_quadrants(tmp_path):
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(ws, _quadrantes(tmp_path, (1.0, 2.0, 3.0, 4.0)))

    lido = MemmapRasterWorkspace(tmp_path / "ws")
    full = np.empty((2 * T, 2 * T))
    for b in lido.blocks():
        full[b.r0:b.r1, b.c0:b.c1] = lido.read_block_core(b, "v")

    assert np.all(full[:T, :T] == 1.0), "quadrante NO"
    assert np.all(full[:T, T:] == 2.0), "quadrante NE"
    assert np.all(full[T:, :T] == 3.0), "quadrante SO"
    assert np.all(full[T:, T:] == 4.0), "quadrante SE"


def test_row_and_column_offsets_are_not_swapped(tmp_path):
    """Trocar row_off por col_off transporia o mosaico — e com tiles
    quadrados o shape continuaria certo, então só o conteúdo denuncia."""
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(ws, _quadrantes(tmp_path, (1.0, 2.0, 3.0, 4.0)))
    lido = MemmapRasterWorkspace(tmp_path / "ws")
    ne = lido.read_block_core(Block(r0=0, r1=4, c0=T, c1=T + 4), "v")
    so = lido.read_block_core(Block(r0=T, r1=T + 4, c0=0, c1=4), "v")
    assert np.all(ne == 2.0) and np.all(so == 3.0)


# ── fronteiras: o que este módulo pode quebrar sozinho ───────────────────────

def test_halo_crosses_boundary_between_two_zarr_files(tmp_path):
    """A janela com halo tem de trazer o valor do tile VIZINHO — que veio
    de outro arquivo — e não o do próprio tile nem nodata."""
    ws = _ws(tmp_path, (2 * T, 2 * T), block=4, halo=1)
    load_zarr_tiles_into_workspace(ws, _quadrantes(tmp_path, (1.0, 2.0, 3.0, 4.0)))
    lido = MemmapRasterWorkspace(tmp_path / "ws")

    # bloco encostado na fronteira vertical: halo à direita cai no tile NE
    blk = Block(r0=0, r1=4, c0=T - 4, c1=T)
    jan = lido.read_block_with_halo(blk, boundary_value=np.nan)["v"]
    assert np.all(jan[1:-1, 1:-1] == 1.0), "núcleo é do tile NO"
    assert np.all(jan[1:-1, -1] == 2.0), "halo direito tem de vir do tile NE"


def test_halo_crosses_horizontal_boundary(tmp_path):
    ws = _ws(tmp_path, (2 * T, 2 * T), block=4, halo=1)
    load_zarr_tiles_into_workspace(ws, _quadrantes(tmp_path, (1.0, 2.0, 3.0, 4.0)))
    lido = MemmapRasterWorkspace(tmp_path / "ws")
    blk = Block(r0=T - 4, r1=T, c0=0, c1=4)
    jan = lido.read_block_with_halo(blk, boundary_value=np.nan)["v"]
    assert np.all(jan[1:-1, 1:-1] == 1.0)
    assert np.all(jan[-1, 1:-1] == 3.0), "halo inferior tem de vir do tile SO"


def test_block_straddling_a_boundary_is_assembled_from_both_files(tmp_path):
    """Um bloco que cai metade num tile e metade no outro precisa das duas
    metades — é o caso que quebra se a montagem for feita tile a tile."""
    ws = _ws(tmp_path, (2 * T, 2 * T), block=4, halo=1)
    load_zarr_tiles_into_workspace(ws, _quadrantes(tmp_path, (1.0, 2.0, 3.0, 4.0)))
    lido = MemmapRasterWorkspace(tmp_path / "ws")
    nucleo = lido.read_block_core(Block(r0=0, r1=4, c0=T - 2, c1=T + 2), "v")
    assert np.all(nucleo[:, :2] == 1.0) and np.all(nucleo[:, 2:] == 2.0)


# ── buracos da malha ─────────────────────────────────────────────────────────

def test_missing_tile_becomes_fill_not_garbage(tmp_path):
    """Buraco real da malha (o MapBiomas tem vários) vira nodata."""
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(ws, _quadrantes(tmp_path, (1.0, 2.0, None, 4.0)))
    lido = MemmapRasterWorkspace(tmp_path / "ws")
    buraco = lido.read_block_core(Block(r0=T, r1=T + 4, c0=0, c1=4), "v")
    assert np.all(np.isnan(buraco))


def test_halo_over_a_hole_is_fill_while_core_keeps_data(tmp_path):
    ws = _ws(tmp_path, (2 * T, 2 * T), block=4, halo=1)
    load_zarr_tiles_into_workspace(ws, _quadrantes(tmp_path, (1.0, 2.0, None, 4.0)))
    lido = MemmapRasterWorkspace(tmp_path / "ws")
    blk = Block(r0=T - 4, r1=T, c0=0, c1=4)          # último bloco do tile NO
    jan = lido.read_block_with_halo(blk, boundary_value=0.0)["v"]
    assert np.all(jan[1:-1, 1:-1] == 1.0), "núcleo mantém o dado"
    assert np.all(np.isnan(jan[-1, 1:-1])), "halo cai no buraco -> fill"


def test_explicit_fill_value_is_used(tmp_path):
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(
        ws, _quadrantes(tmp_path, (1.0, 2.0, None, 4.0)), fill=-9999.0
    )
    lido = MemmapRasterWorkspace(tmp_path / "ws")
    assert np.all(lido.read_block_core(Block(r0=T, r1=T + 4, c0=0, c1=4), "v") == -9999.0)


def test_skip_empty_blocks_leaves_them_zero(tmp_path):
    """Modo de economia de disco: bloco vazio não é escrito, então lê zero
    (não `fill`) — a troca documentada no README."""
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(
        ws, _quadrantes(tmp_path, (1.0, 2.0, None, 4.0)), skip_empty_blocks=True
    )
    lido = MemmapRasterWorkspace(tmp_path / "ws")
    assert np.all(lido.read_block_core(Block(r0=T, r1=T + 4, c0=0, c1=4), "v") == 0.0)


# ── contrato e erros ─────────────────────────────────────────────────────────

def test_single_tile_covering_the_grid_also_works(tmp_path):
    """Variável global (sem tiles) chega como layout de um item só."""
    ws = _ws(tmp_path, (T, T))
    url = _write_tile(tmp_path / "g.zarr", np.arange(T * T, dtype="float64").reshape(T, T))
    load_zarr_tiles_into_workspace(ws, [{
        "tile_id": None, "variable": "v", "url": url,
        "row_off": 0, "col_off": 0, "height": T, "width": T,
    }])
    lido = MemmapRasterWorkspace(tmp_path / "ws")
    assert lido.read_block_core(Block(r0=0, r1=2, c0=0, c1=2), "v")[0, 0] == 0.0


def test_array_name_can_differ_from_variable_name(tmp_path):
    ws = MemmapRasterWorkspace.create(
        tmp_path / "ws", shape=(T, T), arrays={"uso": "float64"},
        block_h=4, block_w=4, halo=1,
    )
    url = _write_tile(tmp_path / "g.zarr", np.ones((T, T)))
    load_zarr_tiles_into_workspace(ws, [{
        "tile_id": None, "variable": "v", "url": url,
        "row_off": 0, "col_off": 0, "height": T, "width": T,
    }], array="uso")
    assert np.all(MemmapRasterWorkspace(tmp_path / "ws")
                  .read_block_core(Block(r0=0, r1=4, c0=0, c1=4), "uso") == 1.0)


def test_empty_tile_list_raises(tmp_path):
    ws = _ws(tmp_path, (T, T))
    with pytest.raises(ValueError, match="vazia"):
        load_zarr_tiles_into_workspace(ws, [])


def test_missing_key_names_what_is_missing(tmp_path):
    ws = _ws(tmp_path, (T, T))
    with pytest.raises(ValueError, match="row_off"):
        load_zarr_tiles_into_workspace(ws, [{"url": "x", "variable": "v"}])


def test_undeclared_array_raises(tmp_path):
    ws = _ws(tmp_path, (T, T))
    url = _write_tile(tmp_path / "g.zarr", np.ones((T, T)))
    with pytest.raises(ValueError, match="não declara"):
        load_zarr_tiles_into_workspace(ws, [{
            "tile_id": None, "variable": "inexistente", "url": url,
            "row_off": 0, "col_off": 0, "height": T, "width": T,
        }])


def test_tile_outside_workspace_raises(tmp_path):
    """Um tile fora dos limites significa grade errada — falhar alto evita
    um mosaico truncado que passaria despercebido."""
    ws = _ws(tmp_path, (T, T))
    url = _write_tile(tmp_path / "g.zarr", np.ones((T, T)))
    with pytest.raises(ValueError, match="não cabe"):
        load_zarr_tiles_into_workspace(ws, [{
            "tile_id": "fora", "variable": "v", "url": url,
            "row_off": T, "col_off": 0, "height": T, "width": T,
        }])


# ── o contrato com o disscube, quando ele está disponível ────────────────────

def test_layout_shape_matches_disscube(tmp_path):
    """Fixa que as chaves que este loader exige são as que o
    CubeClient.tile_layout() produz. Sem o disscube instalado, pula."""
    disscube = pytest.importorskip("disscube")
    from disscube.client import CubeClient
    from disscube.models import DerivedVariable, GridSpec, SpatialSource

    cube = CubeClient(catalog=str(tmp_path / "c.db"), store=str(tmp_path / "s"))
    cube.register_grid(GridSpec(
        id="G", type="local", crs="EPSG:31982", resolution=10.0,
        bbox=[0.0, 0.0, 100.0, 100.0],
    ))
    cube.register_spatial_source(SpatialSource(
        id="G_T1", name="T1", format="raster", asset_url="planned",
        crs="EPSG:31982", bbox=[0.0, 50.0, 50.0, 100.0],
    ))
    cube.catalog.save_derived(DerivedVariable(
        id="v_T1", name="v", grid_id="G", role="test", times=[], dtype="float64",
        derivation_id="d", spec_hash="h", tile_id="T1", asset_url="x.zarr",
    ))

    exigidas = {"url", "variable", "row_off", "col_off", "height", "width"}
    assert exigidas <= set(cube.tile_layout("v", "G")[0])


# ── posições sobrepostas ─────────────────────────────────────────────────────
# Defesa em profundidade: mesmo que o layout venha errado de qualquer origem,
# dois pedaços na mesma posição não devem ser aceitos. O caso real que motivou
# isto: uma variável temporal cujo layout misturava as fatias de vários anos,
# todas nas mesmas posições — carregar isso deixaria o último ano vencer, sem
# erro nenhum.

def test_two_tiles_at_the_same_position_raise(tmp_path):
    ws = _ws(tmp_path, (T, T))
    a = _write_tile(tmp_path / "a.zarr", np.ones((T, T)))
    b = _write_tile(tmp_path / "b.zarr", np.full((T, T), 2.0))
    base = {"variable": "v", "row_off": 0, "col_off": 0, "height": T, "width": T}
    with pytest.raises(ValueError, match="ocupam a posição"):
        load_zarr_tiles_into_workspace(ws, [
            {**base, "tile_id": "1985", "url": a},
            {**base, "tile_id": "1995", "url": b},
        ])


def test_overlap_error_names_both_tiles(tmp_path):
    ws = _ws(tmp_path, (T, T))
    a = _write_tile(tmp_path / "a.zarr", np.ones((T, T)))
    base = {"variable": "v", "row_off": 0, "col_off": 0, "height": T, "width": T}
    with pytest.raises(ValueError) as exc:
        load_zarr_tiles_into_workspace(ws, [
            {**base, "tile_id": "primeiro", "url": a},
            {**base, "tile_id": "segundo", "url": a},
        ])
    assert "primeiro" in str(exc.value) and "segundo" in str(exc.value)


def test_distinct_positions_still_accepted(tmp_path):
    """Guarda: a checagem não pode recusar um mosaico legítimo."""
    ws = _ws(tmp_path, (2 * T, 2 * T))
    load_zarr_tiles_into_workspace(ws, _quadrantes(tmp_path, (1.0, 2.0, 3.0, 4.0)))
