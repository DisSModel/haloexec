"""
Alocação esparsa do MemmapRasterWorkspace.

`create()` dimensiona os `.dat` sem pré-escrever zeros, então eles nascem
esparsos. Estes testes fixam as DUAS metades do contrato:

  * a semântica não mudou — ler região nunca escrita continua devolvendo
    zero, que é o que a pré-escrita dava;
  * o custo em disco mudou — só o que é escrito ocupa blocos.

O segundo depende do sistema de arquivos suportar arquivos esparsos (ext4,
xfs, btrfs suportam). Onde não suportar, o teste é pulado em vez de falhar:
a correção continua válida, só não rende ali.
"""

import numpy as np
import pytest

from haloexec import MemmapRasterWorkspace

SHAPE = (2048, 2048)
ARRAYS = {"uso": "float64"}


def _real_bytes(path):
    return path.stat().st_blocks * 512


def _fs_suporta_esparso(tmp_path) -> bool:
    p = tmp_path / "probe.bin"
    mm = np.memmap(p, dtype="float64", mode="w+", shape=(1024, 1024))
    mm.flush()
    del mm
    return _real_bytes(p) < p.stat().st_size


def _ws(tmp_path, **kw):
    return MemmapRasterWorkspace.create(
        tmp_path / "ws", shape=SHAPE, arrays=ARRAYS,
        block_h=256, block_w=256, halo=1, **kw
    )


# ── semântica: inalterada ────────────────────────────────────────────────────

def test_regiao_nunca_escrita_le_zero(tmp_path):
    """Contrato antigo preservado: workspace novo lê zerado."""
    ws = _ws(tmp_path)
    for block in ws.blocks()[:5]:
        assert np.all(ws.read_block_core(block, "uso") == 0)


def test_halo_de_workspace_novo_le_zero(tmp_path):
    ws = _ws(tmp_path)
    janela = ws.read_block_with_halo(ws.blocks()[10], boundary_value=0)["uso"]
    assert np.all(janela == 0)


def test_escrita_e_leitura_continuam_funcionando(tmp_path):
    ws = _ws(tmp_path)
    blk = ws.blocks()[3]
    dados = np.full((blk.r1 - blk.r0, blk.c1 - blk.c0), 7.5)
    ws.write_block_to_read_slot(blk, "uso", dados)
    ws.flush()
    assert np.array_equal(ws.read_block_core(blk, "uso"), dados)


def test_bloco_vizinho_de_um_escrito_continua_zero(tmp_path):
    """Escrever um bloco não deve materializar valor em outro."""
    ws = _ws(tmp_path)
    blocos = ws.blocks()
    ws.write_block_to_read_slot(
        blocos[0], "uso",
        np.ones((blocos[0].r1 - blocos[0].r0, blocos[0].c1 - blocos[0].c0)),
    )
    ws.flush()
    assert np.all(ws.read_block_core(blocos[1], "uso") == 0)


# ── custo em disco: esse é o ganho ───────────────────────────────────────────

def test_workspace_novo_nao_ocupa_disco(tmp_path):
    if not _fs_suporta_esparso(tmp_path):
        pytest.skip("sistema de arquivos não suporta arquivos esparsos")
    _ws(tmp_path)
    dat = tmp_path / "ws" / "a" / "uso.dat"
    assert dat.stat().st_size == SHAPE[0] * SHAPE[1] * 8   # tamanho aparente cheio
    assert _real_bytes(dat) < dat.stat().st_size // 10     # mas quase nada real


def test_so_o_que_foi_escrito_ocupa_disco(tmp_path):
    if not _fs_suporta_esparso(tmp_path):
        pytest.skip("sistema de arquivos não suporta arquivos esparsos")
    ws = _ws(tmp_path)
    dat = tmp_path / "ws" / "a" / "uso.dat"
    blocos = ws.blocks()
    for blk in blocos[:4]:
        ws.write_block_to_read_slot(
            blk, "uso", np.ones((blk.r1 - blk.r0, blk.c1 - blk.c0))
        )
    ws.flush()
    ocupado = _real_bytes(dat)
    assert ocupado > 0                                     # o que foi escrito conta
    assert ocupado < dat.stat().st_size // 2               # o resto não
