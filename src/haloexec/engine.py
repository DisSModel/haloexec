"""
Primitivas genéricas de Decomposição de Domínio com zonas de Halo
(Ghost Cell Pattern) — sem dependência de dissmodel.

Fundamentação teórica: Kjolstad & Snir (2010), "Ghost Cell Pattern",
ParaPLoP; aplicação em AC-LULC geoespacial: Xia et al. (2025), ISPRS
IJGI 14(3):109. Ver README.md.

Este módulo contém apenas a lógica de particionamento da grade
(Block, make_blocks). A execução em si — chamar a regra de transição
por bloco e reconciliar o resultado — é responsabilidade de quem
consome estas primitivas. A integração concreta com o dissmodel está
em `dissmodel_ca.py` (HaloChunkedRasterCellularAutomaton).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    """Um sub-domínio retangular da grade global (sem halo)."""

    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.r1 - self.r0, self.c1 - self.c0)

    @property
    def core(self) -> tuple[slice, slice]:
        """Slices prontos pra indexar a região deste bloco num array
        global (usado pela camada de disco, ex.: write_block_core)."""
        return (slice(self.r0, self.r1), slice(self.c0, self.c1))


def make_blocks(height: int, width: int, block_h: int, block_w: int) -> list[Block]:
    """Decompõe uma grade (height, width) em blocos regulares de tamanho
    (block_h, block_w). Blocos na borda direita/inferior podem ser
    menores (resíduo), conforme decomposição de domínio regular
    (Xia et al. 2025, Seção 2.1)."""
    blocks = []
    for r0 in range(0, height, block_h):
        r1 = min(r0 + block_h, height)
        for c0 in range(0, width, block_w):
            c1 = min(c0 + block_w, width)
            blocks.append(Block(r0, r1, c0, c1))
    return blocks


def resolve_boundary_value(boundary_value, name: str) -> float:
    """Resolve o valor de preenchimento do halo externo para um array
    específico. Aceita um escalar (mesmo valor para todos os arrays) ou
    um dict {nome: valor}.

    Se `name` terminar em "_past" e não tiver entrada própria no dict,
    cai automaticamente para o valor do nome base (sem "_past") — assim
    quem configura {"solo": -1} não precisa lembrar de duplicar para
    "solo_past" também. Sem esse fallback, "_past" cairia
    silenciosamente no default 0, reintroduzindo o mesmo problema que
    este mecanismo existe para evitar.

    Importante: 0 não é um sentinela seguro para todo domínio — em
    BR-MANGUE, por exemplo, `solo=0` é SOLO_CANAL_FLUVIAL, um código
    de solo VÁLIDO (não "sem dado"). Usar 0 como boundary_value para
    esse array cria fontes de migração fantasmas na borda externa do
    domínio, divergindo do resultado monolítico. Prefira alinhar
    boundary_value ao nodata real de cada array (ex.: TIFF_BANDS do
    domínio), passando um dict em vez de um escalar único.
    """
    if not isinstance(boundary_value, dict):
        return boundary_value
    if name in boundary_value:
        return boundary_value[name]
    if name.endswith("_past"):
        base_name = name[: -len("_past")]
        if base_name in boundary_value:
            return boundary_value[base_name]
    return 0
