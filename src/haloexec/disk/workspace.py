"""
Workspace de arrays em disco (np.memmap) com decomposição em blocos+halo,
double-buffering e checkpoint — para grades grandes demais para caber
inteiras em RAM.

Extraído e generalizado de um protótipo de aluno (chunked_engine.py,
br_mangue_preprocess) que tinha a ideia certa mas amarrada a nomes de
estado específicos do domínio BR-MANGUE. Esta versão é genérica: não
sabe nada sobre "papel", "uso", "alt" ou qualquer domínio — apenas
armazena arrays 2D nomeados com dtype declarado.

Por que existe separado do resto do haloexec
----------------------------------------------
`HaloChunkedRasterCellularAutomaton` e `HaloChunkedSyncRasterModel`
(dissmodel_ca.py, sync_model.py) fazem halo via `np.pad` da grade
INTEIRA em memória — funciona bem até a grade caber em RAM. Quando não
cabe (o caso real de domínios costa Pará-Maranhão em escala), é preciso
nunca materializar a grade inteira: ler só a janela do bloco+halo do
disco, recortada nas bordas quando o bloco toca a borda do domínio.

Fundamentação teórica: mesma do resto do haloexec (Kjolstad & Snir
2010; Xia et al. 2025) — aqui aplicada com a variante em que o halo é
lido diretamente do arquivo em disco, sem padding em memória da grade
global.

Double-buffering
-----------------
Cada array tem dois slots físicos ("a" e "b"). Um passo de tempo lê do
slot corrente e escreve no outro slot — evita que uma célula leia o
valor já atualizado de um vizinho no mesmo passo (hazard clássico de
autômatos celulares síncronos). Ao final do passo, os slots trocam de
papel (swap lógico, sem copiar dados).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..engine import Block, make_blocks, resolve_boundary_value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class HaloWindow:
    """Janela de leitura com halo, recortada nas bordas do domínio.

    global_slices : onde ler no array global (pode ser menor que
        block+2*halo perto das bordas — não há padding em disco).
    core_offset : deslocamento (linha, coluna) do início do "core" do
        bloco dentro da janela lida, já que a janela pode começar mais
        perto do core quando o halo foi recortado numa borda.
    """
    global_slices: tuple[slice, slice]
    core_offset: tuple[int, int]


def halo_window(block: Block, shape: tuple[int, int], halo: int) -> HaloWindow:
    """Calcula a janela com halo de um bloco, recortada nas bordas do domínio."""
    height, width = shape
    r0 = max(0, block.r0 - halo)
    c0 = max(0, block.c0 - halo)
    r1 = min(height, block.r1 + halo)
    c1 = min(width, block.c1 + halo)
    return HaloWindow(
        global_slices=(slice(r0, r1), slice(c0, c1)),
        core_offset=(block.r0 - r0, block.c0 - c0),
    )


class MemmapRasterWorkspace:
    """
    Armazena arrays 2D nomeados em disco (np.memmap), com decomposição
    em blocos+halo, double-buffering e checkpoint de progresso.

    Genérico: não impõe nomes de variável nem domínio. Qualquer modelo
    (dissmodel ou não) que opere sobre arrays nomeados 2D com regra de
    vizinhança local pode usar este workspace.

    Uso típico
    ----------
    >>> ws = MemmapRasterWorkspace.create(
    ...     root=Path("/tmp/meu_workspace"),
    ...     shape=(10000, 10000),
    ...     arrays={"state": np.uint8},
    ...     block_h=512, block_w=512, halo=1,
    ... )
    >>> ws.fill("state", initial_array)
    >>> for step in range(n_steps):
    ...     for block in ws.blocks():
    ...         window = ws.read_block_with_halo(block)   # dict[name, np.ndarray]
    ...         result = minha_regra(window)               # dict[name, np.ndarray]
    ...         ws.write_block_core(block, result)
    ...     ws.swap_buffers()
    ...     ws.checkpoint(step)
    >>> ws.flush()
    """

    METADATA = "metadata.json"
    CHECKPOINT = "checkpoint.json"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        metadata_path = self.root / self.METADATA
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Workspace não inicializado: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.shape = tuple(int(v) for v in self.metadata["shape"])
        self.block_h = int(self.metadata["block_h"])
        self.block_w = int(self.metadata["block_w"])
        self.halo = int(self.metadata["halo"])
        self._dtypes = {n: np.dtype(d) for n, d in self.metadata["arrays"].items()}
        self._slots: dict[str, dict[str, np.memmap]] = {
            "a": self._open_slot("a"),
            "b": self._open_slot("b"),
        }
        checkpoint_path = self.root / self.CHECKPOINT
        self.checkpoint_data = (
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint_path.is_file()
            else {"step": 0, "read_slot": "a"}
        )

    def _slot_path(self, slot: str, name: str) -> Path:
        return self.root / slot / f"{name}.dat"

    def _open_slot(self, slot: str) -> dict[str, np.memmap]:
        return {
            name: np.memmap(self._slot_path(slot, name), dtype=dtype,
                            mode="r+", shape=self.shape)
            for name, dtype in self._dtypes.items()
        }

    @classmethod
    def create(
        cls,
        root: Path,
        shape: tuple[int, int],
        arrays: dict[str, np.dtype],
        block_h: int,
        block_w: int,
        halo: int = 1,
    ) -> "MemmapRasterWorkspace":
        """Cria um workspace novo, com os dois slots do double-buffer.

        Os arquivos `.dat` nascem ESPARSOS: só as regiões efetivamente
        escritas ocupam blocos no disco. Ler uma região nunca escrita
        devolve zero — garantia do próprio sistema de arquivos POSIX,
        idêntica ao que uma pré-escrita de zeros daria. Ou seja, a
        semântica é a mesma de antes desta mudança; o que muda é só o
        custo em disco (medido: um array de 4000x4000 float64 sai de
        122 MB reais para 0 MB até que algo seja escrito).

        Consequência prática, e limite desta economia: ela só aparece se
        quem carrega DEIXAR blocos sem escrever. Um carregador que
        preenche todo bloco — inclusive os vazios, com um sentinela como
        NaN — torna o arquivo denso de novo. E deixar de escrever
        significa que aquela região vale ZERO, não "ausente": para
        domínios em que 0 é um valor válido (um código de classe, uma
        elevação ao nível do mar) os dois casos ficam indistinguíveis.

        Distinguir "ausente" de "zero válido" precisaria de um canal a
        mais, que este formato não tem — ver a nota "Esparsidade e custo
        de disco" no README para o desenho proposto (índice de blocos
        presentes + nodata declarado por array, o mesmo mecanismo que o
        GeoTIFF usa com TileOffsets == 0).

        Nada disso afeta o custo de RAM, que é resolvido pelo memmap em
        si: o kernel traz páginas sob demanda, então percorrer a grade
        inteira nunca a materializa de uma vez.
        """
        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"O diretório do workspace deve ser novo e vazio: {root}")
        root.mkdir(parents=True, exist_ok=True)

        dtypes = {n: np.dtype(d) for n, d in arrays.items()}
        for slot in ("a", "b"):
            for name, dtype in dtypes.items():
                path = root / slot / f"{name}.dat"
                path.parent.mkdir(parents=True, exist_ok=True)
                # mode="w+" dimensiona o arquivo sem tocar os bytes: ele fica
                # esparso. NÃO pré-escrever zeros aqui — isso alocaria tudo
                # fisicamente sem mudar nada do que se lê depois.
                mm = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
                mm.flush()

        _write_json_atomic(root / cls.METADATA, {
            "shape": [int(shape[0]), int(shape[1])],
            "block_h": int(block_h),
            "block_w": int(block_w),
            "halo": int(halo),
            "arrays": {n: str(d) for n, d in dtypes.items()},
        })
        _write_json_atomic(root / cls.CHECKPOINT, {"step": 0, "read_slot": "a"})
        return cls(root)

    # ── acesso a blocos ──────────────────────────────────────────────

    def blocks(self) -> list[Block]:
        return make_blocks(self.shape[0], self.shape[1], self.block_h, self.block_w)

    def fill(self, name: str, array: np.ndarray, slot: str | None = None) -> None:
        """Popula um array inteiro (uso único, ex.: estado inicial). Para
        arrays grandes, prefira escrever bloco a bloco via write_block_core."""
        slot = slot or self.checkpoint_data["read_slot"]
        mm = self._slots[slot][name]
        mm[:] = np.asarray(array, dtype=mm.dtype)
        mm.flush()

    def read_block_with_halo(self, block: Block, boundary_value=0) -> dict[str, np.ndarray]:
        """Lê a janela com halo de todos os arrays para um bloco, do slot
        de leitura atual. Preenche com boundary_value apenas o halo que
        cai fora do domínio (bordas externas da grade) — o resto é lido
        diretamente do disco, sem materializar a grade inteira.

        boundary_value aceita um escalar (mesmo valor para todos os
        arrays) ou um dict {nome: valor} — necessário sempre que 0 não
        for um sentinela seguro para algum array (ex.: um código de
        classe válido no domínio). Ver engine.resolve_boundary_value."""
        read_slot = self._slots[self.checkpoint_data["read_slot"]]
        window = halo_window(block, self.shape, self.halo)
        full_h = block.r1 - block.r0 + 2 * self.halo
        full_w = block.c1 - block.c0 + 2 * self.halo

        result = {}
        for name, mm in read_slot.items():
            name_boundary = resolve_boundary_value(boundary_value, name)
            raw = np.asarray(mm[window.global_slices])
            if raw.shape == (full_h, full_w):
                result[name] = raw.copy()
            else:
                # bloco na borda: janela foi recortada, preenche o
                # halo faltante com o valor de contorno daquele array.
                padded = np.full((full_h, full_w), name_boundary, dtype=raw.dtype)
                dest_r0 = self.halo - window.core_offset[0]
                dest_c0 = self.halo - window.core_offset[1]
                padded[dest_r0:dest_r0 + raw.shape[0], dest_c0:dest_c0 + raw.shape[1]] = raw
                result[name] = padded
        return result

    def write_block_core(self, block: Block, values: dict[str, np.ndarray]) -> None:
        """Escreve a região core (sem halo) de um bloco no slot de
        ESCRITA (o outro slot, não o de leitura) — preserva o
        double-buffer: o passo atual nunca escreve no array que ainda
        está sendo lido por outros blocos do mesmo passo."""
        write_slot_name = "b" if self.checkpoint_data["read_slot"] == "a" else "a"
        write_slot = self._slots[write_slot_name]
        for name, core in values.items():
            write_slot[name][block.core] = core

    def read_block_core(self, block: Block, name: str) -> np.ndarray:
        """Lê só a região core (sem halo) de um array, do slot de
        leitura atual. Usado para sincronização "_past" — não precisa
        de vizinhança, só uma cópia direta."""
        read_slot = self._slots[self.checkpoint_data["read_slot"]]
        return np.asarray(read_slot[name][block.core]).copy()

    def write_block_to_read_slot(self, block: Block, name: str, values: np.ndarray) -> None:
        """Escreve no MESMO slot que está sendo lido no momento (não no
        slot de escrita do ping-pong). Uso exclusivo para popular
        arrays "<nome>_past": eles devem estar disponíveis no slot de
        leitura ANTES de execute() rodar naquele mesmo passo — ao
        contrário dos arrays "correntes", que só ficam prontos no
        próximo passo, após o swap."""
        read_slot = self._slots[self.checkpoint_data["read_slot"]]
        read_slot[name][block.core] = values

    def write_block_core_in_place(self, block: Block, values: dict[str, np.ndarray]) -> None:
        """Escreve vários arrays no MESMO slot de leitura, imediatamente
        visível a qualquer bloco processado depois (dentro da mesma
        varredura). Sem ping-pong: propositalmente diferente de
        write_block_core() (que escreve no slot oposto, só visível
        após swap_buffers()).

        Uso: algoritmos de convergência iterativa (ver
        sweep_until_convergence em convergence.py), onde não existe
        "estado congelado do início do passo" — é refinamento
        sucessivo do MESMO estado até um ponto fixo, e ver a atualização
        do bloco vizinho processado momentos antes acelera a
        convergência (iteração Gauss-Seidel, não Jacobi)."""
        read_slot = self._slots[self.checkpoint_data["read_slot"]]
        for name, core in values.items():
            read_slot[name][block.core] = core

    def swap_buffers(self) -> None:
        """Troca leitura/escrita ao final de um passo de tempo completo."""
        self.checkpoint_data["read_slot"] = (
            "b" if self.checkpoint_data["read_slot"] == "a" else "a"
        )

    def checkpoint(self, step: int) -> None:
        self.checkpoint_data["step"] = int(step)
        _write_json_atomic(self.root / self.CHECKPOINT, self.checkpoint_data)

    def snapshot(self, name: str) -> np.ndarray:
        """Lê o array inteiro do slot de leitura atual (materializa em
        RAM — usar só para inspeção/teste, nunca dentro do loop de
        blocos)."""
        return np.asarray(self._slots[self.checkpoint_data["read_slot"]][name]).copy()

    def flush(self) -> None:
        for slot in self._slots.values():
            for mm in slot.values():
                mm.flush()
