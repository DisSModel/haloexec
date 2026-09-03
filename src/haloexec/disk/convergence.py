"""
Varreduras repetidas até convergência — para problemas de dependência
ESPACIAL NÃO-LIMITADA (conectividade, roteamento de fluxo, delineação
de bacia), onde nenhum halo de tamanho fixo resolve sozinho: o valor
de uma célula pode depender, em princípio, do domínio inteiro.

Generalizado de um protótipo de aluno
(chunked_engine.py::propagar_conectividade), que resolvia exatamente
esse problema para conectividade de maré via scipy.ndimage.binary_propagation
por bloco, repetindo varreduras globais até nenhum bloco mudar mais
nada. A regra específica dele (binary_propagation) NÃO faz parte desta
primitiva — só o padrão de orquestração (halo pequeno + repetição, em
vez de halo grande de uma vez) foi extraído. Qualquer `rule` que
opere sobre uma janela com halo e devolva um núcleo atualizado serve.

Por que isso é o padrão certo para dependência não-limitada
-----------------------------------------------------------------
Um halo de tamanho fixo só deixa informação atravessar UMA fronteira
de bloco por chamada. Repetir a varredura N vezes (N = número de
blocos, no pior caso) deixa a "frente" de propagação andar um bloco
por rodada — depois de N rodadas, necessariamente alcançou qualquer
célula alcançável no domínio inteiro, não importa o tamanho. É o
mesmo princípio de BFS distribuído por rounds limitados.

Gauss-Seidel, não Jacobi
-----------------------------------------------------------------
Cada bloco escreve o resultado IMEDIATAMENTE de volta no mesmo slot de
leitura (via write_block_core_in_place, sem ping-pong) — um bloco
processado depois, na MESMA varredura, já enxerga a atualização de um
bloco processado antes. Isso acelera a convergência (menos varreduras
necessárias) sem comprometer o resultado final, desde que a regra seja
um operador monótono (ex.: conectividade só cresce, nunca encolhe) —
nesse caso o ponto fixo final independe da ordem dos blocos, só o
número de varreduras até chegar lá muda. Para regras não-monótonas,
o resultado final PODE depender da ordem — cabe a quem escreve a regra
garantir monotonicidade se quiser esse comportamento bem definido.
"""

from __future__ import annotations

import numpy as np

from .workspace import MemmapRasterWorkspace, Block


def sweep_until_convergence(
    workspace: MemmapRasterWorkspace,
    rule,
    boundary_value=0,
    max_sweeps: int | None = None,
) -> dict:
    """
    Repete varreduras de blocos+halo até uma varredura inteira não
    mudar nenhum bloco.

    Parameters
    ----------
    rule : callable
        `rule(window: dict[str, np.ndarray]) -> dict[str, np.ndarray]`.
        Recebe a janela com halo de cada array do workspace (mesmo
        formato de MemmapRasterWorkspace.read_block_with_halo) e
        devolve os arrays atualizados, do tamanho do NÚCLEO do bloco
        (sem halo) — mesmo contrato das regras usadas em
        HaloChunkedSyncRasterModel/DiskChunkedSyncRasterModel.
    boundary_value : escalar ou dict, opcional
        Mesmo mecanismo de haloexec.resolve_boundary_value.
    max_sweeps : int, opcional
        Limite de segurança. Default: número de blocos + 1 (mesmo
        limite conservador do protótipo original — cada varredura
        completa avança a frente de propagação em pelo menos um
        bloco, então esse limite basta para qualquer domínio).

    Returns
    -------
    dict com "sweeps" (quantas varreduras rodaram até convergir),
    "blocks_changed_total" (soma de blocos que mudaram em cada
    varredura, ao longo de todas as varreduras).

    Raises
    ------
    RuntimeError
        Se não convergir dentro de max_sweeps — falha alto em vez de
        truncar silenciosamente (mesma disciplina do protótipo
        original).
    """
    blocks: list[Block] = workspace.blocks()
    if max_sweeps is None:
        max_sweeps = len(blocks) + 1

    total_changed_blocks = 0
    for sweep in range(1, max_sweeps + 1):
        changed_this_sweep = 0
        for block in blocks:
            window = workspace.read_block_with_halo(block, boundary_value=boundary_value)
            updates = rule(window)

            block_changed = False
            for name, new_core in updates.items():
                old_core = workspace.read_block_core(block, name)
                if not np.array_equal(old_core, new_core):
                    block_changed = True

            if block_changed:
                workspace.write_block_core_in_place(block, updates)
                changed_this_sweep += 1

        total_changed_blocks += changed_this_sweep
        if changed_this_sweep == 0:
            return {"sweeps": sweep, "blocks_changed_total": total_changed_blocks, "converged": True}

    raise RuntimeError(
        f"sweep_until_convergence não convergiu em {max_sweeps} varreduras "
        f"(limite = número de blocos + 1). Verifique se a regra é monótona "
        f"e termina, ou aumente max_sweeps explicitamente."
    )
