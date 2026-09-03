"""
Integração com dissmodel: HaloChunkedRasterCellularAutomaton.

Estende dissmodel.geo.raster.cellular_automaton.RasterCellularAutomaton
para executar rule() em blocos com halo, em vez de sobre a grade
inteira de uma vez.

Ponto central de design: mantém o MESMO contrato de rule() da classe
base (`rule(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]`).
Isso significa que qualquer RasterCellularAutomaton já escrito para
dissmodel roda em blocos+halo apenas trocando a classe base — nenhuma
mudança na lógica da regra é necessária. É essa propriedade que torna
a migração futura do BR-MANGUE (`chunked_engine.py`) para este motor
uma troca estrutural, não uma reescrita.

Como funciona
-------------
1. Tira um snapshot da grade global (equivalente a `self.backend.past`).
2. Preenche halo global (`np.pad`) em cada array.
3. Para cada bloco, monta um RasterBackend temporário só com a
   sub-grade + halo daquele bloco, e troca `self.backend` para ele.
   Isso é o que garante que chamadas internas da regra como
   `self.backend.focal_sum_mask(...)` operem sobre a forma local
   correta (RasterBackend.focal_sum_mask usa `self.shape` do backend
   ativo) em vez da forma global.
4. Chama `self.rule(block_backend.snapshot())` — mesma assinatura de
   sempre.
5. Recorta o halo do resultado (mantém só a região "core") e escreve
   na posição correspondente da grade global nova.
6. Restaura `self.backend` para o backend global real e aplica as
   atualizações.

Fundamentação teórica: Kjolstad & Snir (2010), Ghost Cell Pattern
(ParaPLoP); Xia et al. (2025), ISPRS IJGI 14(3):109 — ver README.md.
"""

from __future__ import annotations

import numpy as np

from dissmodel.geo.raster.backend import RasterBackend
from dissmodel.geo.raster.cellular_automaton import RasterCellularAutomaton

from ..engine import Block, make_blocks, resolve_boundary_value


class HaloChunkedRasterCellularAutomaton(RasterCellularAutomaton):
    """
    RasterCellularAutomaton que processa a grade em blocos com halo,
    em vez de de uma vez só.

    Uso: qualquer subclasse existente de RasterCellularAutomaton pode
    trocar a herança para esta classe e ganhar decomposição de domínio
    sem alterar `rule()`.

    Examples
    --------
    >>> class GameOfLife(HaloChunkedRasterCellularAutomaton):
    ...     def rule(self, arrays):
    ...         state = arrays["state"]
    ...         neighbors = self.backend.focal_sum_mask(state == 1)
    ...         born = (state == 0) & (neighbors == 3)
    ...         survive = (state == 1) & np.isin(neighbors, [2, 3])
    ...         return {"state": np.where(born | survive, 1, 0)}
    >>> b = RasterBackend(shape=(50, 50))
    >>> b.set("state", np.random.randint(0, 2, (50, 50)))
    >>> env = Environment(start_time=1, end_time=100)
    >>> GameOfLife(backend=b, block_h=10, block_w=10, halo=1)
    >>> env.run()
    """

    def setup(  # type: ignore[override]
        self,
        backend: RasterBackend,
        block_h: int,
        block_w: int,
        halo: int = 1,
        boundary_value: float = 0,
        state_attr: str = "state",
    ) -> None:
        """
        Parameters
        ----------
        backend : RasterBackend
            Backend global compartilhado (mesma semântica da classe base).
        block_h, block_w : int
            Dimensões do bloco de processamento.
        halo : int, optional
            Raio da vizinhança da regra. Deve ser >= alcance máximo de
            dependência espacial de um passo de tempo. Default 1
            (Moore/Von Neumann de vizinho imediato).
        boundary_value : float, optional
            Valor de preenchimento do halo global nas bordas externas
            da grade (fora do domínio simulado). Default 0.
        state_attr : str, optional
            Ver classe base.
        """
        super().setup(backend=backend, state_attr=state_attr)
        self.block_h = block_h
        self.block_w = block_w
        self.halo = halo
        self.boundary_value = boundary_value

    def _block_backend(self, padded: dict[str, np.ndarray], block: Block) -> RasterBackend:
        """Monta um RasterBackend temporário com a sub-grade+halo do bloco."""
        h = self.halo
        block_shape = (block.r1 - block.r0 + 2 * h, block.c1 - block.c0 + 2 * h)
        temp = RasterBackend(shape=block_shape)
        for name, arr in padded.items():
            sub = arr[block.r0: block.r1 + 2 * h, block.c0: block.c1 + 2 * h]
            temp.set(name, sub)
        return temp

    def execute(self) -> None:
        """
        Executa um passo de tempo processando a grade em blocos+halo.

        Substitui o execute() da classe base (que chama rule() uma vez
        sobre a grade inteira) por um loop de blocos, preservando o
        mesmo contrato de rule() para quem escreve a regra.
        """
        real_backend = self.backend
        height, width = real_backend.shape
        h = self.halo

        global_snapshot = real_backend.snapshot()
        padded = {
            name: np.pad(arr, h, mode="constant",
                         constant_values=resolve_boundary_value(self.boundary_value, name))
            for name, arr in global_snapshot.items()
            if arr.ndim == 2  # arrays temporais (time, y, x) não são suportados aqui
        }

        new_arrays: dict[str, np.ndarray] = {
            name: np.zeros_like(arr) for name, arr in global_snapshot.items() if arr.ndim == 2
        }

        for block in make_blocks(height, width, self.block_h, self.block_w):
            block_backend = self._block_backend(padded, block)

            # Troca temporária: garante que chamadas como
            # self.backend.focal_sum_mask(...) dentro de rule() operem
            # sobre a forma local do bloco, não a forma global.
            self.backend = block_backend
            try:
                updates = self.rule(block_backend.snapshot())
            finally:
                self.backend = real_backend

            for name, block_result in updates.items():
                core = block_result[h:-h, h:-h] if h > 0 else block_result
                if name not in new_arrays:
                    new_arrays[name] = np.zeros((height, width), dtype=core.dtype)
                new_arrays[name][block.r0:block.r1, block.c0:block.c1] = core

        for name, arr in new_arrays.items():
            real_backend.arrays[name] = arr
