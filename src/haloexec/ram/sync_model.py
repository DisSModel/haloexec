"""
Integração com dissmodel: HaloChunkedSyncRasterModel.

Diferente de RasterCellularAutomaton (que expõe um hook rule(arrays)
dedicado), modelos baseados em SyncRasterModel/RasterModel — como o
FloodModel do BR-MANGUE — implementam a lógica científica diretamente
em execute(), lendo/escrevendo arrays nomeados no backend
(self.backend.arrays["alt"], self.backend.get("uso_past"), etc.) e
usando self.shape/self.shift/self.dirs herdados de RasterModel.

Este módulo generaliza a mesma estratégia de chunking+halo para esse
padrão, via herança múltipla cooperativa (mixin): HaloChunkedSyncRasterModel
intercepta execute() e setup(), delega a lógica real para a subclasse
concreta via super().execute(), com self.backend/self.shape trocados
temporariamente para uma sub-grade local por bloco.

Isso significa que NENHUMA linha de FloodModel (ou de qualquer outro
SyncRasterModel) precisa mudar — apenas a ordem de herança na
declaração da classe:

    class FloodModelHalo(HaloChunkedSyncRasterModel, FloodModel):
        pass

A ordem importa (MRO): o mixin deve vir primeiro, para que seu
execute()/setup() seja chamado antes, com super() delegando para a
lógica real de FloodModel.execute()/setup().

Limitação conhecida: pre_execute()/post_execute() de SyncRasterModel
(que fazem o snapshot "<name>_past") NÃO são interceptados por este
mixin — continuam operando sobre o backend global real, fora do loop
de blocos. Isso é intencional: sincronizar "_past" é uma cópia simples
de array inteiro, sem dependência de vizinhança, então não precisa de
decomposição de domínio. O halo só é necessário dentro de execute(),
onde há leitura de vizinhos via self.shift.
"""

from __future__ import annotations

import numpy as np

from dissmodel.geo.raster.backend import RasterBackend

from ..engine import make_blocks, resolve_boundary_value


class HaloChunkedSyncRasterModel:
    """
    Mixin que processa execute() de um RasterModel/SyncRasterModel em
    blocos com halo, delegando a lógica científica para a próxima
    classe na MRO via super().

    Parameters (setup, além dos que a subclasse concreta já aceita)
    ------------------------------------------------------------------
    block_h, block_w : int
        Dimensões do bloco de processamento.
    halo : int, optional
        Raio da vizinhança usado pela regra (default 1).

        ATENÇÃO — halo NÃO é sempre igual ao raio nominal do shift
        usado pela regra. Se a regra computa uma quantidade DERIVADA
        de vizinhos (ex.: um "fluxo" que depende de quantos vizinhos
        satisfazem uma condição) e depois lê essa quantidade derivada
        DE UM VIZINHO (não do próprio valor bruto), a dependência real
        é de 2 saltos, não 1 — halo=1 fica sutilmente errado perto de
        fronteiras internas de bloco (não nas bordas do domínio, que
        já são tratadas por boundary_value). Achado documentado em
        tests/test_flood_model_halo_depth_regression.py: o FloodModel
        do BR-MANGUE precisa de halo=2 por esse motivo exato
        (fluxo_viz depende de viz_baixos do vizinho, que depende dos
        vizinhos do vizinho). Ao adaptar uma regra nova, se os testes
        de equivalência passarem com dado sintético simples mas
        falharem em dado real/irregular, suspeite de dependência de
        2+ saltos antes de suspeitar de outra coisa.
    boundary_value : float, optional
        Valor de preenchimento do halo global nas bordas externas da
        grade. Default 0.
    """

    def setup(self, backend: RasterBackend, block_h: int, block_w: int,
              halo: int = 1, boundary_value: float = 0, **kwargs) -> None:
        self.block_h = block_h
        self.block_w = block_w
        self.halo = halo
        self.boundary_value = boundary_value
        super().setup(backend=backend, **kwargs)  # delega para a subclasse real

    def execute(self) -> None:
        real_backend = self.backend
        real_shape = self.shape
        height, width = real_backend.shape
        h = self.halo

        # Todos os arrays estáticos (2D) do backend global, sem distinção
        # de nome — genérico o suficiente para qualquer modelo concreto.
        static_names = [n for n, a in real_backend.arrays.items() if a.ndim == 2]
        padded = {
            n: np.pad(real_backend.arrays[n], h, mode="constant",
                      constant_values=resolve_boundary_value(self.boundary_value, n))
            for n in static_names
        }

        new_arrays: dict[str, np.ndarray] = {}

        for block in make_blocks(height, width, self.block_h, self.block_w):
            block_shape = (block.r1 - block.r0 + 2 * h, block.c1 - block.c0 + 2 * h)
            block_backend = RasterBackend(shape=block_shape)
            for name, arr in padded.items():
                sub = arr[block.r0: block.r1 + 2 * h, block.c0: block.c1 + 2 * h]
                block_backend.set(name, sub)

            # Troca temporária: garante que self.shape e self.backend
            # (usados diretamente dentro de execute() da subclasse real,
            # ex. `rows, cols = self.shape` no FloodModel) reflitam a
            # forma local do bloco, não a forma global.
            self.backend = block_backend
            self.shape = block_backend.shape
            try:
                super().execute()  # lógica real (ex.: FloodModel.execute)
            finally:
                self.backend = real_backend
                self.shape = real_shape

            # Reconcilia: recorta o halo e escreve na grade global nova.
            # Arrays "<name>_past" são ignorados aqui — são geridos pelo
            # synchronize() global em pre_execute()/post_execute(), não
            # devem ser sobrescritos com fatias locais com halo.
            for name, arr in block_backend.arrays.items():
                if name.endswith("_past"):
                    continue
                core = arr[h:-h, h:-h] if h > 0 else arr
                if name not in new_arrays:
                    new_arrays[name] = np.zeros((height, width), dtype=core.dtype)
                new_arrays[name][block.r0:block.r1, block.c0:block.c1] = core

        for name, arr in new_arrays.items():
            real_backend.arrays[name] = arr
