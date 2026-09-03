"""
Integração disco + halo com SyncRasterModel (FloodModel, MangroveModel,
e qualquer outro modelo dissmodel do mesmo padrão), sem modificar o
pacote dissmodel instalado.

Por que não basta reaproveitar HaloChunkedSyncRasterModel (in-memory)
-----------------------------------------------------------------------
SyncRasterModel.pre_execute()/post_execute() chamam synchronize(), que
faz `self.backend.get(name).copy()` sobre o array INTEIRO — se
`self.backend` fosse um wrapper em cima de um np.memmap, esse `.copy()`
materializaria a grade inteira em RAM só para gerar o snapshot
"<nome>_past", anulando o propósito de usar disco.

Este módulo REPLICA localmente a lógica de sincronização "_past" do
SyncRasterModel (não a importa nem a modifica no pacote instalado),
adaptada para copiar bloco a bloco entre memmaps via
MemmapRasterWorkspace. Quando a migração para dissmodel core acontecer,
este é o trecho que deve ser reconciliado com
dissmodel.geo.raster.sync_model.SyncRasterModel.synchronize() — por
ora, os dois vivem em paralelo, sem acoplamento.

Semântica de double-buffer para "_past"
-----------------------------------------
Diferente dos arrays "correntes" (ex. "uso", "alt"), que só ficam
prontos no OUTRO slot após um passo completo (ping-pong clássico), os
arrays "_past" precisam estar disponíveis no MESMO slot que execute()
vai LER naquele passo — por isso usam write_block_to_read_slot(), não
write_block_core(). Ver docstring de MemmapRasterWorkspace.

Uso
---
    class FloodModelDiskHalo(DiskChunkedSyncRasterModel, FloodModel):
        pass

    arrays = workspace_arrays_for_sync_model(
        base={"uso": np.int16, "alt": np.float32},
        land_use_types=["uso", "alt"],
    )
    ws = MemmapRasterWorkspace.create(root=..., shape=..., arrays=arrays,
                                       block_h=.., block_w=.., halo=1)
    ws.fill("uso", uso_inicial)
    ws.fill("alt", alt_inicial)

    env = Environment(start_time=1, end_time=n)
    FloodModelDiskHalo(workspace=ws, taxa_elevacao=0.05)
    env.run()
"""

from __future__ import annotations

import numpy as np

from dissmodel.geo.raster.backend import RasterBackend

from .workspace import MemmapRasterWorkspace


def workspace_arrays_for_sync_model(
    base: dict[str, np.dtype],
    land_use_types: list[str],
) -> dict[str, np.dtype]:
    """Deriva o dict de arrays a declarar em MemmapRasterWorkspace.create(),
    adicionando automaticamente "<nome>_past" para cada nome em
    land_use_types (mesmo dtype do array base)."""
    arrays = dict(base)
    for name in land_use_types:
        arrays[f"{name}_past"] = np.dtype(base[name])
    return arrays


class DiskChunkedSyncRasterModel:
    """
    Mixin que processa um SyncRasterModel (ex.: FloodModel) em blocos
    lidos de um MemmapRasterWorkspace, incluindo a sincronização
    "_past" feita bloco a bloco — sem nunca materializar a grade
    inteira em RAM.

    Ordem de herança (MRO): este mixin deve vir primeiro, ex.:
    `class FloodModelDiskHalo(DiskChunkedSyncRasterModel, FloodModel)`.
    """

    def setup(self, workspace: MemmapRasterWorkspace, halo: int | None = None,
              boundary_value: float = 0, **kwargs) -> None:
        self.workspace = workspace
        self.halo = workspace.halo if halo is None else halo
        self.boundary_value = boundary_value
        self._synced_before_first_execute = False

        # Placeholder leve: RasterBackend(shape=...) não aloca arrays,
        # só existe para satisfazer o contrato de RasterModel.setup()
        # (self.backend = backend; self.shape = backend.shape). O
        # backend real por bloco é criado dentro de execute().
        placeholder = RasterBackend(shape=workspace.shape)
        super().setup(backend=placeholder, **kwargs)  # delega para a subclasse real

    def _synchronize_via_workspace(self) -> None:
        """Equivalente bloco-a-bloco de SyncRasterModel.synchronize():
        copia "<nome>" -> "<nome>_past", dentro do MESMO slot de
        leitura atual (ver docstring do módulo)."""
        for name in getattr(self, "land_use_types", []):
            for block in self.workspace.blocks():
                values = self.workspace.read_block_core(block, name)
                self.workspace.write_block_to_read_slot(block, f"{name}_past", values)

    def pre_execute(self) -> None:
        if not self._synced_before_first_execute:
            self._synchronize_via_workspace()
            self._synced_before_first_execute = True

    def post_execute(self) -> None:
        self._synchronize_via_workspace()

    def execute(self) -> None:
        ws = self.workspace
        real_backend = self.backend
        real_shape = self.shape
        h = self.halo

        for block in ws.blocks():
            window = ws.read_block_with_halo(block, boundary_value=self.boundary_value)
            block_backend = RasterBackend(shape=next(iter(window.values())).shape)
            for name, arr in window.items():
                block_backend.set(name, arr)

            self.backend = block_backend
            self.shape = block_backend.shape
            try:
                super().execute()  # lógica real (ex.: FloodModel.execute)
            finally:
                self.backend = real_backend
                self.shape = real_shape

            updates = {}
            for name, arr in block_backend.arrays.items():
                # IMPORTANTE: não excluir "_past" aqui. Se este modelo não
                # gerencia um determinado "_past" (ex.: FloodModel não
                # gerencia "solo_past", só MangroveModel gerencia), ele
                # ainda precisa ser levado adiante sem alteração através
                # do swap — senão fica órfão no slot novo (nunca escrito,
                # permanece com o valor zerado/obsoleto da alocação
                # inicial do memmap). O "_past" que ESTE modelo gerencia
                # será corretamente sobrescrito por _synchronize_via_workspace
                # em post_execute(), já no slot pós-swap.
                core = arr[h:-h, h:-h] if h > 0 else arr
                updates[name] = core
            ws.write_block_core(block, updates)

        ws.swap_buffers()
