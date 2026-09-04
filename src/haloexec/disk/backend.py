"""
Adaptador de RasterBackend para MemmapRasterWorkspace.

Permite que componentes do ecossistema dissmodel (como visualizadores
RasterMap, exportadores ou coletores) acessem os arrays do slot de leitura
atual do workspace sem materializar nem duplicar a grade inteira em RAM.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .workspace import MemmapRasterWorkspace


class WorkspaceRasterBackend:
    """
    Adaptador leve que expõe um MemmapRasterWorkspace com a interface de
    RasterBackend (shape e dicionário de arrays).

    Os arrays expostos são referências diretas (np.memmap) ao slot de leitura
    atual do workspace, respeitando double-buffering e swap_buffers.

    Parâmetros
    ----------
    workspace : MemmapRasterWorkspace
        Workspace em disco a ser adaptado.
    stride : int, default=1
        Fator de subamostragem espacial (decimação) ao acessar arrays.
        Útil para visualização em larga escala (ex.: 30M+ células),
        reduzindo significativamente o tempo de renderização e o uso de memória
        do matplotlib sem alterar os dados no disco.
    nodata_value : float | int | None, default=None
        Valor de nodata opcional para integração com extent masks do RasterMap.
    """

    def __init__(
        self,
        workspace: MemmapRasterWorkspace,
        stride: int = 1,
        nodata_value: float | int | None = None,
    ) -> None:
        self.workspace = workspace
        self.stride = max(1, int(stride))
        self.nodata_value = nodata_value
        h, w = workspace.shape
        self.shape: tuple[int, int] = (h // self.stride, w // self.stride)

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """Dicionário com os arrays do slot de leitura corrente."""
        slot = self.workspace.checkpoint_data["read_slot"]
        memmaps = self.workspace._slots[slot]
        if self.stride == 1:
            return memmaps
        return {name: mm[::self.stride, ::self.stride] for name, mm in memmaps.items()}

    def get(self, name: str) -> np.ndarray:
        """Retorna o array correspondente ao nome."""
        return self.arrays[name]

    def snapshot(self) -> dict[str, np.ndarray]:
        """Retorna uma cópia dos arrays do slot de leitura atual."""
        return {k: np.asarray(v).copy() for k, v in self.arrays.items()}

    def __repr__(self) -> str:
        names = list(self.workspace.metadata["arrays"].keys())
        return (
            f"WorkspaceRasterBackend(shape={self.shape}, arrays={names}, "
            f"stride={self.stride}, slot={self.workspace.checkpoint_data['read_slot']})"
        )
