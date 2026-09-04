"""
Componentes de visualização e checkpoints para haloexec e dissmodel.

Fornece CheckpointRasterMap para desenhar e salvar quadros (PNG) em passos
ou anos específicos da simulação (evitando overhead em passos intermediários).
"""

from __future__ import annotations

from typing import Any, Iterable

try:
    from dissmodel.visualization.raster_map import RasterMap
    HAS_RASTERMAP = True
except ImportError:
    HAS_RASTERMAP = False


if HAS_RASTERMAP:
    class CheckpointRasterMap(RasterMap):
        """
        Extensão do RasterMap que permite filtrar quais passos ou anos da simulação
        serão desenhados e exportados para PNG.

        Evita o processamento gráfico do matplotlib e a leitura de páginas de memmap
        em passos intermediários, permitindo simulações de longa duração em grandes grades.

        Parâmetros
        ----------
        save_steps : Iterable[int] | None
            Lista ou conjunto de passos (ex.: [1, 5, 10, 20]) nos quais o quadro deve
            ser renderizado e salvo. Se None, comporta-se como o RasterMap padrão
            (obedecendo ao parâmetro `step` da classe base Model).
        **kwargs
            Todos os demais argumentos são repassados ao RasterMap (backend, band,
            color_map, cmap, save_frames, etc.).
        """

        def setup(  # type: ignore[override]
            self,
            *args: Any,
            save_steps: Iterable[int] | None = None,
            **kwargs: Any,
        ) -> None:
            self.save_steps: set[int] | None = set(save_steps) if save_steps is not None else None
            super().setup(*args, **kwargs)

        def execute(self) -> None:
            step = int(self.env.now())
            if self.save_steps is None or step in self.save_steps:
                super().execute()
else:
    class CheckpointRasterMap:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "RasterMap requer dissmodel instalado com extra viz: "
                "pip install 'dissmodel[viz]'"
            )
