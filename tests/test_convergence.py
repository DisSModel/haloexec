"""
Prova de equivalência para sweep_until_convergence, usando exatamente
o caso de uso que motivou a primitiva: propagação de conectividade
(binary_propagation) por blocos+halo+varreduras, comparada a um
binary_propagation monolítico no domínio inteiro de uma vez.

Isso é a prova que o protótipo original (chunked_engine.py::propagar_conectividade)
nunca teve — nenhum teste lá confirmava que a versão em blocos convergia
para o mesmo resultado exato que a versão monolítica.
"""

import numpy as np
import pytest

scipy_ndimage = pytest.importorskip("scipy.ndimage")
from scipy.ndimage import binary_propagation

from haloexec import MemmapRasterWorkspace, sweep_until_convergence


def _connectivity_rule(window: dict[str, np.ndarray], halo: int = 1) -> dict[str, np.ndarray]:
    """Mesma lógica do aluno: dilata 'conectado' através de 'permeavel',
    dentro da janela com halo, e devolve só o núcleo."""
    connected = window["conectado"].astype(bool)
    permeable = window["permeavel"].astype(bool)
    propagated = binary_propagation(connected, mask=permeable)
    core = propagated[halo:-halo, halo:-halo]
    return {"conectado": core.astype(np.uint8)}


def _run_monolithic(seeds: np.ndarray, permeable: np.ndarray) -> np.ndarray:
    return binary_propagation(seeds.astype(bool), mask=permeable.astype(bool)).astype(np.uint8)


def _run_chunked(tmp_path, seeds: np.ndarray, permeable: np.ndarray,
                  block_h: int, block_w: int, halo: int = 1) -> tuple[np.ndarray, dict]:
    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=seeds.shape,
        arrays={"conectado": np.uint8, "permeavel": np.uint8},
        block_h=block_h, block_w=block_w, halo=halo,
    )
    ws.fill("conectado", seeds.astype(np.uint8))
    ws.fill("permeavel", permeable.astype(np.uint8))

    info = sweep_until_convergence(
        ws, lambda w: _connectivity_rule(w, halo), boundary_value=0,
    )
    ws.flush()
    return ws.snapshot("conectado"), info


def _labyrinth_scenario(height: int, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Gera um labirinto de permeabilidade que força a conectividade a
    serpentear por várias fronteiras de bloco antes de convergir —
    testa de verdade a propagação através de múltiplos blocos, não só
    vizinhança imediata de uma fonte central."""
    rng = np.random.default_rng(seed)
    permeable = rng.random((height, width)) < 0.65  # a maioria é permeável
    seeds = np.zeros((height, width), dtype=bool)
    seeds[0, 0] = True  # única fonte, no canto -- força propagação longa
    permeable[0, 0] = True
    return seeds, permeable


@pytest.mark.parametrize(
    "height, width, block_h, block_w, seed, label",
    [
        (40, 40, 10, 10, 42, "grade_divisivel_exatamente"),
        (37, 53, 8, 12, 7, "grade_com_resto_blocos_irregulares"),
        (30, 30, 6, 6, 123, "blocos_pequenos_muitas_fronteiras"),
        (20, 20, 100, 100, 99, "bloco_maior_que_grade"),
    ],
)
def test_sweep_until_convergence_equivalence(tmp_path, height, width, block_h, block_w, seed, label):
    seeds, permeable = _labyrinth_scenario(height, width, seed)

    golden = _run_monolithic(seeds, permeable)
    chunked, info = _run_chunked(tmp_path, seeds, permeable, block_h, block_w)

    n_diff = int(np.sum(golden != chunked))
    assert n_diff == 0, f"[{label}] {n_diff}/{height*width} células divergentes (info={info})"
    assert info["converged"]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_sweep_until_convergence_stress_random_seeds(tmp_path, seed):
    seeds, permeable = _labyrinth_scenario(35, 35, seed)

    golden = _run_monolithic(seeds, permeable)
    chunked, info = _run_chunked(tmp_path, seeds, permeable, block_h=7, block_w=7)

    assert np.array_equal(golden, chunked)


def test_sweep_until_convergence_raises_if_never_converges(tmp_path):
    """Regra que sempre 'muda' algo (nunca estabiliza) deve estourar
    RuntimeError, não travar num loop silencioso."""
    ws = MemmapRasterWorkspace.create(
        root=tmp_path / "workspace", shape=(10, 10),
        arrays={"contador": np.uint8}, block_h=5, block_w=5, halo=1,
    )
    ws.fill("contador", np.zeros((10, 10), dtype=np.uint8))

    def regra_instavel(window):
        # sempre incrementa -- nunca converge
        core = window["contador"][1:-1, 1:-1]
        return {"contador": (core + 1) % 250}

    with pytest.raises(RuntimeError, match="não convergiu"):
        sweep_until_convergence(ws, regra_instavel, max_sweeps=3)


def test_sweep_until_convergence_reports_sweep_count(tmp_path):
    """Uma única célula-fonte isolada (sem vizinho permeável) converge
    na primeira varredura -- caso trivial, serve de sanity check do
    contador de varreduras."""
    seeds = np.zeros((10, 10), dtype=bool)
    seeds[5, 5] = True
    permeable = np.zeros((10, 10), dtype=bool)
    permeable[5, 5] = True  # isolada -- nao tem pra onde propagar

    _, info = _run_chunked(tmp_path, seeds, permeable, block_h=5, block_w=5)
    assert info["sweeps"] == 1
