# haloexec

Motor de execução de Autômatos Celulares por **Decomposição de
Domínio** com **zonas de Halo** (Ghost Cell Pattern), integrado ao
[dissmodel](https://pypi.org/project/dissmodel/) real via dependência
pip — sem viver dentro da pasta do repo `dissmodel` core, que está
atualmente sob revisão JOSS (issue #10827).

## Motivação

O padrão bloco+halo usado no `chunked_engine.py` (BR-MANGUE,
`dissmodel-ca`) não é específico a mangue nem a autômatos celulares em
geral — é decomposição de domínio genérica para qualquer regra de
transição com dependência de vizinhança local, sobre uma grade grande
demais para caber inteira em RAM.

O `dissmodel` core (0.6.3, PyPI) já tem `RasterCellularAutomaton`
(`dissmodel.geo.raster.cellular_automaton`), com o contrato
`rule(arrays: dict) -> dict`, mas **processa a grade inteira de uma
vez** — não há chunking. `haloexec` preenche essa lacuna.

## Design: mesmo contrato de `rule()`

`HaloChunkedRasterCellularAutomaton` estende `RasterCellularAutomaton`
e mantém **exatamente o mesmo contrato de `rule()`** da classe base.
Isso significa que qualquer regra já escrita para dissmodel roda em
blocos+halo trocando apenas a classe base — zero mudança na lógica
científica do modelo:

```python
import numpy as np
from dissmodel.core import Environment
from dissmodel.geo.raster.backend import RasterBackend
from haloexec import HaloChunkedRasterCellularAutomaton

class GameOfLife(HaloChunkedRasterCellularAutomaton):
    def rule(self, arrays):
        state = arrays["state"]
        neighbors = self.backend.focal_sum_mask(state == 1)
        born = (state == 0) & (neighbors == 3)
        survive = (state == 1) & np.isin(neighbors, [2, 3])
        return {"state": np.where(born | survive, 1, 0).astype(np.uint8)}

backend = RasterBackend(shape=(200, 200))
backend.set("state", np.random.randint(0, 2, (200, 200)).astype(np.uint8))

env = Environment(start_time=1, end_time=50)
GameOfLife(backend=backend, block_h=50, block_w=50, halo=1)
env.run()
```

A mesma classe `GameOfLife`, herdando de `RasterCellularAutomaton`
puro (sem `block_h`/`block_w`/`halo`), roda monoliticamente sem
alteração de `rule()` — é exatamente essa propriedade que os testes de
equivalência comprovam.

## Como funciona internamente

`HaloChunkedRasterCellularAutomaton.execute()`:

1. Tira snapshot da grade global e aplica `np.pad` (halo global nas
   bordas externas do domínio).
2. Para cada bloco (`Block`/`make_blocks`, particionamento puro, sem
   dependência de dissmodel), monta um `RasterBackend` temporário só
   com a sub-grade+halo daquele bloco.
3. Troca `self.backend` para o backend temporário do bloco — isso
   garante que chamadas internas da regra como
   `self.backend.focal_sum_mask(...)` operem sobre a forma **local**
   correta (`focal_sum_mask` usa `self.shape` do backend ativo).
4. Chama `self.rule(block_backend.snapshot())` — mesma assinatura de
   sempre.
5. Recorta o halo do resultado e escreve na posição correspondente da
   grade global nova.
6. Restaura `self.backend` para o backend global real.

## Fundamentação teórica

- **Padrão de engenharia:** Kjolstad, F. B.; Snir, M. *Ghost Cell
  Pattern*. In: Proceedings of the 2010 Workshop on Parallel
  Programming Patterns (ParaPLoP '10). ACM, 2010.
  https://doi.org/10.1145/1953611.1953615
- **Aplicação direta em AC-LULC geoespacial:** Xia, W. et al. *Dynamic
  Load Balancing Based on Hypergraph Partitioning for Parallel
  Geospatial Cellular Automata Models*. ISPRS Int. J. Geo-Inf.,
  14(3):109, 2025. https://doi.org/10.3390/ijgi14030109

## Instalação (desenvolvimento)

```bash
pip install -e ".[dev]"
pytest
```

Depende de `dissmodel>=0.6.3` (PyPI) como dependência real de runtime.

## Achado: `boundary_value=0` não é seguro para todo domínio

Ao validar contra `MangroveModel` (não só `FloodModel`), um teste
revelou 62 células divergentes em `solo` mesmo com um único bloco
cobrindo a grade inteira — ou seja, **não era bug de fronteira entre
blocos**, era bug de preenchimento da borda externa do domínio.

Causa: `SOLO_CANAL_FLUVIAL = 0` é um código de solo **válido** no
domínio BR-MANGUE (está inclusive em `SOIL_SOURCES`). Usar `0` como
valor de halo na borda externa cria fontes de migração fantasmas ali,
inexistentes no monolítico. `TIFF_BANDS` do domínio já define o nodata
correto por array (`uso`: 0, `alt`: -9999.0, `solo`: -1) — só não
estava sendo respeitado pelo motor de halo.

**Fix:** `boundary_value` em todas as camadas (`sync_model.py`,
`dissmodel_ca.py`, `disk_backend.py`) agora aceita um `dict {nome: valor}`
além de escalar, via `engine.resolve_boundary_value()`. Nomes
`"<nome>_past"` caem automaticamente para o valor do nome base se não
tiverem entrada própria — sem esse fallback, o bug reaparecia de forma
mais sutil (o `"_past"` gerado por `synchronize()` continuava usando
`0` mesmo com `{"solo": -1}` configurado, já que a chave procurada era
literalmente `"solo_past"`).

```python
boundary_value = {"uso": 0, "alt": -9999.0, "solo": -1}
MangroveModelHalo(backend=backend, taxa_elevacao=0.05,
                   block_h=10, block_w=10, halo=1,
                   boundary_value=boundary_value)
```

Sempre que um domínio usar `0` como código de classe válido em algum
array, `boundary_value` **precisa** ser um dict alinhado ao nodata real
daquele array — nunca confiar no default escalar `0`.

## Achado: halo=1 é insuficiente para `FloodModel` — precisa de halo=2

Validando contra o **dataset real** da Ilha do Maranhão (não sintético)
e comparando contra o golden do TerraME
(`examples/brmangue_validation/validate_against_terrame.py`), `alt`
divergiu do monolítico mesmo com `boundary_value` já corrigido — 91
células no passo 1, crescendo para 1431 no passo 19. `uso`/`solo`
continuavam perfeitos.

**Causa:** `FloodModel.execute()` computa `viz_baixos` (quantos
vizinhos têm elevação ≤ a própria célula) e deriva `fluxo` disso. O
update de uma célula usa `fluxo_viz` — **o fluxo do vizinho**, que por
sua vez depende dos **vizinhos do vizinho**. É uma dependência de 2
saltos, não 1: com `halo=1`, o `fluxo` calculado no próprio anel de
halo já está errado (seus vizinhos de 2 saltos foram zero-padded pelo
`shift2d` local), e esse erro contamina o núcleo do bloco.

Os testes sintéticos anteriores (`test_flood_model_equivalence.py`)
não pegaram isso porque sempre colocavam a fonte de inundação (`MAR`)
numa coluna inteira **na borda do domínio** — onde o artefato de
zero-padding já existe igualmente nos dois lados (monolítico e halo).
O bug só aparece quando uma célula-fonte está perto de uma fronteira
**interna** de bloco, o que só aconteceu com a costa irregular real.

`MangroveModel`, em contraste, só verifica associação direta a um
conjunto (dependência de 1 salto) — `halo=1` é suficiente para ele,
confirmado nos próprios testes.

**Fix:** nenhuma mudança de código — `halo` é responsabilidade de quem
configura o modelo, não algo que o motor possa inferir sozinho. O que
mudou foi a documentação (`sync_model.py`) alertando explicitamente
sobre dependências de 2+ saltos, e um teste de regressão
(`tests/test_flood_model_halo_depth_regression.py`, usando uma fixture
recortada 30x30 do dataset real) que trava `halo=1` como insuficiente
e `halo=2` como correto para este modelo especificamente.

**Lição geral:** o halo correto não é o raio nominal do `shift` usado
pela regra — é a profundidade real da cadeia de dependências. Se a
regra lê uma quantidade derivada de vizinhos (não o valor bruto) de
outra célula, a dependência é de N+1 saltos, não N. Testes sintéticos
com fontes só na borda do domínio podem mascarar esse problema —
validação contra dado real irregular é o que expõe.

## Duas opções de entrada: GeoTIFF/VRT ou Zarr

`geotiff_io.py`/`mosaic_io.py` (TIFF/VRT, via `rasterio`) e `zarr_io.py`
(via `zarr`) são caminhos de entrada **intercambiáveis** — ambos
populam o mesmo `MemmapRasterWorkspace`, bloco a bloco, sem
materializar o array inteiro em RAM. Troca-se o loader; o resto do
pipeline (halo, disco, modelos) não muda.

- **GeoTIFF/VRT**: para dado que já está em raster tradicional, ou
  para mosaicos de tiles (ver seção "Mosaico de tiles" acima).
- **Zarr**: pensado para consumir `DerivedVariable` do
  [disscube](https://github.com/DisSModel/disscube) diretamente — que
  já armazena nativamente em Zarr, já alinhado à grade mestra pelo seu
  `GridAligner` (com resampling por-operador e alinhamento fino para
  variáveis categóricas — mais rigoroso que fazer isso na mão).
  Suporta tanto um grupo com várias variáveis (`variable_map` opcional
  se os nomes diferirem) quanto um array único, e variáveis com
  dimensão temporal (`time_index`, o "Temporal Backend" do disscube).

```python
from haloexec import MemmapRasterWorkspace, load_zarr_into_workspace

ws = MemmapRasterWorkspace.create(root="workspace", shape=(altura, largura),
                                   arrays={"uso": np.int16, "alt": np.float32},
                                   block_h=512, block_w=512, halo=1)
load_zarr_into_workspace(ws, "data/derived/BDC_100m/009002/abc123/",
                          variable_map={"uso": "land_use", "alt": "elevation"})
```

Requer o extra opcional `zarr` (`pip install -e ".[zarr]"`).

## Mosaico de tiles (múltiplos arquivos de satélite)

Resolvido por um pacote **separado**:
[`geomosaic`](https://github.com/LambdaGeo/geomosaic) — descoberta de
tiles, validação de contrato de grade, e construção de VRT, sem
nenhuma dependência de `haloexec`. Mosaico é estritamente anterior a
qualquer motor de execução; o `haloexec` não sabe (nem precisa saber)
que existe mosaico por baixo — um VRT se abre com `rasterio.open()`
exatamente como um GeoTIFF comum, inclusive para blocos que cruzam a
fronteira entre tiles.

```python
from geomosaic import discover_tiles, build_mosaic_contract, write_vrt
from haloexec import MemmapRasterWorkspace, load_geotiff_into_workspace

tiles = discover_tiles("dados/mapbiomas_tiles/")
contract = build_mosaic_contract(tiles)
vrt_path = write_vrt(contract, "dados/mosaico.vrt")

ws = MemmapRasterWorkspace.create(root="workspace", shape=(contract.mosaic_height, contract.mosaic_width),
                                   arrays={"uso": np.int16}, block_h=512, block_w=512, halo=1)
load_geotiff_into_workspace(ws, vrt_path, [("uso", "int16", 0)])
```

`tests/test_geomosaic_integration.py` prova essa integração (requer o
extra opcional `geomosaic`, usado só em teste — nunca importado pelo
código de runtime do `haloexec`).

## Carregando GeoTIFF direto pra disco

`geotiff_io.py` (`load_geotiff_into_workspace`) carrega um GeoTIFF real
bloco a bloco direto para um `MemmapRasterWorkspace`, via
`rasterio.windows.Window` — nunca materializa uma banda inteira em RAM.
Generaliza o mesmo padrão usado em um protótipo de aluno
(`chunked_engine.py`), mas usando a convenção `band_spec` já
estabelecida em `dissmodel.io.raster.load_geotiff` (lista de
`(nome, dtype, nodata)`) em vez de nomes de banda hardcoded.

```python
from haloexec import MemmapRasterWorkspace, load_geotiff_into_workspace

ws = MemmapRasterWorkspace.create(
    root="/dados/workspace", shape=(altura, largura),
    arrays={"uso": np.int16, "alt": np.float32, "solo": np.int16, "mask": np.uint8},
    block_h=512, block_w=512, halo=2,
)
load_geotiff_into_workspace(ws, "dominio.tif", TIFF_BANDS + [("mask","uint8",0)])
```

Requer o extra opcional `geotiff` (`pip install -e ".[geotiff]"`).
Validado (`tests/test_geotiff_io_equivalence.py`) com round-trip exato
contra leitura direta com rasterio, incluindo blocos irregulares e
bloco maior que a grade.

## ⚠️ Problema aberto: disco + Flood+Mangrove combinados no dataset real diverge em `alt`

Ao validar `examples/brmangue_validation/validate_against_terrame_disk.py`
contra o dataset real completo (323×349, máscara irregular real),
`alt` diverge (~300 células de 112.727, crescendo com os passos)
quando `FloodModel`+`MangroveModel` rodam **combinados no mesmo
workspace em disco**. `uso`/`solo` continuam perfeitos.

**O que já foi descartado como causa** (testado e refutado
empiricamente, não por suposição):
- **Não é halo insuficiente** — idêntico com halo=2, 3 e 4.
- **Não é `boundary_value`** — idêntico com `alt`=0 ou `alt`=-9999.
- **Não é chunking** — persiste mesmo com um único bloco cobrindo o
  domínio inteiro.
- **Não é o carregamento do TIFF** — round-trip verificado bit-exato
  para todos os arrays, incluindo `alt`, antes de rodar qualquer modelo.
- **Não é o bug de `_past` órfão** já corrigido — persiste mesmo depois
  do fix, e o teste de regressão sintética para esse bug específico
  continua passando (0 diff), então não é uma regressão dele.

**O que ainda não foi isolado:** a divergência aparece comparando
RAM-halo (bloco único) vs. disco-halo (bloco único) com os MESMOS
dados de entrada — ou seja, é algo na mecânica do
`DiskChunkedSyncRasterModel` que difere do `HaloChunkedSyncRasterModel`
mesmo no caso degenerado sem chunking real, e só se manifesta com o
dataset real grande/com máscara irregular — não aparece em nenhum
teste sintético da suíte, incluindo o teste de combinação Flood+Mangrove
em disco (`test_disk_combined_models_regression.py`), que não incluía
uma máscara irregular real.

**Não travei isso em teste de regressão ainda** porque não sei
reproduzir a causa raiz de forma mínima — reproduzir exige o dataset
real completo. Antes de confiar no caminho disco para os dois modelos
combinados em produção, este problema precisa ser resolvido.
Individualmente, cada modelo no caminho disco está validado (ver
`test_disk_flood_model_equivalence.py`,
`test_disk_mangrove_model_equivalence.py`) — a lacuna é especificamente
a combinação dos dois no dataset real.

## Testando contra o TerraME (dataset real)

`examples/brmangue_validation/` traz o dataset real da Ilha do
Maranhão (`elevacao_pol.zip`, 50.496 células, grade 323×349) e os 20
CSVs golden do TerraME (Bezerra 2014), junto com um script que roda
`FloodModel`+`MangroveModel` monolítico e em blocos+halo, comparando
ambos contra o TerraME e um contra o outro:

```bash
python examples/brmangue_validation/validate_against_terrame.py \
    --end-time 19 --checkpoints 1 5 10 15 19 \
    --block-h 64 --block-w 64 --halo 2
```

Resultado esperado: `uso`/`solo` 100% idênticos entre monolítico e
halo em todos os passos; `alt` idêntico até tolerância de ponto
flutuante (1e-9); ambas as variantes batem com o TerraME na mesma
proporção (~97-100%, a pequena deriva em `alt` é drift de ponto
flutuante Python vs. Lua, não um bug — documentado no
`brmangue-dissmodel` original).

**Nota de escopo:** o resultado acima (100% de equivalência) vale para
o caminho **RAM** (`validate_against_terrame.py`). O caminho **disco**
combinando os dois modelos no dataset real tem uma divergência residual
não resolvida em `alt` — ver "⚠️ Problema aberto" acima antes de usar
`validate_against_terrame_disk.py` como prova de correção.

## Convergência iterativa (dependência espacial não-limitada)

`convergence.py` (`sweep_until_convergence`) resolve um problema
DIFERENTE do que halo de tamanho fixo resolve: conectividade,
roteamento de fluxo, delineação de bacia — onde o valor de uma célula
pode depender, em princípio, do domínio inteiro, não só dos vizinhos
imediatos.

Generalizado de um protótipo de aluno
(`chunked_engine.py::propagar_conectividade`, conectividade de maré via
`scipy.ndimage.binary_propagation`), mas a regra específica dele NÃO
faz parte desta primitiva — só o padrão de orquestração foi extraído:
halo pequeno + repetição de varreduras globais até nenhum bloco mudar,
em vez de halo grande de uma vez. Cada varredura escreve o resultado
**imediatamente** de volta (Gauss-Seidel, via
`write_block_core_in_place` — sem ping-pong), então um bloco processado
depois já enxerga a atualização de um bloco processado antes na mesma
varredura, acelerando a convergência.

```python
from haloexec import MemmapRasterWorkspace, sweep_until_convergence
from scipy.ndimage import binary_propagation

def regra_conectividade(window, halo=1):
    conectado = window["conectado"].astype(bool)
    permeavel = window["permeavel"].astype(bool)
    propagado = binary_propagation(conectado, mask=permeavel)
    return {"conectado": propagado[halo:-halo, halo:-halo].astype(np.uint8)}

info = sweep_until_convergence(ws, regra_conectividade, boundary_value=0)
# info = {"sweeps": N, "blocks_changed_total": M, "converged": True}
```

**Validado contra o que faltava no original:** `tests/test_convergence.py`
prova que a versão em blocos+varreduras converge para o resultado
**exatamente idêntico** a um `binary_propagation` monolítico rodado no
domínio inteiro de uma vez — em 4 configurações de bloco (incluindo um
labirinto de permeabilidade forçando a conectividade a atravessar
várias fronteiras de bloco antes de convergir) + 5 seeds de estresse +
caso de não-convergência (deve estourar `RuntimeError`, não travar
silenciosamente). O protótipo original nunca teve essa prova.

## Testes de equivalência

`tests/test_gameoflife_from_geotiff.py` fecha o ciclo mosaico→TIFF→disco→halo
com um modelo simples: Game of Life carregado de um GeoTIFF (simulando
um mosaico já materializado) direto para `MemmapRasterWorkspace`,
incluindo um caso "grande" (2000×2000 = 4 milhões de células). Antes
deste teste, TIFF só era validado por round-trip (sem rodar nenhum
modelo em cima) e Game of Life só era testado com dado sintético em
RAM/disco — nunca os dois juntos.

`tests/test_equivalence.py` prova, usando `Environment`/`RasterBackend`
reais do dissmodel (não um harness isolado), que o resultado de
`GameOfLifeHalo` (blocos+halo) é idêntico célula a célula ao de
`GameOfLifeMono` (monolítico) após N passos de tempo, em 9 configurações
distintas (grade divisível exatamente, com resto, blocos de 1 linha,
bloco maior que a grade, e stress com 5 seeds aleatórias).

`tests/test_flood_model_equivalence.py` faz o mesmo com o `FloodModel`
**real e inalterado** do
[`brmangue-dissmodel`](https://github.com/DisSModel/brmangue-dissmodel),
via `HaloChunkedSyncRasterModel` (herança múltipla cooperativa — nenhuma
linha do `FloodModel` é modificada). Requer o extra opcional `brmangue`:

```bash
pip install -e ".[dev,brmangue]"
pytest tests/test_flood_model_equivalence.py -v
```

**Escopo deste teste:** verifica apenas que rodar em blocos produz o
mesmo resultado que rodar monoliticamente, usando dado sintético — não
valida a correção científica do `FloodModel` contra o golden TerraME
(há pendências de validação conhecidas, registradas separadamente e
fora do escopo deste pacote). Equivalência bloco-vs-monolítico e
correção científica são propriedades independentes.

`tests/test_disk_backend_equivalence.py` e
`tests/test_disk_flood_model_equivalence.py` fazem o mesmo teste, mas
via `MemmapRasterWorkspace`/`DiskChunkedSyncRasterModel` — a grade
nunca é materializada inteira em memória (leitura direta de blocos+halo
do disco, com recorte nas bordas em vez de padding). O segundo é o
teste mais forte do pacote: `FloodModel` real rodando inteiramente por
disco, com sincronização "_past" bloco a bloco, resultado idêntico ao
monolítico.

`tests/test_mangrove_model_equivalence.py` e
`tests/test_disk_mangrove_model_equivalence.py` fazem o mesmo com
`MangroveModel` (RAM e disco respectivamente) — incluindo um caso com
`acrecao_ativa=True` (exercita o ramo que lê `alt_past` e escreve
`alt`, não coberto pelos casos default). Foi nesse teste que o bug de
`boundary_value` documentado acima foi encontrado.

## Camada de disco (grades maiores que a RAM)

`disk_backend.py` (`MemmapRasterWorkspace`) e `disk_sync_model.py`
(`DiskChunkedSyncRasterModel`) generalizam a mesma decomposição de
domínio para grades que não cabem em memória, usando `np.memmap` com
double-buffer e checkpoint. **Sem dependência de dissmodel** no
`disk_backend.py` — reutilizável por qualquer framework.

Extraído e generalizado de um protótipo de aluno (`chunked_engine.py`,
pipeline de pré-processamento BR-MANGUE) que tinha memmap+double-buffer
+checkpoint corretos, mas amarrados a nomes de estado específicos do
domínio. Aqui os arrays são nomeados genericamente (dict `nome->dtype`),
sem nenhum acoplamento a "papel"/"uso"/"alt" ou a qualquer domínio.

```python
from haloexec import DiskChunkedSyncRasterModel, MemmapRasterWorkspace, workspace_arrays_for_sync_model

class FloodModelDiskHalo(DiskChunkedSyncRasterModel, FloodModel):
    pass

arrays = workspace_arrays_for_sync_model(
    base={"uso": np.int16, "alt": np.float32},
    land_use_types=["uso", "alt"],
)
ws = MemmapRasterWorkspace.create(root="/dados/workspace", shape=(50000, 50000),
                                   arrays=arrays, block_h=512, block_w=512, halo=1)
ws.fill("uso", uso_inicial)
ws.fill("alt", alt_inicial)

env = Environment(start_time=1, end_time=100)
FloodModelDiskHalo(workspace=ws, taxa_elevacao=0.05)
env.run()
```

**Ponto técnico central:** `SyncRasterModel.synchronize()` (que gera os
arrays `"<nome>_past"`) faz `.copy()` do array inteiro — se aplicado
ingenuamente sobre um memmap, materializaria a grade inteira em RAM.
`DiskChunkedSyncRasterModel` replica essa lógica localmente (não
importa nem modifica o `dissmodel` instalado), copiando bloco a bloco.
Isso é o trecho a reconciliar quando este pacote migrar para o core.

## Achado: ordem de eixos em Zarr não é garantida (y, x)

Ao investigar a integração com `disscube` de verdade (não só sintética),
achei que `CubeClient.load()`/`to_lucc_data()` fazem
`.transpose("y", "x")` **defensivamente** antes de usar qualquer array
— evidência de que a ordem de eixos gravada em disco não é garantida.
Reproduzi com `xarray` real, gravando exatamente como
`VariableWriter` do disscube grava (`da.to_dataset(...).to_zarr(...)`):
um array **quadrado** com eixos `(x, y)` em vez de `(y, x)` tem o
**mesmo shape** nos dois casos — a checagem de shape do `zarr_io.py`
não detectava a inversão. Sem correção, isso corrompia linha/coluna
**silenciosamente**, sem erro nenhum.

**Fix:** Zarr v3 grava a ordem real de eixos num campo nativo do
formato (`arr.metadata.dimension_names`, não `attrs` — diferente da
convenção antiga do Zarr v2/xarray, `_ARRAY_DIMENSIONS`).
`load_zarr_into_workspace` agora lê esse metadado e normaliza a leitura
de cada bloco pra `(y, x)`/`(time, y, x)` independente da ordem física
em disco. Testado com os três casos reais (`y,x` correto, `x,y`
invertido em array quadrado, `x,y,time` invertido com dimensão
temporal) em `tests/test_zarr_axis_order_regression.py`, usando
`xarray` de verdade pra gravar — não `zarr` puro — pra reproduzir
exatamente o que o `disscube` produz.

## Testando com arquivos grandes

`scripts/generate_and_benchmark.py` gera dados sintéticos **direto em
disco, bloco a bloco** (nunca materializa a grade inteira em RAM para
gerá-los — RNG determinística por posição de bloco, portanto
reprodutível) e roda Game of Life via `MemmapRasterWorkspace`, medindo
tempo e footprint de memória real.

```bash
python scripts/generate_and_benchmark.py \
    --shape 40000 40000 --block 512 512 --halo 1 \
    --generations 3 --density 0.35 --seed 42 \
    --root /tmp/haloexec_bench
```

Opções: `--shape ALTURA LARGURA`, `--block BLOCO_H BLOCO_W`, `--halo`,
`--generations`, `--density` (fração de células vivas iniciais),
`--seed`, `--root` (diretório do workspace), `--keep` (não apagar o
workspace ao final, para inspecionar os arquivos gerados).

**Métrica reportada e por quê:** o script separa `RssAnon` (heap real
alocado pelo processo — a métrica que prova se a grade foi ou não
materializada) de `RssFile` (cache de páginas do `mmap` tocadas,
reclamável pelo kernel, que cresce com o volume acumulado tocado mas
não representa memória "retida"). Usar só `ru_maxrss`/`VmRSS`
(RssAnon+RssFile somados) é enganoso para workflows baseados em
`mmap` — ele cresce mesmo quando o processo nunca reteve a grade
inteira de uma vez.

Resultado de referência medido em ambiente com ~3.9GB de RAM: grade de
40000×40000 (1,6 bilhão de células, ~1,5GB por array — maior que 40%
da RAM total do container) rodou com `RssAnon` em ~130MB e delta de
~4MB desde o início, confirmando que o footprint do processo não
escala com o tamanho da grade.

Para testar outro modelo em vez de Game of Life, troque a função
`_game_of_life_rule` do script por qualquer regra
`dict[str, np.ndarray] -> dict[str, np.ndarray]` — a mecânica de
geração/benchmark não muda.

## Caminho de migração para dissmodel core

Quando a revisão JOSS estabilizar, `HaloChunkedRasterCellularAutomaton`
deve migrar para dentro de `dissmodel.geo.raster` (ou módulo
equivalente), mantendo a mesma API pública. Como o pacote já depende
de `dissmodel` e reusa suas classes reais (não uma reimplementação),
a migração é literalmente mover a pasta — sem reescrita de lógica.
`tests/test_equivalence.py` serve como suíte de regressão para
confirmar isso.

Aplicação prática esperada: `brmangue-dissmodel` troca
`class FloodModel(SyncRasterModel)` por
`class FloodModel(HaloChunkedSyncRasterModel, SyncRasterModel)`
(ou compõe `FloodModelHalo` como feito no teste), elimina o
`chunked_engine.py` próprio, e ganha chunking com halo validado sem
reescrever a lógica hidrológica.

`dissmodel-abm` **não** usa este motor — agentes móveis exigem halo
dinâmico (transferência de agentes entre blocos), problema diferente
não coberto aqui.

Nenhum patch deste pacote deve ser aplicado ao `dissmodel` core
enquanto este estiver sob revisão.
