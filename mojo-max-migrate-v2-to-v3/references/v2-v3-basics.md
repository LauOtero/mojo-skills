# Esenciales de ModuleV2 vs ModuleV3

Cómo se ven un modelo, una capa y una capa distribuida en la API de grafo V2 (`max.nn`, `TensorValue`, `Graph` explícito) y en ModuleV3 (`max.experimental.nn`, `Tensor`, `F.lazy()` + `compile()`).

## Construir y ejecutar un modelo

### ModuleV2

```python
nn_model = DeepseekV3(config)
nn_model.load_state_dict(state_dict)

with Graph("deepseekV3", input_types=input_types) as g:
    outputs = nn_model(*g.inputs)
    g.output(*outputs)

session.load(g, weights_registry=state_dict)
```

### ModuleV3

```python
with F.lazy():
    nn_model = DeepseekV3(config)

nn_model.compile(*input_types, weights=state_dict)
```

Los tipos de entrada son obligatorios en ambas APIs, y la compilación congela el grafo (`session.load()` o `compile()`). `compile(weights=...)` lanza para un parámetro sin entrada en el state dict (`KeyError`) o con forma o dtype desajustados (`ValueError`). Una entrada del state dict que no coincide con ningún parámetro se ignora sin error.

Las diferencias:

- `F.lazy()` envuelve la construcción del módulo para que MAX registre los tensores de pesos de forma simbólica. Los pesos del checkpoint pasados a `compile()` los reemplazan, así que asignar tensores reales de antemano desperdiciaría memoria.
- `compile(weights=...)` es también el paso de carga, así que pasa el state dict ahí. Elimina la llamada V2 `load_state_dict()` antes de construir el grafo.
- Un desajuste de dtype entre un parámetro y su tensor cargado lanza por defecto. Pasa `auto_cast=True` (o `auto_cast_weights_from_env()` de `max.pipelines.weights.weight_loading`) para permitir casteos de dtype seguros.

## Una capa, lado a lado

### Linear (ModuleV2)

```python
class Linear(Module, Shardable):
    def __init__(self, in_dim, out_dim, dtype, device):
        self.weight = Weight(
            name="weight",
            dtype=dtype,
            shape=(out_dim, in_dim),
            device=device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return x @ self.weight.T
```

### Linear (ModuleV3)

```python
from max.experimental import random
from max.experimental.nn import Module
from max.experimental.tensor import Tensor


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        self.weight = random.normal([out_dim, in_dim])

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T
```

- ModuleV3 usa contextos de `device` y `dtype` por defecto. Mueve un módulo a un dispositivo con `module.to(device)` después de la construcción, o llama a `.to()` directamente sobre el tensor. `Module.to()` rechaza una malla multi-dispositivo; consulta la sección distribuida de abajo.
- No hay clase `Weight`. Los pesos son objetos `Tensor`, creados con `Tensor.zeros()`, `max.experimental.random.*` y fábricas similares.
- `__call__()` pasa a `forward()`.
- Los argumentos del constructor son solo por palabra clave donde V2 aceptaba posicionales: `Embedding(vocab_size, dim=...)` toma su dimensión por palabra clave, y `Linear(in_dim, out_dim)` crea un sesgo por defecto. Pasa `bias=False` cuando el checkpoint no traiga tensores de sesgo, o `compile(weights=...)` fallará su comprobación estricta de claves.

## Una capa distribuida, lado a lado

### Linear distribuida (ModuleV2)

```python
class Linear(Module, Shardable):
    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.weight.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.weight.sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Linear]: ...


linear = Linear(...)
linear.sharding_strategy = ShardingStrategy.rowwise(len(devices))
linear_shards = linear.shard(devices)
```

### Linear distribuida (ModuleV3)

```python
from max.experimental.nn.common_layers.linear import (
    col_parallel,
    row_parallel,
)
from max.experimental.sharding import DeviceMesh

device_mesh = DeviceMesh(devices=(...), mesh_shape=(n,), axis_names=("tp",))
linear = Linear(...)
linear = row_parallel(linear)
```

`col_parallel()` y `row_parallel()` establecen un `NamedMapping` en el peso de la capa: sobre qué eje de la malla se shardea cada dimensión. El modelo asigna la malla antes de la compilación (`kimik2_5_modulev3/model.py` en `max/pipelines/architectures/`):

```python
llm.mesh = DeviceMesh(tuple(self.devices), (n_devices,), (axis_name,))
```

Luego el modelo se construye dentro de `default_device(mesh)`, de modo que cada peso se crea en la malla con su colocación:

```python
with F.lazy(), default_dtype(dtype), default_device(llm.mesh):
    language_nn = KimiK2_5MoEDecoder(llm, kv_params, ep_batch_manager)
```

- La lista `devices` se convierte en un `DeviceMesh`, y un nombre de eje `"tp"` es un requisito previo para los helpers paralelos.
- `ShardingStrategy` más `.shard()` (que devuelve una instancia de módulo por dispositivo) pasa a ser un `Mapping` de colocación en el tensor de pesos. El solver de sharding redistribuye en tiempo de dispatch, así que no hay lista de módulos por dispositivo que iterar.

## Uso eager y compilado

Un módulo ModuleV3 debería funcionar tanto eager como compilado. Toma `Tensor` donde V2 tomaba `TensorValue`, y sus ops vienen de `max.experimental.functional` (`F.gather()`, no `ops.gather()`).

## Imports

### Imports buenos

```python
from max.dtype import DType

from max.experimental import functional as F
from max.experimental.tensor import Tensor

from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Sharded,
)

from max.experimental.nn import Module
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.common_layers.functional_kernels import ...
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
```

La caché KV se reparte entre paquetes: `PagedCacheValues` viene de `max.experimental.nn.common_layers.kv_cache`, mientras que `KVCacheParams`, `KVCacheParamInterface` y `MultiKVCacheParams` se quedan en `max.nn.kv_cache`. `max.nn` también exporta un `PagedCacheValues` propio, el mismo nombre resolviendo al tipo de entrada V2, que carece de `from_upstream()`. La ruta `max.experimental` lleva el tipo V3.

### Revisa estos imports

```python
from max.graph import DeviceRef, TensorValue, ops, ...
from max.nn.layer import LayerList, Module
from max.nn.embedding import VocabParallelEmbedding
```

Cualquier archivo que importe capas de `max.nn` (no `max.experimental.nn`) necesita una segunda mirada.

### Algunos imports de max.nn están bien

`max.nn` está bien cuando la clase o método importado no usa `TensorValue`:

```python
from max.nn.comm.ep import EPCommBuffers
from max.nn.rotary_embedding import DeepseekYarnRopeScalingParams
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
```

## Cuando no hay equivalente `Tensor`

- Cualquier op de grafo o kernel: envuélvelo con `F.functional()`. Convierte una función sobre `TensorValue` en una sobre `Tensor`, y ejecuta la función en el shard de cada dispositivo para una entrada distribuida:

  ```python
  from max.experimental import functional as F
  from max.graph import TensorValue, ops


  def _last_tokens(h: TensorValue, offsets: TensorValue) -> TensorValue:
      return ops.gather(h, offsets[1:] - 1, axis=0)


  last_tokens = F.functional(_last_tokens)
  ```

- Una op faltante usada en varias arquitecturas va a `max/experimental/nn/common_layers/functional_kernels.py` en el paquete `max` instalado. En el repositorio fuente de MAX, el archivo vive en `max/python/max/experimental/nn/common_layers/functional_kernels.py`.
- Una capa faltante recibe una versión ModuleV3, añadida a `common_layers` o al `<arch>/layers/<layer_name>.py` de la propia arquitectura.
- Otros componentes NN: haz que la clase existente maneje tanto `Tensor` como `TensorValue` (`EPBatchManager` es el ejemplo), o escribe un wrapper `Tensor`. Envolviendo `split_batch_replicated()`:

  ```python
  from max.experimental.sharding import DeviceMapping
  from max.experimental.tensor import Tensor
  from max.graph import DeviceRef, TensorValue
  from max.nn.data_parallelism import split_batch_replicated


  def split_replicated_batch(
      h: Tensor,
      input_row_offsets: Tensor,
      input_row_offsets_i64: Tensor,
      data_parallel_splits: Tensor,
      mapping: DeviceMapping,
  ) -> tuple[Tensor, Tensor]:
      """Divide un batch replicado en shards de paralelismo de datos."""
      devices = mapping.mesh.devices

      h_shards, offsets_shards = split_batch_replicated(
          [DeviceRef.from_device(d) for d in devices],
          [TensorValue(shard) for shard in h.local_shards],
          [TensorValue(shard) for shard in input_row_offsets.local_shards],
          TensorValue(input_row_offsets_i64),
          TensorValue(data_parallel_splits),
      )
      return (
          Tensor.from_shard_values(h_shards, mapping),
          Tensor.from_shard_values(offsets_shards, mapping),
      )
  ```
