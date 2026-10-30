---
name: mojo-max-migrate-v2-to-v3
description: >
  Úsala al migrar un modelo MAX de la API de grafo V2 (max.nn, TensorValue,
  construcción explícita de Graph) a ModuleV3 (max.experimental.nn, Tensor,
  F.lazy() + compile()). Se activa con: "migrar a ModuleV3", "portar un modelo
  max.nn a max.experimental", "convertir TensorValue a Tensor", "actualizar mi
  modelo a la API V3", "migración ModuleV3". Flujo: lee el mapa de conceptos,
  reescribe imports y firmas de __call__, y sustituye la construcción de grafos
  por compile(weights=...). Alinea los nombres de pesos con la nueva jerarquía
  de módulos y mueve el unflatten de la caché KV a forward(). Porta el sharding
  a un DeviceMesh cuando el modelo V2 esté shardeado, y luego verifica que las
  salidas greedy coinciden con la implementación V2.
compatibility: Requiere una instalación de MAX (pip o pixi), el código fuente del modelo V2 a migrar, y una GPU para verificar las salidas.
metadata:
  argument-hint: "[ruta a tu código V2, por ejemplo 'models/my_model.py']"
---

# Migra un modelo MAX de V2 a ModuleV3

**Entrada**: el código fuente del modelo V2 a migrar (`$ARGUMENTS`).

ModuleV3 (`max.experimental.nn`) es la API actual de modelos de MAX. Un modelo V2 construye un `Graph` explícito sobre valores `TensorValue` con capas de `max.nn`. Un modelo ModuleV3 declara un `forward()` sobre valores `Tensor` y compila con `F.lazy()` + `compile()`. Lee primero [references/v2-v3-basics.md](references/v2-v3-basics.md). Cubre `F.lazy()`, `compile(weights=...)`, el manejo de dtype con `auto_cast`, `forward()` vs `__call__()`, el mapeo distribuido y las reglas de importación.

> **⚠️ CAMBIOS DE v26.2/v26.3 QUE AFECTAN A LA MIGRACIÓN**
>
> - **v26.2**: `max.nn.module_v3` se renombró a **`max.experimental.nn`** (sin el sufijo `v3`). Las importaciones como `from max.nn.module_v3 import Module` deben pasar a `from max.experimental.nn import Module`.
> - **v26.2**: `max.tensor`, `max.functional` y `max.random` se movieron de nuevo bajo `max.experimental` (es decir, `max.experimental.tensor`, `max.experimental.functional`, `max.experimental.random`).
> - **v26.3**: se añadió `max.experimental.sharding` con `DeviceMesh`, y `Module.compile()` puede distribuir un modelo a través de una malla de dispositivos.
> - **v26.3**: el keyword `fn` emite una advertencia de deprecación y se convertirá en un error grave en la próxima versión.
>
> **Regla**: no uses `fn`; usa `def` y `comptime`. Si tu código V2 aún lo usa, la migración es el momento de eliminarlo.

## Flujo de trabajo

### 1. Confirma que el modelo es V2

Clasifica por imports, y lee el archivo cuando grep y los imports discrepen. El código V2 importa sus capas desde `max.nn` y construye un `Graph` explícito sobre `TensorValue`. El código ModuleV3 importa `max.experimental.nn` y hereda de su `Module`. Greps acotados aceleran la lectura:

```bash
grep -rn "max.experimental" <model-path>
grep -rEn "^\s*(from|import) max\.nn" <model-path>
```

Los aciertos del primero significan que el modelo ya es (al menos en parte) V3 y necesita revisión, no migración. Los aciertos del segundo significan V2. Los imports de `max.graph` por sí solos no deciden: el código V3 mantiene `TensorType` y `DeviceRef` de `max.graph` para las especificaciones de entrada. Un archivo que acierta en ambos es mixto. Léelo antes de migrar nada.

### 2. Conserva el código V2, añade la variante V3

Deja la implementación V2 intacta para que siga funcionando mientras se construye la variante V3. Coloca el código V3 junto a ella: un directorio `<name>_modulev3` cuando migres una arquitectura registrada, o un nuevo archivo de módulo cuando el modelo viva en tu propio paquete. Al migrar una arquitectura registrada, nombra la variante V3 con el sufijo `_ModuleV3`. `max serve --prefer-module-v3` la selecciona mientras el nombre V2 sigue sirviendo por defecto.

### 3. Reescribe los imports

```python
# ELIMINA:
from max.graph import Graph, TensorValue, BufferValue, ops
from max.nn.layer import Module, LayerList
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm

# AÑADE:
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList

# MANTÉN (sin TensorValue, compartido con V2):
from max.nn.kv_cache import KVCacheParams, KVCacheParamInterface
from max.nn.attention import MHAMaskVariant
```

- `TensorType` y `DeviceRef` se mantienen de `max.graph`. Definen especificaciones de entrada, no valores de grafo.
- Los imports de `max.nn` están bien cuando la clase importada no usa `TensorValue`.
- `PagedCacheValues` viene de `max.experimental.nn.common_layers.kv_cache` para el tipo V3. `max.nn` exporta un `PagedCacheValues` propio, un alias del tipo de entrada V2 sin `from_upstream()`. Importa desde la ruta `max.experimental` para obtener el tipo V3.

### 4. Reescribe la superficie del módulo

- `__call__()` pasa a ser `forward()`.
- `TensorValue` y `BufferValue` pasan a ser `Tensor`.
- `Module` toma la firma de llamada:
  `class MyModel(Module[[Tensor, ...], tuple[Tensor, ...]])`. Prefiere tipos específicos, y usa `...` solo cuando `forward()` tenga `*args`.
- Los constructores de capas toman argumentos por palabra clave y eliminan `dtype=` y `device=`. V3 usa contextos de dtype y dispositivo por defecto, y `model.to(device)` mueve el módulo a un dispositivo después de la construcción.
- `LayerList` pasa a ser `ModuleList` de `max.experimental.nn.sequential`. `ModuleList` hereda de `list`, así que toma un único iterable: `ModuleList(layers)`. `ModuleList(*layers)` lanza `TypeError`.
- Los parámetros son objetos `Tensor` planos. Créalos con `Tensor.zeros()`, `max.experimental.random.*` y fábricas similares.

### 5. Reescribe las operaciones

`ops.*` pasa a `F.*` (`ops.gather()` → `F.gather()`). La mayoría de las ops de forma son métodos de `Tensor`: `reshape()`, `transpose()`, `permute()`, `squeeze()`, `unsqueeze()`, `split()`, `cast()`. `flatten()` es la excepción. Es solo funcional (`F.flatten(tensor, start_dim)`).

### 6. Sustituye la construcción del grafo por `compile()`

V2:

```python
nn_model = MyModel(config)
nn_model.load_state_dict(state_dict)

with Graph("my_model", input_types=input_types) as graph:
    outputs = nn_model(*graph.inputs)
    graph.output(*outputs)

model = session.load(graph, weights_registry=state_dict)
```

V3:

```python
with F.lazy():
    nn_model = MyModel(config)
    nn_model.to(device)

model = nn_model.compile(*input_types, weights=state_dict)
```

- `F.lazy()` envuelve la construcción y registra los tensores de pesos de forma simbólica. El checkpoint pasado a `compile(weights=...)` los reemplaza.
- Cada parámetro necesita una entrada en el state dict con forma y dtype coincidentes: una faltante lanza `KeyError` y una desajustada `ValueError`. Una entrada que no coincide con ningún parámetro se ignora sin error, así que compara las claves del state dict contra `parameters` después de migrar (paso 7).
- Un desajuste de dtype entre un parámetro y su tensor cargado lanza por defecto. Pasa `auto_cast=True` para permitir casteos de dtype seguros.
- Llama al modelo compilado directamente (`model(tokens, ...)`). Las salidas compiladas son objetos `Tensor`. El código que consumía salidas `Buffer` lee el buffer a través de `.driver_tensor`.

### 7. Alinea los nombres de pesos con la jerarquía compilada

`compile(weights=...)` resuelve los nombres desde la jerarquía de atributos del módulo raíz compilado. Cuando la migración introduce o renombra atributos wrapper, los mapeos de claves del checkpoint deben seguirlos: un módulo raíz que almacena `self.language_model = ...` necesita que cada peso interno lleve el prefijo `language_model.`. Después de migrar, comprueba que cada clave del state dict sea una ruta de atributo válida desde el módulo raíz.

Los embeddings atados necesitan un manejo explícito: crea condicionalmente `lm_head`, y multiplica por `embed_tokens.weight.T` en `forward()` cuando `tie_word_embeddings` esté activado. Mantén una copia del peso compartido en el state dict.

### 8. Mueve el unflatten de la caché KV a `forward()`

Los tipos de parámetros de la caché KV se quedan en `max.nn`. El tipo de valores tiene un hogar V3 en `max.experimental.nn.common_layers.kv_cache`. Usa los métodos de `kv_params` más `PagedCacheValues.from_upstream()`:

```python
def forward(self, tokens: Tensor, ..., *variadic_args: Tensor):
    kv_inputs = iter(arg._graph_value for arg in variadic_args)
    symbolic_inputs = self.kv_params.unflatten_kv_inputs(kv_inputs)
    kv = PagedCacheValues.from_upstream(symbolic_inputs, tokens.mapping)
```

Necesitas la extracción `._graph_value`: `variadic_args` llegan como objetos `Tensor` mientras los métodos de unflatten esperan valores de grafo crudos. Para atención deslizante más global, declara `MultiKVCacheParams` y usa `unflatten_basic_kv_tree()`, envolviendo cada árbol con su propio `PagedCacheValues.from_upstream()`.

### 9. Porta el sharding cuando el modelo V2 esté shardeado

V2 configura un `ShardingStrategy` en cada capa y llama a `.shard(devices)` para obtener una lista de módulos por dispositivo. V3 coloca un `DeviceMesh` en el modelo y un mapeo de colocación en cada peso:

```python
from max.experimental.nn.common_layers.linear import row_parallel
from max.experimental.sharding import DeviceMesh

model.mesh = DeviceMesh(devices, (n_devices,), ("tp",))
linear = row_parallel(Linear(in_dim, out_dim))
```

`col_parallel()` y `row_parallel()` establecen sobre qué eje de la malla se shardea cada dimensión del peso, y V3 no tiene lista de módulos por dispositivo. La maquinaria distribuida de V2 se mapea a V3 de la siguiente manera:

- **`Allreduce` y `Signals`**: el solver de sharding lee las colocaciones de los tensores e inserta el colectivo que cada op necesita. Por ejemplo, sumar una salida row-parallel (una suma `Partial`) a un residual replicado la all-reduce. Llama a `F.allreduce_sum()`, `F.reduce_scatter()` o `F.allgather()` donde el modelo controle la reducción. `compile()` añade los buffers de señal que usan los colectivos como entradas del grafo y los proporciona en tiempo de ejecución.
- **`.shard()` y listas por dispositivo**: construye el modelo dentro de `default_device(mesh)`, y cada peso se crea en la malla con la colocación que declara. `Module.to()` mueve un módulo a un solo dispositivo y lanza para una malla multi-dispositivo. El código por dispositivo se ejecuta mediante `F.functional()`, o mediante `t.local_shards` y `Tensor.from_shard_values()`.
- **Paralelismo de expertos**: V3 reutiliza `EPBatchManager`. Su `comm_buffers()` devuelve los buffers de comunicación como valores `Tensor`, y la capa MoE pasa el shard de cada dispositivo a las llamadas de dispatch como `TensorValue`. El modelo de pipeline configura el manager en `_init_distributed_runtime()` (`deepseekV3_modulev3/model.py`).

`gemma3_modulev3/` (TP) y `deepseekV3_modulev3/` (TP, TP + EP, DP + EP) muestran cada patrón.

### 10. Verifica contra V2

Ejecuta ambas implementaciones sobre las mismas entradas y compara los tokens greedy (y los logits donde sea conveniente). Salidas idénticas confirman la migración. Ejecuta mypy sobre el módulo migrado. V3 se apoya en las firmas `Module[[...], ...]`, así que los errores de tipo apuntan a bugs de migración.

## Implementaciones de referencia

Las arquitecturas registradas se distribuyen como código Python legible en el paquete MAX instalado, bajo `max/pipelines/architectures/`. Las migraciones V3 que leer:

- V3 de un solo GPU: `olmo3/`, `gpt_oss_modulev3/`, `llama3_modulev3/`
- V3 shardeado: `kimik2_5_modulev3/`, `gemma3_modulev3/`, `deepseekV3_modulev3/`
- Atención híbrida con estado recurrente: `nemotron_h_modulev3/`
- V2 para comparación: `gpt_oss/`, `llama3/`

## Cuando no hay equivalente `Tensor`

Cada op de grafo y kernel es invocable desde V3: `F.functional(fn)` convierte una función sobre `TensorValue` en una sobre `Tensor`. Las opciones, y el patrón wrapper para puentear un helper `TensorValue` por dispositivo a una API `Tensor`, están en [references/v2-v3-basics.md](references/v2-v3-basics.md).

