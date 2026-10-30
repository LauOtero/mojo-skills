# Implementar el módulo

El scaffolding subclasifica una arquitectura donante (`llama3_modulev3`, `olmo3`, …). Ese esqueleto no es tu modelo. El módulo donante computa la matemática del donante hasta que editas cada subcapa que la lista de deltas marcó como diferente de Hugging Face.

No ejecutes `pixi run max serve`, comprobaciones de coherencia ni verificación de logits hasta que pasen los criterios de finalización de esta fase. Un donante sin modificar servido contra un checkpoint extranjero o rechaza los pesos en tiempo de compilación o ejecuta la matemática del donante sobre ellos. De cualquier forma, la verificación falla hasta que el módulo implemente los deltas.

---

## Anti-patrón (no hagas esto)

| Lo que los agentes se saltan                                     | Por qué falla                                                                                                                                                                                                                       |
|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Servir justo después de `scaffold.py`                      | La atención / bloque / MoE del donante sigue ejecutándose                                                                                                                                                                                           |
| Editar solo `arch.py` + `model_config.py`              | El módulo en `<slug>.py` sigue coincidiendo con el donante                                                                                                                                                                                  |
| Asumir que "compatible con Llama" significa que no hay que editar `<slug>.py` | La herencia de HF oculta bloques no-Llama (normas paralelas, MoE sigmoide, capas NoPE, …). Un delta que el donante ya expone como campo de config (`interleaved_rope_weights`) se establece en `from_donor()`; cada otro delta es una edición de módulo |
| Ejecutar `compare_layers.py` antes de que los pesos carguen limpiamente  | Perseguir logits cuando los tensores no están vinculados o están mal mapeados                                                                                                                                                                              |

La Fase 1 produce una **lista de deltas**. Implementar el módulo ejecuta esa lista en código MAX, una subcapa a la vez, contra el `forward()` de HF.

---

## La superficie de la API ModuleV3

Un port es un árbol de subclases de `Module`. Cada una declara su firma de llamada, construye sus parámetros y módulos hijos en `__init__()`, y computa en `forward()`:

```python
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.tensor import Tensor


class MyMLP(Module[[Tensor], Tensor]):
    def __init__(self, *, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

- **Imports.** Las capas vienen de `max.experimental.nn` (`Linear`, `Embedding`, `RMSNorm` en `max.experimental.nn.norm`, `ModuleList` en `max.experimental.nn.sequential`) y los submódulos `max.experimental.nn.common_layers` (`MLP` en `mlp`, embeddings rotatorios en `rotary_embedding`, `ColumnParallelLinear` y `RowParallelLinear` en `linear`, y los kernels de atención en `functional_kernels`). Las ops vienen de `max.experimental.functional`. `PagedCacheValues` viene de `max.experimental.nn.common_layers.kv_cache`. Los tipos de parámetros de caché KV (`KVCacheParams`, `MultiKVCacheParams`, `KVCacheParamInterface`) y `MHAMaskVariant` vienen de `max.nn.kv_cache` y `max.nn.attention`. Sostienen configuración, no tensores. `TensorType` y `DeviceRef` vienen de `max.graph` y describen entradas.
- **Constructores.** Los argumentos de capa son amigables con palabras clave y no llevan `dtype=` ni `device=`: el pipeline construye el módulo bajo un dtype por defecto, y `model.py` lo mueve con `.to(device)`, o lo construye dentro de `default_device(mesh)` para un port multi-GPU. `Linear` crea un sesgo por defecto, así que pasa `bias=False` cuando el checkpoint no traiga ninguno.
- **Parámetros.** Los parámetros son atributos `Tensor`. Sus nombres son las rutas de atributo desde el módulo raíz (`language_model.layers.0.mlp.gate_proj.weight`), y `compile(weights=...)` carga el checkpoint por esos nombres.
- **Ops.** La mayoría de las ops de forma son métodos de `Tensor` (`reshape()`, `transpose()`, `permute()`, `split()`, `cast()`). El resto son `F.*` (`F.gather()`, `F.concat()`, `F.flatten()`). `@` es matmul.
- **Listas de capas.** Usa `ModuleList(layers)`, que toma un iterable, así que los parámetros se registran bajo `layers.<i>.`.
- **Caché KV.** El `forward()` del módulo raíz recibe las entradas de caché KV aplanadas como `*variadic_args` finales. Desaplánalas ahí con `self.kv_params.unflatten_kv_inputs(...)` (o `unflatten_basic_kv_tree()` para cachés deslizantes más globales) y envuelve cada resultado con `PagedCacheValues.from_upstream(...)`. `olmo3/olmo3.py` muestra ambos pasos.
- **Compilación.** `ModuleV3PipelineModelWithKVCache.load_model()` construye el módulo bajo `F.lazy()`, así que los parámetros se registran simbólicamente, y luego llama a `compile(*input_types, weights=state_dict)`. `model.py` solo implementa `_create_model_config()` y `_instantiate_module()`.

### Ops y kernels sin versión `Tensor`

Cada op y kernel a nivel de grafo es invocable desde un `forward()` de ModuleV3. Envuelve una función sobre `TensorValue` con `F.functional()` para obtener una función sobre `Tensor`:

```python
from max.experimental import functional as F
from max.graph import TensorValue, ops


def _last_tokens(h: TensorValue, offsets: TensorValue) -> TensorValue:
    return ops.gather(h, offsets[1:] - 1, axis=0)


last_tokens = F.functional(_last_tokens)
```

En una malla multi-GPU, la función envuelta se ejecuta en el shard de cada dispositivo. Para llamar directamente a un helper por dispositivo, toma los shards con `[TensorValue(s) for s in t.local_shards]` y reensambla los resultados con `Tensor.from_shard_values(values, mapping)`. Para un ejemplo, ver `split_replicated_batch()` en `deepseekV3_modulev3/deepseekV3.py`. Si MAX no tiene un `Module` para una capa, escribe uno en el directorio `layers/` de tu port, construido con estas piezas.

---

## Orden de trabajo

Implementa en este orden, porque cada capa depende de que el cableado anterior sea correcto:

1. **`model_config.py`**: conecta cada clave de `config.json` (tabla de config de la Fase 1).
2. **`list_checkpoint_keys.py`**: metadatos de safetensors del Hub (claves, formas, dtypes).
3. **`weight_adapters.py`**: nombres de safetensors de HF → nombres de parámetros del módulo raíz.
4. **Embedding + norma final + LM head** en `<slug>.py`.
5. **Un bloque decodificador**: haz bien el bloque 0 antes de clonar el patrón.
6. **Pila completa**: repite para todas las capas. Las capas condicionales (deslizante vs completa, MoE vs densa) necesitan lógica explícita por capa que coincida con HF.

Mantén el `modeling_<type>.py` de HF abierto al lado. Para cada módulo MAX que edites, traza la línea del `forward()` de HF que corresponde a cada op de MAX.

---

## Lista de verificación de componentes

Copia esta tabla de la lista de deltas de la Fase 1 y marca cada fila como **hecha** solo cuando MAX coincida con HF para ese componente (no cuando "compile").

| Componente          | Referencia HF                           | Archivo / clase MAX                | Hecho cuando                                                                                          |
|--------------------|----------------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------|
| Config / KV params | `*Config`                              | `model_config.py`               | Cada clave del Hub leída por la config del donante o `from_donor()`, y `construct_kv_params()` coincide con HF   |
| Mapa de pesos         | claves del checkpoint                        | `weight_adapters.py`            | Cada parámetro vinculado, y cada tensor del checkpoint consumido o descartado a propósito                  |
| Embedding          | `embed_tokens`                         | `<slug>.py`                     | Forma + dtype coinciden. Atado con LM head si la config lo dice                                            |
| Atención          | `*Attention.forward`                   | módulo de atención en `<slug>.py` | Diseño Q/K/V, RoPE, máscara, repetición GQA, softcap coinciden con HF                                             |
| MLP / MoE          | `*MLP.forward` / `*SparseMoeBlock`     | módulo MLP o MoE               | Gate/up/down o routing de expertos coincide con HF (el nombre de la activación importa)                                |
| Bloque decodificador      | `*DecoderLayer.forward`                | módulo de bloque                    | **El orden de normas y el cableado residual** coinciden con HF (pre-norm vs paralelo vs doble-norm)                    |
| Norma final         | `model.norm`                           | módulo de texto raíz                | Mismo tipo de norma (RMS vs LayerNorm) y epsilon                                                      |
| LM head            | `ForCausalLM.forward` → `lm_head(...)` | módulo de texto raíz en `<slug>.py` | Embed atado, divisor pre-head (`h / (hidden_size / dim_model_base)`), `logits_scaling` de MuP, softcap |

El orden de normas y el cableado del bloque son los bugs de "parece Llama pero no lo es" más comunes. Lee el `forward()` del bloque de HF antes de editar el bloque donante.

---

## `weight_adapters.py`

Objetivo: después de que corran los adaptadores, **cada parámetro del módulo raíz tiene un tensor de la forma y dtype correctos, y cada tensor del checkpoint aterriza en un parámetro**.

- Haz coincidir proyecciones fusionadas vs divididas (`qkv_proj` vs `q_proj` / `k_proj` / `v_proj` separados).
- Haz coincidir el diseño de claves de expertos MoE (`experts.N.gate_proj` vs tensores agrupados).
- Haz coincidir el prefijo wrapper del módulo raíz (`model.layers.` → `language_model.layers.` en `olmo3`).
- Elimina renombrados solo-donante que no apliquen a tu checkpoint.

`compile(weights=...)` lanza `KeyError` para un parámetro sin tensor y `ValueError` para un desajuste de forma o dtype. Un tensor del checkpoint cuyo nombre no coincide con ningún parámetro se omite sin error, así que ejecuta la auditoría en [state-dict-audit.md](state-dict-audit.md) antes de servir. Ver [rename-weights.md](rename-weights.md) para renombrados.

---

## `<slug>.py`: el módulo

Este archivo es el port. Subclasificar el donante está bien solo para métodos que son idénticos a HF. Cuando la lista de deltas marcó una diferencia:

- **Un método difiere**: subclasifica el módulo donante y sobrescribe ese método.
- **El cableado del bloque difiere**: reescribe el módulo del bloque. No heredes el `forward()` del donante si el orden de normas/residual difiere.
- **Nuevo patrón de atención** (MLA, ventana deslizante por índice de capa): un nuevo `Module` de atención construido con `max.experimental.nn.common_layers` y los kernels en `common_layers.functional_kernels`.
- **NoPE en algunas capas**: da a esas capas un módulo rotatorio con una tabla `freqs_cis` identidad ([divergences.md](divergences.md#17-nope--skip-rope-layers-via-identity-freqs_cis)).
- **Un submódulo que el `__init__` del donante construye** (el modelo de texto interno): llama a `super().__init__(...)` en el módulo raíz de tu port, luego asigna el reemplazo al mismo atributo. El módulo se construye bajo `F.lazy()`, así que el submódulo descartado del donante no asigna nada.

No copies y pegues el módulo donante y cambies el nombre de la clase. Camina por el `forward()` de HF e implementa lo que hace.

### Pilas recurrentes / de pesos compartidos: mezcla la inyección una vez

Algunos modelos iteran la misma pila Transformer varias veces (HRM, algunos encoder-decoders, "looped transformers"). Mezclan un "estado de inyección" separado (`z_H`, `z_L`, embedding de condición, estado de prefijo) en cada invocación de la pila. La inyección ocurre **una vez antes del primer bloque** de la pila, no en cada bloque. El `Stack.forward` de HF es el patrón canónico:

```python
# HF (correcto): inyecta una vez, luego ejecuta los bloques secuencialmente.
def forward(self, x, ...):
    x = x + injection  # mezcla una vez
    for layer in self.layers:
        x = layer(x, ...)
    return self.final_norm(x)
```

Un error común de port es desenrollar la pila en el bucle de nivel superior y escribir:

```python
# INCORRECTO: inyección añadida en cada bloque. mean_sq crece superlinealmente.
for cycle in range(N):
    for i, layer in enumerate(self.layers):
        z = layer(z + injection, ...)  # ← añadido en cada bloque
```

El bucle correcto:

```python
# CORRECTO: inyecta una vez por invocación de la pila, luego itera.
for cycle in range(N):
    z = z + injection
    for i, layer in enumerate(self.layers):
        z = layer(z, ...)
```

Ver ["Stack-vs-block residual"][stack-residual] en la referencia de trampas para la receta de diagnóstico completa. El síntoma es un flujo residual cuyo `mean_sq` crece superlinealmente con la profundidad en MAX, mientras que HF crece linealmente.

Guía de lectura detallada: [read-modeling-code.md](read-modeling-code.md).

---

## Criterios de finalización (obligatorios antes de servir)

Todo debe ser verdadero antes de `pixi run max serve` o cualquier script de verificación:

- [ ] Lista de deltas: cada fila implementada o explícitamente marcada N/A con cita de HF
- [ ] `model_config.py` conecta cada clave de la tabla de config de la Fase 1 / `inspect_hf.py`
- [ ] Metadatos del checkpoint listados (`list_checkpoint_keys.py`), y las formas coinciden con la config
- [ ] `arch.py`: `name=` es igual a `config.json` → `architectures[0]`, y `default_encoding` coincide con el dtype del Hub
- [ ] `weight_adapters.py`: cada parámetro vinculado, y la auditoría del state dict no reporta tensores del checkpoint no consumidos que necesites
- [ ] `<slug>.py`: los módulos de atención, MLP/MoE y bloque coinciden con el `forward()` de HF para el bloque 0 como mínimo, luego a profundidad completa
- [ ] **Forward de nivel superior**: has caminado por el `forward()` más externo de HF (no solo `DecoderLayer.forward`) y confirmado cómo se cablean los bloques entre sí. Para pilas recurrentes / de pesos compartidos: el estado de inyección se mezcla **una vez por invocación de la pila**, no por bloque (ver ["Pilas recurrentes / de pesos compartidos"](#pilas-recurrentes--de-pesos-compartidos-mezcla-la-inyección-una-vez) arriba)
- [ ] Puedes explicar, por delta, qué línea de código de HF refleja el cambio de MAX

Esta fase termina cuando el módulo implementa la matemática de HF. Un módulo que compila y arranca bajo `pixi run max serve` puede seguir computando la matemática incorrecta. La basura o los bucles `&&&&` después de una reescritura normalmente significan que un delta sigue mal (diseño de `freqs_cis` de NoPE, cableado del bloque, combinación MoE). Un delta faltante también puede dejar la salida fluida, así que el texto coherente no termina esta fase. Quédate en el bucle implementar / cazar divergencias hasta que los logits coincidan.

Si alguna casilla está sin marcar, sigues implementando, no verificando.

---

## Registro e imports

Copia los imports y el registro del donante del scaffold en tu paquete MAX instalado, `max/pipelines/architectures/<donor>/`. Para trampas de imports obsoletos y reglas de encoding/dispositivo, ver [pitfalls-config.md § Trampas de importación y API de configuración][import-traps].

[stack-residual]: pitfalls-graph.md#stack-vs-block-residual-in-recurrent--shared-weight-architectures
[import-traps]: pitfalls-config.md#import-and-config-api-traps

