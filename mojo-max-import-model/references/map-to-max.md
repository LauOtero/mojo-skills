# Elegir una arquitectura MAX de partida

Estás respondiendo: ¿qué arquitectura ModuleV3 es la más cercana a la mía? La más cercana se convierte en el punto de partida cuando generas el scaffold. Implementas los deltas mientras implementas el módulo.

## Lista lo que hay disponible

No mantengas una lista estática de slugs aquí, porque se desvía en cada release de MAX. Lista las arquitecturas ModuleV3 que incluye tu MAX instalado:

```bash
pixi run python scripts/list_native_archs.py --donors
```

Ese script, `list_native_archs.py` en el directorio `scripts/` de esta skill, imprime el slug y el nombre registrado de cada arquitectura ModuleV3 de los árboles de archs en disco. Elige un slug donante de la salida, luego abre `max/pipelines/architectures/<slug>/` en tu paquete MAX instalado.

## Tabla de decisión

Mapea los hallazgos del `config.json` a un arch de partida:

| Señal HF                                       | Arch de partida                                                         |
|-------------------------------------------------|-----------------------------------------------------------------------|
| `LlamaForCausalLM` (o compatible)              | `llama3_modulev3`                                                     |
| `Qwen2ForCausalLM`, `Qwen3ForCausalLM`          | `llama3_modulev3`, con QK-norm de `gemma3_modulev3`                |
| `layer_types` mezclando atención deslizante y completa | `olmo3` o `gemma3_modulev3`                                          |
| `Gemma3ForCausalLM`                             | `gemma3_modulev3`                                                     |
| `Phi3ForCausalLM` (`qkv_proj` fusionado)            | `phi3_modulev3`                                                       |
| `GraniteForCausalLM` (escalares MuP)            | `granite_modulev3`                                                    |
| Cualquier `*MoeForCausalLM` (expertos dispersos, top-k)   | `gpt_oss_modulev3`                                                    |
| `DeepseekV2ForCausalLM`, MLA en una GPU         | `deepseekV2_modulev3`                                                 |
| `DeepseekV3ForCausalLM`, MLA + MoE en varias GPUs  | `deepseekV3_modulev3`                                                 |
| Atención híbrida y capas Mamba               | `nemotron_h_modulev3`                                                 |
| Vision-language (imagen + texto)                  | `gemma3multimodal_modulev3`, `idefics3_modulev3`, `kimik2_5_modulev3` |
| `*ForMaskedLM` / encoder de embeddings de frases     | `mpnet_modulev3`, `qwen3_embedding_modulev3`                          |

`gemma3_modulev3` pone QK-norm por head (el diseño de Qwen3). `olmo3` la pone en todo el ancho de la proyección. Comprueba cuál hace HF.

`phi3_modulev3` rota la dimensión completa del head. Ningún donante ModuleV3 de un solo módulo implementa RoPE parcial, así que un `partial_rotary_factor` menor que 1 es un delta que implementas. `ProportionalRotaryEmbedding` en `gemma4_modulev3/layers/rotary_embedding.py` es un ejemplo ModuleV3 para copiar.

Los donantes vision-language construyen varios módulos compilados (torre de visión y modelo de lenguaje). Genera su scaffold con `scaffold.py --full-copy`, que copia cada archivo del donante y lo renombra. El esqueleto de subclase por defecto cubre donantes de un solo módulo.

## Cuando nada encaja

Si tu config tiene múltiples señales poco comunes (por ejemplo, MLA *y* un esquema de routing personalizado, o recurrencia con memoria no estándar), ninguna plantilla coincidirá. Elige una de estas rutas:

- Elige el decodificador más cercano y escribe las piezas únicas desde cero como subclases de `Module` construidas con `max.experimental.nn` y `max.experimental.functional`. Acepta que la comprobación de paridad de la etapa de scaffold fallará hasta que reemplaces el módulo divergente.
- Ver [recognize-walls.md](recognize-walls.md): algunas arquitecturas no son portables con la superficie pública de MAX hoy.

## Lee el arch elegido

Una vez que hayas elegido, lee su código en tu paquete MAX instalado. Empieza por `max/pipelines/architectures/<chosen_slug>/model.py`: su `_instantiate_module()` nombra el módulo raíz, y el archivo que define ese módulo es el siguiente que debes leer (`olmo3/olmo3.py`, `llama3_modulev3/llama3.py`).

Qué estás buscando:

- La clase del modelo de pipeline en `model.py`: subclasifica `ModuleV3PipelineModelWithKVCache` de `max.pipelines.lib`, directamente o a través del modelo de otra arquitectura (`phi3_modulev3` reutiliza el de `llama3_modulev3`).
- El módulo raíz: un `Module` cuyo `forward()` desaplana las entradas de la caché KV y llama a un modelo de texto, a menudo almacenado como `self.language_model`, que establece el prefijo de nombres de pesos.
- La clase del bloque: cada arquitectura define su propio bloque como un `Module` (por ejemplo, `olmo3/layers/transformer.py`).
- La clase de atención: un `Module` sobre `Tensor` y `PagedCacheValues` (por ejemplo, `olmo3/layers/attention.py`), compuesto de `max.experimental.nn.common_layers` y los kernels fusionados en `common_layers.functional_kernels`.
- La clase del MLP: `MLP` de `max.experimental.nn.common_layers.mlp`, o un `Module` construido con `max.experimental.nn.Linear` y las activaciones en `max.experimental.functional`.

Estas clases muestran qué puedes subclasificar cuando necesitas cambiar un método. Una subclase que sobrescribe un método mantiene tu port pequeño.

## Salida de la comparación

Una nota corta para ti:

- **Arch de partida:** `<slug>`
- **Qué reutilizaré sin cambios:** (probablemente el embedding, la RMSNorm final, el LM head)
- **Qué necesito subclasificar:** (¿atención? ¿MLP? ¿bloque?)
- **Qué necesito añadir:** (normas extra, routing MoE, head multi-paso)

Esta nota se convierte en la lista de edición cuando implementas el módulo.

