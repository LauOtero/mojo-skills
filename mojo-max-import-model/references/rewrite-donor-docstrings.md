# Reescribir los docstrings del donante tras un scaffold-by-copy

Copiar un donante y renombrarlo no actualiza **comentarios ni docstrings**. Los nombres de clase se renombran. El texto que registra *lo que el archivo dice hacer* (docstrings de módulo, docstrings de clase, comentarios de código) no. Esto pasa en el modo `scaffold.py --full-copy` y en cualquier archivo donante que copies o subclases y edites a mano. Tras un renombrado, `my_large_moe.py` puede seguir abriendo con un docstring que describe DeepSeek-V3. La nueva clase puede afirmar "TP, TP + EP y DP + EP" cuando el port solo soporta uno de ellos.

El modo `scaffold.py` por defecto escribe docstrings nuevos para sus archivos generados. Esos docstrings describen el esqueleto, y se quedan obsoletos conforme implementas deltas. Ejecuta la auditoría de abajo en todos los modos.

Ningún paso del bucle de test comprueba este texto. ``max serve`` y ``compare_layers.py`` no leen docstrings. Un lector posterior que confíe en un docstring del donante obtiene información incorrecta sobre los modos y capas del port.

Esta referencia define:

1. Una forma positiva que todo docstring de módulo debería seguir.
2. Una comprobación mecánica para ejecutar antes de declarar terminado el scaffold + implementación.

## El docstring de módulo

Cada archivo del directorio de tu port abre con un docstring de módulo con esta forma:

1. **Qué es este archivo**, en términos de tu port, no del donante. ("``my_large_moe`` módulo de generación de texto para servido multi-GPU.")
2. **De qué deriva**, nombrado explícitamente. ("Refleja estructuralmente ``max.pipelines.architectures.deepseekV3_modulev3.deepseekV3`` para el esqueleto MoE shardeado por malla.")
3. **Deltas aplicados**, como lista de viñetas de solo lo que implementaste. ("Bloque decodificador paralelo. LayerNorm sin sesgo (no RMSNorm). RoPE intercalado. Router sigmoid+renorm. NoPE en capas de atención completa vía ``freqs_cis`` identidad.")

Nombra al donante como la fuente ("deriva de deepseekV3_modulev3"). No describas el port como el donante ("este es un modelo DeepSeek-V3"). Un renombrado cambia los nombres en el texto del donante y conserva sus afirmaciones, así que reescribe el docstring desde cero con esta forma.

## Ejemplo resuelto: docstring de módulo

### Texto del donante tras renombrar `deepseekV3_modulev3/deepseekV3.py`

```python
"""Construye un modelo DeepSeek-V3 que soporta single-GPU, TP y DP + EP."""
```

La clase en este archivo ahora es `MyLargeMoE`. No implementa DeepSeek-V3. Los pesos BF16 publicados no caben en una GPU, así que single-GPU no es un modo soportado. Ambas afirmaciones son falsas.

### Reescritura

```python
"""Módulo de generación de texto MyLargeMoE para servido multi-GPU.

Refleja estructuralmente ``max.pipelines.architectures.deepseekV3_modulev3``
para el esqueleto shardeado por malla (``DeviceMesh`` con un eje ``"tp"``,
proyecciones column- y row-parallel, ``EPBatchManager`` para los expertos
enrutados). El checkpoint BF16 necesita al menos cuatro GPUs.

Deltas sobre el esqueleto de deepseekV3_modulev3:

- **Bloque decodificador paralelo** con un único ``input_layernorm`` y
  ``x' = x + attn(norm(x)) + ffn(norm(x))`` (sin norma post-atención).
- **LayerNorm sin sesgo** (no RMSNorm). Lee ``layer_norm_eps`` de la config
  cuando falta ``rms_norm_eps``.
- **RoPE intercalado** con ``interleaved=True`` en el embedding rotatorio
  (verifica contra el ``rotate_half`` de HF, ver [divergences.md](divergences.md)).
- **NoPE en capas de atención completa**: ``freqs_cis`` identidad (cos=1,
  sin=0) en las capas donde HF omite la rotación (``config.layer_types[]``
  aquí).
- **Router sigmoid + renorm por suma** (softmax-then-top-k en el donante).
"""
```

## Ejemplo resuelto: docstring de clase

### Texto del donante tras renombrar

```python
class MyLargeMoE(Module[[Tensor, Tensor, Tensor], tuple[Tensor, ...]]):
    """Modelo unificado DeepSeek-V3 que soporta inferencia single-GPU, TP y DP + EP."""
```

El nombre, los modos de GPU y la familia del modelo están todos mal.

### Reescritura

```python
class MyLargeMoE(Module[[Tensor, Tensor, Tensor], tuple[Tensor, ...]]):
    """Módulo causal-LM MyLargeMoE para inferencia multi-GPU.

    Construye el embedding, la pila :class:`MyLargeMoETransformerBlock`, la
    norma final y el LM head. Las colocaciones de pesos y el cableado de la
    caché KV viven en los módulos de bloque. Esta clase cablea las capas y
    las tablas de RoPE por capa (real vs identidad para NoPE).
    """
```

## Comentarios de código incorrectos

```python
# RMSNorm por head para Q y K (específico de Gemma3)
self.q_norm = RMSNorm(...)
self.k_norm = RMSNorm(...)
```

Si tu modelo no tiene QK-norm, borra el código muerto o explica por qué quedan los campos (p. ej. cableado de sharding del donante). No dejes el comentario específico de Gemma3.

## Afirmaciones de comportamiento incorrectas que un grep de nombres pasa por alto

- ``"""Soporta inferencia single-GPU, TP multi-GPU y DP+EP."""`` en un archivo solo multi-GPU
- ``# All-reduce tras la atención (solo modo TP)`` cuando no hay modo no-TP
- ``# Aplica la norma post-atención`` en un bloque paralelo sin norma post-atención

Buscar con grep nombres de donante no atrapa estos. Lee cada docstring y comentario.

## Auditoría obligatoria antes de declarar terminada la implementación

Ejecuta desde ``<port_dir>/`` (la carpeta del slug con ``arch.py``):

```bash
grep -rniE \
  'qwen|llama|mistral|cohere|gemma|phi|deepseek|olmo|granite|qwen3|mixtral|single-GPU|single GPU|RMSNorm|QK-norm' \
  .
```

Clasifica cada coincidencia:

- **OK**: linaje explícito ("Refleja estructuralmente deepseekV3_modulev3 …"). Déjala.
- **Incorrecta**: comportamiento del donante que no aplica, como "QK-norm específico de Gemma3" sin QK-norm, o "Soporta single-GPU" en un archivo solo multi-GPU. Reescribe o borra.
- **Obsoleta**: números de línea, nombres o rutas incorrectos. Actualiza o elimina.

Luego lee cada docstring de módulo y clase sin grepear.

## Cómo se ve "terminado"

Atestigua explícitamente cuando la implementación esté completa:

> Scaffolding + implementación completos. Los docstrings de módulo siguen el patrón de arriba (deepseekV3_modulev3 bajo "deriva de", no afirmado). ``grep -rni 'deepseek' <port_dir>`` devuelve N coincidencias, todas referencias de linaje legítimas.

No informes la implementación como completa sin esa declaración.

## Por qué no prohibir los renombrados

La estructura de archivos del donante es útil de mantener. El texto incorrecto viene de saltarse la *revisión post-renombrado*, así que la auditoría apunta a ese paso. Produce artefactos concretos: salida de grep y docstrings reescritos.
