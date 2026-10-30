# Causas comunes de divergencia (indexadas por síntoma)

Cuando la caza de divergencias capa por capa muestra divergencia, el bug es casi siempre uno de los patrones de abajo.

## Índice rápido: síntoma → candidatos

Haz coincidir lo que ves con una fila, luego lee todas las causas listadas, incluso después de que una parezca plausible. Varias causas producen el mismo síntoma, así que la primera coincidencia plausible puede ser la incorrecta.

| Lo que observas                                                            | Causas más probables (en orden)                                                                                                                      |
|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Basura en el token 0, token de id bajo sin relación con el prompt           | #1 nombres Q/K/V, #10 QKV fusionado, #4 embeddings atados, #15 variante de norma                                                                     |
| Mismo carácter/token repetido (`&&&&`, bucle de puntuación)                 | #17 diseño de `freqs_cis` identidad para NoPE, #3 empaquetado de RoPE, #1 mapa de pesos                                                              |
| Divergencia leve en cada post-salida de MLP (cos_sim ~0.95)                 | #14 variante de activación, #12 MLP sin puerta, #13 escalares MuP                                                                                    |
| Divergencia en el bloque 0 post-attn, cos_sim 0.1–0.5                       | #2 `repeat_kv` de GQA, #1 nombres Q/K/V                                                                                                              |
| Divergencia en el bloque 0 post-attn, cos_sim 0.7–0.9                       | #3 RoPE split-half vs intercalado, #15 dimensión de QK-norm, #19 softcap final                                                                       |
| Primer token plausible, salida se degrada durante la generación             | #4 embeddings atados, #6 relleno de RoPE parcial, #8 `rope_theta` anidado                                                                             |
| Repetición / bucle en la posición 0 tras el primer token                    | #7 embeddings posicionales absolutos, #6 relleno de RoPE parcial                                                                                     |
| La divergencia crece con la longitud de secuencia, prompts cortos bien     | #6 relleno de RoPE parcial, #8 `rope_theta` anidado, #3 estilo de RoPE                                                                               |
| Prefill bien, decode colapsa a repetición/fragmentos por el token 8-12     | #3 estilo de RoPE: el valor por defecto `RotaryEmbedding(interleaved=True)` no coincide con el `rotate_half` de HF, pasa `interleaved=False`         |
| Divergencia en post-block-norm pero ambas subcapas bien                    | #5 diseño pre/post/peri-norm, #13 `residual_multiplier` (MuP)                                                                                        |
| `actual_token == último token de entrada` en el primer decode               | #5 post-norm degenerando a identidad                                                                                                                 |
| Logits todos NaN desde la referencia HF                                     | [pitfalls-serving.md](pitfalls-serving.md#trust_remote_codetrue-with-tocuda-can-produce-nan) (`trust_remote_code` + `.to("cuda")` con desajuste de dispositivo) |
| Las salidas de capa coinciden pero el texto generado se desvía              | #16 desajuste de tokenizer/plantilla de chat, desajuste de dtype                                                                                     |

Luego recorre las causas de abajo en el orden listado. También están numeradas aproximadamente por frecuencia. Cuando ningún síntoma es obvio, empieza por la #1 y baja.

## 1. Nombres de las proyecciones Q/K/V

Los módulos de atención de MAX pueden tener QKV *fusionado* (un `qkv_proj.weight`) o proyecciones *no fusionadas* (`q_proj.weight`, `k_proj.weight`, `v_proj.weight`, concatenadas en `forward()` como hace `olmo3/layers/attention.py`). Los checkpoints de HF casi siempre traen proyecciones no fusionadas. Asegúrate de que el renombrado de tu `weight_adapters.py` coincide con los nombres de parámetro reales del módulo, no con lo que asumes.

Para verificar: construye el módulo raíz bajo `F.lazy()` e imprime `[name for name, _ in model.parameters]`. Esos nombres son los que tu adaptador debe producir.

**Síntoma:** `compile()` lanza `KeyError` por el parámetro no emparejado. Si el adaptador mapea los tensores a un nombre que existe pero contiene la proyección equivocada, el primer token es basura, a menudo un token de id bajo sin relación con el prompt.

## 2. Falta el `repeat_kv` de GQA

Para modelos con `num_key_value_heads < num_attention_heads`, K y V deben repetirse `num_attention_heads / num_key_value_heads` veces antes del producto punto de atención. Los kernels de atención ragged en `max.experimental.nn.common_layers.functional_kernels` (`flash_attention_ragged`) lo manejan a partir del conteo de heads KV en `kv_params`. La matemática de atención personalizada fuera de esos kernels debe repetir K y V por sí misma.

**Síntoma:** la divergencia empieza en el primer bloque de atención, cos_sim post-attn alrededor de 0.1–0.5.

## 3. Desajuste del estilo de RoPE (split-half vs. intercalado)

Hay dos convenciones de RoPE en el zoo de HF:

- **Split-half** (estilo GPT-NeoX): rota `[x_primera_mitad, x_segunda_mitad]` → `[-x_segunda_mitad, x_primera_mitad]`. Lo usan GPT-NeoX, Llama, Qwen, Mistral, OLMo, OLMoE, Gemma, Phi y la mayoría de los decodificadores modernos.
- **Par adyacente** (estilo GPT-J): rota `[x0, x1, x2, x3, ...]` → `[-x1, x0, -x3, x2, ...]`. Lo usan GPT-J, Helium, ChatGLM/GLM-4, ERNIE 4.5 y un conjunto más pequeño.

Ambas convenciones producen las mismas formas y valores distintos. La comprobación de la etapa de scaffold pasa sin crash ni error, y las salidas de atención están mal desde la primera capa.

**El nombre de función `rotate_half` en HF está sobrecargado.** Los modelos de ambos grupos llaman a su función de rotación RoPE `rotate_half`, y las implementaciones difieren. Lee la implementación en `modeling_<type>.py` y busca la indexación:

```python
# Split-half (Llama, OLMoE, etc.):
x1 = x[..., : x.shape[-1] // 2]
x2 = x[..., x.shape[-1] // 2 :]

# Par adyacente (helium, ChatGLM, etc.):
x1 = x[..., 0::2]
x2 = x[..., 1::2]
```

En MAX, el argumento `interleaved=` de `RotaryEmbedding` selecciona entre ellas. **`RotaryEmbedding(...)` usa por defecto `interleaved=True` (par adyacente estilo GPT-J).** Pasa `interleaved=False` para modelos split-half. `llama3_modulev3` pasa `config.interleaved_rope_weights` como `interleaved`. Su config pone el campo a `False` para checkpoints safetensors y a `True` solo para GGUF con `rope_type="normal"`. Para un modelo safetensors de par adyacente sobre ese donante, establece `cfg.interleaved_rope_weights = True` en el `from_donor()` de tu port.

**Los modelos de par adyacente** también se pueden servir con `interleaved=False` si el adaptador de pesos pre-permuta los pesos de Q/K a orden split-half antes de la carga. Para una permutación relacionada de pesos Q/K, ver `_build_partial_rope_perm` en `step3p5/weight_adapters.py`. Es el movimiento correcto cuando el modelo tiene RoPE de par adyacente *parcial* (algunos canales rotados, otros no) y quieres reutilizar el codepath de `Llama3RotaryEmbedding`.

**Síntoma A (generación de texto):** la salida se degrada con la posición. En casos leves, el prefill produce un primer token o dos sensatos, y luego el decode colapsa a repetición de una sola palabra o bucles de fragmentos alrededor del token 8-12. En casos agudos el primer token ya está mal: ERNIE 4.5 servido con RoPE split-half imprime `' 100000000…'` desde el token 1. En cualquier caso, la comprobación de prefill de `compare_layers.py` reporta `DIVERGED`, mientras que la sonda top-1 multiposición puede seguir coincidiendo en prefijos cortos.

**Síntoma B (volcado de paridad):** divergencia en el bloque 0 post-attn, cos_sim alrededor de 0.7–0.9 (cerca pero visiblemente mal).

## 4. Embeddings atados

Si `config.tie_word_embeddings=True`, cada logit viene del peso del embedding. El patrón ModuleV3 no construye un parámetro `lm_head`: `forward()` computa los logits como `h @ self.embed_tokens.weight.T`, y el adaptador de pesos descarta el `lm_head.weight` del checkpoint (ver [pitfalls-weights.md](pitfalls-weights.md)).

**Síntoma:** la salida del primer token es plausible (el embedding funciona), pero el modelo se degrada conforme avanza la generación. Un port con su propio parámetro `lm_head` carga inicialización aleatoria para cualquier token id que el embedding vea con poca frecuencia.

## 5. Diseño de bloque pre-norm vs. post-norm vs. peri-LN

Un bloque donante estilo Llama (`LlamaTransformerBlock.forward()` en `llama3_modulev3`) está fijado a orden pre-norm: `h = x + attn(norm(x)); out = h + mlp(norm(h))`. Estos patrones lo rompen:

- **Post-norm puro** (EXAONE 4, algunos modelos antiguos): la norma se aplica a la *salida* de la subcapa antes de la suma residual. `h = x + norm(attn(x))`.
- **Peri-LN / doble norma** (Gemma 2, HyperCLOVAX con `use_post_norm=True`): pre-norm *y* una norma adicional en la salida de la subcapa antes de la suma residual.

Ambos requieren un bloque personalizado con un `forward()` reescrito. Ver [read-modeling-code.md](read-modeling-code.md#elegir-una-estrategia-de-edición) para la estrategia de edición.

**Síntoma:** la divergencia aparece primero en post-block-norm, no en post-attn ni post-mlp. El "token real" en la comprobación de coherencia suele ser el *último token de entrada* (el modelo degenera a identidad en el primer paso de decode).

## 6. Relleno de RoPE parcial

El `RotaryEmbedding` base produce pares de frecuencia intercalados `[cos0, sin0, cos1, sin1, ...]`, no split-half. Cuando `partial_rotary_factor < 1.0`, RoPE se aplica solo a parte de la dimensión del head. El relleno de la sección no rotada debe entonces ser *pares identidad intercalados* `[1, 0, 1, 0, ...]`, no split `[1, 1, ..., 0, 0, ...]`.

**Síntoma:** la divergencia crece con la longitud de secuencia. Los prompts cortos pueden pasar, pero los decodes de 200+ tokens producen salida incorrecta.

## 7. Embeddings posicionales absolutos (no-RoPE)

Para modelos sin RoPE (GPT-BigCode, OPT, BLOOM), las posiciones se suman al embedding. Durante el decode autorregresivo, el índice de posición debe venir de `cache_lengths` en las entradas de la caché KV. No lo resetees a 0 en cada paso.

**Síntoma:** el primer token está bien. Cada token posterior atiende a la posición 0, no a su posición real, produciendo salida repetitiva.

## 8. `rope_theta` anidado en `rope_parameters`

Configs más recientes de Gemma 2, Helium y otros modelos anidan `rope_theta` dentro de un dict `rope_parameters`. Leer `config.rope_theta` de nivel superior devuelve `None`, cayendo al valor por defecto 10000.0, que es incorrecto.

```python
theta = getattr(config, "rope_theta", None) or config.rope_parameters.get(
    "rope_theta", 10000.0
)
```

**Síntoma:** las salidas parecen plausibles a longitudes cortas pero se degradan conforme crecen las posiciones, porque el error de RoPE escala con la posición.

## 9. `F.constant` creado con dtype o dispositivo por defecto

`F.constant(value)` sin `dtype` y `device` aterriza en el acelerador como `bfloat16`. Ver [pitfalls-graph.md](pitfalls-graph.md#fconstant-defaults-to-the-accelerator-and-bfloat16).

**Síntoma:** pequeños errores de precisión (un epsilon o escala redondeada a bfloat16), o un desajuste de dispositivo en la op que consume la constante.

## 10. QKV fusionado en checkpoints de HF

GPT-NeoX, BLOOM y algunos otros traen un único peso `query_key_value` fusionado. Para cargar en las proyecciones Q/K/V separadas de MAX, el adaptador de pesos debe dividirlo:

```python
import torch

qkv = checkpoint["...query_key_value.weight"].data()
arr = torch.from_dlpack(qkv)  # NumPy no puede leer bfloat16
q = arr[:hidden_size]
k = arr[hidden_size : 2 * hidden_size]
v = arr[2 * hidden_size:]
```

Para modelos donde el diseño fusionado intercala heads (`[Q0, K0, V0, Q1, K1, V1, ...]`), la división es más elaborada. Deriva el diseño de heads del `forward` de HF.

## 11. Residuales en paralelo

Los bloques transformer estándar usan residuales secuenciales (la salida de atención va al flujo residual, y luego el MLP lee de ese flujo). GPT-NeoX usa *residuales en paralelo*: atención y MLP leen ambos del mismo `x`, y sus salidas se suman.

```text
out = x + attn(ln1(x)) + mlp(ln2(x))   # paralelo
```

vs.

```text
h = x + attn(ln1(x)); out = h + mlp(ln2(h))   # secuencial
```

Un bloque donante estilo Llama solo hace secuencial. Escribe un bloque personalizado cuyo `forward()` coincida con la computación de residual paralelo.

## 12. MLP sin puerta

El `MLP` de serie en MAX es con puerta (SwiGLU): `down(silu(gate(x)) * up(x))`. Para modelos con un MLP plano de dos capas (`fc2(act(fc1(x)))`, común en GPT-style, OPT, XGLM, BLOOM, BioGPT), necesitas una clase MLP personalizada con dos lineales y la activación correcta. Asegúrate de que los nombres de atributo de capa coinciden con las claves de pesos de HF (`fc1`/`fc2` o `dense_h_to_4h`/`dense_4h_to_h`).

## 13. Escalares MuP

Los modelos entrenados con Maximal Update Parametrization (MuP), como Granite, Granite-MoE y HyperCLOVAX, dependen de estos multiplicadores escalares:

- `embedding_multiplier`: escala la salida de la capa de embedding.
- `logits_scaling`: escala los logits *después* de `lm_head` en HF (los divide en algunas familias). Un divisor de ancho pre-head (`h / (hidden_size / dim_model_base)`) es un escalar distinto. Grep `lm_head(` en [read-modeling-code.md](read-modeling-code.md).
- `residual_multiplier`: escala las salidas de subcapa antes de la suma residual.
- `attention_multiplier`: reemplaza `1/sqrt(d)` como escala de atención.

Los valores por defecto son 1.0 (no-op), así que los modelos no-MuP los ignoran. Los modelos MuP les dan valores distintos de 1.0 por defecto, y cada uno debe pasarse a través de las capas. Un multiplicador faltante produce salidas incorrectas sin error.

En MAX, la config de `llama3_modulev3` lee `embedding_multiplier`, `residual_multiplier`, `attention_multiplier` y `logits_scaling`, y sus módulos los aplican. En tu `model_config.py`, tómalos de la config de HF con valores por defecto de `1.0`.

## 14. Variantes de función de activación

`gelu`, `gelu_new`, `gelu_tanh`, `gelu_pytorch_tanh`, `geglu`, `gelu_fast`, `silu`, `swish`. No todas son la misma. Busca la función exacta en `transformers/activations.py` (dict `ACT2FN`) y hazla coincidir en MAX. Trampas comunes:

- `gelu` es la GELU exacta basada en erf.
- `gelu_new` y `gelu_tanh` son la aproximación tanh (misma función, distinto nombre).
- `geglu` es la variante con puerta (usada en T5 y algunos MLPs).

**Síntoma:** divergencia leve en post-mlp, en cada bloque. cos_sim alrededor de 0.95.

## 15. Variantes de norma

- **RMSNorm con escala `+1`** (Gemma 2): `x * weight + x`, donde la RMSNorm estándar usa `x * weight`. Equivale a `x * (1 + weight)`.
- **QK-norm por head** (Olmo2, Qwen3, EXAONE 4): la dimensión de la norma es `head_dim`, no `hidden_size`. La norma se aplica después del reshape por head, no antes.
- **LayerNorm vs. RMSNorm**: la mayoría de los modelos modernos usan RMSNorm, pero algunos embedders, encoders y decodificadores antiguos usan LayerNorm (con escala y sesgo).

## 16. Desajuste de tokenizer / plantilla de chat

Si las salidas capa por capa coinciden pero el *texto generado* no, el modelo en sí está bien. El problema es uno de:

- Tokens especiales distintos (BOS, EOS) añadidos.
- Plantilla de chat distinta para modelos instruct.
- Tokenización distinta (diferencias de merge BPE, normalización).

Para aislarlo, tokeniza el mismo prompt con ambos tokenizers HF y MAX y compara los ids de token. Si difieren, el bug está en la configuración del tokenizer, no en el modelo.

## 17. Capas NoPE / skip-RoPE vía `freqs_cis` identidad

Algunos modelos omiten RoPE en un subconjunto de capas (capas NoPE). La config dice cuáles, y el campo difiere por familia:

- `no_rope_layers` (SmolLM3): una lista por capa, `1` para RoPE y `0` para NoPE. `no_rope_layer_interval` genera la misma lista cuando la lista está ausente.
- `layer_types` (ciertos MoE y híbridos estilo EXAONE): RoPE solo en capas `"sliding_attention"`, NoPE en capas `"full_attention"`.

Lee el `Attention.forward` de HF para ver la condición que realmente comprueba. Un modelo puede llevar ambos campos: las `layer_types` de SmolLM3 son todas `"full_attention"`, y sus capas NoPE vienen de `no_rope_layers`.

HF ramifica en `Attention.forward`, pero los kernels fusionados de MAX (`rope_split_store_ragged`) a menudo **no tienen flag de skip en runtime**. Patrón común en MAX: construir una segunda tabla **identidad** `freqs_cis` (slots cos = 1.0, sin slots = 0.0) y dar a cada capa NoPE la tabla identidad. En `llama3_modulev3`, la atención de cada capa lee su módulo `rope`, así que una capa NoPE puede recibir un módulo rotatorio cuyo `freqs_cis` es la tabla identidad.

La tabla identidad debe coincidir con cómo MAX almacena `freqs_cis`, no con cómo HF nombra el estilo de RoPE. Las clases rotatorias en `max.experimental.nn.common_layers.rotary_embedding` almacenan `freqs_cis` como pares intercalados, `[cos_0, sin_0, cos_1, sin_1, …]`, diga lo que diga el flag `interleaved`. El flag solo selecciona cómo se rotan Q y K. Así que construye la tabla identidad como `[1, 0, 1, 0, …]`. Una identidad split-half, `[1, 1, …, 0, 0, …]`, da a las capas de atención completa una rotación incorrecta. La salida es basura o caracteres repetidos, incluso cuando las capas deslizantes parecen parcialmente correctas.

**Verifica antes de iterar arreglos de MoE o router:**

```python
# Vuelca la primera fila de freqs_cis real desde el módulo rotatorio de MAX vs tu tabla identidad
print(real_freqs_cis[0, :8])
print(identity_freqs_cis[0, :8])
```

**Síntoma:** depende del modelo. Con un diseño identidad incorrecto, la ruta de ventana deslizante parece parcialmente plausible, y luego colapsa a `&&&&` o bucles de puntuación. Con NoPE omitido por completo, la salida puede mantenerse fluida y sobre el tema: SmolLM3 servido con RoPE en cada capa responde con sensatez y se desvía de HF después de unos 12 tokens. Solo la comprobación de logprob lo atrapa (`logits_prefill` reportó `rel_diff=0.2404`), así que una respuesta coherente no descarta NoPE.

## 18. Forma de `F.sum`

En MAX, `F.sum(x, axis=-1)` sobre un tensor `[N, H, D]` devuelve `[N, H, 1]`, mientras que PyTorch devuelve `[N, H]`. Haz squeeze después si necesitas la forma de rango 2.

## 19. Softcap final de logits

Gemma 2 y algunos otros aplican `softcap * tanh(logits / softcap)` a la salida del LM-head. Se sienta entre la capa lineal y el muestreo. Añadirlo es una línea. Sin él, los logits en tokens raros son demasiado grandes, lo que distorsiona el decoding greedy.

