# Leer `config.json`

Cada modelo de Hugging Face incluye un `config.json`. Cada delta arquitectónico aparece en sus campos, así que léelo antes que el código de modelado.

Cárgalo e imprime cada campo:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("<HF_MODEL_ID>", trust_remote_code=True)
print(config)
```

## Campos que mapeas 1:1 a MAX

Estos tienen el mismo significado en todo modelo de la familia transformer. Tu `model_config.py` de MAX los tomará directamente de `pipeline_config.model.huggingface_config`.

| Campo                                          | Significado                                                                    |
|------------------------------------------------|--------------------------------------------------------------------------------|
| `hidden_size`                                  | Ancho del flujo residual                                                       |
| `num_hidden_layers`                            | Número de bloques transformer                                                  |
| `num_attention_heads`                          | Número de heads Q                                                              |
| `num_key_value_heads`                          | Número de heads K/V (igual a `num_attention_heads` cuando no es GQA)            |
| `intermediate_size`                            | Dimensión oculta del MLP                                                       |
| `vocab_size`                                   | Tamaño del vocabulario del tokenizer                                           |
| `max_position_embeddings`                      | Longitud de contexto de entrenamiento                                          |
| `rms_norm_eps` / `layer_norm_eps`              | Epsilon de la norma                                                            |
| `hidden_act`                                   | Activación del MLP (`silu`, `gelu`, `gelu_new`, etc.)                          |
| `tie_word_embeddings`                          | Si el LM head reutiliza la matriz de embedding                                 |
| `rope_theta`                                   | Frecuencia base de RoPE (por defecto 10000.0)                                  |
| `rope_scaling`                                 | Config de extensión de RoPE (Llama3, YaRN, etc.). Si está presente, lee cada subcampo |
| `attention_bias`                               | Si las proyecciones Q/K/V/O tienen término de sesgo                            |
| `bos_token_id`, `eos_token_id`, `pad_token_id` | Tokens especiales                                                              |

## Campos que señalan un delta

Si alguno de estos está presente y no es el valor por defecto, tienes trabajo arquitectónico más allá de un simple renombrado.

| Campo                                                                                   | Señala                                                                                                                 |
|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `sliding_window` (int)                                                                  | Atención de ventana deslizante. La atención estándar no coincidirá.                                                     |
| `attn_logit_softcapping`                                                                | Softcap aplicado a las puntuaciones de atención pre-softmax (Gemma 2).                                                  |
| `final_logit_softcapping`                                                               | Softcap aplicado a los logits finales del LM head (Gemma 2).                                                            |
| `no_rope_layers`, `no_rope_layer_interval`                                              | Capas NoPE: RoPE se omite donde la lista contiene `0` (SmolLM3). Ver divergences.md §17.                                |
| `partial_rotary_factor` (float < 1.0)                                                   | RoPE aplicado solo a una fracción de la dimensión del head.                                                             |
| `head_dim` (cuando no es igual a `hidden_size / num_attention_heads`)                   | Forma de head no estándar.                                                                                              |
| `num_experts` / `num_local_experts`                                                     | Número de capas MoE.                                                                                                    |
| `num_experts_per_tok`                                                                   | Routing top-k.                                                                                                          |
| `router_aux_loss_coef`, `output_router_logits`                                          | Confirma MoE.                                                                                                           |
| `q_lora_rank`, `kv_lora_rank`                                                           | MLA (atención latente estilo DeepSeek).                                                                                  |
| `qk_nope_head_dim`, `qk_rope_head_dim`                                                  | División de heads MLA.                                                                                                  |
| `use_qk_norm` (bool)                                                                    | Q y K se normalizan cada uno antes del producto punto.                                                                  |
| `quantization_config`                                                                   | Los pesos publicados están cuantizados, así que necesitas una ruta de dequantización.                                   |
| `rope_parameters` (dict)                                                                | `rope_theta` está anidado, no en el nivel superior. Lee `rope_parameters['rope_theta']`.                                |
| `use_post_norm`                                                                         | Peri-LN: post-norm aplicada a la salida de cada subcapa antes de la suma residual.                                      |
| `mlp_bias`                                                                              | Las capas lineales del MLP tienen término de sesgo.                                                                     |
| `embedding_multiplier`, `logits_scaling`, `attention_multiplier`, `residual_multiplier` | Escalares MuP. Deben pasarse a través de las capas o la salida es basura.                                               |
| `dim_model_base`                                                                        | A menudo se empareja con un divisor de ancho pre-`lm_head`: `h / (hidden_size / dim_model_base)`. Grep `lm_head(` en el código de modelado. |
| `output_multiplier_scale`, `pre_attn_norm_scale`, ...                                   | Campos de escala personalizados. Lee el código de modelado para ver dónde se aplican.                                   |

## Qué significa "no estándar"

La mayoría de los modelos novedosos tienen dos o tres campos no estándar. Clasifica cada uno en cosmético (un nuevo nombre de campo para un concepto conocido) o estructural (una nueva computación).

Triaje rápido:

- Un nuevo *escalar* (eps, theta, factor de escala) → casi siempre cosmético. Tómalo de la config y multiplícalo o súmalo en el punto correcto, normalmente un cambio de una línea.
- Un nuevo *booleano* (`use_qk_norm`, `use_post_norm`) → habilita condicionalmente un cambio estructural. Lee el código de modelado para ver qué protege.
- Un nuevo *entero de conteo* (`num_experts`, `q_lora_rank`) → estructural. Cambia la forma de la computación.
- Un nuevo *dict* (`rope_scaling`, `quantization_config`) → estructural. Lee cada subcampo.

## Verificar que tienes la config correcta

La config y el checkpoint publicado deben coincidir en forma. Después de cargar la config, haz una comprobación de cordura contra los metadatos de safetensors (sin descargar pesos):

```bash
pixi run python scripts/list_checkpoint_keys.py <HF_MODEL_ID> --summary
```

O inspecciona algunas claves de atención de la tabla de metadatos:

```bash
pixi run python scripts/list_checkpoint_keys.py <HF_MODEL_ID> \
  --prefix model.layers.0.self_attn --limit 10
```

Programáticamente (los mismos datos que el script):

```python
from huggingface_hub import get_safetensors_metadata

meta = get_safetensors_metadata("<HF_MODEL_ID>")
# Una entrada por archivo; un checkpoint shardeado tiene varias.
for file_meta in meta.files_metadata.values():
    for name, info in file_meta.tensors.items():
        if "self_attn.q_proj" in name:
            print(name, info.shape, info.dtype)
```

Si `config.hidden_size = 4096` y `num_attention_heads = 32` pero el `q_proj.weight` del checkpoint es `[4608, 4096]`, el modelo tiene un `head_dim` no estándar `= 4608 / 32 = 144`. La config o tiene un campo `head_dim` explícito o necesitas añadir uno en `model_config.py`.

## Salida de leer `config.json`

Apunta estas listas:

1. **Campos estándar**: nombres + valores, para `model_config.py`.
2. **Señales de atención**: campos no estándar y lo que implican. Esto alimenta la lista de deltas que construyes comparando contra el arch de MAX donante.
