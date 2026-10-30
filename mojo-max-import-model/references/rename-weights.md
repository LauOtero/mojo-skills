# Renombrar pesos de safetensors para MAX

Hugging Face y MAX suelen tener nombres de módulo distintos para los mismos pesos. `weight_adapters.py` es donde los puenteas.

## Descubre las claves del checkpoint primero (solo metadatos)

Antes de escribir o editar `weight_adapters.py`, lista lo que trae el checkpoint del Hub (claves, formas y dtypes) sin descargar los payloads de los tensores:

```bash
pixi run python scripts/list_checkpoint_keys.py <HF_MODEL_ID> --summary
pixi run python scripts/list_checkpoint_keys.py <HF_MODEL_ID> \
  --prefix model. --limit 40
```

Esto usa `huggingface_hub.get_safetensors_metadata` (parseo de cabeceras HTTP / índices). Úsalo para comprobar:

- si el repo es safetensors (falla rápido en repos solo-bin)
- formas por tensor (deriva `head_dim`, diseño de expertos MoE y ancho de expertos compartidos)
- dtypes por tensor (atrapa normas F32 sueltas en checkpoints BF16)
- los prefijos de clave que tu adaptador debe eliminar o descartar

Ejecútalo **después** de leer `config.json` y **antes** de editar el scaffold o el adaptador. Ver también [read-config-json.md](read-config-json.md).

## Cómo funciona el cargador de pesos de MAX

El pipeline corre tu `weight_adapters.py` sobre el checkpoint primero. El adaptador es una función que toma un dict de clave de checkpoint → tensor y devuelve un dict de FQN de MAX → tensor. `compile(weights=...)` luego recorre los parámetros del módulo raíz y busca cada nombre completamente cualificado (FQN) en ese dict. El FQN es la ruta de atributo desde el módulo raíz. Un parámetro sin entrada lanza `KeyError`, y un desajuste de forma o dtype lanza `ValueError`. Una entrada que no coincide con ningún parámetro se omite sin error.

## Descubrir los FQN esperados por MAX

Después del scaffold, antes de depurar nada, lista los FQN que tu modelo MAX espera:

```python
# En un REPL de Python con tu scaffold importable:
from max.experimental import functional as F
from your_package.your_slug import YourModel

# Construye perezosamente, así no se asignan pesos:
with F.lazy():
    m = YourModel(
        ...
    )  # usa los args que pasa el _instantiate_module de tu modelo de pipeline
print(sorted(name for name, _ in m.parameters))
```

Esos son los nombres que tu adaptador debe producir.

Luego lista las claves de safetensors del Hub (preferido, porque omite la descarga):

```bash
pixi run python scripts/list_checkpoint_keys.py <HF_MODEL_ID>
```

O, si ya tienes una caché local, abre los shards con `safe_open`:

```python
from safetensors import safe_open

keys = []
for shard in [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]:
    with safe_open(f"<checkpoint_path>/{shard}", framework="pt") as f:
        keys.extend(f.keys())
print(sorted(keys))
```

Haz diff de las dos listas. Los diffs son lo que tu adaptador debe reescribir.

## Patrones de renombrado comunes

### Reescribe el prefijo `model.`

HF envuelve el decodificador dentro de `model.layers.<i>...`. El módulo raíz de MAX normalmente guarda el modelo de texto bajo un atributo propio, y sus nombres de parámetro empiezan por ese atributo (`language_model.` en `olmo3`). La mayoría de los adaptadores empiezan con un renombrado de prefijo:

```python
PREFIX_MAP = {
    "model.embed_tokens.": "language_model.embed_tokens.",
    "model.norm.": "language_model.norm.",
    "model.layers.": "language_model.layers.",
    "lm_head.": "language_model.lm_head.",
}


def adapt(state):
    out = {}
    for key, value in state.items():
        for before, after in PREFIX_MAP.items():
            if key.startswith(before):
                key = after + key.removeprefix(before)
                break
        out[key] = value
    return out
```

Haz coincidir los prefijos con los nombres de atributo de tu módulo raíz. Un raíz que guarda las capas decodificadoras directamente toma un `removeprefix("model.")` simple.

### Capa de embedding

| HF                          | MAX (varía por arch)                             |
|-----------------------------|--------------------------------------------------|
| `model.embed_tokens.weight` | `embed_tokens.weight` o `tok_embeddings.weight` |
| `model.norm.weight`         | `norm.weight` o `output_norm.weight`            |
| `lm_head.weight`            | `lm_head.weight` (ausente cuando está atado)              |

### Proyecciones de atención

Si la atención de MAX guarda parámetros Q/K/V **no fusionados** y los concatena en `forward()` (`olmo3`, `gemma3_modulev3`):

| HF                                 | MAX                                |
|------------------------------------|------------------------------------|
| `layers.N.self_attn.q_proj.weight` | `layers.N.self_attn.q_proj.weight` |
| `layers.N.self_attn.k_proj.weight` | `layers.N.self_attn.k_proj.weight` |
| `layers.N.self_attn.v_proj.weight` | `layers.N.self_attn.v_proj.weight` |
| `layers.N.self_attn.o_proj.weight` | `layers.N.self_attn.o_proj.weight` |

A menudo no se necesita renombrado más allá del prefijo.

Si la atención de MAX guarda un parámetro Q/K/V **fusionado** (algunos módulos personalizados), concatena los tensores HF a lo largo de la dimensión de salida en el adaptador:

| HF                                                | MAX                                       |
|---------------------------------------------------|-------------------------------------------|
| `q_proj.weight`, `k_proj.weight`, `v_proj.weight` | `qkv_proj.weight` (filas Q, K, V en orden) |

### MLP

Para SwiGLU (familia Llama):

| HF                     | MAX                    |
|------------------------|------------------------|
| `mlp.gate_proj.weight` | `mlp.gate_proj.weight` |
| `mlp.up_proj.weight`   | `mlp.up_proj.weight`   |
| `mlp.down_proj.weight` | `mlp.down_proj.weight` |

Para MLP de dos capas (estilo GPT):

| HF               | MAX              |
|------------------|------------------|
| `mlp.fc1.weight` | `mlp.fc1.weight` |
| `mlp.fc2.weight` | `mlp.fc2.weight` |

Algunos modelos HF usan `dense_h_to_4h` / `dense_4h_to_h`. Renómbralos en el adaptador.

### Normas de capa

| HF                                         | MAX                                        |
|--------------------------------------------|--------------------------------------------|
| `layers.N.input_layernorm.weight`          | `layers.N.input_layernorm.weight`          |
| `layers.N.post_attention_layernorm.weight` | `layers.N.post_attention_layernorm.weight` |

Para modelos post-norm o peri-LN con normas extra (`post_norm1`, `post_feedforward_layernorm`, etc.), añade renombrados explícitos que las apunten a los nombres de atributo del módulo MAX correspondiente.

## Errores comunes de mapeo de pesos

### El diseño del peso de `Conv2d` sigue a `permute`

`max.experimental.nn.Conv2d(..., permute=True)` toma su `weight` en orden PyTorch, `(out_channels, in_channels / num_groups, height, width)`, y lo lee en ese diseño, así que el adaptador pasa el tensor HF sin cambios. Ese modo también toma entrada NCHW, y su diseño de filtro FCRS necesita una GPU con cuDNN. Con `permute=False` (por defecto), la capa espera el orden MAX, `(height, width, in_channels / num_groups, out_channels)`, y el adaptador debe permutar el tensor HF (`w.permute(2, 3, 1, 0)` en torch). Elige uno y mantenlo consistente en el adaptador.

### Los embeddings atados conservan un tensor

Cuando `tie_word_embeddings` está establecido, el módulo no tiene parámetro `lm_head` y multiplica por `embed_tokens.weight.T` en `forward()`. El adaptador pasa `embed_tokens.weight` sin cambios y descarta cualquier `lm_head.weight` que traiga el checkpoint:

```python
def adapt(state):
    out = {
        k.replace("model.", "language_model.", 1): v for k, v in state.items()
    }
    if config.tie_word_embeddings:
        out.pop("lm_head.weight", None)
    return out
```

Un `lm_head.weight` copiado para un modelo atado no coincide con ningún parámetro, así que se omite. Un port que construye un `lm_head` no atado para un checkpoint atado falla `compile()` con un `KeyError` por él.

### Checkpoints con QKV fusionado

GPT-NeoX, BLOOM y algunos otros traen `query_key_value.weight` como un solo tensor fusionado. Divídelo en el adaptador:

```python
import torch

hidden = config.hidden_size
qkv = state["transformer.h.0.attention.query_key_value.weight"].data()
# Diseño 1: [Q, K, V] concatenados a lo largo de la dimensión de salida
arr = torch.from_dlpack(qkv)  # NumPy no puede leer bfloat16
q = arr[:hidden]
k = arr[hidden : 2 * hidden]
v = arr[2 * hidden:]
# Diseño 2 (head-intercalado): cada head es [Q_head, K_head, V_head]
# Comprueba en el código de modelado qué diseño usa el modelo.
```

## Descubrir renombrados haciendo diff

Para escribir un adaptador a partir de las dos listas de claves:

1. Ejecuta `list_checkpoint_keys.py` en el repo del Hub (solo metadatos).
2. Imprime los FQN que espera MAX (desde el árbol de módulos de tu port).
3. Empareja por inspección.
4. Escribe el adaptador para hacer los renombrados.

Este helper imprime ambas diferencias:

```python
def diff_keys(hf_keys, max_keys):
    """Imprime las claves HF que MAX no espera, y las claves MAX que HF no suministra."""
    hf_set, max_set = set(hf_keys), set(max_keys)
    print("HF only:")
    for k in sorted(hf_set - max_set)[:30]:
        print(f"  {k}")
    print("MAX only:")
    for k in sorted(max_set - hf_set)[:30]:
        print(f"  {k}")
```

Si "MAX only" no está vacío después de correr tu adaptador, `compile()` falla con un `KeyError` por la primera. Si "HF only" no está vacío, esos tensores se omiten sin error. Cada uno es o un bug de renombrado o un componente que tu módulo no construye. El adaptador debe producir *cada* FQN de MAX, y cada tensor HF debe aterrizar en uno o descartarse a propósito (ver [state-dict-audit.md](state-dict-audit.md)).

## Verificar el adaptador

Comprueba la salida del adaptador antes de compilar. Esta comprobación de cordura lee una capa pequeña:

```python
import torch

state = convert_safetensor_state_dict(raw_weights, hf_config, pipeline_config)
w = torch.from_dlpack(state["language_model.layers.0.self_attn.q_proj.weight"])
print(w.abs().max())
# Debería ser O(0.1) para un modelo entrenado. Sospecha de O(0.01) o O(1.0).
```

Para una comprobación exhaustiva, ejecuta `compare_layers.py` de la caza de divergencias una vez que los pesos carguen limpiamente y antes de declarar terminados los logits. La primera capa ya debería coincidir. Si no, el adaptador está mal.
