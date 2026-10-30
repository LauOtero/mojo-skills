# ¿Ya está en MAX?

Antes de escribir código, lee el `config.json` del modelo y comprueba si MAX ya registra esa clase de arquitectura.

## Precomprobación de acceso a repos con gating de HF (especialmente para ports VL)

Cohere, Meta, algunos Mistral y muchos repos VL se distribuyen con **gating**. Incluso con un `$HF_TOKEN` válido, tu cuenta puede no estar en la lista de permitidos por repo. Un repo con gating bloquea la auditoría de arquitectura y cualquier intento de servido directo cinco minutos después del bring-up, después de que el resto de la comprobación de guarda haya corrido.

```bash
pixi run python -c "
from huggingface_hub import HfApi
info = HfApi().model_info('<HF_ID>')
print('gated:', info.gated)
# La bandera 'gated' de los metadatos no es fiable. Solo una llamada
# de descarga confirma el acceso:
from huggingface_hub import hf_hub_download
hf_hub_download('<HF_ID>', 'config.json')
print('access ok')
"
```

Si tiene gating, busca un espejo sin gating (a menudo `mlx-community/<base>-bf16` o un repack similar de Apple-MLX). Si no existe, solicita acceso o marca el bring-up como BLOQUEADO por autenticación. No generes un scaffold de un slug que no puedas validar. Este bloqueo es común en repos vision-language con gating cuando tu cuenta carece de acceso a la lista de permitidos.

## Identifica la arquitectura

Cada checkpoint de Hugging Face incluye un `config.json`. Estos campos identifican la arquitectura:

| Campo              | Ejemplo            | Uso en esta guarda                                                                                |
|--------------------|--------------------|--------------------------------------------------------------------------------------------------|
| `architectures[0]` | `Qwen3ForCausalLM` | MAX registra esta clase. Pásala a `--match`.                                                  |
| `model_type`       | `qwen3`            | Localiza `modeling_<type>.py` en Transformers al comparar con un donante. No se usa para esta comprobación. |

MAX mantiene un registro que mapea cada nombre de clase de Hugging Face a un slug interno (por ejemplo `LlamaForCausalLM` → `llama3`). Si el `architectures[0]` de tu modelo ya está en ese registro, sirves los pesos del Hub directamente sin port.

### Lee `architectures[0]`

```bash
pixi run python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('<HF_MODEL_ID>', trust_remote_code=True)
print('architectures[0]:', cfg.architectures[0])
print('model_type:', cfg.model_type)
"
```

O abre el archivo crudo del Hub: `https://huggingface.co/<HF_MODEL_ID>/raw/main/config.json` y lee el array `"architectures"`.

### Comprueba el registro de MAX

```bash
pixi run python scripts/list_native_archs.py --match <architectures[0]>
```

- **Salida 0** e imprime `ClassName\tslug` → MAX ya lo soporta. Ejecuta `pixi run max serve --model <HF_MODEL_ID>` y detente. No se necesita port.
- **Salida 1** (sin salida) → no registrado. Continúa con la Fase 1 (lee la model card).
- **Salida 2** (pista de instalación en stderr) → MAX no está en el entorno Python activo. Arregla primero el entorno pixi. No trates esto como "necesita un port".

Para navegar por todo lo que MAX incluye:

```bash
pixi run python scripts/list_native_archs.py
```

## Escanea muros de port

Antes de generar el scaffold, escanea la config del Hub en busca de bloqueadores duros (ALiBi, pesos solo cuantizados, señales SSM/recurrencia, escala extrema):

```bash
pixi run python scripts/check_walls.py <HF_MODEL_ID>
```

- **Salida 0**: sin bloqueadores.
- **Salida 1**: solo warnings (revisa [recognize-walls.md](recognize-walls.md)).
- **Salida 2**: al menos un bloqueador duro. No generes el scaffold hasta que se resuelva.

## Ejemplo resuelto

`Qwen/Qwen3-8B` tiene esto en `config.json`:

```json
"architectures": ["Qwen3ForCausalLM"],
"model_type": "qwen3"
```

Comprobación:

```bash
pixi run python scripts/list_native_archs.py --match Qwen3ForCausalLM
# Qwen3ForCausalLM    qwen3
# Qwen3ForCausalLM    qwen3_embedding
```

Cada fila es un directorio de arquitectura registrado. Qwen3 tiene dos, uno para generación de texto y otro para embeddings, y MAX elige el que coincide con la tarea que sirves.

Registrado → terminado:

```bash
pixi run max serve --model Qwen/Qwen3-8B
```

Para un modelo cuya clase no está en el registro, como `NewFamilyForCausalLM`, el mismo comando `--match` sale con 1 sin salida. Esa es tu señal para portar (continúa con la Fase 1).

**No confundas `model_type` con `architectures[0]`.** Un checkpoint puede tener `"model_type": "llama"` mientras `"architectures": ["LlamaForCausalLM"]`. Haz siempre la coincidencia por el nombre de clase en `architectures[0]`, no por el nombre de la carpeta.
