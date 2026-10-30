# Servir e iterar (detalle)

**Ruta:** `<port_dir>` es la carpeta del slug con `arch.py` y `ARCHITECTURES` en `__init__.py` (después del scaffold, `<port_dir> = <output_dir>/<slug>`). Pasa el mismo `<port_dir>` a `--custom-architectures` y a `run_oss_gates.py --port-dir`. Sin barra final. MAX lee `<output_dir>/<slug>/` como un nombre de módulo vacío y falla con `Failed to import custom model`.

MAX carga archs personalizados tomando `dirname(<port_dir>)` para `sys.path` e importando `basename(<port_dir>)` como el módulo. No pases el padre de `<port_dir>`. MAX entonces importa el paquete incorrecto (`custom-arch` en lugar de tu slug), y ves `AttributeError: module '…' has no attribute 'ARCHITECTURES'`.

Forma opcional con dos puntos: ``<parent_on_sys.path>:<module_name>`` (mismo efecto que pasar `<port_dir>`).

## Requisitos previos: ejecútalos antes de `pixi run max serve`

`max serve` compila todo el modelo antes de responder a una petición, así que un error que las comprobaciones de abajo atrapan en segundos, de otra forma cuesta una compilación. `run_oss_gates.py` cubre muros, metadatos del checkpoint, y el nombre y encoding de `arch.py`.

### Comprobación del port

`check_port.py` importa el port como lo hace `max serve`, construye el pipeline que `max serve` construiría, y se detiene en `load_model()` justo después de que el port construya su módulo raíz bajo `F.lazy()`. Luego compara los parámetros del módulo con los tensores que tu adaptador de pesos devuelve para el checkpoint real:

```bash
pixi run python scripts/check_port.py <HF_MODEL_ID> --port-dir <port_dir>
```

Para un port de embeddings, añade `--task embeddings_generation`. El informe lista:

- **missing**: un parámetro sin tensor. `compile()` lanza `KeyError`.
- **shape mismatch** y **dtype mismatch**: `compile()` lanza `ValueError`. Castea los tensores float32 o bfloat16 al dtype del parámetro en el adaptador ([pitfalls-weights.md](pitfalls-weights.md)).
- **unconsumed**: un tensor que ningún parámetro lee. `compile()` lo ignora, lo que oculta un renombrado incorrecto.

El script sale con 1 cuando un parámetro falta o está mal. Los tensores no consumidos no lo fallan, porque un port puede descartar tensores a propósito (un head MTP, escalas de caché KV). Explica cada uno. Ver [rename-weights.md](rename-weights.md) y [state-dict-audit.md](state-dict-audit.md).

### Preflight del formato de pesos

MAX carga solo `.safetensors` o `.gguf` (`WeightsFormat` en `max/graph/weights/format.py`). No puede cargar `.bin`. En `<port_dir>/arch.py`, copia `default_weights_format` y `weight_adapters` del donante del scaffold bajo `max/pipelines/architectures/<donor>/arch.py`. Ver "Trampas de importación y de API de configuración" en [pitfalls-config.md](pitfalls-config.md#import-and-config-api-traps).

Comprueba la lista de archivos del repo:

```bash
pixi run python -c "
from huggingface_hub import HfApi
files = HfApi().list_repo_files('<HF_MODEL_ID>')
has_st = any(f.endswith('.safetensors') for f in files)
has_gguf = any(f.endswith('.gguf') for f in files)
has_bin = any(f.endswith('.bin') for f in files)
print(f'safetensors={has_st}  gguf={has_gguf}  bin_only_legacy={has_bin and not has_st}')
if has_bin and not has_st:
    print('STOP: convierte a safetensors o elige un repo GGUF; MAX no puede cargar .bin')
"
```

## Comprobación de cordura de la referencia HF primero

Carga la referencia en float32 cuando quepa. Una referencia bfloat16 añade su propio redondeo, así que un port bfloat16 correcto puede desviarse de ella en unos pocos tokens.

```bash
pixi run python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = '<HF_MODEL_ID>'
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    mid, trust_remote_code=True, device_map='auto', dtype=torch.float32,
).eval()
ids = tok('<MODEL CARD EXAMPLE PROMPT>', return_tensors='pt').input_ids.to(m.device)
with torch.no_grad():
    out = m.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=False))
"
```

## Servir y sondear

```bash
pixi run max serve --model <HF_MODEL_ID> \
  --custom-architectures <port_dir> \
  --quantization-encoding <default_encoding from arch.py>

curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<HF_MODEL_ID>", "prompt": "<MODEL CARD EXAMPLE PROMPT>", "max_tokens": 64}' \
  | pixi run python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])"
```

Usa la plantilla de la model card para el prompt. El campo `model` debe coincidir con `--model` (o `--served-model-name` si lo estableciste).

## Slugs de encoder / embeddings

Usa el mismo `--custom-architectures <port_dir>`, más `--task embeddings_generation`. Sin él, `max serve` elige generación de texto para un nombre de arquitectura que más de una tarea registra (`Qwen3ForCausalLM` es ambas), y el modelo falla al cargar. `scaffold.py` imprime el comando de servido con el flag. El endpoint es `/v1/embeddings`, y toma `input`:

```bash
pixi run max serve --model <HF_MODEL_ID> \
  --custom-architectures <port_dir> --task embeddings_generation

curl -s http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "<HF_MODEL_ID>", "input": ["first text", "second text"]}'
```

## Leer el resultado del primer servido

- **Caída en carga** → config, imports o adaptador de pesos.
- **Tokens basura** → carga [`debug-model`](../../debug-model/SKILL.md). El módulo puede seguir implementando la matemática del donante o un delta latente.
- **Salida corta plausible** → ejecuta `max_tokens=64+` antes de confiar en ella.

## Iterar en arreglos

Cada arreglo cuesta un servido y su compilación: haz un arreglo por servido, y detén el servidor entre ejecuciones con `pkill -f "max serve"`. Cuando los logits diverjan o la salida sea basura, deja de iterar con taps escalares y carga [`debug-model`](../../debug-model/SKILL.md). Usa [layer-by-layer-debugging.md](layer-by-layer-debugging.md) solo para la sonda rápida `compare_layers.py`.
