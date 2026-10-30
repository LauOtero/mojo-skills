# Construir el comparador de tensores por capa

Construir el comparador lleva unos 45 minutos. Cada comparación posterior lleva segundos.

La referencia no tiene que ser `AutoModelForCausalLM`. Cualquier modelo de PyTorch al que puedas enganchar forward hooks funciona igual. Eso incluye un archivo de modelado con `trust_remote_code=True`, o (al depurar una variante cuantizada o multi-GPU) los volcados de referencia de tu propio port MAX ya verificado del bring-up original.

Establece los directorios de volcado mediante variables de entorno o argumentos del script. Usa las mismas rutas en el volcador HF, el volcador MAX y el comparador.

## Artefacto 1: Volcador HF

`hf_layer_dump.py` carga la referencia HF, ejecuta UN prefill y vuelca los estados ocultos en cada checkpoint a disco como numpy FP32.

```python
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "<hf_model_id>"
OUT_DIR = os.environ.get("HF_DUMP_DIR", "./parity_dumps/hf_layers")
os.makedirs(OUT_DIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(REPO)
# Carga la referencia en float32 cuando quepa: el redondeo bfloat16 en la
# referencia aparece como deriva de capas que no está en el port.
model = AutoModelForCausalLM.from_pretrained(
    REPO,
    dtype=torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()

backbone = model.model  # ajústalo para arquitecturas no solo-decodificador
# Confirma que este es el stack decodificador, no un wrapper multimodal/encoder:
# los hooks en el módulo equivocado vuelcan tensores sin contraparte en MAX.
print("backbone:", type(backbone).__name__)


def save(name):
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        np.save(f"{OUT_DIR}/{name}.npy", h[0].detach().float().cpu().numpy())

    return hook


backbone.embed_tokens.register_forward_hook(save("post_embed"))
for i, layer in enumerate(backbone.layers):
    layer.register_forward_hook(save(f"layer_{i:02d}"))
backbone.norm.register_forward_hook(save("post_final_norm"))

text = tok.apply_chat_template(
    [{"role": "user", "content": "Hello!"}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**inputs, use_cache=False)

np.save(f"{OUT_DIR}/last_logits.npy", out.logits[0, -1].float().cpu().numpy())
np.save(f"{OUT_DIR}/prompt_tokens.npy", inputs.input_ids[0].cpu().numpy())
print("done", OUT_DIR)
```

Tiempo de ejecución: ~2 a 3 min de carga del modelo + ~1 s de prefill.

## Artefacto 2: Volcador MAX

Este volcador tiene taps en `forward()` para exponer los estados ocultos y un runner independiente que evita `max serve` para capturar cada salida del modelo.

### Taps en `forward()` (en el archivo del modelo de tu port)

Detrás de un flag de entorno, recoge una copia FP32 del estado oculto después del embedding, después de cada capa decodificadora y después de la norma final. Devuelve esos tensores como salidas extra del `forward()` del módulo raíz:

```python
import os

from max.dtype import DType
from max.experimental.tensor import Tensor

_DUMP = os.environ.get("PORT_DUMP") == "1"


# dentro del forward() del módulo raíz:
dump_tensors: list[Tensor] = []


def _tap(t: Tensor) -> None:
    if _DUMP:
        dump_tensors.append(t.cast(DType.float32))


h = self.embed_tokens(tokens)
_tap(h)  # post-embed
for idx, layer in enumerate(self.layers):
    h = layer(...)
    _tap(h)  # layer_00, layer_01, ...
h = self.norm(h)
_tap(h)  # post-final-norm

return (last_logits, *dump_tensors) if _DUMP else (last_logits,)
```

El flag se lee en tiempo de trazado, así que compila con `PORT_DUMP=1` establecido. Mantén la lista local a `forward()` para que un segundo trazado empiece vacío.

**Castea a FP32 en `forward()`.** `np.from_dlpack` falla con BF16. El casteo dentro del modelo permite que el volcador use numpy plano.

En una malla multi-GPU, el estado oculto es un `Tensor` distribuido. Tapa un tensor replicado, o reúnelo primero (`F.allgather(t, tensor_axis=0)` para uno fragmentado por secuencia), y el runner lee el shard 0.

Si el port comparte subgrafos de capas (`as_subgraph`), mantén los subgrafos activados en modo volcado. Los taps se sitúan entre las llamadas de capa y no cambian la firma del subgrafo.

### Volcador independiente

Este runner usa APIs internas del pipeline que cambian entre releases de MAX. Si una importación o atributo de abajo falla, busca el símbolo en tu paquete `max.pipelines` instalado y ajústalo. Los pasos se mantienen entre releases: construye un contexto, asigna la caché KV para él, prepara las entradas de tokens y llama al modelo compilado.

```python
import os
import numpy as np

from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.context import SamplingParams, TextContext, TokenBuffer
# Registra tu arquitectura igual que lo harías para max serve.

_tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(your_pipeline_config)
pipeline_model = pipeline._pipeline_model
inner_model = pipeline_model.model  # el ModuleV3 compilado invocable
kv_manager = pipeline.kv_manager

ctx = TextContext(
    max_length=n_tokens + 1,
    tokens=TokenBuffer(np.asarray(prompt_ids, dtype=np.int64)),
    sampling_params=SamplingParams(max_new_tokens=1),
)
replica_batches = [[ctx]]

kv_manager.claim(ctx)
kv_manager.alloc(ctx)
kv_inputs = kv_manager.runtime_inputs(replica_batches)
model_inputs = pipeline_model.prepare_initial_token_inputs(
    replica_batches=replica_batches,
    kv_cache_inputs=kv_inputs,
    return_n_logits=1,
)
# Llama al modelo compilado directamente. pipeline_model.execute elimina
# todo excepto los logits. No parchees el invocable compilado.
outs = inner_model(*model_inputs.buffers)
kv_manager.release(ctx)

NAMES = (
    ["last_logits", "post_embed"]
    + [f"layer_{i:02d}" for i in range(n_layers)]
    + ["post_final_norm"]
)
out_dir = os.environ.get("MAX_DUMP_DIR", "./parity_dumps/max_layers")
os.makedirs(out_dir, exist_ok=True)
for name, o in zip(NAMES, outs):
    # Las salidas son Tensors. Una salida distribuida lee el shard 0.
    shard = o.local_shards[0] if o.is_distributed else o
    arr = shard.to_numpy().copy()
    if name == "last_logits":
        arr = arr[0]  # [1, vocab] a [vocab], la forma que guarda el volcador HF
    np.save(f"{out_dir}/{name}.npy", arr)
np.save(f"{out_dir}/prompt_tokens.npy", np.asarray(prompt_ids))
print("done", out_dir)
```

Ejecuta: `PORT_DUMP=1 pixi run python dump_max_layers.py`

El tiempo de compilación en frío coincide con la compilación en frío de serve (~5 a 25 min según el modelo). El volcado en sí tarda menos de un segundo.

### Inspección rápida con `PrintHook`

Los volcadores de tensores de arriba guardan las activaciones de cada capa a disco para diferenciarlas offline. Para una mirada más rápida que no necesita editar el modelo, `max.nn.hooks.PrintHook` imprime las entradas y salidas de cada capa mientras el modelo se ejecuta:

```python
from max.nn.hooks import PrintHook

hook = PrintHook()
hook.name_layers(model)  # nombra cada Module por su ruta de atributo
# compila y ejecuta el modelo
hook.remove()
```

`PrintHook` se engancha a los árboles de `Module` de MAX. Cuando la referencia es otra implementación MAX del mismo modelo, añade el mismo hook a ambas y ejecútalas. Compara los valores de capa impresos para encontrar la primera que discrepe. Cuando la referencia es un modelo de `transformers`, engánchala con los forward hooks de PyTorch del Artefacto 1. Para tensores multi-dispositivo (distribuidos), `F.print(value, name)` de `max.experimental.functional` imprime cada shard con su dispositivo.

`PrintHook` imprime en consola, así que úsalo para triaje rápido. Para comparación de coseno en una ejecución completa, usa el comparador de volcados guardados de arriba. Para las otras opciones de depuración integradas de MAX, consulta [las herramientas de depuración de MAX](https://max.modular.com/develop/debugging/).

> **Novedad (v26.3)**: la guía de *layer comparison* ahora recomienda `PrintHook` junto con la exportación de tensores a `.max` mediante `InferenceSession`. `PrintHook` se instancia antes de construir el grafo para que las ops `print` formen parte del grafo.

## Artefacto 3: Comparador

```python
import numpy as np, os, glob

HF = os.environ.get("HF_DUMP_DIR", "./parity_dumps/hf_layers")
MX = os.environ.get("MAX_DUMP_DIR", "./parity_dumps/max_layers")


def cos(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


names = sorted(
    {os.path.basename(p)[:-4] for p in glob.glob(f"{HF}/*.npy")}
    & {os.path.basename(p)[:-4] for p in glob.glob(f"{MX}/*.npy")}
)

print(
    f"{'checkpoint':25s} {'cos_sim':>9s} {'mean_diff':>10s} {'max_diff':>10s}  spike_position"
)
for n in names:
    h = np.load(f"{HF}/{n}.npy")
    m = np.load(f"{MX}/{n}.npy")
    if h.shape != m.shape:
        print(f"{n:25s}  shape mismatch  hf={h.shape} max={m.shape}")
        continue
    diff = np.abs(h - m)
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    pos = ",".join(str(i) for i in idx)  # (token, dim), o (vocab,) para logits
    spike = f"({pos}) HF={h[idx]:+.2f} MAX={m[idx]:+.2f}"
    print(
        f"{n:25s} {cos(h, m):9.4f} {diff.mean():10.4f} {diff.max():10.4f}  {spike}"
    )
```

**Significado de las columnas:**

- `cos_sim`: similitud coseno del tensor completo (aplanado)
- `mean_diff`: `mean(|HF - MAX|)`, desacuerdo medio elemento a elemento
- `max_diff`: `max(|HF - MAX|)`, mayor desacuerdo de un solo elemento
- `spike_position`: el (token, dim) donde vive la diferencia máxima, más los valores HF y MAX allí

**Error de lectura común**: `max_diff` es desacuerdo elemento a elemento, no el máximo absoluto del tensor. Un `max_diff` grande donde HF tiene un pico y MAX es plano significa que HF formó un ancla que MAX no.

## Más allá del comparador por defecto

Añade cortes por token y por dimensión cuando la salida por defecto muestre divergencia:

```python
def per_token_cos(h, m):
    return [cos(h[t], m[t]) for t in range(h.shape[0])]


def per_dim_cos(h, m):
    return [cos(h[:, d], m[:, d]) for d in range(h.shape[1])]
```

## Volcados de paso de decodificación (fallo greedy en el índice K)

Cuando la verificación greedy falla en el índice de token K, vuelca con prefijo = tokens 0..K-1 y compara en `--token-index K`.

**HF:** extiende tanto `input_ids` como `attention_mask` al añadir tokens forzados por profesor:

```python
ids = tok(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
for tid in prefix_token_ids:
    ids = torch.cat([ids, torch.tensor([[tid]], device=ids.device)], dim=1)
mask = torch.ones_like(ids)
with torch.no_grad():
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
```

**MAX:** construye `TextContext` con la misma lista de tokens, y establece `return_n_logits=1` para los logits de la última posición solamente.

## Indexación de `hidden_states` de HF

**No** asumas que `output_hidden_states[i+1]` es la salida de la capa `i`. Algunos modelos guardan las entradas previas a la capa más una salida de norma final, no las salidas por capa. Usa forward hooks por capa cuando la semántica de tuplas no esté clara. Consulta [comparator-output-patterns.md](comparator-output-patterns.md).

Usa `add_special_tokens=False` cuando tu prompt de MAX no tenga BOS.

## Comprueba RoPE antes de depurar la atención

Para divergencia dependiente de la posición (t=0 coincide, t≥1 mal desde la capa 1):

```python
pos = torch.arange(seq_len, device=model.device).unsqueeze(0)
emb = model.model.rotary_emb(hidden_states, pos)
print("inv_freq max:", model.model.rotary_emb.inv_freq.abs().max().item())
cos, sin = emb
print("cos sample:", cos[0, -1, :4].tolist())
print("sin sample:", sin[0, -1, :4].tolist())
```

Si `inv_freq≈0` → cos≈1, sin≈0 → **NoPE en tiempo de ejecución**. Tu módulo MAX no debe aplicar RoPE plano cuando HF usa rotación identidad en tiempo de ejecución.

## Capas con estado (conv, SSM): volcados de decodificación incremental

Los volcados **solo de prefill** forzados por profesor no son suficientes cuando el modelo lleva estado entre pasos de decodificación (capas conv, bloques SSM, etc.). Un solo `execute()` con el prefijo greedy completo ejercita la matemática de prefill pero puede no coincidir con la ruta de decodificación **incremental** que usa tu verificación.

**Patrón requerido:**

1. Haz prefill del prompt de instrucción una vez.
2. Ejecuta **K pasos de decodificación de un solo token**, actualizando estado + KV en cada paso (refleja `prepare_next_token_inputs` y cualquier salida de estado).
3. Vuelca los estados ocultos en el paso K (prefijo forzado por profesor 0..K−1 **o** greedy autorregresivo de MAX en todo momento).

Cuando el modo volcado devuelva logits + tensores de tap + tensores de estado, documenta el orden de salida en tu `forward()`. Corta las salidas de estado por conteo, no por posición después de los logits solos.

## Extensiones del CLI del comparador

```python
parser.add_argument(
    "--token-index",
    type=int,
    default=-1,
    help="Compara solo la fila t (última fila si -1)",
)
parser.add_argument(
    "--skip",
    nargs="*",
    default=["last_logits"],
    help="Omite logits cuando las formas difieran por el eje de vocab",
)
```

Al comparar tensores de logits, omite el corte de tokens (vectores de vocab 1D).

## Valida los volcadores antes de depurar el port

1. Ejecuta ambos volcadores sobre un modelo que ya sepas que funciona en MAX. La similitud coseno debería ser ~0.999 en todo. Si no, corrige primero los volcadores.
2. Comprueba `prompt_tokens.npy` de ambos lados. Deben ser idénticos.
3. `post_embed.npy` debería tener cos=1.0 (dentro del redondeo FP). Si no, corrige la carga del embedding antes de comparar capas más profundas.
