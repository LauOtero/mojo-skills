# Flags de `max serve`: qué establecer, y cuándo NO

El comando por defecto (`max serve --model <repo>`) auto-detecta dispositivo, dtype y longitud de secuencia. Cada flag de abajo es algo que añades por una razón *específica*. El modo de fallo más común es especificar en exceso: un flag que lucha contra la auto-detección. En caso de duda, déjalo fuera y deja que MAX decida.

> **Cambios recientes**: `--model` está estandarizado en todas las CLI (v26.3). Las variables de entorno usan el prefijo `MODULAR_MAX_SERVE_*` desde v26.3. `--rejection-sampling-strategy` fue eliminado en v26.7.

## Contenido

- [Elegir desde el checkpoint (config.json)](#elegir-desde-el-checkpoint)
- [Dispositivo y memoria](#dispositivo-y-memoria)
- [Precisión / cuantización](#precisión--cuantización)
- [Longitud de secuencia y batching](#longitud-de-secuencia-y-batching)
- [Selección de tarea](#selección-de-tarea)
- [Trust remote code](#trust-remote-code)
- [Arquitecturas personalizadas](#arquitecturas-personalizadas)
- [Paralelismo multi-GPU](#paralelismo-multi-gpu)
- [Ventana deslizante / RoPE](#ventana-deslizante--rope)
- [Tiempo de compilación](#tiempo-de-compilación)
- [Flags de características de servido](#flags-de-características-de-servido)
- [Plantilla de chat](#plantilla-de-chat)
- [ModuleV3](#modulev3)

## Elegir desde el checkpoint

Cuando una elección de flag sea determinante, lee el `config.json` del checkpoint (en el Hub, o `~/.cache/huggingface/...` localmente) y mapea los campos a flags:

| Campo de `config.json` | Te dice | Flag que informa |
|---|---|---|
| `architectures[0]` | Qué arquitectura coincidir | debe ser igual a `SupportedArchitecture.name` |
| `torch_dtype` (`bfloat16`/`float16`/`float32`) | Precisión nativa | `--quantization-encoding` |
| `quantization_config` | Pre-cuantizado (gptq/awq/fp8) | `--quantization-encoding` |
| `max_position_embeddings` / `n_positions` | Contexto máximo | techo para `--max-length` |
| `sliding_window` | Atención con ventana | normalmente auto; `--sliding-window` para sobrescribir |
| `rope_scaling.rope_type` | Variante de RoPE | `--rope-type` (solo GGUF) |

Revisa también los `SupportedArchitecture.supported_encodings` del paquete de arquitectura: el `--quantization-encoding` que pases debe estar en ese conjunto, o el servido lo rechazará.

## Dispositivo y memoria

- `--devices`: el selector de dispositivo de primera clase. `cpu`, `gpu`, `gpu:all`, `gpu:0`, o una lista `gpu:0,1,2,3`. Omítelo para usar el valor por defecto del modelo/config. **No** establezcas también `CUDA_VISIBLE_DEVICES`; los dos se traducen de forma independiente y apilarlos enruta al dispositivo incorrecto.
- `--device-memory-utilization <0..1>`: fracción de la memoria libre del dispositivo que el proceso puede consumir; el resto contiene la caché KV (`kv_workspace = free_mem * util - weights`). Súbela hacia `0.9`–`0.95` en una GPU ocupada cuando necesites más espacio KV; bájala si compartes la GPU.

## Precisión / cuantización

Opciones de `--quantization-encoding`:
`float32 | float16 | bfloat16 | q4_k | q4_0 | q6_k | float8_e4m3fn | float4_e2m1fnx2 | gptq`.

- **Déjalo sin establecer** cuando el repo traiga un solo formato; MAX lo auto-detecta.
- **Establécelo** cuando el repo traiga varios (elige uno) o la auto-detección elija mal. Haz coincidir `torch_dtype`/`quantization_config` de `config.json`, y asegúrate de que el valor está en los `supported_encodings` de la arquitectura.
- El servido en CPU generalmente necesita `float32` (o un modelo con una huella factible en CPU); la GPU comúnmente `bfloat16`.
- `--kv-cache-format {float32|bfloat16|float8_e4m3fn}`: sobrescribe solo el dtype de la caché KV (memoria vs precisión), independientemente del encoding de pesos.

## Longitud de secuencia y batching

- `--max-length`: longitud máxima de secuencia. Por defecto `max_position_embeddings`; MAX puede recortarlo para que quepa en memoria. Establece un valor más pequeño para ahorrar memoria KV cuando no necesites el contexto completo.
- `--max-batch-size`: máximo de peticiones concurrentes en un batch. Auto cuando no se establece. Para un servidor real, establételo a tu concurrencia esperada; para una comprobación de corrección/latencia de un solo flujo, `1` mantiene las cosas simples.
- `--max-batch-input-tokens`, `--max-batch-total-tokens`: presupuestos de tokens objetivo/techo por batch; solo necesarios para tuning de throughput o prefill fragmentado.

## Selección de tarea

- `--task`: `text_generation` (por defecto para LLMs), `embeddings_generation`, `pixel_generation`. Solo necesario para desambiguar una arquitectura registrada bajo un nombre para múltiples tareas, o para servir endpoints de embeddings/imagen. Las rutas de OpenAI solo funcionan con una tarea compatible (por ejemplo `/v1/embeddings` necesita `embeddings_generation`).

## Trust remote code

- `--trust-remote-code`: requerido cuando el repo de HF trae Python personalizado (`modeling_*.py`, `configuration_*.py`, tokenizer personalizado) que el loader ejecuta. Si te encuentras con un error de `trust_remote_code` al cargar, añádelo. Úsalo solo con repos en los que confíes; ejecuta código arbitrario del repo.

## Arquitecturas personalizadas

- `--custom-architectures`: repetible. O bien un nombre de directorio de paquete (cuando ejecutas desde su padre) o `IMPORT_PATH:MODULE_NAME` (robusto, en cualquier ruta). El `__init__.py` del paquete debe exponer `ARCHITECTURES = [<SupportedArch>...]`. Un `name` personalizado que coincida con uno integrado tiene precedencia sobre el integrado.

## Paralelismo multi-GPU

Solo para modelos demasiado grandes para una GPU, o cuando escales deliberadamente. Para una prueba de latencia de un solo flujo, **menos GPUs suelen ser más rápidas**; las comunicaciones de tensor-parallel pueden dominar y perjudicar la latencia de batch-1. Recurre a estos en servido multi-GPU real:

- `--data-parallel-degree`: replica el modelo entre dispositivos.
- `--ep-size`: tamaño de expert-parallelism para MoE (1, o el número total de GPUs).
- `--pipeline-role {prefill_and_decode|prefill_only|decode_only}`: prefill/decode desagregados entre procesos.

## Ventana deslizante / RoPE

- `--sliding-window <N>`: fuerza una máscara causal de ventana deslizante de N tokens. Por defecto el `sliding_window` de la config, o atención causal completa. Sobrescribe solo si la config falta o es incorrecta.
- `--rope-type {none|normal|neox|longrope|yarn}`: fuerza una variante de RoPE. Solo importa para pesos **GGUF**; los modelos safetensors llevan esto en la config.

## Tiempo de compilación

- `--use-subgraphs / --no-use-subgraphs`: los subgrafos (por defecto **activados**) reducen drásticamente el tiempo de compilación para modelos grandes con muchos bloques idénticos. Déjalo activado. Desactivarlo (un resto de depuración) puede disparar el tiempo de compilación en un modelo profundo.
- `--device-graph-capture / --no-device-graph-capture`: captura+replay para ejecución; auto-habilitado para algunas arquitecturas. Desactívalo con `--no-device-graph-capture --force` si es necesario.

## Flags de características de servido

Añádelos sobre cualquier servido funcional; ninguno es necesario para empezar:

- `--enable-prefix-caching`: reutiliza KV entre peticiones que comparten prefijo; reduce TTFT en prompts repetidos.
- `--enable-chunked-prefill`: divide prefills largos en fragmentos (`--max-batch-input-tokens`).
- `--enable-structured-output`: acepta un esquema JSON en `response_format`.
- `--enable-lora` + `--lora-paths`: sirve adaptadores LoRA sobre un modelo base.
- `--temperature`, `--top-k`: valores por defecto de muestreo a nivel de servidor para peticiones que no establecen los suyos.
- `--served-model-name`: sobrescribe el nombre del modelo de cara al cliente (lo que los clientes ponen en el campo `model` de la petición); por defecto `--model`.
- `--port`: puerto HTTP (por defecto `8000`).

## Plantilla de chat

- `--chat-template <path>`: plantilla de chat personalizada para sobrescribir la que trae la config de Hugging Face. Si se proporciona una ruta, el archivo se lee durante la resolución de la config y su contenido se almacena como string. Si es `None`, se usa la plantilla por defecto del modelo. Nueva guía dedicada en v26.7.

## ModuleV3

- `--prefer-module-v3`: selecciona la variante ModuleV3 de una arquitectura registrada cuando existe. Introducido en v26.7.

## Dos trampas del servido que muerden a todos

No derivadas del modelo (vienen de cómo sirve MAX), pero causan síntomas confusos de "el servidor parecía bien, luego falló":

- **`--no-enable-overlap-scheduler --force` para logprobs.** MAX auto-habilita el overlap scheduler para decodificación de un solo paso, y esa ruta no puede servir peticiones de `logprobs` (las herramientas de paridad/verificación las envían). Desactívalo si necesitas logprobs; el `--force` omite la heurística de auto-habilitación. El servido de chat normal no lo necesita.
- **Nunca sirvas en `--port 8001`.** La app ASGI de métricas usa 8001 por defecto y se enlaza antes que el servidor principal de API. Si el servidor principal también cae en 8001, el log imprime `Server ready` y luego inmediatamente `Shutting down workers…` sin error y cada petición falla con un error de decodificación JSON. Usa 8000 u otro puerto; cuando ejecutes múltiples servidores, mueve también el puerto de métricas con la variable de entorno `MODULAR_MAX_SERVE_METRICS_ENDPOINT_PORT`.
