# Servir una arquitectura personalizada: lee el paquete, luego establece los flags

Antes de escribir un comando `max serve`, lee dos archivos: el `arch.py` del paquete (la `SupportedArchitecture`) y el `config.json` del checkpoint. Entre ellos determinan la mayoría de los flags: el encoding, el dispositivo, la longitud máxima, y si necesitas trust-remote-code o una plantilla de chat.

## Encuentra el paquete

El paquete es el directorio cuyo `__init__.py` exporta una lista `ARCHITECTURES` de nivel superior. Si aún no tienes la ruta, búscala con grep:

```bash
grep -rl 'ARCHITECTURES' <repo-or-dir>
```

Abre el `arch.py` de ese directorio (la llamada `SupportedArchitecture(...)`) y el `config.json` del checkpoint; esos dos archivos guían los flags de abajo. MAX importa el paquete, registra cada entrada de `ARCHITECTURES` por `name`, y hace coincidir el `config.json::architectures[0]` de una petición contra un `name` registrado. Un paquete puede registrar varias arquitecturas (por ejemplo una variante de visión y una variante causal solo de texto); MAX elige la que coincide con el checkpoint.

## Lee `arch.py` → deriva flags

La llamada `SupportedArchitecture(...)` es una especificación de flags. Mapea sus campos:

| Campo de `arch.py` | Impulsa | Cómo |
|---|---|---|
| `name=` | qué checkpoint sirve | debe ser igual a `config.json::architectures[0]` **exactamente** |
| `default_encoding=` | `--quantization-encoding` | este es el dtype que trae el Hub; úsalo a menos que tengas una razón para no hacerlo |
| `supported_encodings={...}` | valores válidos de `--quantization-encoding` | lo que pases debe estar en este conjunto, o el servido lo rechaza |
| `multi_gpu_supported=` | si `--devices gpu:0,1,...` está permitido | `False` → solo una GPU |
| `task=` | `--task` | solo pasa `--task` si el mismo `name` está registrado para múltiples tareas |
| `weight_adapters={...}` | qué formatos de pesos cargan | necesita una entrada para el formato del checkpoint (`safetensors`/`gguf`) |
| `requires_max_batch_context_length=True` | Modelos VL | indica que probablemente necesitarás `--max-length` establecido explícitamente |

Un `chat_template.jinja` en el directorio del paquete significa que el modelo necesita `--chat-template path/to/chat_template.jinja` para que las peticiones de chat se formateen correctamente.

## Lee `config.json` → comprueba y limita

- `architectures[0]`: debe coincidir con el `name` de `arch.py`. Un desajuste es el fallo de servido de arquitectura personalizada número 1.
- `torch_dtype` / `quantization_config`: la precisión real del checkpoint. Confirma que `default_encoding` concuerda (un checkpoint fp8 con un `default_encoding` bfloat16 es una bandera roja).
- `max_position_embeddings` (o `n_positions`): el **techo** para `--max-length`. Pasar un valor mayor hace que `max serve` aborte. Los checkpoints de prueba pequeños a menudo declaran 512; limita siempre a `min(lo_que_quieras, esto)`.
- Señales MoE (`num_experts`, `num_local_experts`, `n_routed_experts`): un modelo MoE enrutado se beneficia de `--device-graph-capture` y normalmente quiere un `--max-length` mayor.

## Encoding y dispositivo

MAX sirve en GPU por defecto, así que para el caso común no estableces `--devices` en absoluto. El encoding solo restringe lo que es posible:

| Encoding | Se ejecuta en |
|---|---|
| `float32` | CPU (pasa `--devices cpu`) o GPU |
| `bfloat16`, `float16` | GPU (el valor por defecto) |
| `float8_e4m3fn`, `float8_e5m2`, `float4_e2m1fnx2`, `gptq` | Solo GPU |

MAX no elige el dispositivo a partir del encoding; usa la GPU por defecto en cualquier caso, que es lo que necesitan los encodings solo para GPU. Recurre a `--devices` solo para forzar CPU (un checkpoint `float32` con la RAM para sostenerlo), fijar GPUs específicas, o shardear entre ellas.

## Los flags que no son obvios a partir del modelo

Estos vienen de *cómo sirve MAX*, no del checkpoint. Mordeen repetidamente en los bring-ups de arquitecturas personalizadas:

- **`--no-enable-overlap-scheduler --force`**: MAX auto-habilita el overlap scheduler para decodificación de un solo paso, y esa ruta es incompatible con peticiones de `logprobs` (que envían las herramientas de paridad/verificación). Si vas a pedir logprobs, desactívalo; el `--force` es necesario para omitir la heurística de auto-habilitación. Para servido de chat normal puedes dejarlo activado.
- **`--trust-remote-code`**: necesario cuando el checkpoint trae `modeling_*.py` / `configuration_*.py` personalizados (por ejemplo `internlm2`, `openelm`). El loader de MAX también necesita el `auto_map` del repo en `config.json` *y* esos archivos `.py` presentes en el directorio del modelo. Sin el flag obtienes "Transformers does not recognize this architecture" al cargar.
- **No sirvas en `--port 8001`.** La app ASGI de métricas usa 8001 por defecto y se enlaza *antes* que el servidor principal de API. Si el servidor principal también recibe 8001, el log imprime `Server ready` y luego inmediatamente `Shutting down workers…` sin error, y cada petición falla con un confuso error de decodificación JSON. Usa 8000 u otro puerto; si ejecutas múltiples servidores, mueve también el puerto de métricas con la variable de entorno `MODULAR_MAX_SERVE_METRICS_ENDPOINT_PORT`.

## Un comando de arquitectura personalizada resuelto

Para un decodificador denso bf16 en una GPU, apuntando `--custom-architectures` al directorio del paquete (la ruta absoluta es lo más seguro para scripts y hosts remotos):

```bash
max serve \
  --model <hf-repo-or-local-ckpt> \
  --custom-architectures /abs/path/to/my_arch \
  --devices gpu:0 \
  --quantization-encoding bfloat16 \
  --max-length 4096 \
  --trust-remote-code \
  --served-model-name my_arch \
  --port 8000
```

Para un modelo MoE enrutado, añade `--device-graph-capture` y un `--max-length` mayor; para un modelo con una plantilla incluida, añade `--chat-template /abs/path/to/my_arch/chat_template.jinja`. Tanto la ruta al directorio simple como la forma con dos puntos `IMPORT_PATH:MODULE_NAME` funcionan para `--custom-architectures`; la ruta al directorio es lo que usa la mayoría del tooling.

## Disponibilidad en una compilación lenta (MoE / grande)

Las arquitecturas personalizadas a menudo compilan más tiempo que las integradas. Vigila el log para `Still compiling model (Ns elapsed)`. Un contador que avanza es saludable; solo un contador *congelado* (o un log cuyo mtime deja de moverse mientras el proceso está vivo) es un estancamiento real. Da a los modelos grandes un timeout generoso (10+ minutos) antes de tratar una compilación como colgada.
