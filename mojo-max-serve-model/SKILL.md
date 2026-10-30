---
name: mojo-max-serve-model
description: >
  Sirve un modelo con el comando `max serve` de MAX: configura el entorno (pixi
  o uv con el canal conda max-nightly / índice de wheels nightly), apunta el
  servidor a un repo de Hugging Face o a un checkpoint local, selecciona una
  arquitectura personalizada con `--custom-architectures`, y elige los flags
  de servido adecuados para el modelo. Úsala siempre que el usuario quiera
  ejecutar, lanzar, arrancar u hospedar un modelo en MAX, levantar un endpoint
  compatible con OpenAI, servir una arquitectura personalizada o portada,
  depurar un fallo de arranque de `max serve`, o averiguar qué flags de servido
  (devices, quantization-encoding, max-length, task, trust-remote-code)
  necesita un modelo dado, aunque no diga "max serve" explícitamente.
compatibility: Requiere una instalación de MAX (nightly) con pip o pixi y un repo de Hugging Face o un checkpoint local; sirve en GPU por defecto, CPU para checkpoints float32.
---

# Sirve un modelo con MAX

`max serve` lanza un servidor HTTP compatible con OpenAI para un modelo. Gestiona la tokenización, el batching, la caché KV y todo el stack de servido. Le apuntas a un checkpoint y, si el modelo no está integrado en MAX, a un paquete de arquitectura personalizada. Esta skill te lleva de "sin entorno" a "servidor respondiendo peticiones", y te ayuda a elegir flags que encajen con el modelo específico en lugar de adivinar.

El principio rector: **empieza por el comando más pequeño que pueda funcionar, y añade flags solo cuando el modelo o el hardware te obliguen**. MAX auto-detecta la mayoría de las cosas (dtype, longitud de secuencia, dispositivo por defecto). Especificar flags en exceso es la forma más común de convertir un servido que funciona en uno roto.

**Usa esta skill cuando** quieras ejecutar, lanzar u hospedar un modelo en MAX: levantar un endpoint compatible con OpenAI, servir una arquitectura integrada o personalizada/portada, o depurar un fallo de arranque de `max serve`.

**No uses esta skill cuando** el modelo aún no esté implementado en MAX (sin `arch.py`, grafo y pesos funcionales). Eso es una tarea de *bring-up*: usa `mojo-max-import-model` para portar la arquitectura, y `mojo-max-debug-model` si se sirve pero la salida es incorrecta. Esta skill ejecuta un modelo existente; no lo crea.

## Referencias

| Archivo | Léelo cuando |
|---------|--------------|
| [references/custom-arch.md](references/custom-arch.md) | Sirvas una arquitectura personalizada: el mapeo de `arch.py` a flags, encoding y dispositivo, y las trampas del servido |
| [references/flags.md](references/flags.md) | Elijas cualquier flag de servido más allá de `--model`, `--devices`, `--quantization-encoding` y `--max-length` |
| [references/troubleshooting.md](references/troubleshooting.md) | Un fallo de arranque de `max serve` o un error críptico |

Lee la referencia de lo que estés haciendo, no todas de golpe.

## Ruta rápida (arquitectura personalizada): haz esto primero

Si MAX ya está instalado y tienes un paquete de arquitectura personalizada funcional, este es todo el trabajo en cuatro llamadas. No leas `arch.py` y `config.json` a mano ni razones sobre los flags tú mismo. El inspector incluido hace exactamente eso e imprime un comando listo para ejecutar más el razonamiento:

```bash
# 1. Obtén el comando recomendado + notas (lee arch.py + config.json por ti).
python <skill>/scripts/suggest_serve_command.py \
  --custom-architectures /abs/path/to/my_arch --model <hf-repo-or-path>

# 2. Lánzalo (añade `pixi run` si estás en un proyecto pixi). En una máquina REMOTA,
#    envuélvelo con `setsid ... </dev/null` para que sobreviva a la sesión SSH:
setsid <the suggested command> </dev/null > /tmp/max-serve.log 2>&1 &

# 3. Espera la disponibilidad en UNA llamada (falla rápido ante un crash; ~10 min de
#    presupuesto para una compilación en frío; los heartbeats `Still compiling` que
#    avanzan son normales):
timeout 600 bash -c 'until grep -qE "Server ready|Uvicorn running" /tmp/max-serve.log; do
  grep -qiE "Traceback|CRASHED|Error building|cannot be found|not found in registry" /tmp/max-serve.log && { echo SERVE_FAILED; tail -30 /tmp/max-serve.log; exit 1; }
  sleep 3; done' && echo SERVE_READY

# 4. Confirma con una petición (el campo model debe coincidir con --served-model-name):
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"my_arch","messages":[{"role":"user","content":"The capital of France is"}],"max_completion_tokens":32}'
```

**Lee las `# notes:` del inspector**. Ahí es donde afloran las trampas (un `default_encoding` que no concuerda con el checkpoint, un `name` que colisiona con un integrado, encodings solo para GPU, MoE). Si el comando funciona y la salida es coherente, has terminado. Solo baja a los pasos detallados de abajo cuando una nota o un fallo te lo indiquen. El resto de este documento es el "porqué" detrás de lo que hace el inspector y qué hacer cuando no es suficiente.

## 1. Asegúrate de que MAX está instalado

El usuario necesita un binario `max` de la build **nightly**. Compruébalo primero, y no reinstales si ya está ahí (el comando `max serve` funciona solo si el proyecto incluye las dependencias extra `max[serve]` o `max-serve`):

```bash
max serve --help           # ¿ya estás en un entorno MAX?
pixi run max serve --help   # o dentro de un proyecto pixi
```

Si el comando `max serve` no está disponible, configura un entorno. **pixi** es el predeterminado; el detalle clave es el canal conda `https://conda.modular.com/max-nightly/` más `conda-forge`. Si `pixi` no está instalado, instálalo usando las instrucciones más recientes en <https://pixi.prefix.dev/latest/#installation>, luego:

```bash
# pixi (canales conda)
pixi init my-max-project \
  -c https://conda.modular.com/max-nightly/ -c conda-forge && cd my-max-project
pixi add max-serve
```

Si ya existe un `pixi.toml`, la línea de canales debe leerse exactamente:

```toml
[workspace]                                # o [project] en pixi antiguo
channels = ["https://conda.modular.com/max-nightly/", "conda-forge"]
```

Algunos usuarios prefieren **uv**, que toma MAX del índice de wheels de Modular en lugar de conda. Si `uv` no está instalado, instálalo usando las instrucciones más recientes en <https://docs.astral.sh/uv/getting-started/installation/>, luego:

```bash
# uv (wheels pip)
uv init my-max-project && cd my-max-project
uv venv && source .venv/bin/activate
uv add "max[serve]" --index https://whl.modular.com/nightly/simple/ --prerelease allow
```

Después de una configuración pixi, prefija los comandos con `pixi run` (o entra en `pixi shell`). Después de uv, activa el venv (`source .venv/bin/activate`) y llama a `max` directamente. El resto de esta skill escribe `max serve ...` a secas; añade `pixi run` delante cuando estés en un proyecto pixi y no hayas entrado en la shell.

## 2. ¿Es un modelo integrado o una arquitectura personalizada?

MAX incluye muchas arquitecturas. Si la arquitectura del modelo ya está soportada, no necesitas `--custom-architectures` en absoluto, solo `--model`.

```bash
max list                 # cada arquitectura registrada + IDs de repos de ejemplo
```

Haz coincidir el `config.json::architectures[0]` del checkpoint (por ejemplo `LlamaForCausalLM`, `Qwen2ForCausalLM`) contra esa lista.

- **Listado**: integrado. Salta al paso 4 y omite `--custom-architectures`.
- **No listado**: necesitas un paquete de arquitectura personalizada (paso 3). Si el usuario aún no tiene uno, esta skill no puede fabricarlo; eso es una tarea de *bring-up* (implementar `arch.py`, `model.py`, el grafo, los adaptadores de pesos). Remítelo al flujo de trabajo de bring-up de modelos y detente aquí.

## 3. Apunta a una arquitectura personalizada

Una arquitectura personalizada es un **paquete** de Python (un directorio con `__init__.py`) que expone una lista `ARCHITECTURES` de nivel superior de instancias de `SupportedArchitecture`. Pasas el paquete con `--custom-architectures`; MAX lo importa, registra cada arquitectura por `name`, y en cada petición hace coincidir el `config.json::architectures[0]` del checkpoint contra un `name` registrado.

**Lee el paquete antes de escribir el comando**. El paquete de arquitectura es la fuente de verdad para la mayoría de los flags, así que no los adivines. Abre dos archivos:

- **`arch.py`**: la llamada `SupportedArchitecture(...)`. Su `default_encoding` es el encoding con el que servir; `supported_encodings` es el conjunto al que debe pertenecer tu `--quantization-encoding`; `name` debe coincidir con el checkpoint; `multi_gpu_supported` dice si puedes shardear; un `chat_template.jinja` en el paquete significa que necesitarás `--chat-template`.
- **`config.json`** (el checkpoint): `architectures[0]` debe ser igual al `name` de `arch.py`; `max_position_embeddings` limita `--max-length`; `torch_dtype` / `quantization_config` deberían concordar con `default_encoding`; los campos MoE (`num_experts`, etc.) indican que probablemente quieras `--device-graph-capture`.

**`references/custom-arch.md` es la guía detallada**: un mapeo campo a campo de `arch.py` a flags, la tabla de encoding a dispositivo, y las trampas no obvias del servido (overlap-scheduler vs logprobs, la colisión del puerto 8001 con métricas, trust-remote-code). Léela siempre que sirvas una arquitectura personalizada.

Apunta `--custom-architectures` al **directorio** del paquete (una ruta absoluta es lo más seguro para scripts y hosts remotos). La forma con dos puntos `IMPORT_PATH:MODULE_NAME` también funciona, pero la ruta al directorio es lo que usa la mayoría del tooling:

```bash
max serve --model <hf-repo-or-path> --custom-architectures /abs/path/to/my_arch
```

Ejecuta estas tres comprobaciones por adelantado en lugar de leer un stack trace; evitan casi todos los fallos de servido de arquitecturas personalizadas:

1. `name=` en la `SupportedArchitecture` es **exactamente** igual a `config.json::architectures[0]`.
2. `supported_encodings` incluye el encoding que el checkpoint realmente trae (un checkpoint bf16 necesita `bfloat16`, un checkpoint GPTQ necesita `gptq`, etc.).
3. `weight_adapters` tiene una entrada para el formato de pesos del checkpoint (`WeightsFormat.safetensors` para `.safetensors`, `WeightsFormat.gguf` para `.gguf`).

Si el modelo **no** es ya un paquete de arquitectura personalizada funcional (el grafo, los adaptadores de pesos y la configuración no están implementados), eso es una tarea de **bring-up**, no de servido. Esta skill sirve un paquete existente; no lo crea.

## 4. Construye el comando de servido (por defecto primero)

Empieza con el comando mínimo y ejecútalo. Deja que MAX auto-detecte el resto.

```bash
max serve --model <hf-repo-or-path> [--custom-architectures ...]
```

Eso ya enlaza `0.0.0.0:8000`, sirve en la GPU cuando hay una presente (CPU en caso contrario), infiere el dtype del checkpoint, y establece `max_length` desde `max_position_embeddings`. La GPU es el valor por defecto, así que no pasas `--devices` para el caso común de una sola GPU. Para muchos modelos en una sola GPU, ese comando mínimo es todo el trabajo.

Añade flags solo por una razón concreta. Los tres que más usarás:

| Flag | Añádelo cuando | Ejemplo |
|------|----------------|---------|
| `--devices` | Debas fijar GPUs específicas, shardear entre GPUs, o forzar CPU (la GPU ya es el valor por defecto). | `--devices gpu:0` · `--devices gpu:0,1,2,3` · `--devices gpu:all` · `--devices cpu` |
| `--quantization-encoding` | El repo tenga múltiples formatos, o la auto-detección elija el incorrecto. | `--quantization-encoding bfloat16` |
| `--max-length` | Quieras un contexto más corto que el máximo del modelo (ahorra memoria KV) o necesites limitarlo para que quepa. | `--max-length 4096` |

Para todo lo demás (memoria de dispositivo, tamaño de batch, selección de tarea, ventana deslizante, trust-remote-code, paralelismo multi-GPU, decodificación especulativa) consulta **`references/flags.md`**. Léelo antes de añadir cualquier flag del que no estés seguro; explica qué hace cada uno y cuándo *no* establecerlo.

**Cómo decidir, en orden**:

1. **Por defecto primero**. Prueba el comando mínimo. La auto-detección suele ser correcta para los integrados, y la GPU es el dispositivo por defecto.
2. **Para una arquitectura personalizada, lee el paquete**. El `default_encoding` de `arch.py` es tu `--quantization-encoding`. Ese encoding restringe el dispositivo: fp8, fp4 y gptq son solo para GPU, y la GPU ya es el valor por defecto, así que no añades `--devices` para ellos (ver `references/custom-arch.md`). Limita `--max-length` a `config.json::max_position_embeddings`. Añade `--trust-remote-code` si el checkpoint trae archivos de modelado personalizados, y `--chat-template` si el paquete incluye uno. Estas no son adivinanzas; las lees del paquete y la configuración.
3. **Inspecciona más cuando una elección sea determinante**. Para propiedades inusuales (ventana deslizante, RoPE parcial, enrutamiento MoE), revisa `config.json` y mapea los hallazgos a flags con `references/flags.md`.
4. **Pregunta cuando no estés seguro**. Si una decisión depende de algo que no puedes ver (qué GPUs están libres, cuánto contexto necesitan, CPU vs GPU), pregunta al usuario en lugar de adivinar. Un `--devices` o dtype incorrecto falla lenta y confusamente; una pregunta rápida es más barata.

## 5. Lanza y confirma que funciona

Lanza (en segundo plano, con un log que puedas seguir):

```bash
max serve --model <hf-repo-or-path> [flags] > /tmp/max-serve.log 2>&1 &
```

Si lanzas en una **máquina remota por SSH**, un `&` simple muere cuando la sesión SSH se cierra, y la compilación puede durar más que tu conexión. Sepáralo completamente:

```bash
setsid max serve --model <hf-repo-or-path> [flags] </dev/null > /tmp/max-serve.log 2>&1 &
```

Espera la disponibilidad con una sola llamada que vigila el heartbeat del log y falla rápido ante un crash en lugar de bloquearse durante todo el timeout:

```bash
timeout 600 bash -c 'until grep -qE "Server ready|Uvicorn running" /tmp/max-serve.log; do
  grep -qiE "Traceback|CRASHED|Error building|cannot be found|not found in registry" /tmp/max-serve.log && { echo SERVE_FAILED; tail -30 /tmp/max-serve.log; exit 1; }
  sleep 3; done' && echo SERVE_READY
```

El servidor imprime esta línea cuando está listo:

```output
Server ready on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Los modelos grandes compilan en el primer lanzamiento. Un silencio con heartbeats `Still compiling model (Ns elapsed)` es normal, no un cuelgue: el contador de tiempo transcurrido es la señal de vivacidad. Espera mientras avance, y solo trata el servido como atascado si el contador se congela (o el mtime del log deja de moverse mientras el proceso está vivo).

Confirma la salud, luego envía una petición real:

```bash
curl -s http://localhost:8000/v1/health          # 200 cuando está listo
curl -s http://localhost:8000/v1/models           # nombre del modelo servido

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<hf-repo-or-path>",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "max_completion_tokens": 32
  }'
```

El campo `model` de la petición debe coincidir con lo que pasaste a `--model` (o `--served-model-name` si lo sobrescribiste). Lee la respuesta: un bring-up no está terminado solo porque el servidor esté en verde. Comprueba que el texto es realmente coherente, no repetición ni basura. Si se sirve pero la salida es incorrecta, eso es un problema de paridad/corrección, no de servido.

## Caché de pesos y compilación para re-servidos más rápidos

El primer servido de un modelo descarga sus pesos y compila el grafo, que es la parte lenta. Ambos resultados se cachean, así que los servidos posteriores del mismo modelo son mucho más rápidos.

- **Los pesos** se descargan a través de `huggingface_hub` en la caché compartida de Hugging Face (`~/.cache/huggingface` por defecto; establece `HF_HOME` para reubicarla). Re-servir el mismo repo reutiliza los pesos cacheados sin re-descarga.
- **La compilación** también se cachea. Para calentar ambas cachés por adelantado (antes de una demo o despliegue) de modo que el primer `max serve` se salte la descarga y la espera de `Still compiling`, ejecuta `max warm-cache` primero:

  ```bash
  max warm-cache --model <hf-repo-or-path> [--custom-architectures /abs/path/to/my_arch]
  ```

  El artefacto compilado (MEF) es específico de la plataforma, así que calienta la caché en el mismo hardware (o pasa `--target`, por ejemplo `cuda:sm_90`, para compilar para un objetivo de despliegue desde un host diferente).

## Cambios de versión que afectan a esta skill

| Cambio | Versión | Impacto |
|--------|---------|---------|
| `--model` estandarizado en `max serve`, `max generate`, `max encode`, `max warm-cache` | v26.3 | Usa siempre `--model`; `--model-path` queda obsoleto |
| Prefijo de variables de entorno `MODULAR_MAX_SERVE_*` | v26.3 | Las variables sin prefijo ya no se aplican |
| `--rejection-sampling-strategy` eliminado | v26.7 | No lo uses; la decodificación especulativa se configura de otra forma |
| `--prefer-module-v3` para servir arquitecturas ModuleV3 | v26.7 | Permite seleccionar la variante ModuleV3 de una arquitectura registrada |
| Guía de plantillas de chat (`--chat-template`) | v26.7 | Documentación dedicada para el flag existente |
| Mejoras de programación (KVCache watermarks) | v26.1 | Afecta al rendimiento observado; no cambia los flags |

## Solución de problemas

Haz coincidir el síntoma con **`references/troubleshooting.md`**. Cubre los fallos de arranque que parecen crípticos pero tienen soluciones de una línea (desajuste de encoding, desajuste de nombre de `architectures[0]`, `trust_remote_code`, OOM al cargar, puerto en uso, enrutamiento de dispositivo incorrecto). Lee primero el log de servido; el error real suele estar unas líneas **por encima** del traceback final.
