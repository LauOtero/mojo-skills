# Flags y datasets de `max benchmark`

`max benchmark` envuelve el script de código abierto `benchmark_serving.py` y acepta todas sus mismas opciones. Ejecuta `max benchmark --help` para la lista exhaustiva. Los grupos de abajo cubren lo que realmente se usa. Cada ejecución necesita un servidor en ejecución y `--num-prompts` o `--num-chat-sessions`.

## Contenido

- [Conexión al servidor](#conexión-al-servidor)
- [Generación de carga](#generación-de-carga)
- [Datasets](#datasets)
- [Longitud de salida y muestreo](#longitud-de-salida-y-muestreo)
- [Guardado de resultados](#guardado-de-resultados)
- [Estadísticas y perfilado](#estadísticas-y-perfilado)
- [Archivo de configuración](#archivo-de-configuración)

## Conexión al servidor

- **`--backend`**: tipo de servidor, uno de `modular` (por defecto), `modular-chat`, `vllm`, `vllm-chat`, `sglang`, `trtllm`, y las variantes `-chat`. Usa `modular` para un endpoint `max serve`.
- **`--model`**: un ID de Hugging Face o una ruta local. **Debe ser igual** al `--served-model-name` del servidor o las peticiones fallarán. Léelo de `curl /v1/models`.
- **`--endpoint`**: `/v1/completions` o `/v1/chat/completions` (por defecto `/v1/chat/completions`). Los LMs base usan `/v1/completions`, que no necesita plantilla de chat. Los modelos instruct y chat usan `/v1/chat/completions`, que sí necesita una plantilla de chat.
- **`--base-url`**: la URL base completa. Sobrescribe `--host` y `--port` cuando la estableces.
- **`--host`** y **`--port`**: por defecto `localhost:8000`.
- **`--tokenizer`**: el tokenizer de Hugging Face, que por defecto es `--model`. Si el nombre servido es un *alias* en lugar de un ID de Hugging Face, ese valor por defecto no puede resolverse, así que pasa el ID real del modelo aquí o la ejecución fallará con "not a valid model identifier".

## Generación de carga

- **`--num-prompts`**: el número de prompts de un solo turno. Obligatorio a menos que uses `--num-chat-sessions`. Más prompts estabilizan los promedios y alargan la ejecución.
- **`--num-chat-sessions`**: sesiones multiturno, para datasets multiturno y cargas de trabajo de tipo juez, en lugar de `--num-prompts`.
- **`--max-concurrency`**: el máximo de peticiones en vuelo. Acepta un valor único o un sweep (`1,2,4,8,16,32`). El `--max-batch-size` del servidor debe ser al menos el máximo del sweep, o los puntos más altos solo se encolan.
- **`--request-rate`**: peticiones por segundo. Acepta un valor único o un sweep, y por defecto es `inf` (sin límite). Usa un sweep de tasa para medir el comportamiento a cargas objetivo en lugar de saturar el servidor.
- **`--seed`**: la semilla del RNG de la carga de trabajo, por defecto `24301` (fija para reproducibilidad). `--seed none` extrae una nueva, luego la registra y la guarda.
- **`--workload-config`**: un archivo YAML de opciones de carga de trabajo con claves con guiones, ya sea planas en el nivel superior o anidadas bajo un bloque `native:` junto a `runner: native`. La CLI gana sobre el archivo.
- **`--fit-distributions`**: remodela cargas multiturno usando los flags `random_*` y `--delay-between-chat-turns`. Requiere `--num-chat-sessions` y un dataset compatible.
- **`--agentic-tool-profiles`**: añade un bucle de agente tras cada turno humano, sintetizado a partir de distribuciones por herramienta (no replay). Requiere `--fit-distributions` y `--agentic-rounds-per-turn`. Acepta un archivo YAML o un mapeo YAML/JSON en línea.
- **`--delay-between-chat-turns`**: retardo entre turnos de chat en milisegundos; acepta una constante o una distribución.

## Datasets

`--dataset-name` selecciona la carga de trabajo. Las opciones más útiles son las siguientes:

- **`random`**: sintético y totalmente controlable. Funciona mejor para micro-mediciones limpias y reproducibles. Se le da forma con estos flags:
  - `--random-input-len` (por defecto `1024`) y `--random-output-len` (por defecto `128`). Aceptan una constante o una distribución: `N(mean,std)`, `U(lo,hi)`, `DU(lo,hi)`, `NB(n,p)`, `G(shape,scale)`, o `LN(mean,std)`. Un `;` separa el primer turno de los siguientes.
  - `--random-sys-prompt-ratio`, `--random-max-num-unique-sys-prompt`, y `--warm-shared-prefix` impulsan experimentos de caché de prefijos.
  - `--random-image-count` y `--random-image-size` habilitan el modo visión.
- **`sharegpt`** (**por defecto**): conversaciones reales de humanos y IA del Hugging Face Hub. Funciona mejor para una mezcla de chat representativa.
- **`arxiv-summarization`**: summarización de contexto largo, dimensionada por `--arxiv-summarization-input-len` (por defecto `15000`).
- **`sonnet`**: poesía, dimensionada por `--sonnet-input-len` y `--sonnet-prefix-len`.
- **Datasets de código**: `instruct-coder`, `agentic-code` y `nemotron-opencode` llevan trazas de agentes y de llamadas a herramientas (se activan con `--tool-calls` o `--no-tool-calls`), y `code_debug` cubre contexto largo.
- **Datasets de visión**: `vision-arena`, `local-image` (necesita `--dataset-path`), `batch-job`, y `synthetic-pixel`.
- **`synthetic`**: como `random` pero con IDs de token sintéticos, y soporta multiturno.
- **Sobrescrituras locales**: algunos datasets aceptan `--dataset-path`, y `chat-judge`, `obfuscated-conversations`, `local-image` y `batch-job` lo **requieren**.

## Longitud de salida y muestreo

- **`--max-output-len`**: el máximo de tokens generados por petición. Domina el tiempo de ejecución.
- **`--temperature`**, **`--top-p`** y **`--top-k`**: se reenvían al servidor. Para números de throughput reproducibles, mantén la temperatura baja y fija.

## Guardado de resultados

- **`--result-filename`**: la ruta de salida JSON, y crea los directorios que necesita. Si lo dejas sin establecer, MAX no guarda nada. Establécelo para cualquier cosa que quieras comparar o conservar.
- **`--metadata key=value ...`**: se registra en el JSON, por ejemplo `--metadata version=0.3.3 tp=1 gpu=b200`. Merece la pena cuando comparas ejecuciones después.
- **`--log-dir`**: el directorio de logs por ejecución, por defecto `<backend>-latency-<timestamp>`. Un sweep de concurrencia deja un `results-<N>-median.json` por paso aquí.

## Estadísticas y perfilado

- **`--collect-gpu-stats`** y **`--no-collect-gpu-stats`**: reportan utilización de GPU y memoria pico. Funciona en NVIDIA y, desde v25.7, también en **AMD**. Solo funciona cuando el benchmark se ejecuta en la **misma instancia** que el servidor. **Activado por defecto**.
- **`--collect-cpu-stats`** y **`--collect-server-stats`**: **activados por defecto**.
- **`--profile`**: captura un trace de Nsight Systems y un resumen de los N kernels principales. El **servidor debe estar ya corriendo bajo `nsys launch`**. Ajusta la salida con `--profile-output` y `--profile-top-n` (por defecto 15). `--trace` y `--trace-file` son la forma de bajo nivel. Solo NVIDIA.
- **Flags de LoRA** (desde v25.7): `--lora`, `--lora-paths`, `--lora-uniform-traffic-ratio`, `--per-lora-traffic-ratio`, `--max-concurrent-lora-ops`.

## Archivo de configuración

`--config-file file.yaml` carga opciones desde YAML bajo una clave de nivel superior `benchmark_config:`. **Las claves usan `snake_case`**, así que `--num-prompts` se convierte en `num_prompts`. Los flags de CLI sobrescriben el archivo. Por ejemplo:

```yaml title="bench.yaml"
benchmark_config:
  model: my_arch
  backend: modular
  endpoint: /v1/completions
  host: localhost
  port: 8000
  num_prompts: 200
  dataset_name: random
  random_input_len: 512
  random_output_len: 128
  max_output_len: 128
```

```bash
max benchmark --config-file bench.yaml --max-concurrency 1,2,4,8,16,32 \
  --result-filename results/run.json
```

La mayoría de los harness repetibles siguen un patrón: un archivo YAML con `--section-name benchmark_config` más sobrescrituras de CLI por ejecución (`--base-url`, `--model`, `--result-filename`, `--max-concurrency`). El archivo fija la forma de la carga de trabajo, y la CLI fija el cableado específico de la ejecución.
