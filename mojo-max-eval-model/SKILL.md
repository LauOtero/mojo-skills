---
name: mojo-max-eval-model
description: >
  Mide la precisión de tareas de modelos de texto servidos por MAX usando
  benchmarks estándar como GSM8K, MMLU, HellaSwag, ARC, AIME, GPQA,
  TruthfulQA, WinoGrande y BABILong. Úsala al hacer benchmark de un modelo
  servido, compararlo con las puntuaciones de la model card o de referencia,
  verificar que un nuevo modelo MAX produce respuestas correctas, o ejecutar
  evaluaciones de datasets repetibles contra un endpoint compatible con OpenAI
  de MAX.
argument-hint: "--model <MODEL_ID> --tasks gsm8k [--host localhost]"
disable-model-invocation: true
compatibility: "Pixi, acceso de red para los datasets y un endpoint MAX en ejecución."
---

# Evalúa la precisión de un modelo en MAX

Un modelo que se sirve sin errores puede seguir respondiendo mal las preguntas de un benchmark. Esta skill mide con qué precisión un modelo de texto detrás de MAX responde datasets estándar — GSM8K, MMLU, HellaSwag, ARC, AIME, GPQA, TruthfulQA, WinoGrande y BABILong. Comprueba la compatibilidad del endpoint antes de cada ejecución, separa los fallos de servido de las respuestas incorrectas, y escribe puntuaciones por tarea reproducibles que puedes comparar con una model card.

**Usa esta skill cuando** estés haciendo benchmark de un modelo servido, comparándolo con puntuaciones de model card o de referencia, verificando que un modelo MAX recién importado responde correctamente, o ejecutando evaluaciones de datasets repetibles contra un endpoint compatible con OpenAI de MAX.

**No uses esta skill cuando** el modelo aún no se sirva. Primero levanta el modelo (`mojo-max-import-model`); si se sirve pero genera salida incorrecta, persigue la divergencia con `mojo-max-debug-model` antes de hacer benchmark.

## Referencias

| Archivo | Léelo cuando |
|---------|--------------|
| [references/examples.md](references/examples.md) | Quieras más variantes de comandos — endpoints remotos, rutas de artefactos, contexto largo |
| [references/fast-fail-thresholds.md](references/fast-fail-thresholds.md) | Antes de cambiar los umbrales de seed o añadir una tarea de seed |
| [references/heartbeat-protocol.md](references/heartbeat-protocol.md) | Construyas un programa que consuma el flujo de progreso |

Lee la referencia de lo que estés haciendo, no todas de golpe.

## Configura el evaluador

Resuelve la ruta absoluta al directorio que contiene este `SKILL.md` y guárdala como `SKILL_DIR`. Todos los comandos de abajo dependen de ella, así que establécelo desde la ubicación real de la skill — no desde el directorio de trabajo del usuario ni desde el placeholder literal de abajo. Si te han pasado la ruta de la skill, expórtala directamente; si no, resuélvela desde la ruta a este archivo:

```bash
export SKILL_DIR="$(cd "$(dirname /path/to/eval-model/SKILL.md)" && pwd)"
```

Confirma que apunta al manifiesto antes de continuar, para que un valor incorrecto falle aquí en lugar de a mitad de una ejecución:

```bash
test -f "$SKILL_DIR/pixi.toml" && echo "SKILL_DIR ok: $SKILL_DIR"
```

Instala el entorno del evaluador desde su manifiesto de Pixi:

```bash
pixi install --manifest-path "$SKILL_DIR/pixi.toml"
```

Este entorno contiene las dependencias propias del evaluador (lm-eval, datasets, transformers).

Algunos datasets requieren acceso a internet, autenticación de Hugging Face o aceptación de licencia. Establece `HF_TOKEN` antes de la ejecución cuando el dataset lo requiera.

## Sirve el modelo

Las tareas se dividen en dos grupos que sirven el modelo de forma distinta — decide cuál vas a ejecutar, o sirve para poder ejecutar ambos:

- **Tareas de generación** (`gsm8k`, `aime`): puntúan una respuesta generada y se ejecutan contra un modelo servido normalmente.
- **Tareas de opción múltiple** (`mmlu`, `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande`, `truthfulqa`, `gpqa`): puntúan mediante log probabilidades del prompt, que MAX solo devuelve cuando el servidor arranca con `--enable-echo`.

Sirve desde un entorno donde MAX esté instalado.

Para tareas de generación, arranca el modelo normalmente:

```bash
max serve --model meta-llama/Llama-3.1-8B-Instruct
```

Para tareas de opción múltiple, añade `--enable-echo`:

```bash
max serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-echo
```

No puedes activar echo en un servidor en vivo. Si ya hay uno corriendo sin `--enable-echo`, páralo y reinícialo antes de evaluar una tarea de opción múltiple — no hay forma de añadir la capacidad a un proceso en ejecución. El preflight del evaluador es la comprobación autoritativa: verifica las log probabilidades del prompt antes de descargar datasets y reporta un error de configuración si no están disponibles.

Si el `generation_config.json` de un modelo establece un `repetition_penalty` no predeterminado (la familia Qwen2.5 trae `1.05`, por ejemplo), MAX activa automáticamente su ruta de muestreo con penalización, que es incompatible en forma con echo y hace fallar al worker de servido al arrancar con un error de forma de tensor `repetition_penalty`. Los modelos que dejan `repetition_penalty` sin establecer (como Llama-3.1 y Qwen3) no activan esta ruta. Si te encuentras con el fallo, evalúa un modelo cuya configuración de generación deje `repetition_penalty` sin establecer, o restringe la ejecución a tareas de generación (`gsm8k`, `aime`).

## Ejecuta una evaluación

Invoca el evaluador a través de su manifiesto de Pixi desde cualquier directorio de trabajo:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k
```

Para una comprobación rápida de endpoint y puntuación, baja ambos límites y usa el scorer incluido:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k \
  --seed-limit 4 \
  --full-limit 8 \
  --direct-http
```

Usa `--direct-http` solo para comprobaciones de conectividad y parsing como esta. Su scorer incluido es 0-shot con un prompt personalizado, así que sus números quedan muy por debajo de las puntuaciones few-shot chain-of-thought de una model card — en una ejecución medida, el mismo modelo puntuó 33% bajo el scorer directo de seed y 82% en la pasada completa de lm-eval. No compares resultados de `--direct-http` con benchmarks publicados; quita el flag para cualquier comparación con la card, de modo que se ejecute la pasada lm-eval por defecto.

Para tareas de opción múltiple en un modelo servido con alias, pasa el ID del tokenizer de Hugging Face con `--tokenizer`. Esto es obligatorio, no opcional: lm-eval necesita un tokenizer real para calcular los límites de token, y sin él el tokenizer toma por defecto `--model` y lm-eval falla al intentar cargar el alias como repo de Hugging Face.

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "my-served-alias" \
  --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks mmlu,hellaswag
```

El valor de `--model` debe coincidir con un ID devuelto por `/v1/models`.

`--apply-chat-template` envuelve cada prompt en la plantilla de chat del modelo. Si usarlo o no depende de tu línea base de comparación, no solo de si el modelo está ajustado para instrucciones:

- Para reproducir puntuaciones publicadas de leaderboards de opción múltiple (MMLU, HellaSwag, ARC, WinoGrande), déjalo **desactivado** — esas referencias se miden sobre prompts de completion crudos. Activarlo tiene éxito pero produce números que no se pueden comparar con la card.
- Úsalo solo cuando pretendas medir el modelo tal como se le conversa realmente, y estés comparando contra una línea base con plantilla de chat.

## Interpreta el resultado

El comando usa estos códigos de salida:

- `0`: Todas las tareas completadas, o cada `--target` explícito cumplido.
- `1`: Fallo de configuración, dependencia, endpoint, dataset o lm-eval.
- `2`: Al menos un `--target` explícito no cumplido.
- `3`: Una tarea directa de seed puntuó por debajo de su umbral de fast-fail.

Sin `--target`, una ejecución completa válida tiene estado `completed`; el evaluador no impone objetivos de precisión independientes del modelo. Un estado `completed` o `met` significa que la ejecución terminó limpiamente — no es por sí mismo un veredicto de que el modelo sea correcto o esté a la par. Para convertir una ejecución en un pass/fail o en una comparación, proporciona tú mismo una línea base.

### Compara contra una model card

El evaluador no conoce las puntuaciones esperadas de ningún modelo, así que una comparación real son unos pocos pasos explícitos:

1. Busca la métrica específica en la model card o referencia, haciendo coincidir la variante que produce esta herramienta — para tareas de generación es few-shot chain-of-thought (por ejemplo, Llama-3.1-8B-Instruct reporta GSM8K alrededor de 84.5, 8-shot CoT). Anota el número de shots y si se usó plantilla de chat.
2. Pasa ese número como `--target`. Ten en cuenta que `--target` es un suelo unilateral: la ejecución falla (salida `2`) solo cuando la precisión cae *por debajo*, así que actúa como puerta en lugar de medir un delta bilateral.
3. Lee `accuracy` del JSON de resumen de la tarea y reporta la diferencia real con la referencia. Permite ruido de muestreo — la ejecución completa por defecto es de 200 muestras por tarea (o por subtarea para tareas agrupadas), no el conjunto de test completo, así que trata las brechas pequeñas como ruido en lugar de regresiones. Sube `--full-limit` hacia el dataset completo cuando necesites una estimación más ajustada.

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k,mmlu \
  --target "gsm8k:0.84,mmlu:0.68"
```

Los resúmenes van por defecto a `eval-results/<model-name>/`:

```text
eval-results/meta-llama_Llama-3.1-8B-Instruct/
├── eval_summary.json
├── gsm8k_summary.json
└── mmlu_summary.json
```

Usa `--output-dir` para elegir otra ubicación o `--run-id` para cambiar solo la etiqueta del directorio por defecto. El resumen combinado registra la configuración efectiva de CLI y las versiones de dependencias del evaluador necesarias para interpretar la ejecución.

## Elige tareas

El comportamiento de las tareas es el siguiente:

- `gsm8k` y `aime`: comprobación de seed directa por HTTP, seguida de una ejecución completa de lm-eval. Añade `--direct-http` para usar el scorer exact-match incluido en la ejecución completa.
- `mmlu`, `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande` y `truthfulqa`: ejecución completa de lm-eval a través de `/v1/completions`; requieren `--enable-echo`.
- `gpqa`: ejecución de opción múltiple de lm-eval a través de `/v1/completions`; requiere `--enable-echo`, y el dataset puede requerir acceso a Hugging Face.
- `babilong`: ejecución de chat de lm-eval; por defecto usa metadatos `{"max_seq_lengths": "16k"}`.

Cada ejecución completa usa 200 muestras por defecto (200 por subtarea para tareas agrupadas como `mmlu`; `gpqa` y `babilong` usan 100 por defecto). Esto es una comprobación puntual, no el conjunto de test completo — sube `--full-limit` cuando necesites una estimación más ajustada, y espera pequeña variación entre ejecuciones con el tamaño por defecto. El resumen reporta el número agregado de muestras hijas efectivas para tareas agrupadas.

Solo las tareas de generación (`gsm8k`, `aime`) ejecutan la pasada directa de seed; las tareas de opción múltiple saltan directamente a la ejecución completa, así que ver ninguna precisión de seed para `mmlu` y compañía es lo esperado. El suelo de fast-fail del seed es una puerta de plumbing — atrapa un endpoint inutilizable, una petición malformada o una extracción de respuesta rota, no variación normal de calidad del modelo. Para juzgar la calidad, usa `--target` (ver "Compara contra una model card"), no el suelo de fast-fail.

ChartQA no está registrada porque el formato de prompt de ChartQA de lm-eval estándar no es compatible con la inyección de tokens de imagen de MAX. Añade una configuración de tarea específica de MAX validada antes de habilitarla.

Imprime la configuración efectiva de tareas sin instalar MAX ni arrancar un servidor:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" validate
```

## Evalúa modelos de razonamiento

Para un modelo MAX cuya plantilla de chat acepte `reasoning_effort`, pasa uno de `low`, `medium`, `high` o `xhigh`:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks gsm8k,aime \
  --reasoning-effort low
```

El evaluador envía este valor a través de `chat_template_kwargs={"reasoning_effort": ...}` y selecciona automáticamente el scorer directo HTTP para las tareas de generación soportadas. La petición de preflight falla claramente si el endpoint no acepta el campo.

## Monitoriza el progreso

Los registros de heartbeat van a stdout; los logs legibles por humanos van a stderr. Para archivarlos, captura cada flujo en su propio archivo:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks gsm8k \
  > heartbeats.jsonl \
  2> eval.log
```

Para vigilar la vivacidad *mientras* avanza la ejecución, mantén stdout en una tubería en lugar de un archivo plano, para que un consumidor pueda leer los heartbeats a medida que llegan — el consumidor de ejemplo de la referencia de heartbeat no puede seguir solo una redirección a archivo:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" --tasks gsm8k \
  2> eval.log | tee heartbeats.jsonl | python your_consumer.py
```

Trata un heartbeat ausente durante el doble de `--heartbeat-interval` como un posible estancamiento. Para modelos de razonamiento, una sola generación puede exceder legítimamente el intervalo por defecto de 30 segundos, así que sube `--heartbeat-interval` (y con ello el umbral de estancamiento) para evitar falsas alarmas en ejecuciones lentas pero sanas. El evaluador emite `done` solo después de escribir el resumen de la tarea correspondiente.

## Soluciona fallos

Usa estas comprobaciones:

- **Error de conexión o modelo**: Ejecuta `curl http://localhost:8000/v1/models` y pasa un ID de modelo devuelto exacto.
- **Faltan logprobs del prompt**: Reinicia MAX con `--enable-echo` (no se puede activar en un servidor en ejecución); confirma que el runtime seleccionado soporta log probabilidades.
- **El worker de servido falla al arrancar con echo y un error de forma de tensor `repetition_penalty`**: El `generation_config.json` del modelo servido establece un `repetition_penalty` no predeterminado (común en Qwen2.5), así que MAX activa automáticamente una ruta de penalización que echo no soporta. Usa un modelo cuya configuración de generación deje `repetition_penalty` sin establecer (Llama-3.1, Qwen3) para tareas de opción múltiple, o restringe la ejecución a tareas de generación (`gsm8k`, `aime`).
- **`No module named 'pydantic'` desde `max serve`**: El entorno de servido tiene el paquete `max` slim pero no `max-pipelines`. Instala `max-pipelines`.
- **Paquete faltante**: Ejecuta `pixi install --manifest-path "$SKILL_DIR/pixi.toml"` de nuevo.
- **Error de acceso a dataset**: Autentícate con Hugging Face y acepta los términos del dataset.
- **Error de tarea de lm-eval**: Lista las tareas para la versión instalada de lm-eval con `pixi run --manifest-path "$SKILL_DIR/pixi.toml" lm-eval ls tasks`, luego imprime la configuración efectiva de tareas con `pixi run --manifest-path "$SKILL_DIR/pixi.toml" validate` (la tarea `validate` ejecuta el modo `--validate-thresholds` del script).
- **Error de memoria o longitud de contexto**: Reduce `--num-concurrent`, `--full-limit` o la longitud de contexto servida del modelo según corresponda.

