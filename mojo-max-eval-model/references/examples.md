# Ejemplos de evaluación

Establece `SKILL_DIR` al directorio absoluto de la skill `eval-model`, luego ejecuta `pixi install --manifest-path "$SKILL_DIR/pixi.toml"`.

## Comprueba la calidad de generación

Desde el workspace de Pixi donde MAX esté instalado, arranca el servidor:

```bash
pixi run max serve --model meta-llama/Llama-3.1-8B-Instruct
```

Ejecuta una evaluación de seed y completa de GSM8K:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k
```

## Smoke-test del evaluador

Usa límites pequeños y el scorer completo incluido:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k \
  --seed-limit 4 \
  --full-limit 8 \
  --direct-http
```

Esto es útil para validar conectividad y parsing de salida. No compares resultados de límites pequeños con puntuaciones de benchmarks publicados.

## Ejecuta tareas de opción múltiple

Arranca el servidor con eco de prompt:

```bash
pixi run max serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-echo
```

Luego ejecuta tareas de lm-eval. Para reproducir puntuaciones publicadas de leaderboards, deja `--apply-chat-template` desactivado — esas referencias usan prompts de completion crudos:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks mmlu,hellaswag,arc_easy
```

Si el servidor expone un alias, proporciona el tokenizer por separado. `--tokenizer` es obligatorio para tareas de opción múltiple en un modelo con alias — sin él, lm-eval intenta cargar el alias como repo de Hugging Face y falla:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "production-model" \
  --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks mmlu
```

Añade `--apply-chat-template` solo cuando quieras medir el modelo tal como se le conversa realmente, y estés comparando contra una línea base con plantilla de chat en lugar de números de leaderboard de completion crudo.

## Compara con objetivos explícitos

Los objetivos son opcionales y deben venir de una model card o de una línea base establecida:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tasks gsm8k,mmlu \
  --target "gsm8k:0.50,mmlu:0.60"
```

El comando sale con `2` si un resultado completo no alcanza un objetivo especificado.

## Evalúa un modelo de razonamiento

Usa el esfuerzo de razonamiento solo cuando la plantilla de chat del modelo servido acepte la opción `reasoning_effort` a través de `chat_template_kwargs`:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks gsm8k,aime \
  --reasoning-effort low
```

La petición de preflight reporta un error si el endpoint rechaza el campo.

## Evalúa contexto largo

`babilong` usa por defecto un valor de metadatos de longitud de secuencia de 16K:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks babilong
```

Sobrescribe los metadatos cuando sea necesario:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks babilong \
  --lm-eval-metadata '{"max_seq_lengths": "32k"}'
```

## Captura progreso legible por máquina

Mantén el JSONL de heartbeat separado de los logs humanos:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks gsm8k \
  > heartbeats.jsonl \
  2> eval.log
```

## Usa un endpoint remoto o con prefijo

`--base-url` acepta la raíz del servidor o una base `/v1` de OpenAI:

```bash
OPENAI_API_KEY="<TOKEN>" \
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --base-url "https://example.com/v1" \
  --model "<MODEL_ID>" \
  --tasks gsm8k
```

## Elige rutas de artefactos

Usa una etiqueta corta de ejecución:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --run-id nightly-smoke \
  --tasks gsm8k
```

O establece el directorio de salida completo:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --output-dir artifacts/evals/model-a \
  --tasks gsm8k
```