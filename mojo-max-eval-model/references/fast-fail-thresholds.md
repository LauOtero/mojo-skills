# Umbrales de fast-fail

Fast-fail ejecuta una pequeña muestra directa por HTTP antes de la evaluación completa. Su propósito es atrapar generación inutilizable, problemas de esquema de petición y fallos de extracción de respuesta sin representarlos erróneamente como un benchmark completo.

`scripts/_eval_tasks.py::TASK_CONFIGS` es la fuente de verdad.

## Tareas de seed actuales

| Tarea | Tamaño de seed | Suelo | Métrica |
|---|---:|---:|---|
| `gsm8k` | 24 | 10% | Exact match |
| `aime` | 10 | 5% | Exact match |

Sobrescribe un suelo con `--fast-fail-floor`:

```bash
pixi run --manifest-path "$SKILL_DIR/pixi.toml" eval \
  --model "<MODEL_ID>" \
  --tasks gsm8k,aime \
  --fast-fail-floor "gsm8k:0.15,aime:0.05"
```

Usa `--no-fast-fail` cuando solo se requiera el resultado completo de lm-eval.

## Por qué las tareas de opción múltiple no hacen fast-fail

`mmlu`, `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande` y `truthfulqa` usan lm-eval para la construcción del prompt, los límites de tokenización y el comportamiento de la métrica. Un scorer directo simplificado puede producir números que no son comparables con el benchmark, especialmente para verosimilitud normalizada y TruthfulQA MC2. Estas tareas, por tanto, se saltan la fase directa de seed.

Su preflight sigue verificando que `/v1/completions` devuelve log probabilidades del prompt. Arranca el servidor con `--enable-echo`.

## Cambiar un umbral

Al añadir o cambiar una tarea directa de seed:

1. Usa una selección de muestra determinista.
2. Elige un suelo que detecte una petición o scorer roto, no variación normal de calidad del modelo.
3. Añade o actualiza su `TaskConfig` en `scripts/_eval_tasks.py`.
4. Confirma la configuración efectiva:

   ```bash
   pixi run --manifest-path "$SKILL_DIR/pixi.toml" validate
   ```

5. Actualiza la tabla de este archivo.
```
