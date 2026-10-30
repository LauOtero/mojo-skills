# Leer la salida de depuración capa por capa

> **Para fallos de paridad/coherencia** (el servidor corre pero la salida es incorrecta), carga primero la skill [`debug-model`](../../debug-model/SKILL.md). Ese protocolo exige comparadores de volcado de tensores por capa y agentes de investigación en paralelo. Esta referencia cubre la sonda rápida `compare_layers.py` que `import-model` ejecuta antes de la entrega.

La caza de divergencias tiene dos capas de depuración:

1. **`compare_layers.py`**: sondas de logits automatizadas (lo que corre el script).
2. **Volcados de tensores por capa**: comparadores HF vs MAX (ver `debug-model`). Prefiérelos sobre taps de salida escalares.

## Ejecuta `compare_layers.py`

Desde la raíz de la skill:

```bash
pixi run python scripts/compare_layers.py <HF_MODEL_ID> \
  --port 8000 \
  --prompt "The capital of France is"
```

Requiere `pixi run max serve` con `--custom-architectures <port_dir>` (la carpeta del slug, no su padre). Las sondas leen logprobs, que el overlap scheduler rechaza con HTTP 400. MAX habilita ese scheduler a menos que `arch.py` establezca `supports_overlap_scheduler=False`. Para otros ports, añade `--no-enable-overlap-scheduler` y `--force` al comando de servido: MAX 26.6 y anteriores ignoran el primer flag sin el segundo.

Flags:

| Flag                            | Propósito                                                            |
|---------------------------------|--------------------------------------------------------------------|
| `--dtype float32\|bfloat16`     | dtype de referencia HF (por defecto `float32`)                     |
| `--served-model-name NAME`      | Nombre del modelo al que responde el servidor (por defecto: el ID de modelo HF)        |
| `--skip-hf-layers`              | Omite las estadísticas de estado oculto de HF (más rápido)                                |
| `--skip-multi`                  | Omite la sonda de prefijos multiposición                                  |
| `--long-prompt "..."`           | Prompt para la sonda multiposición (~110 tokens por defecto)              |
| `--positions 1,5,20,50,100,200` | Longitudes de prefijo a comparar                                          |

Preflight antes del servido (guarda + scaffold):

```bash
pixi run python scripts/check_walls.py <HF_MODEL_ID>
pixi run python scripts/run_oss_gates.py <HF_MODEL_ID> --port-dir <port_dir>
```

Después del servido (durante la caza de divergencias):

```bash
pixi run python scripts/run_oss_gates.py <HF_MODEL_ID> \
  --port-dir <port_dir> --phase verify --port 8000
```

---

## Qué imprime el script

### 1. Estadísticas de estado oculto de HF (solo HF)

```text
  layer        mean     max_abs        norm
    0      0.0001      0.4200       12.34
    ...
```

Estas filas son solo diagnóstico. MAX no expone estados ocultos a través de la API de completions. Úsalas para confirmar que HF carga de forma sana (sin NaN, normas razonables). No comparan contra MAX.

### 2. Logprob top-1 del prompt corto

```text
       check            hf           max    rel_diff  verdict
top1_logprob      -0.1234      -0.1250      0.0130  ok
```

| Veredicto    | Significado                                          |
|------------|--------------------------------------------------|
| `ok`       | Diferencia relativa de logprob &lt; 5%              |
| `DIVERGED` | Los logits de prefill discrepan en el último token del prompt |

Si esta fila diverge en el token 0, el bug es temprano: tokenizer, embedding, orden de normas o bloque de atención 0.

### 3. Sonda top-1 multiposición

```text
   pos     hf_id              max_text  verdict
     1      1234                  ' the'  ok
     5      5678                  ' fox'  ok
    50      9012                  ' lap'  DIVERGED
```

El script decodifica el id de token argmax de HF y compara con el texto de completion greedy de MAX en cada longitud de prefijo:

| Veredicto    | Significado                                                                |
|------------|------------------------------------------------------------------------|
| `ok`       | El token top-1 de MAX es el token top-1 de HF                                  |
| `near-tie` | MAX eligió el segundo token de HF o posterior, dentro de 0.1 nats del top-1 de HF   |
| `DIVERGED` | El token top-1 de MAX no está entre los candidatos cercanos de HF                    |

`near-tie` cuenta como coincidencia: el servido en bfloat16 mueve los logprobs superiores en unos 0.05 nats, suficiente para intercambiar dos tokens cercanos en un modelo correcto. `run_oss_gates.py --phase verify` ejecuta esta sonda como su puerta `logits_multi`, y la comprobación del prompt corto de arriba como su puerta `logits_prefill`. Un diseño incorrecto de RoPE puede coincidir en el token top-1 en prefijos cortos, así que la puerta de prefill es la que lo atrapa temprano.

**Primera fila `DIVERGED`** localiza bugs dependientes de la posición:

| Primera divergencia en | Causa probable                                |
|---------------------|-----------------------------------------------|
| Posición 1–5        | Tokenizer, embedding, atención temprana         |
| Posición 5–20       | GQA repeat, QK-norm, diseño de heads              |
| Posición 50+        | Estilo/theta de RoPE, relleno de RoPE parcial        |
| Posición 100+       | Manejo de posición en la ruta de decode, ventana deslizante |

Ver [divergences.md](divergences.md) para arreglos.

---

## Volcados de tensores por capa (cuando las sondas de logits no bastan)

Usa volcados por capa cuando las sondas corta y multiposición pasen pero la salida siga mal, o cuando necesites saber *qué subcapa* divergió. Sigue [`debug-model`](../../debug-model/SKILL.md): construye volcadores HF y MAX, ejecuta el comparador, y lee [`comparator-output-patterns.md`](../../debug-model/references/comparator-output-patterns.md).

Usa salidas extra de `forward()` como sub-taps finos solo *después* de que el agente líder localice el subsistema roto, no como el bucle principal de bisección.

### Interpretar diffs de taps

Si la primera divergencia está en:

- **post-embed**: comprueba `weight_adapters.py` para `embed_tokens.weight`, y comprueba `tie_word_embeddings`.
- **post-attn (capa 0)**: sospecha de la atención. Comprueba nombres Q/K/V, estilo de RoPE, GQA, QK-norm y la máscara de ventana deslizante.
- **post-mlp (capa 0)**: sospecha del MLP. Comprueba la variante de activación (`gelu_new` vs `gelu_tanh`), el diseño con puerta y los términos de sesgo.
- **post-block-norm** pero attn y MLP bien: sospecha del cableado del bloque. Comprueba pre-norm vs paralelo vs doble-norm, orden residual y multiplicadores MuP.

### Bugs de escala vs dirección

- **Dirección similar, magnitud diferente**: factor de escala faltante, escala de softmax incorrecta, formulación de norma incorrecta (`x * weight` vs `x * (1 + weight)`).
- **Magnitud similar, cos_sim bajo**: desajuste de estilo de RoPE, permutación de heads, error de signo en `rotate_half`.

---

## Bucle de iteración

1. Ejecuta `compare_layers.py` (o `run_oss_gates.py --phase verify`).
2. Localiza desde la primera fila `DIVERGED` (fila de logprob o fila multiposición).
3. Abre el `forward()` de HF correspondiente para esa capa/componente.
4. Arregla una operación en `<slug>.py` o `weight_adapters.py`.
5. **Reinicia `pixi run max serve`**. Los grafos compilados obsoletos ocultan arreglos.
6. Vuelve a ejecutar.

Cada pasada debe empujar la primera divergencia más adelante. Si no lo hace, revierte y relee el código fuente de HF.

---

## Limitaciones

- La API de completions de MAX no tiene exportación de estado oculto por capa, así que los taps son manuales.
- La comparación de texto top-1 asume decode greedy, así que usa `temperature=0.0` en el servido.
- La referencia HF carga con `trust_remote_code=True` y `device_map="auto"`.
- `--dtype` establece el dtype de la referencia HF. El lado MAX corre con el encoding del servido. Una referencia bfloat16 añade su propio redondeo: en SmolLM2-135M, un modelo MAX correcto difiere de una referencia bfloat16 en 7.7% en logprob top-1 y de una referencia float32 en 1.6%. Mantén el valor por defecto float32.

Para modelos `trust_remote_code` que dan NaN con `.to("cuda")`, ver [pitfalls-serving.md](pitfalls-serving.md#trust_remote_codetrue-with-tocuda-can-produce-nan).
