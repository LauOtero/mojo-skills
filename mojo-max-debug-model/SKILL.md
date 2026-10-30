---
name: mojo-max-debug-model
description: >
  Depura un modelo MAX que carga, compila, sirve y genera tokens, pero cuya
  salida discrepa de una implementación de referencia. Úsala siempre que la
  depuración de paridad se atasque en taps escalares, el modelo devuelva
  basura o tokens greedy incorrectos, el coseno de logits sea alto pero el
  argmax difiera, o la generación sea coherente y luego diverja. Se aplica
  durante un port de arquitectura, una puesta en marcha de cuantización, una
  conversión multi-GPU o tras una actualización de MAX. Se activa con
  "fallo de paridad", "corrupción silenciosa", "los logits coinciden pero los
  tokens divergen", "desajuste top-1", "divergencia greedy" y "el modelo sirve
  pero genera basura". No es para fallos de carga ni andamiaje previo al
  servido (usa import-model). Exige comparadores de volcado de tensores
  referencia-vs-MAX primero, verificación numérica de las correcciones antes
  de recompilar, y una bisección serve-vs-pipeline cuando los volcados
  coinciden pero el texto diverge.
compatibility: Requiere un entorno pixi con MAX instalado, acceso de red a Hugging Face Hub y una GPU para volcar y servir.
---

# Protocolo de fallo de paridad/coherencia

El modelo se ejecuta sin errores pero la salida es incorrecta. Los taps escalares con `F.print` ocultan bugs direccionales, y cada tap nuevo cuesta una recompilación completa. Construye primero un comparador de volcado de tensores por capa. Cada comprobación posterior se convierte en una lectura de numpy desde disco.

**Usa esta skill cuando** la salida de MAX discrepe de una referencia de PyTorch que puedas ejecutar y enganchar. El caso principal es un port de arquitectura personalizada que sirve pero falla las comprobaciones de paridad o coherencia. El mismo protocolo cubre una variante cuantizada de un port que funciona, una conversión multi-GPU de un port single-GPU que funciona, y una regresión tras una actualización de MAX. Se aplica en cualquier lugar donde exista una referencia de confianza.

**No uses esta skill cuando:**

- El servidor falla al cargar → corrige configuración, pesos o módulo (`import-model`)
- Aún no has terminado de implementar el módulo → `import-model` Fase 2
- Un modelo ya verificado necesita ajustar las tolerancias de comparación de logits → eso es calibración de umbrales, que esta skill no cubre

## Referencias

| Archivo | Léelo cuando |
|---------|--------------|
| [comparator-build.md](references/comparator-build.md) | Construyas los volcadores HF/MAX y el comparador |
| [comparator-output-patterns.md](references/comparator-output-patterns.md) | Interpretes la salida del comparador, falsos acantilados, invariante del token 0 |
| [agent-workflow.md](references/agent-workflow.md) | Despaches agentes de investigación en paralelo |
| [stacked-failures.md](references/stacked-failures.md) | Una corrección ayudó pero la verificación sigue fallando |

Para las opciones de depuración en tiempo de ejecución integradas en MAX (comprobaciones de NaN, tracebacks de fuente, logging de ops), consulta [las herramientas de depuración de MAX](https://max.modular.com/develop/debugging/). `max.nn.hooks.PrintHook` (cubierto en comparator-build.md) imprime las entradas y salidas de las capas para un triaje rápido.

## Protocolo

### Paso 0: Comprobación de cordura de HF

Ejecuta `model.generate(...)` sobre el mismo repo, prompt y checkpoint de HF. Si HF es incoherente, corrige primero el tokenizer o la plantilla de chat. El modelo MAX no es el problema.

```bash
pixi run python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('<repo>')
model = AutoModelForCausalLM.from_pretrained('<repo>', dtype=torch.float32, device_map='auto')
text = tok.apply_chat_template([{'role':'user','content':'Hello!'}], tokenize=False, add_generation_prompt=True)
out = model.generate(**tok(text, return_tensors='pt').to(model.device), max_new_tokens=32, do_sample=False)
print(tok.decode(out[0]))
"
```

### Paso 1: Construye el comparador

Sigue [comparator-build.md](references/comparator-build.md). Necesitas tres artefactos: volcador HF, volcador MAX (taps en `forward()` + runner independiente), y script comparador. Castea los tensores de volcado a FP32 dentro del `forward()` de MAX.

> **Guarda: valida los volcadores antes de confiar en ellos.** Ejecuta ambos volcadores sobre un modelo ModuleV3 que MAX ya sirva correctamente, con los taps añadidos a su `forward()`. Usa `olmo3`, o `llama3_modulev3` servido con `--prefer-module-v3`. Espera cos ≈ 0.999 en cada capa, `prompt_tokens.npy` idéntico en ambos lados, y `post_embed` cos = 1.0. Cualquier cosa menor significa que los volcadores están rotos. Corrígelos antes de leer nada en una comparación sobre tu port.

### Paso 2: Lee la salida del comparador y ramifica

Sigue [comparator-output-patterns.md](references/comparator-output-patterns.md). Comprueba los falsos acantilados (indexación incorrecta de `hidden_states`, `attention_mask` ausente en volcados de prefijo de decodificación) antes de bisecar el modelo.

Elige el siguiente paso a partir de la primera ejecución del comparador que pase la guarda:

- **Alguna capa diverge** → caza de capas. Continúa con los Pasos 3 a 5.
- **Todas las capas coinciden (cos ≥ 0.99) pero la generación aún diverge** → la matemática del modelo probablemente es correcta. Salta al Paso 6. No biseques capas.
- **El patrón coincide con una firma de falso acantilado** → corrige el volcador, vuelve a volcar, vuelve a leer. No depures el modelo contra un comparador roto.

Calcula cortes de coseno por token y por dimensión cuando el coseno global parezca ambiguo:

```python
cos_per_token = [cos(h[t], m[t]) for t in range(h.shape[0])]
cos_per_dim = [cos(h[:, d], m[:, d]) for d in range(h.shape[1])]
```

Un `max_diff` alto donde HF tiene un pico y MAX es plano suele significar que HF formó un ancla de atención que tu port no formó.

### Paso 3: Despacha agentes de investigación

Sigue [agent-workflow.md](references/agent-workflow.md). Un agente líder analiza los volcados y clasifica las hipótesis con evidencia de tensores. Los ayudantes se ejecutan en paralelo (estadísticas de pesos, diff de código, inspección de kernels, preparación de sub-taps). No despaches agentes que intenten corregir hasta que el líder localice el fallo.

### Paso 4: Verifica numéricamente antes de recompilar

Para cada hipótesis: lee los tensores de volcado, calcula lo que produciría la corrección, compara con HF. Si coincide → recompila. Si no coincide → siguiente hipótesis.

### Paso 5: Aplica la corrección, vuelve a volcar, vuelve a comparar

Una compilación, un smoke, una pasada completa del comparador (cos > 0.99 en todas las capas). Si la verificación sigue fallando, consulta [stacked-failures.md](references/stacked-failures.md).

### Paso 6: Serve vs pipeline

Cuando los volcados forzados por profesor en el paso de decodificación K coinciden con HF pero el texto generado diverge, la matemática del modelo probablemente es correcta. Biseca antes de volver a bisecar capas:

| Comprobación | Pasa | Falla → |
|--------------|------|---------|
| Volcado forzado por profesor @ K | cos ≥ 0.99, argmax coincide | Pasos 1 a 5 (bug del modelo) |
| Decodificación incremental del pipeline @ K | el token K coincide con HF | Bug de estado de decodificación (KV, caché de conv) |
| Serve vs pipeline @ K | coinciden | Bug del harness (tokenizer, plantilla de chat, recuperación de tokens) |

Construye si faltan: comparación de decodificación del pipeline, volcado incremental de capas, scripts de comparación de serve. Si tanto el forzado por profesor como el pipeline pasan pero serve falla, no edites el modelo.

## Herramientas de depuración integradas en MAX (nuevas)

MAX incluye opciones que complementan el comparador y que pueden ahorrar una recompilación:

| Opción | Qué hace | Cómo se activa |
|--------|----------|----------------|
| `MODULAR_DEBUG=sensible` | Activa los valores por defecto de depuración (NaN, sync de dispositivo, etc.) | Variable de entorno antes de `max serve` |
| `nan-check` | Inserta una comprobación tras cada grupo de ops fusionadas y señala el grupo que produjo NaN/Inf | `MODULAR_DEBUG=nan-check` |
| `uninitialized-read-check` | Rellena buffers nuevos con un patrón veneno y falla si se lee antes de escribir | `MODULAR_DEBUG=uninitialized-read-check` |
| `source-tracebacks` | Mapea las ops a la ubicación del código Python que las definió | `Graph.debug.source_tracebacks = True` |
| `PrintHook` | Añade un `print()` a cada capa del modelo | `max.nn.hooks.PrintHook` |

> **Nota de versión (v26.3)**: MAX añadió guías de depuración para precisión, errores, GPU y tracing. `MODULAR_DEBUG=sensible` es la forma más rápida de activar un conjunto razonable de comprobaciones.

> **Nota de versión (v26.1)**: `Tensor` y `Module` de `max.experimental` (ModuleV3) son ahora la API principal de desarrollo de modelos. Si tu port usa la API `Graph`/`TensorValue` de V2, la migración a ModuleV3 puede simplificar la depuración porque el modo eager coincide con PyTorch.
