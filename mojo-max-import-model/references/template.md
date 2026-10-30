# Port: `<HF_MODEL_ID>`

Copia este archivo cuando empieces un bring-up. Rellénalo mientras avanzas por las fases de `SKILL.md`. Prefiere datos del `config.json` del Hub y de `inspect_hf.py` sobre suposiciones.

**ID de modelo HF:** `<org/model-name>`
**Slug (nombre de tu directorio de port):** `<slug>`
**Iniciado:** `<YYYY-MM-DD>`
**Operador:**

---

## Fase 1: Decidir y planificar

### Guarda: ¿nativo en MAX?

**`config.json` → `architectures[0]`:** `` `<ArchitecturesClassFromConfig>` ``
**`config.json` → `model_type`:** `` `<model_type>` ``

```bash
pixi run python scripts/inspect_hf.py <HF_MODEL_ID>
pixi run python scripts/list_checkpoint_keys.py <HF_MODEL_ID> --summary
pixi run python scripts/list_native_archs.py --match <ArchitecturesClassFromConfig>
pixi run python scripts/check_walls.py <HF_MODEL_ID>
```

| Comprobación                | Resultado                             |
|-----------------------------|---------------------------------------|
| ¿Registrado en MAX?         | sí / no                               |
| Slug de directorio MAX (si sí) | `` `<native_slug>` ``                 |
| Código de salida de `check_walls.py` | 0 / 1 / 2                             |
| Acción                      | servir nativo / continuar con el port |

Si está registrado, detente aquí:

```bash
pixi run max serve --model <HF_MODEL_ID>
```

### Model card

**URL de la card:** https://huggingface.co/`<HF_MODEL_ID>`

**Familia de arquitectura (de la card):**

**Modificaciones anunciadas** (frases como "we introduce", "unlike prior models"):

**Paper / blog:**

**¿Bloqueadores?** (solo CUDA personalizado, pesos solo FP8, ALiBi, SSM, etc.) → ver `references/recognize-walls.md` si sí.

### `config.json` → API MAX

Pega la tabla `config.json → MAX API` de `inspect_hf.py`, o rellena una fila por clave en el archivo del Hub:

| Clave                   | Valor                         | API MAX                                   | ¿Conectado en `MyConfig`? |
|-------------------------|-------------------------------|-------------------------------------------|----------------------|
| `architectures`         |                               | `SupportedArchitecture.name` en `arch.py` |                      |
| `torch_dtype` / `dtype` |                               | `SupportedArchitecture.default_encoding`  |                      |
| (metadatos de safetensors) | ver `list_checkpoint_keys.py` | confirma formas / dtypes en disco          |                      |

**Metadatos del checkpoint** (`list_checkpoint_keys.py --summary`):

| dtype dominante | número de tensores | notas |
|----------------|--------------|-------|
|                |              |       |

**Deltas** (claves que necesitan código de capa nn personalizado más allá de la config):

1.
2.

### Lista de deltas (HF vs arch MAX donante)

**Slug `--start-from` elegido:** `` `<llama3_modulev3|olmo3|…>` ``
**Por qué este donante** (forma de atención, MLP, MoE, orden de normas, RoPE):

| Componente          | Clase / comportamiento HF | Donante MAX | Cambio necesario |
|--------------------|---------------------|-----------|---------------|
| Atención          |                     |           |               |
| MLP / MoE          |                     |           |               |
| Bloque / orden de normas |                     |           |               |
| LM head            |                     |           |               |

**Archivo de modelado HF:**

```python
import importlib

mod = importlib.import_module(
    "transformers.models.<model_type>.modeling_<model_type>"
)
print(mod.__file__)
```

**Número de diferencias:** ___ (objetivo ≤ 3 deltas estructurales, o elige un donante diferente)

---

## Fase 2: Implementar

### Scaffold (solo esqueleto, no el port)

```bash
pixi run python scripts/scaffold.py <HF_MODEL_ID> \
  --start-from <max_arch_slug> \
  --output-dir <output_dir>
```

**Directorio del port (`<port_dir>`):** `` `<output_dir>/<slug>` `` (sin barra final). Pásalo a `--custom-architectures` y `run_oss_gates.py --port-dir` (no `<output_dir>/`).

| Archivo                 | Escrito | Notas                                                                  |
|----------------------|---------|------------------------------------------------------------------------|
| `arch.py`            | sí     | Caparazón de registro: verifica `name=`                                     |
| `model_config.py`    | sí     | Subclase de config del donante: reconectar                                          |
| `<slug>.py`          | sí     | **Módulo donante: incorrecto hasta que se implemente**                              |
| `weight_adapters.py` | sí     | Delega en los renombrados del donante: reescribir                                    |
| `model.py`           | sí     | `_instantiate_module()` construye tu módulo. Editar si el wrapper HF difiere |

No sirvas todavía: `<slug>.py` sigue ejecutando el módulo donante.

### Implementa el módulo

Ver `references/implement-graph.md`. Cada fila de delta debe tener un cambio de código antes de servir.

| Componente                               | Implementado | Ref. de línea `forward` de HF |
|-----------------------------------------|-------------|-----------------------|
| `model_config.py` (todas las claves de config)     |             |                       |
| `weight_adapters.py` (el checkpoint carga) |             |                       |
| Atención                               |             |                       |
| MLP / MoE                               |             |                       |
| Cableado del bloque decodificador                    |             |                       |
| Norma final + LM head                    |             |                       |
| `arch.py` (`name=`, encoding)           |             |                       |

**¿Implementación completa?** todas las casillas marcadas según `references/implement-graph.md#completion-criteria-required-before-serving`

```bash
pixi run python scripts/run_oss_gates.py <HF_MODEL_ID> --port-dir <port_dir>
```

### Guarda: puerta de humo

```bash
pixi run ruff format <port_dir>
pixi run ruff check <port_dir>
pixi run python scripts/check_port.py <HF_MODEL_ID> --port-dir <port_dir>
```

`ruff check` no reporta errores. Ningún parámetro faltante o mal formado, y cada tensor no consumido explicado.
El preflight de formato de pesos de `references/serve-and-iterate.md` pasa.

---

## Fase 3: Verificar

### Humo de coherencia

**Requisito previo:** implementación completa.

```bash
pixi run max serve --model <HF_MODEL_ID> \
  --custom-architectures <port_dir> \
  --quantization-encoding <encoding from arch.py>
```

**Prompt:** el prompt de la model card, a través de `/v1/chat/completions` para un modelo ajustado a instrucciones

| Resultado                                  | sí / no | Notas |
|------------------------------------------|----------|-------|
| El servidor carga sin caerse               |          |       |
| La salida se mantiene coherente con `max_tokens=64` |          |       |

Si es basura → carga `debug-model` (`../debug-model/SKILL.md`) (revisa los deltas, ya que el cableado del bloque a menudo se pasa por alto).

### Depuración de logits / capas

**Sonda rápida** (luego pasa a `debug-model` si diverge):

**Síntoma** (de `references/divergences.md`):

```bash
pixi run python scripts/compare_layers.py <HF_MODEL_ID> \
  --port 8000 \
  --prompt "The capital of France is"

# O puerta completa después de servir:
pixi run python scripts/run_oss_gates.py <HF_MODEL_ID> \
  --port-dir <port_dir> --phase verify --port 8000
```

| Ejecución | rel_diff del logprob top-1 | Veredicto | Corrección aplicada |
|-----|------------------------|---------|-------------|
| 1   |                        |         |             |
| 2   |                        |         |             |

**Causa raíz (cuando se encuentra):**

**Volcados de tensores por capa** (de `debug-model`, con sub-taps solo después de que el líder localice):

### Coincidencia de texto greedy

**Prompt:** el prompt de coherencia de arriba
**max_tokens:** 64
**Encoding MAX:** `` `<encoding que soporta el modelo, normalmente bfloat16>` ``
**dtype de referencia HF:** float32 cuando quepa

| Lado | Salida |
|------|--------|
| MAX  |        |
| HF   |        |

**¿Coinciden?** sí / no (en el primer token que difiere, los dos logprobs superiores de HF dentro de 0.1 nats es un empate; un claro ganador de HF ahí apunta a tokenizer, dtype o muestreo)

---

## Terminado

- [ ] Fase 2: el módulo implementa HF (no el shim del donante)
- [ ] Fase 3: logits alineados en el/los prompt(s) de prueba
- [ ] Fase 3: la generación greedy coincide con HF en prompt corto
- [ ] Lecciones capturadas (entrada de trampa, de divergencias o parche de skill si es reutilizable)
