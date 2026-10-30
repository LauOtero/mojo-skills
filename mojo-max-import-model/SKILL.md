---
name: mojo-max-import-model
description: >
  Úsala al importar una nueva arquitectura de modelo a MAX desde un ID de
  Hugging Face. Se activa con: "importar un modelo a MAX", "añadir modelo a
  MAX", "bring-up <modelo HF> en MAX". Flujo: inspecciona la configuración y el
  código de modelado de Hugging Face, luego genera un esqueleto a partir de una
  arquitectura ModuleV3 similar. Implementa cada capa como un Module de
  max.experimental.nn que coincida con HF, sirve y verifica contra la referencia
  de Hugging Face. Cuando el servidor funciona pero la salida es incorrecta
  (basura, desajuste greedy, coherente y luego diverge), carga debug-model para
  la caza de divergencias.
compatibility: Requiere un entorno pixi con MAX instalado, acceso de red a Hugging Face Hub y una GPU para servir/verificar.
metadata:
  argument-hint: "[ID de modelo de Hugging Face, por ejemplo 'Qwen/Qwen3-8B']"
---

# Importa un modelo a MAX

**Entrada:** un ID de modelo de Hugging Face (`$ARGUMENTS`).

Copia [references/template.md](references/template.md) para registrar este port mientras avanzas por las fases. Sus referencias a archivos son rutas desde la raíz de esta skill, así que siguen resolviéndose después de copiarlo.

Portar un modelo a MAX significa escribir un modelo ModuleV3 que realice la misma computación que el `modeling_<type>.py` del modelo en Hugging Face `transformers`. Luego lo compilas con los pesos publicados y verificas que las salidas coincidan. ModuleV3 es la API de modelos de MAX. Las capas son subclases de `Module` de `max.experimental.nn` con un `forward()` sobre valores `Tensor`. Las ops vienen de `max.experimental.functional` (importado como `F`). El pipeline construye el módulo bajo `F.lazy()` y lo compila con `compile(weights=...)`. Los modelos de una sola GPU, tensor-parallel y expert-parallel usan ModuleV3. Los modelos multi-GPU se shardean a través de un `DeviceMesh`.

Las fases del flujo son **decidir y planificar**, **implementar** y **verificar**. La Fase 1 es lectura y planificación. La Fase 2 es el port: implementar cada subcapa divergente en el módulo. La Fase 3 es la verificación, solo después de que la implementación esté completa. Las guardas (precondiciones que detienen el flujo) controlan las transiciones entre actividades. No son pasos propios.

**Anti-patrón:** ejecutar `scaffold.py`, retocar `arch.py` y servir mientras `<slug>.py` sigue implementando el donante (`llama3_modulev3`, `olmo3`, …). La verificación de logits falla en ese port porque `<slug>.py` sigue ejecutando la arquitectura donante. No ejecutes scripts de verificación hasta que pasen los criterios de finalización de [implement-graph.md](references/implement-graph.md).

Cada fase enlaza a referencias con los detalles. Lee la referencia de la actividad en la que estás, no todas de golpe.

**Entorno:** ejecuta cada comando a través del entorno pixi que tiene MAX instalado (`pixi run python …`, `pixi run max serve …`). Ejecútalos desde la raíz de la skill donde vive `pixi.toml`, y no uses `python` o `max` desnudos en el PATH del shell:

```bash
cd <path-to-skill>
pixi install
pixi run python scripts/inspect_hf.py <HF_MODEL_ID>
# O: pixi run test-scripts   # smoke-test de todos los scripts (sin GPU)
```

El entorno por defecto se instala en Linux (x86-64 y ARM64) y en macOS con Apple silicon, y no necesita CUDA. Su PyTorch ejecuta el modelo de referencia HF en CPU, o en una GPU de Apple silicon. En un host NVIDIA, `pixi run -e cuda …` ejecuta los mismos comandos con PyTorch CUDA, para que la referencia HF se ejecute también en la GPU.

Los scripts auxiliares viven en el directorio `scripts/` de esta skill (cópialos o véndelos en tu repo). Todos los helpers también son accesibles a través de un despachador unificado con los mismos nombres de argumentos y códigos de salida:

```bash
pixi run python scripts/import_model.py inspect <HF_MODEL_ID>
pixi run python scripts/import_model.py list-archs --donors
pixi run python scripts/import_model.py scaffold <HF_MODEL_ID> --start-from olmo3 --output-dir ./
pixi run python scripts/import_model.py list-archs --match LlamaForCausalLM
pixi run python scripts/import_model.py check-walls <HF_MODEL_ID>
pixi run python scripts/import_model.py list-keys <HF_MODEL_ID> --summary
pixi run python scripts/import_model.py gates <HF_MODEL_ID> --port-dir <port_dir>
pixi run python scripts/import_model.py compare <HF_MODEL_ID> --port 8000
```

Diseño del port:

- **`<port_dir>`**: carpeta slug que contiene `arch.py` y `ARCHITECTURES` en `__init__.py` (normalmente `<output_dir>/<slug>`, sin barra final). Pasa esta ruta tanto a `--custom-architectures` como a `run_oss_gates.py --port-dir`.

MAX resuelve `--custom-architectures <port_dir>` añadiendo `dirname(<port_dir>)` a `sys.path` e importando `basename(<port_dir>)` como el módulo. Pasar el directorio padre importa el nombre de módulo incorrecto (por ejemplo, `custom-arch` en lugar de tu slug).

Si encuentras errores de importación o API mientras editas, copia los imports del arch donante bajo `max/pipelines/architectures/<donor>/` en tu paquete MAX instalado. Ver [pitfalls-config.md § Trampas de importación y API de configuración](references/pitfalls-config.md#import-and-config-api-traps).

---

## Fase 1: Decidir y planificar

> **Guarda: ¿está la arquitectura ya registrada en MAX?**
> Antes de escribir código, comprueba si MAX ya registra la clase de arquitectura en el `config.json::architectures[0]` de tu modelo. Si `pixi run python scripts/list_native_archs.py --match <Class>` devuelve un slug, ejecuta `pixi run max serve --model <HF_MODEL_ID>` y detente. No se necesita port. Procedimiento completo: [native-arch-check.md](references/native-arch-check.md).

### Lee `config.json`

Obtén la configuración y lee cada campo:

```bash
pixi run python -c "from transformers import AutoConfig; \
  print(AutoConfig.from_pretrained('<HF_MODEL_ID>', trust_remote_code=True))"
```

O usa el helper, que obtiene el `config.json` crudo del Hub, ejecuta la comprobación de arquitectura nativa e imprime cada clave mapeada a la API de MAX:

```bash
pixi run python scripts/inspect_hf.py <HF_MODEL_ID>
```

Luego lista los metadatos de safetensors (claves, formas y dtypes, sin descargar pesos):

```bash
pixi run python scripts/list_checkpoint_keys.py <HF_MODEL_ID> --summary
```

El resumen cuenta tensores y dtypes. Ejecútalo de nuevo sin `--summary` para listar los nombres de las claves antes de escribir el adaptador de pesos.

Cada fila de la tabla de `inspect_hf.py` mapea una clave de `config.json` a `pipeline_config.model.huggingface_config` (o a `SupportedArchitecture` en `arch.py` para `architectures` y `torch_dtype`). Una vez que hayas elegido un donante, vuelve a ejecutarlo con `--start-from <donor_slug>`: marca cada clave que las fuentes de configuración del donante nunca mencionan. Esas claves son los deltas de configuración que implementas, un campo de configuración más el código del módulo que lo usa, a menos que ningún código del modelo las lea (`bos_token_id`, `initializer_range`). Significados de campos y deltas comunes: [read-config-json.md](references/read-config-json.md).

Escanea en busca de bloqueadores duros antes de comprometerte con un port:

```bash
pixi run python scripts/check_walls.py <HF_MODEL_ID>
```

Salida 0 → continúa. Salida 1 → revisa [recognize-walls.md](references/recognize-walls.md). Salida 2 → detente hasta que el muro se resuelva o se descarte.

### Lee la model card

Abre `https://huggingface.co/<HF_MODEL_ID>` y lee la model card para:

- **El paper o blog post.** hojea su sección de arquitectura. Los autores listan ahí sus modificaciones (QK-norm, MLA, atención de ventana deslizante, routing MoE).
- **Los "trucos" mencionados en la card.** Frases como "we introduce", "unlike prior models" y "this is the first model to" marcan deltas. Si los pasas por alto ahora, causan bugs durante la implementación.

Si la card dice que el modelo es de una familia conocida (Llama, Mistral, Qwen, Gemma), la actividad de comparación con donantes de abajo empezará desde la variante ya portada más cercana de esa familia.

Algunas señales de la model card requieren una comprobación de muros: kernels CUDA personalizados, atención personalizada sin referencia pública, pesos publicados solo en FP8/FP4, ALiBi y capas de recurrencia o space-state. Si la card menciona alguna, lee [recognize-walls.md](references/recognize-walls.md) antes de continuar. Algunos modelos no se pueden portar solo con la superficie pública de MAX.

### Propón un plan y acepta un veto

Antes de cualquier código, escribe un párrafo corto indicando lo que harías por defecto, y luego espera a que el usuario confirme o vete. Cubre la forma de distribución, el donante, las variantes de cuantización, la profundidad de validación y el objetivo de hardware, todo derivado de lo que ya has leído. Si el modelo necesita una pieza que ModuleV3 aún no tiene (una capa, una primitiva distribuida, una ruta cuantizada), nómbrala y di si la añadirás a `max.experimental.nn` o necesitas que el usuario decida. No hagas preguntas en blanco. Expón un valor por defecto y deja que el usuario lo rechace.

Guía completa y un párrafo de ejemplo: [plan-and-veto.md](references/plan-and-veto.md).

Si los bytes de pesos estimados no caben en una GPU, lee [distributed-transformer.md](references/distributed-transformer.md) antes de elegir `--start-from`: la forma de distribución importa más que la familia de atención por sí sola.

### Compara con otras arquitecturas MAX

Estás eligiendo la arquitectura ModuleV3 más cercana de la que copiar. "Más cercana" significa una coincidencia en:

- Forma de atención (densa vs. GQA vs. MLA)
- Forma de MLP (gated vs. no gated, densa vs. routed)
- Diseño del head (tied vs. untied, Linear simple vs. multi-paso)
- Forma de distribución

Lista las arquitecturas ModuleV3 que incluye tu MAX instalado (no codifiques una lista de slugs):

```bash
pixi run python scripts/list_native_archs.py --donors
```

Pistas heurísticas de señal HF → slug donante en [map-to-max.md](references/map-to-max.md). Versión rápida:

| Tu modelo                                          | Empieza desde                                    |
|-----------------------------------------------------|--------------------------------------------------|
| Llama 3-ish (GQA, RoPE, MLP SwiGLU)                 | `llama3_modulev3`                                |
| QK-norm, atención mixta deslizante y completa       | `olmo3`                                          |
| Gemma-ish (RMSNorm `1 + weight`, doble norm)        | `gemma3_modulev3`                                |
| Phi-ish (`qkv_proj` y `gate_up_proj` fusionados)    | `phi3_modulev3`                                  |
| Granite-ish (escalares MuP)                         | `granite_modulev3`                               |
| MoE (expertos dispersos, routing top-k)             | `gpt_oss_modulev3`                               |
| MLA (KV latente), una sola GPU                      | `deepseekV2_modulev3`                            |
| MLA + MoE, multi-GPU (TP, DP + EP)                  | `deepseekV3_modulev3`                            |
| Atención híbrida y capas space-state                | `nemotron_h_modulev3`                            |
| Vision-language (scaffold con `--full-copy`)        | `gemma3multimodal_modulev3`, `kimik2_5_modulev3` |

Abre el directorio del arch elegido y lee su archivo de modelo (el módulo que `model.py` instancia en `_instantiate_module()`, por ejemplo `olmo3/olmo3.py` o `llama3_modulev3/llama3.py`) y su `layers/`. Estás respondiendo una pregunta: ¿qué módulos cambian y cuáles se quedan igual en tu port?

Ahora lee el archivo de modelado de Hugging Face correspondiente:

```bash
pixi run python -c "from transformers.models.<model_type> import modeling_<model_type>; \
  print(modeling_<model_type>.__file__)"
```

Lee el `__init__`, el `forward` de atención, el `forward` del MLP, la clase del bloque y el head final. Compara cada uno con el equivalente en MAX. La referencia [read-modeling-code.md](references/read-modeling-code.md) cubre qué buscar en cada uno. Comprueba el diseño de RoPE, la fórmula de norma y la activación contra la tabla de síntomas de [divergences.md](references/divergences.md) ahora, antes de servir: una función llamada `rotate_half` puede seguir rotando pares adyacentes (ERNIE 4.5), y el valor por defecto del donante lee los pares equivocados.

Salida de esta actividad: una **lista de deltas**, una fila por cada diferencia real entre HF y el arch donante de MAX (atención, MLP/MoE, cableado del bloque, head, RoPE, máscaras). Implementas cada fila en la Fase 2. Tres o menos deltas estructurales → buena elección de donante. Muchos deltas → elige un donante más cercano o planea reescribir módulos completos. No procedas a la verificación con una lista de deltas vacía o de "parece Llama".

---

## Fase 2: Implementar

### Genera el esqueleto del diseño de archivos

`scaffold.py` solo escribe un esqueleto. No implementa tu modelo.

```bash
pixi run python scripts/scaffold.py <HF_MODEL_ID> --start-from <donor_slug> --output-dir <output_dir>
```

Esto lee `architectures[0]` del `config.json` del Hub para `arch.py::name`, y luego escribe un esqueleto que subclasifica la arquitectura ModuleV3 elegida en `<output_dir>/<slug>/`. El slug toma por defecto el nombre del repo, incluido el sufijo de tamaño (`ernie_4_5_0_3b_pt`). Pasa `--slug` para nombrar el paquete según la familia de arquitectura. Las clases generadas son `<Short>Model` y `<Short>Config` (`Ernie4_5Model`), que pueden compartir nombres con las clases propias de Hugging Face, así que califícalas cuando leas ambas lado a lado.

- `arch.py`: caparazón de registro que lleva cada palabra clave de `SupportedArchitecture` que el donante establece literalmente (`memory_planner=`, `batching=`, `reasoning_parser=`, ...), incluido un planner configurado como `PagedMemoryPlanner.with_activation_reservation(...)`. El scaffold escribe `name=`, los IDs de repo, los encodings, el formato de pesos y las clases y el adaptador propios del port (verifica `name=`, encoding y `memory_planner=`). Cuando el donante no declara planner, el campo se omite con un TODO. Un valor que el scaffold no puede rastrear hasta un import se convierte en TODO y warning.
- `model_config.py`: subclase de la config del donante con un hook `from_donor()` para los campos que el donante no lee (debe reconectarse durante la implementación)
- `model.py`: caparazón del modelo de pipeline. `_create_model_config()` retipifica la config del donante como la tuya, y `_instantiate_module()` construye tu módulo raíz y lo coloca en el dispositivo o malla con la colocación del donante. Hereda el procesador de batch del donante, así que la preparación de entrada sigue funcionando sin un `batch_processor.py`.
- `weight_adapters.py`: delega en los renombrados del donante (debe reescribirse para tu checkpoint)
- `<slug>.py`: **módulo donante** (debe editarse para coincidir con HF durante la implementación)

Después del scaffold, `<slug>.py` sigue computando la matemática del donante. No sirvas hasta que el módulo esté implementado.

El scaffold formatea el port con `ruff format` y reescribe las lecturas de `huggingface_config.rope_theta` del donante a `get_rope_theta(huggingface_config)`, que lee el valor tanto en Transformers 4 como en Transformers 5. Un port generado con una versión anterior de esta skill puede seguir leyendo `huggingface_config.rope_theta` en `model.py` y fallar `check_port.py` con `AttributeError: ... has no attribute 'rope_theta'`. Vuelve a generar el scaffold, o haz la misma edición a mano e importa `get_rope_theta` desde `max.pipelines.lib.pipeline_variants.utils`.

**Los docstrings y comentarios de código del donante sobreviven a cualquier código donante que copies.** El scaffold por defecto escribe docstrings nuevos para sus archivos generados. El modo `--full-copy` conserva el texto del donante, y también cualquier archivo donante que copies o subclases y edites a mano. Los renombrados de clase no tocan el texto que registra *lo que el archivo dice hacer*. Esos archivos abren con docstrings que describen al donante, y afirman comportamientos (soporte single-GPU, QK-norm, post-attention norm, etc.) que tu port puede no tener. Reescribir esos docstrings es un paso de implementación obligatorio. Ver [rewrite-donor-docstrings.md](references/rewrite-donor-docstrings.md) para el patrón de docstring que debe seguir cada docstring de módulo. También tiene la lista de verificación de auditoría que debes ejecutar antes de declarar la implementación terminada.

### Implementa el módulo

La Fase 1 produjo el mapa de config y la lista de deltas. La actividad de implementación los convierte en código.

Lista de verificación completa, orden de trabajo, la superficie de la API ModuleV3, anti-patrones y criterios de finalización: [implement-graph.md](references/implement-graph.md).

En orden:

1. **`model_config.py`**: la config del donante lee las claves que conoce. Para cada otra clave de `config.json` de la Fase 1 / `inspect_hf.py`, declara un campo en la config del port y establécelo en `from_donor()`, que `model.py::_create_model_config()` llama con la config terminada del donante. Confirma que los conteos de heads y `head_dim` de `construct_kv_params()` coinciden con HF. El pipeline dimensiona la caché KV a partir de `construct_kv_params()` de tu config, pero `from_donor()` copia `kv_params` de la config del donante, y los módulos leen esa copia. Si sobrescribes `construct_kv_params()`, `get_head_dim()` o `get_num_layers()`, establece también `kv_params` en `from_donor()` desde tu sobrescritura, para que la caché y los módulos coincidan.
2. **`weight_adapters.py`**: mapea las claves de safetensors de tu checkpoint a los nombres de parámetros que tu módulo raíz produce desde `parameters`, incluido el prefijo wrapper del raíz (`language_model.` en `olmo3`). Ejecuta `list_checkpoint_keys.py` primero. Ver [rename-weights.md](references/rename-weights.md). Conecta la auditoría de cobertura en [state-dict-audit.md](references/state-dict-audit.md): `compile()` rechaza un parámetro faltante o mal formado, y la auditoría atrapa tensores del checkpoint que ningún parámetro consume.
3. **`<slug>.py`**: para **cada fila de la lista de deltas**, edita o reemplaza el módulo donante para que el `forward()` de MAX refleje el `forward()` de HF:
   - Atención (Q/K/V, RoPE, máscara, GQA, softcap, …)
   - MLP o MoE (activación, routing, expertos compartidos, …)
   - Bloque decodificador (**orden de normas y cableado residual**, no intercambiables con Llama)
   - Norma final y LM head (tie, escala de logits, softcap)
4. **`arch.py`**: confirma que `name=` coincide con `architectures[0]` y que `default_encoding` coincide con el `torch_dtype` del Hub. Mantén el `memory_planner=` del scaffold (copiado del donante). Sin él, MAX no presupuesta memoria de activación para la caché KV y omite la inferencia de `max_batch_size`. Si el scaffold dejó un TODO de `memory_planner`, el donante no tenía ninguno. Resuélvelo antes de servir: los ports con caché KV necesitan `memory_planner=PagedMemoryPlanner`, y solo las arquitecturas que hacen su propia estimación de memoria (difusión, embeddings) deberían dejarlo sin establecer. El scaffold también lleva el `batching=` del donante. Cámbialo solo si el batching del port difiere del del donante.
5. **`model.py`**: `_instantiate_module()` construye tu módulo raíz y llama a `.to(self.devices[0])`, o construye el módulo dentro de `default_device(mesh)` para un port multi-GPU. Edita más solo si HF envuelve el backbone de forma diferente (VL, multimodal).

Lee el `modeling_<type>.py` de HF **mientras editas**, no después de que falle la verificación. Subclasifica el donante solo donde HF y el donante coincidan. Reescribe el módulo donde la lista de deltas dijo que difieren.

**La implementación está terminada cuando** cada elemento de [implement-graph.md](references/implement-graph.md#completion-criteria-required-before-serving) está marcado, especialmente estos:

- Cada delta tiene un cambio de código correspondiente.
- Los pesos compilan sin claves huérfanas.
- Has ejecutado la **auditoría de comentarios del scaffold** en [rewrite-donor-docstrings.md](references/rewrite-donor-docstrings.md#mandatory-audit-before-declaring-the-implementation-done) y clasificado cada coincidencia como OK, Incorrecta o Obsoleta.

Ejecuta la auditoría antes de declarar la implementación terminada. `max serve` y `compare_layers.py` no leen docstrings, así que ningún paso posterior encuentra las afirmaciones del donante que la auditoría pasa por alto.

Receta rápida de grep (reglas de clasificación completas en [rewrite-donor-docstrings.md](references/rewrite-donor-docstrings.md)):

```bash
grep -rniE 'qwen|llama|mistral|cohere|gemma|phi|deepseek|olmo|granite|qwen3|mixtral|single-GPU|single GPU|RMSNorm|QK-norm' <port_dir>/
```

En un port en modo subclase, la mayoría de las coincidencias son los imports del donante y las líneas de linaje que escribe el scaffold. Esas son correctas. Lee los docstrings y comentarios entre las coincidencias.

Tu mensaje de implementación completa debe atestiguar explícitamente la auditoría (por ejemplo, `"docstrings reescritos al patrón de rewrite-donor-docstrings.md, y rg devuelve N coincidencias, todas referencias de linaje legítimas"`). No informes la implementación como completa sin esa declaración.

Preflight (config del Hub + registro de arquitectura, ejecutar antes del primer servido):

```bash
pixi run python scripts/run_oss_gates.py <HF_MODEL_ID> --port-dir <port_dir>
```

> **Guarda: puerta de humo local (obligatoria antes de la Fase 3).**
> `max serve` compila todo el modelo antes de responder a una petición, así que un nombre o forma de parámetro incorrectos cuestan una compilación completa para descubrirlo. Ejecuta primero la comprobación del port. Importa el port como lo hace `max serve`, construye el módulo raíz bajo `F.lazy()` sin compilar, y lista cada parámetro que el checkpoint adaptado no rellena, cada desajuste de forma o dtype, y cada tensor que ningún parámetro lee:
>
> ```bash
> pixi run python scripts/check_port.py <HF_MODEL_ID> --port-dir <port_dir>
> ```
>
> Sale con 1 cuando `compile()` fallaría. Lee el checkpoint real, así que la primera ejecución descarga los pesos. Formatea y pasa lint al port antes de cada comprobación, ya que tus ediciones añaden código que el scaffold no formateó:
>
> ```bash
> pixi run ruff format <port_dir>
> pixi run ruff check <port_dir>
> ```
>
> `ruff check` debe reportar cero errores. `run_oss_gates.py` cubre muros, metadatos del checkpoint y el nombre/encoding de `arch.py`. No sustituye esta comprobación. Ver [serve-and-iterate.md](references/serve-and-iterate.md) para el preflight del formato de pesos.

---

## Fase 3: Verificar

### Comprueba si genera texto coherente

**Requisito previo:** implementación del módulo completa. No sirvas para "ver qué pasa" durante la implementación. Arregla config, adaptadores y módulo primero.

**Haz una comprobación de cordura de la referencia HF primero.** Ejecuta HF solo con la plantilla de prompt prevista de la model card, antes de involucrar a MAX. Si HF mismo produce basura, la referencia está rota, y las comparaciones contra ella reportan desajustes que ningún cambio del port arregla.

Luego sirve:

```bash
pixi run max serve --model <HF_MODEL_ID> --custom-architectures <port_dir>
```

`max serve` reserva `--device-memory-utilization` (por defecto `0.9`) de la memoria del dispositivo para la caché KV. Si el arranque falla al asignarla (sin memoria, o un buffer más grande de lo que el dispositivo permite), baja la fracción, por ejemplo `--device-memory-utilization 0.6`. Bájala también cuando la referencia HF se ejecute en la misma máquina que el servidor, donde compiten por memoria: una referencia float32 puede de otro modo derramarse a disco y hacer que cada comparación tarde minutos.

Prueba un modelo ajustado a instrucciones a través de `/v1/chat/completions`, que aplica su plantilla de chat, y un modelo base con un prompt plano a través de `/v1/completions`. Un PrefixLM necesita el formato de prompt que muestra su model card. Para un modelo que razona antes de responder (un bloque `<think>` por defecto), desactiva el razonamiento en ambos lados, con `chat_template_kwargs` en MAX y la misma palabra clave (`enable_thinking=False`) en el `apply_chat_template()` de HF, o la comparación solo cubre el texto de razonamiento. Cuando tokenices texto con plantilla de chat en el lado HF, pasa `add_special_tokens=False`, ya que la plantilla ya los añade. Cada resultado tiene un siguiente paso:

- El servidor se cae durante la carga → arregla config y adaptadores.
- El servidor arranca pero devuelve basura → inicia la caza de divergencias.
- El servidor devuelve texto plausible → ejecuta con `max_tokens=64+`, luego la puerta de verificación. Un port al que le falta un delta puede seguir respondiendo con fluidez.

Comprobación de cordura completa de la referencia HF, flujo de servido de slugs de encoder/embeddings, y disciplina del bucle de corrección-prueba: [serve-and-iterate.md](references/serve-and-iterate.md).

### Fallo de paridad/coherencia (invoca `debug-model`)

El servidor puede arrancar pero producir salida incorrecta: basura, un token greedy incorrecto en el índice K, coseno de logits alto con argmax incorrecto, o texto coherente durante N tokens que luego diverge. Cuando eso ocurre, deja de añadir taps de salida escalares y carga la skill [`debug-model`](../debug-model/SKILL.md).

Esa skill maneja salida incorrecta de un modelo que carga y se ejecuta. Su flujo requiere estos pasos:

1. Comprobación de cordura de HF con el mismo prompt + checkpoint
2. Comparadores de volcado de tensores por capa HF vs MAX (no `F.print` a ojo)
3. Agentes de investigación en paralelo con verificación numérica antes de recompilar
4. Bisección serve-vs-pipeline cuando los volcados coinciden pero el texto generado diverge

Usa `import-model` para el scaffolding de bring-up y las puertas. Usa `debug-model` para la caza de divergencias en sí.

#### Sonda rápida de logits (solo los primeros 5 minutos)

Antes de construir dumpers, ejecuta una comprobación rápida de cordura:

```bash
pixi run python scripts/compare_layers.py <HF_MODEL_ID> \
  --port 8000 \
  --prompt "The capital of France is"
```

Requiere `pixi run max serve` con `--custom-architectures <port_dir>` en el mismo puerto. La sonda lee logprobs, que el overlap scheduler rechaza. MAX habilita ese scheduler a menos que `arch.py` establezca `supports_overlap_scheduler=False`, así que cuando `arch.py` lo deja activado, añade `--no-enable-overlap-scheduler --force` al comando de servido. Este script imprime estadísticas de capas solo de HF y compara el logprob top-1 en el prompt. Ver [layer-by-layer-debugging.md](references/layer-by-layer-debugging.md) para detalles de flags.

Si los logprobs divergen o la salida es basura, cambia a `debug-model`. No iteres solo con taps de salida manuales. El catálogo de síntomas de [divergences.md](references/divergences.md) sigue aplicándose una vez que el comparador localiza la capa que falla.

### Comprueba contra Hugging Face

Ejecuta el modelo de extremo a extremo con pesos preentrenados, luego ejecuta HF con el mismo prompt con muestreo greedy. En el lado MAX, usa el dtype que coincida con el encoding de pesos que el modelo soporta (la mayoría de los modelos se distribuyen en bfloat16). Ejecuta el lado HF en float32 cuando quepa: una referencia bfloat16 añade su propio redondeo, y un port bfloat16 correcto puede desviarse de ella en unos pocos tokens.

En el primer token donde las salidas difieren, mira los dos logprobs superiores de HF. Cuando están dentro de aproximadamente 0.1 nats, los modelos empatan ahí, y cualquiera de los dos tokens es correcto. Una divergencia con un claro ganador de HF en los primeros tokens normalmente significa un desajuste de tokenizer o plantilla de chat, un desajuste de dtype con los pesos publicados, o muestreo distinto de cero en MAX.

Cuando las salidas coinciden, o difieren por primera vez en un empate, el port está terminado **para texto greedy**. Los otros criterios de finalización dependen de la profundidad de validación elegida durante la planificación. Elige un nivel de humo a paridad de logits.

Receta completa de comparación con HF, triaje de divergencias y la tabla de niveles de validación: [validation-tiers.md](references/validation-tiers.md).

---

## Trampas comunes

Usa [pitfalls.md](references/pitfalls.md) como índice: encuentra tu síntoma, luego carga el único archivo de categoría (config, pesos, módulo o servido), y [rewrite-donor-docstrings.md](references/rewrite-donor-docstrings.md) para la auditoría de docstrings específicamente. Las trampas más comunes:

- **Servir el scaffold.** No sirvas ni verifiques hasta que el módulo implemente cada delta en `<slug>.py`, porque el módulo generado ejecuta la matemática del donante.
- **Los renombrados dejan intactos los docstrings del donante.** En archivos donantes que copies (`--full-copy` o a mano), los nombres de clase se renombran, pero los docstrings y comentarios siguen describiendo al donante. Reescríbelos y ejecuta el grep de auditoría antes de declarar la implementación terminada.

## Tests y CI

Cuando añadas tests `pytest` para el modelo portado, minimiza el número de compilaciones por archivo. Compila una vez mediante un fixture con alcance de módulo y reutilízalo en los casos de `@pytest.mark.parametrize`. Para archivos que deban compilar módulos diferentes, paralelízalos con `shard_count` de Bazel para que el archivo se mantenga entero. Patrones y ejemplos completos: [tests-and-ci.md](references/tests-and-ci.md).
