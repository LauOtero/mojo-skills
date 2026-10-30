# Perfilar un modelo cargado con `--custom-architectures`

Usa esto cuando el modelo no sea una arquitectura integrada de MAX sino una personalizada cargada con `--custom-architectures <dir>`. Los pasos de profiling de [SKILL.md](../SKILL.md) siguen aplicándose — este archivo cubre la configuración extra en tiempo de carga y los modos de fallo que no existen para modelos integrados.

La ruta que funciona en MAX actual es `max generate --profile --no-device-graph-capture --custom-architectures <dir> --model <HF_ID>` para una arquitectura *decoder* (generación de texto). Imprime la misma tabla de kernels clasificada que para modelos integrados — decoders densos, modelos MoE dispersos y modelos cuantizados de bajo bit por igual. La *mezcla* de kernels difiere por tipo de modelo, pero el comando y la forma de leer la tabla no (ver [la sección "Leer el resultado" de SKILL.md](../SKILL.md#leer-el-resultado-qué-te-dice-la-mezcla-de-kernels)).

Por ejemplo, un decoder NVFP4 MoE disperso se perfila limpiamente como un decoder de una sola GPU. Sus familias de kernels principales son:

- Matmul agrupado NVFP4 (`*_block_scaled_*_mxf4nvf4`)
- Proyecciones de atención `gemv_split_k` y `sm100_mha`
- Preparación de cuantización FP4 (`*quantize*`, `*block_scales*`)
- Routing MoE (`moe_create_indices`, `topk_*`)

Para números comparables de GPU-time / tok-s entre ejecuciones, vuelve a ejecutar `max generate --profile` sobre el mismo modelo con los mismos ajustes en lugar de reutilizar una ejecución configurada de forma distinta.

## Prerequisito: la arquitectura debe importar en tu MAX

Una arquitectura personalizada está escrita contra la versión de MAX que fije su código. Si tu MAX instalado es más reciente, las líneas `from max... import` de la arquitectura pueden apuntar a módulos movidos y la carga falla antes del profiling. Confirma primero que importa:

```bash
pixi run python -c "import sys; sys.path.insert(0,'<dir_parent>'); \
  import <module>; print([a.name for a in <module>.ARCHITECTURES])"
```

Si eso lanza `ModuleNotFoundError` / `ImportError`, la arquitectura tiene deriva de imports contra tu MAX — actualiza sus imports a tu versión antes de perfilar. Un `TypeError: ... got an unexpected keyword argument` en *tiempo de carga* es deriva de *firma* más profunda — corrige el sitio de llamada. "Los imports funcionan" es la puerta para "se puede perfilar".

## Ejecutar el comando de profiling

```bash
pixi run max generate \
  --profile --no-device-graph-capture \
  --custom-architectures /abs/path/to/<dir> \
  --model <HF_ID> \
  --prompt "Explain what a GPU kernel is." \
  --max-length 512 --num-warmups 1
```

Con una arquitectura y un modelo concretos, queda así:

```bash
pixi run max generate \
  --profile --no-device-graph-capture \
  --custom-architectures /home/me/ports/my_arch \
  --model my-org/My-Model-7B \
  --prompt "Explain what a GPU kernel is." \
  --max-length 512 --num-warmups 1
```

Por flag, y por qué importa aquí:

- `--no-device-graph-capture`: releases antiguas de MAX compilan de otro modo un grafo de dispositivo completo a `max_batch_size=512`, lo que puede tardar minutos. Las releases nuevas dejan la captura desactivada para `max generate`.
- `--custom-architectures /abs/path/to/<dir>`: apunta al directorio de la arquitectura. MAX añade su padre a `sys.path` e importa el nombre del directorio como paquete.
- No establezcas `PYTHONPATH=<dir>`. Pone el directorio de la arquitectura en sí en el path, así que `import <module>` se resuelve a un archivo `.py` de grafo dentro de él en lugar del paquete, lo que produce `module '<module>' has no attribute 'ARCHITECTURES'`. Los fallbacks de import relativo de la propia arquitectura se resuelven correctamente sin él.
- `--max-length` ≥ la longitud del prompt con plantilla de chat: los modelos instruct expanden un prompt corto a cientos de tokens vía su plantilla de chat; si `--max-length` está por debajo de eso obtienes `PromptTooLongError` (no es un fallo de profiling — súbelo a 512).

## Gotchas por tipo de modelo

- Modelo base (sin plantilla de chat): lanza `ValueError: tokenizer.chat_template is not set`. Pasa una plantilla passthrough:

  ```bash
  printf '%s' '{% for m in messages %}{{ m["content"] }}{% endfor %}' > /tmp/raw.jinja
  # añade: --chat-template /tmp/raw.jinja
  ```

- Repo con código de modelado personalizado: añade `--trust-remote-code`.
- Colisión de nombre con una arquitectura integrada: lanza `ValueError: Refusing to override existing architecture for 'X'`. Esto ocurre cuando el nombre de la arquitectura personalizada sombrea una integrada (por ejemplo, una arquitectura derivada de llama que registra `LlamaForCausalLM`). Los nombres genuinamente personalizados no colisionan; para una arquitectura que sombrea, perfila la integrada directamente.

## `max serve --custom-architectures` para embeddings está roto (MAX actual)

Servir una arquitectura *personalizada de embeddings* vía `max 
## references/custom-architectures.md

```markdown
Consecuencia para el profiling: **las arquitecturas decoder se perfilan bien vía `max generate`** (importa las arquitecturas personalizadas a tiempo); **no puedes servir arquitecturas encoder/embeddings personalizadas en MAX actual**, así que los flujos de servido de `nsys.md` no aplican a ellas hasta que MAX arregle el orden. Perfila decoders con `max generate --profile`; para embeddings, recurre a la comprobación de utilización de `gpu_snapshot.py` alrededor de la carga que puedas manejar, o espera al arreglo de serve.
