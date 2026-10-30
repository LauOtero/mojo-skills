# Reconocer arquitecturas difíciles de portar

Algunos modelos no se pueden portar a MAX solo con la superficie pública, al menos no sin un trabajo de infraestructura significativo. Los patrones de abajo son señales de que necesitarás esperar soporte upstream de MAX o contribuir una nueva primitiva.

La lista de modelos "bloqueados" no es exhaustiva. Describe las señales de un modelo que necesita trabajo de infraestructura antes de que un port pueda empezar.

## Kernels CUDA personalizados en el código de modelado

Si `modeling_<type>.py` importa de un paquete de extensión CUDA (p. ej. `from flash_attn import flash_attn_func`), el modelo upstream asume un kernel específico que MAX puede o no tener.

- **MAX tiene flash attention** para máscaras causales estándar. Ventana deslizante y la mayoría de las variantes de atención enmascarada están soportadas.
- **MAX no tiene** variantes personalizadas como Lightning Indexer (DeepSeek V3-Flash), Native Sparse Attention y kernels de atención similares de nivel investigación.

Señal: la sección del paper se titula algo como "Our novel attention mechanism" y un solo archivo CUDA del repo la implementa.

### Compatibilidad de cuantización MoE (preflight antes de dimensionar FP8/NVFP4)

Los kernels de expertos enrutados cuantizados de MAX solo soportan esquemas de escalado específicos. Comprueba el `quantization_config` del checkpoint HF antes de comprometerte con cualquier variante de quant MoE. La ruta ModuleV3 (`QuantizedMoE` en `deepseekV3_modulev3/layers/quant_moe.py`) acepta solo FP8 block-scaled y NVFP4, y lanza `ValueError` para cualquier otro formato. La tabla cubre la ruta graph-API (`max.nn.moe.MoEQuantized`), donde un esquema no soportado falla una aserción de `block_size` en el primer servido.

| Formato HF (compressed-tensors / nativo)  | Escalado de pesos                              | Escalado de activaciones                       | ¿`MoEQuantized` de MAX 26.4?                                                     |
|------------------------------------------|---------------------------------------------|------------------------------------------|----------------------------------------------------------------------------------|
| **FP8 block-scaled** (estilo DeepSeek-V3) | ``block_size=(128, 128)``                   | Block-scaled por entrada                   | ✅ Soportado vía la rama no-FP4 (asserta `(128, 128)`)                     |
| **NVFP4** (``nvfp4-pack-quantized``)     | Bloque ``group_size=16``                     | Bloque, con escalas globales por proyección | ✅ Soportado vía la rama ``is_fp4``                                               |
| **FP8 per-channel de compressed-tensors**   | ``strategy="channel"`` (por canal de salida) | ``strategy="token"``, ``dynamic=true``   | ❌ No soportado. El ``block_size`` de ``input_scale`` es ``None``, así que la aserción se dispara |
| **FBGEMM FP8**                           | escala tensor row-wise                       | escala estática por tensor                  | Limitado. Consulta la lógica del parser                                                  |
| **MXFP4** (estilo ``quark`` / GPT-OSS)    | block-scaled                                | block-scaled                             | ✅ Vía la rama FP4                                                            |

Cómo comprobarlo con antelación:

```bash
pixi run python -c "
from huggingface_hub import hf_hub_download
import json
cfg = json.loads(open(hf_hub_download(repo_id='<HF_ID>', filename='config.json')).read())
qc = cfg.get('quantization_config') or cfg.get('text_config', {}).get('quantization_config') or {}
grp = (qc.get('config_groups') or {}).get('group_0', {})
print('format:', qc.get('format'))
print('weights:', {k: grp.get('weights', {}).get(k) for k in ('strategy','group_size','num_bits','type')})
print('inputs :', {k: grp.get('input_activations', {}).get(k) for k in ('strategy','group_size','num_bits','dynamic')})
"
```

Si la salida es ``weights.strategy='channel'`` e ``inputs.strategy='token'`` con ``dynamic=True``, la variante está en la columna no soportada. Elige una de estas opciones:

1. **Registra la variante en `arch.py` pero difiere el servido.** El adaptador de pesos, la normalización de la ignore-list y el cableado selectivo por capa de ``quant_config`` pueden aterrizar todos. El muro está en los kernels MoE cuantizados de MAX serve. Documenta la brecha, y entrega cuando MAX añada soporte upstream.
2. **Omite esta variante por ahora.** Si la variante BF16 cabe en tu hardware, sirve esa. Marca la variante más pequeña como "solo registro, bloqueada en upstream".

Cualquiera de las dos elecciones encaja en el alcance del bring-up en el paso de plan-and-veto. Ejecuta la comprobación en el momento de leer la config, antes de construir el cableado FP8, para que un esquema no soportado aflore antes de ese trabajo.

### Sesgos de atención en checkpoints NVFP4

Algunas recetas de cuantización NVFP4 absorben el residuo de cuantización por canal de salida en un término `bias` separado en cada proyección. Eso incluye `q_proj` / `k_proj` / `v_proj` / `o_proj` de atención y el `gate` del router. Si el `config.json` de HF dice `attention_bias: false` pero el checkpoint trae tensores `self_attn.o_proj.bias`, el port necesita lineales de atención con sesgo (`Linear(..., bias=True)`) solo para esa variante. Detéctalo desde el state dict (presencia de claves `.bias`) y sobrescribe el flag de config cuando se construye el módulo. El sesgo es específico de la variante, no del modelo. La publicación BF16 del mismo modelo no tendrá sesgos.

## Codificación posicional ALiBi

ALiBi (Attention with Linear Biases, usada en BLOOM, MPT y algunos modelos más antiguos) añade un sesgo lineal por head a las puntuaciones de atención y no usa RoPE. MAX 26.x eliminó la variante `CAUSAL_ALIBI_MASK`. La superficie pública no tiene una ruta ALiBi de primera clase a través de `flash_attention_ragged`.

Señal: `config.json::position_embedding_type == "alibi"`, o el `forward` de atención lee `self.alibi_bias` y lo suma a las puntuaciones.

Si necesitas un modelo ALiBi, prefiere uno de sus descendientes convertidos a RoPE o espera soporte upstream de MAX.

## Modelos que requieren tokenización o plantillas de chat personalizadas

Si el tokenizer no es un tokenizer estándar BPE/SentencePiece/Unigram cargable vía `AutoTokenizer`, pasarás tiempo significativo en el wrapping del tokenizer antes de que importe cualquier trabajo del modelo.

Señal: `tokenizer_config.json` referencia una clase de tokenizer no estándar, o la model card dice explícitamente "usa nuestro script de inferencia personalizado".

## Difusión / samplers multi-paso

La difusión de imagen (variantes de Stable Diffusion, FLUX), la difusión de vídeo y modelos similares usan bucles de muestreo multi-paso fuera de la plantilla estándar de causal LM. MAX soporta algunos de ellos como slugs de arch dedicados (mira la lista de arquitecturas), pero portar un nuevo modelo de difusión no es el mismo flujo que portar un causal LM.

Señal: el modelo tiene un UNet, un VAE y un scheduler, y ninguna pila decodificadora.

## Mixture-of-experts multimodal con routing no compartido

El MoE estándar (cada bloque MoE tiene su propio router) está bien soportado en MAX (`gpt_oss_modulev3`, `deepseekV3_modulev3`). Las variantes donde el router se comparte entre varias capas, o donde los expertos son a su vez capas de atención, no lo están.

Señal: el paper menciona "shared router", "hyper-MoE" o "expert attention".

## Qué hacer cuando chocas con un muro

1. Comprueba si un modelo estrechamente relacionado (misma familia, variante más simple) es portable, y porta ese. A menudo el *primer* modelo de una familia es difícil, y las variantes posteriores reutilizan la infra.
2. Si la arquitectura es novedosa y necesitas portarla, el camino es contribuir una nueva primitiva de MAX upstream. Es un proyecto distinto del flujo de esta skill.

No sortees un muro con una aproximación, como un kernel sustituto que computa algo cercano al original. Un port construido así puede devolver resultados incorrectos sin error. Deja el modelo sin portar hasta que MAX soporte la pieza faltante.
