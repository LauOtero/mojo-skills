# Leer el código de modelado de Hugging Face

El archivo `modeling_<type>.py` de HF define lo que tu implementación de MAX debe computar. Lo lees mientras construyes la lista de deltas, mientras implementas el módulo y de nuevo durante la caza de divergencias.

## Localizar el archivo

```python
import importlib

mod = importlib.import_module(
    "transformers.models.<model_type>.modeling_<model_type>"
)
print(mod.__file__)
```

Para modelos con `trust_remote_code=True`, el código de modelado vive en la caché de Hugging Face bajo `~/.cache/huggingface/hub/<repo>/snapshots/...`.

## Qué leer, en orden

Lee el archivo de arriba abajo siguiendo la jerarquía de clases. Para un causal LM típico, las clases relevantes son:

1. **`<ModelType>Config`**: ya cubierto cuando leíste `config.json`.
2. **`<ModelType>Attention`**: cómo se computan y combinan Q/K/V.
3. **`<ModelType>MLP`**: la capa feedforward.
4. **`<ModelType>DecoderLayer`**: cómo se cablean atención + MLP + normas.
5. **`<ModelType>Model`**: el embedding, la pila de capas, la norma final.
6. **`<ModelType>ForCausalLM`**: el LM head y la pérdida.

Estás mirando `__init__` (qué capas existen) y `forward` (cómo se componen). Salta todo lo demás.

## Qué buscar en `__init__`

Esta es la caza de deltas más fácil. Compara la lista de asignaciones `self.xxx = …` con la clase equivalente en la arquitectura MAX desde la que empiezas.

Hallazgos comunes:

- **Normas extra**: `self.q_norm = nn.RMSNorm(...)`, `self.k_norm = ...` indica QK-norm. Dos normas extra se sientan dentro de la atención. Su dimensión es o `head_dim` (estilo Olmo2, por head) o `hidden_size` (Q/K completos).
- **Proyecciones extra**: `self.q_a_proj`, `self.q_b_proj`, `self.kv_a_proj_with_mqa`, `self.kv_b_proj` es MLA. Proyección Q en dos etapas, proyección KV en dos etapas con un latente en medio.
- **Router**: `self.gate = nn.Linear(hidden, num_experts, bias=False)` es MoE. Seguido de `self.experts = nn.ModuleList([...])`.
- **Embeddings laterales**: `self.alibi_bias`, `self.rotary_emb` a nivel de atención (frente a nivel de módulo) señala variantes de codificación posicional.

## Qué buscar en el `forward` de atención

Aquí es donde la mayoría de las arquitecturas divergen. Lee línea por línea:

1. **Computación Q/K/V**: ¿son separadas (`self.q_proj(x)`, ...) o fusionadas (`self.qkv_proj(x).chunk(3)`)? El adaptador de pesos debe coincidir.
2. **Reformado de forma**: `view(bsz, q_len, num_heads, head_dim)` es estándar. Un `view(bsz, q_len, num_heads, 2, head_dim // 2)` reordena canales para RoPE.
3. **QK norm**: `q = self.q_norm(q)` antes del producto punto. Comprueba la *dimensión* sobre la que actúa la norma (head_dim es por head, hidden_size es completo).
4. **Aplicación de RoPE**: `apply_rotary_pos_emb(q, k, cos, sin)`. Lee el cuerpo de `rotate_half`, no su nombre: `x[..., : d // 2]` y `x[..., d // 2 :]` rotan mitades divididas (Llama), mientras que `x[..., 0::2]` y `x[..., 1::2]` rotan pares adyacentes (ERNIE 4.5, GPT-J), normalmente con `repeat_interleave` en `cos` y `sin`. ¿Se aplica RoPE solo a parte del head (`q[..., :rope_dim]` vs. `q` completo)? Ver [divergences.md](divergences.md#3-rope-style-mismatch-split-half-vs-interleaved).
5. **Repetición GQA**: para `num_key_value_heads < num_attention_heads`, K y V se repiten. `repeat_kv(key_states, self.num_key_value_groups)`.
6. **Máscara**: ventana deslizante vs. causal vs. sink-token. Comprueba si el índice de capa selecciona condicionalmente entre dos máscaras.
7. **Escala de softmax**: `attn_weights / math.sqrt(self.head_dim)` es lo por defecto. Algunos modelos usan una escala diferente (`1 / d` para atención normalizada, `attention_multiplier` de MuP).
8. **Softcap**: `attn_weights = soft_cap * torch.tanh(attn_weights / soft_cap)` se sienta entre la puntuación y el softmax en Gemma 2.
9. **Proyección de salida**: `self.o_proj(attn_output)`, normalmente sin nada destacable.

## Qué buscar en el `forward` del MLP

La mayoría de los MLP toman una de estas formas:

- **SwiGLU con puerta** (familia Llama): `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- **Dos capas estándar** (estilo GPT): `fc2(act(fc1(x)))`. La activación puede ser `gelu`, `gelu_new`, `gelu_tanh`, `relu`.
- **MoE** (Mixtral/Qwen3-MoE/DeepSeek): `gate(x)` → `top_k` → SwiGLU por experto → suma ponderada. Puede incluir un "experto compartido" aplicado a cada token además de los expertos enrutados.

Trampas comunes:

- Nombre de la activación. `gelu_new` y `gelu_tanh` y `gelu` plano *no* son la misma función. Comprueba `ACT2FN` en `transformers/activations.py`.
- Sesgo. La mayoría de los modelos modernos tienen `mlp_bias=False`, pero GPT-NeoX, OPT y algunos embedders tienen sesgos.
- Fusión up/gate. Algunas implementaciones fusionan `gate_proj` y `up_proj` en un único `gate_up_proj` con `chunk(2)`. El adaptador de pesos debe coincidir.

## Qué buscar en el `forward` del bloque

La clase del bloque compone atención y MLP con normas y conexiones residuales. Patrones comunes:

- **Pre-norm** (Llama, Mistral, Qwen, por defecto):

  ```text
  h = x + attn(input_layernorm(x))
  out = h + mlp(post_attention_layernorm(h))
  ```

- **Post-norm** (algunos modelos antiguos, EXAONE 4):

  ```text
  h = x + post_attention_layernorm(attn(x))
  out = h + post_feedforward_layernorm(mlp(h))
  ```

- **Peri-LN / doble norma** (Gemma 2, HyperCLOVAX): tanto pre-norm como una norma *adicional* en la salida de la subcapa antes de la suma residual:

  ```text
  h = x + post_norm1(attn(input_layernorm(x)))
  out = h + post_norm2(mlp(post_attention_layernorm(h)))
  ```

Si tu modelo es cualquier cosa que no sea el primer patrón, un bloque donante estilo Llama no coincidirá. Ver "Elegir una estrategia de edición" abajo.

## Qué buscar en el head final

Lee **`<ModelType>ForCausalLM.forward`**, no solo `__init__`. La clase del LM head a menudo es un `nn.Linear` plano, pero el forward puede escalar los estados ocultos antes de la matmul.

### Busca `lm_head(` y el escalado pre-head

En el `forward` del causal-LM, encuentra la línea que produce los logits. Grep `lm_head(` y lee la expresión pasada:

```bash
grep -n 'lm_head(' modeling_<type>.py
```

Señala cualquier cosa que no sea un `hidden_states` desnudo (o `outputs[0]`):

- **Divisor de ancho**: `self.lm_head(h / (hidden_size / dim_model_base))`, `h / self.scale_width`, o `h * (dim_model_base / hidden_size)`. Común en la familia MiniCPM y algunas configs estilo Cohere. `dim_model_base` a menudo es menor que `hidden_size`. Perder el divisor hace que los logits estén mal sin error de carga.
- **`logits_scaling` de MuP**: multiplica o divide los logits *después* de `lm_head` (ver [divergences.md §13](divergences.md#13-mup-scalars)). No lo confundas con un divisor de ancho pre-head. Comprueba el orden en HF.
- **Softcap final de logits**: Gemma 2: `softcap * tanh(logits / softcap)` después del lineal.

Registra la fórmula en tu lista de deltas (nombre del escalar, claves de config y si el escalado ocurre antes o después de `lm_head`). En MAX, refleja ese orden en `<slug>.py`, normalmente un `*` o `/` en el último estado oculto antes del `Linear` de salida, no solo un retoque de logits a posteriori.

### Otras variantes de head

- **LM head**: `self.lm_head = nn.Linear(hidden, vocab, bias=False)`. Si `config.tie_word_embeddings=True`, el LM head reutiliza la matriz de embedding.
- **Head multi-paso**: algunos modelos aplican capas adicionales entre el bloque final y el LM head (`LayerNorm → Linear → activation → Linear(vocab)`). La plantilla de MAX probablemente termina en un único Linear, así que necesitas añadir las capas extra.

## Elegir una estrategia de edición

Dada tu lista de deltas, elige un enfoque:

| Qué difiere                                             | Estrategia                                         |
|----------------------------------------------------------|--------------------------------------------------|
| Solo nombres de pesos, computación idéntica                 | Editar solo `weight_adapters.py`                   |
| Una subcapa difiere (MLP personalizado, QK-norm)               | Subclasificar la capa, sobrescribir un método          |
| Una variante de atención (ventana deslizante, MLA, softcap)     | Subclasificar la atención donante, sobrescribir `forward` |
| El diseño del bloque difiere (post-norm, peri-LN)                | Subclasificar el bloque donante, sobrescribir `forward`     |
| Head multi-paso                                          | Subclasificar el modelo de nivel superior, sobrescribir el head  |
| La atención es fundamentalmente nueva (recurrencia, state-space) | Escribir un nuevo `Module` desde `max.experimental.nn`  |
| El routing MoE difiere de los archs MoE existentes en MAX          | Escribir desde cero                               |

Prefiere subclasificar. Una capa heredada ya ha pasado la comprobación de paridad del donante, y una capa escrita desde cero no.

## Trampas de herencia

El código de modelado de HF a menudo hereda entre familias de modelos de formas no obvias. Antes de concluir que el modelo es Llama con campos renombrados:

1. Lee el *MRO completo* de la clase. `print(MyClass.__mro__)`.
2. Comprueba si algún padre sobrescribe `forward` o `__init__`, que tu clase luego hereda.
3. Busca ramas `if config.use_xxx:` en clases padre que tu modelo activa vía config.
