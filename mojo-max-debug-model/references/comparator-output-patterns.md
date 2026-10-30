# Patrones de salida del comparador

Lee la tabla de firmas **y** el coseno por token antes de elegir una corrección. Un acantilado en capas tardías puede venir de un volcado HF que indexó el tensor equivocado.

## Tabla de firmas

| Patrón | Forma por token | Categoría probable | Primera comprobación |
|--------|-----------------|--------------------|----------------------|
| `post_embed` cos < 0.99 | todos los tokens mal | Carga de embedding, sharding de vocab, dtype incorrecto | Auditoría de claves de pesos, mismos IDs de token en ambos volcados |
| `post_embed` cos = 1.0, `layer_00` cos < 0.99 en **t=0** | token 0 mal | Atención/MLP de capa 0, no RoPE | Invariante del token 0 (abajo) |
| `post_embed` cos = 1.0, **t=0 coincide**, **t≥1** mal desde la capa 1 | dependiente de posición | RoPE, máscara causal, KV, GQA | cos/sin de RoPE de HF, No-RoPE vs RoPE plano |
| Decaimiento lento y monotónico del coseno | deriva en todas partes | Acumulación BF16, escala de atención incorrecta | Compara en FP32, revisa la escala de atención |
| Acantilado en la capa N, pico en HF, MAX plano | el sumidero no se forma | La atención no puede construir un ancla de registro | Delta de atención en la capa N |
| Acantilado en la capa N, pico en MAX, HF ~0 | activación espuria en MAX | Orden de norma incorrecto, residual malo, bug de volcado | Re-verifica que el volcado HF sea salida de capa, no de norma |
| `last_logits` cos ≥ 0.98, argmax incorrecto | logits casi coincidentes | Deriva del subespacio de rango 1 | Diff de los 10 logits principales |
| Cos de secuencia completa ≈ 1.0, cos del último token malo | enmascarado por otras filas | Bug específico de posición | `--token-index` / bucle por token |

## Invariante del token 0

Bajo máscara causal, la salida de atención en el token 0 es independiente de Q, K, RoPE y la máscara porque el softmax sobre una clave es 1.0:

```text
attn_out[token=0, :] = V[0, :, :] @ o_proj^T
```

1. Calcula analíticamente `V[0] @ o_proj^T` a partir de los pesos + la entrada HF en t=0.
2. Compáralo con `attn_out[t=0]` de MAX en los volcados.
3. Si no coinciden → bug de proyección V, layout de o_proj o escritura KV, no Q/K/RoPE.

Ejecuta esto antes de los sub-taps de atención cuando la atención de la capa 0 diverja. Si t=0 coincide pero t≥1 es malo → aplicación de RoPE o máscara, no V/o_proj.

**Excepción: sumideros de atención aprendidos.** El invariante asume que el softmax sobre una sola clave da peso 1.0. Los modelos que añaden un logit de sumidero aprendido al denominador del softmax (estilo gpt-oss) ponen masa de probabilidad en el sumidero en cada posición, así que `attn_out[t=0] ≠ V[0] @ o_proj^T` por diseño. Omite esta comprobación para modelos con sumidero. Un desajuste en t=0 ahí no es evidencia de un bug de V/o_proj.

Los taps de bisección pueden guardar residuales escalados. Divide por el multiplicador del residual antes de comparar la atención cruda.

## Falsos acantilados (bugs de volcado, no del modelo)

### Layout no estándar de `hidden_states`

Algunos modelos guardan las entradas previas a la capa más la norma final, no las salidas por capa. Indexar `hidden_states[i+1]` como capa `i` crea acantilados falsos en la capa final. Usa forward hooks por capa.

### Prefijo de decodificación sin `attention_mask`

Extiende `attention_mask` al añadir tokens forzados por profesor. De lo contrario, el coseno de `post_embed` colapsa aunque los IDs de token coincidan.

### Sobrescritura de hooks en sub-forwards

Los hooks pueden dispararse varias veces por forward. Prefiere `output_hidden_states=True` más hooks por capa cuando la semántica de tuplas sea ambigua.

## Corrupción parcial

La verificación puede fallar en el índice de token 1+ mientras el smoke pasa:

- Construye volcados del paso de decodificación K (prefijo = tokens greedy 0..K-1)
- Compara en `--token-index K`
- Biseca la primera capa donde el coseno por token caiga por debajo de 0.99

Consulta [stacked-failures.md](stacked-failures.md) cuando aparezca un segundo bug después de corregir el primero.
