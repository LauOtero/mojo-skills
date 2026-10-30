# Fallos apilados

Cuando depuras una salida incorrecta de un modelo que se ejecuta sin errores, a menudo encuentras **múltiples bugs independientes** en secuencia. Corregir el primero expone el siguiente. No marques el port como corregido hasta que la verificación pase de extremo a extremo.

## Clases de bugs (a menudo en este orden)

1. **Bug del comparador / harness**: volcados desalineados, tensor HF equivocado indexado, máscara no extendida, recuperación de tokens incorrecta. Síntoma: cos catastrófico en `post_embed` o acantilado falso en la capa final mientras los IDs de token coinciden.
2. **Bug real del modelo**: RoPE, orden de normas, enrutamiento MoE, etc. Síntoma: después de que el comparador sea fiable, el coseno por token muestra deriva dependiente de la posición.

Corregir (1) sin (2) mueve el índice de fallo hacia adelante (por ejemplo, token greedy 1 → token 2) pero no llega a PASS.

Después de que la matemática del modelo esté probada en prefill forzado por profesor:

3. **Bug de decodificación / KV del pipeline**: deriva incremental de `prepare_next_token_inputs`. Síntoma: el prefill forzado por profesor coincide con HF, pero el **pipeline incremental falla** en el mismo índice.

4. **Bug de serve / harness**: sesión HTTP o recuperación de tokens. Síntoma: el forzado por profesor y el **pipeline incremental pasan ambos**, pero `max serve` o tu comprobación de verificación falla en el mismo índice. Ejecuta la bisección serve-vs-pipeline del Paso 6 de [SKILL.md](../SKILL.md).

## Ejemplo: cuatro bugs en un port

| Orden | Clase de bug | Síntoma | Corrección |
|-------|--------------|---------|------------|
| 1 | Comparador / harness | cos de `post_embed` 0.04, los IDs de token coinciden en ambos lados | Extiende `attention_mask` al añadir tokens forzados por profesor |
| 2 | Comparador / harness | Acantilado falso en la capa final | Vuelca las salidas de capa con forward hooks, no con un índice `hidden_states` equivocado |
| 3 | Modelo | t=0 cos 1.0, t≥1 acantilado desde la capa 1 | Corrige la variante de RoPE (por ejemplo, No-RoPE cuando `inv_freq` es cero en runtime) |
| 4 | Serve / harness | Forzado por profesor + pipeline incremental pasan, serve greedy falla en el índice K | Biseca tokenizer, plantilla de chat, recuperación de tokens HTTP |

## Cómo detectar fallos apilados

- Después de una corrección, los tokens coincidentes **aumentan** pero la verificación sigue fallando
- El comparador en el índice de fallo antiguo está verde, y un nuevo índice necesita nuevos volcados
- Corregiste un subsistema pero la categoría del síntoma no cambió

## Procedimiento

1. Valida los volcadores sobre un **modelo que ya se sabe que funciona en MAX** antes de confiar en acantilados en un port nuevo
2. Después de cada corrección: ejecución de verificación completa, no solo un re-volcado en el mismo índice
