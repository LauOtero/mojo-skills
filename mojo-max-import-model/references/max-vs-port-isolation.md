# ¿Está el bug en MAX o en mi port?

Cuando la precisión regresa, una ruta de característica falla, o un benchmark cae a aleatorio, el bug puede estar en tu port, en MAX serve, o en un kernel compartido. Ejecuta la misma sonda sobre un modelo nativo de MAX con **flags de servido idénticos** para decidir qué lado depurar.

## Cuándo ejecutar esto

- La generación funciona, pero el loglikelihood (ruta echo) falla
- La precisión pasa a 0-shot pero falla a few-shot
- Una ruta de flag de servido (prefill batcheado, echo, KV eviction) falla
- Los prompts con formato de chat o few-shot rompen mientras los planos funcionan

Si el fallo sigue un **flag de servido o un formato de endpoint**, sospecha de MAX primero. Si sigue la **forma de atención, el routing MoE o el cableado de RoPE**, depura el port (ver [layer-by-layer-debugging.md](layer-by-layer-debugging.md)).

## El test de control

1. **Elige un arch pequeno nativo** de la misma familia:
   - Decoder denso: `Qwen/Qwen3-1.7B`
   - MoE: `Qwen/Qwen3-30B-A3B`
   - MLA: `deepseek-ai/DeepSeek-V2-Lite`
2. **Sirve con los mismos flags** que tu port: `--devices`, `--max-batch-size`, `--max-length`, `--enable-echo`, `--quantization-encoding`, etc.
3. **Ejecuta la misma sonda** (misma tarea lm-eval, `num_fewshot`, payload `curl`).
4. **Compara:**
   - El arch nativo **falla de la misma forma** → bug de MAX. Repórtalo upstream con un repro mínimo. Deja de ajustar tu port para ese síntoma.
   - El arch nativo **pasa** → bug del port. Continúa localmente.

## Ejemplo resuelto: colapso de loglikelihood few-shot

Un port MoE grande parecía bien en generación y GSM8K, y HellaSwag 0-shot era razonable. Pero 5-shot y 10-shot estaban cerca de aleatorio (~0.25–0.31 en 4 opciones).

Sospechosos del port: `freqs_cis` de NoPE incorrecto, bug de ventana deslizante, cableado del router. Retocar esos movió las puntuaciones ligeramente pero no arregló la caída.

Control: `Qwen/Qwen3-1.7B` con el **mismo** `--enable-echo` y `--max-batch-size`:

| Tarea                      | Tu port | Qwen3-1.7B (nativo) |
|---------------------------|-----------|---------------------|
| HellaSwag 0-shot acc_norm | 0.76      | 0.49                |
| HellaSwag 5-shot acc_norm | 0.31      | **0.27**            |

El modelo nativo mostró la misma caída 0-shot → 5-shot. Diagnóstico: el bug estaba en el loglikelihood few-shot a través de `--enable-echo` / `compute_log_probabilities`, no en el port personalizado. La ejecución de control tomó unos cinco minutos y descartó el port.

## Bugs que el test de control puede encontrar

El test de control encuentra bugs en características que todos los modelos usan igual, como echo, prefill batcheado y algunos formatos de cuantización. Si el control también falla, el bug está en MAX.

Los bugs que dependen del tamaño o la forma del modelo son más difíciles de separar. Elige un control de la misma clase que tu port, como un control MoE para un port MoE.

El test de control no puede encontrar bugs en la atención, el routing MoE o el cableado del bloque de tu port. Usa [divergences.md](divergences.md) y taps de capa para esos.

## Controles conocidos buenos

| Clase de tu port              | Control                        |
|--------------------------------|--------------------------------|
| Decoder denso, RoPE, RMSNorm   | `Qwen/Qwen3-1.7B`              |
| Ventana deslizante                 | `mistralai/Mistral-7B-v0.3`    |
| MoE, routing top-k             | `Qwen/Qwen3-30B-A3B`           |
| MLA + MoE                      | `deepseek-ai/DeepSeek-V2-Lite` |
| Wrapper multimodal (solo texto) | `google/gemma-3-12b-it`        |

Por defecto usa `Qwen3-1.7B` cuando no estés seguro. Tiene la compilación en frío más rápida.

## Escalar después de que falle el control

1. Minimiza el repro (los flags + sonda más pequeños que aún fallen).
2. Captura `pixi run max --version`.
3. Reporta que el fallo se reproduce en **ambos**, tu port y un arch nativo de serie. Eso lo convierte en un problema de MAX, no en un bug de tu port.
4. Apunta la brecha en las notas de bring-up y entrega lo que pase. No sigas depurando el port para ese síntoma.

