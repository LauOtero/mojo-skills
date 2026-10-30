# Proponer un plan y aceptar un veto

Antes de cualquier código, escribe un párrafo corto indicando lo que harías por defecto, y luego espera a que el usuario confirme o vete. En este punto sabes más de lo que el usuario te contó (leíste la config), así que no hagas preguntas en blanco. Expón un valor por defecto y deja que el usuario lo rechace.

El párrafo debe cubrir cada uno de estos, derivados de lo que ya leíste:

- **Piezas ModuleV3 faltantes.** Nombra cualquier pieza que el modelo necesite y que ModuleV3 aún no tenga (una capa, una primitiva distribuida, una ruta cuantizada), y si la añadirás a `max.experimental.nn` o necesitas que el usuario decida.
- **Forma de distribución.** Densa single-GPU, tensor-parallel multi-GPU, DP+EP multi-GPU (MoE), o port solo de texto de un wrapper multimodal. La regla aproximada: estima `num_params × bytes_per_param` (con el factor MoE cuando aplique) y compáralo con la HBM de una GPU menos el colchón para caché KV + activaciones. Si no cabe en single-GPU, el port se shardea sobre un `DeviceMesh` y tu donante es una arquitectura ModuleV3 shardeada por malla. Ver [distributed-transformer.md](distributed-transformer.md).
- **Donante.** La arquitectura ModuleV3 de la que harás scaffold, y la forma de atención, MLP y distribución que la hacen la coincidencia más cercana (ver [map-to-max.md](map-to-max.md)).
- **Variantes de cuantización en alcance.** Solo BF16, o también FP8 / NVFP4 / GPTQ. Comprueba la org de HF para repos hermanos. Para cualquier cosa no-BF16, haz preflight de los muros de kernel de MAX en [recognize-walls.md](recognize-walls.md). Señala un muro por adelantado, antes de construir el cableado.
- **Profundidad de validación.** Solo humo de generación, evaluación de generación tipo GSM8K, loglikelihood del lm-eval-harness (hellaswag/mmlu), o paridad de logits formal con HF. Ver [validation-tiers.md](validation-tiers.md) para los niveles.
- **Objetivo de hardware.** En qué máquina(s) servirás. Requerido para multi-GPU. Confirma la capacidad antes de descargar checkpoints grandes.

Escribe el párrafo como una propuesta con valores por defecto que el usuario puede aceptar o cambiar. Un ejemplo de forma:

> `org/LargeMoE-70B-bf16`, un causal LM MoE, solo texto (wrapper de visión descartado). Haré scaffold del patrón MoE shardeado por malla de `deepseekV3_modulev3` en 4 GPUs (atención TP, expertos EP) y registro solo BF16 por ahora. Existe el repo hermano FP8, pero el quant MoE tiene una brecha conocida en MAX. Validaré con generación greedy + GSM8K + HellaSwag. ¿Vamos?

El usuario puede decir "vamos" y ya está. O puede vetar un eje ("solo BF16 por ahora") en unas pocas palabras, sin formulario. Procede solo después de que confirme.

Si el usuario ya especificó cualquiera de estos en su petición inicial, tómalo como dado. No vuelvas a preguntar.

Después de que confirmen, declara la decisión de routing en una frase ("Empezaré desde `deepseekV3_modulev3` sobre una malla de 4 GPUs porque los pesos BF16 necesitan ~140 GB de HBM"). Eso da al usuario una oportunidad más de atrapar un routing incorrecto antes de que se reescriban archivos.
