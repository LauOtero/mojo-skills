# Niveles de validación

Ejecuta el modelo de extremo a extremo con pesos preentrenados, luego ejecuta HF con el mismo prompt con muestreo greedy. En el lado MAX, sirve el encoding que usan los pesos publicados del modelo (la mayoría se publica en bfloat16). Carga la referencia HF en float32 cuando quepa: una referencia bfloat16 añade su propio redondeo, y un port bfloat16 correcto puede desviarse de ella en unos pocos tokens.

Envía el prompt de la model card. Para un modelo ajustado a instrucciones, envíalo a través de `/v1/chat/completions` en MAX y `apply_chat_template()` en HF, para que ambos apliquen la misma plantilla. Para un modelo base, envía un prompt plano a través de `/v1/completions`, como abajo. El campo `model` es el ID de modelo de Hugging Face, o el `--served-model-name` que pasaste a `max serve`.

```bash
# MAX: servido con el encoding de pesos publicado
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "<HF_MODEL_ID>", "prompt": "The capital of France is", \
       "max_tokens": 64, "temperature": 0.0}' \
  | pixi run python -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])"

# Referencia de HuggingFace en float32
pixi run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('<HF_MODEL_ID>', trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(
    '<HF_MODEL_ID>', trust_remote_code=True, device_map='auto', dtype='float32',
)
ids = tok('The capital of France is', return_tensors='pt').input_ids.to(m.device)
out = m.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
"
```

Las salidas deben ser idénticas o casi idénticas, y el redondeo bfloat16 puede hacer que se desvíen en generaciones largas. En el primer token donde difieran, mira los dos logprobs superiores de HF. Cuando están dentro de aproximadamente 0.1 nats, los modelos empatan ahí, y cualquiera de los dos tokens es correcto. Una divergencia con un claro ganador de HF en los *primeros* tokens después de que la caza de divergencias haya pasado normalmente significa:

- **Desajuste de tokenizer o plantilla de chat.** Prueba a cambiar el tokenizer/plantilla de chat de HF y vuelve a ejecutar. Si las salidas convergen, el problema era el formato del prompt, no el modelo.
- **Desajuste de dtype con los pesos publicados.** Confirma que MAX usa el encoding en el que se publica el modelo (la mayoría en bfloat16). Compara `arch.py::default_encoding` con el `torch_dtype` de la config del Hub.
- **Deriva de muestreo en MAX.** Confirma `temperature=0.0`, `top_p=1.0` en el lado MAX. Cualquier temperatura distinta de cero diverge de HF greedy.

El texto coincidente significa que el port pasa la comparación de texto greedy. La finalización depende de la profundidad de validación elegida durante plan-and-veto.

## Niveles de validación

Elige el nivel más alto que requiera el alcance de tu bring-up. Los niveles más altos incluyen los más bajos.

| Nivel                      | Qué prueba                                                                             | Herramienta                                            | Criterio de aprobación                                                                                                                                                       |
|---------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1: Humo                  | El texto greedy en un prompt parece plausible                                                 | ``curl /v1/completions``                        | Tokens reales, no basura                                                                                                                                             |
| 2: Batería de coherencia      | 4–6 prompts diversos con ``max_tokens=64`` y ``256`` (factual, razonamiento, código, creativo) | bucle ``curl``                                   | Todas las salidas sobre el tema, sin colapso a bucle de decode                                                                                                                        |
| 3: GSM8K                  | Evaluación de matemáticas de generación 8-shot                                                               | ``run_lm_eval.py``                              | ≥ 0.5 para cualquier modelo base moderno, ≥ 0.7 para >70B                                                                                                                      |
| 4: HellaSwag 0-shot       | Ruta de loglikelihood, contexto corto                                                         | ``run_lm_eval.py`` + ``--enable-echo`` en serve | Dentro del ±5% de la model card de HF                                                                                                                                           |
| 5: Loglikelihood few-shot | hellaswag/mmlu 5-shot                                                                     | ``run_lm_eval.py``                              | Coincide con la model card de HF. **Si es aleatorio (~0.25–0.30): ejecuta la comprobación con qwen3 como control antes de depurar tu port**. Ver [max-vs-port-isolation.md](max-vs-port-isolation.md) |
| 6: Paridad de logits           | Diff de logprob por posición vs referencia HF                                                 | ``compare_layers.py`` + comparadores de debug-model | rel_diff < 5% top-1 en el/los prompt(s) de prueba, coseno por capa de los comparadores                                                                                         |

``run_lm_eval.py`` es el harness de evaluación del repo, en ``max/tests/integration/accuracy/run_lm_eval.py``.

Flags de servido requeridos por nivel:

- Niveles 1–3 (generación): ningún flag más allá de ``--custom-architectures``. Si el arranque falla asignando la caché KV, baja ``--device-memory-utilization`` (por defecto ``0.9``).
- Niveles 4–5 (loglikelihood): pasa ``--enable-echo``. El endpoint compatible con OpenAI ``/v1/completions`` con ``echo=true, logprobs=N`` requiere que el modelo se compile con soporte echo.
- Nivel 6 (paridad): igual que los niveles 4–5. Donde los logprobs diverjan, localiza la capa con los comparadores por capa de la skill debug-model.
- Los niveles 4–6 leen logprobs, que el overlap scheduler rechaza. A menos que ``arch.py`` establezca ``supports_overlap_scheduler=False``, pasa también ``--no-enable-overlap-scheduler --force``.

