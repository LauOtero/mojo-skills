# Flujo de trabajo de agentes en paralelo

Un agente líder analiza los volcados. Los agentes ayudantes reúnen evidencia ortogonal en paralelo. El coordinador aplica las correcciones después de que el líder localice.

No despaches agentes que intenten corregir antes de la localización. Cada corrección equivocada cuesta un ciclo de compilación de 10 a 25 minutos.

**¿Sin soporte de subagentes en tu harness?** El protocolo sigue funcionando. Ejecuta el análisis del agente líder tú mismo, en línea. Los pasos numerados del prompt de abajo son el análisis de cualquier manera. Haz las comprobaciones de los ayudantes una a una a medida que una hipótesis lo requiera. Los ayudantes en paralelo ahorran tiempo de reloj. La parte obligatoria es localizar el bug con números antes de tocar código, y los anti-patrones de abajo siguen aplicando.

## Prompt del agente líder

```text
Eres el AGENTE LÍDER de depuración. Los tensores de estado oculto por capa de HF y MAX sobre el MISMO prompt + los MISMOS pesos están en disco. Produce un enunciado preciso del problema con números. No adivines ni te detengas en la primera hipótesis.

Volcados: HF <ruta>/hf_layers/*.npy, MAX <ruta>/max_layers/*.npy
Salida del comparador: <pega la tabla cos_sim>
Ya verificado: <lista>
Aún sospechoso: <lista>

1. Coseno por token y por dimensión para cada par de capas.
2. Para cualquier acantilado: (token, dim) del máximo desacuerdo; signo y magnitud en cada lado.
3. Invariante del token 0: attn_out[0] = V[0] @ o_proj^T. Compara analítico vs volcados.
4. Hipótesis con citas archivo:línea y firmas de tensor predichas.
5. Clasifica las correcciones por confianza.

Entregable: hipótesis clasificadas con evidencia. Nada de especulación sin números.
```

## Agentes ayudantes (en paralelo con el líder)

| Ayudante | Tarea |
|----------|-------|
| Estadísticas de pesos | Audita los pesos de la capa del acantilado vs capas adyacentes |
| Diff de código | Subsistema roto vs donante y referencia conocida como buena |
| Inspección de kernels | Wrappers de Python y docstrings para ops sospechosas |
| Micro-test | HF vs MAX sobre entradas sintéticas para una op |
| Preparación de sub-taps | Prepara taps más finos en `forward()`. No los ejecutes hasta que el líder decida |

## Anti-patrones

- Agentes que intentan corregir antes del análisis del líder
- Ayudantes recalculando lo que el líder ya calcula
- Pedir al líder que implemente correcciones (solo localiza)
- Re-lanzar al líder sobre volcados idénticos (envía un seguimiento enfocado)

## Lista de verificación del coordinador

1. Síntoma registrado: índice de divergencia, capa que falla o salida de smoke
2. Comparador construido ([comparator-build.md](comparator-build.md))
3. Líder despachado, ayudantes solo después de que existan los volcados
4. Un `max serve` o compilación en frío a la vez por máquina
