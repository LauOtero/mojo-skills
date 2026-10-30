# Importa un modelo a MAX

La skill `import-model` trae una nueva arquitectura de modelo de lenguaje grande a MAX desde un ID de modelo de Hugging Face, escrita en la API ModuleV3 de MAX (`max.experimental.nn`). Impulsa un flujo por fases (decidir y planificar, implementar y verificar), y tú eres el coordinador y validador en cada punto de control.

Este documento describe el procedimiento en detalle. El [`SKILL.md`](SKILL.md) compañero es la versión concisa y mecánica que ejecuta el agente.

## Fase 1: Decidir y planificar

Durante la fase inicial, el agente reúne información sobre el modelo objetivo de Hugging Face y propone un plan de port:

1. **Inspección**: el agente obtiene la configuración del modelo (`config.json`) y lee sus parámetros, comprobando si la clase de arquitectura ya está registrada de forma nativa en MAX.
2. **Selección de donante**: el agente identifica la arquitectura ModuleV3 más cercana en MAX (la plantilla donante, como `llama3_modulev3` u `olmo3`) para usarla como punto de partida.
3. **Análisis de deltas**: el agente compara el código de modelado de Hugging Face con la arquitectura donante para identificar diferencias estructurales (deltas) en mecanismos de atención, capas de norma o funciones de activación.

El agente presenta un plan escrito que lista la arquitectura donante elegida y el catálogo de deltas estructurales. Revisa el plan antes de autorizar al agente a escribir código. Verifica que el agente eligió el donante correcto e identificó todas las propiedades de capa únicas descritas en el paper o la model card del modelo.

## Fase 2: Implementar

Una vez que apruebas el plan, el agente genera los archivos y escribe la implementación en Python:

1. **Scaffolding**: el agente escribe un esqueleto que subclasifica la arquitectura donante en una nueva carpeta con el nombre de tu modelo.
2. **Mapeo de config**: el agente mapea las claves de config de Hugging Face a la clase de configuración tipada a partir de la cual se construye el modelo.
3. **Definición del módulo**: el agente modifica las subclases de `Module` de `max.experimental.nn` para implementar cada diferencia de arquitectura identificada en la lista de deltas.
4. **Traducción de pesos**: el agente escribe adaptadores de pesos que traducen los nombres de pesos del checkpoint de Hugging Face a los nombres de parámetros del módulo MAX.

Si el tipo de modelo aún no está registrado en la biblioteca `transformers` de Hugging Face, el agente puede necesitar escribir un parser de config personalizado. Asegúrate de que el agente actualice todos los docstrings de clase y comentarios de código copiados para que describan tu nuevo modelo, sin referencias obsoletas al donante.

## Fase 3: Verificar y validar

Después de que la implementación esté completa, el agente ejecuta scripts de validación para confirmar el comportamiento correcto:

1. **Comprobaciones previas**: el agente ejecuta `run_oss_gates.py` y las comprobaciones de humo locales (import de humo, construcción perezosa del módulo, diff de claves adaptador-a-parámetro) para atrapar errores de config y registro antes de servir.
2. **Servido local**: el agente lanza el modelo usando `max serve` para comprobar que el módulo compila y carga los pesos del checkpoint sin parámetros faltantes o mal formados.
3. **Comprobación de corrección**: el agente ejecuta generación de tokens greedy en prompts de prueba y compara el texto de salida y los logits de tokens contra el modelo de referencia de Hugging Face.

Revisa las salidas generadas y los informes de verificación. La skill sigue mejorando, así que no garantiza un modelo correcto en la primera ejecución. Si la salida es basura o incoherente, dirige al agente a una caza de divergencias capa por capa. Dile que compare salidas de capas intermedias y pesos contra el modelo de referencia de Hugging Face hasta encontrar y corregir el primer punto de divergencia.
