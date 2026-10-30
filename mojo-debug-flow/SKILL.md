---
name: mojo-debug-flow
description: Flujo de depuración para errores de compilación, test o ejecución en Mojo. Úsalo cuando mojo_build, mojo_test o mojo_run fallen, al interpretar stack traces con símbolos mangleados, o al preparar una sesión de depuración con LLDB. Aplica junto con mojo-syntax.
---

# Flujo de depuración Mojo

## Cuándo usar este skill
- `mojo_build` falla con errores de compilación.
- `mojo_test` falla con fallos de aserción.
- `mojo_run` falla en ejecución o produce un crash.
- Hay un stack trace con símbolos mangleados que hay que interpretar.
- Se necesita preparar una sesión de depuración con LLDB.

**Carga siempre `mojo-syntax` en paralelo** al corregir código Mojo.

## Secuencia de diagnóstico

1. **Obtener el error completo.** Ejecutar `mojo_build` y leer la salida entera. No truncar.
2. **Clasificar el error:**
   - Sintaxis → cargar `mojo-syntax` y aplicar las correcciones de la tabla "Removed syntax".
   - Tipo/origen/convención → consultar `idiomatic-mojo.md`.
   - Linkage (`libtuxedort_nexus.so` no encontrado) → ejecutar `nexus_shared` y reintentar.
   - Dependencias → ejecutar `mojo_precompile` y reintentar `mojo_build`.
3. **Si el error persiste**, preparar depuración:
   - Ejecutar `mojo_debug` para generar el script LLDB.
   - Informar al usuario de la ruta del script y del comando a ejecutar.
4. **Si hay stack trace**, desmangular cada símbolo con `mojo_demangle` antes de interpretarlo.
5. **Si el fallo es en tests**, no ejecutar `mojo_run`: corregir primero el test o el código bajo prueba.

## Orden de herramientas por fase

| Fase | Herramienta | Propósito |
|---|---|---|
| Compilación | `mojo_precompile` | Cachear dependencias si el build es lento |
| Compilación | `mojo_build` | Obtener el error exacto |
| Diagnóstico | `mojo_demangle` | Leer stack traces mangleados |
| Depuración | `mojo_debug` | Generar script LLDB |
| Validación | `mojo_test` | Confirmar que el fix no rompe nada |

## Prohibiciones
- No modificar código sin haber leído el error completo de `mojo_build`.
- No ejecutar `mojo_run` si `mojo_test` falla.
- No invocar `mojo_debug` esperando una sesión interactiva: genera un script, no lanza el depurador.
- No desmangular símbolos a mano: usar siempre `mojo_demangle`.