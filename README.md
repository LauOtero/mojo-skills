[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/modular/skills/ci.yml?branch=main&logo=github)](https://github.com/modular/skills/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Mojo Version](https://img.shields.io/badge/Mojo%20Version-1.2.0-success)](https://github.com/modular/mojo/releases/tag/v1.2.0)
[![MAX Version](https://img.shields.io/badge/MAX%20Version-26.7-success)](https://github.com/modular/max/releases/tag/v26.7)

# 🚀 Skills de Modular

Las habilidades oficiales de agentes de IA de [Modular](https://www.modular.com/) para desarrollar aplicaciones con **Mojo** y gestionar modelos en **MAX**. Estas skills siguen el [Agent Skills Standard](https://agentskills.io/specification) y permiten a cualquier agente de codificación de IA escribir código Mojo, importar, servir y evaluar modelos en MAX.

> 🔧 **Versiones actuales (octubre 2026)**: Mojo stable **1.2.0** (22 oct 2026) y MAX stable **26.7** (22 oct 2026). Las skills se actualizan automáticamente contra el canal nightly cuando una característica llega antes a ese canal.

## 🛠️ Instalación

Elige las instrucciones para tu agente:

### Claude Code

```bash
/plugin marketplace add modular/skills
/plugin install skills@modular
```

Para un subconjunto, instala `max-skills@modular` para trabajar con modelos en MAX, o `mojo-skills@modular` para escribir Mojo.

### Codex y otros agentes

Con [Node.js instalado](https://nodejs.org/en/download):

```bash
npx skills add modular/skills
```

Esto instala las skills en la ubicación correcta para tu agente.

Actualízalas después:

```bash
npx skills update
```

Instala una skill a la vez:

```bash
npx skills add modular/skills --skill mojo-syntax
```

Para instalar a mano, clona [el repositorio](https://github.com/modular/skills) y copia o enlaza simbólicamente los directorios de skills individuales en el directorio de skills de tu agente (para Claude Code, `~/.claude/skills/`)..

---

## 🧩 Skills de Mojo y proyecto

Habilidades que cubren la escritura de Mojo y la configuración de un proyecto:

### 🔹 Fundamentos del lenguaje

- [`/mojo-syntax`](mojo-syntax/SKILL.md): Corrige las suposiciones preentrenadas para que tu agente escriba Mojo moderno. Úsala siempre que un agente escriba Mojo.
- [`/mojo-idiomatic-review`](mojo-idiomatic-review/SKILL.md): Revisa el código Mojo contra las reglas de `idiomatic-mojo.md` antes de dar un cambio por terminado. Escanea el diff, localiza los patrones marcados como INCORRECTO y reporta qué se corrigió y qué se dejó con justificación.
- [`/mojo-metaprogramming`](mojo-metaprogramming/SKILL.md): Metaprogramación con `comptime`, parameterización de structs y funciones, reflexión con `@fieldwise_init`/`@implicit`, y `type_of`. Sustituye los obsoletos `alias` y `@parameter`.

### 🔹 Interoperabilidad

- [`/mojo-python-interop`](mojo-python-interop/SKILL.md): Maneja Mojo llamando a Python y Python llamando a Mojo, incluida la construcción de módulos de extensión de Python. Usa `PythonObject`, `PythonModuleBuilder` y `max.mojo.importer`.
- [`/mojo-c-ffi`](mojo-c-ffi/SKILL.md): Llama a bibliotecas C desde Mojo con `std.ffi`: `external_call`, `CStringSpan`, `OwnedDLHandle`, funciones variádicas y carga dinámica. Para BLAS, CUDA, libc y cualquier API C nativa.

### 🔹 Concurrencia y paralelismo

- [`/mojo-parallelize`](mojo-parallelize/SKILL.md): Uso correcto de `parallelize()`, `sync_parallelize()` y `parallelize_over_rows()` en Mojo 1.2. Cubre closures unificados, configuración del runtime y anti-patrones. No usa el API público de async.
- [`/mojo-concurrency-patterns`](mojo-concurrency-patterns/SKILL.md): Patrones de concurrencia segura sin el API público de async. Cubre `Atomic`, `Ordering`, `BlockingSpinLock`, `BlockingScopedLock`, `SpinWaiter`, partición de datos y anti-patrones como bloquear dentro de `parallelize`.
- [`/mojo-asyncrt-internals`](mojo-asyncrt-internals/SKILL.md): Referencia interna del runtime C++ AsyncRT: `ThreadPoolWorkQueue`, `AsyncValue`, `CPUDevice` y su relación con `parallelize`. No es una API pública; el API async fue eliminado en v1.1.0 y `__async`/`__await` son inestables.

### 🔹 GPU y rendimiento

- [`/mojo-gpu-fundamentals`](mojo-gpu-fundamentals/SKILL.md): Patrones para programar GPUs NVIDIA, AMD y Apple silicon en Mojo. Empareja con `/mojo-syntax`.
- [`/mojo-cpu-profiling`](mojo-cpu-profiling/SKILL.md): Perfila código Mojo y MAX en CPU. Cubre `max.profiler.cpu`, `profiling_range`, `--time-profile` y herramientas externas como `perf` y VTune.

### 🔹 Testing, empaquetado y API

- [`/mojo-testing`](mojo-testing/SKILL.md): Tests unitarios con `TestSuite`, `assert_equal`, `assert_raises` y descubrimiento automático. El subcomando `mojo test` fue eliminado; los tests se ejecutan con `mojo run`.
- [`/mojo-packaging`](mojo-packaging/SKILL.md): Empaqueta y distribuye proyectos Mojo como paquetes conda con `rattler-build` y `mojo precompile`. Cubre `recipe.yaml`, instalación local y publicación en `modular-community`.
- [`/mojo-api-design`](mojo-api-design/SKILL.md): Diseña APIs idiomáticas con traits, `@fieldwise_init`, `@implicit`, convenciones de argumentos y `Optional`/`StringSpan` en lugar de centinelas y tipos propietarios.

### 🔹 Datos y redes

- [`/mojo-serialization`](mojo-serialization/SKILL.md): Serialización en Mojo 1.2. Cubre la ausencia de `std.json` oficial, EmberJSON para JSON con reflexión, interop con Python para JSON/YAML y serialización binaria con `Writable`/`Writer`.
- [`/mojo-networking`](mojo-networking/SKILL.md): Redes en Mojo. Cubre la ausencia de un cliente HTTP público en `std.net`, el uso de **flare** (HTTP/1.1, HTTP/2, HTTP/3, WebSocket, TLS, TCP, UDP) y **lightbug_http**, e interop con Python para prototipos.

### 🔹 Proyectos

- [`/mojo-mix-new-project`](mojo-mix-new-project/SKILL.md): Crea un proyecto nuevo de Mojo o MAX, configurando el entorno `pixi` o `uv`. Es la única skill mixta (prefijo `mojo-mix-`) porque sirve a ambos tipos de proyecto.

---

## Ciclo de vida de modelos

Skills que llevan un modelo desde un checkpoint de Hugging Face hasta un despliegue verificado y medido. Se pasan el testigo entre sí, y cada una nombra el estado en el que espera encontrar el modelo:

- [`/mojo-max-import-model`](mojo-max-import-model/SKILL.md): Importa una nueva arquitectura de modelo a MAX desde un ID de Hugging Face como modelo ModuleV3 (`max.experimental.nn`), generando el esqueleto a partir de una arquitectura registrada similar y verificando que las salidas coinciden. Pasa a `/mojo-max-debug-model` cuando el servidor corre pero el texto es incorrecto.
- [`/mojo-max-serve-model`](mojo-max-serve-model/SKILL.md): Te lleva de ningún entorno a un endpoint compatible con OpenAI en ejecución con `max serve`, eligiendo los flags que necesita un modelo: `--devices`, `--quantization-encoding`, `--max-length`, `--task` y `--trust-remote-code`. Usa `--custom-architectures` para una arquitectura portada con `/mojo-max-import-model`.
- [`/mojo-max-migrate-v2-to-v3`](mojo-max-migrate-v2-to-v3/SKILL.md): Porta un modelo escrito contra la API de grafo V2 (`max.nn`, `TensorValue`, `Graph` explícito) a ModuleV3 (`max.experimental.nn`), dejando el código V2 intacto y verificando que las salidas greedy coinciden.
- [`/mojo-max-debug-model`](mojo-max-debug-model/SKILL.md): Toma el relevo cuando un modelo carga y genera tokens pero la salida es incorrecta. Construye comparadores de volcado de tensores y biseca serve vs pipeline. Para caídas en carga, usa `/mojo-max-import-model`.
- [`/mojo-max-benchmark-model`](mojo-max-benchmark-model/SKILL.md): Maneja carga contra un endpoint iniciado con `/mojo-max-serve-model`. Reporta throughput y latencia (TTFT, TPOT, latencia inter-token), más utilización de GPU cuando corre en el mismo host NVIDIA que el servidor. Pasa a `/mojo-max-profile-model` cuando los números muestran un cuello de botella.
- [`/mojo-max-profile-model`](mojo-max-profile-model/SKILL.md): Encuentra dónde va el tiempo de inferencia y si la GPU está saturada. Funciona de más barato a más caro: un snapshot de utilización, un desglose de kernels con `nsys` o `rocprofv3`, y una inmersión con `ncu` en un solo kernel cuando uno domina.
- [`/mojo-max-eval-model`](mojo-max-eval-model/SKILL.md): Mide la precisión de tareas en benchmarks estándar: GSM8K, MMLU, HellaSwag, ARC, AIME, GPQA, TruthfulQA, WinoGrande y BABILong. Distingue fallos de servido de respuestas incorrectas del modelo. Pídesela por nombre; no se activa sola.
- [`/mojo-max-runtime-tuning`](mojo-max-runtime-tuning/SKILL.md): Ajuste de rendimiento del runtime de MAX. Cubre `M_newRuntimeConfig`/`M_setNumThreads` para controlar hilos, TCMalloc NUMA, afinidad con `MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS`, profiling y límites de workers.

---

## Convención de nombres

Las skills siguen un prefijo según su ámbito:

| Prefijo | Ámbito | Ejemplos |
|---------|--------|----------|
| `mojo-` | Exclusivas de Mojo | `mojo-syntax`, `mojo-parallelize`, `mojo-testing` |
| `mojo-max-` | Exclusivas de MAX | `mojo-max-serve-model`, `mojo-max-eval-model` |
| `mojo-mix-` | Mixtas (Mojo y MAX) | `mojo-mix-new-project` |

---

## 🎯 Ejemplos

Una vez instaladas estas skills, puedes usarlas para muchas tareas comunes. Algunos ejemplos:

### 🚀 Iniciar un proyecto Mojo nuevo

```bash
Quiero crear un proyecto Mojo nuevo llamado "nvfp4-for-metal".
```

### 🔁 Traducir código CUDA C++ a Mojo

```bash
Hay un kernel CUDA en `../example`. Crea un proyecto Mojo nuevo que implemente ese mismo kernel.
```

### ⚙️ Paralelizar un bucle de CPU

```bash
Convierte este bucle en uno que use `parallelize` y explique por qué el número de work items debería ser múltiplo de `parallelism_level()`.
```

### ✅ Añadir tests a una biblioteca Mojo

```bash
Añade tests con `TestSuite` para estas funciones y configura el archivo para que se ejecute con `mojo run`.
```

### 📦 Importar un modelo de Hugging Face a MAX

```bash
Importa `Qwen/Qwen3-8B` a MAX y verifica que la generación greedy coincide con la referencia de Hugging Face.
```

Para varias de estas skills, tu agente de IA puede pedirte más información para aclarar tus objetivos y asegurarse de que usa las herramientas y patrones correctos.

---

## Notas de versión

| Componente | Stable | Nightly |
|------------|--------|---------|
| **Mojo** | 1.2.0 (22 oct 2026) | 1.3.0.dev2026102605 |
| **MAX** | 26.7 (22 oct 2026) | 26.7.0.dev2026101305 |

- **Mojo 1.2.0** introduce coincidencia de patrones (`match`/`case`), hace `StringSpan` inmutable y traslada más de veinte funciones de `std.math` a una implementación compartida CPU/GPU.
- **MAX 26.7** añade nuevos modelos (Muse Glimmer, MiMo-V2.6-Flash), la API `/v1/decisions`, decodificación especulativa más capaz (DFlash, DFlash2, DSpark) y Jenga como caché KV por defecto.

Las skills apuntan a la versión stable más reciente y, cuando una característica llega antes a nightly, lo indican en su propio `SKILL.md`.

---

## 📜 Licencia

Apache 2.0. Consulta el archivo [LICENSE](./LICENSE) para más detalles.
