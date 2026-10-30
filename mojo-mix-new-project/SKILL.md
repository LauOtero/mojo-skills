---
name: mojo-mix-new-project
description: Crea un nuevo proyecto de Mojo o MAX. Úsala cuando quieras empezar un proyecto nuevo de Mojo o MAX, inicializar el entorno de `pixi` o `uv` para usar Mojo o MAX, o cuando el usuario quiera comenzar un proyecto desde cero.
---

Cuando el usuario quiera crear un proyecto nuevo, primero infiere tantas opciones como sea posible de su petición (por ejemplo, "proyecto Mojo nuevo" significa type=Mojo, "llamado foo" significa name=foo). Luego usa un prompt estructurado de opción múltiple (no texto plano) para recoger solo las opciones **restantes no especificadas** en una única interacción. NO preguntes por opciones que el usuario ya haya proporcionado o implícito. Las opciones a determinar son:

- **Nombre del proyecto**: pregunta si el usuario no ha especificado uno.
- **Tipo de proyecto**: Mojo o MAX (infiere del contexto si el usuario dijo "proyecto Mojo" o "proyecto MAX").
- **Gestor de entorno**: `pixi` (recomendado) o `uv`.
- **Tipo de proyecto uv** (solo cuando el gestor sea `uv`): proyecto uv completo (`uv init` + `uv add`, recomendado) o entorno uv rápido (`uv venv` + `uv pip install`, más ligero).
- **Canal**: nightly o stable. Por defecto nightly para proyectos MAX y stable para proyectos Mojo, y pregunta solo si el usuario no ha implícito uno.

Luego sigue la sección correspondiente de abajo (`pixi` o `uv`) para inicializar el proyecto y elegir `max` o `mojo` según corresponda. No fijes una versión manualmente: cada canal ya resuelve a la correcta.

MAX y Mojo se distribuyen juntos pero numeran sus releases de forma distinta, así que sus cadenas de versión no se parecen. En el canal stable actual, `max` es **26.7** mientras que `mojo` es **1.2.0**; en nightly son `26.7.0.dev*` y `1.3.0.dev*`. Eso es lo esperado, no un desajuste.

> [!NOTE]
> Los proyectos Mojo y MAX usan `pixi`. El CLI `magic` está **oficialmente deprecado**: Pixi puede hacer todo lo que hacía Magic, y sus comandos funcionan igual. No uses ni sugieras `magic`; usa `pixi` o el CLI `modular`.

---

## Requisitos previos del sistema

Antes de instalar, verifica que tu sistema cumple los requisitos. Mojo y MAX comparten la mayoría, pero MAX inference y serving requieren más memoria.

| Requisito | Detalle |
|-----------|---------|
| **SO** | macOS Sequoia (15) o posterior; Linux con glibc 2.34+ (Ubuntu 22.04 LTS o posterior recomendado); Windows solo vía WSL2 |
| **CPU** | x86-64-v3 (Haswell o posterior, ~2013+) o ARM64 Neoverse N1+ (Graviton2+) |
| **Python** | 3.10–3.14 |
| **Memoria** | 8 GiB mínimo para desarrollo Mojo; MAX inference requiere bastante más según el modelo |
| **Enlazador C** | `gcc`, `cc` o `clang` (necesario para compilar Mojo) |
| **macOS** | Xcode o Command Line Tools 16+; para GPU Apple, `xcodebuild -downloadComponent MetalToolchain` |

**Usuarios de Windows**: Mojo no se ejecuta de forma nativa en Windows. Instala [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (`wsl --install` en PowerShell), y luego sigue las instrucciones de Linux dentro de tu entorno WSL.

---

## Pixi (recomendado)

Pixi gestiona Python, Mojo y otras dependencias de forma reproducible dentro de un entorno controlado. Es la vía recomendada por Modular para la experiencia más fiable.

Primero, determina si `pixi` está instalado. Si no lo está, instálalo con:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Reinicia la terminal para que los cambios surtan efecto. Puedes actualizar Pixi con `pixi self-update`.

### Stable (recomendado para empezar)

```bash
# Proyecto nuevo — crea el proyecto y añade el stack completo
pixi init [PROJECT] \
  -c https://conda.modular.com/max/ -c conda-forge \
  && cd [PROJECT]
pixi add "max-all==26.7"
pixi shell
```

`max-all` incluye `max[benchmark]`, `max[serve]` y `mojo`, así que las versiones de MAX y Mojo siempre coinciden.

Si solo necesitas una parte del stack, instala el paquete específico:

| Necesitas | Paquete conda/pixi |
|-----------|--------------------|
| Servir modelos (`max serve`) | `max-serve` |
| Benchmarking (`max benchmark`) | `max-benchmark` |
| Todo el stack (incluye Mojo) | `max-all` |
| Solo el compilador de Mojo | `mojo` |

### Nightly

```bash
pixi init [PROJECT] \
  -c https://conda.modular.com/max-nightly/ -c conda-forge \
  && cd [PROJECT]
pixi add max-all
pixi shell
```

Nightly se publica varias veces por semana e incluye las últimas mejoras, pero puede tener bugs nuevos. Úsalo solo si necesitas funciones que aún no están en stable.

### Proyecto existente

Si ya tienes un `pixi.toml`, añade primero el canal a la sección `[workspace]` y luego el paquete:

```toml
[workspace]
channels = ["https://conda.modular.com/max/", "conda-forge"]
```

```bash
pixi add "max-all==26.7"
```

### Proyectos que usan Python

Si tu proyecto usa bibliotecas de Python con Mojo:

```bash
pixi add python
pixi add requests           # paquetes de conda-forge
pixi add --pypi some-pkg    # paquetes solo de PyPI
```

---

## uv

`uv` es un gestor de paquetes rápido y muy popular, familiar para desarrolladores que vienen de Python. También funciona bien con proyectos Mojo.

Si no lo tienes, instálalo con:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reinicia la terminal para que `uv` esté disponible.

### Stable (proyecto)

```bash
uv init [PROJECT] && cd [PROJECT]
uv venv && source .venv/bin/activate
uv add "max[all]"
```

Esto resuelve desde PyPI. El extra `all` incluye `mojo` y el resto del stack de MAX, así que las versiones siempre coinciden.

### Nightly (proyecto)

```bash
uv init [PROJECT] && cd [PROJECT]
uv venv && source .venv/bin/activate
uv add "max[all]" \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow
```

### Stable (entorno rápido)

```bash
mkdir [PROJECT] && cd [PROJECT]
uv venv && source .venv/bin/activate
uv pip install "max[all]"
```

### Nightly (entorno rápido)

```bash
mkdir [PROJECT] && cd [PROJECT]
uv venv && source .venv/bin/activate
uv pip install "max[all]" \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow
```

Cuando uses `uv`, trabaja siempre dentro del entorno del proyecto (`source .venv/bin/activate`) para que `max` y `mojo` estén en el `PATH`.

---

## pip

Gestor de paquetes estándar de Python. Útil si no quieres usar `pixi` ni `uv`.

### Stable

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "max[all]"
```

### Nightly

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --pre "max[all]" \
  --extra-index-url https://whl.modular.com/nightly/simple/
```

Usa `--extra-index-url`, no `--index-url`. Este último reemplaza PyPI, y el índice nightly no contiene dependencias de terceros como `numpy`, así que `pip` retrocede por todas las versiones de `max` en lugar de reportar un error claro.

---

## Conda

Para usuarios de `conda` y `mamba`.

### Stable

```bash
conda install -c conda-forge \
  -c https://conda.modular.com/max/ max-all==26.7
```

### Nightly

```bash
conda install -c conda-forge \
  -c https://conda.modular.com/max-nightly/ max-all
```

---

## Alineación de versiones con MAX

Si usas MAX con kernels Mojo personalizados, ambos deben venir del **mismo canal**. No compares sus números de versión: MAX y Mojo numeran sus releases de forma distinta, así que un par que coincide parece desajustado (stable es `max` **26.7** con `mojo` **1.2.0**).

```bash
# Comprueba que ambos vienen del mismo canal
pixi list | grep -E "^(max|mojo)\b"
```

La forma más sencilla de garantizar la alineación es instalar `max[all]` (pip/uv) o `max-all` (conda/pixi):

```bash
uv add "max[all]"
```

```bash
pixi add max-all
```

`max[all]` / `max-all` incluye `max`, `max[serve]`, `max[benchmark]` y `mojo`, de modo que las versiones de MAX y Mojo siempre coinciden. Mezclar versiones entre ambos causa fallos de compilación de kernels.

> **Nota de versión**: la versión stable actual de Mojo es **1.2.0** (22 oct 2026) y la de MAX es **26.7** (22 oct 2026). En nightly, las versiones son `1.3.0.dev*` y `26.7.0.dev*` respectivamente. El paquete `max-all==26.7` (pixi/conda) o `max[all]` (pip/uv) es la forma recomendada de instalar ambos.

---

## Referencias

- [Mojo releases](https://mojolang.org/releases/) — versiones stable y nightly de Mojo
- [MAX releases](https://max.modular.com/releases/) — versiones stable y nightly de MAX
- [MAX Packages](https://max.modular.com/stable/packages/) — qué incluye `max[all]`, `max[serve]`, `max[benchmark]` y los requisitos del sistema
- [Guía de instalación de Mojo](https://mojolang.org/install)
- [Pixi basics](https://mojolang.org/docs/pixi/) — instalación y uso de Pixi con Mojo
- [Documentación de Mojo Stable](https://mojolang.org/docs/)
- [Documentación de Mojo Nightly](https://mojolang.org/nightly/docs/)
