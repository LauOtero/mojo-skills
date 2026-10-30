---
name: mojo-packaging
description: Empaqueta y distribuye proyectos Mojo 1.2.0 como paquetes conda con rattler-build y mojo precompile. Cubre recipe.yaml, estructura de proyecto, instalación local con pixi y publicación en modular-community. No uses mojo package para distribución entre versiones del compilador.
---

# Packaging de proyectos Mojo 1.2.0

Un proyecto Mojo se distribuye como **paquete conda** construido con `rattler-build`. La guía oficial usa `mojo precompile` en el script de build y publica el paquete en un canal conda (por ejemplo, `modular-community` en prefix.dev).

## Estructura del proyecto

```text
my-mojo-lib/
├── src/
│   └── my_mojo_lib/
│       ├── __init__.mojo
│       └── utils.mojo
├── test.mojo
├── conda.recipe/
│   └── recipe.yaml
├── LICENSE
└── README.md
```

`conda.recipe/recipe.yaml` es la ubicación por defecto que espera `rattler-build`.

## `recipe.yaml` mínimo

```yaml
context:
  version: "0.1.0"

package:
  name: my-mojo-lib
  version: ${{ version }}

source:
  - git: https://github.com/yourname/my-mojo-lib.git
    rev: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2

build:
  number: 0
  script:
    - mojo precompile src -o $PREFIX/lib/mojo/my_mojo_lib.mojopkg
```

`mojo precompile` compila el árbol de fuentes y escribe el artefacto en `$PREFIX/lib/mojo/`, la ruta que el compilador de Mojo descubre automáticamente.

## Instalar `rattler-build`

```bash
curl -fsSL https://pixi.sh/install.sh | sh
pixi global install rattler-build
rattler-build --version
```

## Construir el paquete

```bash
rattler-build build --recipe conda.recipe/recipe.yaml
```

Esto produce un archivo `.conda` listo para subir a un índice conda (prefix.dev, anaconda.org o un bucket S3).

## Instalar el paquete localmente

```bash
pixi add --path ./output/my-mojo-lib-0.1.0.conda
```

O añade el canal del paquete a `pixi.toml` y ejecuta `pixi install`.

## Publicar en `modular-community`

1. Haz fork de [`modular/modular-community`](https://github.com/modular/modular-community).
2. Añade tu `recipe.yaml` en `recipes/<nombre>/`.
3. Abre un PR; el repositorio se encarga de construir y publicar en el canal `modular-community` de prefix.dev.

## `.mojopkg` vs. árbol de fuentes

- `mojo package` produce un `.mojopkg` que **embebe la versión exacta del compilador** y se niega a cargarse contra otra versión. Es útil para distribución interna con la misma versión de Mojo, no para paquetes públicos que deben funcionar en varias versiones.
- `mojo precompile` (usado en `recipe.yaml`) distribuye el **árbol de fuentes** y compila en el entorno del usuario, evitando el acoplamiento a la versión del compilador.
- Para paquetes públicos, usa `rattler-build` con `mojo precompile`. Para artefactos internos de una sola versión, `mojo package` es suficiente.

## Anti-patrones

- **No distribuyas `.mojopkg` como paquete público**: se rompe entre versiones del compilador.
- **No escribas el `.mojopkg` fuera de `$PREFIX/lib/mojo/`**: el compilador no lo encontrará.
- **No omitas los tests en `recipe.yaml`**: `rattler-build` puede ejecutarlos para verificar el build.
- **No fijes la versión de Mojo en el `recipe.yaml`** si quieres que el paquete funcione con varias versiones; deja que el entorno resuelva.

## Reglas prácticas

1. **Estructura `src/<paquete>/`** para que `mojo precompile src` encuentre el paquete.
2. **Usa `rattler-build`** para paquetes públicos; `mojo package` solo para artefactos internos.
3. **Publica en `modular-community`** para máxima visibilidad.
4. **Incluye tests** en `recipe.yaml` y ejecútalos en CI.
5. **Documenta la versión mínima de Mojo** en el README si usas características recientes (por ejemplo, `match`/`case` de 1.2.0).
