---
name: mojo-metaprogramming
description: Metaprogramación en Mojo 1.2.0 con comptime, parameterización de structs y funciones, comptime for/if, reflexión con @fieldwise_init y @implicit, y type_of. Cubre la sustitución de alias/@parameter por comptime y los patrones de generación de código en tiempo de compilación.
---

# Metaprogramación en Mojo 1.2.0

Mojo distingue **parámetros** (valores de tiempo de compilación, entre `[]`) de **argumentos** (valores de runtime, entre `()`). `comptime` es la palabra clave para evaluación en tiempo de compilación y reemplaza a `alias` y `@parameter if`/`for`.

## `comptime` básico

```mojo
comptime N = 1024
comptime MyType = Int

comptime if N > 512:
    print("grande")

comptime for i in range(3):
    print(i)  # se desenrolla en compilación
```

`comptime for` requiere límites conocidos en compilación; el compilador desenrolla el bucle y especializa el código.

## Parameterización de funciones

```mojo
def repeat[count: Int](msg: String):
    comptime for _ in range(count):
        print(msg)

def main():
    repeat[3]("Hola")  # Hola Hola Hola
```

El compilador crea una versión concreta de `repeat` por cada valor único de `count`; si no puede resolver el parámetro, la compilación falla.

## Parameterización de structs

```mojo
struct FixedBuffer[size: Int]:
    var data: InlineArray[UInt8, size]

    def __init__(out self):
        self.data = InlineArray[UInt8, size](fill=0)

@fieldwise_init
struct MyList[T: AnyType]:
    var items: List[Self.T]
```

Los parámetros pueden ser tipos (`T: AnyType`) o valores (`size: Int`).

## Reflexión con `@fieldwise_init` y `@implicit`

`@fieldwise_init` genera un `__init__` que acepta todos los campos como argumentos. `@fieldwise_init("implicit")` genera además un inicializador implícito, siempre que el struct tenga **exactamente un campo de instancia**.

```mojo
@fieldwise_init("implicit")
struct UserId:
    var value: Int

def greet(id: UserId):
    print("Hola", id.value)

def main():
    greet(42)          # conversión implícita
    greet(UserId(42))  # explícita
```

Para conversiones más complejas, escribe un `@implicit` manual:

```mojo
@fieldwise_init
struct Meters:
    var value: Float64

    @implicit
    def __init__(out self, cm: Int):
        self.value = Float64(cm) / 100.0
```

## `type_of` y despacho por tipo

`type_of(x)` devuelve el tipo de un valor en tiempo de compilación:

```mojo
comptime if type_of(x) == Float32:
    ...
```

## `comptime assert`

```mojo
def checked[N: Int]():
    comptime assert N > 0, "N debe ser positivo"
```

`comptime assert` debe estar dentro del cuerpo de una función o método; no en el ámbito de módulo.

## Anti-patrones

- **No uses `alias`**: fue eliminado; usa `comptime`.
- **No uses `@parameter if`/`@parameter for`**: fueron eliminados; usa `comptime if`/`comptime for`.
- **No uses `comptime for` con límites de runtime**: el compilador no puede desenrollar un rango dinámico.
- **No uses `@fieldwise_init("implicit")` con más de un campo**: el compilador lo rechaza.
- **No abuses de `comptime`**: el código generado puede aumentar el tamaño del binario; úsalo cuando el desenrollado aporte rendimiento real.

## Reglas prácticas

1. **Usa `comptime` para constantes y alias de tipo**; reemplaza `alias`.
2. **Usa `comptime for` para desenrollar bucles pequeños** con límites estáticos.
3. **Parameteriza structs para buffers de tamaño fijo** (`InlineArray[T, N]`).
4. **Usa `@fieldwise_init` para structs de datos** y `@fieldwise_init("implicit")` solo con un campo.
5. **Verifica con `comptime assert`** los invariantes que el compilador puede comprobar.
