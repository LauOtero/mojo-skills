---
name: mojo-api-design
description: Diseña APIs idiomáticas en Mojo 1.2.0. Cubre traits, @fieldwise_init, @implicit, @export, convenciones de argumentos (imm/mut/var/deinit/out/ref), Optional en lugar de centinelas, StringSpan/Span en lugar de tipos propietarios y visibilidad de módulos. Úsala al diseñar bibliotecas o structs públicos.
---

# Diseño de APIs en Mojo 1.2.0

Una API Mojo idiomática usa traits para interfaces, `@fieldwise_init` para datos, `@implicit` con moderación y convenciones de argumentos que reflejan la propiedad. El objetivo es que el código del llamador sea claro y que el compilador pueda verificar los contratos.

## Traits como interfaces

```mojo
trait Shape:
    def area(self) -> Float64: ...

struct Circle(Shape):
    var radius: Float64

    def area(self) -> Float64:
        return 3.14159 * self.radius * self.radius

struct Square(Shape):
    var side: Float64

    def area(self) -> Float64:
        return self.side * self.side
```

Los traits pueden tener implementaciones por defecto. Para despacho exhaustivo, conforma a `EnumLike` y usa `match`/`case` (ver `idiomatic-mojo.md` §10).

## `@fieldwise_init` y `@implicit`

`@fieldwise_init` genera el constructor por campos; `@fieldwise_init("implicit")` genera además un inicializador implícito, pero **solo si el struct tiene exactamente un campo de instancia**.

```mojo
@fieldwise_init
struct Point:
    var x: Float64
    var y: Float64

@fieldwise_init("implicit")
struct UserId:
    var value: Int
```

Para conversiones más complejas, usa `@implicit` en un constructor manual con un solo argumento.

## Convenciones de argumentos

| Convención | Uso |
|------------|-----|
| `imm` (implícito) | Solo lectura; el llamador conserva la propiedad |
| `mut` | Referencia mutable |
| `var` | Copia propia (transferencia o copia) |
| `deinit` | Consume el valor |
| `out` | Salida no inicializada (constructores) |
| `ref` | Referencia con origen explícito |

Una función que no muta su argumento no debe tomar `mut`. Un constructor toma `out self`; un método que muta toma `mut self`.

## `Optional[T]` en lugar de centinelas

```mojo
# INCORRECTO
def find(x: Int) -> Int:
    return -1  # centinela

# CORRECTO
def find(x: Int) -> Optional[Int]:
    return None
```

## `StringSpan` / `Span` en lugar de tipos propietarios

Para APIs que solo leen, acepta `StringSpan` o `Span[T]` en lugar de `String` o `List[T]`. Evita copias innecesarias y permite al llamador pasar vistas sin asignar.

## `@export` y visibilidad

- `@export` solo para funciones que deben ser visibles fuera del módulo o desde C/Python.
- No exportes funciones de bajo nivel que no formen parte del contrato público.
- Usa `@doc_hidden` para ocultar helpers de la documentación.
- Usa `@deprecated("msg")` para marcar APIs obsoletas antes de eliminarlas.

## Anti-patrones

- **No uses `mut` en parámetros que no se mutan**: engaña al llamador y bloquea el aliasing.
- **No uses centinelas (`-1`, `""`) en lugar de `Optional`**: el tipo no expresa la ausencia.
- **No abuses de `@implicit`**: la conversión implícita dificulta la lectura; úsala solo cuando el tipo envuelve un valor simple.
- **No exportes todo**: mantén la superficie pública pequeña.
- **No uses `String` en APIs de solo lectura**: usa `StringSpan`.

## Reglas prácticas

1. **Traits para interfaces**, structs para datos.
2. **`@fieldwise_init` para structs de datos**; `@fieldwise_init("implicit")` solo con un campo.
3. **`Optional[T]` para valores ausentes**.
4. **`StringSpan`/`Span` para entradas de solo lectura**.
5. **`@export` solo para el contrato público**.
6. **Documenta las invariantes con `comptime assert` o `debug_assert`**.
