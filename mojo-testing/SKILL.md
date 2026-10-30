---
name: mojo-testing
description: Escribe y ejecuta tests unitarios en Mojo 1.2.0 con TestSuite, assert_equal, assert_true, assert_almost_equal, assert_raises y assert_equal_pyobj. Cubre descubrimiento automático, tests parametrizados, tests que lanzan excepciones y ejecución con mojo run. No usa el eliminado subcomando mojo test.
---

# Testing en Mojo 1.2.0

Mojo incluye un framework de tests en `std.testing`: un conjunto de aserciones y el struct `TestSuite` para descubrimiento y ejecución automáticos. El subcomando `mojo test` **fue eliminado**; los tests se ejecutan con `mojo run` contra un runner `TestSuite`.

## Estructura mínima

```mojo
from std.testing import assert_equal, TestSuite

def inc(n: Int) -> Int:
    return n + 1

def test_inc_zero() raises:
    assert_equal(inc(0), 1)

def test_inc_one() raises:
    assert_equal(inc(1), 2)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

`discover_tests` recoge automáticamente las funciones cuyo nombre empieza por `test_`. Cada test debe ser `def test_xxx() raises`.

```bash
mojo run test_quickstart.mojo
```

Salida típica: `PASS`/`FAIL` por test, tiempo de ejecución y resumen final. Un test falla si lanza un `Error`; si no, pasa.

## Aserciones disponibles

| Función | Uso |
|---------|-----|
| `assert_equal(lhs, rhs)` | Igualdad exacta; falla si `lhs != rhs` |
| `assert_true(cond)` | `cond` debe ser `True` |
| `assert_false(cond)` | `cond` debe ser `False` |
| `assert_almost_equal(lhs, rhs, tolerance)` | Igualdad con tolerancia (Float) |
| `assert_raises()` | Context manager; el bloque debe lanzar |
| `assert_equal_pyobj(lhs, rhs)` | Igualdad entre `PythonObject` |

```mojo
from std.testing import assert_almost_equal, assert_raises, assert_true

def test_math() raises:
    assert_almost_equal(0.1 + 0.2, 0.3, tolerance=1e-9)
    assert_true(2 > 1)

def test_raises() raises:
    with assert_raises():
        raise Error("boom")
```

## Tests parametrizados con `comptime for`

```mojo
from std.testing import assert_equal, TestSuite

def test_squares() raises:
    comptime for i in range(5):
        assert_equal(i * i, i ** 2)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

## Tests con `PythonObject`

```mojo
from std.python import Python
from std.testing import assert_equal_pyobj, TestSuite

def test_python_list() raises:
    var py_list = Python.list(1, 2, 3)
    assert_equal_pyobj(py_list, Python.list(1, 2, 3))
```

## Organización y CI

- Un archivo `test_<modulo>.mojo` por módulo; el `main()` llama a `discover_tests`.
- En CI, ejecuta cada archivo con `mojo run test_*.mojo`; el código de salida es distinto de cero si algún test falla.
- `TestSuite` acepta `cli_args` para filtrar tests; consulta `TestSuite` en la stdlib para `skip_list`/`allow_list`.

## Anti-patrones

- **No uses `mojo test`**: fue eliminado. Usa `mojo run` con `TestSuite`.
- **No dejes tests sin `raises`**: `discover_tests` espera `def() raises thin -> None`; un test que no lanza puede registrarse pero no detectar fallos.
- **No uses `assert_equal` para floats**: usa `assert_almost_equal` con tolerancia.
- **No dependas del orden de ejecución**: los tests deben ser independientes.

## Reglas prácticas

1. **Un `main()` por archivo de tests** que llame a `discover_tests[__functions_in_module()]().run()`.
2. **Nombra los tests `test_<comportamiento>`** para que el descubrimiento funcione.
3. **Usa `assert_raises()`** para verificar errores esperados.
4. **Mantén los tests rápidos**: sin I/O pesado ni sleeps.
5. **Ejecuta con `mojo run`** en CI; falla el job si el código de salida es distinto de cero.
