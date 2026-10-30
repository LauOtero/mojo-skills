# Tests y CI

Cuando añadas tests `pytest` para el modelo portado (tests de capa, tests de módulo o tests de humo del modelo bajo `max/`), minimiza el número de llamadas a `compile()` por archivo de test. Un archivo que recompila para cada combinación de parámetros, o que contiene muchos módulos compilados independientemente, puede agotar el timeout en CI.

Los patrones de abajo evitan timeouts. Úsalos juntos.

## Patrón 1: compila una vez mediante un fixture con alcance de módulo

Compila el módulo una vez en un fixture y reutilízalo en todos los tests del archivo. Usa dimensiones simbólicas en los tipos de entrada, para que un módulo compilado acepte todas las formas, y combínalo con `@pytest.mark.parametrize` para variar entradas sin recompilar.

```python
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.experimental.tensor import TensorType

from my_port.layers import MyMLP


@pytest.fixture(scope="module")
def mlp():
    device = Accelerator()
    layer = MyMLP(hidden_size=64, intermediate_size=256)
    layer.to(device)
    return layer.compile(TensorType(DType.bfloat16, ["seq", 64], device))


@pytest.mark.parametrize("seq_len", [1, 7, 128])
def test_forward(seq_len, mlp): ...  # ejercita el módulo ya compilado
```

pytest crea un fixture con alcance de módulo una vez por archivo de test, así que cada test del archivo comparte el mismo módulo compilado.

## Patrón 2: paraleliza módulos diferentes con `shard_count`

Cuando un mismo archivo deba compilar módulos diferentes (dtypes distintos, variantes de kernel o formas de distribución), no dividas el archivo a mano. Usa el sharding de tests de Bazel para repartir el trabajo entre workers de CI en paralelo:

```bzl
modular_py_test(
    name = "test_attention",
    srcs = ["test_attention.py"],
    shard_count = 3,
)
```

Bazel lanza `N` procesos pytest en paralelo. Cada uno corre aproximadamente `1/N` de los tests. Los fixtures con alcance de módulo son por proceso, así que cada shard compila solo los módulos que sus tests necesitan, y las compilaciones ocurren en paralelo.

El plugin `pytest-shard` añade marcadores finos:

- `@pytest.mark.shard_group("name")` fija cada test del grupo al mismo shard para que compartan compilación.
- `@pytest.mark.unique_shard` da a un test costoso su propio shard.

## Cuándo aplicar

Recurre a cada patrón en situaciones distintas:

- Usa fixtures en cualquier archivo de test que ejercite un módulo compilado más de una vez.
- Usa sharding en cualquier archivo de test cuya suma de wall time se acerque al timeout de CI, especialmente cuando el archivo contiene módulos compilados independientemente que no pueden compartir un fixture.

Los fixtures minimizan compilaciones donde los módulos son compartibles, y el sharding paraleliza compilaciones donde no lo son.