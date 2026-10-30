# Mojo idiomático

La skill principal `mojo-syntax` hace que el código compile; estas reglas hacen que pase la revisión. Antes de enviar o revisar Mojo, escanea el diff en busca de cada patrón INCORRECTO de abajo. Las primeras cinco secciones son las que los revisores señalan con más frecuencia en código escrito por agentes.

## 1. Usa los helpers existentes; nunca los escribas a mano

Busca en `std.math`, `std.gpu`, `nn/`, `linalg/` y la API de TileTensor antes de escribir matemáticas de índices, reducciones, reshapes o activaciones. El código existente a pocas líneas de distancia normalmente ya usa el helper.

```mojo
# INCORRECTO                              # CORRECTO
(M + BM - 1) // BM                       ceildiv(M, BM)          # std.math
(x + a - 1) // a * a                     align_up(x, a)          # std.math
var s = b // H; var h = b % H            var s, h = divmod(b, H)
1 / (1 + exp(-x))                        sigmoid(x)              # nn.activations
bucle manual de shuffle-reduce            warp.sum(val)           # warp de std.gpu
32 lanes codificados a mano               WARP_SIZE
bucle de tap escalar + cast por elemento  (v.load[w](i) * wts).reduce_add(), cast después de reducir
bucle manual de construcción de string ", ".join(items)
```

`ceildiv` funciona con `Int` y con cualquier escalar `SIMD` (`UInt32`, ...). Prefiere `vectorize` sobre bucles escalares escritos a mano, y llama a intrínsecos de LLVM/NVVM en lugar de ensamblador en línea.

Las cargas y almacenes vectorizados de punteros usan por defecto la alineación escalar, lo que puede dividirlos en accesos escalares. Pasa la alineación real; la dirección debe satisfacerla realmente, o el acceso falla:

```mojo
# INCORRECTO
var v = ptr.unsafe_load[width=8](off)
# CORRECTO
comptime align = align_of[SIMD[dtype, 8]]()
var v = ptr.unsafe_load[width=8, alignment=align](off)
```

## 2. No castees valores que ya tienen el tipo correcto

Los campos `thread_idx`, `block_idx`, `block_dim`, `grid_dim` devuelven `Int`. No los envuelvas en `Int(...)`, y no copies ese patrón de código antiguo. Mantén un valor en su tipo natural; un downcast seguido de un upcast es una bandera de revisión.

```mojo
# INCORRECTO                                      # CORRECTO
var token = Int(block_idx.x)                      var token = block_idx.x
var tile_m = UInt32(block_idx.y)  ...  Int(tile_m)   var tile_m = block_idx.y
```

Castea solo en la frontera donde el otro tipo es realmente necesario (por ejemplo, al comparar contra un offset `UInt32`).

## 3. Deja que los parámetros se infieran

Omite las listas de parámetros que los argumentos ya determinan. No añadas un parámetro `rank`/`dtype`/`layout` que el tipo del argumento ya lleva.

```mojo
# INCORRECTO
Self.run_at_tile[c_layout, a_layout, b_layout, c_engine, a_engine, b_engine](c, a, b, m, n)
epilogue[dtype, width, alignment=alignment](idx, val)
def f[rank: Int, axis: Int](shape: IndexList[rank]): ...
x: TileTensor[DType.float32, layout, origin, Engine=engine]

# CORRECTO
Self.run_at_tile(c, a, b, m, n)
epilogue[alignment=alignment](idx, val)
def f[axis: Int](shape: IndexList): ...
x: TileTensor[DType.float32, ...]        # `...` siempre va al final
def g[dtype: DType, //, filter_len: Int](...)   # parámetros solo-inferidos antes de `//`
```

Comprueba el valor por defecto del parámetro antes de eliminarlo — quitar un `[2]` explícito donde el valor por defecto es `1` cambia el comportamiento.

Un valor por defecto no es una restricción: con `b_type: DType = a_type`, una comprobación que compare los dos nunca se dispara para los llamadores que omiten `b_type`.

## 4. Construye coordenadas desde tuplas, no desde `IndexList`/`Index`

Las tuplas convierten implícitamente a tipos de coordenadas. Una tupla de un solo elemento es `(x,)`. No crees un `IndexList` o `Index` solo para pasarlo después.

```mojo
# INCORRECTO                              # CORRECTO
Index(m, n)                               (m, n)
f(Index(tokens))                          f((tokens,))
t.load[width=1](IndexList[2](i, k))       t.load((i, k))
Coord(IndexList[2](a, b))                 Coord(a, b)
Coord(index_list[0], ..., index_list[N])  Coord(index_list)
Coord(Idx[1], Idx[5])                     coord[1, 5]
t.load((Idx[5], Idx[3]))                  t.load(coord[5, 3])
```

Si una tupla de tipos mixtos falla al unificar (por ejemplo, `Tuple[Int, UInt32]`), castea el elemento impar; no recurras a `IndexList`. Para formas y tamaños totalmente estáticos, prefiere el helper `comptime` `coord`.

## 5. `comptime` para trabajo conocido en tiempo de compilación

```mojo
# INCORRECTO                              # CORRECTO
for i in range(CHUNK):                    comptime for i in range(CHUNK):
comptime for anidado x3                   comptime for a, b, c in product(range(X), range(Y), range(Z)):  # std.itertools
for i in range(rank - 1, -1, -1):         comptime for i in reversed(range(rank)):
var scale = 1.0 / sqrt(Float32(D))        comptime scale = Float32(1.0) / sqrt(Float32(D))
if ctx.api() == "metal":  (runtime)       comptime if ...:   # las comprobaciones de target/vendor son siempre comptime
```

No desenrolles un bucle que no se beneficie — los revisores preguntan "¿necesitas desenrollar aquí?". Si una llamada matemática `comptime Float32` falla al plegarse en GPU, abre un issue en lugar de cambiar silenciosamente el dtype.

Mantén las dimensiones comptime estáticas hasta los layouts. Las heurísticas de dispatch leen `static_shape`; pasar un valor comptime como un `Int` de runtime lo borra y enruta silenciosamente a una ruta más lenta.

```mojo
# INCORRECTO: N es comptime, pero c.static_shape[1] == -1
var c = TileTensor(ptr, row_major((M, N)))
# CORRECTO
var c = TileTensor(ptr, row_major((M, Idx[N])))
```

## 6. No uses `rebind` a menos que los tipos ya coincidan

`rebind` es una promesa de tiempo de compilación de que dos tipos son idénticos tras la elaboración, no un cast. Intenta eliminar cada `rebind` que añadas o toques; consérvalo solo si el código deja de compilar sin él. Para reinterpretar un puntero, usa `ptr.unsafe_bitcast[T]()`; para convertir un valor `SIMD`, usa `x.cast[DType]()`.

```mojo
# INCORRECTO: rebindea un tipo a sí mismo
p.init_pointee_copy(rebind[StrideType](StrideType()))
# CORRECTO
p.init_pointee_copy(StrideType())

# INCORRECTO: puentea layouts de TileTensor diferentes; la forma de Coord almacenada configura la ABI,
# así que row_major[4, 4]() y row_major((Idx[4], 4)) difieren en tamaño
rebind[TileTensor[dtype, LayoutB, origin]](tile_a)
# CORRECTO: construye la tesela con el layout que el consumidor espera

# INCORRECTO: deletrear parámetros y luego rebindear para que la inferencia funcione
# CORRECTO: pasa tuplas/valores directamente y deja que los parámetros se infieran
```

- Después de una migración, comprueba si los rebinds antiguos siguen siendo necesarios.
- Legítimo: `rebind[IndexList[rank]](x.canonicalize())` (válido solo después de `canonicalize()`), reconciliación de ancho/dtype de `SIMD` dentro de lambdas de epílogo, y discrepancias nominales en la frontera de retorno.
- Usa un helper de coerción existente (por ejemplo, `_coerce_dynamic[T]`) en lugar de repetir `rebind[T](Scalar[...](...))`.
- Para una restricción que el compilador no puede probar, usa una cláusula `where` o `comptime assert`, no un rebind que oculte la discrepancia. Una cláusula `where` empuja la comprobación al llamador (detectada por el LSP, pero los llamadores genéricos deben repetirla); un `comptime assert` la empuja al cuerpo (igualmente detectada, solo en elaboración). Usa `where` para APIs llamadas mayoritariamente con tipos concretos (`UInt8(1) in (1, 2, 3)` requiere `T in Self.Ts`), y `comptime assert` para APIs pensadas para contextos genéricos, donde una cláusula `where` obligaría a cada llamador a repetir la restricción.
- Más de un rebind por función, o uno que no puedas justificar en una frase, es una bandera roja.

Lo mismo aplica a los orígenes: no llames a `as_any_origin()` ni uses `ImmutAnyOrigin` sin necesidad, y usa `TileTensor.as_imm()` en lugar de sobrescribir `mut=False`.

## 7. Marca los datos de solo lectura como inmutables

Las entradas que nunca se escriben toman orígenes inmutables para que la firma muestre qué argumentos son salidas. Cuestiona cada `MutAnyOrigin`.

```mojo
# INCORRECTO                                    # CORRECTO
src: Pointer[Float32, MutAnyOrigin]            src: ImmPointer[Float32, _]
```

Que un llamador tenga datos mutables no hace que el parámetro sea mutable. Un valor mutable convierte implícitamente a un parámetro inmutable con un origen genérico (`ImmTileTensor[dtype, L, _]`); con un origen concreto como `ImmutAnyOrigin`, el llamador pasa `.as_imm()`. Nunca ensanches un parámetro de solo lectura a `mut=True` para que coincida con un llamador.

Las cargas desde datos inmutables son invariantes por defecto; no pases `invariant=True` ni otros argumentos no-op. Prefiere los prefijos de tipo con nombre para describir la mutabilidad de los datos subyacentes, como `ImmPointer`, `MutPointer`, `ImmTileTensor` y `MutTileTensor`, sobre el prefijo `mut=True` o `mut=False`.

## 8. TileTensor sobre punteros crudos

Los kernels usan TileTensor, indexación por coordenadas y sintaxis de slice en lugar de offsets de puntero planos. El acceso directo a punteros era un workaround para los acantilados de rendimiento de los tipos de tensor anteriores (llevaban bytes muertos); un TileTensor totalmente estático tiene la misma ABI que un puntero, así que indexa el tensor directamente.

```mojo
# INCORRECTO                              # CORRECTO
out.ptr[row * N + col] = v                out[row, col] = v
src.slice(...)                            src[:h, :w]
```

No indexes punteros crudos ni calcules offsets a partir de strides en kernels. Algunos aceleradores no exponen punteros crudos (sus tensores están respaldados por un memref), así que `ptr[offset]` y las matemáticas manuales `i * stride + j` no portan a ellos. Deja que TileTensor sea dueño del layout: indexa por coordenada, y usa `tile`, `vectorize` y `distribute` para subvistas en lugar de aritmética de punteros.

```mojo
# INCORRECTO
var p = t.unsafe_ptr()
p[row * row_stride + col] = v
var tile_ptr = p + (tm * BM) * N + tn * BN
# CORRECTO
t[row, col] = v
var tile = t.tile[BM, BN](tm, tn)
```

No uses `TileTensor.ptr`. Cuando necesites el puntero, llama a `TileTensor.unsafe_ptr()`, y solo cuando sepas que el tensor está respaldado por un puntero (no, por ejemplo, un memref).

Los usos restantes de `raw_load`/`raw_store` necesitan un issue de seguimiento.

## 9. Simplifica el flujo de control; elimina duplicación

- Devuelve temprano en lugar de anidar `if/else`; sin `else` después de `return`.
- Usa `clamp(x, lo, hi)` sobre llamadas anidadas a min/max, en lugar de `min(max(x, lo), hi)` o `min(x, max(lo, hi))`.
- Itera colecciones directamente (`for d in dtypes:`), no por índice.
- Empieza los acumuladores en su identidad (`accum = 0`) en lugar de pelar la primera iteración.
- Deriva listas de las existentes (`float_dtypes.join(integer_dtypes)`).
- Eleva bloques repetidos a un helper o alias `comptime`.
- Devuelve una `Tuple` desde un helper en lugar de hacer que los llamadores hagan aritmética de offsets `+1, +2`; empaqueta argumentos escalares de kernel en un `InlineArray` y buffer en lugar de varios buffers o `Pointer`s falsos.
- Prefiere t-strings sobre concatenación de strings. Por ejemplo:

  ```mojo
  # INCORRECTO
  "expected " + String(expected_dtype) + " but got " + String(actual_dtype)
  # CORRECTO
  t"expected {expected_dtype} but got {actual_dtype}"
  ```

## 10. Coincidencia de patrones: `match` sobre cadenas `elif`

`match` (y `comptime match`, `if case`, `while case`) están estabilizados. Cuando varios casos despachan sobre un mismo sujeto, un `match` reemplaza la cadena `elif`. Cuando las ramas tocan expresiones no relacionadas, mantén `if`/`elif`. Dos condiciones de seguridad antes de reescribir: `match` evalúa su sujeto **una vez** y cada caso ve esa instantánea, así que una escalera cuyas ramas reevalúan o mutan el sujeto entre pruebas no es equivalente; y un `comptime match` es la forma para especialización guiada por parámetros — un `match` de runtime sobre parámetros compila ambos lados de cada rama.

```mojo
# INCORRECTO: escalera elif sobre un sujeto
if dtype == .float32:
    return 1
elif dtype == .float16 or dtype == .bfloat16:
    return 2
else:
    return 3

# CORRECTO
match dtype:
    case .float32:
        return 1
    case .float16 | .bfloat16:
        return 2
    case _:
        return 3
```

El sujeto se evalúa una vez; los casos se prueban en orden. Para un sujeto parámetro, `comptime match` garantiza dispatch en tiempo de compilación y enlaza nombres como parámetros (sin `var`/`ref` ahí — son sin sentido y se rechazan). Dos reglas duras para la forma comptime: el sujeto debe ser un valor de tiempo de compilación (un parámetro o expresión comptime — `comptime match` sobre un argumento de runtime es "cannot use a dynamic value in 'match' subject"; usa un `match` de runtime ahí), y los guards también deben ser evaluables en tiempo de compilación.

```mojo
comptime match (m, n, k):
    case (16, 16, 16):
        ...                      # ruta 16³ ajustada a mano
    case (mt, nt, kt) if mt * nt <= 256:
        ...                      # tamaños ligados usables como parámetros
    case _:
        ...
```

La forma de kernel más común es un dispatch de `dtype` (parámetro) o valor de enum (`X.tag`). La reescritura mecánica — ramas `.NAME` para un parámetro `DType`, ramas `== value` para un tag `Int`:

```mojo
# INCORRECTO                                      # CORRECTO
comptime if dtype == .float32:                    comptime match dtype:
    path_f32()                                        case .float32:
elif dtype == .float16 or dtype == .bfloat16:              path_f32()
    path_f16()                                    case .float16 | .bfloat16:
else:                                                  path_f16()
    comptime assert False, "unsupported"          case _:
                                                      comptime assert False, (
                                                          t"unsupported {dtype}"
                                                      )
```

### Formas de patrón

```mojo
match x:
    case 0 | 1 | 2:              # literales + patrones or
    case == expected:            # compara contra un valor/expresión; ver abajo
    case == (expected + 1):
    case n if n < 0:             # guard; nunca cuenta para exhaustividad
    case (0, y):                 # tuplas, anidadas arbitrariamente
    case Point(x=x, y=0):        # campos de struct por nombre
    case {x=0, y=0}:             # campos tipados del sujeto por nombre
    case == {1, offset=2}:       # compara contra un valor de lista de inicialización
    case (0, var x) | (var x, 1):# patrón or: los mismos nombres enlazados en cada
    case .Some(0):               #   alternativa (los casos sin payload no enlazan nada)
    case .Some(ref v):           # ref al payload mutable escribe de vuelta
    case var y:                  # copia propia (mutable); nombre desnudo es imm
    case 0 as n:                 # enlaza el sujeto completo tras probarlo
    case _:                      # wildcard; irrefutable, cierra el match
```

### `== expr` — cuándo una rama lo necesita

Usa la ortografía nativa de la rama cuando compara por sí sola; añade `==` solo donde la gramática lo requiera. Matriz verificada por el compilador:

| Rama | ¿Compara sola? | Notas |
|------|----------------|-------|
| `case 1:` / `case "a":` / `case True:` | sí | `== 1:` **avisa "redundante"** |
| `case -1:` (literal negativo) | no — error | escribe `case == -1:` |
| `case {1, 2}:` (lista de inicialización) | no — error | nombra los campos del sujeto, como `case {x=1, y=2}:` |
| `case .Some(0):` / `case DType.int32:` / `case _Ops.BARRIER.value:` | sí | los tags de enum y cualquier ruta con puntos comparan; sin `==` |
| `case LOAD_DRAM:` (nombre desnudo) | **enlaza** | incluso un alias comptime — sombrea silenciosamente, las ramas posteriores avisan inalcanzables; escribe `case == LOAD_DRAM:` |
| `case e + 1:` / `case a[i]:` / llamadas | no — error | escribe `case == (e + 1):` |
| `case == None:` | — | `None` no es un literal plano; `==` requerido |

La regla: **una ruta con puntos o un caso de enum compara; un nombre desnudo enlaza; todo lo demás necesita `==`.** Recurre a los payloads (`.Some(expected)` enlaza; `.Some(== expected)` compara).

```mojo
# INCORRECTO: enlaza x a un nuevo nombre W; no compara
match x:
    case W:
    case _:

# CORRECTO — == solo donde se requiere
match x:
    case == W:                    # nombre desnudo: == requerido
    case _Ops.BARRIER.value:      # ruta con puntos: sin ==
    case == (expected + 1):       # expresión: == requerido
    case _:
```

Los tipos son valores comptime, así que `comptime match T` con ramas `== Type` es el "type switch". No hay `is` — `==` es identidad de tipo en Mojo:

```mojo
comptime match type_of(x):       # builtin es `type_of`; `typeof` no analiza
    case == Float32 | == Float64:
        ...
    case _:
        ...

# Para una pregunta de un solo tipo, comptime if se lee mejor:
comptime if type_of(x) == Float32:
    ...
```

No puedes elevar alternativas `==` a un conjunto compartido (`== FPTypes` compara contra el valor del conjunto) ni poner predicados dentro de patrones — deletrea cada `== T`, y mueve las preguntas de trait a un guard sobre el valor *dtype* (`case dt if dt.is_floating_point():` — un *tipo* enlazado no tiene tal método) o a un `comptime if` plano.

### Structs disfrazados de enum: conforma a `EnumLike`

Cuando un match (o una cadena elif que lo sería) despacha sobre un struct cuyos casos lógicos son constantes `comptime Self(...)` — un struct actuando como enum — conforma a `EnumLike` y usa ramas `.CASE`: dispatch por discriminante en lugar del `__eq__` del struct, más diagnósticos de match exhaustivo. Miembros del contrato:

|Miembro|Contrato|
|---|---|
|`_enum_case_names`|`ParameterList.of[...].values`: nombres de constantes en orden de índice de caso.|
|`_enum_case_types`|Valores de `TypeList` en el mismo orden; `NoneType` para casos sin payload.|
|`_get_enum_discriminant(self) -> Int`|Mapea el valor de respaldo a su índice de caso.|
|`_unsafe_get_enum_payload[id: Int](ref self)`|Devuelve `ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[id]`; los enums sin payload hacen `comptime assert False`.|

Si pueden existir valores de respaldo sin nombre (por ejemplo, un centinela `-1` "no soportado"), establece `comptime _enum_is_exhaustive = False` y mantén `case _`. Para structs respaldados por strings, mapea strings a índices de caso en `_get_enum_discriminant` (un `match self.name:` de runtime está bien); `_enum_case_names` lista los **nombres de constantes**, no los strings de respaldo. Conformidades en el árbol: `TuningGroup`, `FlashAttentionAlgorithm`, `GEMMKind`. Conversiones recientes de la misma forma: `UMMAKind`, `MaskName` (ver los PRs de kernel que aplican esta regla).

```mojo
# INCORRECTO — Kind es un struct con constantes FIRST/SECOND pero sin EnumLike
if kind == Kind.FIRST:
    return 10
elif kind == Kind.SECOND:
    return 20
else:
    return -1

# CORRECTO — los valores de respaldo sin nombre siguen siendo posibles
@fieldwise_init
struct Kind(EnumLike, TrivialRegisterPassable):
    var value: Int32
    comptime FIRST = Self(0)
    comptime SECOND = Self(1)
    comptime _enum_is_exhaustive = False
    comptime _enum_case_names = ParameterList.of[
        "FIRST".value, "SECOND".value
    ].values
    comptime _enum_case_types = TypeList.splat[2, NoneType].values

    def _get_enum_discriminant(self) -> Int:
        return Int(self.value)

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[
        Trait=AnyType, Self._enum_case_types
    ]()[id]:
        comptime assert False, "Kind no tiene payload"

def classify(kind: Kind) -> Int:
    match kind:
        case .FIRST:
            return 10
        case .SECOND:
            return 20
        case _:
            return -1
```

### Exhaustividad y advertencias

El compilador avisa sobre sujetos `EnumLike` no exhaustivos (incl. `Bool` y tuplas de esos), sobre casos duplicados y sobre ramas inalcanzables. Los guards nunca contribuyen a la cobertura, así que un `case _ if cond:` **no** es un catch-all. En kernels, prefiere un match exhaustivo con un `case _:` que haga `comptime assert`, para que un nuevo valor de enum falle en tiempo de compilación, no en producción:

```mojo
comptime match dtype:
    case .float32 | .float64:
        ...
    case _:
        comptime assert False, t"dtype no soportado {dtype}"
```

Una vez cubiertos todos los casos de un match de runtime, un `case _:` final avisa inalcanzable — elimínalo. Los enums abiertos (`_enum_is_exhaustive = False`) todavía necesitan `case _`; cubrir los casos conocidos no cierra un conjunto abierto.

El modelo de cobertura (qué cuenta como "cubierto"):

| Sujeto / patrón | Comportamiento de cobertura |
|-----------------|----------------------------|
| `case _` o enlace desnudo | irrefutable — cierra todo el espacio |
| `case X if cond:` (cualquier guard) | nunca cuenta para la cobertura |
| `case .Some(0):` | no cubre `Some` (payload refinado) |
| `case .Some(_):` | cubre `Some` |
| `case (True, True):` sobre `Bool × Bool` | cubre la combinación; `(False, False)` al lado todavía deja dos celdas abiertas — la cobertura es por combinación, no por campo |
| enum abierto (`_enum_is_exhaustive = False`) | los casos conocidos nunca cierran el conjunto; `case _` sigue siendo alcanzable |
| literales `Int`/`String` | los duplicados avisan, pero no hay requisito de exhaustividad |

"Sin advertencia" no es prueba de dispatch exhaustivo o no redundante: los duplicados de payload refinado y los literales de tipo abierto anidados no se diagnostican, y la identidad de literal se basa en la ortografía (`4` vs `0x4` son claves diferentes).

### `if case` / `while case` / `or_else`

Elige por forma, en orden:

- **Fallback a un valor por defecto** → `opt.or_else(default)`. Una expresión, sin rama: `var global_rank = my_rank.or_else(Int(ctx.id()))`. Funciona con Optionals de runtime y comptime por igual. Esto reemplaza el ternario `opt.value() if opt else default` y la danza de dos líneas de poner por defecto y luego sobrescribir.
- **Rama según presencia** (usa el payload solo cuando existe, otro flujo de control en caso contrario) → `if case .Some(x) = opt:`. El patrón es el guard Y el enlace.
- **Dispatch según la forma del payload** (múltiples casos, o el lado None lleva su propia lógica) → `match opt: case .Some(x): ... case .None: ...`.
- ¿Presencia garantizada por construcción (una condición envolvente ya lo prueba, y la ausencia es un bug del llamador)? Mantén `.value()` — aborta ruidosamente en lugar de saltarse trabajo silenciosamente.

```mojo
# INCORRECTO: guard y luego desenvolver
var global_rank = my_rank.value() if my_rank else Int(ctx.id())

# CORRECTO: el fallback es una expresión
var global_rank = my_rank.or_else(Int(ctx.id()))
```

Para la forma de rama según presencia, `if case pattern = value:` evalúa el valor una vez, prueba un patrón y enlaza nombres solo para ese bloque; el fallo pasa al siguiente `elif`/`else`. El patrón es tanto el guard como el enlace, así que desenvuelve un `Optional` sin la danza de guard `is not None` / `Bool(opt)` + `.value()`:

```mojo
# INCORRECTO: guard y luego desenvolver
if sink_weight is not None:
    sink_logit = sink_weight.value()

var stored = get_global_or_null(key)
if not stored or Int(stored.value()) != Int(record):
    raise Error("missing")

# CORRECTO: el patrón ES el guard; el payload enlaza en línea
if case .Some(w) = sink_weight:
    sink_logit = w

if case .Some(s) = get_global_or_null(key):
    if Int(s) != Int(record):
        raise Error("missing")

while case .Some(item) = get_next():   # consume mientras produce
    process(item)
```

`if case` compone con la disciplina de rama else: un desenvolvimiento en una rama de una cadena condicional más grande sigue siendo legible como `if case .Some(x) = opt: ... elif <otra cond>: ...`. Prefiérelo incluso cuando el guard sea `comptime` (`comptime if case .Some(fn) = wrapper:`).

Ambos componen con `comptime` — los parámetros enlazan como parámetros:

```mojo
comptime if case == Int = T:
    ...
elif case == String = T:
    ...
else:
    ...
```

Un patrón or debe enlazar el mismo nombre de forma compatible en cada alternativa; `ref` enlaza una referencia a cualquiera de los elementos que coincidió, y mantiene la mutabilidad del sujeto:

```mojo
match point:
    case (2, ref x) | (ref x, 3):
        use(x)                  # x referencia point[1] o point[0]
    case _:
        ...
```

### Comprobaciones de revisión

- Una cadena de 2+ ramas `elif x == ...` sobre un sujeto se convierte en un `match` — a menos que las ramas reevalúen o muten el sujeto entre pruebas; `match` toma una instantánea del sujeto una vez.
- Las cadenas `elif` sobre expresiones no relacionadas se quedan como `if`/`elif`. Los sujetos parámetro toman `comptime match`.
- Un nombre desnudo en un patrón enlaza; si se pretendía una comparación, usa `== name` (excepto casos de enum: `case Color.red:` / `case .red:`).
- Los casos con guard no cuentan para la exhaustividad; la falta de *cobertura* avisa (el `case _:` final de un match cubierto avisa inalcanzable en su lugar).
- Los enlaces `ref` necesitan un origen de memoria — `case ref y:` sobre un sujeto pasable por registro es un error; úsalo dentro de tuplas o payloads de enum mutables.

## 11. Nomenclatura

- Sin clase de fabricante o dispositivo en APIs genéricas (`CompilationTarget(MI355X)`, no una variante prefijada con `gpu_`).
- Acrónimos en mayúsculas: `TMem`, no `Tmem`.
- No sombrees builtins (`min_value`, no `min`).
- Descriptivo sobre críptico (`tA`, `tR` se señalan).
- Declara siempre las locales con `var`.

## 12. Diseño de API

- `Optional[T]` en lugar de valores por defecto centinela.
- Un punto de entrada genérico (`is[NVIDIA_GPU]()`) sobre un conjunto creciente de métodos `is_xyz()`.
- Toma `StringSlice`/`Span`/`Iterable` en lugar de tipos propietarios.
- Nunca aceptes un argumento y lo ignores silenciosamente en algunos targets.
- Mantén las funciones avanzadas de bajo nivel fuera de las exportaciones del módulo.
- Para `call_location()`, usa un wrapper delgado `@inline(.always)` que reenvíe la ubicación a un `_impl` `@inline(.never)`.
- Usa resultados con nombre con `rebind[type_of(out)](x)` en lugar de repetir un tipo de retorno largo (solo cuando el rebind sea necesario por lo demás).

## 13. Invariantes y hechos de hardware

- Consulta constantes (`WARP_SIZE`, límites del dispositivo); no codifiques `32` o `128` a mano.
- Declara supuestos de forma y configuración con `comptime assert` / `debug_assert`; no elimines comprobaciones de límites existentes.
- En tests, usa centinelas `NaN`/`Inf`, no números mágicos arbitrarios, y usa longitudes primas y que crucen teselas (`13`, `63`/`64`/`65`) para que el enmascaramiento y el código de cola se ejecuten.
- Una corrección de escritura fuera de límites necesita una banda de guarda después de la salida, rellena con un valor que el kernel no pueda producir y comprobada después; comparar elementos dentro del rango no puede ver el desbordamiento.
- La captura de grafo de dispositivo reproduce las elecciones del host, así que el código del host no debe elegir un kernel ni dimensionar una rejilla a partir de valores por petición o de datos del dispositivo copiados de vuelta al host, y debe manejar lanzamientos de calentamiento con `M == 0`.
- Declara `MAX_THREADS_PER_BLOCK_METADATA` para que coincida con el tamaño de bloque real.

## 14. Escribe kernels portables; limita solo lo específico del fabricante

Especializar un kernel para un fabricante está bien cuando usa el hardware de ese fabricante (TMA, WGMMA, tcgen05 en NVIDIA; MFMA, esperas `lgkmcnt` en AMD; ensamblador en línea). Un kernel construido solo con primitivas genéricas (`thread_idx`, `block_dim`, `barrier`, `warp.sum`, `WARP_SIZE`, memoria compartida, TileTensor) no es específico del fabricante en el que se escribió, así que no lo limites ni limites su test a ese fabricante por defecto. Esto aplica en ambas direcciones: un kernel escrito y ajustado en AMD no es más solo-AMD que uno escrito en NVIDIA es solo-NVIDIA.

```bzl
# INCORRECTO: kernel genérico, limitado al fabricante en el que se escribió
gpu_constraints = ["//:nvidia_gpu"],
gpu_constraints = ["//:amd_gpu"],
# CORRECTO: cualquier GPU (añade un select `incompatible` de Apple solo si falla allí)
gpu_constraints = ["//:has_gpu"],
```

- Antes de añadir una puerta de fabricante, ejecuta el kernel también en el otro fabricante (una caja de desarrollo AMD MI300X/MI355X o NVIDIA H100/B200, o deja que el otro carril de CI lo ejecute). Si también tiene una ruta de host, intenta eliminar la restricción de GPU por completo para que se ejecute en todos los aceleradores.
- Al revisar, si un kernel limitado a un fabricante no usa intrínsecos de ese fabricante, pide al autor que intente habilitarlo para todas las GPUs, o todos los aceleradores.
- Mantén pequeña la parte específica del fabricante: pon la ruta rápida detrás de `comptime if is_nvidia_gpu()` / `is_amd_gpu()` y mantén un fallback genérico, en lugar de limitar todo el kernel.
- No limites optimizaciones genéricas al hardware. Las cargas y almacenes vectorizados, `vectorize`, el desenrollado y las teselas más anchas por hilo suelen ayudar a todos los fabricantes, así que aplícalos incondicionalmente y dimensiónalos con `simd_width_of` en lugar de ponerlos detrás de `is_nvidia_gpu()` o `is_amd_gpu()`. Limita uno solo si un benchmark muestra que regresiona a un fabricante.
- Si realmente falla en otro fabricante, limítalo con un `TODO(<ticket>)` que nombre el fallo (un intrínseco faltante o una brecha de numérica), no solo "solo NVIDIA" o "solo AMD".
- No codifiques los hechos de un fabricante (tamaño de warp 32 vs 64, tamaño de memoria compartida) en código genérico; ver §13.

## 15. Higiene

- Cada hack temporal o abuso de API lleva un `TODO(<ticket>)` que enlaza a un issue archivado.
- Mantén los comentarios cortos: una o dos líneas sobre el *porqué*, nunca un párrafo. No expliques lo que el código muestra claramente; elimina comentarios que no añaden nada (`# (basado en TileTensor)`).
- No hagas commit de estado de agente o notas de diseño junto al código fuente.
- Las entradas de changelog describen el estado final, no los pasos intermedios.
