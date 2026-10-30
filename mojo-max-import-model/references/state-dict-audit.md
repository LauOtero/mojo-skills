# Auditoría de cobertura del state dict en la carga

`compile(weights=...)` comprueba el state dict contra los parámetros del módulo desde el lado de los parámetros. Cada parámetro necesita un tensor con forma y dtype coincidentes, o la compilación lanza: `KeyError` para un tensor faltante, y `ValueError` para uno desajustado. No comprueba la otra dirección. Un tensor del checkpoint cuyo nombre no coincide con ningún parámetro se omite sin warning.

Todo adaptador de pesos tiene esta brecha. Toma un typo en una tabla de renombrado, un prefijo obsoleto de una revisión de port anterior, o un componente del checkpoint que el módulo nunca construyó. En cada caso el modelo sigue compilando, sigue sirviendo y sigue emitiendo tokens de aspecto real. El error solo aparece en benchmarks de precisión.

Un fallo típico: el checkpoint trae `self_attn.q_norm.weight` y `self_attn.k_norm.weight` en cada capa, pero el módulo de atención del port se copió de un donante sin QK-norm. La compilación tiene éxito porque el módulo no pide pesos de norma, los tensores de norma se descartan, y la generación parece plausible hasta que un benchmark de precisión expone la brecha.

Esta referencia describe una auditoría que reporta cada tensor del checkpoint no consumido en tiempo de carga, en segundos.

## Qué comprueba la auditoría

La auditoría compara dos conjuntos de nombres: las claves que produjo el adaptador, y los nombres de parámetro que el módulo raíz produce desde `parameters`. Cada clave sin parámetro es o un bug o un tensor que el port descarta a propósito. Los tensores descartados incluyen una torre de visión en un port solo de texto, o un head de predicción multi-token que el port no sirve. La auditoría lanza por el primer tipo e ignora el segundo por prefijo.

## De dónde vienen los nombres de parámetro

`parameters` recorre el módulo como un árbol y produce cada tensor bajo su ruta de atributo con puntos, así que los nombres que la auditoría compara son los mismos nombres que `load_state_dict()` y `compile(weights=...)` buscan. Llama al recorrido directamente cuando la auditoría necesite las rutas como conjunto:

```python
from max import tree

parameter_names = set(tree.paths(nn_model, leaf=Tensor))
```

`leaf=Tensor` detiene el recorrido en los tensores, que es el conjunto contra el que la auditoría compara. Quita el argumento y la misma llamada también producirá los atributos no-tensor del módulo, como una config de cuantización o una constante entera, y esos no son parámetros.

## Implementación genérica

Añade esta función al `model.py` de tu slug:

```python
from collections.abc import Iterable
from collections.abc import Set as AbstractSet


def _audit_unconsumed_keys(
    *,
    checkpoint_keys: AbstractSet[str],
    parameter_names: Iterable[str],
    dropped_prefixes: tuple[str, ...] = (),
) -> None:
    """Lanza si un tensor del checkpoint no coincide con ningún parámetro del módulo.

    Las claves bajo ``dropped_prefixes`` pertenecen a componentes que el port
    omite a propósito y se ignoran.
    """
    unconsumed = sorted(
        key
        for key in checkpoint_keys - set(parameter_names)
        if not key.startswith(dropped_prefixes)
    )
    if unconsumed:
        sample = unconsumed[:8]
        raise RuntimeError(
            f"State-dict audit: {len(unconsumed)} tensor(s) del checkpoint no "
            f"coinciden con ningún parámetro del módulo. Causa probable: un "
            f"renombrado obsoleto en weight_adapters.py o un componente que el "
            f"módulo no construye. Primeros {len(sample)}: {sample}"
        )
```

## Dónde conectarla

`ModuleV3PipelineModelWithKVCache.load_model()` llama a `_prepare_state_dict()` con el state dict adaptado, luego construye el módulo en `_instantiate_module()` bajo `F.lazy()`, luego compila. Registra las claves en el primer hook y audita en el segundo:

```python
class MyModel(DonorModel):
    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> dict[str, Any]:
        state_dict = super()._prepare_state_dict(state_dict, model_config)
        self._checkpoint_keys = frozenset(state_dict)
        return state_dict

    def _instantiate_module(self, model_config: Any) -> Any:
        nn_model = MyTopLevelModule(model_config, self.kv_params)
        nn_model.to(self.devices[0])
        _audit_unconsumed_keys(
            checkpoint_keys=self._checkpoint_keys,
            parameter_names=(name for name, _ in nn_model.parameters),
        )
        return nn_model
```

El módulo se construye perezosamente, así que listar sus parámetros no asigna pesos. La auditoría es aritmética de conjuntos sobre ~10K nombres y corre una vez por carga.

## Qué atrapa

- **Typos en renombrados del adaptador.** ``mlp.routed.experts`` cuando la ruta del módulo es ``mlp.experts``: cada tensor de experto enrutado se reporta como no consumido, y la compilación también fallará por los parámetros ``mlp.experts`` faltantes.
- **Prefijo obsoleto.** ``model.layers`` sin renombrar cuando el módulo raíz espera ``language_model.layers``: cada tensor de capa se reporta.
- **Componentes que el módulo nunca construyó.** QK-norm, sesgos de atención, expertos compartidos o una norma post-feedforward que el checkpoint trae y el donante no tiene. Solo la compilación no puede verlos.
- **Conteo de expertos incorrecto.** Un checkpoint con 128 expertos cargado en un módulo construido para 64: los expertos 64–127 se reportan.

## Qué no atrapa

- **Renombrados a una ruta incorrecta pero existente.** Si el adaptador envía ``layers.0.mlp.experts.{j}.gate_proj.weight`` a ``layers.0.mlp.shared_experts.gate_proj.weight``, y las formas coinciden, el tensor es consumido por el parámetro equivocado. Compara la salida del adaptador contra el código de modelado de HF para cualquier renombrado que hayas escrito a mano.
- **Valores de tensores.** La auditoría y las comprobaciones de compilación verifican nombres, formas y dtypes. Un tensor transpuesto o mal permutado con la forma correcta carga sin queja. El comparador de [`debug-model`](../../debug-model/SKILL.md) encuentra esos.

## Estilo de reporte

Cuando la auditoría pase, regístralo explícitamente para que el usuario lo vea:

```python
logger.info(
    "State-dict audit: los %d tensores del checkpoint son consumidos por el módulo.",
    len(self._checkpoint_keys),
)
```

Cuando falle, el mensaje del ``RuntimeError`` basta para diagnosticar el bug. Las primeras 8 claves no consumidas normalmente hacen pattern-match con una sola clase de bug ("cada clave ``q_norm``/``k_norm``" → QK-norm faltante, "todo bajo ``model.layers``" → renombrado de prefijo).
