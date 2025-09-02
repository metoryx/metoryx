# API reference

## Fundamentals

### Module Transformation

```{eval-rst}
.. autofunction:: metoryx.transform
.. autofunction:: metoryx.init
.. autofunction:: metoryx.apply
.. autofunction:: metoryx.checkpoint
```

### Modules, States, and Parameters

```{eval-rst}
.. autoclass:: metoryx.Module
   :members:
.. autoclass:: metoryx.State
   :members:
.. autoclass:: metoryx.Parameter
   :members:
```

### Random Numbers

```{eval-rst}
.. autoclass:: metoryx.PRNGKeys
   :members:
.. autofunction:: metoryx.next_rng_key
```

## Common Modules and Functions

### Linear

```{eval-rst}
.. autoclass:: metoryx.Dense
   :members:
.. autoclass:: metoryx.Conv
   :members:
.. autoclass:: metoryx.Embed
   :members:
```

### Normalization

```{eval-rst}
.. autoclass:: metoryx.BatchNorm
   :members:
.. autoclass:: metoryx.LayerNorm
   :members:
.. autoclass:: metoryx.RMSNorm
   :members:
```

### Dropout

```{eval-rst}
.. autofunction:: metoryx.dropout
```

### Pooling

```{eval-rst}
.. autofunction:: metoryx.avg_pool
.. autofunction:: metoryx.max_pool
.. autofunction:: metoryx.min_pool
```

### Activation Functions

Most activation functions are exported from `jax.nn` for convenience.

```{eval-rst}
.. autofunction:: metoryx.celu
.. autofunction:: metoryx.elu
.. autofunction:: metoryx.gelu
.. autofunction:: metoryx.glu
.. autofunction:: metoryx.hard_sigmoid
.. autofunction:: metoryx.hard_silu
.. autofunction:: metoryx.hard_tanh
.. autofunction:: metoryx.identity
.. autofunction:: metoryx.leaky_relu
.. autofunction:: metoryx.log_sigmoid
.. autofunction:: metoryx.log_softmax
.. autofunction:: metoryx.mish
.. autofunction:: metoryx.one_hot
.. autofunction:: metoryx.relu
.. autofunction:: metoryx.relu6
.. autofunction:: metoryx.selu
.. autofunction:: metoryx.sigmoid
.. autofunction:: metoryx.silu
.. autofunction:: metoryx.soft_sign
.. autofunction:: metoryx.softmax
.. autofunction:: metoryx.softplus
.. autofunction:: metoryx.sparse_plus
.. autofunction:: metoryx.sparse_sigmoid
.. autofunction:: metoryx.squareplus
.. autofunction:: metoryx.standardize
```

## Initializers

Initializers are exported from `jax.nn.initializers` for convenience.

```{eval-rst}
.. autofunction:: metoryx.initializers.zeros
.. autofunction:: metoryx.initializers.ones
.. autofunction:: metoryx.initializers.constant
.. autofunction:: metoryx.initializers.uniform
.. autofunction:: metoryx.initializers.normal
.. autofunction:: metoryx.initializers.truncated_normal
.. autofunction:: metoryx.initializers.variance_scaling
.. autofunction:: metoryx.initializers.glorot_uniform
.. autofunction:: metoryx.initializers.glorot_normal
.. autofunction:: metoryx.initializers.xavier_uniform
.. autofunction:: metoryx.initializers.xavier_normal
.. autofunction:: metoryx.initializers.he_uniform
.. autofunction:: metoryx.initializers.he_normal
.. autofunction:: metoryx.initializers.kaiming_uniform
.. autofunction:: metoryx.initializers.kaiming_normal
.. autofunction:: metoryx.initializers.lecun_uniform
.. autofunction:: metoryx.initializers.lecun_normal
.. autofunction:: metoryx.initializers.orthogonal
.. autofunction:: metoryx.initializers.delta_orthogonal
```

## Utilities

```{eval-rst}
.. autoclass:: metoryx.utils.AverageMeter
   :members:
.. autofunction:: metoryx.assign_variables
```