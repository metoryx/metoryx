<div align="center">
    <h1>Metoryx</h1>
</div>

Metoryx is a neural network library for JAX that combines the functional init/apply style with a flexible model definition. It cleanly separates logic (pure functions) from state (parameters) while keeping the workflow intuitive and easy to extend. If you value reusability, testability, and compatibility with the JAX ecosystem, Metoryx is for you.

> [!CAUTION]
> This project is a work in progress, and the API may change.

## Features

- Flexible model definitions: Define models in a Pythonic, PyTorch-like style.
- Seamless JAX integration: Transform models into pure init/apply functions that work with jax.jit, jax.vmap, and jax.pmap.
- Easy customization: Implement LoRA, transfer learning, and other techniques with minimal boilerplate; parameters are regular Python objects you can inspect and modify.

## Installation

```bash
pip install metoryx
```

## Quick Example

Define a model by subclassing `nn.Module`.

```python
import jax
import jax.numpy as jnp

import metoryx as mx
from metoryx import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Dense(16, 128)
        self.dense2 = nn.Dense(128, 10)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.relu(x)
        return self.dense2(x)


model = MyModel()

# Model instance is a Python object that can be manipulated freely.
model.dense2 = nn.Dense(128, 5)  # Replace the last layer
model.dense1.kernel.value = jnp.zeros((16, 128), jnp.float32)  # Set initial weights directly

# Transform the module into JAX-native pure functions.
init, apply = mx.transform(model)

# Create PRNG keys
rng = jax.random.PRNGKey(0)
init_rng, apply_rng = jax.random.split(rng)

# Initialize parameters
variables = init(init_rng)

# Forward pass (with JIT compilation)
inputs = jnp.zeros((16,))
outputs, new_variables = jax.jit(apply)(variables, apply_rng, inputs)
```

More examples can be found in [metoryx/examples](https://github.com/metoryx/examples).