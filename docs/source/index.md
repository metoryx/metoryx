# Metoryx

Metoryx is a neural network library for JAX that combines the functional init/apply style with a flexible model definition. It cleanly separates logic (pure functions) from state (parameters) while keeping the workflow intuitive and easy to extend. If you value reusability, testability, and compatibility with the JAX ecosystem, Metoryx is for you.

## Features

- **Flexible model definitions**: Define models in a Pythonic, PyTorch-like style.
- **Seamless JAX integration**: Transform models into pure init/apply functions that work with jax.jit, jax.vmap, and jax.pmap.
- **Easy customization**: Implement LoRA, transfer learning, and other techniques with minimal boilerplate; parameters are regular Python objects you can inspect and modify.

## Installation

Install via pip:

```bash
pip install metoryx
```

## Minimal Example

```python
# Define a simple MLP model
class Mlp(mx.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.dense1 = mx.Dense(in_size, 128)
        self.dense2 = mx.Dense(128, out_size)

    def __call__(self, x):
        x = self.dense1(x)
        x = mx.relu(x)
        return self.dense2(x)


# Instantiate the MLP model and transform it into init/apply functions
mlp = Mlp(in_size=16, out_size=10)
init, apply = mx.transform(mlp)

# Initialize parameters and run a forward pass
x = jax.numpy.zeros((16,))
variables = init(jax.random.PRNGKey(42))
y, new_variables = apply(variables, jax.random.PRNGKey(0), x)
```

<!-- Table of Contents. Not shown on the body -->
```{toctree}
:maxdepth: 2
:hidden:

tutorial
concepts
examples/index
advanced/index
contributing
reference
```
