# Concepts

Metoryx is designed to reconcile two powerful paradigms: **flexible model definition** and **robust model execution**. It provides the intuitive, imperative workflow of frameworks like PyTorch, combined with the high performance and reproducibility of functional programming with JAX—all within a single, seamless workflow.

To achieve this, Metoryx is built upon two distinct worlds:

## The two worlds of metoryx

### 1. The world of definition: flexibility and intuition

In this world, model definition is **imperative** and highly flexible. After creating a model instance, you can dynamically modify its structure or directly set initial parameter values.

```python
# First create an instance of the model
model = ConvNet()

# Later, you can freely replace parts of the architecture
model.fc2 = mx.Dense(128, 10, kernel_init=mx.initializers.glorot_uniform())

# You can also directly specify parameter initial values using NumPy/JAX arrays
# (Useful for transfer learning or specialized initialization)
model.fc1.kernel.value = jnp.zeros((7 * 7 * 64, 128))
```

This world serves as a **design studio** where you can experiment with ideas and iterate quickly.

### 2. The world of execution: robustness and performance

Once the model architecture is finalized, you transition to the **world of execution** by calling `mx.transform`. In this world, the model is treated as an **immutable, pure function**, fully adhering to the functional paradigm of JAX.

```python
# "Freeze" the model definition and convert it into a pure function
init_fn, apply_fn = mx.transform(model)

# The resulting functions are fully compatible with JAX transformations (jit, vmap, etc.)
jitted_apply_fn = jax.jit(apply_fn)
```

Here, you can confidently train and run your model without worrying about unintended side effects.

## `mx.transform`: The bridge between two worlds

The `mx.transform` function is the crucial bridge connecting these two distinct worlds.

It creates a **snapshot** of the model's definition at the moment it's called and converts it into the pure functions `init_fn` and `apply_fn`.

Technically, it uses a **deep copy** of the model instance, so any subsequent modifications to the original `model` object have no effect on the generated functions. This guarantees a clean separation between the flexibility of definition and the reproducibility of execution.

## Conclusion

Metoryx reconciles what were previously considered incompatible advantages: "interactive flexibility" and "functional robustness." This is made possible by the clear separation point provided by `mx.transform`. This unique workflow allows researchers and developers to prototype rapidly and then seamlessly transition their work into robust, high-performance training loops where reproducibility is paramount.