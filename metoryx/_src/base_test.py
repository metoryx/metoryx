import jax
import jax.numpy as jnp

from metoryx._src.base import (
    Module,
    Parameter,
    State,
    apply,
    get_states,
    init,
    next_rng_key,
    transform,
)
from metoryx._src.initializers import ones, zeros


class SimpleModule(Module):
    def __init__(self):
        super().__init__()
        self.param = Parameter(ones(), shape=(3, 4))
        self.state = State("batch_stats", zeros(), shape=(3,), mutable=True)

    def __call__(self, x):
        return self.param.value + x


def test_parameter_creation():
    param = Parameter(ones(), shape=(3, 4))
    assert param.shape == (3, 4)
    assert param.dtype == jnp.float32
    assert param.col == "params"
    assert not param.mutable


def test_state_creation():
    state = State("batch_stats", zeros(), shape=(3,), mutable=True)
    assert state.shape == (3,)
    assert state.dtype == jnp.float32
    assert state.col == "batch_stats"
    assert state.mutable


def test_get_states():
    module = SimpleModule()
    states = get_states(module)
    
    assert "params" in states
    assert "batch_stats" in states
    # Check that we have the expected structure
    assert isinstance(states["params"], dict)
    assert isinstance(states["batch_stats"], dict)


def test_init_function():
    module = SimpleModule()
    init_fn = init(module)
    
    key = jax.random.PRNGKey(0)
    variables = init_fn(key)
    
    assert "params" in variables
    assert "batch_stats" in variables
    # Check that variables have the expected structure
    assert isinstance(variables["params"], dict)
    assert isinstance(variables["batch_stats"], dict)


def test_apply_function():
    module = SimpleModule()
    apply_fn = apply(module)
    
    key = jax.random.PRNGKey(0)
    variables = init(module)(key)
    
    x = jnp.ones((3, 4))
    output, new_variables = apply_fn(variables, None, x)
    
    assert output.shape == (3, 4)
    assert "params" in new_variables
    assert "batch_stats" in new_variables


def test_transform():
    module = SimpleModule()
    transformed = transform(module)
    
    key = jax.random.PRNGKey(0)
    variables = transformed.init(key)
    
    x = jnp.ones((3, 4))
    output, new_variables = transformed.apply(variables, None, x)
    
    assert output.shape == (3, 4)
    assert "params" in new_variables


def test_next_rng_key():
    # Test that next_rng_key works within a module context
    module = SimpleModule()
    init_fn = init(module)
    
    key = jax.random.PRNGKey(0)
    variables = init_fn(key)
    
    # next_rng_key should work during apply
    class TestModule(Module):
        def __call__(self, x):
            rng = next_rng_key()
            return x + jax.random.normal(rng, shape=())
    
    test_module = TestModule()
    apply_fn = apply(test_module)
    
    x = jnp.array(1.0)
    output, _ = apply_fn({}, jax.random.PRNGKey(1), x)
    
    assert output.shape == ()
    assert output != x  # Should be different due to random noise


class StatefulModule(Module):
    def __init__(self):
        super().__init__()
        self.counter = State("counters", zeros(), shape=(), dtype=jnp.int32, mutable=True)

    def __call__(self, x):
        self.counter.value = self.counter.value + 1
        return x + self.counter.value


def test_mutable_state():
    module = StatefulModule()
    init_fn = init(module)
    apply_fn = apply(module)
    
    key = jax.random.PRNGKey(0)
    variables = init_fn(key)
    
    # First call
    x = jnp.array(1.0)
    output1, variables1 = apply_fn(variables, None, x)
    
    # Second call with updated variables
    output2, variables2 = apply_fn(variables1, None, x)
    
    assert output1 == 2.0  # 1 + 1 (counter incremented from 0 to 1)
    assert output2 == 3.0  # 1 + 2 (counter incremented from 1 to 2)
    # Check that counter values were updated
    assert "counters" in variables1
    assert "counters" in variables2


def test_nested_modules():
    class NestedModule(Module):
        def __init__(self):
            super().__init__()
            self.sub1 = SimpleModule()
            self.sub2 = SimpleModule()

        def __call__(self, x):
            return self.sub1(x) + self.sub2(x)

    module = NestedModule()
    states = get_states(module)
    
    assert "params" in states
    assert "batch_stats" in states
    # Check nested structure exists
    assert isinstance(states["params"], dict)
    assert isinstance(states["batch_stats"], dict)


def test_module_with_list():
    class ListModule(Module):
        def __init__(self):
            super().__init__()
            self.layers = [
                Parameter(ones(), shape=(2, 2)),
                Parameter(ones(), shape=(2, 2)),
            ]

        def __call__(self, x):
            for layer in self.layers:
                x = x @ layer.value
            return x

    module = ListModule()
    states = get_states(module)
    
    assert "params" in states
    # Check that params contains layers structure
    assert isinstance(states["params"], dict)
    
    # Test that init works
    init_fn = init(module)
    variables = init_fn(jax.random.PRNGKey(0))
    assert "params" in variables


def test_empty_module():
    class EmptyModule(Module):
        def __call__(self, x):
            return x

    module = EmptyModule()
    states = get_states(module)
    
    assert len(states) == 0
    
    init_fn = init(module)
    variables = init_fn(jax.random.PRNGKey(0))
    assert len(variables) == 0