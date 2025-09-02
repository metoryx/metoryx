import chex
import jax
import jax.random as jr
from flax import linen

from . import normalization as norm
from .base import apply


class TestBatchNorm:
    def test_apply(self):
        flax_norm = linen.BatchNorm(use_running_average=False)
        morphax_norm = norm.BatchNorm(size=8)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 8))
        variables = flax_norm.init(jr.PRNGKey(2), inputs)

        flax_output, flax_updates = flax_norm.apply(variables, inputs, mutable=True)
        morphax_output, morphax_updates = apply(morphax_norm)(
            variables, None, inputs, is_training=True
        )

        chex.assert_trees_all_close(morphax_output, flax_output)
        chex.assert_trees_all_close(morphax_updates, flax_updates)

    def test_apply_with_noscale(self):
        flax_norm = linen.BatchNorm(use_scale=False, use_running_average=False)
        morphax_norm = norm.BatchNorm(size=8, use_scale=False)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 8))
        variables = flax_norm.init(jr.PRNGKey(2), inputs)

        flax_output, flax_updates = flax_norm.apply(variables, inputs, mutable=True)
        morphax_output, morphax_updates = apply(morphax_norm)(
            variables, None, inputs, is_training=True
        )

        chex.assert_trees_all_close(morphax_output, flax_output)
        chex.assert_trees_all_close(morphax_updates, flax_updates)

    def test_apply_with_nobias(self):
        flax_norm = linen.BatchNorm(use_bias=False, use_running_average=False)
        morphax_norm = norm.BatchNorm(size=8, use_bias=False)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 8))
        variables = flax_norm.init(jr.PRNGKey(2), inputs)

        flax_output, flax_updates = flax_norm.apply(variables, inputs, mutable=True)
        morphax_output, morphax_updates = apply(morphax_norm)(
            variables, None, inputs, is_training=True
        )

        chex.assert_trees_all_close(morphax_output, flax_output)
        chex.assert_trees_all_close(morphax_updates, flax_updates)

    def test_apply_with_notraining(self):
        flax_norm = linen.BatchNorm()
        morphax_norm = norm.BatchNorm(size=8)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 8))
        variables = flax_norm.init(jr.PRNGKey(2), inputs, use_running_average=False)

        # First update the running statistics
        _, variables = flax_norm.apply(variables, inputs, use_running_average=False, mutable=True)

        # Then test in eval mode
        flax_output = flax_norm.apply(variables, inputs, use_running_average=True)
        morphax_output, _ = apply(morphax_norm)(variables, None, inputs, is_training=False)

        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_vectorized_apply(self):
        flax_norm = linen.BatchNorm(axis_name="batch", use_running_average=False)
        morphax_norm = norm.BatchNorm(size=8, axis_name="batch")

        flax_apply_fn = linen.apply(
            lambda m, x: m(x),
            flax_norm,
            mutable=True,
        )

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 4, 8))
        variables = jax.vmap(flax_norm.init)(jr.split(jr.PRNGKey(2), 4), inputs)

        flax_output, flax_updates = jax.vmap(
            lambda variables, input: flax_apply_fn(variables, inputs),
            axis_name="batch",
        )(variables, inputs)

        morphax_output, morphax_updates = jax.vmap(
            lambda variables, input: apply(morphax_norm)(variables, None, inputs, is_training=True),
            axis_name="batch",
        )(variables, inputs)

        chex.assert_trees_all_close(morphax_output, flax_output)
        chex.assert_trees_all_close(morphax_updates, flax_updates)


class TestLayerNorm:
    def test_apply(self):
        flax_norm = linen.LayerNorm()
        morphax_norm = norm.LayerNorm(size=8)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 8))
        variables = flax_norm.init(jr.PRNGKey(2), inputs)

        flax_output = flax_norm.apply(variables, inputs)
        morphax_output, _ = apply(morphax_norm)(variables, None, inputs)

        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_noscale(self):
        flax_norm = linen.LayerNorm(use_scale=False)
        morphax_norm = norm.LayerNorm(size=8, use_scale=False)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 8))
        variables = flax_norm.init(jr.PRNGKey(2), inputs)

        flax_output = flax_norm.apply(variables, inputs)
        morphax_output, _ = apply(morphax_norm)(variables, None, inputs)

        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_nobias(self):
        flax_norm = linen.LayerNorm(use_bias=False)
        morphax_norm = norm.LayerNorm(size=8, use_bias=False)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 8))
        variables = flax_norm.init(jr.PRNGKey(2), inputs)

        flax_output = flax_norm.apply(variables, inputs)
        morphax_output, _ = apply(morphax_norm)(variables, None, inputs)

        chex.assert_trees_all_close(morphax_output, flax_output)


class TestRMSNorm:
    def test_apply(self):
        flax_norm = linen.RMSNorm()
        morphax_norm = norm.RMSNorm(size=8)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 8))
        variables = flax_norm.init(jr.PRNGKey(2), inputs)

        flax_output = flax_norm.apply(variables, inputs)
        morphax_output, _ = apply(morphax_norm)(variables, None, inputs)

        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_noscale(self):
        flax_norm = linen.RMSNorm(use_scale=False)
        morphax_norm = norm.RMSNorm(size=8, use_scale=False)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 8))
        variables = flax_norm.init(jr.PRNGKey(2), inputs)

        flax_output = flax_norm.apply(variables, inputs)
        morphax_output, _ = apply(morphax_norm)(variables, None, inputs)

        chex.assert_trees_all_close(morphax_output, flax_output)
