from typing import cast

import chex
import jax.numpy as jnp
import jax.random as jr
from flax import linen

from . import linear
from .base import Variables, apply, init


class TestConv:
    def test_apply(self):
        flax_conv = linen.Conv(features=3, kernel_size=(2, 2), use_bias=True)
        morphax_conv = linear.Conv(in_size=4, out_size=8, kernel_size=(3, 3))

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 4, 4, 3))
        flax_output, variables = flax_conv.init_with_output(jr.PRNGKey(2), inputs)
        morphax_output, _ = apply(morphax_conv)(cast(Variables, variables), None, inputs)
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_nobias(self):
        flax_conv = linen.Conv(features=8, kernel_size=(3, 3), use_bias=False)
        morphax_conv = linear.Conv(in_size=3, out_size=8, kernel_size=(3, 3), use_bias=False)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 4, 4, 3))
        flax_output, variables = flax_conv.init_with_output(jr.PRNGKey(2), inputs)
        morphax_output, _ = apply(morphax_conv)(cast(Variables, variables), None, inputs)
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_same_padding(self):
        flax_conv = linen.Conv(features=8, kernel_size=(3, 3), padding="SAME", use_bias=True)
        morphax_conv = linear.Conv(in_size=3, out_size=8, kernel_size=(3, 3), padding="SAME")

        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 6, 6, 3))
        flax_output, variables = flax_conv.init_with_output(jr.PRNGKey(2), inputs)
        morphax_output, _ = apply(morphax_conv)(cast(Variables, variables), None, inputs)
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_sequence_padding(self):
        flax_conv = linen.Conv(
            features=5, kernel_size=(3, 3), padding=((1, 2), (2, 1)), use_bias=True
        )
        morphax_conv = linear.Conv(
            in_size=4, out_size=5, kernel_size=(3, 3), padding=((1, 2), (2, 1))
        )

        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 8, 8, 4))
        flax_output, variables = flax_conv.init_with_output(jr.PRNGKey(2), inputs)
        morphax_output, _ = apply(morphax_conv)(cast(Variables, variables), None, inputs)
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_strides(self):
        flax_conv = linen.Conv(features=6, kernel_size=(3, 3), strides=(2, 2), use_bias=True)
        morphax_conv = linear.Conv(in_size=3, out_size=6, kernel_size=(3, 3), strides=(2, 2))

        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 8, 8, 3))
        flax_output, variables = flax_conv.init_with_output(jr.PRNGKey(2), inputs)
        morphax_output, _ = apply(morphax_conv)(cast(Variables, variables), None, inputs)
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_dilations(self):
        # Note: Flax uses 'kernel_dilation' instead of 'dilation'
        flax_conv = linen.Conv(
            features=7, kernel_size=(3, 3), kernel_dilation=(2, 2), use_bias=True
        )
        morphax_conv = linear.Conv(in_size=3, out_size=7, kernel_size=(3, 3), dilation=(2, 2))

        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 10, 10, 3))
        flax_output, variables = flax_conv.init_with_output(jr.PRNGKey(2), inputs)
        morphax_output, _ = apply(morphax_conv)(cast(Variables, variables), None, inputs)
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_groups(self):
        # Test grouped convolution
        flax_conv = linen.Conv(
            features=12, kernel_size=(3, 3), feature_group_count=3, use_bias=True
        )
        morphax_conv = linear.Conv(in_size=6, out_size=12, kernel_size=(3, 3), groups=3)

        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 5, 5, 6))
        flax_output, variables = flax_conv.init_with_output(jr.PRNGKey(2), inputs)
        morphax_output, _ = apply(morphax_conv)(cast(Variables, variables), None, inputs)
        chex.assert_trees_all_close(morphax_output, flax_output)


class TestDense:
    def test_apply(self):
        flax_dense = linen.Dense(features=3, use_bias=True)
        morphax_dense = linear.Dense(in_size=2, out_size=3, use_bias=True)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 2))
        flax_output, variables = flax_dense.init_with_output(jr.PRNGKey(2), inputs)
        morphax_output, _ = apply(morphax_dense)(cast(Variables, variables), None, inputs)
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_nobias(self):
        flax_dense = linen.Dense(features=3, use_bias=False)
        morphax_dense = linear.Dense(in_size=2, out_size=3, use_bias=False)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 2))
        flax_output, variables = flax_dense.init_with_output(jr.PRNGKey(2), inputs)
        morphax_output, _ = apply(morphax_dense)(cast(Variables, variables), None, inputs)
        chex.assert_trees_all_close(morphax_output, flax_output)


class TestEmbed:
    def test_init(self):
        morphax_embed = linear.Embed(size=5, num_embeddings=10)
        assert morphax_embed.embedding.shape == (10, 5)

    def test_apply(self):
        flax_embed = linen.Embed(num_embeddings=10, features=5)
        morphax_embed = linear.Embed(size=5, num_embeddings=10)

        inputs = jr.randint(jr.PRNGKey(1), shape=(4,), minval=0, maxval=10)
        flax_output, variables = flax_embed.init_with_output(jr.PRNGKey(2), inputs)
        morphax_output, _ = apply(morphax_embed)(cast(Variables, variables), None, inputs)
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_attend(self):
        flax_embed = linen.Embed(num_embeddings=10, features=5)
        flax_apply_fn = linen.apply(lambda m, x: m.attend(x), flax_embed, mutable=False)
        morphax_embed = linear.Embed(size=5, num_embeddings=10)

        inputs = jr.normal(jr.PRNGKey(1), shape=(4, 5))
        inputs /= jnp.sum(inputs, axis=-1, keepdims=True)
        variables = init(morphax_embed)(jr.PRNGKey(2))

        flax_output = flax_apply_fn(variables, inputs)
        morphax_output, _ = apply(morphax_embed, to_callable=lambda m: getattr(m, "attend"))(
            cast(Variables, variables), None, inputs
        )
        chex.assert_trees_all_close(morphax_output, flax_output)
