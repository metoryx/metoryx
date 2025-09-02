import chex
import jax.random as jr
from flax import linen

from . import pool


class TestMaxPool:
    def test_apply(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 8, 8, 3))
        
        flax_output = linen.max_pool(inputs, window_shape=(2, 2), strides=(2, 2))
        morphax_output = pool.max_pool(inputs, kernel_size=(2, 2), strides=(2, 2))
        
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_same_padding(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 7, 7, 3))
        
        flax_output = linen.max_pool(inputs, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        morphax_output = pool.max_pool(inputs, kernel_size=(3, 3), strides=(2, 2), padding="SAME")
        
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_non_square_kernel(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 8, 6, 3))
        
        flax_output = linen.max_pool(inputs, window_shape=(3, 2), strides=(2, 1))
        morphax_output = pool.max_pool(inputs, kernel_size=(3, 2), strides=(2, 1))
        
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_single_stride(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 8, 8, 3))
        
        flax_output = linen.max_pool(inputs, window_shape=(2, 2), strides=(2, 2))
        morphax_output = pool.max_pool(inputs, kernel_size=(2, 2), strides=2)
        
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_large_kernel(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 8, 8, 3))
        
        flax_output = linen.max_pool(inputs, window_shape=(4, 4), strides=(4, 4))
        morphax_output = pool.max_pool(inputs, kernel_size=(4, 4), strides=(4, 4))
        
        chex.assert_trees_all_close(morphax_output, flax_output)


class TestAvgPool:
    def test_apply(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 8, 8, 3))
        
        flax_output = linen.avg_pool(inputs, window_shape=(2, 2), strides=(2, 2))
        morphax_output = pool.avg_pool(inputs, kernel_size=(2, 2), strides=(2, 2))
        
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_same_padding(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 7, 7, 3))
        
        flax_output = linen.avg_pool(inputs, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        morphax_output = pool.avg_pool(inputs, kernel_size=(3, 3), strides=(2, 2), padding="SAME")
        
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_non_square_kernel(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 8, 6, 3))
        
        flax_output = linen.avg_pool(inputs, window_shape=(3, 2), strides=(2, 1))
        morphax_output = pool.avg_pool(inputs, kernel_size=(3, 2), strides=(2, 1))
        
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_single_stride(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 8, 8, 3))
        
        flax_output = linen.avg_pool(inputs, window_shape=(2, 2), strides=(2, 2))
        morphax_output = pool.avg_pool(inputs, kernel_size=(2, 2), strides=2)
        
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_large_kernel(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 8, 8, 3))
        
        flax_output = linen.avg_pool(inputs, window_shape=(4, 4), strides=(4, 4))
        morphax_output = pool.avg_pool(inputs, kernel_size=(4, 4), strides=(4, 4))
        
        chex.assert_trees_all_close(morphax_output, flax_output)

    def test_apply_with_overlapping_windows(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 8, 8, 3))
        
        flax_output = linen.avg_pool(inputs, window_shape=(3, 3), strides=(1, 1))
        morphax_output = pool.avg_pool(inputs, kernel_size=(3, 3), strides=(1, 1))
        
        chex.assert_trees_all_close(morphax_output, flax_output)


class TestMinPool:
    def test_apply(self):
        # Flax doesn't have min_pool, so we test basic functionality
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 4, 4, 3))
        
        output = pool.min_pool(inputs, kernel_size=(2, 2), strides=(2, 2))
        
        # Basic shape check
        assert output.shape == (2, 2, 2, 3)
        
        # Check that output values are minimum from each window
        # For example, check first window of first batch and first channel
        window = inputs[0, 0:2, 0:2, 0]
        expected_min = window.min()
        chex.assert_trees_all_close(output[0, 0, 0, 0], expected_min)

    def test_apply_with_same_padding(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 3, 3, 2))
        
        output = pool.min_pool(inputs, kernel_size=(2, 2), strides=(2, 2), padding="SAME")
        
        # Check output shape with SAME padding
        assert output.shape == (2, 2, 2, 2)

    def test_apply_with_non_square_kernel(self):
        inputs = jr.normal(jr.PRNGKey(1), shape=(2, 4, 6, 3))
        
        output = pool.min_pool(inputs, kernel_size=(2, 3), strides=(2, 3))
        
        assert output.shape == (2, 2, 2, 3)