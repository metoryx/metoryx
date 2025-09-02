from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jax import lax

from metoryx._src.base import Array
from metoryx._src.padding import PaddingLike, canonicalize_padding


def pool(
    inputs: Array,
    init,
    computation,
    kernel_size: Sequence[int],
    strides: int | Sequence[int] = 1,
    padding: PaddingLike = "VALID",
) -> Array:
    """Performs pooling on the input array.

    Args:
        inputs: Input array of shape (*batch_size, height, width, channels).
        init: Initial value for the reduction.
        computation: Computation to perform on the input.
        kernel_size: Size of the pooling window.
        strides: Strides of the pooling operation.
        padding: Padding method, either 'SAME' or 'VALID'.

    Returns:
        The pooled output array.
    """
    padding = canonicalize_padding(padding, kernel_size)
    strides = (strides,) * len(kernel_size) if isinstance(strides, int) else strides

    # Determine the number of batch dimensions
    num_batch_dims = inputs.ndim - len(kernel_size) - 1  # -1 for channel dimension

    # Create window dimensions with 1s for batch and channel dimensions
    window_dimensions = (1,) * num_batch_dims + tuple(kernel_size) + (1,)

    # Create strides with 1s for batch and channel dimensions
    window_strides = (1,) * num_batch_dims + tuple(strides) + (1,)

    # Adjust padding to include batch and channel dimensions
    if isinstance(padding, str):
        full_padding = padding
    else:
        full_padding = ((0, 0),) * num_batch_dims + tuple(padding) + ((0, 0),)

    output = lax.reduce_window(
        inputs,
        init,
        computation,
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding=full_padding,
    )
    return output


def avg_pool(
    inputs: Array,
    kernel_size: Sequence[int],
    strides: int | Sequence[int],
    padding: PaddingLike = "VALID",
) -> Array:
    output = pool(inputs, 0.0, lax.add, kernel_size, strides, padding)
    output /= np.prod(kernel_size)
    return output


def max_pool(
    inputs: Array,
    kernel_size: Sequence[int],
    strides: int | Sequence[int],
    padding: PaddingLike = "VALID",
) -> Array:
    output = pool(inputs, -jnp.inf, lax.max, kernel_size, strides, padding)
    return output


def min_pool(
    inputs: Array,
    kernel_size: Sequence[int],
    strides: int | Sequence[int],
    padding: PaddingLike = "VALID",
) -> Array:
    output = pool(inputs, jnp.inf, lax.min, kernel_size, strides, padding)
    return output
