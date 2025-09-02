from typing import Sequence

import jax.numpy as jnp
from jax import lax

from metoryx._src.base import (
    Array,
    DType,
    Initializer,
    Module,
    Parameter,
)
from metoryx._src.initializers import lecun_normal, variance_scaling, zeros
from metoryx._src.padding import PaddingLike, canonicalize_padding


class Dense(Module):
    """Applies an affine linear transformation."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        use_bias: bool = True,
        kernel_init: Initializer = lecun_normal(),
        bias_init: Initializer = zeros(),
        dtype: DType = jnp.float32,
        param_dtype: DType | None = None,
    ):
        """Initializes the Dense module.

        Args:
            in_size: Size of the input features.
            out_size: Size of the output features.
            use_bias: Whether to include a bias term.
            kernel_init: Initializer for the weight matrix.
            bias_init: Initializer for the bias term.
            dtype: Data type of the parameters.
            param_dtype: Data type for computation.
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.use_bias = use_bias

        self.kernel = Parameter(
            kernel_init,
            shape=(in_size, out_size),
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.bias = None
        if use_bias:
            self.bias = Parameter(
                bias_init,
                shape=(out_size,),
                dtype=dtype,
                param_dtype=param_dtype,
            )

    def __call__(self, inputs: Array) -> Array:
        r"""Applies an affine transformation to the input.

        Args:
            inputs: Input array of shape (\*batch_size, in_size).

        Returns:
            The transformed output array of shape (\*batch_size, out_size).
        """
        output = inputs @ self.kernel
        if self.bias is not None:
            output += self.bias
        return output


def _maybe_broadcast(name: str, x: int | Sequence[int], kernel_size: Sequence[int]) -> list[int]:
    if isinstance(x, int):
        x = (x,) * len(kernel_size)
    if len(x) != len(kernel_size):
        raise ValueError(f"{name} must be an int or a sequence of the same length as kernel_size")
    return list(x)


def _conv_dimension_numbers(input_shape: tuple[int, ...]) -> lax.ConvDimensionNumbers:
    """Computes the dimension numbers based on the input shape.

    Modify from: https://github.com/google/flax/blob/f73aea5cc605d9e4530132ca3569b04721942f36/flax/linen/linear.py#L427
    """
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class Conv(Module):
    """Applies a convolution."""

    padding: PaddingLike

    def __init__(
        self,
        in_size: int,
        out_size: int,
        kernel_size: Sequence[int],
        padding: PaddingLike = "SAME",
        strides: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        kernel_init: Initializer = lecun_normal(),
        bias_init: Initializer = zeros(),
        dtype: DType = jnp.float32,
        param_dtype: DType | None = None,
    ):
        """Initializes the Conv module.

        Args:
            in_size: Number of input channels.
            out_size: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            padding: Padding method, either 'SAME', 'VALID', or a sequence of padding tuples.
            strides: Strides of the convolution.
            dilation: Dilation of the convolution.
            groups: Number of groups for grouped convolution.
            use_bias: Whether to include a bias term.
            kernel_init: Initializer for the convolutional kernel.
            bias_init: Initializer for the bias term.
            dtype: Data type of the parameters.
            param_dtype: Data type for computation.
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias

        if in_size % groups != 0:
            raise ValueError("in_size must be divisible by groups")
        kernel_shape = (*kernel_size, in_size // groups, out_size)
        self.kernel = Parameter(
            kernel_init,
            shape=kernel_shape,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.bias = None
        if use_bias:
            self.bias = Parameter(
                bias_init,
                shape=(out_size,),
                dtype=dtype,
                param_dtype=param_dtype,
            )

    def __call__(self, inputs: Array) -> Array:
        r"""Applies the convolution to the input.

        Args:
            inputs: Input array of shape (\*batch_size, height, width, in_size).

        Returns:
            The convolved output array.
        """
        padding = canonicalize_padding(self.padding, self.kernel_size)
        window_strides = _maybe_broadcast("strides", self.strides, self.kernel_size)
        dilation = _maybe_broadcast("dilation", self.dilation, self.kernel_size)

        batch_dims = inputs.shape[: -(len(self.kernel_size) + 1)]
        input_dims = inputs.shape[-(len(self.kernel_size) + 1) :]
        inputs = jnp.reshape(inputs, (-1, *input_dims))

        dimension_numbers = _conv_dimension_numbers(inputs.shape)
        output = lax.conv_general_dilated(
            inputs,
            jnp.asarray(self.kernel.value),
            window_strides=window_strides,
            padding=padding,
            lhs_dilation=None,
            rhs_dilation=dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.groups,
        )

        if self.bias is not None:
            output += self.bias

        output_dims = output.shape[1:]
        output = jnp.reshape(output, (*batch_dims, *output_dims))
        return output


class Embed(Module):
    """Embeds the inputs along the last dimension."""

    def __init__(
        self,
        size: int,
        num_embeddings: int,
        embedding_init: Initializer = variance_scaling(1.0, "fan_in", "normal", out_axis=0),
        dtype: DType = jnp.float32,
        param_dtype: DType | None = None,
    ):
        """Initializes the Embed module.

        Args:
            size: Size of each embedding vector.
            num_embeddings: Number of unique embeddings.
            embedding_init: Initializer for the embedding matrix.
            dtype: Data type of the parameters.
            param_dtype: Data type for computation.
        """
        super().__init__()
        self.size = size
        self.num_embeddings = num_embeddings

        self.embedding = Parameter(
            embedding_init,
            shape=(num_embeddings, size),
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def __call__(self, inputs: Array) -> Array:
        r"""Embeds the input indices.

        Args:
            inputs: Input array of shape (\*batch_size,) with integer indices.

        Returns:
            The embedded output array of shape (\*batch_size, size).
        """
        return jnp.take(self.embedding.value, inputs, axis=0)

    def attend(self, query: Array) -> Array:
        r"""Computes the attention scores between the query and the embeddings.

        Args:
            query: Query array of shape (\*batch_size, size).

        Returns:
            The attention scores of shape (\*batch_size, num_embeddings).
        """
        return query @ self.embedding.T
