from typing import Any

import jax
import jax.numpy as jnp

from metoryx._src.base import (
    Array,
    DType,
    Initializer,
    Module,
    Parameter,
    State,
)
from metoryx._src.initializers import ones, zeros


class BatchNorm(Module):
    """Batch normalization.

    Ref. https://arxiv.org/abs/1502.03167

    Batch normalization keep moving average of batch statistics.
    We will keep them in `batch_stats` collection.

    Attributes:
        size: size of input features.
        momentum: momentum for the moving average.
        epsilon: small constant for numerical stability.
        use_scale: whether to use a scale parameter.
        use_bias: whether to use a bias parameter.
        scale_init: initializer for the scale parameter.
        bias_init: initializer for the bias parameter.
        dtype: data type for computation.
        param_dtype: data type of the parameters.
        axis_name: axis name to sync batch statistics along devices.
        axis_index_groups: axis index groups for distributed training.
    """

    def __init__(
        self,
        size: int,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        use_scale: bool = True,
        use_bias: bool = True,
        scale_init: Initializer = ones(),
        bias_init: Initializer = zeros(),
        dtype: DType = jnp.float32,
        param_dtype: DType | None = None,
        axis_name: Any | None = None,
        axis_index_groups: Any | None = None,
    ):
        super().__init__()
        self.size = size
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.scale_init = scale_init
        self.bias_init = bias_init
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups

        self.mean = State(
            "batch_stats",
            zeros(),
            shape=(size,),
            dtype=dtype,
            param_dtype=param_dtype,
            mutable=True,
        )

        self.var = State(
            "batch_stats",
            ones(),
            shape=(size,),
            dtype=dtype,
            param_dtype=param_dtype,
            mutable=True,
        )

        self.scale = None
        if use_scale:
            self.scale = Parameter(
                self.scale_init,
                shape=(size,),
                dtype=dtype,
                param_dtype=param_dtype,
            )

        self.bias = None
        if use_bias:
            self.bias = Parameter(
                self.bias_init,
                shape=(size,),
                dtype=dtype,
                param_dtype=param_dtype,
            )

    def __call__(self, input: Array, is_training: bool = False) -> Array:
        if is_training:
            axis = tuple(i for i in range(input.ndim - 1))
            mean = jnp.mean(input, axis=axis)
            mean2 = jnp.mean(jnp.square(input), axis=axis)

            if self.axis_name:
                synced: tuple[Array, Array] = jax.lax.pmean(
                    (mean, mean2),
                    self.axis_name,
                    axis_index_groups=self.axis_index_groups,
                )
                mean, mean2 = synced

            var = mean2 - jnp.square(mean)
            output = (input - mean) / jnp.sqrt(var + self.epsilon)

            # Update batch stats
            self.mean.value = self.momentum * self.mean + (1 - self.momentum) * mean
            self.var.value = self.momentum * self.var + (1 - self.momentum) * var
        else:
            output = (input - self.mean) / jnp.sqrt(self.var + self.epsilon)

        if self.scale is not None:
            output *= self.scale
        if self.bias is not None:
            output += self.bias

        return output


class LayerNorm(Module):
    """Layer normalization.

    Ref. https://arxiv.org/abs/1607.06450

    Attributes:
        size: size of input features.
        epsilon: small constant for numerical stability.
        use_scale: whether to use a scale parameter.
        use_bias: whether to use a bias parameter.
        scale_init: initializer for the scale parameter.
        bias_init: initializer for the bias parameter.
        dtype: data type for computation.
        param_dtype: data type of the parameters.
    """

    def __init__(
        self,
        size: int,
        epsilon: float = 1e-6,
        use_scale: bool = True,
        use_bias: bool = True,
        scale_init: Initializer = ones(),
        bias_init: Initializer = zeros(),
        dtype: DType = jnp.float32,
        param_dtype: DType | None = None,
    ):
        super().__init__()
        self.size = size
        self.epsilon = epsilon
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.scale_init = scale_init
        self.bias_init = bias_init
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.scale = None
        if use_scale:
            self.scale = Parameter(
                self.scale_init,
                shape=(size,),
                dtype=dtype,
                param_dtype=param_dtype,
            )

        self.bias = None
        if use_bias:
            self.bias = Parameter(
                self.bias_init,
                shape=(size,),
                dtype=dtype,
                param_dtype=param_dtype,
            )

    def __call__(self, input: Array) -> Array:
        mean = jnp.mean(input, axis=-1, keepdims=True)
        mean2 = jnp.mean(jnp.square(input), axis=-1, keepdims=True)
        var = mean2 - jnp.square(mean)
        output = (input - mean) / jnp.sqrt(var + self.epsilon)

        if self.scale is not None:
            output *= self.scale
        if self.bias is not None:
            output += self.bias

        return output


class RMSNorm(Module):
    """RMS layer normalization.

    Ref. https://arxiv.org/abs/1910.07467

    Attributes:
        size: size of input features.
        epsilon: small constant for numerical stability.
        use_scale: whether to use a scale parameter.
        scale_init: initializer for the scale parameter.
        dtype: data type for computation.
        param_dtype: data type of the parameters.
    """

    def __init__(
        self,
        size: int,
        epsilon: float = 1e-6,
        use_scale: bool = True,
        scale_init: Initializer = ones(),
        dtype: DType = jnp.float32,
        param_dtype: DType | None = None,
    ):
        super().__init__()
        self.size = size
        self.epsilon = epsilon
        self.use_scale = use_scale
        self.scale_init = scale_init
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.scale = None
        if use_scale:
            self.scale = Parameter(
                self.scale_init,
                shape=(size,),
                dtype=dtype,
                param_dtype=param_dtype,
            )

    def __call__(self, input: Array) -> Array:
        mean2 = jnp.mean(jnp.square(input), axis=-1, keepdims=True)
        norm = jnp.sqrt(mean2 + self.epsilon)
        output = input / norm

        if self.scale is not None:
            output *= self.scale

        return output
