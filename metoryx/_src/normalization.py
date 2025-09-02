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

    Batch normalization keeps a moving average of batch statistics.
    These are stored in the `batch_stats` collection.
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
        """Initialize BatchNorm layer.

        Args:
            size: Size of input features.
            momentum: Momentum for the moving average.
            epsilon: Small constant for numerical stability.
            use_scale: Whether to use a scale parameter.
            use_bias: Whether to use a bias parameter.
            scale_init: Initializer for the scale parameter.
            bias_init: Initializer for the bias parameter.
            dtype: Data type for computation.
            param_dtype: Data type of the parameters.
            axis_name: Axis name to sync batch statistics along devices.
            axis_index_groups: Axis index groups for distributed training.
        """
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

    def __call__(self, inputs: Array, is_training: bool = False) -> Array:
        if is_training:
            axis = tuple(i for i in range(inputs.ndim - 1))
            mean = jnp.mean(inputs, axis=axis)
            mean2 = jnp.mean(jnp.square(inputs), axis=axis)

            if self.axis_name:
                synced: tuple[Array, Array] = jax.lax.pmean(
                    (mean, mean2),
                    self.axis_name,
                    axis_index_groups=self.axis_index_groups,
                )
                mean, mean2 = synced

            var = mean2 - jnp.square(mean)
            output = (inputs - mean) / jnp.sqrt(var + self.epsilon)

            # Update batch stats
            self.mean.value = self.momentum * self.mean + (1 - self.momentum) * mean
            self.var.value = self.momentum * self.var + (1 - self.momentum) * var
        else:
            output = (inputs - self.mean) / jnp.sqrt(self.var + self.epsilon)

        if self.scale is not None:
            output *= self.scale
        if self.bias is not None:
            output += self.bias

        return output


class LayerNorm(Module):
    """Layer normalization.

    Ref. https://arxiv.org/abs/1607.06450
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
        """Initialize LayerNorm layer.

        Args:
            size: Size of input features.
            epsilon: Small constant for numerical stability.
            use_scale: Whether to use a scale parameter.
            use_bias: Whether to use a bias parameter.
            scale_init: Initializer for the scale parameter.
            bias_init: Initializer for the bias parameter.
            dtype: Data type for computation.
            param_dtype: Data type of the parameters.
        """
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

    def __call__(self, inputs: Array) -> Array:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        mean2 = jnp.mean(jnp.square(inputs), axis=-1, keepdims=True)
        var = mean2 - jnp.square(mean)
        output = (inputs - mean) / jnp.sqrt(var + self.epsilon)

        if self.scale is not None:
            output *= self.scale
        if self.bias is not None:
            output += self.bias

        return output


class RMSNorm(Module):
    """RMS layer normalization.

    Ref. https://arxiv.org/abs/1910.07467
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
        """Initialize RMSNorm layer.

        Args:
            size: Size of input features.
            epsilon: Small constant for numerical stability.
            use_scale: Whether to use a scale parameter.
            scale_init: Initializer for the scale parameter.
            dtype: Data type for computation.
            param_dtype: Data type of the parameters.
        """
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

    def __call__(self, inputs: Array) -> Array:
        mean2 = jnp.mean(jnp.square(inputs), axis=-1, keepdims=True)
        norm = jnp.sqrt(mean2 + self.epsilon)
        output = inputs / norm

        if self.scale is not None:
            output *= self.scale

        return output
