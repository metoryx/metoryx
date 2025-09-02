import jax.numpy as jnp
import jax.random as jr

from metoryx._src.base import (
    Array,
    next_rng_key,
)


def dropout(
    inputs: Array,
    rate: float,
    is_training: bool,
    *,
    rng_collection: str | None = None,
) -> Array:
    """Applies dropout to the input array.

    During training, randomly sets a fraction `rate` of input units to zero
    at each update step, which helps prevent overfitting. During evaluation,
    the input is returned unchanged.

    Args:
        inputs: Input array.
        rate: Fraction of the input units to drop. Must be between 0 and 1.
        is_training: Whether the model is in training mode.
        rng_collection: Name of the RNG collection to use for generating
            dropout masks. If None, uses the default RNG collection.

    Returns:
        The array after applying dropout.
    """
    if not 0 <= rate < 1:
        raise ValueError("Dropout rate must be in the range [0, 1).")

    if not is_training or rate == 0.0:
        return inputs

    keep_prob = 1.0 - rate
    rng = next_rng_key(rng_collection)
    mask = jr.bernoulli(rng, p=keep_prob, shape=inputs.shape)
    return jnp.where(mask, inputs / keep_prob, 0)
