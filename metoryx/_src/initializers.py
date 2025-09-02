from typing import Literal, Sequence

from jax.nn import initializers as init

from metoryx._src.base import ArrayLike, DType, Initializer

__all__ = [
    "constant",
    "zeros",
    "ones",
    "delta_orthogonal",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "kaiming_normal",
    "kaiming_uniform",
    "lecun_normal",
    "lecun_uniform",
    "normal",
    "uniform",
    "orthogonal",
    "truncated_normal",
    "variance_scaling",
    "xavier_normal",
    "xavier_uniform",
]


def constant(value: ArrayLike, dtype: DType | None = None) -> Initializer:
    """Builds an initializer that generates arrays initialized to a constant value.

    Args:
        value: The constant value to initialize the array.
        dtype: The data type of the initialized array.

    Returns:
        An initializer that returns an array initialized to the specified constant value.
    """
    return init.constant(value, dtype)


def zeros(dtype: DType | None = None) -> Initializer:
    """Builds an initializer that generates arrays initialized to 0.

    Args:
        dtype: The data type of the initialized array.

    Returns:
        An initializer that returns an array initialized to 0.
    """
    return constant(0.0, dtype)


def ones(dtype: DType | None = None) -> Initializer:
    """Builds an initializer that generates arrays initialized to 1.

    Args:
        dtype: The data type of the initialized array.

    Returns:
        An initializer that returns an array initialized to 1.
    """
    return constant(1.0, dtype)


def delta_orthogonal(
    scale: float = 1.0, column_axis: int = -1, dtype: DType | None = None
) -> Initializer:
    """
    Builds an initializer for delta orthogonal kernels.

    Args:
      scale: the upper bound of the uniform distribution.
      column_axis: the axis that contains the columns that should be orthogonal.
      dtype: the default dtype of the weights.

    Returns:
      A delta orthogonal initializer. The shape passed to the initializer must
      be 3D, 4D, or 5D.
    """
    return init.delta_orthogonal(scale, column_axis, dtype)


def glorot_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    """Builds a Glorot normal initializer (aka Xavier normal initializer).

    A Glorot normal initializer is a specialization of
    variance_scaling where ``scale = 1.0``,
    ``mode="fan_avg"``, and ``distribution="truncated_normal"``.

    Args:
      in_axis: axis or sequence of axes of the input dimension in the weights
        array.
      out_axis: axis or sequence of axes of the output dimension in the weights
        array.
      batch_axis: axis or sequence of axes in the weight array that should be
        ignored.
      dtype: the dtype of the weights.

    Returns:
      An initializer.
    """
    return init.glorot_normal(in_axis, out_axis, batch_axis, dtype)


def glorot_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    """Builds a Glorot uniform initializer (aka Xavier uniform initializer).

    A Glorot uniform initializer is a specialization of
    variance_scaling where ``scale = 1.0``,
    ``mode="fan_avg"``, and ``distribution="uniform"``.

    Args:
      in_axis: axis or sequence of axes of the input dimension in the weights
        array.
      out_axis: axis or sequence of axes of the output dimension in the weights
        array.
      batch_axis: axis or sequence of axes in the weight array that should be
        ignored.
      dtype: the dtype of the weights.

    Returns:
      An initializer.
    """
    return init.glorot_uniform(in_axis, out_axis, batch_axis, dtype)


def he_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    """Builds a He normal initializer (aka Kaiming normal initializer).

    A He normal initializer is a specialization of
    variance_scaling where ``scale = 2.0``,
    ``mode="fan_in"``, and ``distribution="truncated_normal"``.

    Args:
      in_axis: axis or sequence of axes of the input dimension in the weights
        array.
      out_axis: axis or sequence of axes of the output dimension in the weights
        array.
      batch_axis: axis or sequence of axes in the weight array that should be
        ignored.
      dtype: the dtype of the weights.

    Returns:
      An initializer.
    """
    return init.he_normal(in_axis, out_axis, batch_axis, dtype)


def he_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    """Builds a He uniform initializer (aka Kaiming uniform initializer).

    A He uniform initializer is a specialization of
    variance_scaling where ``scale = 2.0``,
    ``mode="fan_in"``, and ``distribution="uniform"``.

    Args:
      in_axis: axis or sequence of axes of the input dimension in the weights
        array.
      out_axis: axis or sequence of axes of the output dimension in the weights
        array.
      batch_axis: axis or sequence of axes in the weight array that should be
        ignored.
      dtype: the dtype of the weights.

    Returns:
      An initializer.
    """
    return init.he_uniform(in_axis, out_axis, batch_axis, dtype)


def kaiming_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    """Builds a He normal initializer (aka Kaiming normal initializer).

    A He normal initializer is a specialization of
    variance_scaling where ``scale = 2.0``,
    ``mode="fan_in"``, and ``distribution="truncated_normal"``.

    Args:
      in_axis: axis or sequence of axes of the input dimension in the weights
        array.
      out_axis: axis or sequence of axes of the output dimension in the weights
        array.
      batch_axis: axis or sequence of axes in the weight array that should be
        ignored.
      dtype: the dtype of the weights.

    Returns:
      An initializer.
    """
    return init.he_normal(in_axis, out_axis, batch_axis, dtype)


def kaiming_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    """Builds a He uniform initializer (aka Kaiming uniform initializer).

    A He uniform initializer is a specialization of
    variance_scaling where ``scale = 2.0``,
    ``mode="fan_in"``, and ``distribution="uniform"``.

    Args:
      in_axis: axis or sequence of axes of the input dimension in the weights
        array.
      out_axis: axis or sequence of axes of the output dimension in the weights
        array.
      batch_axis: axis or sequence of axes in the weight array that should be
        ignored.
      dtype: the dtype of the weights.

    Returns:
      An initializer.
    """
    return init.he_uniform(in_axis, out_axis, batch_axis, dtype)


def lecun_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    """Builds a Lecun normal initializer.

    A Lecun normal initializer is a specialization of
    variance_scaling where ``scale = 1.0``,
    ``mode="fan_in"``, and ``distribution="truncated_normal"``.

    Args:
      in_axis: axis or sequence of axes of the input dimension in the weights
        array.
      out_axis: axis or sequence of axes of the output dimension in the weights
        array.
      batch_axis: axis or sequence of axes in the weight array that should be
        ignored.
      dtype: the dtype of the weights.

    Returns:
      An initializer.
    """
    return init.lecun_normal(in_axis, out_axis, batch_axis, dtype)


def lecun_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    """Builds a Lecun uniform initializer.

    A Lecun uniform initializer is a specialization of
    variance_scaling where ``scale = 1.0``,
    ``mode="fan_in"``, and ``distribution="uniform"``.

    Args:
      in_axis: axis or sequence of axes of the input dimension in the weights
        array.
      out_axis: axis or sequence of axes of the output dimension in the weights
        array.
      batch_axis: axis or sequence of axes in the weight array that should be
        ignored.
      dtype: the dtype of the weights.

    Returns:
      An initializer.
    """
    return init.lecun_uniform(in_axis, out_axis, batch_axis, dtype)


def normal(stddev: float = 1e-2, dtype: DType | None = None) -> Initializer:
    """Builds an initializer that returns real normally-distributed random arrays.

    Args:
      stddev: optional; the standard deviation of the distribution.
      dtype: optional; the initializer's default dtype.

    Returns:
      An initializer that returns arrays whose values are normally distributed
      with mean ``0`` and standard deviation ``stddev``.
    """
    return init.normal(stddev, dtype)


def uniform(scale: float = 1e-2, dtype: DType | None = None) -> Initializer:
    """Builds an initializer that returns real uniformly-distributed random arrays.

    Args:
      scale: optional; the upper bound of the random distribution.
      dtype: optional; the initializer's default dtype.

    Returns:
      An initializer that returns arrays whose values are uniformly distributed in
      the range ``[0, scale)``.
    """
    return init.uniform(scale, dtype)


def orthogonal(
    scale: float = 1.0, column_axis: int = -1, dtype: DType | None = None
) -> Initializer:
    """
    Builds an initializer that returns uniformly distributed orthogonal matrices.

    If the shape is not square, the matrices will have orthonormal rows or columns
    depending on which side is smaller.

    Args:
      scale: the upper bound of the uniform distribution.
      column_axis: the axis that contains the columns that should be orthogonal.
      dtype: the default dtype of the weights.

    Returns:
      An orthogonal initializer.
    """
    return init.orthogonal(scale, column_axis, dtype)


def truncated_normal(
    stddev: float = 1e-2, dtype: DType | None = None, lower: float = -2.0, upper: float = 2.0
) -> Initializer:
    """Builds an initializer that returns truncated-normal random arrays.

    Args:
      stddev: optional; the standard deviation of the untruncated distribution.
        Note that this function does not apply the stddev correction as is done in
        the variancescaling initializers, and users are expected to apply this
        correction themselves via the stddev arg if they wish to employ it.
      dtype: optional; the initializer's default dtype.
      lower: Float representing the lower bound for truncation. Applied before
        the output is multiplied by the stddev.
      upper: Float representing the upper bound for truncation. Applied before
        the output is multiplied by the stddev.

    Returns:
      An initializer that returns arrays whose values follow the truncated normal
      distribution with mean ``0`` and standard deviation ``stddev``, and range
      lower * stddev < x < upper * stddev.
    """
    return init.truncated_normal(stddev, dtype, lower, upper)


def variance_scaling(
    scale: float,
    mode: Literal["fan_in", "fan_out", "fan_avg", "fan_geo_avg"],
    distribution: Literal["truncated_normal", "normal", "uniform"],
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    """
    Initializer that adapts its scale to the shape of the weights tensor.

    With ``distribution="truncated_normal"`` or ``distribution="normal"``, samples
    are drawn from a (truncated) normal distribution with a mean of zero
    and a standard deviation (after truncation, if applicable) of
    sqrt(scale/n), where `n` is, for each ``mode``:

    * ``"fan_in"``: the number of inputs
    * ``"fan_out"``: the number of outputs
    * ``"fan_avg"``: the arithmetic average of the numbers of inputs and outputs
    * ``"fan_geo_avg"``: the geometric average of the numbers of inputs and outputs

    This initializer can be configured with ``in_axis``, ``out_axis``, and
    ``batch_axis`` to work with general convolutional or dense layers; axes that
    are not in any of those arguments are assumed to be the "receptive field"
    (convolution kernel spatial axes).

    With ``distribution="truncated_normal"``, the absolute values of the samples
    are truncated at 2 standard deviations before scaling.

    With ``distribution="uniform"``, samples are drawn from:

    * a uniform interval, if `dtype` is real, or
    * a uniform disk, if `dtype` is complex,

    with a mean of zero and a standard deviation of sqrt(scale/n)
    where `n` is defined above.

    Args:
      scale: scaling factor (positive float).
      mode: one of ``"fan_in"``, ``"fan_out"``, ``"fan_avg"``, and ``"fan_geo_avg"``.
      distribution: random distribution to use. One of ``"truncated_normal"``,
        ``"normal"`` and ``"uniform"``.
      in_axis: axis or sequence of axes of the input dimension in the weights
        array.
      out_axis: axis or sequence of axes of the output dimension in the weights
        array.
      batch_axis: axis or sequence of axes in the weight array that should be
        ignored.
      dtype: the dtype of the weights.
    """
    return init.variance_scaling(scale, mode, distribution, in_axis, out_axis, batch_axis, dtype)


def xavier_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    """Builds a Glorot normal initializer (aka Xavier normal initializer).

    A Glorot normal initializer is a specialization of
    variance_scaling where ``scale = 1.0``,
    ``mode="fan_avg"``, and ``distribution="truncated_normal"``.

    Args:
      in_axis: axis or sequence of axes of the input dimension in the weights
        array.
      out_axis: axis or sequence of axes of the output dimension in the weights
        array.
      batch_axis: axis or sequence of axes in the weight array that should be
        ignored.
      dtype: the dtype of the weights.

    Returns:
      An initializer.
    """
    return init.glorot_normal(in_axis, out_axis, batch_axis, dtype)


def xavier_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    """Builds a Glorot uniform initializer (aka Xavier uniform initializer).

    A Glorot uniform initializer is a specialization of
    variance_scaling where ``scale = 1.0``,
    ``mode="fan_avg"``, and ``distribution="uniform"``.

    Args:
      in_axis: axis or sequence of axes of the input dimension in the weights
        array.
      out_axis: axis or sequence of axes of the output dimension in the weights
        array.
      batch_axis: axis or sequence of axes in the weight array that should be
        ignored.
      dtype: the dtype of the weights.

    Returns:
      An initializer.
    """
    return init.glorot_uniform(in_axis, out_axis, batch_axis, dtype)
