from typing import Literal, Sequence

from jax.nn import initializers

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
    return initializers.constant(value, dtype=dtype)


def zeros(dtype: DType | None = None) -> Initializer:
    return constant(0, dtype=dtype)


def ones(dtype: DType | None = None) -> Initializer:
    return constant(1, dtype=dtype)


def delta_orthogonal(
    scale: float = 1.0, column_axis: int = -1, dtype: DType | None = None
) -> Initializer:
    return initializers.delta_orthogonal(
        scale=scale,
        column_axis=column_axis,
        dtype=dtype,
    )


def glorot_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    return initializers.glorot_normal(
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def glorot_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    return initializers.glorot_uniform(
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def he_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    return initializers.he_normal(
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def he_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    return initializers.he_uniform(
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def kaiming_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    return initializers.kaiming_normal(
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def kaiming_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    return initializers.kaiming_uniform(
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def lecun_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    return initializers.lecun_normal(
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def lecun_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    return initializers.lecun_uniform(
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def normal(stddev: float = 0.01, dtype: DType | None = None) -> Initializer:
    return initializers.normal(stddev=stddev, dtype=dtype)


def uniform(scale: float = 0.01, dtype: DType | None = None) -> Initializer:
    return initializers.uniform(scale=scale, dtype=dtype)


def orthogonal(
    scale: float = 1.0, column_axis: int = -1, dtype: DType | None = None
) -> Initializer:
    return initializers.orthogonal(scale=scale, column_axis=column_axis, dtype=dtype)


def truncated_normal(
    stddev: float = 0.01,
    dtype: DType | None = None,
    lower: float = -2.0,
    upper: float = 2.0,
) -> Initializer:
    return initializers.truncated_normal(stddev=stddev, dtype=dtype, lower=lower, upper=upper)


def variance_scaling(
    scale: float,
    mode: Literal["fan_in", "fan_out", "fan_avg", "fan_geo_avg"] = "fan_in",
    distribution: Literal["truncated_normal", "normal", "uniform"] = "truncated_normal",
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    return initializers.variance_scaling(
        scale=scale,
        mode=mode,
        distribution=distribution,
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def xavier_normal(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    return initializers.xavier_normal(
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def xavier_uniform(
    in_axis: int | Sequence[int] = -2,
    out_axis: int | Sequence[int] = -1,
    batch_axis: int | Sequence[int] = (),
    dtype: DType | None = None,
) -> Initializer:
    return initializers.xavier_uniform(
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )
