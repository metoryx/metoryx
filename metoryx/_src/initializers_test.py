import jax
import jax.numpy as jnp

from metoryx._src import initializers


def test_constant():
    key = jax.random.PRNGKey(0)
    shape = (3, 4)

    init_fn = initializers.constant(5.0)
    params = init_fn(key, shape)
    assert params.shape == shape
    assert jnp.all(params == 5.0)

    init_fn = initializers.constant(2.5, dtype=jnp.float32)
    params = init_fn(key, shape)
    assert params.dtype == jnp.float32
    assert jnp.all(params == 2.5)


def test_zeros():
    key = jax.random.PRNGKey(0)
    shape = (3, 4)

    init_fn = initializers.zeros()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert jnp.all(params == 0.0)

    init_fn = initializers.zeros(dtype=jnp.int32)
    params = init_fn(key, shape)
    assert params.dtype == jnp.int32
    assert jnp.all(params == 0)


def test_ones():
    key = jax.random.PRNGKey(0)
    shape = (3, 4)

    init_fn = initializers.ones()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert jnp.all(params == 1.0)

    init_fn = initializers.ones(dtype=jnp.int32)
    params = init_fn(key, shape)
    assert params.dtype == jnp.int32
    assert jnp.all(params == 1)


def test_delta_orthogonal():
    key = jax.random.PRNGKey(0)
    shape = (8, 8, 3, 3)

    init_fn = initializers.delta_orthogonal()
    params = init_fn(key, shape)
    assert params.shape == shape

    init_fn = initializers.delta_orthogonal(scale=2.0, column_axis=-1)
    params = init_fn(key, shape)
    assert params.shape == shape


def test_glorot_normal():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.glorot_normal()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.glorot_normal(in_axis=-2, out_axis=-1, dtype=jnp.float32)
    params = init_fn(key, shape)
    assert params.dtype == jnp.float32


def test_glorot_uniform():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.glorot_uniform()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.glorot_uniform(in_axis=-2, out_axis=-1, batch_axis=())
    params = init_fn(key, shape)
    assert params.shape == shape


def test_he_normal():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.he_normal()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.he_normal(in_axis=-2, out_axis=-1)
    params = init_fn(key, shape)
    assert params.shape == shape


def test_he_uniform():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.he_uniform()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.he_uniform(batch_axis=())
    params = init_fn(key, shape)
    assert params.shape == shape


def test_kaiming_normal():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.kaiming_normal()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.kaiming_normal(in_axis=-2, out_axis=-1)
    params = init_fn(key, shape)
    assert params.shape == shape


def test_kaiming_uniform():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.kaiming_uniform()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.kaiming_uniform(in_axis=-2, out_axis=-1, batch_axis=())
    params = init_fn(key, shape)
    assert params.shape == shape


def test_lecun_normal():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.lecun_normal()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.lecun_normal(in_axis=-2, out_axis=-1)
    params = init_fn(key, shape)
    assert params.shape == shape


def test_lecun_uniform():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.lecun_uniform()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.lecun_uniform(batch_axis=())
    params = init_fn(key, shape)
    assert params.shape == shape


def test_normal():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.normal()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.normal(stddev=0.05, dtype=jnp.float32)
    params = init_fn(key, shape)
    assert params.dtype == jnp.float32

    std = jnp.std(params)
    assert jnp.abs(std - 0.05) < 0.01


def test_uniform():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.uniform()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32
    assert jnp.all(params >= -0.01) and jnp.all(params <= 0.01)

    init_fn = initializers.uniform(scale=0.1, dtype=jnp.float32)
    params = init_fn(key, shape)
    assert params.dtype == jnp.float32
    assert jnp.all(params >= -0.1) and jnp.all(params <= 0.1)


def test_orthogonal():
    key = jax.random.PRNGKey(0)
    shape = (20, 20)

    init_fn = initializers.orthogonal()
    params = init_fn(key, shape)
    assert params.shape == shape

    init_fn = initializers.orthogonal(scale=2.0, column_axis=-1)
    params = init_fn(key, shape)
    assert params.shape == shape


def test_truncated_normal():
    key = jax.random.PRNGKey(0)
    shape = (100, 100)

    init_fn = initializers.truncated_normal()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.truncated_normal(stddev=0.05, lower=-1.0, upper=1.0)
    params = init_fn(key, shape)
    assert jnp.all(params >= -1.0) and jnp.all(params <= 1.0)


def test_variance_scaling():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.variance_scaling(scale=2.0)
    params = init_fn(key, shape)
    assert params.shape == shape

    init_fn = initializers.variance_scaling(scale=2.0, mode="fan_out", distribution="normal")
    params = init_fn(key, shape)
    assert params.shape == shape

    init_fn = initializers.variance_scaling(
        scale=1.0, mode="fan_avg", distribution="uniform", in_axis=-2, out_axis=-1
    )
    params = init_fn(key, shape)
    assert params.shape == shape


def test_xavier_normal():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.xavier_normal()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.xavier_normal(in_axis=-2, out_axis=-1, dtype=jnp.float32)
    params = init_fn(key, shape)
    assert params.dtype == jnp.float32


def test_xavier_uniform():
    key = jax.random.PRNGKey(0)
    shape = (10, 20)

    init_fn = initializers.xavier_uniform()
    params = init_fn(key, shape)
    assert params.shape == shape
    assert params.dtype == jnp.float32

    init_fn = initializers.xavier_uniform(batch_axis=(), dtype=jnp.float32)
    params = init_fn(key, shape)
    assert params.dtype == jnp.float32
