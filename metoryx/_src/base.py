import collections
import contextlib
import copy
from contextvars import ContextVar
from typing import Any, Callable, Literal, Mapping, NamedTuple, Optional, Protocol, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = [
    "Array",
    "ArrayLike",
    "Variables",
    "Shape",
    "PRNGKey",
    "DType",
    "Initializer",
    "Status",
    "InitFn",
    "ApplyFn",
    "PRNGKeys",
    "next_rng_key",
    "State",
    "Parameter",
    "Module",
    "Transformed",
    "init",
    "apply",
    "transform",
    "assign_variables",
]


#
#  Types
#
type PyTree[T] = T | list[PyTree[T]] | dict[str, PyTree[T]]
type Array = jax.Array
type ArrayLike = jax.typing.ArrayLike
type States = dict[str, PyTree[State]]
type Variables = dict[str, PyTree[Array]]
type Shape = Sequence[int]
type PRNGKey = Array
type DType = jax.typing.DTypeLike
type Status = Literal["initializing", "applying"]


class Initializer(Protocol):
    def __call__(self, key: PRNGKey, shape: Shape, dtype: Optional[DType] = None) -> Array: ...


class InitFn(Protocol):
    def __call__(self, rng: PRNGKey) -> Variables: ...


class ApplyFn(Protocol):
    def __call__(
        self,
        variables: Variables,
        rngs: Optional[PRNGKey | dict[str, PRNGKey]] = None,
        *args,
        **kwargs,
    ) -> tuple[Any, Variables]: ...


#
#  Contextvars
#
status_context = ContextVar[Status | None]("status_context", default=None)
array_context = ContextVar[dict[str, Array]]("array_context")
rng_context = ContextVar[dict[str, PRNGKey]]("rng_context")


def get_context[T](ctx: ContextVar[T]) -> T:
    """Get the value of the context variable. Raises an error if it is not set.

    Args:
        ctx: The context variable to get the value from.

    Returns:
        The value of the context variable.
    """

    try:
        return ctx.get()
    except LookupError:
        raise RuntimeError(f"Context '{ctx.name}' is not set.")


@contextlib.contextmanager
def using_context[T](ctx: ContextVar[T], value: T):
    token = ctx.set(value)
    try:
        yield
    finally:
        ctx.reset(token)


#
#  Randomness
#
def PRNGKeys(default: PRNGKey | None = None, /, **kwargs: PRNGKey) -> dict[str, PRNGKey]:
    """Create a dictionary of PRNGKeys to feed into the apply function.

    Args:
        default: The default PRNGKey to use if none is provided.
        **kwargs: Additional PRNGKeys to use for specific purposes.

    Returns:
        A dictionary of PRNGKeys to feed into the apply function.
    """
    if "__default__" in kwargs:
        raise ValueError('"__default__" is a reserved key for the default PRNGKey.')
    rngs = dict(kwargs)
    if default is not None:
        rngs["__default__"] = default
    return rngs


def next_rng_key(
    name: str | None = None,
    num: int | tuple[int, ...] | None = None,
    *,
    strict: bool = False,
) -> PRNGKey:
    """Get the next PRNGKey. This function is only available within the `applying` phase.

    Args:
        name: The name of the PRNGKey to use. If None, uses the default PRNGKey.
        num: If provided, splits the PRNGKey into `num` keys.
        strict: If True, raises an error if the specified name is not found in the context.
            If False, falls back to the default PRNGKey if the specified name is not found.

    Returns:
        The next PRNGKey.

    Note:
        This function also updates the context with the new PRNGKey.
        Thus, subsequent calls to this function will return different keys.

    Raises:
        ValueError: If the context is not set, or if the specified name is not found in the
            context and `strict` is True, or if the default PRNGKey is not found in the context.
    """

    rngs = get_context(rng_context)
    if name is None:
        name = "__default__"
    elif name not in rngs:
        if strict:
            raise ValueError(f"PRNGKey for '{name}' is not found in the context.")
        else:
            name = "__default__"

    if name not in rngs:
        raise ValueError("Default PRNGKey is not found in the context.")

    rng = rngs[name]
    next_rng, new_rng = jr.split(rng)

    # Update the context with the new PRNGKey.
    new_rngs = rngs.copy()
    new_rngs[name] = new_rng
    rng_context.set(new_rngs)

    # Split `next_rng` if necessary.
    if num is not None:
        next_rng = jr.split(new_rng, num)

    return next_rng


#
#  State and Parameter
#
class State:
    """A container for a stateful variable in a module. Lazily initialized and can be mutable."""

    def __init__(
        self,
        col: str,
        init: Initializer,
        shape: Shape,
        dtype: DType = jnp.float32,
        param_dtype: DType | None = None,
        mutable: bool = False,
    ):
        """Initializes the State.

        Args:
            col: The collection name for the state variable.
            init: The initializer function for the state variable.
            shape: The shape of the state variable.
            dtype: The data type of the state variable.
            param_dtype: The data type for computation. If None, uses `dtype`.
            mutable: Whether the state variable is mutable during the `applying` phase.
        """
        self.col = col
        self.init = init
        self.shape = shape
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.mutable = mutable

    @property
    def value(self) -> Array:
        """Get or initialize the value of the state variable.

        This property is only available during the *initializing* or *applying* phase.
        In the *initializing* phase, it initializes the variable using the provided initializer.
        In the *applying* phase, it retrieves the current value from the context.

        This property can also be set to an array with the same shape as the state variable.
        In the *applying* phase, if the state variable is mutable, setting this property
        updates the value in the context. Otherwise, setting this property updates the initializer
        to return the provided array during the next initialization.
        """
        status = get_context(status_context)
        if status == "initializing":
            rng = next_rng_key()
            param_dtype = self.param_dtype or self.dtype
            return self.init(rng, self.shape, param_dtype)
        elif status == "applying":
            arrays = get_context(array_context)
            array = arrays.get(self.id, None)
            if array is None:
                class_name = self.__class__.__name__
                raise ValueError(
                    f"Array for {class_name} with id '{self.id}' is not found in the context."
                )
            return jnp.asarray(array, dtype=self.dtype)
        else:
            raise RuntimeError("Variable value is only available during initializing or applying.")

    @value.setter
    def value(self, array: Array) -> None:
        status = get_context(status_context)
        if status == "applying":
            if self.mutable:
                arrays = get_context(array_context)
                arrays[self.id] = array
                array_context.set(arrays)
            else:
                raise RuntimeError("Cannot set value of an immutable variable.")
        else:
            if self.shape != array.shape:
                # TODO: consider broadcastable arrays.
                raise ValueError(f"Shape mismatch: {self.shape} != {array.shape}")
            self.init = lambda rng, shape, dtype: jnp.asarray(array, dtype=dtype)

    @property
    def id(self) -> str:
        """A unique identifier for the state variable."""
        return str(id(self))

    def __jax_array__(self) -> Array:
        return self.value

    def __add__(self, other) -> Array:
        return self.value + other

    def __radd__(self, other) -> Array:
        return other + self.value

    def __iadd__(self, other) -> Array:
        self.value = self.value + other
        return self.value

    def __sub__(self, other) -> Array:
        return self.value - other

    def __rsub__(self, other) -> Array:
        return other - self.value

    def __isub__(self, other) -> Array:
        self.value = self.value - other
        return self.value

    def __mul__(self, other) -> Array:
        return self.value * other

    def __rmul__(self, other) -> Array:
        return other * self.value

    def __imul__(self, other) -> Array:
        self.value = self.value * other
        return self.value

    def __truediv__(self, other) -> Array:
        return self.value / other

    def __rtruediv__(self, other) -> Array:
        return other / self.value

    def __itruediv__(self, other) -> Array:
        self.value = self.value / other
        return self.value

    def __floordiv__(self, other) -> Array:
        return self.value // other

    def __rfloordiv__(self, other) -> Array:
        return other // self.value

    def __ifloordiv__(self, other) -> Array:
        self.value = self.value // other
        return self.value

    def __pow__(self, other) -> Array:
        return self.value**other

    def __rpow__(self, other) -> Array:
        return other**self.value

    def __ipow__(self, other) -> Array:
        self.value = self.value**other
        return self.value

    def __matmul__(self, other) -> Array:
        return self.value @ other

    def __rmatmul__(self, other) -> Array:
        return other @ self.value

    def __imatmul__(self, other) -> Array:
        self.value = self.value @ other
        return self.value

    @property
    def T(self) -> Array:
        """Transpose of the state variable."""
        return self.value.T

    @property
    def ndim(self) -> int:
        """Number of dimensions of the state variable."""
        return len(self.shape)


class Parameter(State):
    """A container for a parameter variable in a module. Typically immutable during applying."""

    def __init__(
        self,
        init: Initializer,
        shape: Shape,
        dtype: DType = jnp.float32,
        param_dtype: DType | None = None,
    ):
        """Initializes the Parameter.

        Args:
            init: The initializer function for the parameter variable.
            shape: The shape of the parameter variable.
            dtype: The data type of the parameter variable.
            param_dtype: The data type for computation. If None, uses `dtype`.
        """
        super().__init__(
            "params",
            init,
            shape,
            dtype,
            param_dtype,
            mutable=False,
        )


#
#  Module
#
class Module:
    """Base class for all neural network modules."""

    pass


def get_states(obj: Any) -> States:
    """Recursively get all State instances from the given object."""

    if isinstance(obj, State):
        return {obj.col: obj}

    it = dict()
    if isinstance(obj, (tuple, list, collections.deque)):
        it = enumerate(obj)
    elif isinstance(obj, (Mapping, Module)):
        it = obj.items() if isinstance(obj, Mapping) else obj.__dict__.items()

    states = collections.defaultdict(dict)
    for key, val in it:
        for col, vars in get_states(val).items():
            states[col][str(key)] = vars

    return dict(states)


def assign_variables(module: Module, variables: Variables) -> Module:
    """Assign arrays to the module.

    Args:
        module: The module to assign arrays to.
        variables: The variables to assign to the module.

    Returns:
        The module with assigned arrays.

    Notes:
        - The assigned arrays will be reflected when the module is initialized.
        - Currently, the array tree structure must match the module's variable structure.
    """

    def assign(state: State, variable: Array) -> None:
        state.value = variable

    module = copy.deepcopy(module)
    states = get_states(module)
    jax.tree.map(assign, states, variables)
    return module


#
#  Module Transformation
#
class Transformed(NamedTuple):
    """A transformed module with separate init and apply functions."""

    init: InitFn
    apply: ApplyFn


def init(module: Module) -> InitFn:
    """Transform a module into an initialization function.

    Args:
        module: The module to transform.

    Returns:
        The initialization function for the module.
    """
    module = copy.deepcopy(module)
    states = get_states(module)

    def init_fn(rng: PRNGKey) -> Variables:
        with (
            using_context(status_context, "initializing"),
            using_context(rng_context, PRNGKeys(rng)),
        ):
            variables = jax.tree.map(lambda v: v.value, states)
            return variables

    return init_fn


def _make_bound_arrays(states: States, variables: Variables) -> dict[str, Array]:
    bound_arrays = {}

    def bind(state: State, array: Array):
        bound_arrays[state.id] = array

    jax.tree.map(bind, states, variables)
    return bound_arrays


def apply(
    module: Module,
    *,
    to_callable: Callable[[Module], Callable[..., Any]] | None = None,
) -> ApplyFn:
    """Transform a module into an apply function.

    Args:
        module: The module to transform.
        to_callable: An optional function to convert module into a callable.
            If None, the module itself must be callable.

    Returns:
        The apply function for the module.
    """

    module = copy.deepcopy(module)
    if to_callable is None:
        if not callable(module):
            raise ValueError("Module is not callable.")
        to_callable = lambda m: getattr(m, "__call__")

    def apply_fn(
        variables: Variables,
        rngs: PRNGKey | dict[str, PRNGKey] | None = None,
        *args,
        **kwargs,
    ):
        # deepcopy module again to avoid side-effects caused by `fn(m)`.
        _module = copy.deepcopy(module)

        if rngs is None:
            rngs = dict()
        elif isinstance(rngs, jax.Array):
            rngs = PRNGKeys(rngs)

        states = get_states(_module)
        bound_arrays = _make_bound_arrays(states, variables)

        with (
            using_context(status_context, "applying"),
            using_context(array_context, bound_arrays),
            using_context(rng_context, rngs),
        ):
            outputs = to_callable(_module)(*args, **kwargs)
            new_variables = jax.tree.map(lambda v: bound_arrays[v.id], states)
            return outputs, new_variables

    return apply_fn


def transform(
    module: Module,
    *,
    to_callable: Callable[[Module], Callable[..., Any]] | None = None,
) -> Transformed:
    """Transform a module into initialization and applying functions.

    Args:
        module: The module to transform.
        to_callable: An optional function to convert module into a callable.
            If None, the module itself must be callable.

    Returns:
        A transformed module with separate init and apply functions.
    """

    return Transformed(
        init=init(module),
        apply=apply(module, to_callable=to_callable),
    )


def _lift(
    to_lift: Callable,
    module: Module,
    *args,
    to_callable: Callable[[Module], Callable[..., Any]] | None = None,
    **kwargs,
):
    module = copy.deepcopy(module)
    if to_callable is None:
        if not callable(module):
            raise ValueError("Module is not callable.")
        to_callable = lambda m: getattr(m, "__call__")

    def to_apply(
        variables: Variables, rngs: PRNGKey | dict[str, PRNGKey] | None = None, *args, **kwargs
    ):
        # deepcopy module again to avoid side-effects caused by `fn(m)`.
        _module = copy.deepcopy(module)

        if rngs is None:
            rngs = dict()
        elif isinstance(rngs, jax.Array):
            rngs = PRNGKeys(rngs)

        states = get_states(_module)
        bound_arrays = _make_bound_arrays(states, variables)

        with (
            using_context(array_context, bound_arrays),
            using_context(rng_context, rngs),
        ):
            outputs = to_callable(_module)(*args, **kwargs)
            new_rngs = get_context(rng_context)
            new_variables = jax.tree.map(lambda v: bound_arrays[v.id], states)
            return outputs, new_rngs, new_variables

    to_apply = to_lift(to_apply, *args, **kwargs)

    def update_array_context(states, variables: Array):
        arrays = get_context(array_context).copy()

        def update(state: State, variable: Array):
            arrays[state.id] = variable

        jax.tree.map(update, states, variables)
        array_context.set(arrays)

    def lifted(*args, **kwargs):
        if get_context(status_context) != "applying":
            raise RuntimeError("Lifted function can only be called during the applying phase.")

        rngs = get_context(rng_context)
        states = get_states(module)
        variables = jax.tree.map(lambda v: v.value, states)

        outputs, new_rngs, new_variables = to_apply(variables, rngs, *args, **kwargs)

        # Update the context with the new rngs and variables.
        rng_context.set(new_rngs)
        update_array_context(states, new_variables)

        return outputs

    return lifted


def checkpoint(
    module: Module,
    *,
    to_callable: Callable[[Module], Callable] | None = None,
    concrete: bool = False,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: Callable[..., bool] | None = None,
) -> Callable[..., Any]:
    """Transform a module into a function that applies `jax.checkpoint` to its call method.

    Args:
        module: The module to wrap.
        to_callable: An optional function to convert module into a callable.
            If None, the module itself must be callable.
        concrete: Whether to use concrete mode in jax.checkpoint.
        prevent_cse: Whether to prevent common subexpression elimination in jax.checkpoint.
        static_argnums: Static argument numbers to pass to jax.checkpoint.
        policy: A custom policy function to pass to jax.checkpoint.

    Returns:
        A function that applies `jax.checkpoint` to the module's call method.
    """
    return _lift(
        jax.checkpoint,  # type: ignore
        module,
        to_callable=to_callable,
        concrete=concrete,
        prevent_cse=prevent_cse,
        static_argnums=static_argnums,
        policy=policy,
    )
