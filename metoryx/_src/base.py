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
    """Get the value of the context variable. Raises an error if not set.

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
    """Prepare PRNGKeys.

    Args:
        default: The default PRNGKey to use if none is provided.
            Note that this is positional-only, and must be provided as the first argument.
        **kwargs: Additional PRNGKeys to use for specific purposes.

    Returns:
        A dictionary mapping string names to PRNGKeys.
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
    """

    Attributes:
        col: The collection name for the state management.
        init: The initializer function for the state.
        shape: The shape of the state.
        dtype: The data type of the state.
        param_dtype: The parameter data type of the state.
            If None, defaults to `dtype`.
        mutable: Whether the state is mutable.
            If false, the state value is not settable during applying.
    """

    def __init__(
        self,
        col: str,
        init: Initializer,
        shape: Shape,
        dtype: DType = jnp.float32,
        param_dtype: DType | None = None,
        mutable: bool = False,
    ):
        self.col = col
        self.init = init
        self.shape = shape
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.mutable = mutable

    @property
    def value(self) -> Array:
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
                # todo: consider broadcast-able.
                raise ValueError(f"Shape mismatch: {self.shape} != {array.shape}")
            self.init = lambda rng, shape, dtype: jnp.asarray(array, dtype=dtype)

    @property
    def id(self) -> str:
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
        return self.value.T

    @property
    def ndim(self) -> int:
        return len(self.shape)


class Parameter(State):
    def __init__(
        self,
        init: Initializer,
        shape: Shape,
        dtype: DType = jnp.float32,
        param_dtype: DType | None = None,
    ):
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
        arrays: The arrays to assign to the module.

    Returns:
        The module with assigned arrays.

    Notes:
        - The assigned arrays will be reflected when the module is initialized.
        - Currently, arrays tree structure must match the module's variable structure.
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
    """Transform module into initialization function.

    Args:
        module: The module to transform.

    Returns:
        The initialization function for the module.

    Note:
        This function deepcopy the module to detach it from the original context.
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


def apply(module: Module, fn: Callable[[Module], Callable] | None = None) -> ApplyFn:
    """Transform module into applying function.

    Args:
        module: The module to transform.
        fn: An optional function to convert module into a callable.
            Useful for wrapping non-callable modules or applying child module.

    Returns:
        The applying function for the module.

    Note:
        This function deepcopy the module to detach it from the original context.
    """

    module = copy.deepcopy(module)
    if fn is None:
        if not callable(module):
            raise ValueError("Module is not callable.")
        fn = lambda m: getattr(m, "__call__")

    def assign(states: States, variables: Variables) -> dict[str, Array]:
        arrays_to_assign = {}

        def f(state: State, array: Array):
            arrays_to_assign[state.id] = array

        jax.tree.map(f, states, variables)
        return arrays_to_assign

    def apply_fn(
        variables: Variables,
        rngs: PRNGKey | dict[str, PRNGKey] | None = None,
        *args,
        **kwargs,
    ):
        # deepcopy module again to avoid side-effects caused in `fn(m)`.
        _module = copy.deepcopy(module)

        if rngs is None:
            rngs = dict()
        elif isinstance(rngs, jax.Array):
            rngs = PRNGKeys(rngs)

        states = get_states(_module)
        assigned_variables = assign(states, variables)

        with (
            using_context(status_context, "applying"),
            using_context(array_context, assigned_variables),
            using_context(rng_context, rngs),
        ):
            outputs = fn(_module)(*args, **kwargs)
            new_variables = jax.tree.map(lambda v: assigned_variables[v.id], states)
            return outputs, new_variables

    return apply_fn


def transform(
    module: Module,
    fn: Callable[[Module], Callable] | None = None,
) -> Transformed:
    """Transform module into initialization and applying functions.

    Args:
        module: The module to transform.
        fn: An optional function to convert module into a callable.
            Useful for wrapping non-callable modules or applying child module.

    Returns:
        A transformed module with separate init and apply functions.

    Note:
        This function deepcopy the module to detach it from the original context.
    """

    return Transformed(
        init=init(module),
        apply=apply(module, fn),
    )
