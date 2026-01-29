"""Decorators for tracking JAX function execution."""

from functools import wraps
from typing import Callable, TypeVar, overload

from jax.profiler import TraceAnnotation

F = TypeVar("F", bound=Callable)

PERFORMAX_PREFIX = "performax/"


@overload
def track(fn: F) -> F: ...


@overload
def track(*, name: str) -> Callable[[F], F]: ...


def track(fn: F | None = None, *, name: str | None = None) -> F | Callable[[F], F]:
    """Decorator that wraps a function with a TraceAnnotation for profiling.

    This decorator uses JAX's TraceAnnotation to add timing information that
    appears in profiler traces. The annotation works at Python level and
    captures the time spent in the decorated function.

    Can be used as:
        @track
        def fn(): ...

        @track(name="custom_name")
        def fn(): ...

    Args:
        fn: The function to wrap (when used without arguments).
        name: Optional custom name for the annotation (defaults to function name).

    Returns:
        The wrapped function with profiling annotation.

    Note:
        When used inside JIT-compiled functions, the annotation captures the
        time for the entire JIT execution, not individual operations within it.
        For best results, use @track on functions that are called from Python
        level, including functions that internally use @jax.jit.
    """

    def decorator(func: F) -> F:
        scope_name = name if name is not None else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with TraceAnnotation(f"{PERFORMAX_PREFIX}{scope_name}"):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)

    return decorator
