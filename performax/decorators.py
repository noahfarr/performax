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
