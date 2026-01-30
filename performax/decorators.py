from functools import wraps
from typing import Callable, TypeVar, overload

from jax.lax import optimization_barrier
from jax.profiler import TraceAnnotation

F = TypeVar("F", bound=Callable)

PERFORMAX_PREFIX = "performax/"

_barrier_enabled: bool = False


def enable_barriers() -> None:
    global _barrier_enabled
    _barrier_enabled = True


def disable_barriers() -> None:
    global _barrier_enabled
    _barrier_enabled = False


def barriers_enabled() -> bool:
    return _barrier_enabled


@overload
def track(fn: F) -> F: ...


@overload
def track(*, name: str | None = None, barrier: bool | None = None) -> Callable[[F], F]: ...


def track(
    fn: F | None = None, *, name: str | None = None, barrier: bool | None = None
) -> F | Callable[[F], F]:
    def decorator(func: F) -> F:
        scope_name = name if name is not None else func.__name__
        use_barrier = barrier if barrier is not None else _barrier_enabled

        @wraps(func)
        def wrapper(*args, **kwargs):
            with TraceAnnotation(f"{PERFORMAX_PREFIX}{scope_name}"):
                result = func(*args, **kwargs)
                if use_barrier:
                    result = optimization_barrier(result)
                return result

        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorator(fn)

    return decorator
