import inspect
from functools import wraps
from typing import Any, Callable, Optional, Sequence, TypeVar, Union

from signpost.properties import ContextProperty, DataFrameTypeError, Property

T = TypeVar("T")
F = Callable[..., T]


def df_args(
    *prop_args: Union[Property, ContextProperty],
    **prop_kwargs: Union[Property, ContextProperty]
) -> Callable[[F[T]], F[T]]:
    def decorator(func: F[T]) -> F[T]:
        sig = inspect.signature(func)
        bound_props = sig.bind_partial(*prop_args, **prop_kwargs)

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for df_name, prop in bound_props.arguments.items():
                df = bound.arguments[df_name]
                result = (
                    prop.check_with_context(df, bound.arguments)
                    if isinstance(prop, ContextProperty)
                    else prop.check(df)
                )
                if result is not None:
                    raise DataFrameTypeError(result)
            return func(*args, **kwargs)

        return wrapped

    return decorator


def df_return(
    *properties: Optional[Union[Property, ContextProperty]]
) -> Callable[[F[T]], F[T]]:
    def decorator(func: F[T]) -> F[T]:
        sig = inspect.signature(func)

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            out = func(*args, **kwargs)
            # wrap the output
            out_seq = out if isinstance(out, Sequence) else (out,)
            for df, prop in zip(out_seq, properties):
                if prop is not None:
                    result = (
                        prop.check_with_context(df, bound.arguments)
                        if isinstance(prop, ContextProperty)
                        else prop.check(df)
                    )
                    if result is not None:
                        raise DataFrameTypeError(result)
            return out

        return wrapped

    return decorator
