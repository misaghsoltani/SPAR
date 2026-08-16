"""Utility decorators for the SPAR framework."""

from __future__ import annotations

from functools import partial
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Generic, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from inspect import Signature

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T", bound=type)


class OptionalAbstractMethod(Generic[P, R]):
    """Require a subclass implementation before binding an abstract method.

    Attributes:
        method: The abstract method whose implementation is required.
    """

    def __init__(self, method: Callable[..., R]) -> None:
        """Initialize the descriptor.

        Args:
            method: The abstract method whose implementation is required.
        """
        self.method: Callable[..., R] = method

    def __get__(self, instance: T | None, owner: type[T]) -> Callable[..., R] | OptionalAbstractMethod[P, R]:
        """Bind the subclass implementation to an instance.

        Args:
            instance: The instance receiving the bound method, or ``None``
                when accessed through the class.
            owner: The class that owns the descriptor.

        Returns:
            The bound method, or the descriptor when accessed through the class.

        Raises:
            AttributeError: If the subclass does not implement the method.
        """
        if instance is None:
            return self
        cls: type[T] = owner
        method_name: str = getattr(self.method, "__name__", type(self.method).__name__)
        if method_name not in vars(cls):
            raise AttributeError(f"{method_name} is not implemented in {cls.__name__}")

        return partial(self.method, instance)


def optional_abstract_method(func: Callable[P, R]) -> OptionalAbstractMethod[P, R]:
    """Wrap an abstract method with a subclass implementation check.

    Args:
        func: The abstract method whose implementation is required.

    Returns:
        A descriptor that validates the subclass implementation on access.
    """
    return OptionalAbstractMethod(func)


def enforce_init_defaults(cls: type) -> type:
    """Require default values for each subclass constructor parameter.

    Args:
        cls: The class to wrap.

    Returns:
        A subclass that validates constructor defaults during class creation.
    """
    # Use the class's own hook when present and otherwise use the built-in hook.
    original_impl: Callable[..., None] = cls.__dict__.get("__init_subclass__", type.__init_subclass__)

    def _validate_init_defaults(subclass: type, **kwargs: type) -> None:
        """Validate a newly defined subclass."""
        original_impl(subclass, **kwargs)

        # Prefer a constructor defined directly on the subclass.
        init_func: Callable[..., None] = subclass.__dict__.get("__init__", type.__init__)
        init_signature: Signature = signature(init_func)

        for param in init_signature.parameters.values():
            if param.name == "self":
                continue
            if param.default is Parameter.empty:
                raise TypeError(
                    f"All parameters in the '__init__' method of '{subclass.__name__}' must have "
                    f"default values, but parameter '{param.name}' does not."
                )

    # Construct a wrapper without using a dynamic class statement so static
    # analyzers can verify the base tuple as ordinary class objects.
    wrapper_namespace: dict[str, Callable[..., None]] = {"__init_subclass__": _validate_init_defaults}
    wrapper: type = type(cls.__name__, (cls,), wrapper_namespace)

    # Retain the original class metadata used by introspection.
    wrapper.__name__ = cls.__name__
    wrapper.__qualname__ = cls.__qualname__
    wrapper.__module__ = cls.__module__
    return wrapper
