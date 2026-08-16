# src/myapp/samplers.py

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Generic, TypeVar
from urllib.parse import quote, unquote

from numpy.random import default_rng

if TYPE_CHECKING:
    from typing import TypeAlias

    from numpy.random import Generator


_DEFAULT_RNG: Generator = default_rng()

SampleT_co = TypeVar("SampleT_co", covariant=True)
ChoiceT = TypeVar("ChoiceT")
SamplerScalar: TypeAlias = str | int | float | bool | None

_RANGE_PREFIX: str = "RangeSampler(low="
_INTEGER_RANGE_PREFIX: str = "IntRangeSampler(low="
_CHOICE_PREFIX: str = "ChoiceSampler(options="


@dataclass
class Sampler(Generic[SampleT_co]):
    """Base class for all samplers."""

    def sample(self, rng: Generator) -> SampleT_co:
        """Sample a value using the provided random number generator."""
        raise NotImplementedError("Subclasses must implement this method.")


@dataclass
class RangeSampler(Sampler[float]):
    """Sampler that generates random floating-point values within a specified range.

    Attributes:
        low: Lower bound of the range (inclusive).
        high: Upper bound of the range (exclusive).
    """

    low: float
    high: float

    def sample(self, rng: Generator) -> float:
        """Sample a random float value from the range [low, high).

        Args:
            rng: Random number generator instance.

        Returns:
            A random float value within the specified range.
        """
        return rng.uniform(self.low, self.high)


@dataclass
class IntRangeSampler(Sampler[int]):
    """Sampler that generates random integer values within a specified range.

    Attributes:
        low: Lower bound of the range (inclusive).
        high: Upper bound of the range (inclusive).
    """

    low: int
    high: int

    def sample(self, rng: Generator) -> int:
        """Sample a random integer value from the range [low, high].

        Args:
            rng: Random number generator instance.

        Returns:
            A random integer value within the specified range (inclusive).
        """
        # Use integers for inclusive range sampling
        return int(rng.integers(self.low, self.high + 1))


@dataclass
class ChoiceSampler(Sampler[ChoiceT], Generic[ChoiceT]):
    """Sampler that randomly selects one option from a sequence of choices.

    Attributes:
        options: Sequence of options to choose from.
    """

    options: Sequence[ChoiceT]

    def sample(self, rng: Generator) -> ChoiceT:
        """Sample a random choice from the available options.

        Args:
            rng: Random number generator instance.

        Returns:
            A randomly selected option from the sequence.
        """
        index: int = int(rng.integers(0, len(self.options)))
        return self.options[index]


def encode_range_sampler(low: float, high: float) -> str:
    """Encode a floating-point sampler as an OmegaConf-safe scalar.

    Args:
        low: Inclusive lower bound.
        high: Exclusive upper bound.

    Returns:
        A deterministic sampler specification.
    """
    return f"{_RANGE_PREFIX}{low!r}, high={high!r})"


def encode_integer_range_sampler(low: int, high: int) -> str:
    """Encode an integer sampler as an OmegaConf-safe scalar.

    Args:
        low: Inclusive lower bound.
        high: Inclusive upper bound.

    Returns:
        A deterministic sampler specification.
    """
    return f"{_INTEGER_RANGE_PREFIX}{low}, high={high})"


def _encode_choice_value(value: SamplerScalar) -> str:
    """Encode one choice value with an explicit scalar type tag."""
    if value is None:
        return "n:"
    if isinstance(value, bool):
        return f"b:{int(value)}"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        return f"f:{value!r}"
    return f"s:{quote(value, safe='')}"


def encode_choice_sampler(*options: SamplerScalar | Sequence[SamplerScalar]) -> str:
    """Encode choice values as an OmegaConf-safe scalar.

    Args:
        *options: Individual options or one non-string option sequence.

    Returns:
        A deterministic sampler specification.

    Raises:
        TypeError: If option sequences are mixed with additional arguments.
        ValueError: If no choices are provided.
    """
    normalized: list[SamplerScalar]
    sole_option: SamplerScalar | Sequence[SamplerScalar] | None = options[0] if len(options) == 1 else None
    if sole_option is not None and isinstance(sole_option, Sequence) and not isinstance(sole_option, str):
        normalized = list(sole_option)
    else:
        normalized = []
        for option in options:
            if isinstance(option, Sequence) and not isinstance(option, str):
                raise TypeError("Pass one option sequence or individual scalar options, not both")
            normalized.append(option)
    if not normalized:
        raise ValueError("ChoiceSampler requires at least one option")
    payload = "|".join(_encode_choice_value(option) for option in normalized)
    return f"{_CHOICE_PREFIX}{payload})"


def _decode_choice_value(token: str) -> SamplerScalar:
    """Decode one explicitly typed choice value."""
    tag, separator, payload = token.partition(":")
    if not separator:
        raise ValueError(f"Invalid ChoiceSampler option token: {token!r}")
    if tag == "n":
        return None
    if tag == "b":
        if payload not in {"0", "1"}:
            raise ValueError(f"Invalid boolean ChoiceSampler option: {payload!r}")
        return payload == "1"
    if tag == "i":
        return int(payload)
    if tag == "f":
        return float(payload)
    if tag == "s":
        return unquote(payload)
    raise ValueError(f"Unknown ChoiceSampler option tag: {tag!r}")


def _decode_bounds(spec: str, prefix: str) -> tuple[str, str] | None:
    """Extract low and high bound strings from a sampler specification."""
    if not spec.startswith(prefix) or not spec.endswith(")"):
        return None
    low_text, separator, high_text = spec[len(prefix) : -1].partition(", high=")
    if not separator:
        raise ValueError(f"Invalid sampler range specification: {spec!r}")
    return low_text, high_text


@cache
def sampler_from_spec(value: str) -> Sampler[SamplerScalar] | None:
    """Decode an OmegaConf-safe sampler specification.

    Args:
        value: Configuration scalar to inspect.

    Returns:
        The decoded sampler, or ``None`` when the value is not a sampler specification.
    """
    bounds: tuple[str, str] | None = _decode_bounds(value, _RANGE_PREFIX)
    if bounds is not None:
        return RangeSampler(float(bounds[0]), float(bounds[1]))
    bounds = _decode_bounds(value, _INTEGER_RANGE_PREFIX)
    if bounds is not None:
        return IntRangeSampler(int(bounds[0]), int(bounds[1]))
    if value.startswith(_CHOICE_PREFIX) and value.endswith(")"):
        payload: str = value[len(_CHOICE_PREFIX) : -1]
        if not payload:
            raise ValueError("ChoiceSampler specification has no options")
        return ChoiceSampler([_decode_choice_value(token) for token in payload.split("|")])
    return None
