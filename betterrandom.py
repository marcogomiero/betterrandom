"""betterRandom: unbiased random integers from the OS CSPRNG, fast and safe.

Design:
  1. All randomness comes from the operating system CSPRNG (os.urandom,
     which is getrandom / BCryptGenRandom / getentropy underneath). Nothing
     can be more random than that, so nothing tries to be.
  2. Range reduction uses Lemire's method on a 64 bit word: exactly
     unbiased, one OS call in practice, no modulo and no retry loop in the
     common case. Ranges wider than 64 bits use plain rejection sampling.
  3. The generator is stateless: no buffered output, nothing to leak from a
     memory dump, nothing duplicated by fork(), nothing to lock between
     threads.
  4. Optional paranoid mode (BetterRandom(paranoid=True)) mixes extra
     sources into every read with SHA3-512 and XORs the result with a fresh
     OS read. It is a hedge against a hypothetical flawed OS generator and
     it can never be weaker than the OS generator alone. It is much slower.

Python support: 3.8 and newer. Newer interpreters get extra features,
detected at runtime, never at the cost of compatibility:
  - 3.15: PEP 810 lazy imports (via __lazy_modules__), PEP 814 frozendict,
          PEP 661 sentinel, bytearray.take_bytes().
"""

from __future__ import annotations

import itertools
import operator
import os

# PEP 810 (Python 3.15): modules listed here are imported lazily, on first
# use. Older interpreters simply ignore this list, so it is harmless there.
# Only plain "import x" statements are covered, hence no "from x import y".
__lazy_modules__ = ["argparse", "array", "hashlib", "secrets", "threading", "time"]

import argparse
import array
import hashlib
import secrets
import threading
import time

# Avoid importing typing at runtime: it is only needed by type checkers.
TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import (
        Callable,
        List,
        Mapping,
        MutableSequence,
        Optional,
        Sequence,
        TypeVar,
    )

    T = TypeVar("T")

__all__ = [
    "BetterRandom",
    "better_random",
    "betterRandom",
    "better_random_float",
    "better_random_many",
    "better_choice",
    "better_shuffle",
    "ENTROPY_SOURCES",
    "MISSING",
]
__version__ = "0.4.0"

_WORD_BYTES = 8
_WORD_RANGE = 1 << (8 * _WORD_BYTES)  # 2**64
_FLOAT_SCALE = 1.0 / (1 << 53)  # an exact power of two
_MASK64 = _WORD_RANGE - 1
_counter = itertools.count()


# --- Python 3.15 features with fallbacks for older versions ---------------

try:
    # PEP 661: the sentinel builtin, new in 3.15.
    MISSING = sentinel("MISSING")  # type: ignore[name-defined]  # noqa: F821
except NameError:

    class _Missing:
        """Fallback sentinel: a unique object with a readable repr."""

        __slots__ = ()

        def __repr__(self) -> str:
            return "MISSING"

        def __copy__(self):
            return self

        def __deepcopy__(self, memo):
            return self

        def __reduce__(self):
            return "MISSING"

    MISSING = _Missing()

try:
    # PEP 814: the frozendict builtin, new in 3.15.
    _freeze = frozendict  # type: ignore[name-defined]  # noqa: F821
except NameError:
    from types import MappingProxyType as _freeze

# bytearray.take_bytes() exists from 3.15: it hands out the bytes without
# copying them.
_HAS_TAKE_BYTES = hasattr(bytearray, "take_bytes")


# --- Paranoid entropy (opt-in) ---------------------------------------------


def _u64(value: int) -> bytes:
    """Encode an int as 8 bytes, truncating to 64 bits."""
    return (value & _MASK64).to_bytes(8, "little")


# Name -> callable returning bytes. Immutable, in a fixed order, and public so
# that it can be inspected. Used only by paranoid mode. frozendict on 3.15, a
# read-only view before that.
ENTROPY_SOURCES: Mapping[str, Callable[[], bytes]] = _freeze(
    {
        "os.urandom": lambda: os.urandom(64),
        "time.time_ns": lambda: _u64(time.time_ns()),
        "time.perf_counter_ns": lambda: _u64(time.perf_counter_ns()),
        "time.monotonic_ns": lambda: _u64(time.monotonic_ns()),
        "os.getpid": lambda: _u64(os.getpid()),
        "threading.get_ident": lambda: _u64(threading.get_ident()),
        "object.id": lambda: _u64(id(object())),
        "counter": lambda: _u64(next(_counter)),
        "secrets.token_bytes": lambda: secrets.token_bytes(64),
    }
)


def _entropy_block() -> bytes:
    """Return 64 bytes of mixed entropy."""
    h = hashlib.sha3_512()
    for source in ENTROPY_SOURCES.values():
        h.update(source())
    digest = h.digest()
    # XOR with a fresh OS read: never weaker than the OS CSPRNG alone.
    fresh = os.urandom(len(digest))
    return bytes(a ^ b for a, b in zip(digest, fresh))


def _paranoid_bytes(n: int) -> bytes:
    """Return n bytes built from mixed entropy blocks."""
    buf = bytearray()
    while len(buf) < n:
        buf += _entropy_block()
    if _HAS_TAKE_BYTES:
        return buf.take_bytes(n)  # no copy on 3.15
    return bytes(buf[:n])


# --- Unbiased range reduction ------------------------------------------------


def _bounded(n: int, read: Callable[[int], bytes], nbytes: int = _WORD_BYTES) -> int:
    """Return a uniform integer in [0, n) for 1 <= n <= 2**(8 * nbytes).

    Lemire's method ("Fast random integer generation in an interval", 2019):
    multiply a random word by n and keep the high half. The few words that
    would favour some results are discarded, which makes the output exactly
    uniform. nbytes is a parameter so the tests can prove this exhaustively
    on a tiny word size.
    """
    bits = 8 * nbytes
    mask = (1 << bits) - 1
    m = int.from_bytes(read(nbytes), "little") * n
    if (m & mask) < n:  # rare slow path, probability n / 2**bits
        threshold = ((1 << bits) - n) % n
        while (m & mask) < threshold:
            m = int.from_bytes(read(nbytes), "little") * n
    return m >> bits


def _bounded_wide(n: int, read: Callable[[int], bytes]) -> int:
    """Return a uniform integer in [0, n) for any n, by rejection sampling."""
    k = (n - 1).bit_length()
    nbytes = (k + 7) // 8
    shift = nbytes * 8 - k
    while True:
        r = int.from_bytes(read(nbytes), "big") >> shift
        if r < n:
            return r


def _as_int(value: object) -> int:
    """Slow path of argument checking: accept __index__ types, reject bool."""
    if isinstance(value, bool):
        raise TypeError("low and high must be integers, not bool")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError("low and high must be integers") from None


# --- The generator -----------------------------------------------------------


class BetterRandom:
    """A stateless random number generator backed by the OS CSPRNG.

    BetterRandom()                 fast default, straight from the OS
    BetterRandom(paranoid=True)    mixes extra entropy into every read
    """

    __slots__ = ("paranoid", "_read")

    def __init__(self, paranoid: bool = False) -> None:
        self.paranoid = bool(paranoid)
        self._read = _paranoid_bytes if self.paranoid else os.urandom

    def __repr__(self) -> str:
        return "BetterRandom(paranoid=%r)" % self.paranoid

    def _below(self, n: int) -> int:
        if n <= _WORD_RANGE:
            return _bounded(n, self._read)
        return _bounded_wide(n, self._read)

    def randint(self, low: int = 0, high: int = 100) -> int:
        """Return a random integer N such that low <= N <= high (inclusive)."""
        if type(low) is not int or type(high) is not int:
            low, high = _as_int(low), _as_int(high)
        if low > high:
            raise ValueError("low must be less than or equal to high")
        return low + self._below(high - low + 1)

    def randints(self, count: int, low: int = 0, high: int = 100) -> List[int]:
        """Return count random integers in [low, high], inclusive.

        Much faster than calling randint in a loop: the random words come
        from a single OS call.
        """
        count = operator.index(count)
        if count < 0:
            raise ValueError("count must not be negative")
        if type(low) is not int or type(high) is not int:
            low, high = _as_int(low), _as_int(high)
        if low > high:
            raise ValueError("low must be less than or equal to high")
        n = high - low + 1
        if count == 0:
            return []
        words = array.array("Q")
        if n > _WORD_RANGE or words.itemsize != _WORD_BYTES:
            return [low + self._below(n) for _ in range(count)]
        words.frombytes(self._read(_WORD_BYTES * count))
        read = self._read
        threshold = None
        out = []
        for word in words:
            m = word * n
            if (m & _MASK64) < n:
                if threshold is None:
                    threshold = (_WORD_RANGE - n) % n
                while (m & _MASK64) < threshold:
                    m = int.from_bytes(read(_WORD_BYTES), "little") * n
            out.append(low + (m >> 64))
        return out

    def random(self) -> float:
        """Return a random float in [0.0, 1.0) with 53 bits of precision."""
        return (int.from_bytes(self._read(_WORD_BYTES), "little") >> 11) * _FLOAT_SCALE

    def choice(self, seq: Sequence[T], default: object = MISSING) -> T:
        """Return a random element of a sequence.

        If the sequence is empty, return default when given, otherwise raise
        IndexError (same idea as next(iterator, default)).
        """
        if len(seq) == 0:
            if default is MISSING:
                raise IndexError("cannot choose from an empty sequence")
            return default  # type: ignore[return-value]
        return seq[self._below(len(seq))]

    def shuffle(self, seq: MutableSequence[T]) -> None:
        """Shuffle a mutable sequence in place (Fisher-Yates, unbiased)."""
        below = self._below
        for i in range(len(seq) - 1, 0, -1):
            j = below(i + 1)
            seq[i], seq[j] = seq[j], seq[i]


# --- Module level API --------------------------------------------------------

_DEFAULT = BetterRandom()

better_random = _DEFAULT.randint
better_random_many = _DEFAULT.randints
better_random_float = _DEFAULT.random
better_choice = _DEFAULT.choice
better_shuffle = _DEFAULT.shuffle

# CamelCase alias, as in the project name.
betterRandom = better_random


def main(argv: Optional[List[str]] = None) -> int:
    """Command line entry point: betterrandom [LOW] HIGH [-n COUNT]."""
    parser = argparse.ArgumentParser(
        prog="betterrandom",
        description="Print random integers in [LOW, HIGH], both included.",
    )
    parser.add_argument(
        "bounds",
        type=int,
        nargs="*",
        help="HIGH (LOW defaults to 0) or LOW HIGH; defaults to 0 100",
    )
    parser.add_argument("-n", "--count", type=int, default=1)
    parser.add_argument(
        "--paranoid",
        action="store_true",
        help="mix extra entropy sources into every read (much slower)",
    )
    args = parser.parse_args(argv)
    if len(args.bounds) == 0:
        low, high = 0, 100
    elif len(args.bounds) == 1:
        low, high = 0, args.bounds[0]
    elif len(args.bounds) == 2:
        low, high = args.bounds
    else:
        parser.error("expected at most two bounds")
    if args.count < 0:
        parser.error("count must not be negative")
    rng = BetterRandom(paranoid=args.paranoid)
    for value in rng.randints(args.count, low, high):
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())