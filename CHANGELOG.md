# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## 0.5.1 (2026-10-13)

### Changed

- `BetterSystemRandom`: faster single calls. The range reduction used by
  `randint`, `randrange`, `choice`, `shuffle` and `sample` is now written out
  inside `_randbelow`, which saves two function calls per draw. A test compares
  it with the proven reduction on the edge cases, so the two cannot drift apart.

## 0.5.0 (first public release)

### Added

- `better_random(low, high)`, `better_random_many(count, low, high)`,
  `better_random_float()`, `better_choice(seq, default=...)` and
  `better_shuffle(seq)`, plus the CamelCase alias `betterRandom`.
- `better_randrange(start, stop, step)`, with the semantics of `range`.
- `better_sample(population, k)`: distinct elements, O(k) memory, no copy of
  the population.
- `better_choices(population, weights, k)`: exact proportional choice with
  integer weights.
- `BetterRandom` class, and its optional paranoid mode that mixes extra
  entropy sources into every read.
- `BetterSystemRandom` and the ready made instance `rng`: the whole
  `random.Random` API (`gauss`, `uniform`, `expovariate`, `binomialvariate` and
  the rest) on top of betterRandom, as a `random.Random` subclass. Only the
  primitives are replaced, so no standard library algorithm is reimplemented,
  and `randint`, `randrange`, `choice`, `shuffle` and `sample` use the unbiased
  range reduction on every Python from 3.8 to 3.15. It is created on first use.
  `seed(value)`, `getstate()` and `setstate()` raise `NotImplementedError`, and
  `gauss` is stateless (it calls `normalvariate`), which makes it slower than
  `SystemRandom.gauss`.
- Command line interface: `betterrandom [LOW] HIGH [-n COUNT] [--paranoid]`.
- `bias_demo.py`, which shows the bias of `% n` with exact counts.
- Python 3.15 features with fallbacks for older versions: lazy imports through
  `__lazy_modules__`, `frozendict`, `sentinel` and `bytearray.take_bytes()`.

### Design

- All randomness comes from the OS CSPRNG. The generator is stateless.
- Range reduction uses Lemire's method on 64 bit words, with rejection
  sampling for ranges wider than 64 bits. The test suite proves exhaustively
  that it is unbiased on small word sizes.
- Supports Python 3.8 to 3.15, including the free-threaded 3.15 build.

### Development history

Versions 0.1.0 to 0.4.0 were internal iterations and were never published.
The first of them mixed extra entropy into every call, which made it 25 to 65
times slower than the standard library with no measurable gain. That mixing
is now an opt-in paranoid mode.