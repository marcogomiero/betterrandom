# betterRandom

[![CI](https://github.com/marcogomiero/betterRandom/actions/workflows/ci.yml/badge.svg)](https://github.com/marcogomiero/betterRandom/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.8%20to%203.15-blue)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)

**Unbiased random integers from the OS CSPRNG. Fast, stateless, one file.**

```python
from betterrandom import better_random, better_random_many

better_random(1, 6)             # a die roll, both ends included
better_random_many(1000, 1, 6)  # a thousand of them, several times faster than secrets
```

## Why this exists

It started as a half-joking question: what is the most random way to pick a
number in Python?

Chasing the answer taught a useful lesson. Nothing a library can do makes
numbers more random than the operating system CSPRNG, and Python already
exposes it through `secrets`. The first version of this project mixed extra
entropy into every call, and it was 25 to 65 times slower than the standard
library for no measurable gain. That detour shaped everything that followed:
the project stopped chasing "more random" and started chasing "right".

So betterRandom has a more honest purpose:

1. **Be correct, and prove it.** A lot of hand-rolled code does
   `os.urandom(n) % k`, which is quietly biased. Here the range reduction is
   Lemire's method and the test suite proves exhaustively that it is fair.
2. **Be small enough to audit.** One file, no dependencies, readable in a
   single sitting. Random number code is the kind of code you want to be able
   to read before you trust it.
3. **Fill the gaps next to the standard library.** A batch API for secure
   integers, a safe default for `choice` on empty input, and a stateless design
   that is safe with `fork()` and threads.
4. **Show how to adopt new Python features without breaking old ones.** The
   module uses Python 3.15 features when they exist (lazy imports,
   `frozendict`, `sentinel`, `bytearray.take_bytes()`) and still runs on 3.8.
   The section on versions below explains how, and where the limits are.

### Who it is for

- Developers who need unbiased integers from a secure source: raffles and
  draws, sampling, shuffling, test data for security tools.
- Anyone who wants a small, readable reference implementation of fair
  random integers in Python.

### Who it is not for

- Reproducible simulations. Use `random` with a seed.
- Keys, tokens and passwords. Use `secrets.token_bytes` and friends.
- Anything that needs a certified or audited generator. This project has had
  no external review.

## What "better" means here

No library can make random numbers more random than the operating system
CSPRNG, which is the best source Python can reach. So betterRandom does not
try to. It tries to get everything else right:

- **Exactly unbiased.** Range reduction uses Lemire's method, and the test
  suite proves it exhaustively on small word sizes: every result is produced
  by exactly the same number of input words, and exactly the words that would
  skew the result are rejected. No `% n` bias.
- **Secure by default.** Every number comes straight from `os.urandom`. There
  is no seed and nothing to guess from past outputs.
- **Stateless.** No buffered output means nothing to leak from a memory dump,
  nothing duplicated by `fork()`, and no lock to share between threads. It
  also runs on the free-threaded 3.15 build.
- **Fast where it counts.** Within about 30% of `secrets.randbelow` for a
  single number (a bit slower or a bit faster depending on the machine), and
  3 to 8 times faster for batches, because `better_random_many` gets all its
  random words from a single OS call.
- **No dependencies, one file.** Python 3.8 and newer.

## Install

```
pip install git+https://github.com/marcogomiero/betterRandom.git
```

or just copy `betterrandom.py` into your project. It is a single file.

## Usage

```python
from betterrandom import (
    better_random, better_random_many, better_randrange, better_random_float,
    better_choice, better_choices, better_sample, better_shuffle,
)

better_random(1, 6)                 # integer in [1, 6]
better_random()                     # integer in [0, 100]
better_random(-10, 10)              # negative ranges work
better_random(0, 2**256)            # so do huge ones
better_random_many(1000, 1, 6)      # list of 1000 integers
better_randrange(0, 10, 3)          # one of 0, 3, 6, 9 (stop excluded, like range)
better_random_float()               # float in [0.0, 1.0) with 53 bits

better_choice(["red", "green", "blue"])
better_choice([], default=None)     # None instead of IndexError

# Weighted choice with integer weights: exactly proportional, no floats.
better_choices(["common", "rare"], [99, 1], k=5)

# Distinct elements without replacement, in O(k) memory.
better_sample(range(10**18), 3)

deck = list(range(52))
better_shuffle(deck)                # in place, Fisher-Yates, unbiased
```

`better_choices` takes integer weights on purpose. `random.choices` works on
floating point weights and cannot be exactly proportional. Here the draw is an
exact uniform integer below the sum of the weights, so each element has
probability exactly `weight / sum(weights)`. Scale float weights to integers
first, for example `[round(w * 1000) for w in weights]`.

`betterRandom` is available as a CamelCase alias of `better_random`.

### The whole `random` API

If you want everything the standard `random` module has (`gauss`, `uniform`,
`expovariate`, `betavariate`, `binomialvariate` and the rest) with betterRandom
underneath, use `rng`:

```python
from betterrandom import rng

rng.randint(1, 6)
rng.gauss(0, 1)
rng.uniform(1.5, 2.5)
rng.expovariate(0.5)
rng.sample(range(100), 5)
rng.shuffle(deck)
rng.randints(1000, 1, 6)            # batch, from a single OS call
```

`rng` is an instance of `BetterSystemRandom`, a subclass of `random.Random`,
the same idea as `random.SystemRandom`. Only the primitives are replaced
(`random`, `getrandbits`, `randbytes` and the internal range reduction), so
every method of `random.Random` exists and no algorithm is reimplemented. It
also works as a drop in for code that calls `random.something()`:

```python
from betterrandom import rng as random   # random.randint(1, 6) now uses betterRandom
```

What you should know about it:

- **The integer functions are exactly uniform.** `randint`, `randrange`,
  `choice`, `shuffle` and `sample` all go through the same unbiased range
  reduction as the rest of the module, on every Python from 3.8 to 3.15. The
  tests check this with scripted byte sources, so it is not an assumption.
- **The continuous distributions are the standard library's.** `gauss`,
  `uniform`, `betavariate` and the others are the standard algorithms running
  on 53 bit uniform floats. They are as good as the standard library's and not
  better, and nothing here claims more.
- **`rng.choices` is the standard float version.** For exactly proportional
  choice use `better_choices` with integer weights.
- **There is no seed.** `rng.seed(42)` raises `NotImplementedError` instead of
  being silently ignored, and `getstate()` and `setstate()` raise too.
  `rng.seed()` with no argument is accepted.
- **`gauss` is stateless, and slower because of it.** The standard `gauss`
  keeps a second value between calls, which is state and is not thread safe.
  Here `gauss` calls `normalvariate`, which is roughly 2.3 to 2.7 times slower
  than `SystemRandom.gauss` in `bench.py`. If you do not need that guarantee,
  `random.SystemRandom` is the faster choice for `gauss`.
- **Single calls are a little slower than `SystemRandom`.** Both do the same
  OS call, and betterRandom adds a few Python level calls on top of it. In
  `bench.py` on a Windows desktop that is between equal and about 10% slower
  (`randint` 0.47 us against 0.44 us, `choice` 0.45 against 0.43, `shuffle`
  equal, `random()` 0.34 against 0.31). The advantage is `randints`, which is 3.5 to 9 times faster
  than a loop of `SystemRandom.randint`, depending on the machine.
- **It is created on first use.** `import betterrandom` does not import the
  `random` module. On Python 3.15 it stays unloaded until you touch `rng`. On
  earlier versions `secrets` already imports it.

`BetterSystemRandom(paranoid=True)` gives the instance that mixes extra
entropy into every read, see below.

From the command line:

```
$ betterrandom 1 6
4
$ python -m betterrandom 100 -n 5      # five integers in [0, 100]
$ betterrandom 1 6 --paranoid          # see below
```

### Paranoid mode

```python
from betterrandom import BetterRandom, BetterSystemRandom

paranoid = BetterRandom(paranoid=True)
paranoid.randint(1, 6)

# The same mode for the whole random API:
BetterSystemRandom(paranoid=True).gauss(0, 1)
```

In paranoid mode every read mixes extra sources (nanosecond clocks, pid,
thread id, a counter, a second CSPRNG read) into a SHA3-512 hash, then XORs
the result with a fresh OS read. The sources are listed in the read-only
`betterrandom.ENTROPY_SOURCES` mapping.

Be clear about what this is: a hedge against a hypothetical flawed OS
generator, not an improvement on a working one. It can never be weaker than
the OS generator alone, and it is roughly 10 to 15 times slower, so it is off
by default.

## How it works

1. Bytes come from `os.urandom` (`getrandom`, `BCryptGenRandom`, `getentropy`).
2. A 64 bit word is multiplied by the size of the range and the high half is
   the result. The few words that would favour some results are discarded.
   This needs one OS call in practice and no division in the common case.
3. Ranges wider than 64 bits use plain rejection sampling.
4. Shuffling is Fisher-Yates on top of the same unbiased primitive.

## Why not just use `%`?

The classic shortcut `os.urandom(n) % k` is biased whenever `k` does not
divide the number of possible words. `python bias_demo.py` shows it with exact
counts, nothing sampled. For a range of 3 values and one random byte, 256
words cannot be split evenly:

```
Range of 3 values, one random byte (256 equally likely words)
  result   words with `% n`   words with betterRandom
       0                 86                        85
       1                 85                        85
       2                 85                        85
  betterRandom rejects 1 of 256 words and draws again
```

With 64 bit words the same flaw becomes large for large ranges. For a range of
3 * 2**62 values, `% n` lands in the lowest third of the range exactly 50% of
the time instead of 33.33%. betterRandom rejects the 25% of words that would
cause that, and every result is produced by exactly one of the others.

To be fair about it: `% 6` on a 64 bit word is off by only about 3e-19, which
is harmless in practice. The bias matters for small words, for ranges close to
the word size, and whenever you want fairness to be something you can prove
instead of something you hope for.

## Python versions and 3.15 features

The minimum is Python 3.8. Newer interpreters get extra behavior, detected at
runtime, so nothing breaks on older ones.

| Feature | Where it is used | Before 3.15 |
|---|---|---|
| PEP 810 lazy imports, via `__lazy_modules__` | `argparse`, `array`, `hashlib`, `secrets`, `threading` and `time` load on first use, so `import betterrandom` is nearly free and the default path never loads `hashlib` | the list is ignored, normal imports |
| PEP 814 `frozendict` | `ENTROPY_SOURCES` is a real immutable, hashable mapping | `types.MappingProxyType` |
| PEP 661 `sentinel` | `betterrandom.MISSING`, the default marker of `better_choice` | a small equivalent class |
| `bytearray.take_bytes()` | paranoid mode hands out bytes without an extra copy | `bytes(buf[:n])` |
| PEP 799 sampling profiler | profile the benchmark with `python -m profiling.sampling run bench.py` | `python -m cProfile bench.py` |

A note on lazy imports: the `lazy import x` syntax from PEP 810 is a
`SyntaxError` before 3.15, so it cannot live in a file that must still import
on 3.8. `__lazy_modules__` is the compatible form that the PEP provides for
exactly this case. The same reason keeps the 3.15 unpacking in comprehensions
(`[*xs for xs in xss]`) out of this module.

## How it compares

| | `random.randint` | `secrets.randbelow` | `better_random` |
|---|---|---|---|
| Source | Mersenne Twister (PRNG) | OS CSPRNG | OS CSPRNG |
| Predictable from past outputs | yes | no | no |
| Seedable and reproducible | yes | no | no |
| Unbiased range reduction | yes | yes | yes, proven exhaustively in the tests |
| Stateless, fork and thread safe | no | yes | yes |
| Batch API | no | no | `better_random_many` |
| Weighted choice | float weights, not exactly proportional | no | `better_choices`, exact integer weights |
| Sampling without replacement | `random.sample` | no | `better_sample`, secure, O(k) memory |
| Ranges beyond 64 bits | yes | yes | yes |
| The whole `random.Random` API | yes | no | yes, as `rng`, secure and unseedable |
| Single number, die roll | fastest | baseline | within about 30% of `secrets` |
| 10 000 numbers in one go | fast | slow | 3 to 8 times faster than `secrets`, 1.2 to 3 times faster than `random` |

Run `python bench.py` to measure on your own machine. Absolute numbers vary a
lot between machines and Python versions, the ratios are what matter. On a
Windows desktop with a recent Python the batch case came out at 0.12 us per
number for `better_random_many`, against 0.41 us for `secrets.randbelow` and
0.16 us for `random.randint`. On the same machine `BetterSystemRandom.randint`
took 0.56 us against 0.44 us for `SystemRandom.randint`, and `randints` 0.12 us
against 0.43 us for a loop of `SystemRandom.randint`.

## Honest notes

- The quality of the randomness is the quality of your operating system
  generator. betterRandom adds correctness, safety and speed around it, not
  extra randomness.
- `random.randint` is still faster for a single number, because it is not a
  cryptographic generator. Use it for simulations that need a seed.
- Do not use any of this to invent your own cryptography. For keys and tokens
  use `secrets.token_bytes` and friends.

## Tests

```
python -m unittest -v
```

99 tests, including an exhaustive proof of unbiasedness on 8 and 16 bit
words, scripted byte sources that force the rejection paths and check the
exact mapping of `better_sample`, `better_choices` and of the `random.Random`
methods that `rng` inherits, chi-square and bit balance checks, thread tests,
and checks for the CLI and the 3.15 features. The statistical tests use a
significance level of 1e-6, so a green CI is not luck. Tests that need 3.15 are
skipped on older versions. The suite is run on every Python from 3.8 to 3.15,
including the free-threaded 3.15 build and `-X lazy_imports=all`. Every minor
version is in the matrix on purpose: `random.Random` handles an empty
`choice` differently in 3.8, 3.9 and 3.10, and only a real run on each of them
catches that.

The tests were also checked against deliberately injected bugs (off by one in
the range length, wrong swap bookkeeping in `better_sample`, `bisect_left`
instead of `bisect_right` in `better_choices`, a removed rejection threshold,
a lost `_randbelow` hook, a stateful `gauss`, a silently ignored seed, and
others). Every one of them made the suite fail.

## Contributing

Issues and pull requests are welcome. Please keep the module dependency free,
keep it importable on Python 3.8, and add a test with every change.

## License

MIT