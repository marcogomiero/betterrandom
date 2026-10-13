"""Tiny benchmark: betterRandom vs the standard library.

Usage: python bench.py

To see where the time goes:
  Python 3.15:  python -m profiling.sampling run bench.py
  older:        python -m cProfile -s cumtime bench.py
"""

import random
import secrets
import timeit

from betterrandom import BetterRandom, better_random, better_random_many

N = 100_000
BATCH = 10_000

paranoid = BetterRandom(paranoid=True)


def bench(label, fn, per_call=1):
    number = max(1, N // per_call)
    seconds = min(timeit.repeat(fn, number=number, repeat=3))
    per_item_us = seconds / (number * per_call) * 1e6
    print(f"{label:<44} {per_item_us:7.2f} us/number  {1e6 / per_item_us:12,.0f} numbers/s")


if __name__ == "__main__":
    print("One number per call, die roll in [1, 6]")
    bench("random.randint (Mersenne Twister, not secure)", lambda: random.randint(1, 6))
    bench("secrets.randbelow (OS CSPRNG)", lambda: secrets.randbelow(6) + 1)
    bench("better_random (OS CSPRNG)", lambda: better_random(1, 6))
    bench("BetterRandom(paranoid=True).randint", lambda: paranoid.randint(1, 6))
    print()
    print(f"Batches of {BATCH} numbers in [1, 6]")
    bench(
        "random.randint in a list comprehension (not secure)",
        lambda: [random.randint(1, 6) for _ in range(BATCH)],
        per_call=BATCH,
    )
    bench(
        "secrets.randbelow in a list comprehension",
        lambda: [secrets.randbelow(6) + 1 for _ in range(BATCH)],
        per_call=BATCH,
    )
    bench(
        "better_random_many (OS CSPRNG, one OS call)",
        lambda: better_random_many(BATCH, 1, 6),
        per_call=BATCH,
    )