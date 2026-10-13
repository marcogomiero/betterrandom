import array
import contextlib
import io
import math
import os
import pickle
import random as std_random
import subprocess
import sys
import threading
import unittest
from collections import Counter

import betterrandom
from betterrandom import (
    ENTROPY_SOURCES,
    BetterRandom,
    better_choice,
    better_choices,
    better_random,
    better_random_float,
    better_random_many,
    better_randrange,
    better_sample,
    better_shuffle,
    betterRandom,
    main,
)

# Chi-square critical values at p = 1e-6 (computed with scipy). Using such a
# tiny p keeps the false failure rate of each statistical test negligible.
CHI2_P1E6 = {2: 27.63, 3: 30.66, 5: 35.89, 9: 44.81, 11: 48.87}


def chi_square(counts, categories, total):
    expected = total / categories
    return sum((counts[i] - expected) ** 2 / expected for i in range(categories))


def chi_square_expected(counts, expected):
    """Chi-square statistic for a dict of expected counts, keyed like counts."""
    return sum((counts.get(key, 0) - e) ** 2 / e for key, e in expected.items())


def scripted(chunks):
    """A read() replacement that returns the given byte chunks, in order."""
    pending = list(chunks)

    def read(n):
        chunk = pending.pop(0)
        assert len(chunk) == n, (len(chunk), n)
        return chunk

    return read


def words(*values):
    """Pack 64 bit words into bytes, in the machine order used by array."""
    return array.array("Q", values).tobytes()


def word_for(result, n):
    """Return a 64 bit word that the Lemire mapping sends to result for range n.

    The word is chosen so that it is never rejected, which keeps the scripted
    tests deterministic.
    """
    full = 1 << 64
    threshold = (full - n) % n
    word = -(-(result * full + threshold) // n)  # ceiling division
    assert (word * n) >> 64 == result
    assert ((word * n) & (full - 1)) >= threshold
    return word


def le_word(value):
    """Encode a 64 bit word the way the range reduction reads it (little endian)."""
    return value.to_bytes(8, "little")


class Feed:
    """A scripted byte source that also remembers what is left and what was read."""

    def __init__(self, chunks):
        self.pending = list(chunks)
        self.sizes = []

    def __call__(self, n):
        self.sizes.append(n)
        chunk = self.pending.pop(0)
        assert len(chunk) == n, (len(chunk), n)
        return chunk


class CountingSource:
    """The real OS source, but it records the size of every read."""

    def __init__(self):
        self.sizes = []

    def __call__(self, n):
        self.sizes.append(n)
        return os.urandom(n)


class Index:
    """Minimal object implementing __index__, like a numpy integer."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class BoundedExhaustiveTests(unittest.TestCase):
    """Prove the range reduction is exactly unbiased on small word sizes."""

    def check_exhaustive(self, n, nbytes):
        bits = 8 * nbytes
        mask = (1 << bits) - 1
        threshold = ((1 << bits) - n) % n
        counts = Counter()
        for word in range(1 << bits):
            supplied = iter([word.to_bytes(nbytes, "little")])

            def read(_k, supplied=supplied):
                return next(supplied)  # StopIteration means: needs a redraw

            try:
                value = betterrandom._bounded(n, read, nbytes)
            except StopIteration:
                # Exactly the words below the threshold must be rejected.
                self.assertLess((word * n) & mask, threshold, (n, word))
                continue
            self.assertGreaterEqual((word * n) & mask, threshold, (n, word))
            counts[value] += 1
        self.assertEqual(set(counts), set(range(n)), n)
        # Every result must be produced by exactly the same number of words.
        self.assertEqual(len(set(counts.values())), 1, n)

    def test_every_n_on_8_bit_words(self):
        for n in range(1, 257):
            self.check_exhaustive(n, 1)

    def test_selected_n_on_16_bit_words(self):
        for n in (1, 2, 3, 6, 100, 1000, 40000, 65535, 65536):
            self.check_exhaustive(n, 2)


class FakeSourceTests(unittest.TestCase):
    """Drive the real API with scripted bytes to hit the rejection paths."""

    @staticmethod
    def scripted(chunks):
        pending = list(chunks)

        def read(n):
            chunk = pending.pop(0)
            assert len(chunk) == n, (len(chunk), n)
            return chunk

        return read

    @staticmethod
    def words(*values):
        return array.array("Q", values).tobytes()

    def test_randint_redraws_a_rejected_word(self):
        # For n = 3 the threshold is 2**64 mod 3 = 1, so the word 0 is
        # rejected. The next word 2**63 maps to 1.
        rng = BetterRandom()
        rng._read = self.scripted([self.words(0), self.words(2**63)])
        self.assertEqual(rng.randint(10, 12), 11)

    def test_randints_redraws_a_rejected_word(self):
        rng = BetterRandom()
        rng._read = self.scripted(
            [self.words(0, 2**63), self.words(2**62)]  # 0 rejected, redraw 2**62
        )
        # Word 2**62 maps to 0, word 2**63 maps to 1.
        self.assertEqual(rng.randints(2, 0, 2), [0, 1])

    def test_extreme_words(self):
        rng = BetterRandom()
        rng._read = self.scripted([self.words(2**64 - 1)])
        self.assertEqual(rng.randint(0, 5), 5)
        rng._read = self.scripted([self.words(1)])
        self.assertEqual(rng.randint(0, 5), 0)


class BetterRandomTests(unittest.TestCase):
    def test_range_inclusive(self):
        seen = {better_random(1, 6) for _ in range(2000)}
        self.assertEqual(seen, {1, 2, 3, 4, 5, 6})

    def test_defaults(self):
        for _ in range(200):
            self.assertTrue(0 <= better_random() <= 100)

    def test_single_value(self):
        self.assertEqual(better_random(7, 7), 7)

    def test_negative_range(self):
        for _ in range(500):
            self.assertTrue(-10 <= better_random(-10, -5) <= -5)

    def test_word_boundaries(self):
        for high in (2**64 - 2, 2**64 - 1, 2**64, 2**64 + 1, 2**100, 2**256):
            for _ in range(50):
                value = better_random(0, high)
                self.assertTrue(0 <= value <= high)
        # The full 64 bit range really uses the high bits.
        self.assertTrue(any(better_random(0, 2**64 - 1) >= 2**63 for _ in range(200)))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            better_random(5, 1)
        with self.assertRaises(TypeError):
            better_random(1.5, 3)
        with self.assertRaises(TypeError):
            better_random(1, "6")
        with self.assertRaises(TypeError):
            better_random(True, 3)
        with self.assertRaises(TypeError):
            better_random(1, False)

    def test_accepts_index_types(self):
        for _ in range(100):
            self.assertTrue(1 <= better_random(Index(1), Index(3)) <= 3)

    def test_uniformity_chi_square(self):
        n = 20000
        counts = Counter(better_random(0, 9) for _ in range(n))
        self.assertLess(chi_square(counts, 10, n), CHI2_P1E6[9])

    def test_no_modulo_bias_on_awkward_range(self):
        n = 30000
        counts = Counter(better_random(0, 2) for _ in range(n))
        self.assertLess(chi_square(counts, 3, n), CHI2_P1E6[2])

    def test_bits_are_balanced(self):
        draws = 2000
        ones = sum(bin(better_random(0, 2**64 - 1)).count("1") for _ in range(draws))
        total = 64 * draws
        sigma = math.sqrt(total) / 2
        self.assertLess(abs(ones - total / 2), 6.5 * sigma)

    def test_float_range_and_mean(self):
        values = [better_random_float() for _ in range(20000)]
        self.assertTrue(all(0.0 <= v < 1.0 for v in values))
        # Standard deviation of the mean is 0.2887 / sqrt(20000) = 0.00204.
        self.assertLess(abs(sum(values) / len(values) - 0.5), 6.5 * 0.00204)

    def test_alias(self):
        self.assertIs(betterRandom, better_random)


class ManyTests(unittest.TestCase):
    def test_length_and_range(self):
        values = better_random_many(1000, 1, 6)
        self.assertEqual(len(values), 1000)
        self.assertEqual(set(values), {1, 2, 3, 4, 5, 6})

    def test_zero_and_errors(self):
        self.assertEqual(better_random_many(0), [])
        with self.assertRaises(ValueError):
            better_random_many(-1)
        with self.assertRaises(TypeError):
            better_random_many(1.5)
        with self.assertRaises(ValueError):
            better_random_many(3, 9, 1)

    def test_wide_range_fallback(self):
        values = better_random_many(20, 0, 2**100)
        self.assertEqual(len(values), 20)
        self.assertTrue(all(0 <= v <= 2**100 for v in values))

    def test_full_word_range(self):
        values = better_random_many(50, 0, 2**64 - 1)
        self.assertTrue(all(0 <= v < 2**64 for v in values))

    def test_uniformity_chi_square(self):
        n = 20000
        counts = Counter(better_random_many(n, 0, 9))
        self.assertLess(chi_square(counts, 10, n), CHI2_P1E6[9])


class ChoiceShuffleTests(unittest.TestCase):
    def test_choice(self):
        items = ["a", "b", "c"]
        seen = {better_choice(items) for _ in range(500)}
        self.assertEqual(seen, set(items))
        with self.assertRaises(IndexError):
            better_choice([])

    def test_shuffle_is_permutation(self):
        items = list(range(50))
        shuffled = items[:]
        better_shuffle(shuffled)
        self.assertEqual(sorted(shuffled), items)
        self.assertNotEqual(shuffled, items)

    def test_shuffle_trivial_inputs(self):
        for items in ([], [1]):
            copy = items[:]
            better_shuffle(copy)
            self.assertEqual(copy, items)

    def test_shuffle_first_position_uniform(self):
        n = 20000
        counts = Counter()
        for _ in range(n):
            items = [0, 1, 2, 3]
            better_shuffle(items)
            counts[items[0]] += 1
        self.assertLess(chi_square(counts, 4, n), CHI2_P1E6[3])


class RangeLenTests(unittest.TestCase):
    def test_matches_len_of_range_on_a_grid(self):
        for start in range(-12, 13):
            for stop in range(-12, 13):
                for step in list(range(-7, 0)) + list(range(1, 8)):
                    self.assertEqual(
                        betterrandom._range_len(start, stop, step),
                        len(range(start, stop, step)),
                        (start, stop, step),
                    )

    def test_zero_step(self):
        with self.assertRaises(ValueError):
            betterrandom._range_len(0, 10, 0)

    def test_beyond_sys_maxsize(self):
        self.assertEqual(betterrandom._range_len(0, 2**100, 2**90), 2**10)


class RandrangeTests(unittest.TestCase):
    def test_all_call_forms_cover_exactly_the_range(self):
        for args in [(5,), (2, 6), (10, 0, -3), (0, 10, 3), (-5, 5, 2), (7, 8)]:
            expected = set(range(*args))
            seen = {better_randrange(*args) for _ in range(800)}
            self.assertEqual(seen, expected, args)

    def test_stop_is_excluded(self):
        for _ in range(500):
            self.assertLess(better_randrange(3), 3)

    def test_empty_ranges_and_zero_step(self):
        for args in [(0,), (5, 5), (5, 0), (0, 5, -1), (-3,)]:
            with self.assertRaises(ValueError, msg=str(args)):
                better_randrange(*args)
        with self.assertRaises(ValueError):
            better_randrange(0, 10, 0)

    def test_type_errors(self):
        with self.assertRaises(TypeError):
            better_randrange(2.5)
        with self.assertRaises(TypeError):
            better_randrange(True)
        with self.assertRaises(TypeError):
            better_randrange(0, 10, 1.5)
        self.assertTrue(0 <= better_randrange(Index(5)) < 5)

    def test_huge_ranges(self):
        for _ in range(100):
            value = better_randrange(0, 2**100, 2**90)
            self.assertTrue(0 <= value < 2**100 and value % 2**90 == 0)
        for _ in range(100):
            self.assertLess(better_randrange(2**64 + 5), 2**64 + 5)

    def test_scripted_mapping(self):
        rng = BetterRandom()
        rng._read = scripted([words(word_for(2, 4))])
        self.assertEqual(rng.randrange(0, 10, 3), 6)  # the third of 0, 3, 6, 9
        rng._read = scripted([words(word_for(1, 4))])
        self.assertEqual(rng.randrange(10, 0, -3), 7)  # the second of 10, 7, 4, 1

    def test_uniformity_chi_square(self):
        n = 20000
        counts = Counter(better_randrange(10, 0, -3) for _ in range(n))
        expected = {value: n / 4 for value in (10, 7, 4, 1)}
        self.assertEqual(set(counts), set(expected))
        self.assertLess(chi_square_expected(counts, expected), CHI2_P1E6[3])


class SampleTests(unittest.TestCase):
    def test_edges(self):
        self.assertEqual(better_sample([1, 2, 3], 0), [])
        everything = better_sample(list(range(20)), 20)
        self.assertEqual(sorted(everything), list(range(20)))

    def test_errors(self):
        with self.assertRaises(ValueError):
            better_sample([1, 2, 3], 4)
        with self.assertRaises(ValueError):
            better_sample([1, 2, 3], -1)
        with self.assertRaises(TypeError):
            better_sample([1, 2, 3], 1.5)
        with self.assertRaises(TypeError):
            better_sample({1, 2, 3}, 2)

    def test_elements_are_distinct_positions_and_population_is_untouched(self):
        population = list(range(50))
        before = population[:]
        for _ in range(200):
            picked = better_sample(population, 10)
            self.assertEqual(len(set(picked)), 10)
        self.assertEqual(population, before)

    def test_duplicates_in_the_population_are_kept(self):
        self.assertEqual(better_sample([7, 7, 7], 3), [7, 7, 7])

    def test_other_sequence_types(self):
        self.assertEqual(sorted(better_sample("abcdef", 6)), list("abcdef"))
        self.assertEqual(sorted(better_sample((1, 2, 3), 3)), [1, 2, 3])
        # No copy and O(k) memory. len() of a range is limited by sys.maxsize.
        picked = better_sample(range(10**18), 5)
        self.assertEqual(len(set(picked)), 5)
        self.assertTrue(all(0 <= v < 10**18 for v in picked))

    def test_scripted_mapping(self):
        rng = BetterRandom()
        rng._read = scripted([words(word_for(2, 4)), words(word_for(1, 3))])
        # Step 0 picks position 2 (c) and moves a there. Step 1 draws from
        # the remaining a, b, d and picks index 1 of them, which is a again
        # because it now sits where c was.
        self.assertEqual(rng.sample("abcd", 2), ["c", "a"])

    def test_every_ordered_selection_is_equally_likely(self):
        n = 24000
        counts = Counter(tuple(better_sample(range(4), 2)) for _ in range(n))
        self.assertEqual(len(counts), 12)  # 4 * 3 ordered pairs
        expected = {pair: n / 12 for pair in counts}
        self.assertLess(chi_square_expected(counts, expected), CHI2_P1E6[11])


class ChoicesTests(unittest.TestCase):
    def test_zero_weights_are_never_chosen(self):
        picked = set(better_choices("abcd", [0, 5, 0, 5], k=2000))
        self.assertEqual(picked, {"b", "d"})

    def test_proportions_chi_square(self):
        n = 20000
        counts = Counter(better_choices("abcd", [1, 2, 3, 4], k=n))
        expected = {"a": n * 0.1, "b": n * 0.2, "c": n * 0.3, "d": n * 0.4}
        self.assertLess(chi_square_expected(counts, expected), CHI2_P1E6[3])

    def test_default_k_and_empty_result(self):
        self.assertEqual(len(better_choices("abc", [1, 1, 1])), 1)
        self.assertEqual(better_choices("abc", [1, 1, 1], k=0), [])

    def test_errors(self):
        with self.assertRaises(ValueError):
            better_choices("abc", [1, 2])
        with self.assertRaises(ValueError):
            better_choices("abc", [0, 0, 0])
        with self.assertRaises(ValueError):
            better_choices([], [])
        with self.assertRaises(ValueError):
            better_choices("abc", [1, -1, 1])
        with self.assertRaises(TypeError):
            better_choices("abc", [1, 0.5, 1])
        with self.assertRaises(TypeError):
            better_choices("abc", [1, True, 1])
        with self.assertRaises(ValueError):
            better_choices("abc", [1, 1, 1], k=-1)
        with self.assertRaises(TypeError):
            better_choices("abc", [1, 1, 1], k=1.5)

    def test_huge_and_index_weights(self):
        picked = set(better_choices("ab", [2**100, 2**100], k=200))
        self.assertEqual(picked, {"a", "b"})
        self.assertEqual(better_choices("ab", [Index(0), Index(3)], k=3), ["b"] * 3)

    def test_scripted_boundaries(self):
        # Weights 1, 2, 3 give the cumulative sums 1, 3, 6. The draws 0 to 5
        # must map to a, b, b, c, c, c, exactly.
        rng = BetterRandom()
        rng._read = scripted([words(*[word_for(r, 6) for r in range(6)])])
        self.assertEqual(
            rng.choices("abc", [1, 2, 3], k=6), ["a", "b", "b", "c", "c", "c"]
        )


def run_python(code):
    """Run code in a fresh interpreter that can import betterrandom."""
    here = os.path.dirname(os.path.abspath(__file__))
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=here,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# Calls that exercise every known public method of random.Random.
STD_CALLS = {
    "betavariate": lambda r: r.betavariate(2, 3),
    "binomialvariate": lambda r: r.binomialvariate(10, 0.5),
    "choice": lambda r: r.choice("abc"),
    "choices": lambda r: r.choices("abc", k=2),
    "expovariate": lambda r: r.expovariate(1.0),
    "gammavariate": lambda r: r.gammavariate(2, 1),
    "gauss": lambda r: r.gauss(0, 1),
    "getrandbits": lambda r: r.getrandbits(10),
    "lognormvariate": lambda r: r.lognormvariate(0, 1),
    "normalvariate": lambda r: r.normalvariate(0, 1),
    "paretovariate": lambda r: r.paretovariate(2),
    "randbytes": lambda r: r.randbytes(4),
    "randint": lambda r: r.randint(1, 6),
    "random": lambda r: r.random(),
    "randrange": lambda r: r.randrange(10),
    "sample": lambda r: r.sample(range(10), 3),
    "shuffle": lambda r: r.shuffle([1, 2, 3]),
    "triangular": lambda r: r.triangular(0, 1),
    "uniform": lambda r: r.uniform(1, 2),
    "vonmisesvariate": lambda r: r.vonmisesvariate(0, 1),
    "weibullvariate": lambda r: r.weibullvariate(1, 1),
}


class StdlibWrapperTests(unittest.TestCase):
    """BetterSystemRandom: the whole random.Random API on top of betterRandom."""

    def setUp(self):
        self.rng = betterrandom.BetterSystemRandom()

    def test_it_is_a_random_subclass(self):
        self.assertIsInstance(betterrandom.rng, std_random.Random)
        self.assertIsInstance(betterrandom.rng, betterrandom.BetterSystemRandom)
        self.assertEqual(betterrandom.BetterSystemRandom.__module__, "betterrandom")
        self.assertEqual(repr(self.rng), "BetterSystemRandom(paranoid=False)")
        self.assertFalse(self.rng.paranoid)

    def test_every_public_method_of_random_is_there(self):
        for name in dir(std_random.Random):
            if name.startswith("_") or not callable(getattr(std_random.Random, name)):
                continue
            with self.subTest(method=name):
                self.assertTrue(callable(getattr(self.rng, name)))

    def test_every_known_method_works(self):
        for name, call in STD_CALLS.items():
            if not hasattr(std_random.Random, name) and name != "randbytes":
                continue  # for example binomialvariate before Python 3.12
            with self.subTest(method=name):
                for _ in range(20):
                    call(self.rng)

    def test_values_have_the_expected_ranges(self):
        r = self.rng
        for _ in range(300):
            self.assertTrue(0.0 <= r.betavariate(2, 3) <= 1.0)
            self.assertGreaterEqual(r.expovariate(1.0), 0.0)
            self.assertTrue(1.0 <= r.uniform(1, 2) <= 2.0)
            self.assertTrue(0.0 <= r.triangular(0, 1, 0.5) <= 1.0)
            self.assertGreaterEqual(r.paretovariate(2), 1.0)
            self.assertTrue(0 <= r.getrandbits(10) < 1024)
            self.assertEqual(len(r.randbytes(7)), 7)
        self.assertEqual(r.randbytes(0), b"")
        self.assertEqual(r.getrandbits(0), 0)
        with self.assertRaises(ValueError):
            r.getrandbits(-1)
        with self.assertRaises(ValueError):
            r.randbytes(-1)

    def test_extra_batch_method(self):
        values = self.rng.randints(500, 1, 6)
        self.assertEqual(len(values), 500)
        self.assertEqual(set(values), {1, 2, 3, 4, 5, 6})

    def test_stdlib_semantics_are_kept(self):
        # Float weights still work here, only better_choices is integer only.
        self.assertEqual(len(self.rng.choices("ab", [0.25, 0.75], k=10)), 10)
        # random.Random.__init_subclass__ only keeps a _randbelow that is in the
        # class body, and otherwise puts its own on the class. Check whose it is
        # first: the stdlib version loops forever on a negative n, and a hang is
        # a worse failure than a red test.
        self.assertEqual(type(self.rng)._randbelow.__module__, "betterrandom")
        for empty in ([], (), ""):
            with self.assertRaises(IndexError) as caught:
                self.rng.choice(empty)
            self.assertIn("empty sequence", str(caught.exception))
        with self.assertRaises(ValueError):
            self.rng._randbelow(-1)
        with self.assertRaises(ValueError):
            self.rng.randrange(5, 5)
        with self.assertRaises(ValueError):
            self.rng.randint(5, 1)
        with self.assertRaises(ValueError):
            self.rng.sample([1, 2], 3)

    # --- The standard library functions really use betterRandom's primitives.

    def test_randint_goes_through_the_unbiased_reduction(self):
        feed = Feed([le_word(word_for(2, 6))])
        self.rng._core._read = feed
        self.assertEqual(self.rng.randint(1, 6), 3)
        self.assertEqual(feed.pending, [])
        self.assertEqual(feed.sizes, [8])

    def test_randrange_with_a_negative_step(self):
        feed = Feed([le_word(word_for(1, 4))])
        self.rng._core._read = feed
        self.assertEqual(self.rng.randrange(10, 0, -3), 7)  # 10, 7, 4, 1
        self.assertEqual(feed.pending, [])

    def test_choice_goes_through_the_unbiased_reduction(self):
        feed = Feed([le_word(word_for(2, 3))])
        self.rng._core._read = feed
        self.assertEqual(self.rng.choice("abc"), "c")
        self.assertEqual(feed.pending, [])

    def test_shuffle_goes_through_the_unbiased_reduction(self):
        # i = 3, 2, 1 draw from 4, 3 and 2 values. Pick j = 1, 0, 1.
        feed = Feed(
            [le_word(word_for(1, 4)), le_word(word_for(0, 3)), le_word(word_for(1, 2))]
        )
        self.rng._core._read = feed
        items = list("abcd")
        self.rng.shuffle(items)
        # abcd -> swap 3,1 -> adcb -> swap 2,0 -> cdab -> swap 1,1 -> cdab
        self.assertEqual(items, list("cdab"))
        self.assertEqual(feed.pending, [])

    def test_inlined_reduction_matches_the_proven_one(self):
        # _randbelow repeats the arithmetic of _bounded for speed. Feed both the
        # same words, including the ones at the rejection edge, and compare the
        # result and how many words each one consumed.
        full = 1 << 64
        sizes_n = [1, 2, 3, 5, 6, 7, 10, 100, 1000, 2**32 - 1, 2**32, 2**32 + 1]
        sizes_n += [2**63 - 1, 2**63, 2**63 + 1, 3 * 2**62, full - 3, full - 1, full]
        rejected_seen = 0
        for n in sizes_n:
            threshold = (full - n) % n
            first_words = {0, 1, 2, full - 1, full // 2, threshold, threshold + 1}
            first_words |= {max(threshold - 1, 0), min(full - 1, threshold + n)}
            first_words = {w for w in first_words if 0 <= w < full}
            fallback = word_for(0, n)  # a word that is never rejected
            for first in sorted(first_words):
                chunks = [le_word(first), le_word(fallback), le_word(fallback)]
                reference = Feed(chunks)
                expected = betterrandom._bounded(n, reference)
                feed = Feed(chunks)
                self.rng._core._read = feed
                with self.subTest(n=n, first=first):
                    self.assertEqual(self.rng._randbelow(n), expected)
                    self.assertEqual(feed.sizes, reference.sizes)
                if len(reference.sizes) > 1:
                    rejected_seen += 1
        # The rejection path must really have been exercised.
        self.assertGreater(rejected_seen, 5)

    def test_inlined_reduction_for_ranges_wider_than_64_bits(self):
        for n in (full + 1 for full in (1 << 64, 1 << 100, 1 << 200)):
            for _ in range(50):
                self.assertTrue(0 <= self.rng._randbelow(n) < n)

    def test_sample_and_big_ranges_read_whole_words(self):
        source = CountingSource()
        self.rng._core._read = source
        self.assertEqual(len(set(self.rng.sample(range(40), 6))), 6)
        self.rng.randrange(10**6)
        self.assertGreaterEqual(len(source.sizes), 7)
        # Without the hook the standard library would ask for 1 to 3 bytes.
        self.assertEqual(set(source.sizes), {8})

    def test_ranges_wider_than_64_bits(self):
        for _ in range(100):
            value = self.rng.randrange(2**200)
            self.assertTrue(0 <= value < 2**200)

    def test_getrandbits_mapping(self):
        feed = Feed([b"\xab\xcd", b"\x80", b"\xff\xff\xff"])
        self.rng._core._read = feed
        self.assertEqual(self.rng.getrandbits(12), 0xABC)  # drops the low 4 bits
        self.assertEqual(self.rng.getrandbits(1), 1)
        self.assertEqual(self.rng.getrandbits(17), 2**17 - 1)
        self.assertEqual(feed.sizes, [2, 1, 3])

    def test_random_mapping(self):
        self.rng._core._read = Feed([le_word(1 << 63), le_word(2**64 - 1), le_word(0)])
        self.assertEqual(self.rng.random(), 0.5)
        self.assertEqual(self.rng.random(), (2**53 - 1) / 2**53)
        self.assertEqual(self.rng.random(), 0.0)

    def test_randbytes_mapping(self):
        feed = Feed([b"abc"])
        self.rng._core._read = feed
        self.assertEqual(self.rng.randbytes(3), b"abc")

    # --- No seed and no state.

    def test_a_seed_is_never_silently_ignored(self):
        for value in (0, 42, "text", 1.5, b"x"):
            with self.assertRaises(NotImplementedError, msg=repr(value)):
                self.rng.seed(value)
        self.rng.seed()
        self.rng.seed(None)

    def test_no_state(self):
        with self.assertRaises(NotImplementedError):
            self.rng.getstate()
        with self.assertRaises(NotImplementedError):
            self.rng.setstate(0)

    def test_gauss_keeps_no_state_between_calls(self):
        for _ in range(200):
            self.rng.gauss(0, 1)
            self.assertIsNone(self.rng.gauss_next)

    def test_gauss_delegates_to_normalvariate(self):
        self.rng.normalvariate = lambda mu, sigma: ("called", mu, sigma)
        self.assertEqual(self.rng.gauss(2.0, 3.0), ("called", 2.0, 3.0))
        self.assertEqual(self.rng.gauss(), ("called", 0.0, 1.0))

    # --- Statistics.

    def test_randint_uniformity_chi_square(self):
        n = 20000
        counts = Counter(self.rng.randint(0, 9) for _ in range(n))
        self.assertLess(chi_square(counts, 10, n), CHI2_P1E6[9])

    def test_shuffle_first_position_uniform(self):
        n = 20000
        counts = Counter()
        for _ in range(n):
            items = [0, 1, 2, 3]
            self.rng.shuffle(items)
            counts[items[0]] += 1
        self.assertLess(chi_square(counts, 4, n), CHI2_P1E6[3])

    def test_gauss_mean_and_variance(self):
        n = 20000
        values = [self.rng.gauss(0, 1) for _ in range(n)]
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        self.assertLess(abs(mean), 6.5 / math.sqrt(n))  # std of the mean is 1/sqrt(n)
        self.assertLess(abs(variance - 1), 6.5 * math.sqrt(2 / (n - 1)))

    def test_expovariate_mean(self):
        n = 20000
        mean = sum(self.rng.expovariate(2.0) for _ in range(n)) / n
        self.assertLess(abs(mean - 0.5), 6.5 * 0.5 / math.sqrt(n))

    # --- Paranoid mode, threads and lazy creation.

    def test_paranoid_mode(self):
        rng = betterrandom.BetterSystemRandom(paranoid=True)
        self.assertTrue(rng.paranoid)
        self.assertEqual(repr(rng), "BetterSystemRandom(paranoid=True)")
        self.assertEqual(set(rng.randint(1, 3) for _ in range(100)), {1, 2, 3})
        self.assertEqual(len(rng.randbytes(10)), 10)
        self.assertTrue(0 <= rng.getrandbits(100) < 2**100)
        self.assertEqual(len(rng.sample(range(30), 5)), 5)

    def test_concurrent_use_of_the_shared_instance(self):
        errors = []

        def work():
            try:
                for _ in range(500):
                    assert 1 <= betterrandom.rng.randint(1, 6) <= 6
                    betterrandom.rng.gauss(0, 1)
                items = list(range(20))
                betterrandom.rng.shuffle(items)
                assert sorted(items) == list(range(20))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertIsNone(betterrandom.rng.gauss_next)

    def test_created_on_first_use_only(self):
        out = run_python(
            "import betterrandom as b\n"
            "before = ('rng' in vars(b), 'BetterSystemRandom' in vars(b))\n"
            "b.rng\n"
            "after = ('rng' in vars(b), 'BetterSystemRandom' in vars(b))\n"
            "print(before, after, b.rng is b.rng, isinstance(b.rng, b.BetterSystemRandom))"
        )
        self.assertEqual(out, "(False, False) (True, True) True True")

    def test_star_import_and_from_import(self):
        out = run_python(
            "from betterrandom import *\n"
            "from betterrandom import rng as other\n"
            "print(rng is other, rng.randint(4, 4), BetterSystemRandom.__name__)"
        )
        self.assertEqual(out, "True 4 BetterSystemRandom")

    def test_unknown_attribute_still_raises(self):
        with self.assertRaises(AttributeError):
            betterrandom.no_such_thing
        self.assertFalse(hasattr(betterrandom, "no_such_thing"))

    def test_racing_threads_share_one_instance(self):
        out = run_python(
            "import threading, betterrandom as b\n"
            "seen = []\n"
            "barrier = threading.Barrier(8)\n"
            "def work():\n"
            "    barrier.wait()\n"
            "    seen.append(b.rng)\n"
            "threads = [threading.Thread(target=work) for _ in range(8)]\n"
            "[t.start() for t in threads]\n"
            "[t.join() for t in threads]\n"
            "print(len({id(x) for x in seen}), len({type(x) for x in seen}))"
        )
        self.assertEqual(out, "1 1")

    @unittest.skipUnless(sys.version_info >= (3, 15), "PEP 810 needs Python 3.15")
    def test_random_module_is_not_imported_until_needed(self):
        out = run_python(
            "import sys, betterrandom as b\n"
            "b.better_random(1, 6); b.better_shuffle([1, 2, 3])\n"
            "before = 'random' in sys.modules\n"
            "b.rng.randint(1, 6)\n"
            "print(before, 'random' in sys.modules)"
        )
        self.assertEqual(out, "False True")


class ParanoidNewApiTests(unittest.TestCase):
    def test_new_methods_work_in_paranoid_mode(self):
        rng = BetterRandom(paranoid=True)
        self.assertTrue(0 <= rng.randrange(10) < 10)
        self.assertEqual(len(set(rng.sample(range(100), 10))), 10)
        self.assertEqual(len(rng.choices("abc", [1, 2, 3], k=20)), 20)

    def test_module_level_names_use_the_default_instance(self):
        self.assertIs(better_randrange.__self__, better_sample.__self__)
        self.assertIs(better_choices.__self__, better_random.__self__)
        self.assertFalse(better_randrange.__self__.paranoid)


class ParanoidTests(unittest.TestCase):
    def setUp(self):
        self.rng = BetterRandom(paranoid=True)

    def test_repr_and_flag(self):
        self.assertTrue(self.rng.paranoid)
        self.assertFalse(BetterRandom().paranoid)
        self.assertEqual(repr(self.rng), "BetterRandom(paranoid=True)")

    def test_range(self):
        seen = {self.rng.randint(1, 6) for _ in range(500)}
        self.assertEqual(seen, {1, 2, 3, 4, 5, 6})
        self.assertTrue(0.0 <= self.rng.random() < 1.0)
        self.assertEqual(len(self.rng.randints(20, 0, 9)), 20)

    def test_uniformity_chi_square(self):
        n = 5000
        counts = Counter(self.rng.randint(0, 9) for _ in range(n))
        self.assertLess(chi_square(counts, 10, n), CHI2_P1E6[9])

    def test_shuffle_and_choice(self):
        items = list(range(20))
        self.rng.shuffle(items)
        self.assertEqual(sorted(items), list(range(20)))
        self.assertIn(self.rng.choice(items), items)


class ThreadTests(unittest.TestCase):
    def test_concurrent_use(self):
        errors = []

        def work():
            try:
                for _ in range(1000):
                    assert 1 <= better_random(1, 6) <= 6
                values = better_random_many(500, 0, 9)
                assert len(values) == 500 and all(0 <= v <= 9 for v in values)
                items = list(range(30))
                better_shuffle(items)
                assert sorted(items) == list(range(30))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class CliTests(unittest.TestCase):
    def run_cli(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(main(argv), 0)
        return [int(line) for line in buf.getvalue().split()]

    def test_two_bounds(self):
        values = self.run_cli(["1", "6", "-n", "20"])
        self.assertEqual(len(values), 20)
        self.assertTrue(all(1 <= v <= 6 for v in values))

    def test_single_bound(self):
        values = self.run_cli(["5", "-n", "50"])
        self.assertTrue(all(0 <= v <= 5 for v in values))

    def test_paranoid_flag_and_zero_count(self):
        values = self.run_cli(["1", "6", "-n", "10", "--paranoid"])
        self.assertTrue(all(1 <= v <= 6 for v in values))
        self.assertEqual(self.run_cli(["1", "6", "-n", "0"]), [])


class CompatTests(unittest.TestCase):
    """Features that use Python 3.15 when available and fall back otherwise."""

    def test_choice_default_on_empty(self):
        self.assertIsNone(better_choice([], default=None))
        self.assertEqual(better_choice((), default=0), 0)
        self.assertEqual(better_choice([5], default=0), 5)

    def test_sentinel_repr_and_pickle(self):
        missing = betterrandom.MISSING
        self.assertEqual(repr(missing), "MISSING")
        self.assertIs(pickle.loads(pickle.dumps(missing)), missing)

    def test_entropy_sources_are_immutable_and_produce_bytes(self):
        with self.assertRaises(TypeError):
            ENTROPY_SOURCES["extra"] = lambda: b""
        self.assertGreaterEqual(len(ENTROPY_SOURCES), 5)
        for name, source in ENTROPY_SOURCES.items():
            self.assertIsInstance(source(), bytes, name)

    def test_entropy_blocks_differ(self):
        blocks = {betterrandom._entropy_block() for _ in range(100)}
        self.assertEqual(len(blocks), 100)

    def test_paranoid_bytes_length(self):
        for n in (0, 1, 63, 64, 65, 200):
            self.assertEqual(len(betterrandom._paranoid_bytes(n)), n)

    @unittest.skipUnless(
        hasattr(bytearray, "take_bytes"), "bytearray.take_bytes needs Python 3.15"
    )
    def test_take_bytes_used_on_new_pythons(self):
        self.assertTrue(betterrandom._HAS_TAKE_BYTES)
        buf = bytearray(b"abcdef")
        self.assertEqual(buf.take_bytes(3), b"abc")

    @unittest.skipUnless(sys.version_info >= (3, 15), "PEP 810 needs Python 3.15")
    def test_lazy_imports_stay_lazy_on_the_default_path(self):
        here = os.path.dirname(os.path.abspath(__file__))
        code = (
            "import sys, betterrandom as b; b.better_random(1, 6); "
            "b.better_random_many(5, 1, 6); "
            "print([m for m in ('argparse', 'hashlib', 'secrets', 'threading') "
            "if m in sys.modules])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # Only the array module is needed by randints, and it is not in the list.
        self.assertEqual(out, "[]")

    @unittest.skipUnless(sys.version_info >= (3, 15), "PEP 814 needs Python 3.15")
    def test_frozendict_on_new_pythons(self):
        self.assertIsInstance(ENTROPY_SOURCES, frozendict)  # noqa: F821


if __name__ == "__main__":
    unittest.main()