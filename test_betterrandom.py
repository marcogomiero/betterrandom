import array
import contextlib
import io
import math
import os
import pickle
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
    better_random,
    better_random_float,
    better_random_many,
    better_shuffle,
    betterRandom,
    main,
)

# Chi-square critical values at p = 1e-6 (computed with scipy). Using such a
# tiny p keeps the false failure rate of each statistical test negligible.
CHI2_P1E6 = {2: 27.63, 3: 30.66, 9: 44.81}


def chi_square(counts, categories, total):
    expected = total / categories
    return sum((counts[i] - expected) ** 2 / expected for i in range(categories))


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