import copy
import unittest
from live_probe.intervals import merge_intervals


class IntervalProbe(unittest.TestCase):
    def test_closed_interval_cases_and_no_mutation(self):
        cases = [([], []), ([(1, 2)], [(1, 2)]),
                 ([(5, 7), (1, 3)], [(1, 3), (5, 7)]),
                 ([(1, 3), (3, 5)], [(1, 5)]),
                 ([(1, 9), (2, 3), (4, 8)], [(1, 9)]),
                 ([(2, 2), (2, 2)], [(2, 2)]),
                 ([(-5, -2), (-3, 0), (4, 4)], [(-5, 0), (4, 4)]),
                 ([(8, 10), (1, 4), (4, 8)], [(1, 10)])]
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                original = copy.deepcopy(inputs)
                self.assertEqual(merge_intervals(inputs), expected)
                self.assertEqual(inputs, original)

    def test_reversed_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            merge_intervals([(3, 1)])


if __name__ == "__main__":
    unittest.main()
