import unittest

import numpy as np

from scripts.build_stage43 import holm_adjust, repeated_measures_anova


class Stage43Tests(unittest.TestCase):
    def test_holm_is_monotone_in_sorted_order(self):
        raw = [0.04, 0.001, 0.02]
        adjusted = holm_adjust(raw)
        self.assertEqual(len(adjusted), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        ordered = sorted(zip(raw, adjusted))
        self.assertLessEqual(ordered[0][1], ordered[1][1])
        self.assertLessEqual(ordered[1][1], ordered[2][1])

    def test_repeated_measures_anova_detects_large_model_effect(self):
        values = np.array(
            [
                [0.01, 0.20, 0.40],
                [0.02, 0.23, 0.39],
                [0.03, 0.19, 0.44],
                [0.04, 0.25, 0.41],
                [0.02, 0.21, 0.43],
                [0.01, 0.24, 0.38],
            ]
        )
        result = repeated_measures_anova(values)
        self.assertLess(result["corrected_p_value"], 0.05)
        self.assertGreater(result["statistic"], 1.0)


if __name__ == "__main__":
    unittest.main()
