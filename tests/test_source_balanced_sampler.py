import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_act_source_balanced.py"
SPEC = importlib.util.spec_from_file_location("source_balanced", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SourceBalancedSamplerTest(unittest.TestCase):
    def make_sampler(self):
        original = {i: np.arange(i * 10, i * 10 + 10) for i in range(2)}
        correction = {i: np.arange(i * 4, i * 4 + 4) for i in range(3)}
        strata = {0: ("moving_away", "timeout"),
                  1: ("moving_away", "timeout"),
                  2: ("gate_stall", "success")}
        return MODULE.SourceBalancedSampler(
            original, correction, strata, seed=1000
        )

    def test_exact_source_ratio_and_bounds(self):
        sampler = self.make_sampler()
        records = sampler.stream_records(0, 1000)
        counts = {key: int(np.count_nonzero(records["sources"] == key))
                  for key in MODULE.SOURCE_NAMES}
        # Quotas are exact per deterministic dataset-length epoch.  Small toy
        # epochs may differ from the asymptotic target by one sample.
        self.assertLessEqual(abs(counts["original"] / 1000 - 0.70), 0.02)
        self.assertLessEqual(abs(counts["correction"] / 1000 - 0.30), 0.02)
        self.assertTrue(np.all(records["indices"] >= 0))
        self.assertTrue(np.all(records["indices"] < len(sampler)))

    def test_deterministic_and_stratified(self):
        first = self.make_sampler().stream_records(0, 320)
        second = self.make_sampler().stream_records(0, 320)
        np.testing.assert_array_equal(first["indices"], second["indices"])
        groups = [group for group in first["correction_strata"] if group is not None]
        counts = {group: groups.count(group) for group in set(groups)}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 2)

    def test_configurable_twenty_percent_correction(self):
        sampler = self.make_sampler()
        sampler.source_targets = MODULE._source_targets(0.20)
        records = sampler.stream_records(0, 1000)
        correction = int(np.count_nonzero(records["sources"] == "correction"))
        self.assertLessEqual(abs(correction / 1000 - 0.20), 0.02)
        self.assertEqual(sampler.source_targets, {"original": 0.8, "correction": 0.2})

    def test_invalid_correction_fraction(self):
        for value in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE._source_targets(value)


if __name__ == "__main__":
    unittest.main()
