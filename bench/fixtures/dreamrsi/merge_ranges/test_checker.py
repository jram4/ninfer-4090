import importlib.util
import tempfile
import unittest
from pathlib import Path


CHECKER_PATH = Path(__file__).with_name("checker.py")
SPEC = importlib.util.spec_from_file_location("merge_ranges_fixture_checker", CHECKER_PATH)
checker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(checker)


CORRECT = '''\
def merge_ranges(ranges):
    ordered = sorted((start, end) for start, end in ranges)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
'''

FAULTY = {
    "does-not-merge-adjacency": '''\
def merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
''',
    "does-not-sort": '''\
def merge_ranges(ranges):
    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
''',
    "fills-gaps": '''\
def merge_ranges(ranges):
    if not ranges:
        return []
    ordered = sorted(ranges)
    return [(ordered[0][0], max(end for _, end in ordered))]
''',
    "mutates-input": '''\
def merge_ranges(ranges):
    ranges.sort()
    return list(ranges)
''',
    "negative-endpoint-bug": '''\
def merge_ranges(ranges):
    ordered = sorted(ranges)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 1 and start >= 0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
''',
    "nested-interval-end-bug": '''\
def merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged
''',
}

ALTERNATING = '''\
_calls = 0

def merge_ranges(ranges):
    global _calls
    _calls += 1
    if _calls % 2 == 0:
        return []
    ordered = sorted((start, end) for start, end in ranges)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
'''


class CheckerSelfTests(unittest.TestCase):
    def _write_and_check(self, source):
        with tempfile.TemporaryDirectory() as temp:
            candidate = Path(temp) / "merge_ranges.py"
            candidate.write_text(source, encoding="utf-8")
            return checker.check_candidate(candidate)

    def test_accepts_correct_implementation(self):
        report = self._write_and_check(CORRECT)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["checked"], 3625)

    def test_rejects_faults_for_key_interval_semantics(self):
        for label, source in FAULTY.items():
            with self.subTest(label=label):
                report = self._write_and_check(source)
                self.assertFalse(report["ok"], report)
                self.assertGreater(report["failure_count"], 0)

    def test_rejects_alternating_correct_and_fresh_wrong_results(self):
        report = self._write_and_check(ALTERNATING)
        self.assertFalse(report["ok"], report)
        self.assertIn("second invocation: got [], expected [(4, 4)]", "\n".join(report["failures"]))


if __name__ == "__main__":
    unittest.main()
