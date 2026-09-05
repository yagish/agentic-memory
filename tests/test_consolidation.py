import unittest

from memory.consolidation import consolidate_old_sessions, prune_old_transcripts


class TestRemovedConsolidation(unittest.TestCase):
    def test_consolidation_removed(self):
        with self.assertRaises(RuntimeError):
            consolidate_old_sessions()

    def test_pruning_removed(self):
        with self.assertRaises(RuntimeError):
            prune_old_transcripts()


if __name__ == "__main__":
    unittest.main()
