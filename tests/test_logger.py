import os
import tempfile
import unittest
from unittest.mock import patch

from memory import logger


class TestErrorLogTimestamps(unittest.TestCase):
    def test_error_log_prefixes_each_traceback_line_with_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            error_log_path = os.path.join(tmp, "error.log")
            with patch.object(logger, "_MEMORY_DIR", tmp), \
                 patch.object(logger, "ERROR_LOG", error_log_path):
                try:
                    raise ValueError("boom")
                except ValueError as exc:
                    logger.error_log("test", "something failed", exc=exc)

            with open(error_log_path) as f:
                lines = f.read().splitlines()

        self.assertGreaterEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("20"))
        self.assertIn("[test      ]  ERROR", lines[0])
        self.assertIn("something failed", lines[0])
        for line in lines[1:]:
            self.assertTrue(line.startswith("20"), msg=line)
            self.assertIn("[test      ]  ERROR", line)
            self.assertIn("|", line)


if __name__ == "__main__":
    unittest.main()
