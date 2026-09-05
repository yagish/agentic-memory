import unittest

from memory import logger


class TestLoggerCompatibility(unittest.TestCase):
    def test_noop_loggers_do_not_raise(self):
        logger.activity_log("test", "action", value=1)
        logger.error_log("test", "something failed")


if __name__ == "__main__":
    unittest.main()
