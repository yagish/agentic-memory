# logger.py — compatibility no-op logger.
#
# The retained file logs are facts.log, episodic.log, and daemon.log.
# Older call sites still import activity_log/error_log, so keep these helpers
# as safe no-ops instead of creating extra log files.

def activity_log(component: str, action: str, **kwargs) -> None:
    del component, action, kwargs



def error_log(component: str, message: str, exc: BaseException | None = None) -> None:
    del component, message, exc


