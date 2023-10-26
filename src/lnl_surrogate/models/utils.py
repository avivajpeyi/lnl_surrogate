import signal

from ..logger import logger


def fmt_val_upper_lower(val: float, up: float, low: float) -> str:
    return f"{val}^{{+{up}}}_{{-{low}}}"

def signal_wrapper(method):
    """
    Decorator to wrap a method of a class to set system signals before running
    and reset them after.

    Parameters
    ==========
    method: callable
        The method to call, this assumes the first argument is `self`
        and that `self` has a `write_current_state_and_exit` method.

    Returns
    =======
    output: callable
        The wrapped method.
    """

    def wrapped(self, *args, **kwargs):
        try:
            old_term = signal.signal(signal.SIGTERM, self.write_current_state_and_exit)
            old_int = signal.signal(signal.SIGINT, self.write_current_state_and_exit)
            old_alarm = signal.signal(signal.SIGALRM, self.write_current_state_and_exit)
            _set = True
        except (AttributeError, ValueError):
            _set = False
            logger.debug(
                "Setting signal attributes unavailable on this system. "
                "This is likely the case if you are running on a Windows machine "
                "and can be safely ignored."
            )
        output = method(self, *args, **kwargs)
        if _set:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGALRM, old_alarm)
        return output

    return wrapped

