"""Minimal stand-in for the loguru logger (only used when the real loguru package is not installed).

Written in ASD-STE100 Simplified Technical English. [IndoML-T2 change]: new file. The vendored DeepFilterNet
files import `from loguru import logger`; for inference they only log messages. This shim sends the
messages to the standard logging module (logger name 'df').
"""
import logging

_log = logging.getLogger('df')


class _Logger:
    def _f(level):  # noqa: N805
        def f(self, msg, *args, **kwargs):
            try:
                text = str(msg).format(*args, **kwargs) if (args or kwargs) else str(msg)
            except Exception:  # noqa: BLE001
                text = str(msg)
            _log.log(level, text)
        return f

    trace = _f(5)
    debug = _f(logging.DEBUG)
    info = _f(logging.INFO)
    success = _f(logging.INFO)
    warning = _f(logging.WARNING)
    error = _f(logging.ERROR)
    critical = _f(logging.CRITICAL)
    exception = _f(logging.ERROR)

    def log(self, level, msg, *args, **kwargs):
        lv = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
        _log.log(lv, str(msg))

    def opt(self, *args, **kwargs):
        return self

    def bind(self, *args, **kwargs):
        return self

    def add(self, *args, **kwargs):
        return 0

    def remove(self, *args, **kwargs):
        return None

    def level(self, *args, **kwargs):
        return None

    def disable(self, *args, **kwargs):
        return None

    def enable(self, *args, **kwargs):
        return None


logger = _Logger()
