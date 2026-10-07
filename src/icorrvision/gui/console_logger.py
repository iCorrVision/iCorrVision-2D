import logging
from PySide6.QtCore import QObject, Signal

from icorrvision.gui.widgets.console_view import ConsoleLogView

from icorrvision.gui.widgets.utils.theme import THEME

_LEVEL_COLORS = {
    logging.DEBUG: THEME.log.debug,
    logging.INFO: THEME.log.info,
    logging.WARNING: THEME.log.warning,
    logging.ERROR: THEME.log.error,
    logging.CRITICAL: THEME.log.critical,
}


class _LogSignalEmitter(QObject):
    """Minimal QObject that carries a signal for log records.

    Bridges python logging system to Qts's signal-slot. Separate from
    _QtLoggingHandler because logging.Handler and QObject have different
    metaclasses (`Shiboken` for QObject and `type` for logging.Handler).
    Prevents reliable mutiple inheritance.
    """

    log_record = Signal(logging.LogRecord)


class _QtLoggingHandler(logging.Handler):
    """Logging handler that forwads records to a Qt compatible callback.

    _QtLoggingHandler extends logging.Handler, gets called by the logging
    system each time a record arrives, and forwards it via a callback to
    Qt's signal system so it can be safely delivered to the main thread.
    """

    def __init__(self, callback) -> None:
        """Initialises the handler with a record callback.

        Args:
            callback: A callable that accepts a single logging.LogRecord.
                Typically the emit method of a _LogSignalEmitter signal.
        """
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        """Forwards an incomming log to the registered callback.

        Called by the logging system on every log recorded that passes the
        handler's level filter. Can be called from any thread.

        Args:
            record: incoming log output.
        """
        self._callback(record)


class ConsoleLogger:
    """Connects Python's logging system to the ConsoleLogView.

    Installs a handler on the root logger at construction time so that every
    log feeds into the UI console without any per module wiring. formats records
    as HTML to make them pretty and/or menacing :).

    Attributes:
     _view: ConsoleLogView instance to drive and deliver to.
     _emitter: QObject that carries the log_record signal.
     _handler: Logging handler to install on the root logger.
    """

    def __init__(self, view: ConsoleLogView) -> None:
        """Initialises the presenter and installs the root log handler.

        Args:
            view: ConsoleLogView instance that will display the logs.
        """
        self._view = view

        self._emitter = _LogSignalEmitter()
        self._emitter.log_record.connect(self._on_record)

        self._handler = _QtLoggingHandler(self._emitter.log_record.emit)
        self._handler.setLevel(logging.DEBUG)

        root = logging.getLogger()
        root.addHandler(self._handler)
        root.setLevel(logging.DEBUG)

        self._view.log_level_signal.connect(self._on_log_level_changed)

    def _on_record(self, record: logging.LogRecord) -> None:
        """Formats a log record and sends it to the view.

            Always called on the main thread via Qt's queued connection.
        Formats the record as an HTML span with a timestamp, level tag,
        and per-level colour, then passes it to the view for display.

        Args:
            record: The log record to format and display.
        """
        color = _LEVEL_COLORS.get(record.levelno, "#c5c9d6")
        time = logging.Formatter().formatTime(record, datefmt="%H:%M:%S")
        self._view.append_message(
            f'<span style="color:{color}">[{time}] [{record.levelname}] [{record.filename}] [{record.lineno}] {record.getMessage()}</span>'
        )

    def _on_log_level_changed(self, level_name: str) -> None:
        """Updates the logging level based on the user's selection.

        Falls back to INFO if the level name is unrecognized.

        Args:
            level_name: The logging level as a string (e.g., "DEBUG", "INFO").
        """
        level = getattr(logging, level_name, logging.INFO)
        self._handler.setLevel(level)
        logging.debug(f"Console log level changed to {level_name}")
