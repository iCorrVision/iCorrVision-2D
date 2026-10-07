from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QComboBox,
)
from PySide6.QtCore import Signal


class ConsoleLogView(QWidget):
    """Read-only console showing timestamped log messages, coloured by level."""

    log_level_signal: Signal = Signal(str)
    path_to_log_signal: Signal = Signal(str)

    def __init__(self, parent=None):
        """Build the read-only text area."""
        super().__init__(parent)
        self._console_log = QTextEdit()
        self._console_log.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self._console_log)

        h_box = QHBoxLayout()
        box = QComboBox()
        box.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        box.currentIndexChanged.connect(
            lambda: self.log_level_signal.emit(box.currentText())
        )
        h_box.addWidget(box)
        h_box.addStretch()

        layout.addLayout(h_box)

    def append_message(self, message: str):
        """Append a message, usually HTML-formatted, to the console."""
        self._console_log.append(message)
