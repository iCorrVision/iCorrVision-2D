from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
)

from UI.console_view import ConsoleLogView


class BottomPanelView(QWidget):
    """Bottom panel holding the console log tab."""

    def __init__(self, parent=None):
        """Build the tab widget and its console tab."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()

        self.console = ConsoleLogView()

        self._tabs.addTab(self.console, "Console Log")

        layout.addWidget(self._tabs)
