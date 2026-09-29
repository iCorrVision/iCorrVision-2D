from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
)


class ModePanelView(QWidget):
    """Tabbed panel of the workflow modes (Session, Setup, Correlation)."""

    def __init__(self, modes: dict[str, type[QWidget]], parent=None):
        """Add one tab per entry of `modes`, a mapping of name to view class."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._tabs = QTabWidget()
        self._views: dict[str, QWidget] = {}

        for panel_name, panel_ui in modes.items():
            instance = panel_ui()
            self._views[panel_name] = instance
            self._tabs.addTab(instance, panel_name)

        layout.addWidget(self._tabs)

    def append_panel(self, panel_name: str, panel_ui: type[QWidget]) -> None:
        instance = panel_ui()
        self._views[panel_name] = instance
        self._tabs.addTab(instance, panel_name)

    def get_view(self, name: str) -> QWidget:
        """Return the view of the named mode."""
        return self._views[name]
