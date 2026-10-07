from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QSplitter,
    QSizePolicy,
)
from icorrvision.gui.widgets.mode_panel_view import ModePanelView
from icorrvision.gui.widgets.bottom_panel_view import BottomPanelView
from icorrvision.gui.session.panel import SessionPanel
from icorrvision.gui.setup.panel import SetupPanel
from icorrvision.gui.correlation.panel import CorrelationPanel
from icorrvision.gui.menu.view import MenuBar


class MainView(QMainWindow):
    modes: dict[str, type[QWidget]] = {
        "Session": SessionPanel,
        "Setup": SetupPanel,
        "Correlation": CorrelationPanel,
    }

    mode_panel: ModePanelView
    bottom_panel: BottomPanelView

    def __init__(self, parent=None):
        """Build the main window and its panels."""
        super().__init__(parent)
        self.setWindowTitle("DICCorrelation")

        main_layout = QSplitter(Qt.Orientation.Vertical)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.menu: MenuBar = MenuBar()
        self.setMenuBar(self.menu)

        self.mode_panel = ModePanelView(modes=self.modes)
        main_layout.addWidget(self.mode_panel)
        self.mode_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_layout.setStretchFactor(0, 3)

        self.bottom_panel = BottomPanelView()
        main_layout.addWidget(self.bottom_panel)
        main_layout.setStretchFactor(0, 1)

        self.setCentralWidget(main_layout)
