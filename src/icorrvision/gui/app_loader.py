from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QColor

from icorrvision.gui.widgets.main_view import MainView
from icorrvision.gui.main_presenter import MainPresenter

from icorrvision.gui.widgets.utils.theme import THEME


class AppLoader:
    def __init__(self, app: QApplication) -> None:
        self._app: QApplication = app
        self._splash = self._build_splash()
        self._view: MainView | None = None
        self._presenter: MainPresenter | None = None

    def start(self) -> None:
        self._splash.show()
        QTimer.singleShot(0, self._startup)

    def _startup(self) -> None:
        self._update_splash("Building iCorrVision Correlation interface")

        self._view = MainView()
        self._presenter = MainPresenter(self._view)

        self._splash.finish(self._view)
        self._view.show()

    def _update_splash(self, message: str) -> None:
        self._splash.showMessage(
            message,
            Qt.AlignmentFlag.AlignCenter,
            QColor(THEME.splash.text),
        )
        self._app.processEvents()

    @staticmethod
    def _build_splash() -> QSplashScreen:
        pixmap = QPixmap(400, 80)
        pixmap.fill(QColor(THEME.splash.background))
        return QSplashScreen(pixmap)
