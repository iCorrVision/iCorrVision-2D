from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from icorrvision.gui.menu.view import MenuBar


class MenuPresenter(QObject):
    # new_requested_signal = Signal()
    # open_requested_signal = Signal()
    # save_requested_signal = Signal()
    # exit_requested_signal = Signal()
    load_results_requested_signal = Signal(Path)

    def __init__(self, menu: MenuBar, parent=None):
        super().__init__(parent)
        self.menu = menu

        # menu.new_requested.connect(self.new_requested_signal)
        # menu.open_requested.connect(self.open_requested_signal)
        # menu.save_requested.connect(self.save_requested_signal)
        # menu.exit_requested.connect(self.exit_requested_signal)
        menu.load_results_requested.connect(self._on_load_results)

    def _on_load_results(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self.menu, "Load correlation results", "", "iCorrVision archive (*.icorr)"
        )
        if not path_str:
            return
        self.load_results_requested_signal.emit(Path(path_str))
