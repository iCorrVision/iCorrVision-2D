from PySide6.QtWidgets import QMenuBar
from PySide6.QtGui import QKeySequence
from PySide6.QtCore import Signal


class MenuBar(QMenuBar):
    # new_requested = Signal()
    # open_requested = Signal()
    # save_requested = Signal()
    # exit_requested = Signal()
    load_results_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # file_menu.addAction("New").triggered.connect(self.new_requested)
        # file_menu.addAction("Open").triggered.connect(self.open_requested)
        # file_menu.addAction("Save").triggered.connect(self.save_requested)

        file_menu = self.addMenu("&File")
        load_results_action = file_menu.addAction("Load Results...")
        load_results_action.setShortcut(QKeySequence("Ctrl+O"))
        load_results_action.triggered.connect(self.load_results_requested)

        # file_menu.addAction("Exit").triggered.connect(self.exit_requested)

        # file_menu = self.addMenu("&Edit")
        # file_menu = self.addMenu("&View")
        # file_menu = self.addMenu("&Help")
