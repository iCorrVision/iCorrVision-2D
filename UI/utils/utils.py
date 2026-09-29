from PySide6.QtCore import SignalInstance
from PySide6.QtWidgets import (
    QLayout,
    QWidget,
    QPushButton,
    QStyle,
    QFileDialog,
)
from typing import Callable


def make_button(
    parent: QWidget,
    icon_enum: QStyle.StandardPixmap,
    slot: Callable[[], None],
    layout: QLayout,
) -> QPushButton:
    """Create a button with a standard icon, connect it to `slot` and add it to `layout`."""
    button = QPushButton(parent)
    button.setIcon(button.style().standardIcon(icon_enum))
    button.clicked.connect(slot)
    layout.addWidget(button)
    return button


def ask_export_folder(parent: QWidget, callback: Callable[[str], None]) -> None:
    """Ask for an export folder and pass it to `callback`; nothing on cancel."""
    path = QFileDialog.getExistingDirectory(parent, "Select Export Folder")
    if path:
        callback(path)


def ask_load_file(
    parent: QWidget, callback: Callable[[str], None], filetype: str
) -> None:
    file, _ = QFileDialog.getOpenFileName(parent, "Load file", " ", filetype)
    if file:
        callback(file)


def make_labeled_toggle(
    parent: QWidget,
    label_on: str,
    label_off: str,
    initial: bool,
    on_toggle: SignalInstance,
) -> QPushButton:
    button = QPushButton(parent)
    button.setCheckable(True)
    button.setChecked(initial)
    button.setText(label_on if initial else label_off)

    def _update_label(checked: bool) -> None:
        button.setText(label_on if checked else label_off)

    button.clicked.connect(_update_label)
    button.clicked.connect(lambda checked: on_toggle.emit(checked))

    return button
