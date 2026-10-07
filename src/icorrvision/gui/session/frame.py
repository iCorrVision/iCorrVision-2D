from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton

from icorrvision.gui.widgets.frame_display import FrameDisplay


class SessionFrame(QWidget):
    """Display panel for the currently selected reference and deformed images."""

    move_def_image_signal: Signal = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the image previews and their navigation controls."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self.reference_frame: FrameDisplay = FrameDisplay()
        self.reference_label: QLabel = QLabel("No frame selected")

        self.deformed_frame: FrameDisplay = FrameDisplay()
        self.deformed_label: QLabel = QLabel("No frame selected")

        layout.addLayout(self._make_reference_panel())
        layout.addLayout(self._make_deformed_panel())

    def _make_reference_panel(self) -> QVBoxLayout:
        """Build the reference-image preview and its label."""
        layout = QVBoxLayout()

        layout.addWidget(self.reference_frame)
        layout.addWidget(self.reference_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        return layout

    def update_reference_label(self, label: str) -> None:
        """Set the label under the reference image."""
        self.reference_label.setText(label)

    def _make_deformed_panel(self) -> QVBoxLayout:
        """Build the deformed-image preview and its navigation buttons."""
        layout = QVBoxLayout()
        frame_layout = QHBoxLayout()

        prev_button = QPushButton("<")
        prev_button.setAutoRepeat(True)
        prev_button.setAutoRepeatDelay(400)
        prev_button.setAutoRepeatInterval(50)
        prev_button.clicked.connect(lambda: self.move_def_image_signal.emit(-1))

        next_button = QPushButton(">")
        next_button.setAutoRepeat(True)
        next_button.setAutoRepeatDelay(400)
        next_button.setAutoRepeatInterval(50)
        next_button.clicked.connect(lambda: self.move_def_image_signal.emit(1))

        frame_layout.addWidget(prev_button)
        frame_layout.addWidget(self.deformed_frame)
        frame_layout.addWidget(next_button)

        layout.addLayout(frame_layout)
        layout.addWidget(self.deformed_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        return layout

    def update_deformed_label(self, label: str) -> None:
        """Set the label under the deformed image."""
        self.deformed_label.setText(label)
