from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QSizePolicy,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QComboBox,
    QListWidget,
    QAbstractItemView,
    QGroupBox,
    QStyle,
    QSplitter,
)
from PySide6.QtGui import QPixmap

from icorrvision.gui.session.frame import SessionFrame

from icorrvision.gui.widgets.utils.utils import ask_export_folder, make_button


class SessionPanel(QWidget):
    """Panel for choosing the session folder, camera feed and images."""

    folder_path_signal: Signal = Signal(str)
    camera_feed_selected_signal: Signal = Signal(str)

    reference_selected_signal: Signal = Signal(int)
    deformed_selected_signal: Signal = Signal(list)  # list[str]

    reference_confirmed_signal: Signal = Signal(str)
    deformed_confirmed_signal: Signal = Signal(list)  # list[str]

    def __init__(self, parent: QWidget | None = None) -> None:
        """Lay out the folder, feed and image widgets."""
        super().__init__(parent)

        self._folder_path_line: QLineEdit = QLineEdit()

        self._def_img_list: QListWidget = QListWidget()

        self._camera_feed_combo: QComboBox
        self._ref_img_list: QListWidget

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(0, 0, 0, 0)

        self.frame: SessionFrame = SessionFrame()
        splitter.addWidget(self.frame)
        self.frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        splitter.setStretchFactor(0, 3)

        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(4, 4, 4, 4)
        control_layout.setSpacing(8)

        control_layout.addWidget(self._make_folder_layout())
        control_layout.addWidget(self._make_camera_feed_picker())
        control_layout.addWidget(self._make_reference_image_list())
        control_layout.addWidget(self._make_deformed_image_list())
        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        splitter.addWidget(control_widget)
        splitter.setStretchFactor(1, 1)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(splitter)

    # ---------------------------------------------------------------
    # GUI Construction
    # ---------------------------------------------------------------

    def _make_folder_layout(self) -> QGroupBox:
        """Build the folder field and its browse button."""
        group = QGroupBox("Folder")
        layout = QHBoxLayout(group)

        self._folder_path_line.setPlaceholderText("Please select a folder")
        self._folder_path_line.setReadOnly(True)
        layout.addWidget(self._folder_path_line)

        set_folder_button: QPushButton = make_button(
            self,
            QStyle.StandardPixmap.SP_DirIcon,
            lambda: ask_export_folder(self, self._on_folder_selected),
            layout,
        )
        layout.addWidget(set_folder_button)

        return group

    def _on_folder_selected(self, path: str) -> None:
        """Store the selected folder and notify the presenter."""
        self._folder_path_line.setText(path)
        self.folder_path_signal.emit(path)

    def _make_camera_feed_picker(self) -> QGroupBox:
        """Build the camera-feed selector (mono or stereo)."""
        group = QGroupBox("Camera Feed")
        layout = QHBoxLayout(group)

        self._camera_feed_combo = QComboBox()
        self._camera_feed_combo.currentTextChanged.connect(
            self.camera_feed_selected_signal
        )
        layout.addWidget(self._camera_feed_combo)

        return group

    def _make_reference_image_list(self) -> QGroupBox:
        """Build the reference-image list and its confirm button."""
        group = QGroupBox("Reference Image")
        layout = QVBoxLayout(group)

        self._ref_img_list = QListWidget()
        self._ref_img_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        layout.addWidget(self._ref_img_list)
        self._ref_img_list.currentRowChanged.connect(self.reference_selected_signal)

        confirm_ref_button = QPushButton("Use as Reference")
        confirm_ref_button.clicked.connect(self._on_reference_confirmed)
        layout.addWidget(confirm_ref_button)

        return group

    def _make_deformed_image_list(self) -> QGroupBox:
        """Build the deformed-image list and its confirm button."""
        group = QGroupBox("Deformed Images")
        layout = QVBoxLayout(group)

        self._def_img_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        layout.addWidget(self._def_img_list)
        self._def_img_list.itemSelectionChanged.connect(self._on_deformed_list_selected)

        confirm_def_button = QPushButton("Use as Deformed Set")
        confirm_def_button.clicked.connect(self._on_deformed_confirmed)
        layout.addWidget(confirm_def_button)

        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self._def_img_list.selectAll)
        layout.addWidget(select_all_button)

        return group

    # ---------------------------------------------------------------
    # Confirm handlers
    # ---------------------------------------------------------------

    def _on_reference_confirmed(self) -> None:
        """Send the selected reference image to the presenter."""
        item = self._ref_img_list.currentItem()
        self.reference_confirmed_signal.emit(item.text())

    def _selected_deformed_names(self) -> list[str]:
        """Selected deformed images in list order, not in the order they were clicked."""
        items = sorted(self._def_img_list.selectedItems(), key=self._def_img_list.row)
        return [item.text() for item in items]

    def _on_deformed_confirmed(self) -> None:
        """Send the confirmed deformed images to the presenter."""
        names = self._selected_deformed_names()
        if not names:
            return
        self.deformed_confirmed_signal.emit(names)

    def _on_deformed_list_selected(self) -> None:
        """Preview the currently selected deformed images."""
        names = self._selected_deformed_names()
        self.deformed_selected_signal.emit(names)

    # ---------------------------------------------------------------
    # Presenter API
    # ---------------------------------------------------------------

    def set_camera_feeds(self, feed_names: list[str]) -> None:
        """Replace the camera-feed options once a manifest is loaded."""
        self._camera_feed_combo.clear()
        self._camera_feed_combo.addItems(feed_names)

    def set_image_candidates(self, image_names: list[str]) -> None:
        """Fill the reference and deformed image lists."""
        self._def_img_list.clear()
        self._ref_img_list.clear()

        self._def_img_list.addItems(image_names)
        self._ref_img_list.addItems(image_names)

    def set_reference_pixmap(self, image: QPixmap, label: str) -> None:
        """Show the reference image and its filename."""
        self.frame.reference_frame.set_pixmap(image)
        self.frame.update_reference_label(label)

    def set_deformed_pixmap(self, image: QPixmap, label: str) -> None:
        """Show the deformed image and its filename."""
        self.frame.deformed_frame.set_pixmap(image)
        self.frame.update_deformed_label(label)
