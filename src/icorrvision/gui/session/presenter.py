import logging
import numpy as np

from PySide6.QtCore import QObject, Slot, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMessageBox

from icorrvision.gui.session.panel import SessionPanel
from icorrvision.io.folder import AmbiguousManifestError, FolderParser
from icorrvision.contracts import CameraChannel


class SessionPresenter(QObject):
    reference_confirmed_signal: Signal = Signal(object, str)
    deformed_confirmed_signal: Signal = Signal(list)
    folder_parser_ready_signal: Signal = Signal(object)

    def __init__(self, view: SessionPanel, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._folder: FolderParser

        self._view: SessionPanel = view
        self._active_channel: CameraChannel | None = None
        self._reference_confirmed: str | None = None
        self._deformed_confirmed: list[str] = []
        self._deformed_selected: list[str] = []
        self._deformed_images: list[str] = []

        self._deformed_count: int = 0
        self._deformed_max: int

        self._view.folder_path_signal.connect(self._on_folder_selected)
        self._view.camera_feed_selected_signal.connect(self._on_camera_feed_selected)

        self._view.reference_selected_signal.connect(self._on_reference_selected)
        self._view.reference_confirmed_signal.connect(self._on_reference_confirmed)

        self._view.deformed_selected_signal.connect(self._on_deformed_selected)
        self._view.deformed_confirmed_signal.connect(self._on_deformed_confirmed)

        self._view.frame.move_def_image_signal.connect(self._on_move_deformed_selected)

    @property
    def is_stereo(self) -> bool:
        """Offer mono or stereo feeds, depending on the loaded manifest."""
        return bool(self._folder.records) and self._folder.records[0].has_right

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_folder_selected(self, folder_path: str) -> None:
        try:
            folder = FolderParser(folder_path)
        except (AmbiguousManifestError, KeyError, ValueError, OSError) as exc:
            # The previous folder, if any, is left in place: nothing is half-loaded.
            self._reset_lists()
            message = (
                str(exc)
                if isinstance(exc, AmbiguousManifestError)
                else f"Could not read the manifest in {folder_path}: {exc!r}"
            )
            logging.critical(message)
            QMessageBox.warning(self._view, "Session", message)
            return
        self._folder = folder

        if not self._folder.records:
            self._reset_lists()
            logging.critical(f"Manifest in {folder_path} contains no rows.")
            return

        self.folder_parser_ready_signal.emit(self._folder)
        feeds = (
            [CameraChannel.LEFT.value, CameraChannel.RIGHT.value]
            if self.is_stereo
            else [CameraChannel.LEFT.value]
        )
        self._view.set_camera_feeds(feeds)

    @Slot(str)
    def _on_camera_feed_selected(self, feed_name: str) -> None:
        if not feed_name or not self._folder.records:
            return

        try:
            channel = CameraChannel(feed_name)
        except ValueError:
            return

        self._active_channel = channel
        self._reference_confirmed = None
        self._deformed_images = []

        image_names = [
            name
            for record in self._folder.records
            if (name := record.filename(channel)) is not None
        ]
        self._view.set_image_candidates(image_names)

    @Slot(str)
    def _on_reference_confirmed(self, image_name: str) -> None:
        self._reference_confirmed = image_name
        arr = self._folder.pull_image(image_name)
        if arr is not None:
            self.reference_confirmed_signal.emit(
                np.ascontiguousarray(arr, dtype=np.uint8), image_name
            )

    @Slot(list)
    def _on_deformed_confirmed(self, image_names: list[str]) -> None:
        self._deformed_confirmed = [
            n for n in image_names if n != self._reference_confirmed
        ]
        self.deformed_confirmed_signal.emit(self._deformed_confirmed)

    @Slot(int)
    def _on_reference_selected(self, idx: int) -> None:
        if self._active_channel is None:
            return
        image_name = self._folder.records[idx].filename(self._active_channel)
        if image_name is None:
            return
        self._view.set_reference_pixmap(self._array_to_pixmap(image_name), image_name)

    @Slot(list)
    def _on_deformed_selected(self, deformed_list: list[str]) -> None:
        if self._active_channel is None:
            return

        self._deformed_selected = deformed_list
        image_name = self._deformed_selected[0]
        self._deformed_count = 0
        self._deformed_max = len(deformed_list) - 1
        self._view.set_deformed_pixmap(self._array_to_pixmap(image_name), image_name)

    @Slot(int)
    def _on_move_deformed_selected(self, move: int) -> None:
        new_idx = self._deformed_count + move
        if new_idx < 0 or new_idx > self._deformed_max:
            return
        self._deformed_count = new_idx
        image_name = self._deformed_selected[self._deformed_count]
        self._view.set_deformed_pixmap(self._array_to_pixmap(image_name), image_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_lists(self) -> None:
        self._active_channel = None
        self._reference_confirmed = None
        self._deformed_images = []
        self._view.set_camera_feeds([])
        self._view.set_image_candidates([])

    # Kept in the presenter rather than the view: it translates model output (an array)
    # into view input (a pixmap).
    def _array_to_pixmap(self, image_name: str) -> QPixmap:
        arr = np.ascontiguousarray(self._folder.pull_image(image_name), dtype=np.uint8)
        height, width = arr.shape
        qimage = QImage(arr.data, width, height, width, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap
