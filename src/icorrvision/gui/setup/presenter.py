import numpy as np

from PIL import Image
import logging
from PySide6.QtCore import QObject, Slot, Signal, QPointF, QLineF
from PySide6.QtGui import QImage, QPixmap, QPainterPath
from PySide6.QtWidgets import QFileDialog, QMessageBox

from icorrvision.engine.engine import node_grid
from icorrvision.gui.setup.panel import SetupPanel
from icorrvision.gui.setup.roi_drawer import RoiModel, CanvasMode


def _compute_grid_nodes(
    step: int,
    subset_size: int | None,
    image_shape: tuple[int, int],
    roi_mask: np.ndarray | None = None,
) -> list[QPointF]:
    """Return the node positions the engine will use, normalised to the image."""
    height, width = image_shape[:2]
    if step <= 0 or width <= 0 or height <= 0:
        logging.warning("not enough information to compute grid nodes")
        return []

    mask = np.ones((height, width), dtype=bool) if roi_mask is None else roi_mask
    try:
        grid_x, grid_y, inside = node_grid(mask, subset_size or 1, step)
    except ValueError:  # the ROI vanishes once eroded by the subset half-width
        return []
    px, py = grid_x[inside], grid_y[inside]
    return [QPointF(x / width, y / height) for x, y in zip(px.tolist(), py.tolist())]


class SetupPresenter(QObject):

    roi_mask_confirmed_signal: Signal = Signal(np.ndarray)
    grid_confirmed_signal: Signal = Signal(int, int)
    scale_confirmed_signal: Signal = Signal(float)

    def __init__(self, view: SetupPanel, parent=None) -> None:
        super().__init__(parent)
        self._view: SetupPanel = view
        self._roi_model: RoiModel = RoiModel()
        self._reference_frame: np.ndarray | None = None
        self._roi_mask: np.ndarray | None = None
        self._scale_line: tuple[QPointF, QPointF] | None = None
        self._mm_measurement: float = 1.0
        self._subset_size: int | None = None
        self._step_size: int | None = None
        self._grid_anchor: QPointF | None = None
        self._grid_visible: bool = False

        self._view.roi_mode_changed_signal.connect(self._on_mode_changed)
        self._view.roi_clear_signal.connect(self._on_roi_clear)
        self._view.roi_undo_signal.connect(self._on_roi_undo)
        self._view.frame.roi_shape_committed.connect(self._on_shape_committed)
        self._view.roi_confirmed_signal.connect(self._on_roi_confirmed)
        self._view.roi_export_signal.connect(self._on_roi_export)
        # self._view.roi_load_signal.connect(self._on_roi_load)

        self._view.grid_commited_signal.connect(self._on_grid_commited)
        self._view.grid_confirmed_signal.connect(self._on_grid_confirmed)
        self._view.grid_visuals_signal.connect(self._on_grid_visuals_toggled)
        self._view.frame.grid_anchor_dragged.connect(self._on_grid_anchor_dragged)

        self._view.scale_draw_signal.connect(self._on_scale_draw)
        self._view.scale_clear_signal.connect(self._on_scale_clear)
        self._view.scale_value_signal.connect(self._on_scale_value_changed)
        self._view.frame.scale_line_committed.connect(self._on_line_committed)
        self._view.scale_confirmed_signal.connect(self._on_scale_confirmed)

    # --------------------------------------------------------------------------
    # Frame Slots
    # --------------------------------------------------------------------------

    @Slot(object)  # handoff from Session
    def set_reference_image(self, array: np.ndarray) -> None:
        """Receive the confirmed reference frame from SessionPresenter, via MainPresenter."""
        arr = np.ascontiguousarray(array, dtype=np.uint8)
        self._reference_frame = arr
        height, width = arr.shape
        qimage = QImage(arr.data, width, height, width, QImage.Format.Format_Grayscale8)
        self._view.frame.set_pixmap(QPixmap.fromImage(qimage))

        # A new reference image invalidates the ROI and scale line drawn on the old one.
        self._roi_model.clear()
        self._roi_mask = None
        self._view.set_roi_path(self._roi_model.path)
        self._scale_line = None
        self._view.set_scale_line(None)

    # --------------------------------------------------------------------------
    # ROI Slots
    # --------------------------------------------------------------------------

    @Slot(object)
    def _on_mode_changed(self, mode: CanvasMode) -> None:
        self._view.set_drawing_mode(mode)

    @Slot()
    def _on_roi_clear(self) -> None:
        self._roi_model.clear()
        self._view.set_roi_path(self._roi_model.path)
        self._roi_mask = None
        self._update_grid_preview()

    @Slot()
    def _on_roi_undo(self) -> None:
        if self._roi_model.undo():
            self._view.set_roi_path(self._roi_model.path)
            self._update_grid_preview()

    @Slot(QPainterPath, object)
    def _on_shape_committed(self, shape: QPainterPath, op) -> None:
        self._roi_model.commit(shape, op)
        self._view.set_roi_path(self._roi_model.path)
        if self._reference_frame is not None:
            self._roi_mask = self._roi_model.to_mask(self._reference_frame)
            self._update_grid_preview()

    @Slot()
    def _on_roi_confirmed(self) -> None:
        if self._roi_mask is not None:
            self.roi_mask_confirmed_signal.emit(self._roi_mask)
        else:
            self.roi_mask_confirmed_signal.emit(
                np.ones(self._reference_frame.shape[:2], dtype=bool)
            )

    @Slot()
    def _on_roi_export(self) -> None:
        if self._roi_mask is None:
            logging.warning("No ROI mask to export")
            return
        # The presenter is not a widget, so the view is the dialog's parent.
        file_path, _ = QFileDialog.getSaveFileName(
            self._view, "Save Mask", "mask.tiff", "TIFF Files (*.tif *.tiff)"
        )
        if not file_path:
            return

        try:
            if not file_path.lower().endswith((".tif", ".tiff")):
                file_path += ".tiff"
            Image.fromarray(self._roi_mask).save(file_path)

        except Exception as e:
            QMessageBox.critical(self._view, "Error", f"Could not save file: {e}")

    # --------------------------------------------------------------------------
    # Grid Slots
    # --------------------------------------------------------------------------

    @Slot(int, int)
    def _on_grid_commited(self, subset: int, step: int) -> None:
        self._subset_size = subset
        self._step_size = step
        self._update_grid_preview()

    @Slot()
    def _on_grid_confirmed(self) -> None:
        if self._subset_size is None or self._step_size is None:
            logging.warning("please set valid grid parameters")
            return
        self.grid_confirmed_signal.emit(self._subset_size, self._step_size)

    @Slot(bool)
    def _on_grid_visuals_toggled(self, visible: bool) -> None:
        self._grid_visible = visible
        self._update_grid_preview()

    @Slot(QPointF)
    def _on_grid_anchor_dragged(self, anchor: QPointF) -> None:
        self._grid_anchor = anchor
        self._update_grid_preview()

    def _update_grid_preview(self) -> None:
        if (
            not self._grid_visible
            or self._reference_frame is None
            or self._step_size is None
        ):
            self._view.set_grid_preview([], None, None, False)
            return

        height, width = self._reference_frame.shape[:2]

        if self._grid_anchor is None:
            self._grid_anchor = self._default_anchor(width, height)

        nodes = _compute_grid_nodes(
            step=self._step_size,
            subset_size=self._subset_size,
            image_shape=(height, width),
            roi_mask=self._roi_mask,
        )

        half_extent = None
        if self._subset_size is not None:
            half_extent = (
                (self._subset_size / 2) / width,
                (self._subset_size / 2) / height,
            )

        self._view.set_grid_preview(nodes, self._grid_anchor, half_extent, True)

    def _default_anchor(self, width: int, height: int) -> QPointF:
        """Return the centroid of the ROI mask, or the image centre without one."""
        if self._roi_mask is not None and self._roi_mask.any():
            ys, xs = np.nonzero(self._roi_mask)
            return QPointF(float(xs.mean()) / width, float(ys.mean()) / height)
        return QPointF(0.5, 0.5)

    # --------------------------------------------------------------------------
    # Scale slots
    # --------------------------------------------------------------------------

    @Slot()
    def _on_scale_draw(self) -> None:
        self._view.set_drawing_mode(CanvasMode.DRAW_LINE)

    @Slot()
    def _on_scale_clear(self) -> None:
        self._scale_line = None
        self._view.set_scale_line(None)

    @Slot(QPointF, QPointF)
    def _on_line_committed(self, start: QPointF, end: QPointF) -> None:
        self._scale_line = (start, end)
        self._view.set_scale_line(self._scale_line)
        self._view.set_drawing_mode(CanvasMode.IDLE)

    @Slot(float, float)
    def _on_scale_value_changed(self, value: float, magnitude: float) -> None:
        if value <= 0:
            logging.warning("Please input a valid scale size")
            return
        self._mm_measurement = value * magnitude

    @Slot()
    def _on_scale_confirmed(self) -> None:
        if self._scale_line is None or self._reference_frame is None:
            return
        height, width = self._reference_frame.shape[:2]
        start, end = self._scale_line
        start_px = QPointF(start.x() * width, start.y() * height)
        end_px = QPointF(end.x() * width, end.y() * height)
        px_distance = QLineF(start_px, end_px).length()
        if px_distance <= 0:
            logging.warning("Scale line has zero length")
            return
        calibration_mm_per_px = self._mm_measurement / px_distance
        self._view.show_calibration(calibration_mm_per_px)
        self.scale_confirmed_signal.emit(calibration_mm_per_px)
