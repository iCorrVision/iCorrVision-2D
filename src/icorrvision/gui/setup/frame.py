from PySide6.QtCore import Signal, Qt, QPointF, QRectF

from PySide6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QTransform,
    QBrush,
    QPolygonF,
)

from icorrvision.gui.widgets.frame_display import FrameDisplay
from icorrvision.gui.setup.roi_drawer import (
    CanvasMode,
    SelectionOp,
    CanvasTool,
    RoiToolStrategy,
    RectRoiTool,
    EllipseRoiTool,
    PolygonRoiTool,
    LineScaleTool,
)


class SetupFrame(FrameDisplay):
    """FrameDisplay extended with ROI drawing tools.

    Holds only the stroke in progress (through the active RoiToolStrategy).
    The committed ROI lives in SetupPresenter's RoiModel and arrives through
    set_roi_path; SetupFrame renders it and reports finished shapes through
    roi_shape_committed. The view never combines shapes itself.
    """

    roi_shape_committed: Signal = Signal(QPainterPath, object)
    _ROI_COLOR = QColor(0, 0, 255, 200)
    _ROI_FILL_COLOR = QColor(0, 0, 255, 40)

    scale_line_committed: Signal = Signal(QPointF, QPointF)
    _SCALE_COLOR = QColor(0, 255, 0, 200)

    grid_anchor_dragged: Signal = Signal(QPointF)
    _GRID_DOT_COLOR = QColor(255, 140, 0, 220)
    _GRID_ANCHOR_COLOR = QColor(255, 140, 0, 220)
    _GRID_ANCHOR_FILL_COLOR = QColor(255, 140, 0, 40)

    _TOOL_FACTORIES: dict[CanvasMode, type[CanvasTool]] = {
        CanvasMode.DRAW_RECT: RectRoiTool,
        CanvasMode.DRAW_ELLIPSE: EllipseRoiTool,
        CanvasMode.DRAW_POLYGON: PolygonRoiTool,
        CanvasMode.DRAW_LINE: LineScaleTool,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode: CanvasMode = CanvasMode.IDLE
        self._tool: CanvasTool | None = None
        self._pending_op: SelectionOp = SelectionOp.REPLACE
        self._committed_path: QPainterPath = QPainterPath()
        self._scale_line: tuple[QPointF, QPointF] | None = None

        self._committed_pen = QPen(self._ROI_COLOR, 2)
        self._committed_brush = QBrush(self._ROI_FILL_COLOR)
        self._preview_pen = QPen(self._ROI_COLOR, 1, Qt.PenStyle.DashLine)
        self._scale_pen = QPen(self._SCALE_COLOR, 2)
        self._scale_preview_pen = QPen(self._SCALE_COLOR, 1, Qt.PenStyle.DashLine)

        self._grid_nodes: list[QPointF] = []
        self._grid_anchor: QPointF | None = None
        self._grid_half_extent: tuple[float, float] | None = None
        self._grid_visible: bool = False
        self._anchor_drag_active: bool = False
        self._anchor_preview: QPointF | None = None

        self._grid_dot_pen = QPen(self._GRID_DOT_COLOR, 4)
        self._grid_dot_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self._grid_anchor_pen = QPen(self._GRID_ANCHOR_COLOR, 2)
        self._grid_anchor_brush = QBrush(self._GRID_ANCHOR_FILL_COLOR)

    # ------------------------------------------------------------------
    # Presenter-facing API
    # ------------------------------------------------------------------

    def set_mode(self, mode: CanvasMode) -> None:
        """Switch the active drawing tool (or return to pan/zoom via IDLE)."""
        self._mode = mode
        factory = self._TOOL_FACTORIES.get(mode)
        self._tool = factory() if factory else None
        self._drag_origin = None
        self.update()

    def set_roi_path(self, path: QPainterPath) -> None:
        """Receive the composed ROI from RoiModel."""
        self._committed_path = path
        self.update()

    def set_scale_line(self, line: tuple[QPointF, QPointF] | None) -> None:
        """Receive the scale line, as set_roi_path does the ROI."""
        self._scale_line = line
        self.update()

    def set_grid_preview(
        self,
        nodes: list[QPointF],
        anchor: QPointF | None,
        half_extent: tuple[float, float] | None,
        visible: bool,
    ) -> None:
        """Receive the grid preview state."""
        self._grid_nodes = nodes
        self._grid_anchor = anchor
        self._grid_half_extent = half_extent
        self._grid_visible = visible
        if not visible:
            self._anchor_drag_active = False
        self.update()

    # ------------------------------------------------------------------
    # FrameDisplay hooks
    # ------------------------------------------------------------------

    def _suppress_pan(self) -> bool:
        return self._tool is not None

    def _handle_press(self, norm_pos, event) -> None:
        if self._tool is not None:
            if not self._tool.is_active:
                self._pending_op = self._op_from_modifiers(event.modifiers())
            self._tool.on_press(norm_pos)
            self.update()
            return

        if self._grid_visible and self._hit_test_anchor(norm_pos):
            self._anchor_drag_active = True
            self._anchor_preview = QPointF(*norm_pos)
            self.update()

    def _handle_move(self, norm_pos, event) -> None:
        if self._tool is not None:
            self._tool.on_move(norm_pos)
            self.update()
            return

        if self._anchor_drag_active:
            self._anchor_preview = QPointF(*norm_pos)
            self.update()

    def _handle_release(self, norm_pos, event) -> None:
        if self._tool is not None:
            result = self._tool.on_release(norm_pos)  # a QPainterPath, or a point pair for LineScaleTool
            if result is not None:
                self._emit_commit(result)
            self.update()
            return

        if self._anchor_drag_active:
            self._anchor_drag_active = False
            self.grid_anchor_dragged.emit(QPointF(*norm_pos))
            self.update()

    def _hit_test_anchor(self, norm_pos: tuple[float, float]) -> bool:
        if self._grid_anchor is None or self._grid_half_extent is None:
            return False
        half_x, half_y = self._grid_half_extent
        dx = abs(norm_pos[0] - self._grid_anchor.x())
        dy = abs(norm_pos[1] - self._grid_anchor.y())
        return dx <= half_x and dy <= half_y

    def _handle_double_click(self, norm_pos, event) -> None:
        if not isinstance(self._tool, RoiToolStrategy):
            return
        path = self._tool.on_double_click(norm_pos)
        if path is not None:
            self._emit_commit(path)
        self.update()

    def _emit_commit(self, result) -> None:
        """Route a finished tool result to the right signal by mode.

        Line mode produces a (QPointF, QPointF) pair for calibration; every
        other mode produces a QPainterPath meant to be combined into the ROI.
        """
        if self._mode is CanvasMode.DRAW_LINE:
            start, end = result
            self.scale_line_committed.emit(start, end)
        else:
            self.roi_shape_committed.emit(result, self._pending_op)

    def _paint_overlay(self, painter: QPainter, transform: QTransform) -> None:
        if not self._committed_path.isEmpty():
            self._committed_pen.setColor(self._ROI_COLOR)
            painter.setPen(self._committed_pen)
            painter.setBrush(self._committed_brush)
            painter.drawPath(transform.map(self._committed_path))

        if self._scale_line is not None:
            start, end = self._scale_line
            painter.setPen(self._scale_pen)
            painter.drawLine(transform.map(start), transform.map(end))

        if self._tool is not None and self._tool.preview_path is not None:
            preview_pen = (
                self._scale_preview_pen
                if self._mode is CanvasMode.DRAW_LINE
                else self._preview_pen
            )
            painter.setPen(preview_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(transform.map(self._tool.preview_path))

        if self._grid_visible:
            self._paint_grid(painter, transform)

    def _paint_grid(self, painter: QPainter, transform: QTransform) -> None:
        if self._grid_nodes:
            painter.setPen(self._grid_dot_pen)
            painter.drawPoints(
                QPolygonF([transform.map(node) for node in self._grid_nodes])
            )
        anchor = self._anchor_preview if self._anchor_drag_active else self._grid_anchor
        if anchor is not None and self._grid_half_extent is not None:
            half_x, half_y = self._grid_half_extent
            rect = QRectF(
                transform.map(QPointF(anchor.x() - half_x, anchor.y() - half_y)),
                transform.map(QPointF(anchor.x() + half_x, anchor.y() + half_y)),
            )
            painter.setPen(self._grid_anchor_pen)
            painter.setBrush(self._grid_anchor_brush)
            painter.drawRect(rect)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _op_from_modifiers(modifiers) -> SelectionOp:
        """Read Shift/Alt at stroke-start into a SelectionOp."""
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        alt = bool(modifiers & Qt.KeyboardModifier.AltModifier)
        if shift and alt:
            return SelectionOp.INTERSECT
        if shift:
            return SelectionOp.ADD
        if alt:
            return SelectionOp.SUBTRACT
        return SelectionOp.REPLACE
