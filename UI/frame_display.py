from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import QPointF, Qt, QRectF
from PySide6.QtGui import QPainter, QPixmap, QTransform


class FrameDisplay(QWidget):
    """Image view with zoom and pan.

    Mouse-wheel zoom within limits, click-drag panning, double-click to reset
    the view; the aspect ratio is preserved and the image centred.

    The widget knows nothing about ROIs, grids or subsets. Subclasses with
    mouse-driven tools (e.g. SetupFrame's ROI drawing) override the `_handle_*`,
    `_paint_overlay` and `_suppress_pan` hooks rather than the event handlers.
    """

    _ZOOM_MIN: float = 1.0
    MAX_PX_PER_IMAGE_PX = 16

    def __init__(self, parent=None):
        """Start with no image, zoom 1 and no pan."""
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._pixmap: QPixmap | None = None
        self._zoom: float = 1.0
        self._pan: QPointF = QPointF(0.0, 0.0)

        self._drag_origin: QPointF | None = None  # set on mouse click
        self._pan_at_drag: QPointF = QPointF(0.0, 0.0)

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Show `pixmap`, keeping the current zoom and pan."""
        self._pixmap = pixmap
        self.update()

    def reset_view(self) -> None:
        """Reset zoom and pan so the image fits the widget."""
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def _widget_to_norm(self, pos: QPointF) -> tuple[float, float] | None:
        rect = self._render_rect()
        if rect is None or rect.width() <= 0 or rect.height() <= 0:
            return None
        x = (pos.x() - rect.x()) / rect.width()
        y = (pos.y() - rect.y()) / rect.height()
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        return (x, y)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _base_scale(self) -> float:
        """Return the scale (widget px per image px) that fits the image in the widget."""
        if self._pixmap is None or self._pixmap.isNull():
            return 1.0
        return min(
            self.width() / self._pixmap.width(),
            self.height() / self._pixmap.height(),
        )

    def _render_rect(self) -> QRectF | None:
        """Return the widget rectangle the image is drawn in, or None without an image.

        Accounts for the base scale, the zoom and the pan offset.
        """
        if self._pixmap is None or self._pixmap.isNull():
            return None

        pix_w = self._pixmap.width()
        pix_h = self._pixmap.height()
        w = self.width()
        h = self.height()

        total_scale = self._base_scale() * self._zoom

        rendered_w = pix_w * total_scale
        rendered_h = pix_h * total_scale

        # Centre then apply pan
        x = (w - rendered_w) / 2.0 + self._pan.x()
        y = (h - rendered_h) / 2.0 + self._pan.y()

        return QRectF(x, y, rendered_w, rendered_h)

    def _total_scale(self) -> float:
        """Return the base scale times the zoom (widget px per image px)."""
        if self._pixmap is None or self._pixmap.isNull():
            return 1.0
        return self._base_scale() * self._zoom

    # --------------------------------------------------------------------------
    # Subclass hooks
    # --------------------------------------------------------------------------
    # The default implementations are no-ops, so a plain FrameDisplay behaves
    # as a simple image view.

    def _suppress_pan(self) -> bool:
        """Return True to disable click-drag panning.

        Overridden when a subclass tool needs left-click-drag, e.g. to draw a
        rectangle.
        """
        return False

    def _handle_press(self, norm_pos: tuple[float, float], event) -> None:
        """Left press, at a position in normalised image coordinates."""

    def _handle_move(self, norm_pos: tuple[float, float], event) -> None:
        """Mouse move, at a position in normalised image coordinates."""

    def _handle_release(self, norm_pos: tuple[float, float], event) -> None:
        """Left release, at a position in normalised image coordinates."""

    def _handle_double_click(self, norm_pos: tuple[float, float], event) -> None:
        """Left double click while _suppress_pan() is True.

        Otherwise a double click resets the view (see mouseDoubleClickEvent).
        """

    def _paint_overlay(self, painter: QPainter, transform: QTransform) -> None:
        """Draw subclass content at the end of paintEvent.

        Args:
            painter: active QPainter, set up for the widget.
            transform: maps normalised (0..1) image coordinates to widget
                coordinates; apply it to any path or point built in normalised
                space before drawing.
        """

    # --------------------------------------------------------------------------
    # Painting
    # --------------------------------------------------------------------------

    def paintEvent(self, event):
        """Render the pixmap and any subclass overlay to the widget."""
        rect = self._render_rect()
        if rect is None:
            return

        with QPainter(self) as painter:
            # sharp pixel edges at high zoom
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

            src = QRectF(self._pixmap.rect())
            painter.drawPixmap(rect, self._pixmap, src)

            transform = QTransform()
            transform.translate(rect.x(), rect.y())
            transform.scale(rect.width(), rect.height())
            self._paint_overlay(painter, transform)

    # --------------------------------------------------------------------------
    # Mouse actions
    # --------------------------------------------------------------------------

    def _max_zoom(self) -> float:
        """Return the zoom at which one image pixel spans MAX_PX_PER_IMAGE_PX."""
        if self._pixmap is None:
            return 1.0  # no image: zoom is fixed at 1
        return self.MAX_PX_PER_IMAGE_PX / self._base_scale()

    def wheelEvent(self, event):
        """Zoom about the cursor, within the zoom limits."""
        if self._pixmap is None:
            return

        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15

        old_zoom = self._zoom
        new_zoom = max(self._ZOOM_MIN, min(self._max_zoom(), old_zoom * factor))
        if new_zoom == old_zoom:
            return

        cursor = QPointF(event.position())
        w_centre = QPointF(self.width() / 2.0, self.height() / 2.0)

        ratio = new_zoom / old_zoom
        self._pan = cursor - w_centre - (cursor - w_centre - self._pan) * ratio
        self._zoom = new_zoom
        self.update()

    def mousePressEvent(self, event):
        """Dispatch a press to the subclass hook, then start panning."""
        pos = QPointF(event.position())

        if event.button() == Qt.MouseButton.RightButton:
            self._drag_origin = pos
            self._pan_at_drag = QPointF(self._pan)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        norm = self._widget_to_norm(pos)
        if norm is not None:
            self._handle_press(norm, event)

    def mouseMoveEvent(self, event):
        """Dispatch a move to the subclass hook, then pan."""
        pos = QPointF(event.position())
        norm = self._widget_to_norm(pos)
        if norm is not None:
            self._handle_move(norm, event)

        if self._drag_origin is not None and (
            event.buttons() & Qt.MouseButton.RightButton
        ):
            delta = pos - self._drag_origin
            self._pan = self._pan_at_drag + delta
            self.update()

    def mouseReleaseEvent(self, event):
        """Dispatch a release to the subclass hook and end panning."""
        if event.button() == Qt.MouseButton.RightButton:
            self._drag_origin = None
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = QPointF(event.position())
        norm = self._widget_to_norm(pos)
        if norm is not None:
            self._handle_release(norm, event)

    def mouseDoubleClickEvent(self, event):
        """Reset the view, or pass the double click to a subclass tool."""
        if event.button() == Qt.MouseButton.RightButton:
            self.reset_view()

        if event.button() == Qt.MouseButton.LeftButton:
            if self._suppress_pan():
                pos = QPointF(event.position())
                norm = self._widget_to_norm(pos)
                if norm is not None:
                    self._handle_double_click(norm, event)
            else:
                self.reset_view()
