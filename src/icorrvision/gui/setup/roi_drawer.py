from enum import Enum, auto
from PySide6.QtGui import QPainterPath, QPainter, QImage, QTransform
from PySide6.QtCore import QPointF, QRectF, Qt
from abc import ABC, abstractmethod
import numpy as np


class CanvasMode(Enum):
    """Interaction mode for SetupFrame's canvas.

    IDLE preserves FrameDisplay's default pan/zoom behavior. Every other
    value both selects an ROI drawing tool and suppresses panning while
    that tool is active (see SetupFrame._suppress_pan).
    """

    IDLE = auto()
    DRAW_RECT = auto()
    DRAW_ELLIPSE = auto()
    DRAW_POLYGON = auto()
    DRAW_LINE = auto()


class SelectionOp(Enum):
    """How a newly drawn shape combines with the existing ROI.

    Mirrors the Photoshop/GIMP/Krita modifier-key convention:
    Shift=ADD, Alt=SUBTRACT, Shift+Alt=INTERSECT, no modifier=REPLACE.
    """

    REPLACE = auto()
    ADD = auto()
    SUBTRACT = auto()
    INTERSECT = auto()


class RoiModel:
    painter_scale: int = 4096  # scale of the normalised paths while combining; ideally the image size

    def __init__(self) -> None:
        self._path: QPainterPath = QPainterPath()
        self._history: list[QPainterPath] = []

    @property
    def path(self) -> QPainterPath:
        """A copy of the current composed ROI path."""
        return QPainterPath(self._path)

    def is_empty(self) -> bool:
        return self._path.isEmpty()

    def commit(self, shape: QPainterPath, op: SelectionOp) -> None:
        """Combine a newly drawn shape into the ROI per the given op."""
        self._history.append(QPainterPath(self._path))

        if op is SelectionOp.REPLACE:
            self._path = shape
            return

        up = QTransform().scale(self.painter_scale, self.painter_scale)
        down = QTransform().scale(1.0 / self.painter_scale, 1.0 / self.painter_scale)

        a = up.map(self._path)
        b = up.map(shape)

        if op is SelectionOp.ADD:
            result = a.united(b)
        elif op is SelectionOp.SUBTRACT:
            result = a.subtracted(b)
        elif op is SelectionOp.INTERSECT:
            result = a.intersected(b)

        self._path = down.map(result)

    def undo(self) -> bool:
        """Undo the last committed change; return False if there is none."""
        if not self._history:
            return False
        self._path = self._history.pop()
        return True

    def clear(self) -> None:
        if self._path.isEmpty():
            return
        self._history.append(QPainterPath(self._path))
        self._path = QPainterPath()

    def to_mask(self, reference_image: np.ndarray) -> np.ndarray:
        "Rasterize ROI"
        height, width = reference_image.shape[:2]
        img = QImage(width, height, QImage.Format.Format_Grayscale8)
        img.fill(0)

        transform = QTransform()
        transform.scale(width, height)
        scaled_path = transform.map(self._path)

        painter = QPainter(img)
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing, False
        )  # hard edges for a mask
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawPath(scaled_path)
        painter.end()

        ptr = img.bits()
        buf = np.frombuffer(ptr, dtype=np.uint8, count=img.sizeInBytes())
        arr = buf.reshape(height, img.bytesPerLine())[:, :width]

        return arr > 0


class CanvasTool(ABC):
    """Base for anything that drives SetupFrame's canvas via press/move/release.

    Subclasses build the shape in progress as the user interacts, and
    SetupFrame reads preview_path on every paint to draw it. The meaning of a
    finished interaction is left to each tool: most return a QPainterPath to
    commit into the ROI mask, LineScaleTool returns a segment.
    """

    def __init__(self) -> None:
        self._path: QPainterPath | None = None

    @property
    def is_active(self) -> bool:
        """Whether a stroke is currently in progress."""
        return self._path is not None

    @property
    def preview_path(self) -> QPainterPath | None:
        """The current in-progress path, for live rendering. None if idle."""
        return self._path

    def reset(self) -> None:
        self._path = None

    @abstractmethod
    def on_press(self, norm_pos: tuple[float, float]) -> None: ...

    @abstractmethod
    def on_move(self, norm_pos: tuple[float, float]) -> None: ...


class RoiToolStrategy(CanvasTool, ABC):
    """Base for tools whose finished shape gets unioned into the ROI mask."""

    @abstractmethod
    def on_release(self, norm_pos: tuple[float, float]) -> QPainterPath | None: ...

    def on_double_click(self, norm_pos: tuple[float, float]) -> QPainterPath | None:
        """Handle a double click; used by the polygon tool, a no-op by default."""
        return None


class RectRoiTool(RoiToolStrategy):
    """Click-drag rectangle, anchored at the press point."""

    def __init__(self) -> None:
        super().__init__()
        self._start: tuple[float, float] | None = None

    def on_press(self, norm_pos: tuple[float, float]) -> None:
        self._start = norm_pos
        self._path = self._build(norm_pos, norm_pos)

    def on_move(self, norm_pos: tuple[float, float]) -> None:
        if self._start is None:
            return
        self._path = self._build(self._start, norm_pos)

    def on_release(self, norm_pos: tuple[float, float]) -> QPainterPath | None:
        if self._start is None:
            return None
        path = self._build(self._start, norm_pos)
        self._start = None
        self._path = None
        return path if not path.isEmpty() else None

    @staticmethod
    def _build(a: tuple[float, float], b: tuple[float, float]) -> QPainterPath:
        rect = QRectF(QPointF(*a), QPointF(*b)).normalized()
        path = QPainterPath()
        path.addRect(rect)
        return path


class EllipseRoiTool(RectRoiTool):
    """Click-drag ellipse, inscribed in the same bounding box a rectangle would use.

    Reuses RectRoiTool's press, move and release handling: the bounding box
    is dragged out in the same way, and only the path geometry differs.
    """

    @staticmethod
    def _build(a: tuple[float, float], b: tuple[float, float]) -> QPainterPath:
        rect = QRectF(QPointF(*a), QPointF(*b)).normalized()
        path = QPainterPath()
        path.addEllipse(rect)
        return path


class PolygonRoiTool(RoiToolStrategy):
    """Click to place vertices, double-click to close.

    Qt delivers a double click as press, release, doubleClick, release. The
    first press of the closing double click therefore adds one more vertex at
    or near the closing point before on_double_click runs. The finished path
    carries this near-duplicate vertex, which is harmless unless the vertex
    count must be exact.
    """

    _MIN_VERTICES: int = 3

    def __init__(self) -> None:
        super().__init__()
        self._points: list[tuple[float, float]] = []

    def on_press(self, norm_pos: tuple[float, float]) -> None:
        self._points.append(norm_pos)
        self._path = self._build(self._points)

    def on_move(self, norm_pos: tuple[float, float]) -> None:
        if not self._points:
            return
        self._path = self._build(self._points + [norm_pos])

    def on_release(self, norm_pos: tuple[float, float]) -> QPainterPath | None:
        return None  # commitment only happens via on_double_click

    def on_double_click(self, norm_pos: tuple[float, float]) -> QPainterPath | None:
        if len(self._points) < self._MIN_VERTICES:
            self._points = []
            self._path = None
            return None
        path = self._build(self._points, closed=True)
        self._points = []
        self._path = None
        return path

    @staticmethod
    def _build(points: list[tuple[float, float]], closed: bool = False) -> QPainterPath:
        path = QPainterPath()
        if not points:
            return path
        path.moveTo(*points[0])
        for p in points[1:]:
            path.lineTo(*p)
        if closed:
            path.closeSubpath()
        return path


class LineScaleTool(CanvasTool):
    """Draws one reference segment for pixel↔real-world scale calibration.

    Outside the RoiModel combination: on_release returns a pair of QPointF
    endpoints rather than a QPainterPath for the ROI mask. The presenter turns
    the pixel length of the segment into a scale factor once the known
    real-world distance is entered.
    """

    def __init__(self) -> None:
        super().__init__()
        self._start: QPointF | None = None
        self._end: QPointF | None = None

    @property
    def is_active(self) -> bool:
        return self._start is not None

    @property
    def preview_path(self) -> QPainterPath | None:
        if self._start is None or self._end is None:
            return None
        path = QPainterPath()
        path.moveTo(self._start)
        path.lineTo(self._end)
        return path

    def on_press(self, norm_pos: tuple[float, float]) -> None:
        self._start = QPointF(*norm_pos)
        self._end = QPointF(*norm_pos)

    def on_move(self, norm_pos: tuple[float, float]) -> None:
        if self._start is not None:
            self._end = QPointF(*norm_pos)

    def on_release(
        self, norm_pos: tuple[float, float]
    ) -> tuple[QPointF, QPointF] | None:
        if self._start is None:
            return None
        end = QPointF(*norm_pos)
        start = self._start
        self._start = self._end = None
        return start, end
