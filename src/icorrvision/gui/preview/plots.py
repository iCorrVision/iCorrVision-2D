import numpy as np
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from icorrvision.contracts import HeatmapPayload

from icorrvision.results.payload import ColorScale
from icorrvision.results.render import HeatmapFigure


class PlotView(QWidget):
    """Frame-by-frame heatmap over the image.

    The colour scale is fixed for the run or rescaled to each frame.
    """

    frame_changed = Signal(int)
    opacity_changed = Signal(float)
    scale_mode_changed = Signal(bool)

    _ZOOM_STEP = 1.05

    def __init__(self, n_frames: int, parent=None) -> None:
        super().__init__(parent)

        self._figure = Figure()
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._heatmap = HeatmapFigure(self._figure)
        self._pan_origin: tuple[float, float, tuple, tuple] | None = None

        self.toolbar = NavigationToolbar2QT(self._canvas, self)
        self._scale_label = QLabel()
        self._frame_label = QLabel()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(self._make_color_scale_button())
        layout.addWidget(self.toolbar)
        layout.addWidget(self._canvas, stretch=1)
        layout.addLayout(self._make_frame_slider(n_frames))
        layout.addLayout(self._make_opacity_slider())

        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("resize_event", self._on_canvas_resize)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)

    # --- UI construction ------------------------------------------------------

    def _make_color_scale_button(self) -> QHBoxLayout:
        scale_row = QHBoxLayout()
        scale_row.addWidget(self._scale_label)
        scale_row.addStretch()
        self._per_frame_box = QCheckBox("Rescale to each frame")
        self._per_frame_box.toggled.connect(self.scale_mode_changed.emit)
        scale_row.addWidget(self._per_frame_box)
        return scale_row

    def _make_frame_slider(self, n_frames: int) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(self._frame_label)
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, max(0, n_frames - 1))
        self._frame_slider.valueChanged.connect(self.frame_changed.emit)
        row.addWidget(self._frame_slider)
        return row

    def _make_opacity_slider(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Heatmap opacity"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(70)
        self._opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        row.addWidget(self._opacity_slider)
        self._opacity_value_label = QLabel("70%")
        row.addWidget(self._opacity_value_label)
        return row

    def _on_opacity_slider_changed(self, value: int) -> None:
        self._opacity_value_label.setText(f"{value}%")
        self.opacity_changed.emit(value / 100.0)

    # --- Presenter API --------------------------------------------------------

    def set_color_scale(self, scale: ColorScale, per_frame: bool = False) -> None:
        self._heatmap.set_scale(scale)
        scope = "this frame only" if per_frame else "fixed across run"
        self._scale_label.setText(f"Scale ({scope}): {scale.describe()}")
        self._figure.tight_layout()

    def render_frame(
        self,
        image: np.ndarray,
        payload: HeatmapPayload,
        opacity: float,
        frame_info_text: str,
    ) -> None:
        self._heatmap.draw(image, payload, opacity, keep_view=True)
        self._apply_cover_fit()
        self._canvas.draw_idle()
        self._frame_label.setText(frame_info_text)

    # --- View fitting ---------------------------------------------------------

    def _on_canvas_resize(self, event) -> None:
        if not self._heatmap.has_content:
            return
        self._apply_cover_fit()
        self._canvas.draw_idle()

    def _apply_cover_fit(self) -> None:
        """Widen the shorter data range so the view fills the axes box exactly."""
        ax = self._heatmap.ax
        pos = ax.get_position(original=True)
        fig_w_in, fig_h_in = self._figure.get_size_inches()
        box_w, box_h = pos.width * fig_w_in, pos.height * fig_h_in
        if box_w <= 0 or box_h <= 0:
            return
        box_aspect = box_w / box_h

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        dx, dy = xlim[1] - xlim[0], abs(ylim[1] - ylim[0])
        if dx <= 0 or dy <= 0:
            return

        cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
        if dx / dy > box_aspect:
            half = dy * box_aspect / 2
            ax.set_xlim(cx - half, cx + half)
        else:
            half = dx / box_aspect / 2
            y_inverted = ylim[0] > ylim[1]  # imshow puts row 0 at the top
            ax.set_ylim(
                (cy + half, cy - half) if y_inverted else (cy - half, cy + half)
            )

    # --- Mouse interaction ----------------------------------------------------

    def _on_scroll(self, event) -> None:
        ax = self._heatmap.ax
        if event.inaxes is not ax or event.xdata is None:
            return  # scrolling over the colorbar must not zoom it

        factor = 1 / self._ZOOM_STEP if event.button == "up" else self._ZOOM_STEP
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        rel_x = (xlim[1] - event.xdata) / (xlim[1] - xlim[0])
        rel_y = (ylim[1] - event.ydata) / (ylim[1] - ylim[0])
        width = (xlim[1] - xlim[0]) * factor
        height = (ylim[1] - ylim[0]) * factor

        ax.set_xlim(event.xdata - width * (1 - rel_x), event.xdata + width * rel_x)
        ax.set_ylim(event.ydata - height * (1 - rel_y), event.ydata + height * rel_y)
        self._canvas.draw_idle()

    def _on_press(self, event) -> None:
        ax = self._heatmap.ax
        if event.button == 3 and event.inaxes is ax:
            self._pan_origin = (event.x, event.y, ax.get_xlim(), ax.get_ylim())

    def _on_motion(self, event) -> None:
        if self._pan_origin is None or event.x is None or event.y is None:
            return
        ax = self._heatmap.ax
        x0, y0, xlim, ylim = self._pan_origin

        inverse = ax.transData.inverted()
        (dx0, dy0), (dx1, dy1) = inverse.transform([(x0, y0), (event.x, event.y)])
        shift_x, shift_y = dx1 - dx0, dy1 - dy0

        ax.set_xlim(xlim[0] - shift_x, xlim[1] - shift_x)
        ax.set_ylim(ylim[0] - shift_y, ylim[1] - shift_y)
        self._canvas.draw_idle()

    def _on_release(self, event) -> None:
        if event.button == 3:
            self._pan_origin = None
