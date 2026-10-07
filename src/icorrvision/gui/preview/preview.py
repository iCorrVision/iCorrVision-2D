from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from icorrvision.contracts import HeatmapPayload

from icorrvision.results.diagnostics import FrameHealth
from icorrvision.results.payload import ColorScale
from icorrvision.gui.preview.plots import PlotView
from icorrvision.results.render import HeatmapFigure


class _ResultTab(QWidget):
    """Last-frame displacement and strain maps side by side."""

    _OPACITY = 0.7

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        maps_row = QHBoxLayout()
        self._panels: list[tuple[FigureCanvasQTAgg, HeatmapFigure]] = []
        for _ in range(2):
            figure = Figure(figsize=(5, 5))
            canvas = FigureCanvasQTAgg(figure)
            self._panels.append((canvas, HeatmapFigure(figure)))
            maps_row.addWidget(canvas)
        layout.addLayout(maps_row)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

    def display_results(
        self,
        image: np.ndarray,
        disp_payload: HeatmapPayload,
        disp_scale: ColorScale,
        strain_payload: HeatmapPayload,
        strain_scale: ColorScale,
        summary_text: str,
    ) -> None:
        self._summary_label.setText(summary_text)
        contents = ((disp_payload, disp_scale), (strain_payload, strain_scale))
        for (canvas, heatmap), (payload, scale) in zip(self._panels, contents):
            heatmap.set_scale(scale)
            heatmap.draw(image, payload, self._OPACITY)
            canvas.draw_idle()


class _EvolutionTab(PlotView):
    """PlotView plus a selector for the quantity shown."""

    quantity_changed = Signal(str)

    def __init__(
        self, n_frames: int, quantity_options: dict[str, str], parent=None
    ) -> None:
        super().__init__(n_frames, parent=parent)
        row = QHBoxLayout()
        row.addWidget(QLabel("Quantity"))
        self._quantity_box = QComboBox()
        for value, label in quantity_options.items():
            self._quantity_box.addItem(label, userData=value)
        self._quantity_box.currentIndexChanged.connect(
            lambda _index: self.quantity_changed.emit(self._quantity_box.currentData())
        )
        row.addWidget(self._quantity_box)
        self.layout().insertLayout(0, row)


class _AnalysisTab(QWidget):
    """Run diagnostics plus CSV export controls."""

    save_as_requested = Signal()
    copy_path_requested = Signal()
    open_folder_requested = Signal()
    frame_changed = Signal(int)

    def __init__(self, n_frames: int, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._summary_label)

        self._figure = Figure(figsize=(8, 5.5), layout="tight")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax_health = self._figure.add_subplot(2, 2, 1)
        self._ax_health_twin = (
            None  # created on first use, only if anything was recovered
        )
        self._ax_scores = self._figure.add_subplot(2, 2, 2)
        self._ax_strain = self._figure.add_subplot(2, 1, 2)
        layout.addWidget(self._canvas)

        frame_row = QHBoxLayout()
        self._frame_label = QLabel()
        frame_row.addWidget(self._frame_label)
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, max(0, n_frames - 1))
        self._frame_slider.valueChanged.connect(self.frame_changed.emit)
        frame_row.addWidget(self._frame_slider)
        layout.addLayout(frame_row)

        self._path_label = QLabel()
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)

        button_row = QHBoxLayout()
        for text, signal in (
            ("Save CSV As...", self.save_as_requested),
            ("Copy path", self.copy_path_requested),
            ("Open containing folder", self.open_folder_requested),
        ):
            button = QPushButton(text)
            button.clicked.connect(signal.emit)
            button_row.addWidget(button)
        layout.addLayout(button_row)

    # --- Presenter API --------------------------------------------------------

    def set_summary(self, lines: list[str]) -> None:
        self._summary_label.setText("<br>".join(lines))

    def set_csv_info(self, info_text: str) -> None:
        self._path_label.setText(info_text)

    def set_frame_index(self, index: int) -> None:
        """Move the slider without emitting frame_changed; the caller renders."""
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(index)
        self._frame_slider.blockSignals(False)

    def render_health(self, health: list[FrameHealth]) -> None:
        idx = [h.frame_index for h in health]
        ax = self._ax_health
        ax.clear()
        ax.plot(idx, [h.valid_pct for h in health], "-o", ms=3)

        if self._ax_health_twin is not None:
            self._ax_health_twin.remove()
            self._ax_health_twin = None
        recovered = [h.recovered for h in health]
        if any(recovered):
            twin = self._ax_health_twin = ax.twinx()
            twin.bar(idx, recovered, alpha=0.3, color="tab:orange")
            twin.set_ylabel("recovered nodes", fontsize=8)
            twin.tick_params(labelsize=7)

        ax.set_xlabel("frame", fontsize=8)
        ax.set_ylabel("valid nodes [% of ROI]", fontsize=8)
        ax.set_title("Run health", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 102)
        ax.set_yticks(range(0, 101, 20))
        self._canvas.draw_idle()

    def render_strain_spread(self, spread: dict, component_label: str) -> None:
        ax = self._ax_strain
        ax.clear()
        idx = spread["frame_index"]
        ax.fill_between(
            idx,
            spread["p1"],
            spread["p99"],
            alpha=0.3,
            color="tab:blue",
            label="p1 - p99",
        )
        ax.plot(idx, spread["median"], "-o", ms=3, color="tab:blue", label="median")
        ax.plot(idx, spread["max"], "--^", ms=3, color="tab:red", label="max")
        ax.plot(idx, spread["min"], "--v", ms=3, color="tab:purple", label="min")
        ax.set_xlabel("frame", fontsize=8)
        ax.set_ylabel(component_label, fontsize=8)
        ax.set_title("Strain spread per frame", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)
        self._canvas.draw_idle()

    def render_scores(
        self,
        counts: np.ndarray,
        edges: np.ndarray,
        match_threshold: float,
        frame_info: str,
    ) -> None:
        ax = self._ax_scores
        ax.clear()
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(
            centres, counts, width=edges[1] - edges[0], color="tab:green", alpha=0.75
        )
        ax.axvline(
            match_threshold,
            color="tab:red",
            linestyle="--",
            linewidth=1,
            label=f"threshold {match_threshold:g}",
        )
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel("correlation score at converged warp", fontsize=8)
        ax.set_ylabel("nodes", fontsize=8)
        ax.set_title("Match quality", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)
        self._frame_label.setText(frame_info)
        self._canvas.draw_idle()

    def prompt_save_as_path(self, current_path: Path) -> Path | None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save correlation CSV", str(current_path), "CSV files (*.csv)"
        )
        return Path(path_str) if path_str else None


class CorrelationPreviewDialog(QDialog):
    """Preview dialog holding the Result, Evolution and Analysis tabs."""

    save_results_requested: Signal = Signal()

    def __init__(
        self, n_frames: int, quantity_options: dict[str, str], parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Correlation preview")
        self.resize(900, 750)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.result_tab = _ResultTab()
        self.evolution_tab = _EvolutionTab(n_frames, quantity_options)
        self.analysis_tab = _AnalysisTab(n_frames)

        self.tabs.addTab(self.result_tab, "Result")
        self.tabs.addTab(self.evolution_tab, "Evolution")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        layout.addWidget(self.tabs)

        save_button = QPushButton("Save Results")
        save_button.clicked.connect(self.save_results_requested.emit)
        layout.addWidget(save_button)
