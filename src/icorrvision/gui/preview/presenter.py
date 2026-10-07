import logging
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from components.archive.writer import write_icorr_archive
from components.contracts import CorrelationConfig, CorrelationResult, SetupOutput

from .diagnostics import (
    frame_health,
    run_status_line,
    score_distribution,
    strain_percentiles,
    summary_lines,
)
from .export import export_full_csv
from .payload import (
    STRAIN_COMPONENT_LABELS,
    ColorScale,
    build_payload,
    color_scale,
)
from .preview import CorrelationPreviewDialog


class _SaveWorker(QObject):
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, **archive_kwargs) -> None:
        """Take the keyword arguments of write_icorr_archive, passed on unchanged."""
        super().__init__()
        self._archive_kwargs = archive_kwargs

    @Slot()
    def run(self) -> None:
        try:
            write_icorr_archive(**self._archive_kwargs)
        except Exception as exc:
            self.error_signal.emit(str(exc))
            return
        self.finished_signal.emit()


def _to_grayscale_display(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else image.mean(axis=2)


class PreviewPresenter(QObject):
    """Turn a CorrelationResult into payloads and scales for the preview tabs."""

    _IMAGE_CACHE_SIZE = 8
    _RESULT_STRAIN_COMPONENT = "e1"

    def __init__(
        self,
        view: CorrelationPreviewDialog,
        result: CorrelationResult,
        setup: SetupOutput,
        frame_loader: Callable[[str], np.ndarray],
        csv_path: Path,
        correlation_config: CorrelationConfig,
        reference_frame_name: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._view = view
        self._result = result
        self._setup = setup
        self._frame_loader = frame_loader
        self._csv_path = Path(csv_path)
        self._correlation_config = correlation_config
        self._reference_frame_name = reference_frame_name

        self._save_thread: QThread | None = None
        self._save_worker: _SaveWorker | None = None

        self._image_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._run_scales: dict[str, ColorScale] = {}

        self._frame_index = 0
        self._opacity = 0.7
        self._quantity = "magnitude"
        self._per_frame = False  # matches the checkbox's initial (unchecked) state
        self._strain_component = (
            self._RESULT_STRAIN_COMPONENT
        )  # what the Analysis tab shows

        self._connect_view()
        self._initialize_view()

    def _connect_view(self) -> None:
        view = self._view
        view.save_results_requested.connect(self._on_save_results_requested)

        tab = view.evolution_tab
        tab.frame_changed.connect(self._on_frame_changed)
        tab.quantity_changed.connect(self._on_quantity_changed)
        tab.opacity_changed.connect(self._on_opacity_changed)
        tab.scale_mode_changed.connect(self._on_scale_mode_changed)

        view.analysis_tab.save_as_requested.connect(self._on_csv_save_as)
        view.analysis_tab.copy_path_requested.connect(self._on_csv_copy_path)
        view.analysis_tab.open_folder_requested.connect(self._on_csv_open_folder)
        view.analysis_tab.frame_changed.connect(self._on_analysis_frame_changed)

    def _initialize_view(self) -> None:
        last = len(self._result.frames) - 1
        component = self._RESULT_STRAIN_COMPONENT

        self._view.result_tab.display_results(
            self._get_image(last),
            build_payload(self._result, "magnitude", last),
            color_scale(self._result, "magnitude", frame_index=last),
            build_payload(self._result, component, last),
            color_scale(self._result, component, frame_index=last),
            f"{run_status_line(self._result)}\nCSV exported to: {self._csv_path}",
        )

        self._render_evolution()

        self._view.analysis_tab.render_health(frame_health(self._result))
        self._refresh_component_diagnostics()
        self._view.analysis_tab.set_frame_index(last)
        self._on_analysis_frame_changed(last)
        self._update_csv_info()

    # ==========================================================================
    # Data access
    # ==========================================================================

    def _run_scale(self, quantity: str) -> ColorScale:
        if quantity not in self._run_scales:
            self._run_scales[quantity] = color_scale(self._result, quantity)
        return self._run_scales[quantity]

    def _get_image(self, index: int) -> np.ndarray:
        cached = self._image_cache.get(index)
        if cached is not None:
            self._image_cache.move_to_end(index)
            return cached

        if index == 0:
            raw = self._setup.reference_frame
        else:
            name = self._result.frames[index].frame_name
            raw = self._frame_loader(name)
            if raw is None:  # FolderParser.pull_image returns None on a missing file
                logging.warning(f"Could not load {name}; showing a blank frame.")
                raw = np.zeros_like(self._setup.reference_frame)

        image = _to_grayscale_display(np.asarray(raw))
        self._image_cache[index] = image
        if len(self._image_cache) > self._IMAGE_CACHE_SIZE:
            self._image_cache.popitem(last=False)  # evict least recently used
        return image

    def _frame_info(self, index: int) -> str:
        name = self._result.frames[index].frame_name
        return f"Frame {index} / {len(self._result.frames) - 1}: {name}"

    # ==========================================================================
    # Rendering
    # ==========================================================================

    def _render_evolution(self) -> None:
        index, quantity = self._frame_index, self._quantity
        scale = (
            color_scale(self._result, quantity, frame_index=index)
            if self._per_frame
            else self._run_scale(quantity)
        )
        tab = self._view.evolution_tab
        tab.set_color_scale(scale, per_frame=self._per_frame)
        tab.render_frame(
            self._get_image(index),
            build_payload(self._result, quantity, index),
            self._opacity,
            self._frame_info(index),
        )

    def _refresh_component_diagnostics(self) -> None:
        """Analysis content that depends on the selected strain component."""
        component = self._strain_component
        tab = self._view.analysis_tab
        tab.set_summary(summary_lines(self._result, component))
        tab.render_strain_spread(
            strain_percentiles(self._result, component),
            STRAIN_COMPONENT_LABELS[component],
        )

    def _update_csv_info(self) -> None:
        try:
            size_text = f"{self._csv_path.stat().st_size / 1024:.1f} KB"
        except OSError:
            size_text = "unknown size"
        self._view.analysis_tab.set_csv_info(
            f"Full CSV ({size_text}): {self._csv_path}"
        )

    # ==========================================================================
    # Slots: evolution tab
    # ==========================================================================
    @Slot(int)
    def _on_frame_changed(self, index: int) -> None:
        self._frame_index = index
        self._render_evolution()

    @Slot(str)
    def _on_quantity_changed(self, quantity: str) -> None:
        self._quantity = quantity
        if (
            quantity in STRAIN_COMPONENT_LABELS
        ):  # the Analysis tab follows the strain choice
            self._strain_component = quantity
            self._refresh_component_diagnostics()
        self._render_evolution()

    @Slot(float)
    def _on_opacity_changed(self, opacity: float) -> None:
        self._opacity = opacity
        self._render_evolution()

    @Slot(bool)
    def _on_scale_mode_changed(self, per_frame: bool) -> None:
        self._per_frame = per_frame
        self._render_evolution()

    # ==========================================================================
    # Slots: analysis tab
    # ==========================================================================

    @Slot(int)
    def _on_analysis_frame_changed(self, index: int) -> None:
        counts, edges = score_distribution(self._result, index)
        self._view.analysis_tab.render_scores(
            counts,
            edges,
            self._correlation_config.match_threshold,
            self._frame_info(index),
        )

    @Slot()
    def _on_csv_save_as(self) -> None:
        new_path = self._view.analysis_tab.prompt_save_as_path(self._csv_path)
        if new_path is None:
            return
        export_full_csv(self._result, new_path)
        self._csv_path = new_path
        self._update_csv_info()

    @Slot()
    def _on_csv_copy_path(self) -> None:
        QApplication.clipboard().setText(str(self._csv_path))

    @Slot()
    def _on_csv_open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._csv_path.parent)))

    # ==========================================================================
    # Slots: archive saving
    # ==========================================================================

    @Slot()
    def _on_save_results_requested(self) -> None:
        if self._save_thread is not None:
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self._view, "Save correlation results", "", "iCorrVision archive (*.icorr)"
        )
        if not path_str:
            return
        archive_path = Path(path_str)
        if archive_path.suffix != ".icorr":
            archive_path = archive_path.with_suffix(".icorr")

        deformed_names = [f.frame_name for f in self._result.frames[1:]]

        self._save_thread = QThread()
        self._save_worker = _SaveWorker(
            result=self._result,
            setup_output=self._setup,
            correlation_config=self._correlation_config,
            frame_loader=self._frame_loader,
            reference_frame_name=self._reference_frame_name,
            deformed_frames=deformed_names,
            archive_path=archive_path,
        )
        self._save_worker.moveToThread(self._save_thread)
        self._save_thread.started.connect(self._save_worker.run)
        self._save_worker.finished_signal.connect(
            lambda: self._on_save_finished(archive_path)
        )
        self._save_worker.error_signal.connect(self._on_save_failed)
        self._save_thread.start()

    def _on_save_finished(self, path: Path) -> None:
        self._teardown_save_thread()
        QMessageBox.information(self._view, "Saved", f"Results saved to:\n{path}")

    def _on_save_failed(self, message: str) -> None:
        self._teardown_save_thread()
        QMessageBox.critical(self._view, "Save failed", message)

    def _teardown_save_thread(self) -> None:
        if self._save_thread is not None:
            self._save_thread.quit()
            self._save_thread.wait()
        self._save_thread = None
        self._save_worker = None
