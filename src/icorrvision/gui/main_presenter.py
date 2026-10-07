import logging
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog

from icorrvision.gui.menu.presenter import MenuPresenter
from icorrvision.gui.console_logger import ConsoleLogger
from icorrvision.contracts import (
    CorrelationResult,
    SessionOutput,
    SetupOutput,
    CorrelationConfig,
)
from icorrvision.io.folder import FolderParser
from icorrvision.gui.session.presenter import SessionPresenter
from icorrvision.gui.setup.presenter import SetupPresenter
from icorrvision.io.archive.reader import read_icorr_archive, LoadedArchive

from icorrvision.gui.correlation.presenter import CorrelationPresenter

from icorrvision.results.payload import QUANTITY_LABELS
from icorrvision.io.export import export_full_csv, export_summary_csv

from icorrvision.gui.preview.preview import CorrelationPreviewDialog
from icorrvision.gui.preview.presenter import PreviewPresenter


class _LoadWorker(QObject):
    finished_signal = Signal(object)  # LoadedArchive
    error_signal = Signal(str)

    def __init__(self, path: Path):
        super().__init__()
        self._path = path

    @Slot()
    def run(self) -> None:
        try:
            archive = read_icorr_archive(self._path)
        except Exception as exc:
            self.error_signal.emit(str(exc))
            return
        self.finished_signal.emit(archive)


class MainPresenter(QObject):
    """Connect the sub-presenters with each other and with their views."""

    def __init__(self, view) -> None:
        super().__init__()
        self._view = view
        self._folder: FolderParser | None = None
        self._frame_loader = None
        self.session_output: SessionOutput = SessionOutput()
        self.setup_output: SetupOutput = SetupOutput()
        self._load_thread: QThread | None = None
        self._load_worker: _LoadWorker | None = None
        self._load_progress: QProgressDialog | None = None

        self._console_presenter: ConsoleLogger = ConsoleLogger(
            view.bottom_panel.console
        )
        self._session_presenter: SessionPresenter = SessionPresenter(
            view.mode_panel.get_view("Session")
        )
        self._setup_presenter: SetupPresenter = SetupPresenter(
            view.mode_panel.get_view("Setup")
        )
        self._correlation_presenter: CorrelationPresenter = CorrelationPresenter(
            view.mode_panel.get_view("Correlation")
        )
        self._menu_presenter: MenuPresenter = MenuPresenter(view.menu)

        # --- session signal connections ---

        self._session_presenter.reference_confirmed_signal.connect(
            self._on_reference_confirmed
        )
        self._session_presenter.deformed_confirmed_signal.connect(
            self._on_deformed_confirmed
        )
        self._session_presenter.folder_parser_ready_signal.connect(
            self._on_folder_ready
        )

        # --- setup signal connections ---

        self._setup_presenter.roi_mask_confirmed_signal.connect(self._on_roi_confirmed)
        self._setup_presenter.grid_confirmed_signal.connect(self._on_grid_confirmed)
        self._setup_presenter.scale_confirmed_signal.connect(self._on_scale_confirmed)

        # --- correlation signal connections ---

        self._correlation_presenter.result_ready_signal.connect(
            self._on_correlation_result
        )

        # --- menu signal connections ---

        self._menu_presenter.load_results_requested_signal.connect(
            self.load_results_archive
        )
        # self._menu_presenter.new_requested_signal.connect(self._on_new_requested)
        # self._menu_presenter.open_requested_signal.connect(self._on_open_requested)
        # self._menu_presenter.save_requested_signal.connect(self._on_save_requested)
        # self._menu_presenter.exit_requested_signal.connect(self._on_exit_requested)

    # --------------------------------------------------------------------------
    # Session Slots
    # --------------------------------------------------------------------------

    def _on_reference_confirmed(self, frame, name: str) -> None:
        self._setup_presenter.set_reference_image(frame)
        self.session_output.reference_frame = name
        # ROI, grid and scale were confirmed against the previous reference: they
        # must be confirmed again for this one.
        self.setup_output = replace(
            self.setup_output,
            reference_frame=frame,
            roi_mask=None,
            subset_size=None,
            step_size=None,
            calibration_mm_per_px=1.0,
        )
        logging.info(f"Frame {name} set as reference frame.")
        self._maybe_forward_setup_output()

    def _on_deformed_confirmed(self, names: list[str]) -> None:
        self.session_output.deformed_frames = names
        self.setup_output = replace(self.setup_output, deformed_list=names)
        logging.info("New set of deformed frames confirmed")
        self._maybe_forward_setup_output()

    def _on_folder_ready(self, folder: FolderParser) -> None:
        self._folder = folder
        self._frame_loader = self._folder.pull_image
        self._correlation_presenter.set_frame_loader(self._frame_loader)

    # --------------------------------------------------------------------------
    # Setup Slots
    # --------------------------------------------------------------------------

    def _on_roi_confirmed(self, mask: np.ndarray) -> None:
        self.setup_output = replace(self.setup_output, roi_mask=mask)
        self._maybe_forward_setup_output()

    def _on_grid_confirmed(self, subset_size: int, step_size: int) -> None:
        self.setup_output = replace(
            self.setup_output, subset_size=subset_size, step_size=step_size
        )
        self._maybe_forward_setup_output()

    def _on_scale_confirmed(self, calibration: float) -> None:
        self.setup_output = replace(
            self.setup_output, calibration_mm_per_px=calibration
        )
        self._maybe_forward_setup_output()

    # --------------------------------------------------------------------------
    # Correlation Slots
    # --------------------------------------------------------------------------

    def _maybe_forward_setup_output(self) -> None:
        """Forward the setup output to the correlation presenter once it is complete.

        Until then the correlation presenter is told what is still missing, and
        any setup it held before is withdrawn.
        """
        setup = self.setup_output
        required = {
            "reference image": setup.reference_frame,
            "ROI": setup.roi_mask,
            "grid": setup.subset_size,
            "deformed frames": setup.deformed_list,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            self._correlation_presenter.clear_setup_output(missing)
        else:
            self._correlation_presenter.set_setup_output(setup)

    def _on_correlation_result(
        self, result: CorrelationResult, config: CorrelationConfig
    ) -> None:
        full_csv_path = Path(tempfile.gettempdir()) / (
            f"dic_correlation_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        summary_csv_path = Path(tempfile.gettempdir()) / (
            f"dic_summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )

        export_full_csv(result, full_csv_path)
        export_summary_csv(result, summary_csv_path)

        dialog = CorrelationPreviewDialog(
            n_frames=len(result.frames),
            quantity_options=QUANTITY_LABELS,
            parent=self._view,
        )
        # the presenter computes; the dialog only displays
        _presenter = PreviewPresenter(
            view=dialog,
            result=result,
            setup=self.setup_output,
            frame_loader=self._frame_loader,
            csv_path=full_csv_path,
            correlation_config=config,
            reference_frame_name=self.session_output.reference_frame,
            parent=dialog,
        )
        dialog.exec()

    # --------------------------------------------------------------------------
    # File Loader
    # --------------------------------------------------------------------------

    def load_results_archive(self, path: Path) -> None:
        if self._load_thread is not None:
            return

        # progress dialog while the archive loads
        self._load_progress = QProgressDialog(
            "Loading results...", None, 0, 0, self._view
        )
        self._load_progress.setWindowTitle("Loading")
        self._load_progress.setWindowModality(Qt.WindowModality.WindowModal)
        # The load worker cannot be stopped part-way, so offer no Cancel button.
        self._load_progress.setCancelButton(None)
        self._load_progress.setMinimumDuration(0)
        self._load_progress.show()

        self._load_thread = QThread()
        self._load_worker = _LoadWorker(path)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished_signal.connect(self._on_archive_loaded)
        self._load_worker.error_signal.connect(self._on_archive_load_failed)
        self._load_thread.start()

    def _on_archive_loaded(self, archive: LoadedArchive) -> None:
        if self._load_progress is not None:
            self._load_progress.close()
            self._load_progress = None
        self._teardown_load_thread()

        setup = SetupOutput(
            reference_frame=archive.reference_thumbnail,
            roi_mask=archive.roi_mask,
            subset_size=archive.subset_size,
            step_size=archive.step_size,
            deformed_list=archive.deformed_frames,
            calibration_mm_per_px=archive.calibration_mm_per_px,
        )
        dialog = CorrelationPreviewDialog(
            n_frames=len(archive.result.frames),
            quantity_options=QUANTITY_LABELS,
            parent=self._view,
        )
        _presenter = PreviewPresenter(
            view=dialog,
            result=archive.result,
            setup=setup,
            frame_loader=archive.frame_loader,
            csv_path=archive.csv_path,
            correlation_config=archive.correlation_config,
            reference_frame_name=archive.reference_frame_name,
            parent=dialog,
        )
        dialog.exec()

    def _on_archive_load_failed(self, message: str) -> None:
        if self._load_progress is not None:
            self._load_progress.close()
            self._load_progress = None
        self._teardown_load_thread()
        QMessageBox.critical(self._view, "Load failed", message)
        logging.critical(f"Archive load failed: {message}")

    def _teardown_load_thread(self) -> None:
        if self._load_thread is not None:
            self._load_thread.quit()
            self._load_thread.wait()
        self._load_thread = None
        self._load_worker = None
