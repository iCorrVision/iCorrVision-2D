import logging
import threading
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from components.correlation.panel import CorrelationPanel
from components.contracts import (
    CorrelationConfig,
    CorrelationResult,
    SetupOutput,
)
from components.correlation.orchestration.factory import build_engine


class _CorrelationWorker(QObject):
    """Run `run_correlation` off the GUI thread.

    Built afresh for every run: `setup` and `config` are immutable snapshots of
    one run's inputs, so there is no state worth resetting between runs.
    """

    progress_signal: Signal = Signal(int, int)
    finished_signal: Signal = Signal(object)  # CorrelationResult
    error_signal: Signal = Signal(str)

    def __init__(
        self,
        setup: SetupOutput,
        config: CorrelationConfig,
        frame_loader: Callable[[str], np.ndarray],
        abort_event: threading.Event,
    ) -> None:
        super().__init__()
        self._setup = setup
        self._config = config
        self._frame_loader = frame_loader
        self._abort_event = abort_event

    @Slot()
    def run(self) -> None:
        try:
            logging.info(self._config)
            engine = build_engine(self._config, self._setup, self._frame_loader)
            result = engine.run(
                progress_cb=self.progress_signal.emit,
                abort_check=self._abort_event.is_set,
            )
        except Exception as exc:
            logging.exception("Correlation run failed")
            self.error_signal.emit(str(exc))
            return
        self.finished_signal.emit(result)


class CorrelationPresenter(QObject):
    """Presenter of the correlation panel; MainPresenter supplies two inputs.

    - `set_setup_output(setup)`: a complete `SetupOutput` (reference frame,
      roi_mask, subset_size, step_size, calibration_mm_per_px, deformed_list).
      `deformed_list` comes from Session and the rest from Setup, so
      MainPresenter assembles it once both are confirmed
      (`MainPresenter._maybe_forward_setup_output`).
    - `set_frame_loader(loader)`: a callable from image name to array, the
      folder's `pull_image`. A plain method rather than a Slot, since it is
      wired once per session.
    """

    result_ready_signal: Signal = Signal(
        object, object
    )  # CorrelationResult, CorrelationConfig

    def __init__(self, view: CorrelationPanel, parent=None) -> None:
        super().__init__(parent)
        self._view = view
        self._setup: SetupOutput | None = None
        self._frame_loader: Callable[[str], np.ndarray] | None = None
        self._thread: QThread | None = None
        self._worker: _CorrelationWorker | None = None
        self._abort_event = threading.Event()
        self._last_config: CorrelationConfig | None = None

        self._view.run_requested_signal.connect(self._on_run_requested)
        self._view.abort_requested_signal.connect(self._on_abort_requested)

    # ------------------------------------------------------------------
    # Setup/Session -> Correlation handoff
    # ------------------------------------------------------------------

    @Slot(object)
    def set_setup_output(self, setup: SetupOutput) -> None:
        self._setup = setup
        self._view.set_setup_missing([])
        self._view.set_minimum_search_size(setup.subset_size)
        self._view.set_grid_limits(setup.subset_size, setup.step_size)

    def clear_setup_output(self, missing: list[str]) -> None:
        """Withdraw the setup until the listed items are confirmed."""
        self._setup = None
        self._view.set_setup_missing(missing)

    def set_frame_loader(self, loader: Callable[[str], np.ndarray]) -> None:
        self._frame_loader = loader

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_run_requested(self, config: CorrelationConfig) -> None:
        if self._setup is None:
            logging.warning("Correlation requested before setup was confirmed.")
            return
        if self._frame_loader is None:
            logging.warning("Correlation requested before a frame loader was provided.")
            return
        if self._thread is not None:
            return  # a run is already in progress

        self._last_config = config
        self._abort_event.clear()
        self._thread = QThread()
        self._worker = _CorrelationWorker(
            self._setup, config, self._frame_loader, self._abort_event
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress_signal.connect(self._view.set_progress)
        self._worker.finished_signal.connect(self._on_run_finished)
        self._worker.error_signal.connect(self._on_run_failed)

        self._view.set_running(True)
        self._thread.start()

    @Slot()
    def _on_abort_requested(self) -> None:
        self._abort_event.set()

    @Slot(object)
    def _on_run_finished(self, result: CorrelationResult) -> None:
        self._teardown_thread()
        self._view.set_running(False)
        self._view.set_result_summary(len(result.frames), result.aborted)
        self.result_ready_signal.emit(result, self._last_config)

    @Slot(str)
    def _on_run_failed(self, message: str) -> None:
        self._teardown_thread()
        self._view.set_running(False)
        self._view.set_failure(message)
        logging.critical(f"Correlation failed: {message}")

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
