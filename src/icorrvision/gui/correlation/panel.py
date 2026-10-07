import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QProgressBar,
    QMessageBox,
)

from components.correlation.pipeline.search import (
    MIN_COARSE_SUBSET_PX,
    coarsest_pyramid_subset,
)
from components.correlation.pipeline.tensors import minimum_cloud_radius
from components.contracts import (
    CorrelationConfig,
    InterpolationConfig,
    RefinementConfig,
    SearchConfig,
    StrainConfig,
    RecoveryConfig,
)


class CorrelationPanel(QWidget):
    run_requested_signal: Signal = Signal(object)  # emits a built CorrelationConfig
    abort_requested_signal: Signal = Signal()

    _CRITERION_LABELS: dict[str, str] = {
        "znssd": "Zero-normalized SSD (ZNSSD)",
        "zncc": "Zero-normalized cross-correlation (ZNCC)",
    }
    _REFINEMENT_LABELS: dict[str, str] = {"icgn": "Gradient-based (ICGN)"}
    _CORRELATION_MODE_LABELS: dict[str, str] = {
        "spatial": "Direct (vs. reference frame)",
        "incremental": "Incremental (vs. previous frame)",
    }
    # Tracking follows the correlation mode: nodes stay on the reference grid in
    # direct correlation and follow the material in incremental correlation, the
    # only combinations the engine accepts.
    _TRACKING_FOR_MODE: dict[str, str] = {
        "spatial": "eulerian",
        "incremental": "lagrangian",
    }
    _INTERPOLATION_LABELS: dict[str, str] = {
        "bicubic_spline": "Bicubic spline",
        "biquintic_spline": "Biquintic B-spline",
        "8tap": "8 Tap",
    }

    _SEARCH_LABELS: dict[str, str] = {
        "brute": "Brute Search",
        "pyramid": "Pyramid Search",
    }

    _STRAIN_LABELS: dict[str, str] = {
        "lagrange_q4": "Q4 VSG Green-Lagrange",
        "lagrange_q9": "Q9 VSG Green-Lagrange",
        "lagrange_cloud": "Point Cloud Green-Lagrange",
    }

    _SHAPE_FUNCTION_LABELS: dict[str, str] = {
        "affine": "First Order (Affine)",
        "quadratic": "Second Order (Quadratic)",
    }

    def __init__(self, parent=None) -> None:
        """Stack the settings sections and the run controls."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(self._make_criterion_settings())
        layout.addWidget(self._make_window_settings())
        layout.addWidget(self._make_refinement_settings())
        layout.addWidget(self._make_tracking_settings())
        layout.addWidget(self._make_strain_settings())
        layout.addWidget(self._make_recovery_settings())
        layout.addWidget(self._make_subpixel_settings())
        layout.addWidget(self._make_run_controls())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _make_criterion_settings(self) -> QGroupBox:
        """Build the group for the matching criterion (ZNSSD or ZNCC)."""
        group = QGroupBox("Matching algorithm")
        form = QVBoxLayout(group)

        self._criterion_box = self._make_combo(self._CRITERION_LABELS, default="znssd")
        self._add_row(form, "Criterion", self._criterion_box)

        return group

    def _make_window_settings(self) -> QGroupBox:
        """Build the group for the search method, window, match threshold and workers."""
        group = QGroupBox("Correlation window")
        form = QVBoxLayout(group)

        self._search_method = self._make_combo(self._SEARCH_LABELS)
        self._search_method.currentIndexChanged.connect(self._on_search_method_changed)
        self._add_row(form, "Search mode", self._search_method)

        self._search_threshold = QSpinBox()
        self._search_threshold.setRange(13, 401)
        self._search_threshold.setSingleStep(2)
        self._search_threshold.setValue(61)
        self._search_threshold.setSuffix(" px")
        self._search_threshold.valueChanged.connect(self._enforce_odd_search)
        self._add_row(form, "Search size", self._search_threshold)

        self._pyramid_widget = self._make_pyramid_settings()
        form.addWidget(self._pyramid_widget)

        self._criterion_spin = QDoubleSpinBox()
        self._criterion_spin.setRange(0.0, 1.0)
        self._criterion_spin.setSingleStep(0.05)
        self._criterion_spin.setValue(0.6)
        self._add_row(form, "Match threshold", self._criterion_spin)

        self._n_workers_spin = QSpinBox()
        self._n_workers_spin.setRange(1, 32)
        self._n_workers_spin.setValue(2)
        self._add_row(form, "Worker threads", self._n_workers_spin)

        self._on_search_method_changed()

        return group

    def _make_refinement_settings(self) -> QGroupBox:
        group = QGroupBox("Refinement")
        form = QVBoxLayout(group)

        # Hidden while IC-GN is the only refinement method.
        # self._refinement_method_box = self._make_combo(self._REFINEMENT_LABELS)
        # self._add_row(form, "Refinement method", self._refinement_method_box)

        self._shape_function_box = self._make_combo(self._SHAPE_FUNCTION_LABELS)
        self._add_row(form, "Shape Function", self._shape_function_box)

        return group

    def _make_tracking_settings(self) -> QGroupBox:
        """Build the group for the correlation mode (direct or incremental)."""
        group = QGroupBox("Tracking")
        form = QVBoxLayout(group)

        self._correlation_mode_box = self._make_combo(self._CORRELATION_MODE_LABELS)
        self._add_row(form, "Correlation mode", self._correlation_mode_box)

        return group

    def _make_strain_settings(self) -> QGroupBox:
        group = QGroupBox("Strain")
        form = QVBoxLayout(group)

        self._strain_method_box: QComboBox = self._make_combo(
            self._STRAIN_LABELS, default=StrainConfig().method
        )
        self._strain_method_box.currentIndexChanged.connect(
            self._on_strain_method_changed
        )
        self._add_row(form, "Strain method", self._strain_method_box)

        self._point_cloud_widget = self._make_point_cloud_settings()
        form.addWidget(self._point_cloud_widget)

        self._strain_window_widget = self._make_strain_window_settings()
        form.addWidget(self._strain_window_widget)

        self._on_strain_method_changed()

        return group

    def _make_point_cloud_settings(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Start from the engine's defaults so the GUI and the CLI agree.
        defaults = StrainConfig()

        self._min_neighbors = QSpinBox()
        self._min_neighbors.setRange(3, 100)
        self._min_neighbors.setValue(defaults.min_neighbors)
        self._min_neighbors.valueChanged.connect(self._update_grid_limits)
        self._add_row(layout, "Minimum neighbours", self._min_neighbors)

        self._radius_px = QSpinBox()
        self._radius_px.setRange(1, 100)
        self._radius_px.setValue(int(defaults.radius_px))
        self._add_row(layout, "Radius (px)", self._radius_px)

        return container

    def _make_strain_window_settings(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # A count of grid nodes per side, not pixels; it must be odd.
        self._strain_window = QSpinBox()
        self._strain_window.setRange(3, 99)
        self._strain_window.setSingleStep(2)
        self._strain_window.setValue(StrainConfig().strain_window)
        self._strain_window.valueChanged.connect(self._enforce_odd_strain_window)
        self._add_row(layout, "Strain window (nodes)", self._strain_window)

        return container

    def _make_recovery_settings(self) -> QGroupBox:
        group = QGroupBox("Recovery of lost subsets")
        group.setCheckable(True)
        group.setChecked(False)
        self._recovery_group = group
        form = QVBoxLayout(group)

        self._max_lost_spin = QSpinBox()
        self._max_lost_spin.setRange(1, 10)
        self._max_lost_spin.setValue(2)
        self._max_lost_spin.setSuffix(" frames")
        self._add_row(form, "Max lost frames", self._max_lost_spin)

        self._continuity_tol_spin = QDoubleSpinBox()
        self._continuity_tol_spin.setRange(0.1, 100.0)
        self._continuity_tol_spin.setSingleStep(0.5)
        self._continuity_tol_spin.setValue(5.0)
        self._continuity_tol_spin.setSuffix(" px")
        self._add_row(form, "Continuity tolerance", self._continuity_tol_spin)

        self._fit_radius_spin = QSpinBox()
        self._fit_radius_spin.setRange(1, 10)
        self._fit_radius_spin.setValue(3)
        self._fit_radius_spin.setSuffix(" nodes")
        self._add_row(form, "Neighbour fit radius", self._fit_radius_spin)

        self._min_fit_points_spin = QSpinBox()
        self._min_fit_points_spin.setRange(4, 50)  # a plane has 3 dof
        self._min_fit_points_spin.setValue(6)
        self._add_row(form, "Min. fit neighbours", self._min_fit_points_spin)

        self._recovery_note = QLabel()
        self._recovery_note.setWordWrap(True)
        form.addWidget(self._recovery_note)

        self._correlation_mode_box.currentIndexChanged.connect(
            self._on_correlation_mode_changed
        )
        self._on_correlation_mode_changed()
        return group

    def _make_subpixel_settings(self) -> QGroupBox:
        """Build the group for the image interpolant."""
        group = QGroupBox("Sub-pixel accuracy")
        form = QVBoxLayout(group)

        self._interpolation_box = self._make_combo(
            self._INTERPOLATION_LABELS, default="biquintic_spline"
        )
        self._interpolation_box.setToolTip(
            "Used by the IC-GN refinement to sample the deformed image at sub-pixel positions."
        )
        self._add_row(form, "Image interpolation", self._interpolation_box)
        return group

    def _make_run_controls(self) -> QGroupBox:
        """Build the group for the progress bar and the Run and Abort buttons."""
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._setup_label = QLabel()
        layout.addWidget(self._setup_label)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        button_row = QHBoxLayout()
        self._run_button = QPushButton("Run correlation")
        self._run_button.clicked.connect(self._on_run_clicked)
        button_row.addWidget(self._run_button)

        self._abort_button = QPushButton("Abort")
        self._abort_button.setEnabled(False)
        self._running = False
        self.set_setup_missing(["reference image", "ROI", "grid", "deformed frames"])
        self._abort_button.clicked.connect(self._on_abort_clicked)
        button_row.addWidget(self._abort_button)

        layout.addLayout(button_row)
        return group

    @staticmethod
    def _add_row(form: QVBoxLayout, label: str, widget: QWidget) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(widget)
        form.addLayout(row)

    @staticmethod
    def _make_combo(labels: dict[str, str], default: str | None = None) -> QComboBox:
        box = QComboBox()
        for value, label in labels.items():
            box.addItem(label, userData=value)
        if default is not None:
            box.setCurrentIndex(list(labels).index(default))
        return box

    # ------------------------------------------------------------------
    # Behavior
    # ------------------------------------------------------------------

    def _enforce_odd_window(self, value: int) -> None:
        if value % 2 == 0:
            self._search_threshold.blockSignals(True)
            self._search_threshold.setValue(value + 1)
            self._search_threshold.blockSignals(False)

    def _enforce_odd_search(self, value: int) -> None:
        if value % 2 == 0:
            self._search_threshold.blockSignals(True)
            self._search_threshold.setValue(value + 1)
            self._search_threshold.blockSignals(False)

    def _enforce_odd_strain_window(self, value: int) -> None:
        if value % 2 == 0:
            self._strain_window.blockSignals(True)
            self._strain_window.setValue(value + 1)
            self._strain_window.blockSignals(False)

    def _make_pyramid_settings(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._pyramid_levels_spin = QSpinBox()
        self._pyramid_levels_spin.setRange(1, 6)
        self._pyramid_levels_spin.setValue(3)
        self._add_row(layout, "Pyramid levels", self._pyramid_levels_spin)

        self._pyramid_downsample_spin = QSpinBox()
        self._pyramid_downsample_spin.setRange(2, 4)
        self._pyramid_downsample_spin.setValue(2)
        self._add_row(layout, "Downsample factor", self._pyramid_downsample_spin)
        self._pyramid_downsample_spin.valueChanged.connect(self._update_grid_limits)

        self._pyramid_fine_window_spin = QSpinBox()
        self._pyramid_fine_window_spin.setRange(2, 64)
        self._pyramid_fine_window_spin.setSingleStep(2)
        self._pyramid_fine_window_spin.setValue(8)
        self._pyramid_fine_window_spin.setSuffix(" px")
        self._add_row(layout, "Fine window size", self._pyramid_fine_window_spin)

        return container

    def _on_search_method_changed(self) -> None:
        is_pyramid = self._search_method.currentData() == "pyramid"
        self._pyramid_widget.setVisible(is_pyramid)

    def _on_strain_method_changed(self) -> None:
        is_point_cloud = self._strain_method_box.currentData() == "lagrange_cloud"
        self._point_cloud_widget.setVisible(is_point_cloud)
        self._strain_window_widget.setVisible(not is_point_cloud)
        self._strain_window.setMinimum(
            5 if self._strain_method_box.currentData() == "lagrange_q9" else 3
        )

    def _on_correlation_mode_changed(self) -> None:
        incremental = self._correlation_mode_box.currentData() == "incremental"
        self._recovery_group.setEnabled(incremental)
        self._recovery_note.setText(
            ""
            if incremental
            else "Not used in direct mode: every frame is matched against the "
            "reference, so a subset that fails once is simply retried."
        )

    def _on_abort_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Abort Correlation",
            "Are you sure you want to abort the correlation process?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.abort_requested_signal.emit()

    def _on_run_clicked(self) -> None:
        try:
            config = self.build_config()
        except ValueError as exc:  # a setting the config rejects
            self.set_failure(str(exc))
            return
        self.run_requested_signal.emit(config)

    # ------------------------------------------------------------------
    # Presenter API
    # ------------------------------------------------------------------

    def build_config(self) -> CorrelationConfig:
        return CorrelationConfig(
            search_size=self._search_threshold.value(),
            correlation_mode=self._correlation_mode_box.currentData(),
            tracking_mode=self._TRACKING_FOR_MODE[self._correlation_mode_box.currentData()],
            criterion_name=self._criterion_box.currentData(),
            match_threshold=self._criterion_spin.value(),
            search=SearchConfig(
                method=self._search_method.currentData(),
                pyramid_levels=self._pyramid_levels_spin.value(),
                pyramid_downsample_factor=self._pyramid_downsample_spin.value(),
                pyramid_fine_window=self._pyramid_fine_window_spin.value(),
            ),
            strain=StrainConfig(
                method=self._strain_method_box.currentData(),
                min_neighbors=self._min_neighbors.value(),
                radius_px=self._radius_px.value(),
                strain_window=self._strain_window.value(),
            ),
            n_workers=self._n_workers_spin.value(),
            image_interpolation=InterpolationConfig(
                self._interpolation_box.currentData(),
            ),
            recovery=RecoveryConfig(
                enabled=self._recovery_group.isChecked(),
                max_lost_frames=self._max_lost_spin.value(),
                continuity_tol_px=self._continuity_tol_spin.value(),
                fit_radius=self._fit_radius_spin.value(),
                min_fit_points=self._min_fit_points_spin.value(),
            ),
            refinement=RefinementConfig(
                method="icgn",  # the only refinement method, so not offered in the GUI
                shape_function=self._shape_function_box.currentData(),
            ),
        )

    def set_grid_limits(self, subset_size: int, step_size: int) -> None:
        """Bound the pyramid depth and the cloud radius by the confirmed grid."""
        self._grid = (subset_size, step_size)
        self._update_grid_limits()

    def _update_grid_limits(self) -> None:
        grid = getattr(self, "_grid", None)
        if grid is None:
            return
        subset_size, step_size = grid
        factor = self._pyramid_downsample_spin.value()
        levels = 1
        while levels < 6 and coarsest_pyramid_subset(
            subset_size, levels + 1, factor
        ) >= MIN_COARSE_SUBSET_PX:
            levels += 1
        self._pyramid_levels_spin.setMaximum(levels)
        radius = minimum_cloud_radius(step_size, self._min_neighbors.value())
        self._radius_px.setMinimum(int(math.ceil(radius)))

    def set_minimum_search_size(self, subset_size: int) -> None:
        """Enforce build_engine's rule that search_size exceed subset_size.

        Called once Setup confirms subset_size.
        """
        minimum = (
            subset_size + 2
        )  # smallest odd value strictly greater than subset_size
        self._search_threshold.setMinimum(minimum)
        if self._search_threshold.value() < minimum:
            self._search_threshold.setValue(minimum)

    def set_setup_missing(self, missing: list[str]) -> None:
        """Show which setup items are still unconfirmed; Run needs none missing."""
        self._setup_missing = list(missing)
        self._setup_label.setText(
            "Setup confirmed." if not missing else "To confirm: " + ", ".join(missing)
        )
        self._run_button.setEnabled(not missing and not self._running)

    def set_running(self, running: bool) -> None:
        self._running = running
        self._run_button.setEnabled(not running and not self._setup_missing)
        self._abort_button.setEnabled(running)
        if running:
            self._progress_bar.setValue(0)
            self._status_label.setText("Running...")

    def set_progress(self, frame_index: int, n_frames: int) -> None:
        pct = int(100 * frame_index / n_frames) if n_frames else 0
        self._progress_bar.setValue(pct)
        self._status_label.setText(f"Frame {frame_index} / {n_frames}")

    def set_result_summary(self, n_frames: int, aborted: bool) -> None:
        if aborted:
            self._status_label.setText(f"Aborted after {n_frames} frame(s).")
        else:
            self._status_label.setText(f"Done: {n_frames} frame(s) processed.")

    def set_failure(self, message: str) -> None:
        self._status_label.setText(f"Failed: {message}")
