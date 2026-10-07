from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QGroupBox,
    QSpinBox,
    QLabel,
    QButtonGroup,
    QDoubleSpinBox,
    QComboBox,
    QSizePolicy,
    QSplitter,
)
from PySide6.QtCore import QPointF, Signal, Qt
from icorrvision.gui.setup.roi_drawer import CanvasMode
from icorrvision.gui.setup.frame import SetupFrame
from PySide6.QtGui import QPainterPath
from icorrvision.gui.widgets.utils.utils import make_labeled_toggle


class SetupPanel(QWidget):
    roi_confirmed_signal: Signal = Signal()
    roi_export_signal: Signal = Signal()
    roi_load_signal: Signal = Signal()
    roi_mode_changed_signal: Signal = Signal(object)
    roi_clear_signal: Signal = Signal()
    roi_undo_signal: Signal = Signal()

    _TOOL_LABELS: dict[CanvasMode, str] = {
        CanvasMode.DRAW_RECT: "Rect",
        CanvasMode.DRAW_ELLIPSE: "Ellipse",
        CanvasMode.DRAW_POLYGON: "Polygon",
    }

    grid_commited_signal: Signal = Signal(int, int)
    grid_visuals_signal: Signal = Signal(bool)
    grid_confirmed_signal: Signal = Signal()

    scale_value_signal: Signal = Signal(float, float)
    scale_clear_signal: Signal = Signal()
    scale_draw_signal: Signal = Signal()
    scale_confirmed_signal: Signal = Signal()

    _AVAILABLE_SCALE: dict[str, float] = {
        "m": 1000.0,
        "dm": 100.0,
        "cm": 10.0,
        "mm": 1.0,
        "μm": 0.001,
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.frame: SetupFrame = SetupFrame()
        self.frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.tool_buttons: dict[CanvasMode, QPushButton] = {}
        self._subset_spin: QSpinBox
        self._step_spin: QSpinBox

        setup_layout = QVBoxLayout()
        setup_layout.addWidget(self._make_roi_settings())
        setup_layout.addWidget(self._make_grid_parameters())
        setup_layout.addWidget(self._make_scale_paramaters())
        setup_widget = QWidget()
        setup_widget.setLayout(setup_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.frame)
        splitter.addWidget(setup_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(splitter)

    def _make_roi_settings(self) -> QGroupBox:
        group = QGroupBox("ROI Parameters")
        layout = QVBoxLayout(group)

        tool_row = QHBoxLayout()
        tool_group = QButtonGroup(self)
        tool_group.setExclusive(False)

        for mode, label in self._TOOL_LABELS.items():
            btn = QPushButton(label)
            btn.setCheckable(True)
            tool_group.addButton(btn)
            self.tool_buttons[mode] = btn
            btn.clicked.connect(
                lambda checked, m=mode: self._on_tool_clicked(m, checked)
            )
            tool_row.addWidget(btn)

        layout.addLayout(tool_row)

        action_row = QHBoxLayout()
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.roi_clear_signal)
        action_row.addWidget(clear_button)

        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self.roi_undo_signal)
        action_row.addWidget(undo_button)
        layout.addLayout(action_row)

        accept_row = QHBoxLayout()

        confirm_button = QPushButton("confirm ROI")
        confirm_button.clicked.connect(self.roi_confirmed_signal)
        accept_row.addWidget(confirm_button)

        export_button = QPushButton("export ROI")
        export_button.clicked.connect(self.roi_export_signal)
        accept_row.addWidget(export_button)

        # TODO: FUTURE WORK
        # load_button = QPushButton("load ROI")
        # load_button.clicked.connect(self.roi_load_signal)
        # accept_row.addWidget(load_button)

        layout.addLayout(accept_row)

        hint = QLabel("Shift = add \u00b7 Alt = subtract \u00b7 Shift+Alt = intersect")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        return group

    def _on_tool_clicked(self, mode: CanvasMode, checked: bool) -> None:
        if checked:
            for m, btn in self.tool_buttons.items():
                if m is not mode:
                    btn.setChecked(False)
            self.roi_mode_changed_signal.emit(mode)
        else:
            self.roi_mode_changed_signal.emit(CanvasMode.IDLE)

    def _make_grid_parameters(self) -> QGroupBox:
        group = QGroupBox("Grid Parameters")
        layout = QVBoxLayout(group)

        self._subset_spin = QSpinBox()
        self._subset_spin.setRange(11, 201)
        self._subset_spin.setSingleStep(2)
        self._subset_spin.setValue(31)
        self._subset_spin.setSuffix(" px")

        self._step_spin = QSpinBox()
        self._step_spin.setRange(1, 200)
        self._step_spin.setValue(10)
        self._step_spin.setSuffix(" px")

        for label_text, spin in [
            ("Subset size", self._subset_spin),
            ("Step size", self._step_spin),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            row.addWidget(spin)
            layout.addLayout(row)

        self._subset_spin.valueChanged.connect(self._on_grid_commited)
        self._step_spin.valueChanged.connect(self._on_grid_commited)

        toggle_grid = make_labeled_toggle(
            self,
            "grid visuals off",
            "grid visuals on",
            True,
            self.grid_visuals_signal,
        )
        layout.addWidget(toggle_grid)

        confirm_button = QPushButton("confirm grid")
        confirm_button.clicked.connect(self._on_grid_commited)
        confirm_button.clicked.connect(self.grid_confirmed_signal)
        layout.addWidget(confirm_button)

        return group

    def _make_scale_paramaters(self) -> QGroupBox:
        group = QGroupBox("Scale Parameters")

        layout = QVBoxLayout(group)

        buttons_layout = QHBoxLayout()
        layout.addLayout(buttons_layout)

        draw_scale_button = QPushButton("Draw")
        draw_scale_button.clicked.connect(self.scale_draw_signal)
        buttons_layout.addWidget(draw_scale_button)

        clear_scale_button = QPushButton("Clear")
        clear_scale_button.clicked.connect(self.scale_clear_signal)
        buttons_layout.addWidget(clear_scale_button)

        scale_size_layout = QHBoxLayout()
        layout.addLayout(scale_size_layout)

        # Known length of the drawn line. A spin box follows the locale's decimal
        # separator, and every change of value or unit is sent at once, so the
        # confirmed scale always uses what is on screen.
        self._scale_length = QDoubleSpinBox()
        self._scale_length.setRange(0.001, 1e6)
        self._scale_length.setDecimals(3)
        self._scale_length.setValue(1.0)
        self._scale_length.setToolTip("Known length of the drawn line")
        scale_size_layout.addWidget(self._scale_length)

        self._scale_unit = QComboBox()
        self._scale_unit.addItems(list(self._AVAILABLE_SCALE.keys()))
        self._scale_unit.setCurrentText("mm")
        scale_size_layout.addWidget(self._scale_unit)

        self._scale_length.valueChanged.connect(self._emit_scale_value)
        self._scale_unit.currentTextChanged.connect(self._emit_scale_value)

        confirm_button = QPushButton("confirm scale")
        confirm_button.clicked.connect(self.scale_confirmed_signal)
        layout.addWidget(confirm_button)

        self._scale_result = QLabel("Scale: not set (1 mm/px)")
        layout.addWidget(self._scale_result)

        return group

    # --------------------------------------------------------------------------
    # Confirm handlers
    # --------------------------------------------------------------------------

    def _on_grid_commited(self) -> None:
        subset = self._subset_spin.value()
        if subset % 2 == 0:  # odd signals only
            self._subset_spin.blockSignals(True)
            self._subset_spin.setValue(subset + 1)
            self._subset_spin.blockSignals(False)
            subset += 1
        self.grid_commited_signal.emit(subset, self._step_spin.value())

    # --------------------------------------------------------------------------
    # Presenter API
    # --------------------------------------------------------------------------

    def set_drawing_mode(self, mode: CanvasMode) -> None:
        self.frame.set_mode(mode)

    def set_roi_path(self, path: QPainterPath) -> None:
        self.frame.set_roi_path(path)

    def _emit_scale_value(self) -> None:
        self.scale_value_signal.emit(
            self._scale_length.value(),
            self._AVAILABLE_SCALE[self._scale_unit.currentText()],
        )

    def show_calibration(self, mm_per_px: float) -> None:
        """Show the confirmed scale."""
        self._scale_result.setText(f"Scale: {mm_per_px:.6g} mm/px")

    def set_scale_line(self, line: tuple[QPointF, QPointF] | None) -> None:
        self.frame.set_scale_line(line)

    def set_grid_preview(
        self,
        nodes: list[QPointF],
        anchor: QPointF | None,
        half_extent: tuple[float, float] | None,
        visible: bool,
    ) -> None:
        self.frame.set_grid_preview(nodes, anchor, half_extent, visible)
