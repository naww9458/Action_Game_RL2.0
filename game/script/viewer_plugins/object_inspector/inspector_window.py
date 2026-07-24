from __future__ import annotations

from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .control_bridge import BodyState, JointState, PinnedBodyFields, PinnedJointFields
from .metadata import BodyParamSpec, JointParamSpec, ObjectInspectorSpec, PlayerActionSpec

if TYPE_CHECKING:
    from .control_bridge import ControlBridge
    from .metadata import InspectorCatalog
    from script.viewer_controls import SimulationControl, ViewerControlsConfig


class ParamRow(QWidget):
    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
        pin_enabled: bool = True,
        always_pinned: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._always_pinned = always_pinned
        self.pin_box = QCheckBox("Pin") if pin_enabled else None

        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(4)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self._min = minimum
        self._max = maximum
        self._updating = False

        self.spin.valueChanged.connect(self._spin_to_slider)
        self.slider.valueChanged.connect(self._slider_to_spin)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label, minimumWidth=80))
        layout.addWidget(self.spin, stretch=1)
        layout.addWidget(self.slider, stretch=2)
        if self.pin_box is not None:
            layout.addWidget(self.pin_box)

        self._sync_slider_from_spin()

    def _sync_slider_from_spin(self):
        value = float(self.spin.value())
        t = 0.0 if self._max == self._min else (value - self._min) / (self._max - self._min)
        self.slider.blockSignals(True)
        self.slider.setValue(int(max(0, min(1000, round(t * 1000)))))
        self.slider.blockSignals(False)

    def _spin_to_slider(self, value: float):
        if self._updating:
            return
        self._updating = True
        self._sync_slider_from_spin()
        self._updating = False

    def _slider_to_spin(self, value: int):
        if self._updating:
            return
        self._updating = True
        t = value / 1000.0
        self.spin.setValue(self._min + t * (self._max - self._min))
        self._updating = False

    def set_value(self, value: float):
        self._updating = True
        self.spin.setValue(value)
        self._sync_slider_from_spin()
        self._updating = False

    def value(self) -> float:
        return float(self.spin.value())

    def is_pinned(self) -> bool:
        if self._always_pinned:
            return True
        return self.pin_box.isChecked() if self.pin_box is not None else False

    def set_pinned(self, pinned: bool):
        if self.pin_box is not None:
            self.pin_box.setChecked(pinned)

    def should_sync_from_sim(self) -> bool:
        if self.is_pinned():
            return False
        return not (self.spin.hasFocus() or self.slider.hasFocus())

    def set_enabled(self, enabled: bool):
        self.spin.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        if self.pin_box is not None:
            self.pin_box.setEnabled(enabled)


class InspectorWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Object Inspector")
        self.resize(520, 720)
        self._spec: Optional[ObjectInspectorSpec] = None
        self._world_idx = 0

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        header = QGroupBox("Target")
        header_layout = QFormLayout(header)
        self.label_combo = QComboBox()
        self.label_combo.currentTextChanged.connect(self._on_label_changed)
        self.world_spin = QSpinBox()
        self.world_spin.setMinimum(0)
        self.world_spin.valueChanged.connect(self._on_world_changed)
        self.info_label = QLabel("")
        header_layout.addRow("Object", self.label_combo)
        header_layout.addRow("Env", self.world_spin)
        header_layout.addRow("Info", self.info_label)
        root.addWidget(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self.body_tab = QWidget()
        body_layout = QVBoxLayout(self.body_tab)
        self.body_selector = QComboBox()
        self.body_selector.currentIndexChanged.connect(self._on_body_selection_changed)
        body_layout.addWidget(self.body_selector)

        lock_row = QHBoxLayout()
        self.lock_pos_btn = QPushButton("Lock Position")
        self.lock_rot_btn = QPushButton("Lock Rotation")
        self.lock_lin_vel_btn = QPushButton("Lock Lin Vel")
        self.lock_ang_vel_btn = QPushButton("Lock Ang Vel")
        for btn in (
            self.lock_pos_btn,
            self.lock_rot_btn,
            self.lock_lin_vel_btn,
            self.lock_ang_vel_btn,
        ):
            btn.setCheckable(True)
        self.lock_pos_btn.clicked.connect(lambda: self._toggle_body_group_lock("pos"))
        self.lock_rot_btn.clicked.connect(lambda: self._toggle_body_group_lock("quat"))
        self.lock_lin_vel_btn.clicked.connect(lambda: self._toggle_body_group_lock("lin_vel"))
        self.lock_ang_vel_btn.clicked.connect(lambda: self._toggle_body_group_lock("ang_vel"))
        lock_row.addWidget(self.lock_pos_btn)
        lock_row.addWidget(self.lock_rot_btn)
        lock_row.addWidget(self.lock_lin_vel_btn)
        lock_row.addWidget(self.lock_ang_vel_btn)
        body_layout.addLayout(lock_row)

        self.body_scroll = QScrollArea()
        self.body_scroll.setWidgetResizable(True)
        self.body_panel = QWidget()
        self.body_form = QFormLayout(self.body_panel)
        self.body_scroll.setWidget(self.body_panel)
        body_layout.addWidget(self.body_scroll, stretch=1)
        self.tabs.addTab(self.body_tab, "Bodies")

        self.joint_tab = QWidget()
        joint_layout = QVBoxLayout(self.joint_tab)
        self.joint_selector = QComboBox()
        self.joint_selector.currentIndexChanged.connect(self._on_joint_selection_changed)
        joint_layout.addWidget(self.joint_selector)
        self.joint_scroll = QScrollArea()
        self.joint_scroll.setWidgetResizable(True)
        self.joint_panel = QWidget()
        self.joint_form = QFormLayout(self.joint_panel)
        self.joint_scroll.setWidget(self.joint_panel)
        joint_layout.addWidget(self.joint_scroll, stretch=1)
        self.tabs.addTab(self.joint_tab, "Joints")

        self.rl_action_tab = QWidget()
        rl_layout = QVBoxLayout(self.rl_action_tab)
        self.rl_action_enable_checkbox = QCheckBox("Enable RL Action override")
        self.rl_action_enable_checkbox.setChecked(False)
        self.rl_action_enable_checkbox.toggled.connect(self._on_rl_action_enable_toggled)
        rl_layout.addWidget(self.rl_action_enable_checkbox)
        rl_btn_row = QHBoxLayout()
        self.rl_action_zero_btn = QPushButton("Clear to Zero")
        self.rl_action_zero_btn.clicked.connect(self._clear_rl_action_values)
        rl_btn_row.addWidget(self.rl_action_zero_btn)
        rl_btn_row.addStretch()
        rl_layout.addLayout(rl_btn_row)
        self.rl_action_scroll = QScrollArea()
        self.rl_action_scroll.setWidgetResizable(True)
        self.rl_action_panel = QWidget()
        self.rl_action_layout = QVBoxLayout(self.rl_action_panel)
        self.rl_action_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.rl_action_scroll.setWidget(self.rl_action_panel)
        rl_layout.addWidget(self.rl_action_scroll, stretch=1)
        self.tabs.addTab(self.rl_action_tab, "RL Action")

        self.commands_tab = QWidget()
        commands_layout = QVBoxLayout(self.commands_tab)
        commands_btn_row = QHBoxLayout()
        self.commands_zero_btn = QPushButton("Clear to Zero")
        self.commands_zero_btn.clicked.connect(self._clear_command_values)
        commands_btn_row.addWidget(self.commands_zero_btn)
        commands_btn_row.addStretch()
        commands_layout.addLayout(commands_btn_row)
        self.commands_scroll = QScrollArea()
        self.commands_scroll.setWidgetResizable(True)
        self.commands_panel = QWidget()
        self.commands_layout = QVBoxLayout(self.commands_panel)
        self.commands_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.commands_scroll.setWidget(self.commands_panel)
        commands_layout.addWidget(self.commands_scroll, stretch=1)
        self.tabs.addTab(self.commands_tab, "Commands")

        self.controls_tab = QWidget()
        controls_layout = QVBoxLayout(self.controls_tab)
        sim_group = QGroupBox("Simulation")
        sim_form = QFormLayout(sim_group)
        self.auto_reset_checkbox = QCheckBox("Auto reset on env end")
        self.manual_reset_checkbox = QCheckBox("Enable manual reset")
        self.paused_status_label = QLabel("Paused: No")
        self.game_over_status_label = QLabel("Game over: No")
        sim_form.addRow(self.auto_reset_checkbox)
        sim_form.addRow(self.manual_reset_checkbox)
        sim_form.addRow("Status", self.paused_status_label)
        sim_form.addRow("", self.game_over_status_label)

        self.gravity_x = QDoubleSpinBox()
        self.gravity_y = QDoubleSpinBox()
        self.gravity_z = QDoubleSpinBox()
        for spin in (self.gravity_x, self.gravity_y, self.gravity_z):
            spin.setRange(-500.0, 500.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._on_gravity_changed)
        gravity_row = QHBoxLayout()
        gravity_row.addWidget(self.gravity_x)
        gravity_row.addWidget(self.gravity_y)
        gravity_row.addWidget(self.gravity_z)
        gravity_widget = QWidget()
        gravity_widget.setLayout(gravity_row)
        sim_form.addRow("Gravity XYZ", gravity_widget)

        controls_layout.addWidget(sim_group)

        self.controls_help_scroll = QScrollArea()
        self.controls_help_scroll.setWidgetResizable(True)
        self.controls_help_panel = QWidget()
        self.controls_help_layout = QVBoxLayout(self.controls_help_panel)
        self.controls_help_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.controls_help_scroll.setWidget(self.controls_help_panel)
        controls_layout.addWidget(self.controls_help_scroll, stretch=1)
        self.tabs.addTab(self.controls_tab, "Controls")

        btn_row = QHBoxLayout()
        self.apply_force_btn = QPushButton("Apply Force")
        self.apply_force_btn.clicked.connect(lambda: self._apply_impulse(force=True, torque=False))
        self.apply_torque_btn = QPushButton("Apply Torque")
        self.apply_torque_btn.clicked.connect(lambda: self._apply_impulse(force=False, torque=True))
        self.apply_joint_torque_btn = QPushButton("Apply Joint Torque")
        self.apply_joint_torque_btn.clicked.connect(self._apply_joint_torque)
        btn_row.addWidget(self.apply_force_btn)
        btn_row.addWidget(self.apply_torque_btn)
        btn_row.addWidget(self.apply_joint_torque_btn)
        root.addLayout(btn_row)

        self._body_rows: Dict[str, ParamRow] = {}
        self._joint_rows: Dict[str, ParamRow] = {}
        self._rl_action_rows: Dict[int, ParamRow] = {}
        self._command_rows: Dict[int, ParamRow] = {}
        self._body_field_pins: Dict[str, Dict[str, bool]] = {}
        self._joint_field_pins: Dict[str, Dict[str, bool]] = {}
        self._body_field_values: Dict[str, Dict[str, float]] = {}
        self._joint_field_values: Dict[str, Dict[str, float]] = {}
        self._rl_action_values: Dict[str, Dict[int, float]] = {}
        self._command_values: Dict[str, Dict[int, float]] = {}
        self._command_pins: Dict[str, Dict[int, bool]] = {}
        self._command_labels: List[str] = []
        self._active_body_storage_key: Optional[str] = None
        self._active_joint_storage_key: Optional[str] = None
        self._player_action: Optional[PlayerActionSpec] = None
        self._impulse_callbacks: List[Callable] = []
        self._simulation_control: Optional["SimulationControl"] = None
        self._gravity_changed_cb: Optional[Callable[[float, float, float], None]] = None
        self._controls_syncing = False

    def setup_controls_tab(
        self,
        simulation_control: "SimulationControl",
        viewer_controls_cfg: "ViewerControlsConfig",
        gameplay_bindings: Optional[List[dict]] = None,
        initial_gravity: Optional[List[float]] = None,
    ):
        from script.viewer_controls import format_keys

        self._simulation_control = simulation_control
        self._controls_syncing = True
        self.auto_reset_checkbox.setChecked(simulation_control.auto_reset_on_env_end)
        self.manual_reset_checkbox.setChecked(simulation_control.manual_reset_enabled)
        if initial_gravity is not None and len(initial_gravity) >= 3:
            self.gravity_x.setValue(float(initial_gravity[0]))
            self.gravity_y.setValue(float(initial_gravity[1]))
            self.gravity_z.setValue(float(initial_gravity[2]))
        self._controls_syncing = False

        self.auto_reset_checkbox.toggled.connect(self._on_auto_reset_toggled)
        self.manual_reset_checkbox.toggled.connect(self._on_manual_reset_toggled)

        while self.controls_help_layout.count():
            item = self.controls_help_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for category, bindings in viewer_controls_cfg.bindings_by_category().items():
            group = QGroupBox(viewer_controls_cfg.category_title(category))
            form = QFormLayout(group)
            for binding in bindings:
                label = format_keys(binding.keys)
                description = binding.description
                if binding.context == "free_camera":
                    description = f"{description} (free camera only)"
                form.addRow(label, QLabel(description))
            self.controls_help_layout.addWidget(group)

        if gameplay_bindings:
            group = QGroupBox("Gameplay")
            form = QFormLayout(group)
            for item in gameplay_bindings:
                form.addRow(item["keys"], QLabel(item["description"]))
            self.controls_help_layout.addWidget(group)

        self.sync_controls_status(game_over=False)

    def _on_auto_reset_toggled(self, checked: bool):
        if self._controls_syncing or self._simulation_control is None:
            return
        self._simulation_control.auto_reset_on_env_end = checked

    def _on_manual_reset_toggled(self, checked: bool):
        if self._controls_syncing or self._simulation_control is None:
            return
        self._simulation_control.manual_reset_enabled = checked

    def _on_gravity_changed(self):
        if self._controls_syncing or self._gravity_changed_cb is None:
            return
        self._gravity_changed_cb(
            float(self.gravity_x.value()),
            float(self.gravity_y.value()),
            float(self.gravity_z.value()),
        )

    def set_gravity_changed_callback(self, cb: Callable[[float, float, float], None]):
        self._gravity_changed_cb = cb

    def set_gravity_values(self, gravity: List[float]):
        if len(gravity) < 3:
            return
        
        if any(spin.hasFocus() for spin in (self.gravity_x, self.gravity_y, self.gravity_z)):
            return
        self._controls_syncing = True
        self.gravity_x.setValue(float(gravity[0]))
        self.gravity_y.setValue(float(gravity[1]))
        self.gravity_z.setValue(float(gravity[2]))
        self._controls_syncing = False

    def get_gravity_values(self) -> List[float]:
        return [
            float(self.gravity_x.value()),
            float(self.gravity_y.value()),
            float(self.gravity_z.value()),
        ]

    def sync_controls_status(self, game_over: bool = False):
        if self._simulation_control is None:
            return
        self._controls_syncing = True
        self.auto_reset_checkbox.setChecked(self._simulation_control.auto_reset_on_env_end)
        self.manual_reset_checkbox.setChecked(self._simulation_control.manual_reset_enabled)
        self._controls_syncing = False
        self.paused_status_label.setText(
            f"Paused: {'Yes' if self._simulation_control.paused else 'No'}"
        )
        self.game_over_status_label.setText(f"Game over: {'Yes' if game_over else 'No'}")

    def closeEvent(self, event):
        event.ignore()
        self.hide()

    def set_labels(self, labels: List[str]):
        current = self.label_combo.currentText()
        self.label_combo.blockSignals(True)
        self.label_combo.clear()
        self.label_combo.addItems(labels)
        if current in labels:
            self.label_combo.setCurrentText(current)
        self.label_combo.blockSignals(False)

    def set_spec(self, spec: Optional[ObjectInspectorSpec], world_idx: int = 0):
        same_catalog = (
            self._spec is not None
            and spec is not None
            and self._spec.catalog_key == spec.catalog_key
        )
        prev_body_idx = self.body_selector.currentIndex() if same_catalog else -1
        prev_joint_idx = self.joint_selector.currentIndex() if same_catalog else -1

        self._save_body_field_values()
        self._save_joint_field_values()
        self._save_rl_action_values()
        self._save_command_values()

        self._spec = spec
        self._world_idx = world_idx
        if spec is None:
            self.info_label.setText("No object selected")
            self._player_action = None
            self.tabs.setTabEnabled(2, False)
            self.tabs.setTabEnabled(3, len(self._command_labels) > 0)
            return
        self.world_spin.blockSignals(True)
        self.world_spin.setMaximum(max(0, 999))
        self.world_spin.setValue(world_idx)
        self.world_spin.blockSignals(False)
        self.info_label.setText(
            f"{spec.catalog_key} | role={spec.local_role_idx} | view={spec.view_obj_idx} | "
            f"kind={spec.body_kind} | env={world_idx}"
        )
        self.body_selector.blockSignals(True)
        self.body_selector.clear()
        for b in spec.bodies:
            self.body_selector.addItem(b.display_name, b)
        if same_catalog and 0 <= prev_body_idx < self.body_selector.count():
            self.body_selector.setCurrentIndex(prev_body_idx)
        self.body_selector.blockSignals(False)
        self.joint_selector.blockSignals(True)
        self.joint_selector.clear()
        for j in spec.joints:
            self.joint_selector.addItem(j.display_name, j)
        if same_catalog and 0 <= prev_joint_idx < self.joint_selector.count():
            self.joint_selector.setCurrentIndex(prev_joint_idx)
        self.joint_selector.blockSignals(False)
        self.tabs.setTabEnabled(1, len(spec.joints) > 0)
        self._player_action = spec.player_action
        self.tabs.setTabEnabled(2, spec.player_action is not None)
        self.tabs.setTabEnabled(3, len(self._command_labels) > 0)
        self._rebuild_body_panel()
        self._rebuild_joint_panel()
        self._rebuild_rl_action_panel()
        self._rebuild_commands_panel()

    def _clear_layout(self, layout: QFormLayout):
        while layout.rowCount():
            layout.removeRow(0)

    def _on_body_selection_changed(self):
        self._save_body_field_values()
        self._rebuild_body_panel()
        self._update_apply_buttons()

    def _on_joint_selection_changed(self):
        self._save_joint_field_values()
        self._rebuild_joint_panel()

    def _update_apply_buttons(self):
        self.apply_torque_btn.setVisible(True)

    def _body_field_defs(self, body: BodyParamSpec):
        defs = [
            ("pos_x", "Pos X", -50.0, 50.0, 0.01, True, body.can_edit_position),
            ("pos_y", "Pos Y", -50.0, 50.0, 0.01, True, body.can_edit_position),
            ("pos_z", "Pos Z", -5.0, 5.0, 0.01, True, body.can_edit_position),
            ("quat_w", "Quat W", -1.0, 1.0, 0.01, True, body.can_edit_orientation),
            ("quat_x", "Quat X", -1.0, 1.0, 0.01, True, body.can_edit_orientation),
            ("quat_y", "Quat Y", -1.0, 1.0, 0.01, True, body.can_edit_orientation),
            ("quat_z", "Quat Z", -1.0, 1.0, 0.01, True, body.can_edit_orientation),
            ("vel_x", "Vel X", -20.0, 20.0, 0.05, True, True),
            ("vel_y", "Vel Y", -20.0, 20.0, 0.05, True, True),
            ("vel_z", "Vel Z", -20.0, 20.0, 0.05, True, True),
            ("omega_x", "Omega X", -20.0, 20.0, 0.05, True, True),
            ("omega_y", "Omega Y", -20.0, 20.0, 0.05, True, True),
            ("omega_z", "Omega Z", -20.0, 20.0, 0.05, True, True),
            ("force_x", "Force X", -500.0, 500.0, 1.0, False, True),
            ("force_y", "Force Y", -500.0, 500.0, 1.0, False, True),
            ("force_z", "Force Z", -500.0, 500.0, 1.0, False, True),
            ("torque_x", "Torque X", -500.0, 500.0, 1.0, False, True),
            ("torque_y", "Torque Y", -500.0, 500.0, 1.0, False, True),
            ("torque_z", "Torque Z", -500.0, 500.0, 1.0, False, True),
        ]
        return [item for item in defs if item[6]]

    _BODY_LOCK_GROUPS = {
        "pos": ("pos_x", "pos_y", "pos_z"),
        "quat": ("quat_w", "quat_x", "quat_y", "quat_z"),
        "lin_vel": ("vel_x", "vel_y", "vel_z"),
        "ang_vel": ("omega_x", "omega_y", "omega_z"),
    }

    @staticmethod
    def _parse_env_suffix_storage_key(key: str) -> Optional[tuple[str, int, str]]:
        env_marker = "|env"
        env_idx = key.rfind(env_marker)
        if env_idx < 0:
            return None
        catalog_key = key[:env_idx]
        rest = key[env_idx + 1 :]
        parts = rest.split("|", 1)
        if len(parts) != 2:
            return None
        env_part, suffix = parts
        if not env_part.startswith("env"):
            return None
        try:
            world_idx = int(env_part[3:])
        except ValueError:
            return None
        return catalog_key, world_idx, suffix

    @staticmethod
    def _parse_env_only_storage_key(key: str) -> Optional[tuple[str, int]]:
        env_marker = "|env"
        env_idx = key.rfind(env_marker)
        if env_idx < 0:
            return None
        catalog_key = key[:env_idx]
        env_part = key[env_idx + 1 :]
        if not env_part.startswith("env"):
            return None
        try:
            world_idx = int(env_part[3:])
        except ValueError:
            return None
        return catalog_key, world_idx

    @staticmethod
    def _body_state_from_values(values: Dict[str, float]) -> BodyState:
        def _val(field_key: str, default: float = 0.0) -> float:
            return float(values.get(field_key, default))

        return BodyState(
            pos=[_val("pos_x"), _val("pos_y"), _val("pos_z")],
            quat=[_val("quat_x"), _val("quat_y"), _val("quat_z"), _val("quat_w", 1.0)],
            lin_vel=[_val("vel_x"), _val("vel_y"), _val("vel_z")],
            ang_vel=[_val("omega_x"), _val("omega_y"), _val("omega_z")],
        )

    @staticmethod
    def _pinned_body_from_pins(pins: Dict[str, bool]) -> PinnedBodyFields:
        def _group(*field_keys: str) -> bool:
            return any(pins.get(field_key, False) for field_key in field_keys)

        return PinnedBodyFields(
            pos=_group("pos_x", "pos_y", "pos_z"),
            quat=_group("quat_w", "quat_x", "quat_y", "quat_z"),
            lin_vel=_group("vel_x", "vel_y", "vel_z"),
            ang_vel=_group("omega_x", "omega_y", "omega_z"),
        )

    @staticmethod
    def _joint_state_from_values(values: Dict[str, float]) -> JointState:
        return JointState(
            q=float(values.get("q", 0.0)),
            qd=float(values.get("qd", 0.0)),
            torque=float(values.get("torque", 0.0)),
        )

    @staticmethod
    def _pinned_joint_from_pins(pins: Dict[str, bool]) -> PinnedJointFields:
        return PinnedJointFields(
            q=pins.get("q", False),
            qd=pins.get("qd", False),
        )

    def set_command_labels(self, labels: List[str]):
        self._command_labels = list(labels)
        self.tabs.setTabEnabled(3, len(self._command_labels) > 0)
        self._rebuild_commands_panel()

    def flush_pinned_storage(self):
        self._save_body_field_values()
        self._save_joint_field_values()
        self._save_rl_action_values()
        self._save_command_values()

    def iter_stored_body_pins(self, catalog: "InspectorCatalog"):
        for key, pins in self._body_field_pins.items():
            pinned = self._pinned_body_from_pins(pins)
            if not (pinned.pos or pinned.quat or pinned.lin_vel or pinned.ang_vel):
                continue
            parsed = self._parse_env_suffix_storage_key(key)
            if parsed is None:
                continue
            catalog_key, world_idx, body_name = parsed
            spec = catalog.get_by_catalog_key(catalog_key)
            if spec is None:
                continue
            body = next((b for b in spec.bodies if b.display_name == body_name), None)
            if body is None:
                continue
            values = self._body_field_values.get(key, {})
            yield spec, world_idx, body, self._body_state_from_values(values), pinned

    def iter_stored_joint_pins(self, catalog: "InspectorCatalog"):
        for key, pins in self._joint_field_pins.items():
            pinned = self._pinned_joint_from_pins(pins)
            if not (pinned.q or pinned.qd):
                continue
            parsed = self._parse_env_suffix_storage_key(key)
            if parsed is None:
                continue
            catalog_key, world_idx, joint_name = parsed
            spec = catalog.get_by_catalog_key(catalog_key)
            if spec is None:
                continue
            joint = next((j for j in spec.joints if j.display_name == joint_name), None)
            if joint is None:
                continue
            values = self._joint_field_values.get(key, {})
            yield spec, world_idx, joint, self._joint_state_from_values(values), pinned

    def iter_stored_rl_actions(self, catalog: "InspectorCatalog"):
        for key, values in self._rl_action_values.items():
            if not values:
                continue
            parsed = self._parse_env_only_storage_key(key)
            if parsed is None:
                continue
            catalog_key, world_idx = parsed
            spec = catalog.get_by_catalog_key(catalog_key)
            if spec is None or spec.player_action is None:
                continue
            yield spec, world_idx, values

    def iter_stored_command_pins(self):
        for key, pins in self._command_pins.items():
            pinned_dims = [idx for idx, pinned in pins.items() if pinned]
            if not pinned_dims:
                continue
            world_idx = self._parse_command_storage_key(key)
            if world_idx is None:
                continue
            values = self._command_values.get(key, {})
            yield world_idx, values, pinned_dims

    @staticmethod
    def _parse_command_storage_key(key: str) -> Optional[int]:
        if not key.startswith("env"):
            return None
        try:
            return int(key[3:])
        except ValueError:
            return None

    def _env_storage_prefix(self) -> Optional[str]:
        if self._spec is None:
            return None
        return f"{self._spec.catalog_key}|env{self._world_idx}"

    def _body_pin_storage_key(self) -> Optional[str]:
        prefix = self._env_storage_prefix()
        body = self.current_body()
        if prefix is None or body is None:
            return None
        return f"{prefix}|{body.display_name}"

    def _joint_pin_storage_key(self) -> Optional[str]:
        prefix = self._env_storage_prefix()
        joint = self.current_joint()
        if prefix is None or joint is None:
            return None
        return f"{prefix}|{joint.display_name}"

    def _rl_value_storage_key(self) -> Optional[str]:
        return self._env_storage_prefix()

    def _command_storage_key(self) -> str:
        return f"env{self._world_idx}"

    def _save_command_values(self):
        key = self._command_storage_key()
        if not self._command_rows:
            return
        self._command_values[key] = {idx: row.value() for idx, row in self._command_rows.items()}
        self._command_pins[key] = {
            idx: row.is_pinned() for idx, row in self._command_rows.items() if row.pin_box is not None
        }

    def _restore_command_values(self):
        key = self._command_storage_key()
        saved = self._command_values.get(key, {})
        saved_pins = self._command_pins.get(key, {})
        for idx, value in saved.items():
            row = self._command_rows.get(idx)
            if row is not None:
                row.set_value(value)
        for idx, row in self._command_rows.items():
            if row.pin_box is None:
                continue
            pinned = saved_pins.get(idx, False)
            row.pin_box.blockSignals(True)
            row.set_pinned(pinned)
            row.pin_box.blockSignals(False)
            row.pin_box.toggled.connect(
                lambda checked, dim_idx=idx: self._on_command_pin_toggled(dim_idx, checked)
            )

    def _on_command_pin_toggled(self, dim_idx: int, pinned: bool):
        key = self._command_storage_key()
        self._command_pins.setdefault(key, {})[dim_idx] = pinned
        if pinned:
            row = self._command_rows.get(dim_idx)
            if row is not None:
                self._command_values.setdefault(key, {})[dim_idx] = row.value()

    def _clear_command_values(self):
        for row in self._command_rows.values():
            row.set_value(0.0)
        key = self._command_storage_key()
        self._command_values[key] = {idx: 0.0 for idx in self._command_rows}

    def _save_body_field_values(self):
        key = self._active_body_storage_key
        if key is None or not self._body_rows:
            return
        self._body_field_values[key] = {field_key: row.value() for field_key, row in self._body_rows.items()}

    def _restore_body_field_values(self):
        key = self._body_pin_storage_key()
        if key is None:
            return
        saved = self._body_field_values.get(key, {})
        for field_key, value in saved.items():
            row = self._body_rows.get(field_key)
            if row is not None:
                row.set_value(value)

    def _save_joint_field_values(self):
        key = self._active_joint_storage_key
        if key is None or not self._joint_rows:
            return
        self._joint_field_values[key] = {field_key: row.value() for field_key, row in self._joint_rows.items()}

    def _restore_joint_field_values(self):
        key = self._joint_pin_storage_key()
        if key is None:
            return
        saved = self._joint_field_values.get(key, {})
        for field_key, value in saved.items():
            row = self._joint_rows.get(field_key)
            if row is not None:
                row.set_value(value)

    def _save_rl_action_values(self):
        key = self._rl_value_storage_key()
        if key is None or not self._rl_action_rows:
            return
        self._rl_action_values[key] = {idx: row.value() for idx, row in self._rl_action_rows.items()}

    def _restore_rl_action_values(self):
        key = self._rl_value_storage_key()
        if key is None:
            return
        saved = self._rl_action_values.get(key, {})
        for idx, value in saved.items():
            row = self._rl_action_rows.get(idx)
            if row is not None:
                row.set_value(value)

    def _clear_rl_action_values(self):
        for row in self._rl_action_rows.values():
            row.set_value(0.0)
        key = self._rl_value_storage_key()
        if key is not None:
            self._rl_action_values[key] = {idx: 0.0 for idx in self._rl_action_rows}

    def _set_body_field_pinned(self, field_key: str, pinned: bool):
        row = self._body_rows.get(field_key)
        if row is None or row.pin_box is None:
            return
        storage_key = self._body_pin_storage_key()
        if storage_key is None:
            return
        row.set_pinned(pinned)
        self._body_field_pins.setdefault(storage_key, {})[field_key] = pinned
        if pinned:
            self._body_field_values.setdefault(storage_key, {})[field_key] = row.value()

    def _restore_body_field_pins(self):
        storage_key = self._body_pin_storage_key()
        if storage_key is None:
            return
        saved = self._body_field_pins.get(storage_key, {})
        for field_key, row in self._body_rows.items():
            if row.pin_box is None:
                continue
            pinned = saved.get(field_key, False)
            row.pin_box.blockSignals(True)
            row.set_pinned(pinned)
            row.pin_box.blockSignals(False)
            row.pin_box.toggled.connect(
                lambda checked, fk=field_key: self._on_body_field_pin_toggled(fk, checked)
            )

    def _on_body_field_pin_toggled(self, field_key: str, pinned: bool):
        storage_key = self._body_pin_storage_key()
        if storage_key is None:
            return
        self._body_field_pins.setdefault(storage_key, {})[field_key] = pinned
        if pinned:
            row = self._body_rows.get(field_key)
            if row is not None:
                self._body_field_values.setdefault(storage_key, {})[field_key] = row.value()
        self._update_body_lock_buttons()

    def _is_body_group_locked(self, group: str) -> bool:
        keys = self._BODY_LOCK_GROUPS.get(group, ())
        present = [self._body_rows[k] for k in keys if k in self._body_rows and self._body_rows[k].pin_box is not None]
        if not present:
            return False
        return all(row.is_pinned() for row in present)

    def _update_body_lock_buttons(self):
        buttons = {
            "pos": self.lock_pos_btn,
            "quat": self.lock_rot_btn,
            "lin_vel": self.lock_lin_vel_btn,
            "ang_vel": self.lock_ang_vel_btn,
        }
        for group, btn in buttons.items():
            btn.blockSignals(True)
            btn.setChecked(self._is_body_group_locked(group))
            btn.setText(
                {
                    "pos": "Unlock Position" if self._is_body_group_locked("pos") else "Lock Position",
                    "quat": "Unlock Rotation" if self._is_body_group_locked("quat") else "Lock Rotation",
                    "lin_vel": "Unlock Lin Vel" if self._is_body_group_locked("lin_vel") else "Lock Lin Vel",
                    "ang_vel": "Unlock Ang Vel" if self._is_body_group_locked("ang_vel") else "Lock Ang Vel",
                }[group]
            )
            btn.blockSignals(False)

    def _toggle_body_group_lock(self, group: str):
        keys = self._BODY_LOCK_GROUPS.get(group, ())
        lockable = [k for k in keys if k in self._body_rows and self._body_rows[k].pin_box is not None]
        if not lockable:
            self._update_body_lock_buttons()
            return
        should_lock = not self._is_body_group_locked(group)
        storage_key = self._body_pin_storage_key()
        for field_key in lockable:
            if should_lock and storage_key is not None:
                self._body_field_values.setdefault(storage_key, {})[field_key] = self._body_rows[field_key].value()
            self._set_body_field_pinned(field_key, should_lock)
        self._update_body_lock_buttons()

    def _rebuild_body_panel(self):
        self._clear_layout(self.body_form)
        self._body_rows.clear()
        body: BodyParamSpec = self.body_selector.currentData()
        if body is None:
            return
        for key, label, lo, hi, step, pin, _enabled in self._body_field_defs(body):
            row = ParamRow(label, lo, hi, step, pin_enabled=pin)
            self._body_rows[key] = row
            self.body_form.addRow(row)
        self._restore_body_field_pins()
        self._restore_body_field_values()
        self._update_body_lock_buttons()
        self._update_apply_buttons()
        self._active_body_storage_key = self._body_pin_storage_key()

    def _restore_joint_field_pins(self):
        storage_key = self._joint_pin_storage_key()
        if storage_key is None:
            return
        saved = self._joint_field_pins.get(storage_key, {})
        for field_key, row in self._joint_rows.items():
            if row.pin_box is None:
                continue
            pinned = saved.get(field_key, False)
            row.pin_box.blockSignals(True)
            row.set_pinned(pinned)
            row.pin_box.blockSignals(False)
            row.pin_box.toggled.connect(
                lambda checked, fk=field_key: self._on_joint_field_pin_toggled(fk, checked)
            )

    def _on_joint_field_pin_toggled(self, field_key: str, pinned: bool):
        storage_key = self._joint_pin_storage_key()
        if storage_key is None:
            return
        self._joint_field_pins.setdefault(storage_key, {})[field_key] = pinned
        if pinned:
            row = self._joint_rows.get(field_key)
            if row is not None:
                self._joint_field_values.setdefault(storage_key, {})[field_key] = row.value()

    def _rebuild_joint_panel(self):
        self._clear_layout(self.joint_form)
        self._joint_rows.clear()
        joint: JointParamSpec = self.joint_selector.currentData()
        if joint is None:
            return
        lo, hi = joint.limit_min, joint.limit_max
        if lo >= hi:
            lo, hi = -3.14159, 3.14159
        for key, label, pin in [
            ("q", "Angle (rad)", True),
            ("qd", "Ang Vel", True),
            ("torque", "Torque", False),
        ]:
            span = max(abs(lo), abs(hi), 1.0)
            row = ParamRow(label, -span * 2, span * 2, 0.01, pin_enabled=pin)
            self._joint_rows[key] = row
            self.joint_form.addRow(row)
        self._restore_joint_field_pins()
        self._restore_joint_field_values()
        self._active_joint_storage_key = self._joint_pin_storage_key()

    def _clear_vbox_layout(self, layout: QVBoxLayout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _rebuild_rl_action_panel(self):
        self._clear_vbox_layout(self.rl_action_layout)
        self._rl_action_rows.clear()
        player_action = self._player_action
        if player_action is None:
            label = QLabel("No RL action space for this object.")
            self.rl_action_layout.addWidget(label)
            return

        if player_action.rl_action_row < 0:
            self.rl_action_layout.addWidget(
                QLabel("This player is not mapped to an RL action row (view only).")
            )

        for ability in player_action.abilities:
            title = QLabel(ability.ability_name)
            title.setStyleSheet("font-weight: bold; margin-top: 8px;")
            self.rl_action_layout.addWidget(title)
            for dim in ability.dims:
                row = ParamRow(
                    dim.display_name,
                    dim.lo,
                    dim.hi,
                    dim.step,
                    pin_enabled=False,
                    always_pinned=False,
                )
                row.spin.valueChanged.connect(self._on_rl_action_value_changed)
                self._rl_action_rows[dim.dim_index] = row
                self.rl_action_layout.addWidget(row)
        self._restore_rl_action_values()
        self._update_rl_action_controls_enabled()

    def _update_rl_action_controls_enabled(self):
        enabled = self.is_rl_action_enabled()
        self.rl_action_zero_btn.setEnabled(enabled)
        for row in self._rl_action_rows.values():
            row.set_enabled(enabled)

    def _on_rl_action_enable_toggled(self, enabled: bool):
        self._update_rl_action_controls_enabled()
        if enabled:
            self._save_rl_action_values()

    def _on_rl_action_value_changed(self, _value: float):
        if self.is_rl_action_enabled():
            self._save_rl_action_values()

    def _rebuild_commands_panel(self):
        self._clear_vbox_layout(self.commands_layout)
        self._command_rows.clear()
        if not self._command_labels:
            self.commands_layout.addWidget(QLabel("This level has no command inputs."))
            return
        title = QLabel("Velocity command for selected env")
        title.setStyleSheet("font-weight: bold;")
        self.commands_layout.addWidget(title)
        for idx, label in enumerate(self._command_labels):
            row = ParamRow(label, -1.0, 1.0, 0.01, pin_enabled=True, always_pinned=False)
            self._command_rows[idx] = row
            self.commands_layout.addWidget(row)
        self._restore_command_values()

    def _on_label_changed(self, label: str):
        if hasattr(self, "_label_changed_cb") and self._label_changed_cb:
            self._label_changed_cb(label, self._world_idx)

    def _on_world_changed(self, world: int):
        if hasattr(self, "_world_changed_cb") and self._world_changed_cb:
            self._world_changed_cb(world)
        else:
            self._world_idx = world

    def set_label_changed_callback(self, cb: Callable[[str, int], None]):
        self._label_changed_cb = cb

    def set_world_changed_callback(self, cb: Callable[[int], None]):
        self._world_changed_cb = cb

    def set_impulse_callback(self, cb: Callable):
        self._impulse_callbacks = [cb]

    def current_spec(self) -> Optional[ObjectInspectorSpec]:
        return self._spec

    def current_world(self) -> int:
        return self._world_idx

    def current_body(self) -> Optional[BodyParamSpec]:
        return self.body_selector.currentData()

    def current_joint(self) -> Optional[JointParamSpec]:
        return self.joint_selector.currentData()

    def get_body_state(self) -> BodyState:
        s = BodyState()
        if not self._body_rows:
            return s

        def _val(key: str, default: float = 0.0) -> float:
            row = self._body_rows.get(key)
            return row.value() if row is not None else default

        s.pos = [_val("pos_x"), _val("pos_y"), _val("pos_z")]
        s.quat = [_val("quat_x"), _val("quat_y"), _val("quat_z"), _val("quat_w", 1.0)]
        s.lin_vel = [_val("vel_x"), _val("vel_y"), _val("vel_z")]
        s.ang_vel = [_val("omega_x"), _val("omega_y"), _val("omega_z")]
        s.force = [_val("force_x"), _val("force_y"), _val("force_z")]
        s.torque = [_val("torque_x"), _val("torque_y"), _val("torque_z")]
        return s

    def set_body_state(self, state: BodyState):
        if not self._body_rows:
            return
        mapping = [
            ("pos_x", state.pos[0]),
            ("pos_y", state.pos[1]),
            ("pos_z", state.pos[2]),
            ("quat_x", state.quat[0]),
            ("quat_y", state.quat[1]),
            ("quat_z", state.quat[2]),
            ("quat_w", state.quat[3]),
            ("vel_x", state.lin_vel[0]),
            ("vel_y", state.lin_vel[1]),
            ("vel_z", state.lin_vel[2]),
            ("omega_x", state.ang_vel[0]),
            ("omega_y", state.ang_vel[1]),
            ("omega_z", state.ang_vel[2]),
        ]
        for key, value in mapping:
            row = self._body_rows.get(key)
            if row is not None and row.should_sync_from_sim():
                row.set_value(value)

    def get_joint_state(self) -> JointState:
        js = JointState()
        if "q" in self._joint_rows:
            js.q = self._joint_rows["q"].value()
        if "qd" in self._joint_rows:
            js.qd = self._joint_rows["qd"].value()
        if "torque" in self._joint_rows:
            js.torque = self._joint_rows["torque"].value()
        return js

    def set_joint_state(self, state: JointState):
        if "q" in self._joint_rows and self._joint_rows["q"].should_sync_from_sim():
            self._joint_rows["q"].set_value(state.q)
        if "qd" in self._joint_rows and self._joint_rows["qd"].should_sync_from_sim():
            self._joint_rows["qd"].set_value(state.qd)

    def get_body_pinned(self) -> PinnedBodyFields:
        if not self._body_rows:
            return PinnedBodyFields()

        def _group_pinned(keys: tuple[str, ...]) -> bool:
            return any(self._body_rows[k].is_pinned() for k in keys if k in self._body_rows)

        return PinnedBodyFields(
            pos=_group_pinned(("pos_x", "pos_y", "pos_z")),
            quat=_group_pinned(("quat_w", "quat_x", "quat_y", "quat_z")),
            lin_vel=_group_pinned(("vel_x", "vel_y", "vel_z")),
            ang_vel=_group_pinned(("omega_x", "omega_y", "omega_z")),
        )

    def get_joint_pinned(self) -> PinnedJointFields:
        return PinnedJointFields(
            q=self._joint_rows.get("q") is not None and self._joint_rows["q"].is_pinned(),
            qd=self._joint_rows.get("qd") is not None and self._joint_rows["qd"].is_pinned(),
        )

    def current_player_action(self) -> Optional[PlayerActionSpec]:
        return self._player_action

    def get_rl_action_values(self) -> Dict[int, float]:
        return {idx: row.value() for idx, row in self._rl_action_rows.items()}

    def set_rl_action_values(self, values: Dict[int, float], force: bool = False):
        if self.is_rl_action_enabled() and not force:
            return
        for idx, value in values.items():
            row = self._rl_action_rows.get(idx)
            if row is not None and (force or row.should_sync_from_sim()):
                row.set_value(value)

    def get_pinned_rl_action_indices(self) -> List[int]:
        return [idx for idx, row in self._rl_action_rows.items() if row.is_pinned()]

    def is_rl_action_enabled(self) -> bool:
        return self.rl_action_enable_checkbox.isChecked()

    def get_command_values(self) -> Dict[int, float]:
        return {idx: row.value() for idx, row in self._command_rows.items()}

    def set_command_values(self, values: Dict[int, float], force: bool = False):
        for idx, value in values.items():
            row = self._command_rows.get(idx)
            if row is not None and (force or row.should_sync_from_sim()):
                row.set_value(value)

    def get_pinned_command_indices(self) -> List[int]:
        return [idx for idx, row in self._command_rows.items() if row.is_pinned()]

    def _apply_impulse(self, force: bool, torque: bool):
        for cb in self._impulse_callbacks:
            cb(force=force, torque=torque, joint=False)

    def _apply_joint_torque(self):
        for cb in self._impulse_callbacks:
            cb(force=False, torque=False, joint=True)
