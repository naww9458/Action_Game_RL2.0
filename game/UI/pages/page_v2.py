import sys
import os
import re
import yaml
import ast
import inspect
import copy
import multiprocessing as mp
import subprocess

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

from pathlib import Path
from pydantic import BaseModel

from script.levels.level_cfg import EnvironmentConfig

from script.simulate.solvers.base_solver import SolverRegistry
from script.simulate.solvers.coupled import (
    CoupledSolverModel,
    RIGID_CAPABLE_SOLVERS,
    SOFT_CAPABLE_SOLVERS,
    coupled_solvers_list,
    prune_coupled_solver_configs,
    resolve_coupled_domains,
)
from script.role.objects.base_object import ObjectRegistry
from script.role.objects.object_template.loader import load_object_templates
from script.role.base_role import RoleRegistry
from script.role.abilities.ability import Ability

from utils.get_pydantic_default import get_pydantic_default
from UI.pages.experiment_hub import ExperimentHubPage

from typing import List, get_origin, get_args, Union

_MISSING_FIELD_DEFAULT = object()

# 角色屬性白名單：只有對該角色所有物件（無論 object template 為何）都通用、
# 且不進入 object template 的欄位才能作為「角色屬性」。
# 不在白名單中的非模型屬性欄位，一律視為「物件屬性」。
_ROLE_ATTR_FIELDS = frozenset({
    "type", "id", "name", "color",
    "default_position", "default_rotation",
    "default_velocity", "default_angular_velocity",
    "controller", "team_id", "health",
    # dict 容器角色（entity/ability_generated_object）的 dict key 即為
    # 「物件子角色」（object_sub_role），等同於 list 角色的 id：是單一配置
    # 自身的唯一鍵，不進入模板
    # 工具每物件配置：連接宿主 / 自動掛載不進入模板，是單一工具自身的設定
    "host_player_index",
    "host_player_id",
    "proximity_threshold",
    "proximity_height_threshold",
    "start_attached",
})

# 加入物理引擎時輸入的「模型屬性」欄位（在「物件屬性」內最底部的次級區域）
_MODEL_ATTR_FIELD = "object"

# 物件模板（template.yaml）頂層中保留給模板本身使用的 key，
# 套用模板時不會複製到角色配置
_OBJECT_TEMPLATE_RESERVED_KEYS = frozenset({
    "id", "display_name", "object_type", "control_config_path", "object",
})


def get_editor_field_default(field_info):
    """Return a YAML-safe Pydantic field default for an absent editor value."""
    if field_info.is_required():
        return _MISSING_FIELD_DEFAULT

    value = field_info.get_default(call_default_factory=True)
    if value is None:
        return _MISSING_FIELD_DEFAULT
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value


_ABILITIES_SCHEMA_CACHE = None


def _load_abilities_schema() -> dict:
    """Load the raw abilities_default_cfg.yaml as {ability_name: {param: value}}.

    Used by the abilities editor to render per-ability parameter input boxes.
    """
    global _ABILITIES_SCHEMA_CACHE
    if _ABILITIES_SCHEMA_CACHE is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "script"
            / "role"
            / "abilities"
            / "abilities_default_cfg.yaml"
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            _ABILITIES_SCHEMA_CACHE = raw if isinstance(raw, dict) else {}
        except Exception as e:  # pragma: no cover - defensive UI fallback
            print(f"Failed to load abilities schema for editor: {e}")
            _ABILITIES_SCHEMA_CACHE = {}
    return _ABILITIES_SCHEMA_CACHE


def _ability_param_fields(schema: dict) -> dict:
    """Return editable numeric params per ability.

    ``{ability_name: {field: ("float" | "vec3", default)}}``. Skips nested
    ``key`` / ``action_space`` dicts; numeric scalars become spin boxes and
    3-element numeric lists become XYZ vector inputs.
    """
    result = {}
    for name, raw in (schema or {}).items():
        if not isinstance(raw, dict):
            continue
        fields = {}
        for field, val in raw.items():
            if field in ("key", "action_space") or isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                fields[field] = ("float", float(val))
            elif (
                isinstance(val, list)
                and len(val) >= 3
                and all(isinstance(v, (int, float)) for v in val)
            ):
                fields[field] = ("vec3", [float(v) for v in val[:3]])
        if fields:
            result[name] = fields
    return result





# --- 多選下拉選單組件 ---
class MultiSelectComboBox(QComboBox):
    selectionChanged = pyqtSignal(list)

    def __init__(self, items, selected_items, parent=None):
        # 必須先初始化 list_widget 為 None，避免 eventFilter 提早觸發導致報錯
        self.list_widget = None 
        super().__init__(parent)
        
        self.setEditable(True)
        # 隱藏原生的編輯框，改用我們自定義的多行顯示區域
        self.lineEdit().setReadOnly(True)
        self.lineEdit().hide()

        # 1. 創建多行顯示區域 (獨占一行顯示)
        self._display_text = QPlainTextEdit()
        self._display_text.setReadOnly(True)
        self._display_text.setFont(self.font()) 
        self._display_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._display_text.setFrameStyle(QFrame.Shape.NoFrame)
        # 設定滑鼠穿透，讓點擊事件傳遞給 ComboBox
        self._display_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # 為了讓外觀更像 ComboBox，可以加上背景透明
        self._display_text.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 2, 25, 2)
        layout.addWidget(self._display_text)

        # 2. 設置下拉列表
        self.list_widget = QListWidget()
        for item in items:
            lw_item = QListWidgetItem(item, self.list_widget)
            lw_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check_state = Qt.CheckState.Checked if item in selected_items else Qt.CheckState.Unchecked
            lw_item.setCheckState(check_state)
            
        self.setModel(self.list_widget.model())
        self.setView(self.list_widget)
        
        # 3. 安裝過濾器處理點擊 CheckBox 不關閉選單，以及點擊外部關閉選單
        self.list_widget.viewport().installEventFilter(self)
        self.list_widget.itemChanged.connect(self.on_item_changed)
        
        self.update_text()

    def eventFilter(self, obj, event):
        # 如果變數尚未初始化，不處理事件
        if self.list_widget is None:
            return super().eventFilter(obj, event)

        # 核心修復：點擊 CheckBox 區域時切換狀態，且不讓事件傳遞給 QComboBox 導致選單關閉
        if obj == self.list_widget.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            index = self.list_widget.indexAt(event.pos())
            if index.isValid():
                item = self.list_widget.item(index.row())
                # 手動切換勾選狀態
                new_state = Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
                item.setCheckState(new_state)
                return True # 攔截事件，防止選單關閉
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        # 修復：點擊整個方框（包含文字區域）都能打開選單
        self.showPopup()

    def on_item_changed(self):
        selected = self.get_selected()
        self.update_text()
        self.selectionChanged.emit(selected)

    def update_text(self):
        selected = self.get_selected()
        # 更新文字
        self._display_text.setPlainText("\n".join(selected))
        
        # --- 修改：根據字體大小動態計算高度 ---
        # 獲取當前字體的行間距 (Line Spacing)
        metrics = self.fontMetrics()
        line_height = metrics.lineSpacing() 
        
        # 計算總高度：行數 * 行高 + 上下留白 (Margins)
        line_count = max(1, len(selected))
        # 10px 為上下 Margin 的補償值
        total_height = (line_count * line_height) + 10 
        
        self.setMinimumHeight(total_height)
        self.setFixedHeight(total_height) # 強制更新高度
        # -----------------------------------

    def get_selected(self):
        if self.list_widget is None: return []
        res = []
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                res.append(it.text())
        return res


class CollapsibleSection(QWidget):
    """可摺疊的表單區塊：點擊標題列展開/收起內容。"""

    def __init__(
        self,
        title: str,
        accent_color: str = "#90caf9",
        expanded: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle.setStyleSheet(
            "QToolButton {"
            " border: none;"
            " font-weight: bold;"
            " text-align: left;"
            " padding: 4px 2px;"
            " color: #ddd;"
            "}"
            "QToolButton:hover { background-color: #3a3a3a; }"
        )
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle)

        self._content = QFrame()
        self._content.setFrameShape(QFrame.Shape.StyledPanel)
        self._content.setStyleSheet(
            f"QFrame {{ border-left: 3px solid {accent_color};"
            f" background-color: #262626; margin: 0 0 4px 0; padding: 4px; }}"
        )
        self.form_layout = QFormLayout(self._content)
        self.form_layout.setContentsMargins(8, 4, 8, 4)
        layout.addWidget(self._content)

        self._content.setVisible(expanded)

    def _on_toggled(self, expanded: bool) -> None:
        self._content.setVisible(expanded)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )


class AbilityConfigEditor(QWidget):
    """Edits a role's ``abilities`` as dict form ``{ability_name: {param: value}}``.

    A multi-select combo enables/disables abilities; below it, each enabled
    ability gets its own group box with numeric parameter spin boxes sourced
    from ``abilities_default_cfg.yaml``. Legacy list form (``["Shoot", ...]``)
    is normalized to dict form with empty parameter dicts on load.
    """

    valueChanged = pyqtSignal()

    def __init__(self, all_abilities, current_value, param_fields, parent=None):
        super().__init__(parent)
        self._all_abilities = list(all_abilities or [])
        self._param_fields = param_fields or {}
        if isinstance(current_value, dict):
            self._value = {
                str(k): dict(v) for k, v in current_value.items() if str(k) in self._all_abilities
            }
        else:
            self._value = {
                str(n): {} for n in (current_value or []) if str(n) in self._all_abilities
            }

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._combo = MultiSelectComboBox(self._all_abilities, list(self._value.keys()))
        self._combo.selectionChanged.connect(self._on_selection_changed)
        self._layout.addWidget(self._combo)

        self._params_host = QWidget()
        self._params_layout = QVBoxLayout(self._params_host)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        self._params_layout.setSpacing(2)
        self._layout.addWidget(self._params_host)

        self._rebuild_params()

    def _on_selection_changed(self, selected):
        new_value = {}
        for name in selected:
            new_value[name] = self._value.get(name, {})
        self._value = new_value
        self._rebuild_params()
        self.valueChanged.emit()

    def _clear_params(self):
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _rebuild_params(self):
        self._clear_params()
        for name in self._value:
            fields = self._param_fields.get(name, {})
            if not fields:
                continue
            group = QGroupBox(name)
            group.setStyleSheet("QGroupBox { font-weight: bold; }")
            form = QFormLayout(group)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            form.setContentsMargins(6, 2, 6, 2)
            entry = self._value[name]
            for field, (kind, default) in fields.items():
                label = QLabel(field)
                label.setStyleSheet("color: #888;")
                current = entry.get(field, default)
                if kind == "vec3":
                    widget = self._create_vec3(name, field, current)
                else:
                    widget = self._create_spin(name, field, current)
                form.addRow(label, widget)
            self._params_layout.addWidget(group)

    def _create_spin(self, name, field, value):
        spin = FlexibleDoubleSpinBox()
        try:
            spin.setValue(float(value))
        except (TypeError, ValueError):
            spin.setValue(0.0)
        spin.valueChanged.connect(
            lambda v, n=name, f=field: self._update_scalar(n, f, v)
        )
        return spin

    def _create_vec3(self, name, field, value):
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(2)
        if isinstance(value, list) and len(value) >= 3:
            vec = [float(v) for v in value[:3]]
        else:
            vec = [0.0, 0.0, 0.0]
        for i in range(3):
            s = FlexibleDoubleSpinBox()
            s.setValue(vec[i])
            s.valueChanged.connect(
                lambda v, n=name, f=field, i=i: self._update_vec3(n, f, i, v)
            )
            h.addWidget(s)
        return container

    def _update_scalar(self, name, field, value):
        self._value.setdefault(name, {})[field] = float(value)
        self.valueChanged.emit()

    def _update_vec3(self, name, field, idx, value):
        entry = self._value.setdefault(name, {})
        vec = entry.get(field)
        if not isinstance(vec, list):
            vec = [0.0, 0.0, 0.0]
        while len(vec) < 3:
            vec.append(0.0)
        vec[idx] = float(value)
        entry[field] = vec
        self.valueChanged.emit()

    def get_value(self):
        """Return the dict-form abilities value (keeps empty-override abilities)."""
        return dict(self._value)





# --- 1. 翻譯管理模塊 ---
class TransMgr:
    def __init__(self):
        self.current_lang = "zh"
        self.data = {}
        self.load_lang(self.current_lang)


    def load_lang(self, lang_code):
        self.current_lang = lang_code
        file_path = f"{lang_code}.yaml"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading language {lang_code}: {e}")

    def get(self, key):
        return self.data.get(key, key)

TR = TransMgr()





# --- 2. 新增子關卡對話框 ---
class NewSubLevelDialog(QDialog):
    def __init__(self, series_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(TR.get("select_series_title"))
        self.setFixedSize(350, 180)
        layout = QVBoxLayout(self)
        desc_label = QLabel(TR.get("select_series_desc"))
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        self.combo = QComboBox()
        self.combo.addItems(series_list)
        layout.addWidget(self.combo)
        layout.addStretch()
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_selected_series(self):
        return self.combo.currentText()





# --- 3D 預覽組件 ---
class GLPreviewWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = {}
        self.camera_rot = [-50, 45]
        self.camera_dist = 5.0
        self.last_mouse_pos = QPoint()

        self.role_keys = RoleRegistry.get_all_keys()

    def set_data(self, data):
        self.data = data
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.12, 0.12, 0.12, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h > 0 else 1, 0.1, 100.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -self.camera_dist)
        glRotatef(self.camera_rot[0], 1, 0, 0)
        glRotatef(self.camera_rot[1], 0, 0, 1)

        self.draw_grid()

        if not self.data: return

        for key in self.role_keys:
            role_data = self.data.get(key + "_configs", [])

            if isinstance(role_data, list):
                for d in role_data:
                    self.draw_object(d)

            elif isinstance(role_data, dict):
                for k, d in role_data.items():
                    self.draw_object(d)

    def draw_grid(self):
        glBegin(GL_LINES)
        glColor3f(0.3, 0.3, 0.3)
        size = 100
        for i in range(-size, size + 5):
            glVertex3f(i, -size, 0); glVertex3f(i, size, 0)
            glVertex3f(-size, i, 0); glVertex3f(size, i, 0)
        glEnd()

    def _get_avg_val(self, val):
        if isinstance(val, list) and len(val) >= 2:
            return (val[0] + val[1]) / 2.0
        return float(val) if val is not None else 0.0

    def draw_object(self, cfg):
        raw_pos = cfg.get("default_position", [0, 0, 0])
        actual_pos = [0, 0, 0]
        ranges = [[0,0], [0,0], [0,0]]
        is_random = False

        for i in range(3):
            val = raw_pos[i] if i < len(raw_pos) else 0
            if isinstance(val, list) and len(val) >= 2:
                is_random = True
                ranges[i] = [val[0], val[1]]
                actual_pos[i] = (val[0] + val[1]) / 2.0
            else:
                actual_pos[i] = val
                ranges[i] = [val, val]

        if is_random:
            self.draw_range_box(ranges)

        raw_rot = cfg.get("default_rotation", [0, 0, 0])
        if not isinstance(raw_rot, list): raw_rot = [raw_rot, 0, 0]
        actual_rot = [self._get_avg_val(v) for v in raw_rot]

        color = [c/255.0 for c in cfg.get("color", [200, 200, 200])]


        glPushMatrix()
        glTranslatef(actual_pos[0], actual_pos[1], actual_pos[2])
        if len(actual_rot) >= 3:
            glRotatef(actual_rot[2], 0, 0, 1)
            glRotatef(actual_rot[1], 0, 1, 0)
            glRotatef(actual_rot[0], 1, 0, 0)

        glColor3f(*color[:3])

        object_data = cfg.get("object", None)
        if object_data is None:
            print("Warning: No object data found for config:", cfg) # TODO 警告信息或許應該加上顔色字體
            return

        object_type = object_data.get("type")

        if object_type == "rigid_box":
            raw_size = object_data.get("size", [0.0, 0.0, 0.0])
            if not isinstance(raw_size, list): raw_size = [raw_size]
            s = [self._get_avg_val(v) for v in raw_size]

            self.draw_cube(s[0], s[1], s[2])

        elif object_type == "rigid_sphere":
            r = object_data.get("radius", 0.0)

            self.draw_sphere(r)

        glPopMatrix()

    def draw_range_box(self, ranges):
        x1, x2 = ranges[0]; y1, y2 = ranges[1]; z1, z2 = ranges[2]
        glColor4f(1.0, 1.0, 0.0, 0.3)
        glBegin(GL_LINES)
        glVertex3f(x1, y1, z1); glVertex3f(x2, y1, z1); glVertex3f(x2, y1, z1); glVertex3f(x2, y2, z1)
        glVertex3f(x2, y2, z1); glVertex3f(x1, y2, z1); glVertex3f(x1, y2, z1); glVertex3f(x1, y1, z1)
        glVertex3f(x1, y1, z2); glVertex3f(x2, y1, z2); glVertex3f(x2, y1, z2); glVertex3f(x2, y2, z2)
        glVertex3f(x2, y2, z2); glVertex3f(x1, y2, z2); glVertex3f(x1, y2, z2); glVertex3f(x1, y1, z2)
        glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z2); glVertex3f(x2, y1, z1); glVertex3f(x2, y1, z2)
        glVertex3f(x2, y2, z1); glVertex3f(x2, y2, z2); glVertex3f(x1, y2, z1); glVertex3f(x1, y2, z2)
        glEnd()

    def draw_cube(self, w, l, h):
        w, l, h = w, l, h
        glBegin(GL_QUADS)
        faces = [
            [(-w,-l,h),(w,-l,h),(w,l,h),(-w,l,h)], [(-w,-l,-h),(-w,l,-h),(w,l,-h),(w,-l,-h)],
            [(-w,l,-h),(-w,l,h),(w,l,h),(w,l,-h)], [(-w,-l,-h),(w,-l,-h),(w,-l,h),(-w,-l,h)],
            [(w,-l,-h),(w,l,-h),(w,l,h),(w,-l,h)], [(-w,-l,-h),(-w,-l,h),(-w,l,h),(-w,l,-h)]
        ]
        for face in faces:
            for v in face: glVertex3f(*v)
        glEnd()
        glColor3f(0, 0, 0)
        glBegin(GL_LINE_LOOP)
        for v in faces[0]: glVertex3f(*v)
        glEnd()

    def draw_sphere(self, r):
        quad = gluNewQuadric()
        gluSphere(quad, r, 16, 16)

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.position().x() - self.last_mouse_pos.x()
        dy = event.position().y() - self.last_mouse_pos.y()
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.camera_rot[1] += dx * 0.5
            self.camera_rot[0] += dy * 0.5
        self.last_mouse_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        self.camera_dist -= event.angleDelta().y() * 0.005
        self.camera_dist = max(1.0, min(50.0, self.camera_dist))
        self.update()





# --- 基礎頁面類 ---
class BasePage(QWidget):
    def __init__(self, title_key, parent=None):
        super().__init__(parent)
        self.title_key = title_key
        self.layout = QVBoxLayout(self)
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; margin: 10px;")
        self.layout.addWidget(self.title_label)
        
    def retranslate_ui(self):
        self.title_label.setText(TR.get(self.title_key))





class FlexibleDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDecimals(10)  # 設置足夠大的精度
        self.setRange(-999999, 999999)
        self.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)

    def textFromValue(self, value):
        # 邏輯：保留到非零值，或者至少保留一位小數 (.0)
        # 比如 0.0001 -> "0.0001", 0.0000 -> "0.0"
        s = format(value, '.10f').rstrip('0')
        if s.endswith('.'):
            s += '0'
        return s

# --- 編輯頁面 ---
class EditPage(BasePage):
    def __init__(self):
        super().__init__("edit_mode")

        self.role_infos: dict = RoleRegistry.get_all_infos()

        # --- 內部索引管理 ---
        self._temp_registry = {}      # 臨時 ID -> (data_type, index, model_cls)
        self._current_temp_id = None  # 當前正在編輯的臨時 ID
        self._temp_counter = 0

        self.title_label.hide()
        self.current_yaml_data = {}
        self.current_file_path = ""
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.row_counter = 0  # 用於斑馬紋計數

        # --- 左側：物件列表 ---
        self.list_container = QWidget()
        list_layout = QVBoxLayout(self.list_container)

        header_layout = QHBoxLayout()
        self.list_title = QLabel(); self.list_title.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.list_title)
        header_layout.addStretch() 
        
        self.btn_show_add_menu = QPushButton("新增 ✚")
        self.btn_show_add_menu.setStyleSheet("background-color: #3d3d3d; font-weight: bold; padding: 2px 10px;")
        self.btn_show_add_menu.clicked.connect(self.toggle_add_menu)
        header_layout.addWidget(self.btn_show_add_menu)

        self.obj_tree = QTreeWidget(); self.obj_tree.setHeaderHidden(True)
        self.obj_tree.itemClicked.connect(self.on_object_selected)

        self.add_sub_widget = QWidget()
        self.add_btns_layout = QVBoxLayout(self.add_sub_widget) 
        self.add_btns_layout.setContentsMargins(5, 5, 5, 5)
        self.add_sub_widget.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        self.add_sub_widget.hide() 

        self.init_dynamic_buttons()

        self.del_btn_layout = QHBoxLayout()
        self.btn_delete = QPushButton()
        self.btn_delete.setStyleSheet("color: #ff5555; font-weight: bold;")
        self.btn_delete.clicked.connect(self.delete_selected_object)
        self.del_btn_layout.addWidget(self.btn_delete)

        list_layout.addLayout(header_layout) 
        list_layout.addWidget(self.add_sub_widget) 
        list_layout.addWidget(self.obj_tree)
        list_layout.addLayout(self.del_btn_layout)

        # --- 中側和右側：屬性編輯與預覽 ---
        self.right_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.attr_container = QWidget()
        attr_layout = QVBoxLayout(self.attr_container)
        self.attr_title = QLabel(); self.attr_title.setStyleSheet("font-weight: bold;")
        self.attr_scroll = QScrollArea(); self.attr_scroll.setWidgetResizable(True)
        self.attr_widget = QWidget(); self.attr_form = QFormLayout(self.attr_widget)
        # 設置 form 邊距為 0 以便斑馬紋填滿
        self.attr_form.setContentsMargins(0, 5, 0, 5)
        self.attr_form.setSpacing(0)
        self.attr_scroll.setWidget(self.attr_widget)
        attr_layout.addWidget(self.attr_title); attr_layout.addWidget(self.attr_scroll)

        self.preview_container = QFrame()
        self.preview_container.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555;")
        preview_layout = QVBoxLayout(self.preview_container)
        self.preview_widget = GLPreviewWidget() 
        preview_layout.addWidget(self.preview_widget)

        self.right_splitter.addWidget(self.attr_container); self.right_splitter.addWidget(self.preview_container)
        self.main_splitter.addWidget(self.list_container); self.main_splitter.addWidget(self.right_splitter)
        
        self.main_splitter.setStretchFactor(0, 1); self.main_splitter.setStretchFactor(1, 4)
        self.right_splitter.setStretchFactor(0, 1); self.right_splitter.setStretchFactor(1, 2)
        
        self.layout.addWidget(self.main_splitter)
        
        self.bottom_btns_layout = QHBoxLayout()
        self.cancel_btn = QPushButton(); self.cancel_btn.clicked.connect(self.exit_without_saving)
        self.back_btn = QPushButton(); self.back_btn.clicked.connect(self.save_and_exit)
        self.bottom_btns_layout.addWidget(self.cancel_btn); self.bottom_btns_layout.addWidget(self.back_btn)
        self.layout.addLayout(self.bottom_btns_layout)
        self.retranslate_ui()

    def toggle_add_menu(self):
        """切換新增選單的顯示狀態"""
        if self.add_sub_widget.isHidden():
            self.add_sub_widget.show()
            self.btn_show_add_menu.setText("收起 ▲")
        else:
            self.add_sub_widget.hide()
            self.btn_show_add_menu.setText("新增 ✚")

    def init_dynamic_buttons(self):
        """根據 role_infos 自動生成新增按鈕"""
        for role_name, info in self.role_infos.items():
            btn = QPushButton(TR.get(f"add_{role_name}"))
            btn.clicked.connect(
                lambda checked, i=info, rn=role_name: self.add_object_from_registry(i, rn)
            )
            self.add_btns_layout.addWidget(btn)

    def load_config(self, file_path):
        self.current_file_path = file_path
        self._temp_registry.clear()
        self._current_temp_id = None
        
        if not os.path.exists(file_path): return
        with open(file_path, 'r', encoding='utf-8') as f:
            self.current_yaml_data = yaml.safe_load(f) or {}
            
        if "environment_configs" not in self.current_yaml_data: 
            self.current_yaml_data["environment_configs"] = copy.deepcopy(get_pydantic_default(EnvironmentConfig))
            
        self.refresh_object_tree()
        self.clear_attr_form()
        self.preview_widget.set_data(self.current_yaml_data) 

    def _get_next_temp_id(self, info_tuple):
        self._temp_counter += 1
        tid = self._temp_counter
        self._temp_registry[tid] = info_tuple
        return tid

    def refresh_object_tree(self):
        # 記住當前正在編輯的定位資訊，以便重建後找回 ID
        last_info = self._temp_registry.get(self._current_temp_id)
        
        self.obj_tree.clear()
        self._temp_registry.clear()
        
        # 環境變數根節點
        env_info = ("env", "env_root", EnvironmentConfig)
        env_tid = self._get_next_temp_id(env_info)
        env_item = QTreeWidgetItem(self.obj_tree, [TR.get("env_cfg")])
        env_item.setData(0, Qt.ItemDataRole.UserRole, env_tid)
        
        new_selected_item = None
        for role_name, info in self.role_infos.items():
            path_key = info["path"]
            model_cls = info["model"]
            container = info["container"]
            root_item = QTreeWidgetItem(self.obj_tree, [TR.get(f"{role_name.lower()}_cfg")])
            
            data_source = self.current_yaml_data.get(path_key, [] if container == "list" else {})
            
            if container == "list":
                for i, obj_data in enumerate(data_source):
                    obj_id = str(obj_data.get("id") or "")
                    disp_name = obj_data.get("name") or obj_data.get("object", {}).get("pattern") or f"{role_name} {i}"
                    label = f"{obj_id} ({disp_name})" if obj_id and obj_id != disp_name else (obj_id or disp_name)
                    child = QTreeWidgetItem(root_item, [label])
                    info_tuple = (path_key, i, model_cls)
                    tid = self._get_next_temp_id(info_tuple)
                    child.setData(0, Qt.ItemDataRole.UserRole, tid)
                    if last_info == info_tuple:
                        self._current_temp_id = tid
                        new_selected_item = child
            elif container == "dict":
                for key, obj_data in data_source.items():
                    disp_name = obj_data.get("name") or obj_data.get("object", {}).get("pattern") or key
                    label = f"{key} ({disp_name})" if key != disp_name else key
                    child = QTreeWidgetItem(root_item, [label])
                    info_tuple = (path_key, key, model_cls)
                    tid = self._get_next_temp_id(info_tuple)
                    child.setData(0, Qt.ItemDataRole.UserRole, tid)
                    if last_info == info_tuple:
                        self._current_temp_id = tid
                        new_selected_item = child
                    
        self.obj_tree.expandAll()
        if new_selected_item:
            self.obj_tree.setCurrentItem(new_selected_item)
        self.preview_widget.update() 

    def clear_attr_form(self):
        self.row_counter = 0 # 重置斑馬紋
        while self.attr_form.count():
            child = self.attr_form.takeAt(0)
            if child.widget(): child.widget().deleteLater()

    def get_target_data(self, data_type, index):
        if data_type == "env":
            return self.current_yaml_data.get("environment_configs", {})
        container = self.current_yaml_data.get(data_type)
        if isinstance(container, list) and isinstance(index, int):
            return container[index]
        elif isinstance(container, dict):
            return container.get(index, {})
        return {}

    def on_object_selected(self, item):
        if item is None: return
        tid = item.data(0, Qt.ItemDataRole.UserRole)
        self._current_temp_id = tid
        self.refresh_editor_by_temp_id(tid)

    def refresh_editor_by_temp_id(self, tid):
        """根據內部臨時索引刷新編輯器，避免依賴 Tree Widget 狀態"""
        user_data = self._temp_registry.get(tid)
        if not user_data: return
        data_type, index, model_cls = user_data
        
        self.clear_attr_form()
        if index is None or ("_root" in str(index) and data_type != "env"): return

        target_data = self.get_target_data(data_type, index)
        if isinstance(index, str) and data_type != "env":
            name_input = QLineEdit(str(index))
            name_input.editingFinished.connect(lambda: self.rename_dict_key(data_type, index, name_input.text()))
            # dict 容器角色的 key = 物件子角色（object_sub_role）
            self.add_zebra_row(f"{TR.get('obj_sub_role')}:", name_input)

        if target_data and model_cls:
            self.render_editor_recursive(target_data, model_cls)

    def add_zebra_row(self, label_text, widget):
        """添加帶有斑馬紋背景的行"""
        container = QWidget()
        bg_color = "#2b2b2b" if self.row_counter % 2 == 0 else "#333333"
        container.setStyleSheet(f"background-color: {bg_color};")
        
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 5, 10, 5)
        
        lbl = QLabel(label_text)
        lbl.setFixedWidth(160) # 固定寬度對齊
        layout.addWidget(lbl)
        layout.addWidget(widget)
        
        self.attr_form.addRow(container)
        self.row_counter += 1

    # 明確指定連接對象（host_player_id）且 start_attached 時，
    # 工具的初始姿態參數由宿主繼承，無需編輯
    _POSE_FIELDS_WHEN_HOST_BOUND = (
        "default_position",
        "default_rotation",
        "default_velocity",
        "default_angular_velocity",
        "host_player_index",
    )

    def _should_hide_pose(self, data_source) -> bool:
        """工具明確綁定宿主且啟動即掛載時，初始姿態由宿主繼承，隱藏編輯欄位。

        取消 start_attached 後（或未指定 host_player_id）初始姿態欄位重新出現。
        """
        return bool(
            data_source
            and data_source.get("host_player_id")
            and data_source.get("start_attached")
        )

    def _add_section_header(self, text, color="#888", border_color=None):
        """添加分區標題。

        color 為文字顏色；border_color 提供時會在左側畫一條強調色邊條，
        用於讓「角色屬性 / 物件屬性 / 模型屬性」三個區段明顯區分。
        """
        label = QLabel(text)
        if border_color:
            label.setStyleSheet(
                f"color: {color}; font-weight: bold;"
                f"border-left: 4px solid {border_color};"
                f"background-color: #2a2a2a;"
                f"padding: 4px 8px; margin: 6px 0;"
            )
        else:
            label.setStyleSheet(f"color: {color}; font-weight: bold; margin: 5px 10px;")
        self.attr_form.addRow(label)

    def _prepare_field_value(self, data_source, field_name, field_info):
        val = data_source.get(field_name)
        if val is None:
            default_value = get_editor_field_default(field_info)
            if default_value is not _MISSING_FIELD_DEFAULT:
                val = default_value
                data_source[field_name] = val
        return val

    def _render_field(self, data_source, field_name, field_info):
        """渲染單一欄位（含嵌套模型處理）"""
        val = self._prepare_field_value(data_source, field_name, field_info)
        field_type = field_info.annotation
        if hasattr(field_type, '__metadata__'):
            field_type = field_type.__args__[0]
        # 嵌套模型處理
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            self._add_section_header(f"--- {field_name.upper()} ---")
            if not isinstance(val, dict):
                val = {}
                data_source[field_name] = val
            self.render_editor_recursive(val, field_type, indent=True)
            return
        self.add_editable_row(field_name, val, data_source, field_type=field_type)

    def render_editor_recursive(self, data_source, model_class, indent=False):
        """遞歸渲染 Pydantic 模型屬性，依欄位性質分為三大區段：

          1. 角色屬性  — 角色屬性白名單內的欄位，對該角色所有物件都通用
                         （type / id / name / 初始坐標旋轉速度角速度 / controller /
                         team_id / health，以及 tool 的 host_player_id、start_attached 等）
                        置於最頂部，與物件/模型屬性明顯區分。
          2. 物件屬性  — 不在白名單內的非模型屬性欄位，與特定物件/模板相關、
                         會進入 object template 並可批量套用（abilities、掛載介面
                         mount_anchor_name / tool_base_body_prim_suffix 等）。
                         區段頂部提供 Add Object Template 下拉框可批量套用。
          3. 模型屬性  — 加入物理引擎時輸入的參數（object 的 radius / mass /
                         friction / 模型路徑等），作為「物件屬性」底部緊貼的次級
        """
        hide_pose = self._should_hide_pose(data_source)

        # 判斷目前編輯的是否為 dict 容器角色（entity / ability_generated_object）。
        # dict 角色以 dict key（物件子角色 object_sub_role）識別，而非 id 欄位，
        # 因此隱藏 id 欄位（key 由上方編輯列維護）。
        current_path = None
        if self._current_temp_id in self._temp_registry:
            current_path = self._temp_registry[self._current_temp_id][0]
        is_dict_role = any(
            info["container"] == "dict" and info["path"] == current_path
            for info in self.role_infos.values()
        )

        role_fields = []
        object_fields = []
        special_fields = []

        for field_name, field_info in model_class.model_fields.items():
            if hide_pose and field_name in self._POSE_FIELDS_WHEN_HOST_BOUND:
                # 初始姿態由宿主繼承，隱藏編輯欄位（保留已存在值以免破壞舊配置）
                continue
            if is_dict_role and field_name == "id":
                # dict 角色的唯一鍵由上方編輯列維護，不在此重複編輯
                continue
            if field_name in (_MODEL_ATTR_FIELD, "solver_config"):
                special_fields.append(field_name)
            elif field_name in _ROLE_ATTR_FIELDS:
                role_fields.append(field_name)
            else:
                object_fields.append(field_name)

        # 角色模型（具備 type 判別欄位）才套用三區段排版；
        # 環境配置等不屬角色的模型維持原本平鋪排版
        is_role_model = "type" in model_class.model_fields
        # 只有樹上選中的頂層物件（具備 object 模型欄位）才顯示
        # Add Object Template 入口；嵌套模型由 _render_field 遞歸進入，
        # 不應出現模板選擇器
        is_top_level_object = (
            _MODEL_ATTR_FIELD in model_class.model_fields
            and self._current_temp_id in self._temp_registry
            and self._temp_registry[self._current_temp_id][2] is model_class
        )

        # 1. 角色屬性（與物件/模型屬性明顯區分，置於最頂部）
        if role_fields and is_role_model:
            self._add_section_header(
                TR.get("role_attr_section"), color="#b39ddb", border_color="#9575cd"
            )
            for field_name in role_fields:
                self._render_field(data_source, field_name, model_class.model_fields[field_name])
        elif role_fields:
            for field_name in role_fields:
                self._render_field(data_source, field_name, model_class.model_fields[field_name])

        # 2. 物件屬性（頂部提供 Add Object Template 批量套用入口，
        #    模型屬性緊貼其後渲染）
        if is_role_model and (object_fields or is_top_level_object):
            self._add_section_header(
                TR.get("object_attr_section"), color="#ffb74d", border_color="#fb8c00"
            )
            if is_top_level_object:
                self._add_object_template_selector(data_source)
            for field_name in object_fields:
                self._render_field(data_source, field_name, model_class.model_fields[field_name])
        elif object_fields:
            for field_name in object_fields:
                self._render_field(data_source, field_name, model_class.model_fields[field_name])

        # 3. 模型屬性（物件屬性內最底部的次級區域，緊貼物件屬性）
        for field_name in special_fields:
            self.render_special_union_section(
                field_name, data_source.get(field_name), data_source
            )

    def _add_object_template_selector(self, data_source):
        """在「物件屬性」區段頂部提供 Add Object Template 下拉框。

        選擇模板後一次套用模型屬性（object 欄位）與物件屬性
        （abilities、掛載介面等），再刷新整個編輯器。
        """
        template_combo = QComboBox()
        template_combo.addItem("None", None)
        self._object_templates = load_object_templates()
        for template_id, template_data in self._object_templates.items():
            template_combo.addItem(template_data.get("display_name", template_id), template_id)

        def on_template_selected(_label):
            template_id = template_combo.currentData()
            if not template_id:
                return
            template_data = self._object_templates.get(template_id)
            if not template_data:
                return
            # 1. 套用模型屬性（object 欄位：radius/mass/friction/模型路徑等）
            object_type = template_data.get("object_type", "usd")
            template_object = copy.deepcopy(template_data.get("object", {}))
            template_object["type"] = object_type
            data_source["object"] = template_object
            # 2. 批量套用物件屬性（僅套用「物件屬性」白名單內、且本角色模型具備的欄位）
            user_data = self._temp_registry.get(self._current_temp_id)
            role_model_cls = user_data[2] if user_data else None
            if role_model_cls is not None:
                for t_key, t_val in template_data.items():
                    if t_key in _OBJECT_TEMPLATE_RESERVED_KEYS:
                        continue
                    # 角色屬性白名單內的欄位是每個物件的通用配置，不隨模板套用
                    if t_key in _ROLE_ATTR_FIELDS:
                        continue
                    if t_key in role_model_cls.model_fields:
                        data_source[t_key] = copy.deepcopy(t_val)
            if self._current_temp_id:
                self.refresh_editor_by_temp_id(self._current_temp_id)

        template_combo.currentIndexChanged.connect(on_template_selected)
        self.attr_form.addRow(f"    {TR.get('add_object_template')}:", template_combo)

    def render_special_union_section(self, key, current_val, data_source):
        """渲染 object / solver_config 專用的 Dropdown 與彩色屬性區域"""
        # 標題與容器
        section_text = (
            TR.get("model_attr_section") if key == "object"
            else f"{key.upper()} Configuration:"
        )
        label = QLabel(section_text)
        
        # 根據 key 的種類，動態獲取 Registry 及框線樣式顏色
        if key == "object":
            raw_models = ObjectRegistry.get_all_models()
            border_color = "#a5d6a7"  # 綠色
        elif key == "solver_config":
            raw_models = SolverRegistry.get_all_models()
            border_color = "#90caf9"  # 藍色
        else:
            return

        label.setStyleSheet(f"font-weight: bold; color: {border_color}; margin: 10px;")
        self.attr_form.addRow(label)

        type_map = {}
        for m in raw_models:
            if 'type' in m.model_fields:
                t_name = m.model_fields['type'].default
                type_map[t_name] = m

        if not isinstance(current_val, dict):
            if type_map:
                first_type_name = list(type_map.keys())[0]
                current_val = get_pydantic_default(type_map[first_type_name])
                data_source[key] = current_val
            else:
                return 

        # 下拉選單
        combo = QComboBox()
        combo.addItems(list(type_map.keys()))
        combo.setCurrentText(current_val.get("type", ""))

        def on_type_changed(new_type_name):
            target_model = type_map[new_type_name]
            # 更新數據源並刷新畫面
            data_source[key] = copy.deepcopy(get_pydantic_default(target_model))
            # 修改點：使用內部 ID 刷新，解決閃退
            if self._current_temp_id:
                self.refresh_editor_by_temp_id(self._current_temp_id)

        combo.currentTextChanged.connect(on_type_changed)
        self.attr_form.addRow(f"    Select {key.capitalize()}:", combo)

        current_type_name = current_val.get("type")
        if current_type_name == "coupled":
            self._render_coupled_solver_config(current_val, border_color)
            return

        # 彩色標記區域 — 求解器參數可摺疊
        params_section = CollapsibleSection(
            TR.get("solver_params_section"),
            accent_color=border_color,
            expanded=True,
        )
        frame_layout = params_section.form_layout

        # 暫時替換渲染目標
        original_form = self.attr_form
        original_zebra = self.row_counter 
        self.attr_form = frame_layout

        current_type_name = current_val.get("type")
        model_cls = type_map.get(current_type_name)

        if model_cls:
            for f_name, f_info in model_cls.model_fields.items():
                if f_name == "type": continue
                f_val = current_val.get(f_name)
                if f_val is None:
                    default_value = get_editor_field_default(f_info)
                    if default_value is not _MISSING_FIELD_DEFAULT:
                        f_val = default_value
                        current_val[f_name] = f_val
                self.add_editable_row(f_name, f_val, current_val, use_zebra=False, field_type=f_info.annotation)

        self.attr_form = original_form
        self.row_counter = original_zebra 
        self.attr_form.addRow(params_section)

    def _ensure_coupled_solver_config_entry(self, coupled_val: dict, solver_name: str) -> dict:
        """確保 solver_configs 中存在指定求解器的配置字典（不含 type 欄位）。"""
        if coupled_val.get("solver_configs") is None:
            coupled_val["solver_configs"] = {}
        solver_configs = coupled_val["solver_configs"]
        if not isinstance(solver_configs, dict):
            solver_configs = {}
            coupled_val["solver_configs"] = solver_configs

        if solver_name not in solver_configs or not isinstance(solver_configs.get(solver_name), dict):
            handler = SolverRegistry.get_handler(solver_name)
            if handler is None:
                solver_configs[solver_name] = {}
            else:
                entry = copy.deepcopy(get_pydantic_default(handler.model_cls))
                entry.pop("type", None)
                solver_configs[solver_name] = entry
        return solver_configs[solver_name]

    def _sync_coupled_domain_fields(self, coupled_val: dict) -> tuple[str, str]:
        """同步 rigid_solver / soft_solver / solvers 三個欄位。"""
        rigid, soft = resolve_coupled_domains(coupled_val)
        coupled_val["rigid_solver"] = rigid
        coupled_val["soft_solver"] = soft
        coupled_val["solvers"] = coupled_solvers_list(rigid, soft)
        prune_coupled_solver_configs(coupled_val)
        return rigid, soft

    def _render_sub_solver_config_block(
        self,
        parent_layout: QFormLayout,
        coupled_val: dict,
        solver_name: str,
        accent_color: str,
    ) -> None:
        """在 Coupled 區塊內渲染單一子求解器的屬性表單。"""
        cfg = self._ensure_coupled_solver_config_entry(coupled_val, solver_name)
        handler = SolverRegistry.get_handler(solver_name)
        if handler is None:
            return

        model_cls = handler.model_cls
        defaults = get_pydantic_default(model_cls)
        defaults.pop("type", None)
        for field_name, default_val in defaults.items():
            if field_name not in cfg and default_val is not None:
                cfg[field_name] = default_val

        section_title = TR.get("solver_params_title").format(solver=solver_name.upper())
        params_section = CollapsibleSection(
            section_title,
            accent_color=accent_color,
            expanded=False,
        )
        sub_form = params_section.form_layout

        original_form = self.attr_form
        self.attr_form = sub_form
        for field_name, field_info in model_cls.model_fields.items():
            if field_name == "type":
                continue
            field_val = cfg.get(field_name)
            if field_val is None:
                default_value = get_editor_field_default(field_info)
                if default_value is not _MISSING_FIELD_DEFAULT:
                    field_val = default_value
                    cfg[field_name] = field_val
            self.add_editable_row(
                field_name,
                field_val,
                cfg,
                use_zebra=False,
                field_type=field_info.annotation,
            )
        self.attr_form = original_form
        parent_layout.addRow(params_section)

    def _render_coupled_solver_config(self, coupled_val: dict, border_color: str) -> None:
        """Coupled 求解器專用 UI：剛體/軟體下拉選單 + 各自屬性 + 耦合參數。"""
        rigid_name, soft_name = self._sync_coupled_domain_fields(coupled_val)
        if coupled_val.get("solver_configs") is None:
            coupled_val["solver_configs"] = {}
        solver_configs = coupled_val["solver_configs"]
        if not isinstance(solver_configs, dict):
            solver_configs = {}
            coupled_val["solver_configs"] = solver_configs

        self._ensure_coupled_solver_config_entry(coupled_val, rigid_name)
        self._ensure_coupled_solver_config_entry(coupled_val, soft_name)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ border-left: 4px solid {border_color};"
            f" background-color: #2b2b2b; margin: 5px 10px; padding: 5px; }}"
        )
        frame_layout = QFormLayout(frame)

        def on_rigid_changed(new_name: str) -> None:
            coupled_val["rigid_solver"] = new_name
            self._sync_coupled_domain_fields(coupled_val)
            self._ensure_coupled_solver_config_entry(coupled_val, new_name)
            if self._current_temp_id:
                self.refresh_editor_by_temp_id(self._current_temp_id)

        rigid_combo = QComboBox()
        rigid_combo.addItems(list(RIGID_CAPABLE_SOLVERS))
        if rigid_name in RIGID_CAPABLE_SOLVERS:
            rigid_combo.setCurrentText(rigid_name)
        rigid_combo.currentTextChanged.connect(on_rigid_changed)
        frame_layout.addRow(TR.get("coupled_rigid_solver"), rigid_combo)
        self._render_sub_solver_config_block(
            frame_layout, coupled_val, rigid_name, "#81c784"
        )

        def on_soft_changed(new_name: str) -> None:
            coupled_val["soft_solver"] = new_name
            self._sync_coupled_domain_fields(coupled_val)
            self._ensure_coupled_solver_config_entry(coupled_val, new_name)
            if self._current_temp_id:
                self.refresh_editor_by_temp_id(self._current_temp_id)

        soft_combo = QComboBox()
        soft_combo.addItems(list(SOFT_CAPABLE_SOLVERS))
        if soft_name in SOFT_CAPABLE_SOLVERS:
            soft_combo.setCurrentText(soft_name)
        soft_combo.currentTextChanged.connect(on_soft_changed)
        frame_layout.addRow(TR.get("coupled_soft_solver"), soft_combo)
        self._render_sub_solver_config_block(
            frame_layout, coupled_val, soft_name, "#ffb74d"
        )

        coupling_section = CollapsibleSection(
            TR.get("coupled_params_section"),
            accent_color="#90caf9",
            expanded=True,
        )

        original_form = self.attr_form
        self.attr_form = coupling_section.form_layout

        for field_name in ("coupling_mode", "mass_scale", "proxy_iterations"):
            field_info = CoupledSolverModel.model_fields[field_name]
            field_val = coupled_val.get(field_name)
            if field_val is None:
                default_value = get_editor_field_default(field_info)
                if default_value is not _MISSING_FIELD_DEFAULT:
                    field_val = default_value
                    coupled_val[field_name] = field_val
            self.add_editable_row(
                field_name,
                field_val,
                coupled_val,
                use_zebra=False,
                field_type=field_info.annotation,
            )
        self.attr_form = original_form
        frame_layout.addRow(coupling_section)
        self.attr_form.addRow(frame)

    def add_object_from_registry(self, info: dict, role_name: str = ""):
        model_cls = info["model"]
        path_key = info["path"]
        container_type = info["container"]
        new_data = copy.deepcopy(get_pydantic_default(model_cls))

        # list 容器角色以「物件 ID」(id 欄位) 識別；
        # dict 容器角色（entity/ability_generated_object）的 dict key 即為
        # 「物件子角色」（object_sub_role），key 本身即為該角色的唯一鍵。
        base = role_name or path_key.replace("_configs", "")

        if container_type == "list":
            if path_key not in self.current_yaml_data:
                self.current_yaml_data[path_key] = []
            new_id = f"{base}_{len(self.current_yaml_data[path_key])}"
            new_data["id"] = new_id
            if not new_data.get("name"):
                new_data["name"] = new_id
            self.current_yaml_data[path_key].append(new_data)
        else:
            if path_key not in self.current_yaml_data:
                self.current_yaml_data[path_key] = {}
            new_key = f"new_{base}_{len(self.current_yaml_data[path_key])}"
            self.current_yaml_data[path_key][new_key] = new_data
        self.refresh_object_tree()

    def delete_selected_object(self):
        item = self.obj_tree.currentItem()
        if not item: return
        tid = item.data(0, Qt.ItemDataRole.UserRole)
        user_data = self._temp_registry.get(tid)
        if not user_data: return

        data_type, index, _ = user_data
        if data_type == "env": return 
        if index is None or "_root" in str(index): return
        container = self.current_yaml_data.get(data_type)
        if isinstance(container, list) and isinstance(index, int):
            container.pop(index)
        elif isinstance(container, dict) and isinstance(index, str):
            container.pop(index)

        self._current_temp_id = None
        self.refresh_object_tree()
        self.clear_attr_form()
        self.preview_widget.update()

    def rename_dict_key(self, path_key, old_name, new_name):
        if old_name == new_name or not new_name.strip(): return
        configs = self.current_yaml_data.get(path_key, {})
        if new_name in configs: return 
        configs[new_name] = configs.pop(old_name)
        # dict key 即為物件子角色（object_sub_role），無需同步內部欄位
        # 更新 registry 中的 index (key) 以保持連貫性
        if self._current_temp_id in self._temp_registry:
            dt, idx, mc = self._temp_registry[self._current_temp_id]
            self._temp_registry[self._current_temp_id] = (dt, new_name, mc)
        self.refresh_object_tree()

    def _on_role_id_edited(self, data_type, index, new_id, data_source):
        """物件 ID 欄位編輯後刷新樹（list 容器角色以 id 欄位為物件 ID）。

        dict 容器角色（entity/ability_generated_object）以物件子角色
        object_sub_role（dict key）識別，不透過 id 欄位改名。
        """
        self.refresh_object_tree()

    def create_vector_input(self, key, labels, data_source, is_range=False, is_int=False):
        container = QWidget()
        v_layout = QVBoxLayout(container); v_layout.setContentsMargins(0, 0, 0, 0); v_layout.setSpacing(4)
        current_list = data_source.get(key, [])
        if not isinstance(current_list, list): current_list = [current_list]
        while len(current_list) < len(labels): current_list.append(0 if is_int else 0.0)

        for i, lbl_text in enumerate(labels):
            row_widget = QWidget()
            h_layout = QHBoxLayout(row_widget); h_layout.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(lbl_text); lbl.setFixedWidth(20) 
            h_layout.addWidget(lbl)

            if is_range:
                val = current_list[i] if isinstance(current_list[i], list) else [current_list[i], current_list[i]]
                if not isinstance(current_list[i], list): current_list[i] = [val[0], val[1]]

                def make_range_updater(idx, sub_idx):
                    return lambda v: [current_list[idx].__setitem__(sub_idx, v), self.preview_widget.update()]

                spins = []
                for sub_idx in [0, 1]:
                    spin = QSpinBox() if is_int else FlexibleDoubleSpinBox()
                    if is_int: spin.setRange(-9999, 9999)
                    spin.setValue(int(val[sub_idx]) if is_int else float(val[sub_idx]))
                    spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                    spin.valueChanged.connect(make_range_updater(i, sub_idx))
                    h_layout.addWidget(spin)
            else:
                val = current_list[i]
                spin = QSpinBox() if is_int else FlexibleDoubleSpinBox()
                if is_int: spin.setRange(-9999, 9999)
                if val is not None:
                    spin.setValue(int(val) if is_int else float(val))
                spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                def make_updater(idx):
                    return lambda v: [current_list.__setitem__(idx, v), self.preview_widget.update()]
                spin.valueChanged.connect(make_updater(i))
                h_layout.addWidget(spin)
            v_layout.addWidget(row_widget)
        return container

    def add_editable_row(self, key, value, data_source, use_zebra=True, field_type=None):
        if key == "id":
            label_text = TR.get("obj_id")
        else:
            label_text = TR.get(key) if key in ("controller", "start_attached", "host_player_id") else f"{key}:"
        update_call = self.preview_widget.update 
        # 修改：將 default_rotation 移出 generic 列表，單獨處理以支持 Random Range
        vector_keys = ["space_xyz", "gravity", "default_velocity", "default_angular_velocity", "size", "color"]
        
        is_int = False
        if field_type:
            origin = get_origin(field_type)
            args = get_args(field_type)
            if field_type is int or (origin is list and args and args[0] is int):
                is_int = True
        if key == "color": is_int = True 

        widget = None

        if key == "controller":
            widget = QComboBox()
            widget.addItems(["Human", "RL", "Bot"])
            current = str(data_source.get(key, "Human"))
            if current not in ("Human", "RL", "Bot"):
                current = "Human"
                data_source[key] = current
            widget.setCurrentText(current)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            widget.currentTextChanged.connect(
                lambda v: [data_source.update({key: v}), update_call()]
            )

        elif key == "abilities":
            all_abilities = list(Ability._registry.keys())
            current_vals = data_source.get(key, [])
            param_fields = _ability_param_fields(_load_abilities_schema())
            widget = AbilityConfigEditor(all_abilities, current_vals, param_fields)
            widget.valueChanged.connect(
                lambda: [data_source.update({key: widget.get_value()}), update_call()]
            )

        elif key == "host_player_id":
            # 明確指定要連接的宿主物件（依 player_configs 的物件 ID / id 欄位）
            widget = QComboBox()
            widget.addItem(TR.get("tool_host_none"), None)
            for p_cfg in self.current_yaml_data.get("player_configs", []):
                p_id = str(p_cfg.get("id") or p_cfg.get("name") or "")
                if p_id:
                    widget.addItem(p_id, p_id)
            current = data_source.get(key)
            if current:
                idx = widget.findData(current)
                widget.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                widget.setCurrentIndex(0)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

            def on_host_player_id_changed(_idx):
                selected = widget.currentData()
                data_source[key] = selected
                if selected and data_source.get("start_attached"):
                    # 明確指定連接且啟動即掛載時，初始姿態由宿主繼承，從配置中移除
                    for pose_key in self._POSE_FIELDS_WHEN_HOST_BOUND:
                        data_source.pop(pose_key, None)
                # 依 start_attached 狀態隱藏/重新顯示初始姿態欄位
                if self._current_temp_id:
                    self.refresh_editor_by_temp_id(self._current_temp_id)

            widget.currentIndexChanged.connect(on_host_player_id_changed)

        elif key in vector_keys:
            current_val = data_source.get(key, [])
            if key == "color":
                labels = ["R:", "G:", "B:"]
            else:
                labels = ["X:", "Y:", "Z:"] if len(current_val) >= 3 else (["W:", "L:"] if len(current_val) == 2 else ["V:"])
            widget = self.create_vector_input(key, labels, data_source, is_int=is_int)

        elif key == "separation":
            # 專屬屬性 separation 修改為 XYZ 輸入方式，但不含 Random Range
            widget = self.create_vector_input(key, ["X:", "Y:", "Z:"], data_source, is_int=is_int)

        elif key == "damping":
            widget = self.create_vector_input(key, ["線:", "旋:"], data_source, is_int=is_int)

        elif key == "default_position":
            container = QWidget(); v_layout = QVBoxLayout(container); v_layout.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox("Random Range")
            is_random = isinstance(value[0], list) if (isinstance(value, list) and len(value)>0) else False
            cb.setChecked(is_random)
            cb.stateChanged.connect(lambda s: self.toggle_random_pos(s == Qt.CheckState.Checked.value, data_source))
            v_layout.addWidget(cb)
            pos_widget = self.create_vector_input(key, ["X:", "Y:", "Z:"], data_source, is_range=is_random, is_int=is_int)
            v_layout.addWidget(pos_widget)
            widget = container

        elif key == "default_rotation":
            # 新增：default_rotation 支持 Random Range 選項，保持 W L 標籤
            container = QWidget(); v_layout = QVBoxLayout(container); v_layout.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox("Random Range")
            is_random = isinstance(value[0], list) if (isinstance(value, list) and len(value)>0) else False
            cb.setChecked(is_random)
            cb.stateChanged.connect(lambda s: self.toggle_random_rot(s == Qt.CheckState.Checked.value, data_source))
            v_layout.addWidget(cb)
            rot_widget = self.create_vector_input(key, ["W:", "L:"], data_source, is_range=is_random, is_int=is_int)
            v_layout.addWidget(rot_widget)
            widget = container

        elif key == "start_attached":
            # 專屬的「啟動時自動掛載」選擇框：勾選 = 環境啟動（及每次 reset）後工具自動掛載
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

            def on_start_attached_changed(s):
                data_source[key] = (s == Qt.CheckState.Checked.value)
                update_call()
                # 取消 start_attached 後初始姿態欄位重新出現
                if self._current_temp_id:
                    self.refresh_editor_by_temp_id(self._current_temp_id)

            widget.stateChanged.connect(on_start_attached_changed)

        elif isinstance(value, bool):
            widget = QComboBox(); widget.addItems(["True", "False"]); widget.setCurrentText(str(value))
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            widget.currentTextChanged.connect(lambda v: [data_source.update({key: v == "True"}), update_call()])

        elif is_int or isinstance(value, int):
            widget = QSpinBox(); widget.setRange(-99999, 99999); 
            if value is not None: widget.setValue(int(value))
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            widget.valueChanged.connect(lambda v: [data_source.update({key: v}), update_call()])

        elif isinstance(value, (float)):
            widget = FlexibleDoubleSpinBox(); widget.setValue(float(value))
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            widget.valueChanged.connect(lambda v: [data_source.update({key: v}), update_call()])

        else:
            widget = QLineEdit(str(value)); widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            widget.textChanged.connect(lambda v: [data_source.update({key: v}), update_call()])
            if key == "name":
                widget.editingFinished.connect(lambda: self.refresh_object_tree())
            elif key == "id":
                # 物件 ID 欄位：dict 容器同步 key，list 容器直接更新顯示
                user_data = self._temp_registry.get(self._current_temp_id)
                data_type, index = (user_data[0], user_data[1]) if user_data else (None, None)
                widget.editingFinished.connect(
                    lambda: self._on_role_id_edited(data_type, index, widget.text(), data_source)
                )

        if use_zebra:
            self.add_zebra_row(label_text, widget)
        else:
            self.attr_form.addRow(label_text, widget)

    def toggle_random_pos(self, state, data_source):
        old_pos = data_source["default_position"]
        new_pos = []
        if state:
            for v in old_pos: new_pos.append([v, v] if not isinstance(v, list) else v)
        else:
            for v in old_pos: new_pos.append(v[0] if isinstance(v, list) else v)
        data_source["default_position"] = new_pos
        # 使用內部 ID 刷新
        if self._current_temp_id:
            self.refresh_editor_by_temp_id(self._current_temp_id)

    def toggle_random_rot(self, state, data_source):
        """處理旋轉屬性的 Random Range 切換"""
        old_rot = data_source["default_rotation"]
        new_rot = []
        if state:
            for v in old_rot: new_rot.append([v, v] if not isinstance(v, list) else v)
        else:
            for v in old_rot: new_rot.append(v[0] if isinstance(v, list) else v)
        data_source["default_rotation"] = new_rot
        if self._current_temp_id:
            self.refresh_editor_by_temp_id(self._current_temp_id)

    def update_list_data(self, key, text, data_source):
        try:
            new_val = ast.literal_eval(text)
            if isinstance(new_val, list): data_source[key] = new_val
        except: pass

    def _prune_coupled_solver_config_in_yaml(self) -> None:
        """保存前清理 coupled 配置中未使用的 solver_configs 條目。"""
        env = self.current_yaml_data.get("environment_configs", {})
        solver_config = env.get("solver_config")
        if isinstance(solver_config, dict):
            prune_coupled_solver_configs(solver_config)

    def save_and_exit(self):
        self._prune_coupled_solver_config_in_yaml()
        self.window().save_ui_settings(self.main_splitter.sizes(), self.right_splitter.sizes())
        if self.current_file_path:
            with open(self.current_file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.current_yaml_data, f, allow_unicode=True, sort_keys=False)
        self.window().switch_page(0)

    def exit_without_saving(self):
        self.window().switch_page(0)

    def retranslate_ui(self):
        super().retranslate_ui()
        self.list_title.setText(TR.get("obj_list"))
        self.attr_title.setText(TR.get("attr_edit"))
        self.back_btn.setText(TR.get("save_back"))
        self.cancel_btn.setText(TR.get("cancel_no_save"))
        self.btn_delete.setText(TR.get("delete_obj"))





# --- 設置頁面 ---
class SettingsPage(BasePage):
    def __init__(self, main_app):
        super().__init__("sys_settings")
        self.main_app = main_app
        
        # 顯示三行自動計算的路徑 (唯讀提示)
        self.preview_group = QFrame()
        self.preview_group.setStyleSheet("color: #888; font-size: 14px; padding: 5px;")
        preview_layout = QVBoxLayout(self.preview_group)
        self.lbl_level_path = QLabel()
        self.lbl_shape_path = QLabel()
        self.lbl_reward_path = QLabel()
        preview_layout.addWidget(self.lbl_level_path)
        preview_layout.addWidget(self.lbl_shape_path)
        preview_layout.addWidget(self.lbl_reward_path)
        self.layout.addWidget(self.preview_group)

        self.font_size_label = QLabel()
        self.layout.addWidget(self.font_size_label)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 40)
        self.font_size_spin.setValue(18)
        self.font_size_spin.valueChanged.connect(self.main_app.apply_global_settings)
        self.layout.addWidget(self.font_size_spin)

        self.lang_label = QLabel()
        self.layout.addWidget(self.lang_label)
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("繁體中文", "zh")
        self.lang_combo.addItem("English", "en")
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        self.layout.addWidget(self.lang_combo)

        self.layout.addStretch()
        self.save_btn = QPushButton()
        self.save_btn.clicked.connect(self.save_and_back)
        self.layout.addWidget(self.save_btn)
        
        self.update_path_previews() # 初始化顯示
        self.retranslate_ui()

    # --- 內部路徑處理邏輯 ---
    def get_project_path(self):
        return self.main_app.project_root

    def get_level_path(self):
        return os.path.join(self.get_project_path(), "game", "script", "levels")

    def get_shape_path(self):
        return os.path.join(self.get_project_path(), "game", "script", "role", "shapes")

    def get_reward_path(self):
        return os.path.join(self.get_project_path(), "game", "script", "levels", "rewards")

    def update_path_previews(self):
        """更新 UI 上的三行路徑文字"""
        self.lbl_level_path.setText(f"{TR.get('calc_level_path')} {self.get_level_path()}")
        self.lbl_shape_path.setText(f"{TR.get('calc_shape_path')} {self.get_shape_path()}")
        self.lbl_reward_path.setText(f"{TR.get('calc_reward_path')} {self.get_reward_path()}")

    def retranslate_ui(self):
        super().retranslate_ui()
        self.font_size_label.setText(TR.get("font_size"))
        self.lang_label.setText(TR.get("language_select"))
        self.save_btn.setText(TR.get("save_back"))
        self.update_path_previews()

    def on_language_changed(self):
        TR.load_lang(self.lang_combo.currentData())
        self.main_app.retranslate_all()

    def save_and_back(self):
        config_data = {
            "font_size": self.font_size_spin.value(),
            "language": self.lang_combo.currentData()
        }
        self.main_app.global_config.update(config_data)
        self.main_app.save_app_settings()
        self.main_app.switch_page(0)





class TestPage(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.layout = QVBoxLayout(self)
        
        self.title = QLabel(TR.get("test_settings"))
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        self.layout.addWidget(self.title)

        # 表單區域
        self.form_container = QScrollArea()
        self.form_container.setWidgetResizable(True)
        self.form_widget = QWidget()
        self.form_layout = QFormLayout(self.form_widget)
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        # render_mode
        self.combo_render = QComboBox()
        self.combo_render.addItems(["window", "headless"])
        self.form_layout.addRow(TR.get("render_mode"), self.combo_render)

        # model_obs_type
        self.combo_obs_type = QComboBox()
        # self.combo_obs_type.addItems(["game_screen", "state_based", "mixed"]) # TODO
        self.combo_obs_type.addItems(["state_based"])
        self.combo_obs_type.setCurrentText("state_based")
        self.form_layout.addRow(TR.get("model_obs_type"), self.combo_obs_type)

        # # obs_width / height # TODO
        # self.spin_width = QSpinBox(); self.spin_width.setRange(1, 4000); self.spin_width.setValue(400)
        # self.spin_height = QSpinBox(); self.spin_height.setRange(1, 4000); self.spin_height.setValue(225)
        # self.form_layout.addRow(TR.get("obs_width"), self.spin_width)
        # self.form_layout.addRow(TR.get("obs_height"), self.spin_height)

        # max_episode_step / num_env
        self.spin_max_step = QSpinBox(); self.spin_max_step.setRange(1, 100000); self.spin_max_step.setValue(3000)
        self.spin_num_env = QSpinBox(); self.spin_num_env.setRange(1, 10000); self.spin_num_env.setValue(1)
        self.form_layout.addRow(TR.get("max_episode_step"), self.spin_max_step)
        self.form_layout.addRow(TR.get("num_env"), self.spin_num_env)

        # # capture_per_second # TODO
        # self.input_cps = QLineEdit("None")
        # self.input_cps.setPlaceholderText(TR.get("none_tip"))
        # self.form_layout.addRow(TR.get("capture_per_second"), self.input_cps)

        # requires_grad (Bool)
        self.check_grad = QCheckBox(); self.check_grad.setChecked(False)
        self.form_layout.addRow(TR.get("requires_grad"), self.check_grad)


        # is_lock_fps
        self.check_lock_fps = QCheckBox(); self.check_lock_fps.setChecked(True)
        self.form_layout.addRow(TR.get("is_lock_fps"), self.check_lock_fps)

        self.form_container.setWidget(self.form_widget)
        self.layout.addWidget(self.form_container)

        # 按鈕欄
        btn_layout = QHBoxLayout()
        self.btn_back = QPushButton(TR.get("cancel"))
        self.btn_back.clicked.connect(lambda: self.main_app.switch_page(0))
        
        self.btn_start = QPushButton(TR.get("start_test"))
        self.btn_start.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 10px;")
        self.btn_start.clicked.connect(self.run_test_process)
        
        btn_layout.addWidget(self.btn_back)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_start)
        self.layout.addLayout(btn_layout)

        # 用於儲存當前關卡資訊
        self.target_level = 0
        self.target_sub_level = 0

    def set_target(self, level, sub_level):
        self.target_level = level
        self.target_sub_level = sub_level

    def run_test_process(self):
        # 獲取參數
        try:
            # cps = None if self.input_cps.text().lower() == "none" else int(self.input_cps.text()) # TODO
            cps = None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"參數格式錯誤: {e}")
            return

        # 獲取項目路徑並加入 sys.path
        project_path = self.main_app.settings_page.get_project_path()
        if project_path not in sys.path:
            sys.path.append(project_path)

        # 準備參數字典
        cfg = {
            "render_mode": self.combo_render.currentText(),
            "model_obs_type": self.combo_obs_type.currentText(),
            # "obs_width": self.spin_width.value(), # TODO
            # "obs_height": self.spin_height.value(),
            "obs_width": 0,
            "obs_height": 0,
            "max_episode_step": self.spin_max_step.value(),
            "num_env": self.spin_num_env.value(),
            "capture_per_second": cps,
            "requires_grad": self.check_grad.isChecked(),
            "is_lock_fps": self.check_lock_fps.isChecked(),
            "level": self.target_level,
            "sub_level": self.target_sub_level
        }

        try:
            # 執行測試邏輯 (這裡我們呼叫一個獨立的方法，方便維護)
            p = mp.Process(target=execute_game_logic, args=(cfg,))
            p.start()
        except Exception as e:
            QMessageBox.critical(self, "Runtime Error", f"無法啟動遊戲:\n{e}")
        
def execute_game_logic(cfg):
    from queue import Queue
    from game.script.game import Game
    from game.script.simulate.physics_manager import PhysicsManager
    from game.script.custom_viewergl import CustomViewerGL

    DEVICE = "cuda:0" # TODO Hardcode

    event_is_window_setup_ready = mp.Event()
    event_is_game_logic_keymapping_setup_ready = mp.Event()

    physics_manager_state_queue = Queue(maxsize=1)
    human_input_queue = Queue(maxsize=1)

    viewerGL = CustomViewerGL(
        event_is_window_setup_ready=event_is_window_setup_ready, 
        human_input_queue=human_input_queue,
        follow_body_index=0, 
    )
    physics_manager = PhysicsManager(device=DEVICE, viewerGL=viewerGL)

    game = Game(
        render_mode=cfg["render_mode"], 
        model_obs_type=cfg["model_obs_type"],
        obs_width=cfg["obs_width"],
        obs_height=cfg["obs_height"],
        device=DEVICE,
        physics_manager=physics_manager,
        max_episode_step=cfg["max_episode_step"],
        player_configs=None, 
        platform_configs=None,
        environment_configs=None,
        num_env=cfg["num_env"],
        level=cfg["level"], 
        sub_level=cfg["sub_level"], 
        capture_per_second=cfg["capture_per_second"],
        requires_grad=cfg["requires_grad"],
    )

    game.run_game_human(
        event_is_window_setup_ready=event_is_window_setup_ready, 
        event_is_game_logic_keymapping_setup_ready=event_is_game_logic_keymapping_setup_ready, 
        physics_manager_state_queue=physics_manager_state_queue, 
        human_input_queue=human_input_queue,
        is_lock_fps=cfg["is_lock_fps"]
    )





# --- 主窗口 ---
class MainWindow(QMainWindow):
    def __init__(self, project_root: str | None = None):
        super().__init__()
        self.project_root = project_root or str(Path(__file__).resolve().parent.parent.parent.parent)
        self.global_config = {}
        self.resize(1200, 800)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # 初始化頁面
        self.init_main_ui()
        self.settings_page = SettingsPage(self)
        self.edit_page = EditPage()
        self.test_page = TestPage(self)
        self.train_page = ExperimentHubPage(self, TR.get)
        
        # 註冊頁面
        self.stack.addWidget(self.main_widget)   # 0
        self.stack.addWidget(self.settings_page) # 1
        self.stack.addWidget(self.edit_page)     # 2
        self.stack.addWidget(self.train_page)    # 3
        self.stack.addWidget(self.test_page)     # 4
        
        self.check_initial_config()
        self.update_button_states() # 初始化按鈕狀態

    def init_main_ui(self):
        self.main_widget = QWidget()
        layout = QVBoxLayout(self.main_widget)
        
        self.set_btn = QPushButton()
        self.set_btn.clicked.connect(lambda: self.switch_page(1))
        layout.addWidget(self.set_btn)
        
        h = QHBoxLayout()
        self.list_label = QLabel()
        self.add_btn = QPushButton("+")
        self.add_btn.setFixedWidth(80)
        self.add_btn.clicked.connect(self.show_add_menu)
        h.addWidget(self.list_label); h.addStretch(); h.addWidget(self.add_btn)
        layout.addLayout(h)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemSelectionChanged.connect(self.update_button_states)
        layout.addWidget(self.tree)
        
        b = QHBoxLayout()
        self.edit_btn = QPushButton()
        self.train_btn = QPushButton()
        self.test_btn = QPushButton()
        
        self.edit_btn.clicked.connect(self.go_to_edit)
        self.train_btn.clicked.connect(self.go_to_train)
        self.test_btn.clicked.connect(self.go_to_test)
        
        b.addWidget(self.edit_btn); 
        b.addWidget(self.train_btn);
        b.addWidget(self.test_btn)
        layout.addLayout(b)

    def update_button_states(self):
        """根據選擇決定按鈕是否可用"""
        s = self.tree.selectedItems()
        # 只有選中子關卡（即有父節點的項）時，按鈕才啟用
        is_sub_level = bool(s and s[0].parent())
        self.edit_btn.setEnabled(is_sub_level)
        self.train_btn.setEnabled(is_sub_level)
        self.test_btn.setEnabled(is_sub_level)

    def get_selected_indices(self):
        """解析選中項的 level index 和 sub_level index"""
        s = self.tree.selectedItems()
        if not s or not s[0].parent(): return None, None
        
        try:
            sub_text = s[0].text(0) # "子關卡 0"
            sub_idx = int(re.search(r'\d+', sub_text).group())
            
            series_name = s[0].parent().text(0) 
            series_idx = int(re.search(r'\d+', series_name).group())
            return series_idx, sub_idx
        except:
            return None, None

    def go_to_train(self):
        lv, sub_lv = self.get_selected_indices()
        if lv is not None:
            self.train_page.set_target(lv, sub_lv)
            self.switch_page(3)

    def go_to_test(self):
        lv, sub_lv = self.get_selected_indices()
        if lv is not None:
            self.test_page.set_target(lv, sub_lv)
            self.switch_page(4)

    def go_to_edit(self):
        lv, sub_lv = self.get_selected_indices()
        if lv is not None:
            series_name = f"level{lv}"
            path = os.path.join(self.settings_page.get_level_path(), series_name, f"level_{lv}_{sub_lv}_default_cfg.yaml")
            self.edit_page.load_config(path)
            self.switch_page(2)

    def show_add_menu(self):
        m = QMenu(self); a1 = m.addAction(TR.get("new_sub_level")); a2 = m.addAction(TR.get("new_level_series"))
        a = m.exec(self.add_btn.mapToGlobal(self.add_btn.rect().bottomLeft()))
        if a == a1: self.handle_new_sub_level()
        elif a == a2: self.handle_new_series()

    def handle_new_sub_level(self):
        root = self.settings_page.get_level_path()
        if not os.path.exists(root): os.makedirs(root, exist_ok=True)
        dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and re.match(r'level\d+', d)])
        if not dirs: return
        d = NewSubLevelDialog(dirs, self)
        if d.exec() == QDialog.DialogCode.Accepted:
            sel = d.get_selected_series(); s_path = os.path.join(root, sel); s_idx = re.search(r'level(\d+)', sel).group(1)
            subs = [int(re.search(fr'level_{s_idx}_(\d+)_default_cfg\.yaml', f).group(1)) for f in os.listdir(s_path) if re.search(fr'level_{s_idx}_(\d+)_default_cfg\.yaml', f)]
            nxt = max(subs) + 1 if subs else 0
            p = os.path.join(s_path, f"level_{s_idx}_{nxt}_default_cfg.yaml")
            with open(p, 'w', encoding='utf-8') as f: yaml.dump({}, f)
            self.update_level_list(root); self.edit_page.load_config(p); self.switch_page(2)

    def handle_new_series(self):
        root = self.settings_page.get_level_path()
        if not os.path.exists(root): os.makedirs(root, exist_ok=True)
        idxs = [int(re.search(r'level(\d+)', d).group(1)) for d in os.listdir(root) if re.search(r'level(\d+)', d)]
        nxt = max(idxs) + 1 if idxs else 0
        s_path = os.path.join(root, f"level{nxt}"); os.makedirs(s_path, exist_ok=True)
        p = os.path.join(s_path, f"level_{nxt}_0_default_cfg.yaml")
        with open(p, 'w', encoding='utf-8') as f: yaml.dump({}, f)
        self.update_level_list(root); self.edit_page.load_config(p); self.switch_page(2)

    def retranslate_all(self):
        self.setWindowTitle(TR.get("window_title")); self.set_btn.setText(TR.get("sys_settings")); self.list_label.setText(TR.get("level_list")); self.add_btn.setText(f"{TR.get('add_new')} +")
        self.edit_btn.setText(TR.get("edit")); 
        self.train_btn.setText(TR.get("train"));
        self.test_btn.setText(TR.get("test"))
        for i in range(self.stack.count()):
            p = self.stack.widget(i)
            if hasattr(p, "retranslate_ui"): p.retranslate_ui()
            
        r = self.settings_page.get_level_path()
        if os.path.exists(r): self.update_level_list(r)

    def apply_global_settings(self, size): self.setStyleSheet(f"QWidget {{ font-size: {size}px; }}")

    def check_initial_config(self):
        config_path = os.path.join(self.project_root, "app_settings.yaml")
        if os.path.exists(config_path):
            self.load_app_config(config_path)
        else:
            self.retranslate_all()
            self.switch_page(1)

    def save_ui_settings(self, main_sizes, right_sizes):
        self.global_config["splitter_main"] = main_sizes
        self.global_config["splitter_right"] = right_sizes
        self.save_app_settings()

    def save_app_settings(self):
        config_path = os.path.join(self.project_root, "app_settings.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.global_config, f, allow_unicode=True)

    def load_app_config(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                d = yaml.safe_load(f) or {}
                self.global_config = {
                    k: v for k, v in d.items()
                    if k not in ("project_path", "config_dir")
                }
                if d:
                    TR.load_lang(d.get("language", "zh"))
                    self.settings_page.font_size_spin.setValue(d.get("font_size", 18))
                    self.apply_global_settings(d.get("font_size", 18))
                    if "splitter_main" in d: self.edit_page.main_splitter.setSizes(d["splitter_main"])
                    if "splitter_right" in d: self.edit_page.right_splitter.setSizes(d["splitter_right"])
                    self.retranslate_all(); self.switch_page(0)
        except: self.retranslate_all(); self.switch_page(1)

    def switch_page(self, i): self.stack.setCurrentIndex(i)

    def update_level_list(self, root):
        self.tree.clear()
        if not os.path.exists(root): return
        for subdir in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]):
            m = re.fullmatch(r'level(\d+)', subdir)
            if m:
                p = QTreeWidgetItem(self.tree, [subdir]); s_idx = m.group(1)
                subs = sorted([int(re.match(fr'level_{s_idx}_(\d+)_default_cfg\.yaml', f).group(1)) for f in os.listdir(os.path.join(root, subdir)) if re.match(fr'level_{s_idx}_(\d+)_default_cfg\.yaml', f)])
                for idx in subs: QTreeWidgetItem(p, [f"{TR.get('sub_item')} {idx}"])
                p.setExpanded(False)





if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(project_root=str(Path(__file__).resolve().parent.parent.parent.parent))
    window.show()
    sys.exit(app.exec())