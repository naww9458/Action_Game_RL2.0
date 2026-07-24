"""Schema-driven form editor for training preset YAML files."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import yaml

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "training" / "preset_editor_schema.yaml"


def _load_schema() -> dict:
    with open(_schema_path(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_nested(data: dict, path: list[str]) -> dict:
    node = data
    for key in path:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    return node


def _get_nested_value(data: dict, path: list[str], key: str) -> Any:
    node = _get_nested(data, path) if path else data
    if "." in key:
        parts = key.split(".")
        cur = node
        for part in parts[:-1]:
            if not isinstance(cur.get(part), dict):
                cur[part] = {}
            cur = cur[part]
        return cur.get(parts[-1])
    return node.get(key)


def _set_nested_value(data: dict, path: list[str], key: str, value: Any) -> None:
    node = _get_nested(data, path) if path else data
    if "." in key:
        parts = key.split(".")
        cur = node
        for part in parts[:-1]:
            if not isinstance(cur.get(part), dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value
        return
    node[key] = value


def _reward_component_options() -> list[str]:
    try:
        from training.reward_imports import ensure_all_rewards_registered
        from script.levels.rewards.reward_calculator import RewardComponent

        ensure_all_rewards_registered()
        return RewardComponent.get_registered_names()
    except Exception:
        return []


class _MultiSelectComboBox(QComboBox):
    selection_changed = pyqtSignal(list)

    def __init__(self, items: list[str], selected_items: list[str], parent=None):
        self.list_widget: QListWidget | None = None
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().hide()

        self._display_text = QPlainTextEdit()
        self._display_text.setReadOnly(True)
        self._display_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._display_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._display_text.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 2, 25, 2)
        layout.addWidget(self._display_text)

        self.list_widget = QListWidget()
        for item in items:
            lw_item = QListWidgetItem(item, self.list_widget)
            lw_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            state = Qt.CheckState.Checked if item in selected_items else Qt.CheckState.Unchecked
            lw_item.setCheckState(state)

        self.setModel(self.list_widget.model())
        self.setView(self.list_widget)
        self.list_widget.viewport().installEventFilter(self)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self._update_text()

    def eventFilter(self, obj, event):
        if self.list_widget is None:
            return super().eventFilter(obj, event)
        if obj == self.list_widget.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            index = self.list_widget.indexAt(event.pos())
            if index.isValid():
                item = self.list_widget.item(index.row())
                new_state = (
                    Qt.CheckState.Unchecked
                    if item.checkState() == Qt.CheckState.Checked
                    else Qt.CheckState.Checked
                )
                item.setCheckState(new_state)
                return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        self.showPopup()

    def _on_item_changed(self):
        self._update_text()
        self.selection_changed.emit(self.selected())

    def _update_text(self):
        selected = self.selected()
        self._display_text.setPlainText("\n".join(selected))
        metrics = self.fontMetrics()
        line_height = metrics.lineSpacing()
        line_count = max(1, len(selected))
        self._display_text.setFixedHeight(line_count * line_height + 8)

    def selected(self) -> list[str]:
        if self.list_widget is None:
            return []
        result = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                result.append(item.text())
        return result

    def set_selected(self, values: list[str]) -> None:
        if self.list_widget is None:
            return
        value_set = set(values)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.text() in value_set else Qt.CheckState.Unchecked
            )
        self._update_text()


class _FieldBinding:
    def __init__(
        self,
        section_id: str,
        getter: Callable[[], Any],
        setter: Callable[[Any], None],
        widget: QWidget,
        field_spec: dict,
        path: list[str],
        refresh_visibility: Callable[[], None] | None = None,
    ):
        self.section_id = section_id
        self.getter = getter
        self.setter = setter
        self.widget = widget
        self.field_spec = field_spec
        self.path = path
        self.refresh_visibility = refresh_visibility


class PresetFormEditor(QWidget):
    """Collapsible three-section editor for training preset YAML."""

    data_changed = pyqtSignal()

    def __init__(self, lang: str = "zh", parent=None):
        super().__init__(parent)
        self._lang = lang if lang in ("zh", "en") else "zh"
        self._schema = _load_schema()
        self._data: dict[str, Any] = {}
        self._bindings: list[_FieldBinding] = []
        self._section_boxes: dict[str, QGroupBox] = {}

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._form_host = QWidget()
        self._form_layout = QVBoxLayout(self._form_host)
        self._form_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(self._form_host)
        root_layout.addWidget(scroll)

        self._build_sections()

    def set_language(self, lang: str) -> None:
        self._lang = lang if lang in ("zh", "en") else "zh"
        self._rebuild_ui()

    def _label(self, spec: dict) -> str:
        key = f"label_{self._lang}"
        return spec.get(key) or spec.get("label_en") or spec.get("id", "")

    def _description(self, field_spec: dict) -> str:
        key = f"description_{self._lang}"
        return field_spec.get(key) or field_spec.get("description_en") or ""

    def _rebuild_ui(self) -> None:
        data = copy.deepcopy(self._data)
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._bindings.clear()
        self._section_boxes.clear()
        self._build_sections()
        self.set_data(data)

    def _build_sections(self) -> None:
        for section in self._schema.get("sections", []):
            box = QGroupBox(self._label(section))
            box.setCheckable(True)
            box.setChecked(section.get("default_expanded", True))
            box.toggled.connect(lambda checked, b=box: self._toggle_group_content(b, checked))
            section_layout = QVBoxLayout(box)

            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            self._add_fields(form, section.get("fields", []), section["id"], [])
            section_layout.addLayout(form)

            for group in section.get("groups", []):
                group_box = self._build_group_box(section["id"], group)
                section_layout.addWidget(group_box)

            self._wrap_group_content(box, section_layout)
            self._form_layout.addWidget(box)
            self._section_boxes[section["id"]] = box

    def _build_group_box(self, section_id: str, group: dict) -> QGroupBox:
        box = QGroupBox(self._label(group))
        box.setCheckable(True)
        box.setChecked(group.get("default_expanded", False))
        box.toggled.connect(lambda checked, b=box: self._toggle_group_content(b, checked))

        layout = QVBoxLayout(box)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        base_path = list(group.get("path", []))
        self._add_fields(form, group.get("fields", []), section_id, base_path)
        layout.addLayout(form)

        for subgroup in group.get("subgroups", []):
            sub_box = QGroupBox(self._label(subgroup))
            sub_box.setCheckable(True)
            sub_box.setChecked(True)
            sub_box.toggled.connect(lambda checked, b=sub_box: self._toggle_group_content(b, checked))
            sub_layout = QVBoxLayout(sub_box)
            sub_form = QFormLayout()
            sub_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            sub_path = base_path + list(subgroup.get("path", []))
            self._add_fields(sub_form, subgroup.get("fields", []), section_id, sub_path)
            sub_layout.addLayout(sub_form)
            self._wrap_group_content(sub_box, sub_layout)
            layout.addWidget(sub_box)

        self._wrap_group_content(box, layout)
        return box

    def _wrap_group_content(self, box: QGroupBox, inner_layout: QVBoxLayout) -> None:
        content = QWidget()
        content.setLayout(inner_layout)
        wrapper = QVBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(content)
        box.setLayout(wrapper)
        content.setVisible(box.isChecked())

    def _toggle_group_content(self, box: QGroupBox, checked: bool) -> None:
        layout = box.layout()
        if layout and layout.count() > 0:
            item = layout.itemAt(0)
            if item and item.widget():
                item.widget().setVisible(checked)

    def _add_fields(
        self,
        form: QFormLayout,
        fields: list[dict],
        section_id: str,
        nested_path: list[str],
    ) -> None:
        for field_spec in fields:
            widget, binding = self._create_field_widget(field_spec, section_id, nested_path)
            label = QLabel(field_spec["key"])
            desc = self._description(field_spec)
            if desc:
                label.setToolTip(desc)
                widget.setToolTip(desc)
            form.addRow(label, widget)
            self._bindings.append(binding)

    def _section_data(self, section_id: str) -> dict:
        if section_id not in self._data or not isinstance(self._data[section_id], dict):
            self._data[section_id] = {}
        return self._data[section_id]

    def _create_field_widget(
        self,
        field_spec: dict,
        section_id: str,
        nested_path: list[str],
    ) -> tuple[QWidget, _FieldBinding]:
        key = field_spec["key"]
        field_type = field_spec.get("type", "string")

        def getter() -> Any:
            node = self._section_data(section_id)
            return _get_nested_value(node, nested_path, key)

        def setter(value: Any) -> None:
            node = self._section_data(section_id)
            _set_nested_value(node, nested_path, key, value)
            self.data_changed.emit()

        widget: QWidget
        refresh_visibility: Callable[[], None] | None = None

        if field_type == "string":
            widget = QLineEdit()
            widget.textChanged.connect(lambda text: setter(text))

        elif field_type == "int":
            spin = QSpinBox()
            spin.setRange(field_spec.get("min", -999999999), field_spec.get("max", 999999999))
            spin.valueChanged.connect(lambda v: setter(v))
            widget = spin

        elif field_type == "float":
            spin = QDoubleSpinBox()
            spin.setDecimals(field_spec.get("decimals", 6))
            spin.setRange(field_spec.get("min", -1e12), field_spec.get("max", 1e12))
            spin.setSingleStep(10 ** -min(spin.decimals(), 4))
            spin.valueChanged.connect(lambda v: setter(v))
            widget = spin

        elif field_type == "bool":
            check = QCheckBox()
            check.stateChanged.connect(lambda state: setter(state == Qt.CheckState.Checked.value))
            widget = check

        elif field_type in ("enum", "enum_nullable"):
            combo = QComboBox()
            options = list(field_spec.get("options", []))
            if field_type == "enum_nullable":
                combo.addItem("null", None)
            for opt in options:
                combo.addItem(str(opt), opt)
            if field_spec.get("allow_custom"):
                combo.setEditable(True)
            def on_combo_changed(_idx: int, c=combo, ft=field_type) -> None:
                if ft == "enum_nullable" and c.currentData() is None:
                    setter(None)
                else:
                    setter(c.currentText())

            combo.currentIndexChanged.connect(on_combo_changed)
            widget = combo

        elif field_type == "multi_select_unique":
            options = self._resolve_options(field_spec)
            widget = _MultiSelectComboBox(options, [])
            widget.selection_changed.connect(lambda vals: setter(vals))

        elif field_type == "string_list":
            edit = QPlainTextEdit()
            edit.setPlaceholderText("RL\nBot")
            edit.setMaximumHeight(80)
            edit.textChanged.connect(lambda: setter(self._parse_string_list(edit.toPlainText())))
            widget = edit

        elif field_type == "yaml_mapping":
            edit = QPlainTextEdit()
            edit.setFont(edit.font())
            edit.textChanged.connect(lambda: self._on_yaml_mapping_changed(edit, setter))
            widget = edit

        else:
            widget = QLineEdit()
            widget.textChanged.connect(lambda text: setter(text))

        binding = _FieldBinding(section_id, getter, setter, widget, field_spec, nested_path, refresh_visibility)

        if field_spec.get("visible_when"):
            refresh_visibility = lambda b=binding: self._apply_visibility(b)
            binding.refresh_visibility = refresh_visibility
            self._wire_visibility_triggers(binding)

        return widget, binding

    def _resolve_options(self, field_spec: dict) -> list[str]:
        source = field_spec.get("options_source")
        if source == "reward_components":
            return _reward_component_options()
        return list(field_spec.get("options", []))

    def _parse_string_list(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines and "," in text:
            lines = [part.strip() for part in text.split(",") if part.strip()]
        return lines

    def _on_yaml_mapping_changed(self, edit: QPlainTextEdit, setter: Callable[[Any], None]) -> None:
        text = edit.toPlainText().strip()
        if not text:
            setter({})
            return
        try:
            value = yaml.safe_load(text)
            if value is None:
                value = {}
            if not isinstance(value, dict):
                return
            setter(value)
        except yaml.YAMLError:
            return

    def _wire_visibility_triggers(self, binding: _FieldBinding) -> None:
        condition = binding.field_spec.get("visible_when", {})
        dep_key = condition.get("field")
        if not dep_key:
            return

        for other in self._bindings:
            if other.field_spec.get("key") == dep_key and other.path == binding.path:
                if isinstance(other.widget, QComboBox):
                    other.widget.currentIndexChanged.connect(lambda: self._apply_visibility(binding))
                break

    def _apply_visibility(self, binding: _FieldBinding) -> None:
        condition = binding.field_spec.get("visible_when", {})
        dep_key = condition.get("field")
        if not dep_key:
            binding.widget.setVisible(True)
            return

        dep_value = None
        for other in self._bindings:
            if other.field_spec.get("key") == dep_key and other.path == binding.path:
                dep_value = other.getter()
                break

        visible = True
        if condition.get("not_null"):
            visible = dep_value is not None and dep_value not in ("null", "none", "")
        binding.widget.setVisible(visible)
        label = binding.widget.parent()
        # QLabel is sibling in QFormLayout; hide row via widget only

    def _set_widget_value(self, binding: _FieldBinding, value: Any) -> None:
        widget = binding.widget
        field_type = binding.field_spec.get("type", "string")

        if field_type == "string":
            widget.setText("" if value is None else str(value))

        elif field_type == "int":
            widget.setValue(int(value or 0))

        elif field_type == "float":
            widget.setValue(float(value or 0.0))

        elif field_type == "bool":
            widget.setChecked(bool(value))

        elif field_type in ("enum", "enum_nullable"):
            if value is None:
                idx = widget.findData(None)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            else:
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                elif widget.isEditable():
                    widget.setEditText(str(value))

        elif field_type == "multi_select_unique":
            widget.set_selected(list(value or []))

        elif field_type == "string_list":
            widget.setPlainText("\n".join(value or []))

        elif field_type == "yaml_mapping":
            mapping = value or {}
            widget.blockSignals(True)
            widget.setPlainText(
                yaml.dump(mapping, default_flow_style=False, sort_keys=False, allow_unicode=True)
                if mapping
                else ""
            )
            widget.blockSignals(False)

        if binding.refresh_visibility:
            binding.refresh_visibility()

    def set_data(self, data: dict[str, Any] | None) -> None:
        self._data = copy.deepcopy(data or {})
        for section in self._schema.get("sections", []):
            section_id = section["id"]
            if section_id not in self._data:
                self._data[section_id] = {}

        for binding in self._bindings:
            node = self._section_data(binding.section_id)
            value = _get_nested_value(node, binding.path, binding.field_spec["key"])
            if value is None and binding.field_spec.get("nullable") is False:
                if binding.field_spec.get("type") == "bool":
                    value = False
                elif binding.field_spec.get("type") == "yaml_mapping":
                    value = {}
                elif binding.field_spec.get("type") in ("multi_select_unique", "string_list"):
                    value = []
            binding.widget.blockSignals(True)
            try:
                self._set_widget_value(binding, value)
            finally:
                binding.widget.blockSignals(False)

    def get_data(self) -> dict[str, Any]:
        for binding in self._bindings:
            value = self._read_widget_value(binding)
            node = self._section_data(binding.section_id)
            _set_nested_value(node, binding.path, binding.field_spec["key"], value)
        self._normalize_preprocessor_nulls()
        return copy.deepcopy(self._data)

    def _normalize_preprocessor_nulls(self) -> None:
        preprocessors = self._data.get("model", {}).get("preprocessors", {})
        for slot in ("state", "value"):
            ref = preprocessors.get(slot)
            if not isinstance(ref, dict):
                continue
            if ref.get("type") in (None, "null", "none"):
                preprocessors[slot] = {"type": None}

    def _read_widget_value(self, binding: _FieldBinding) -> Any:
        widget = binding.widget
        field_type = binding.field_spec.get("type", "string")

        if field_type == "string":
            return widget.text()

        if field_type == "int":
            return widget.value()

        if field_type == "float":
            return widget.value()

        if field_type == "bool":
            return widget.isChecked()

        if field_type == "enum":
            text = widget.currentText()
            return text

        if field_type == "enum_nullable":
            data = widget.currentData()
            if data is None:
                return None
            return widget.currentText()

        if field_type == "multi_select_unique":
            selected = widget.selected()
            if binding.field_spec.get("unique", True):
                return list(dict.fromkeys(selected))
            return selected

        if field_type == "string_list":
            return self._parse_string_list(widget.toPlainText())

        if field_type == "yaml_mapping":
            text = widget.toPlainText().strip()
            if not text:
                return {}
            value = yaml.safe_load(text)
            if value is None:
                return {}
            if not isinstance(value, dict):
                raise ValueError(f"{binding.field_spec['key']} must be a YAML mapping")
            return value

        return widget.text()

    def validate(self) -> tuple[bool, str]:
        try:
            data = self.get_data()
            from training.schema import TrainingPresetConfig

            TrainingPresetConfig.model_validate(data)
            return True, ""
        except Exception as exc:
            return False, str(exc)
