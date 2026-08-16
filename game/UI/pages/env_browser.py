"""Environment catalog tree: nested folders, context menus, hover intro."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, QPoint, Qt
from PyQt6.QtGui import QBrush, QColor, QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PyQt6 import sip

from script.environments.env_catalog import (
    ROOT_CUSTOM,
    ROOT_TEMPLATE,
    CatalogNode,
    as_catalog_leaf,
    copy_catalog_node,
    create_env_yaml,
    create_folder_node,
    delete_catalog_node,
    destination_for,
    ensure_environment_roots,
    scan_catalog,
)


NODE_ROLE = Qt.ItemDataRole.UserRole


def _qt_alive(obj) -> bool:
    return obj is not None and not sip.isdeleted(obj)


class NameIntroDialog(QDialog):
    def __init__(self, title: str, tr_get, parent=None):
        super().__init__(parent)
        self.TR = tr_get
        self.setWindowTitle(title)
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.intro_edit = QPlainTextEdit()
        self.intro_edit.setPlaceholderText(self.TR("env_intro_optional"))
        self.intro_edit.setFixedHeight(90)
        form.addRow(self.TR("env_name_label"), self.name_edit)
        form.addRow(self.TR("env_intro_label"), self.intro_edit)
        layout.addLayout(form)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_btn.setEnabled(False)
        self.ok_btn.setText(self.TR("confirm"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.TR("cancel"))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.name_edit.textChanged.connect(self._sync_ok)

    def _sync_ok(self, text: str) -> None:
        self.ok_btn.setEnabled(bool(text.strip()))

    def values(self) -> tuple[str, str]:
        return self.name_edit.text().strip(), self.intro_edit.toPlainText().strip()


class FollowCursorTip(QLabel):
    """Tooltip that follows the cursor and never intercepts mouse clicks."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
        self.setWordWrap(True)
        self.setMaximumWidth(360)
        self.setStyleSheet(
            "QLabel { background: #222; color: #f5f5f5; border: 1px solid #666; "
            "padding: 8px; }"
        )
        self.hide()

    def show_at_cursor(self, title: str, intro: str) -> None:
        body = title if not intro else f"{title}\n\n{intro}"
        self.setText(body)
        self.adjustSize()
        pos = QCursor.pos() + QPoint(16, 16)
        self.move(pos)
        self.show()


class EnvTreeController(QObject):
    def __init__(
        self,
        tree: QTreeWidget,
        intro_label: QLabel,
        tr_get,
        host: QWidget,
        *,
        parent: QWidget | None = None,
        root_filter: str | None = None,
        flatten_root: bool = False,
        on_catalog_changed=None,
    ):
        super().__init__(parent if parent is not None else host)
        self.tree = tree
        self.intro_label = intro_label
        self.TR = tr_get
        self.host = host
        self.root_filter = root_filter
        self.flatten_root = flatten_root
        self.on_catalog_changed = on_catalog_changed
        self.hover_enabled = False
        self._tip = FollowCursorTip()
        self._hover_item = None
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.viewport().installEventFilter(self)
        self.tree.installEventFilter(self)
        self.tree.destroyed.connect(self._detach_event_filters)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.destroyed.connect(self._detach_event_filters)

    def _detach_event_filters(self, *_args) -> None:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._hover_item = None
        if _qt_alive(self._tip):
            self._tip.hide()
            self._tip.deleteLater()

    def reload(self) -> None:
        if not _qt_alive(self.tree):
            return
        self._hover_item = None
        self._hide_tip()
        ensure_environment_roots()
        self.tree.clear()
        nodes = scan_catalog()
        if self.root_filter:
            nodes = [node for node in nodes if node.root == self.root_filter]
            if self.flatten_root:
                flattened: list[CatalogNode] = []
                for node in nodes:
                    flattened.extend(node.children)
                nodes = flattened
        for node in nodes:
            self._add_node(self.tree.invisibleRootItem(), node, expanded=True)
        self._on_selection_changed()

    def selected_payload(self) -> dict | None:
        if not _qt_alive(self.tree):
            return None
        items = self.tree.selectedItems()
        if not items:
            return None
        data = items[0].data(0, NODE_ROLE)
        return data if isinstance(data, dict) else None

    def eventFilter(self, watched, event) -> bool:
        if not _qt_alive(self.tree):
            return False
        etype = event.type()
        hovering = self._hover_item is not None or self.tree.underMouse()
        if hovering and etype in (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride) and event.key() == Qt.Key.Key_Shift:
            if etype == QEvent.Type.KeyPress and not event.isAutoRepeat():
                self.hover_enabled = not self.hover_enabled
                if not self.hover_enabled:
                    self._hide_tip()
                elif self._hover_item is not None:
                    self._show_tip_for_item(self._hover_item)
            return False
        viewport = self.tree.viewport()
        if watched is viewport and _qt_alive(viewport):
            if event.type() == QEvent.Type.Leave:
                self._hover_item = None
                self._hide_tip()
            elif event.type() == QEvent.Type.MouseMove:
                item = self.tree.itemAt(event.position().toPoint())
                self._hover_item = item
                if self.hover_enabled and item is not None:
                    self._show_tip_for_item(item)
                else:
                    self._hide_tip()
        return super().eventFilter(watched, event)

    def _add_node(self, parent_item, node: CatalogNode, *, expanded: bool = False) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent_item, [node.name])
        item.setData(
            0,
            NODE_ROLE,
            {
                "node_type": node.node_type,
                "path": str(node.path),
                "name": node.name,
                "intro": node.intro,
                "root": node.root,
                "bucket": node.bucket,
                "env_id": node.env_id,
            },
        )
        item.setExpanded(expanded or node.node_type in ("root", "bucket"))
        color = self._color_for_node(node.node_type)
        if color is not None:
            item.setForeground(0, QBrush(color))
        for child in node.children:
            self._add_node(item, child)
        return item

    def _color_for_node(self, node_type: str) -> QColor | None:
        getter = getattr(self.host, "env_tree_colors", None)
        if not callable(getter):
            return None
        colors = getter() or {}
        if node_type == "env":
            key = "env"
        elif node_type == "category":
            key = "category"
        else:
            key = "series"
        raw = colors.get(key)
        if not raw:
            return None
        color = QColor(str(raw))
        return color if color.isValid() else None

    def _on_selection_changed(self) -> None:
        if not _qt_alive(self.intro_label):
            return
        payload = self.selected_payload()
        if payload is None:
            self.intro_label.setText("")
            return
        title = payload.get("name") or ""
        intro = payload.get("intro") or ""
        self.intro_label.setText(title if not intro else f"{title}\n{intro}")

    def _show_tip_for_item(self, item: QTreeWidgetItem) -> None:
        if not _qt_alive(self._tip):
            return
        payload = item.data(0, NODE_ROLE)
        if not isinstance(payload, dict):
            self._hide_tip()
            return
        self._tip.show_at_cursor(payload.get("name") or "", payload.get("intro") or "")

    def _hide_tip(self) -> None:
        if _qt_alive(self._tip):
            self._tip.hide()

    def _on_context_menu(self, pos) -> None:
        if not _qt_alive(self.tree):
            return
        item = self.tree.itemAt(pos)
        if item is None:
            return
        payload = item.data(0, NODE_ROLE)
        if not isinstance(payload, dict):
            return
        node_type = payload.get("node_type")
        root = payload.get("root")
        path = Path(payload["path"])
        menu = QMenu(self.tree)

        if root == ROOT_CUSTOM and node_type == "bucket":
            menu.addAction(self.TR("new_series"), lambda: self._create_folder(path, "new_series"))
        if root == ROOT_CUSTOM and node_type in ("series", "category"):
            menu.addAction(self.TR("new_category"), lambda: self._create_folder(path, "new_category"))
            menu.addAction(self.TR("new_environment"), lambda: self._create_env(path))
        if node_type in ("series", "category", "env"):
            if root == ROOT_TEMPLATE:
                menu.addAction(self.TR("copy_to_custom"), lambda: self._copy_node(path))
            elif root == ROOT_CUSTOM:
                menu.addAction(self.TR("upload_to_template"), lambda: self._copy_node(path))
            menu.addSeparator()
            menu.addAction(self.TR("delete_env_node"), lambda: self._delete_node(path, payload.get("name") or path.name))

        if menu.actions():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _create_folder(self, parent: Path, title_key: str) -> None:
        dialog = NameIntroDialog(self.TR(title_key), self.TR, self.host)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, intro = dialog.values()
        if not name:
            return
        create_folder_node(parent, name, intro)
        self.reload()

    def _create_env(self, parent: Path) -> None:
        dialog = NameIntroDialog(self.TR("new_environment"), self.TR, self.host)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, intro = dialog.values()
        if not name:
            return
        path = create_env_yaml(parent, name, intro)
        self.reload()
        opener = getattr(self.host, "open_env_editor", None)
        if callable(opener):
            opener(str(path))

    def _copy_node(self, src: Path) -> None:
        src_leaf = as_catalog_leaf(src)
        dest = destination_for(src_leaf)
        overwrite = False
        if dest is not None and dest.exists():
            reply = QMessageBox.question(
                self.host,
                self.TR("overwrite_title"),
                self.TR("overwrite_dest_confirm"),
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Ok:
                return
            overwrite = True
        try:
            copy_catalog_node(src_leaf, overwrite=overwrite)
        except Exception as exc:
            QMessageBox.critical(self.host, self.TR("copy_failed"), str(exc))
            return
        self.reload()
        callback = self.on_catalog_changed
        if callable(callback):
            callback()

    def _delete_node(self, path: Path, name: str) -> None:
        reply = QMessageBox.question(
            self.host,
            self.TR("delete_env_title"),
            self.TR("delete_env_confirm").format(name=name),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            return
        try:
            delete_catalog_node(path)
        except Exception as exc:
            QMessageBox.critical(self.host, self.TR("delete_failed"), str(exc))
            return
        self.reload()
        callback = self.on_catalog_changed
        if callable(callback):
            callback()


class TemplateCatalogDialog(QDialog):
    """Browse official template environments: delete unused ones or copy to custom."""

    def __init__(self, tr_get, host: QWidget, parent=None):
        super().__init__(parent)
        self.TR = tr_get
        self.setWindowTitle(self.TR("env_templates"))
        self.resize(520, 640)
        layout = QVBoxLayout(self)
        hint = QLabel(self.TR("env_templates_hint"))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        layout.addWidget(tree, 1)
        intro = QLabel()
        intro.setWordWrap(True)
        intro.setAlignment(Qt.AlignmentFlag.AlignTop)
        intro.setMinimumHeight(72)
        layout.addWidget(intro)
        self.browser = EnvTreeController(
            tree,
            intro,
            tr_get,
            host,
            parent=self,
            root_filter=ROOT_TEMPLATE,
            flatten_root=True,
            on_catalog_changed=getattr(host, "reload_env_catalog", None),
        )
        self.browser.reload()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setText(self.TR("close"))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
