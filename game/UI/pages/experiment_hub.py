import os
import sys
import subprocess
import yaml

from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QLineEdit, QFormLayout, QMessageBox,
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices

from UI.widgets.preset_form_editor import PresetFormEditor


class ExperimentHubPage(QWidget):
    def __init__(self, main_app, tr_get):
        super().__init__()
        self.main_app = main_app
        self.TR = tr_get
        self.target_level = None
        self.target_sub_level = None
        self._tb_process = None
        self._tb_active_run = None
        self._tb_port = 6006
        self._train_process = None
        self._eval_process = None

        self.layout = QVBoxLayout(self)
        self.title = QLabel(self.TR("exp_hub_title"))
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        self.layout.addWidget(self.title)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self._build_preset_tab()
        self._build_runs_tab()
        self._build_launch_tab()

        btn_layout = QHBoxLayout()
        self.btn_back = QPushButton(self.TR("back_main"))
        self.btn_back.clicked.connect(lambda: self.main_app.switch_page(0))
        btn_layout.addWidget(self.btn_back)
        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)

        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

    def _game_dir(self) -> str:
        project = self.main_app.settings_page.get_project_path()
        game_dir = os.path.join(project, "game")
        if os.path.isdir(game_dir):
            return game_dir
        return project

    def _project_root(self) -> str:
        return self.main_app.settings_page.get_project_path()

    def _launcher_script(self) -> str:
        return os.path.join(self._game_dir(), "rl_launcher.py")

    def _spawn_launcher(self, *args: str) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, self._launcher_script(), *args],
            cwd=self._project_root(),
        )

    def _ensure_training_imports(self):
        game_dir = self._game_dir()
        if game_dir not in sys.path:
            sys.path.insert(0, game_dir)

    def set_target(self, level, sub_level):
        self.target_level = level
        self.target_sub_level = sub_level
        self.refresh_presets()
        if self.preset_combo.count() > 0:
            self.preset_combo.setCurrentIndex(0)

    def _build_preset_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        left = QVBoxLayout()
        self.preset_list = QListWidget()
        self.preset_list.currentItemChanged.connect(self._load_selected_preset)
        left.addWidget(QLabel(self.TR("preset_list")))
        left.addWidget(self.preset_list)

        preset_btns = QHBoxLayout()
        self.btn_reload_preset = QPushButton(self.TR("reload_preset"))
        self.btn_reload_preset.clicked.connect(self.refresh_presets)
        self.btn_save_preset = QPushButton(self.TR("save_preset"))
        self.btn_save_preset.clicked.connect(self.save_preset_yaml)
        preset_btns.addWidget(self.btn_reload_preset)
        preset_btns.addWidget(self.btn_save_preset)
        left.addLayout(preset_btns)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        self.lbl_preset_editor = QLabel(self.TR("preset_editor"))
        right.addWidget(self.lbl_preset_editor)
        lang = getattr(self.main_app, "global_config", {}).get("language", "zh")
        self.preset_editor = PresetFormEditor(lang=lang)
        right.addWidget(self.preset_editor)
        layout.addLayout(right, 2)

        self.tabs.addTab(tab, self.TR("tab_preset"))

    def _build_runs_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        left = QVBoxLayout()
        left.addWidget(QLabel(self.TR("run_list")))
        self.run_list = QListWidget()
        self.run_list.currentItemChanged.connect(self._on_run_selected)
        left.addWidget(self.run_list)
        self.btn_refresh_runs = QPushButton(self.TR("refresh"))
        self.btn_refresh_runs.clicked.connect(self.refresh_runs)
        left.addWidget(self.btn_refresh_runs)
        layout.addLayout(left, 2)

        right = QVBoxLayout()
        right.addWidget(QLabel(self.TR("checkpoint_select")))
        self.checkpoint_combo = QComboBox()
        right.addWidget(self.checkpoint_combo)

        self.spin_eval_episodes = QSpinBox()
        self.spin_eval_episodes.setRange(1, 10000)
        self.spin_eval_episodes.setValue(50)
        right.addWidget(QLabel(self.TR("eval_episodes")))
        right.addWidget(self.spin_eval_episodes)

        self.spin_eval_num_envs = QSpinBox()
        self.spin_eval_num_envs.setRange(1, 65536)
        self.spin_eval_num_envs.setValue(2)
        self.spin_eval_num_envs.valueChanged.connect(self._sync_eval_window_envs_range)
        right.addWidget(QLabel(self.TR("eval_num_envs")))
        right.addWidget(self.spin_eval_num_envs)

        self.check_eval_headless = QCheckBox()
        self.check_eval_headless.setChecked(False)
        self.check_eval_headless.stateChanged.connect(self._update_eval_window_envs_visibility)
        right.addWidget(QLabel(self.TR("eval_headless")))
        right.addWidget(self.check_eval_headless)

        self.lbl_eval_window_envs = QLabel(self.TR("window_envs"))
        self.spin_eval_window_envs = QSpinBox()
        self.spin_eval_window_envs.setRange(1, 65536)
        self.spin_eval_window_envs.setValue(1)
        right.addWidget(self.lbl_eval_window_envs)
        right.addWidget(self.spin_eval_window_envs)

        self.lbl_eval_algorithm = QLabel("")
        self.lbl_eval_algorithm.setStyleSheet("color: #546e7a; font-style: italic;")
        right.addWidget(self.lbl_eval_algorithm)

        self.btn_eval = QPushButton(self.TR("eval_model"))
        self.btn_eval.clicked.connect(self.launch_eval)
        self.btn_stop_eval = QPushButton(self.TR("stop_eval"))
        self.btn_stop_eval.setStyleSheet("background-color: #c62828; color: white;")
        self.btn_stop_eval.clicked.connect(self.stop_eval)
        self.btn_stop_eval.setEnabled(False)
        self.btn_tb = QPushButton(self.TR("launch_tb"))
        self.btn_tb.clicked.connect(self.launch_tensorboard)
        self.btn_stop_tb = QPushButton(self.TR("stop_tb"))
        self.btn_stop_tb.clicked.connect(self.stop_tensorboard)
        self.btn_stop_tb.setEnabled(False)
        self.btn_open_tb = QPushButton(self.TR("open_tb_browser"))
        self.btn_open_tb.clicked.connect(self.open_tensorboard_browser)
        self.btn_open_tb.setEnabled(False)
        self.btn_open_folder = QPushButton(self.TR("open_folder"))
        self.btn_open_folder.clicked.connect(self.open_run_folder)
        eval_row = QHBoxLayout()
        eval_row.addWidget(self.btn_eval)
        eval_row.addWidget(self.btn_stop_eval)
        right.addLayout(eval_row)
        tb_row = QHBoxLayout()
        tb_row.addWidget(self.btn_tb)
        tb_row.addWidget(self.btn_stop_tb)
        tb_row.addWidget(self.btn_open_tb)
        right.addLayout(tb_row)
        right.addWidget(self.btn_open_folder)
        right.addStretch()
        layout.addLayout(right, 1)

        self.tabs.addTab(tab, self.TR("tab_runs"))
        self._update_eval_window_envs_visibility()

    def _sync_eval_window_envs_range(self, _value: int | None = None):
        max_envs = self.spin_eval_num_envs.value()
        self.spin_eval_window_envs.setMaximum(max_envs)
        if self.spin_eval_window_envs.value() > max_envs:
            self.spin_eval_window_envs.setValue(max_envs)

    def _update_eval_window_envs_visibility(self, _state: int | None = None):
        visible = not self.check_eval_headless.isChecked()
        self.lbl_eval_window_envs.setVisible(visible)
        self.spin_eval_window_envs.setVisible(visible)
        if visible:
            self._sync_eval_window_envs_range()

    def _default_eval_num_envs(self, run) -> int:
        if run.algorithm.upper() == "APG":
            return 2
        if run.num_envs is not None:
            return min(run.num_envs, 40)
        return 40

    def _update_eval_run_hints(self, run):
        if run is None:
            self.lbl_eval_algorithm.setText("")
            self.btn_eval.setText(self.TR("eval_model"))
            return

        algo = (run.algorithm or "PPO").upper()
        if algo == "APG":
            self.btn_eval.setText(self.TR("eval_apg"))
            self.lbl_eval_algorithm.setText(self.TR("eval_apg_hint"))
        else:
            self.btn_eval.setText(self.TR("eval_model"))
            self.lbl_eval_algorithm.setText(self.TR("eval_ppo_hint"))

        self.spin_eval_num_envs.setValue(self._default_eval_num_envs(run))

    def _build_launch_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)

        self.preset_combo = QComboBox()
        form.addRow(self.TR("preset_list"), self.preset_combo)

        self.spin_train_envs = QSpinBox()
        self.spin_train_envs.setRange(1, 65536)
        self.spin_train_envs.setValue(4096)
        self.spin_train_envs.valueChanged.connect(self._sync_window_envs_range)
        form.addRow(self.TR("num_envs"), self.spin_train_envs)

        self.check_headless = QCheckBox()
        self.check_headless.setChecked(True)
        self.check_headless.stateChanged.connect(self._update_window_envs_visibility)
        form.addRow(self.TR("headless_train"), self.check_headless)

        self.lbl_window_envs = QLabel(self.TR("window_envs"))
        self.spin_window_envs = QSpinBox()
        self.spin_window_envs.setRange(1, 4096)
        self.spin_window_envs.setValue(1)
        form.addRow(self.lbl_window_envs, self.spin_window_envs)

        self.input_resume = QLineEdit()
        self.input_resume.setPlaceholderText("runs/.../checkpoints/agent_100.pt")
        form.addRow(self.TR("resume_checkpoint"), self.input_resume)

        self.btn_start_train = QPushButton(self.TR("start_train"))
        self.btn_start_train.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold; padding: 10px;")
        self.btn_start_train.clicked.connect(self.launch_train)
        self.btn_stop_train = QPushButton(self.TR("stop_train"))
        self.btn_stop_train.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 10px;")
        self.btn_stop_train.clicked.connect(self.stop_training)
        self.btn_stop_train.setEnabled(False)
        train_row = QHBoxLayout()
        train_row.addWidget(self.btn_start_train)
        train_row.addWidget(self.btn_stop_train)
        form.addRow(train_row)

        self.tabs.addTab(tab, self.TR("tab_launch"))
        self._update_window_envs_visibility()

    def _sync_window_envs_range(self, _value: int | None = None):
        max_envs = self.spin_train_envs.value()
        self.spin_window_envs.setMaximum(max_envs)
        if self.spin_window_envs.value() > max_envs:
            self.spin_window_envs.setValue(max_envs)

    def _update_window_envs_visibility(self, _state: int | None = None):
        visible = not self.check_headless.isChecked()
        self.lbl_window_envs.setVisible(visible)
        self.spin_window_envs.setVisible(visible)
        if visible:
            self._sync_window_envs_range()

    def refresh_presets(self):
        self._ensure_training_imports()
        from training.registry import TrainingPresetRegistry

        presets = TrainingPresetRegistry.list_presets()
        if self.target_level is not None:
            presets = [
                p for p in presets
                if p["level"] == self.target_level and p["sub_level"] == self.target_sub_level
            ]

        self.preset_list.clear()
        self.preset_combo.clear()
        for preset in presets:
            label = f"{preset['display_name']} ({preset['id']})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, preset)
            self.preset_list.addItem(item)
            self.preset_combo.addItem(label, preset["id"])

        if self.preset_list.count() > 0:
            self.preset_list.setCurrentRow(0)

    def _load_selected_preset(self, current, _previous):
        if current is None:
            return
        preset = current.data(Qt.ItemDataRole.UserRole)
        path = preset.get("path")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.preset_editor.set_data(data)

    def save_preset_yaml(self):
        item = self.preset_list.currentItem()
        if item is None:
            return
        preset_meta = item.data(Qt.ItemDataRole.UserRole)
        ok, error = self.preset_editor.validate()
        if not ok:
            QMessageBox.critical(self, self.TR("yaml_error"), error)
            return

        try:
            data = self.preset_editor.get_data()
        except Exception as e:
            QMessageBox.critical(self, self.TR("yaml_error"), str(e))
            return

        self._ensure_training_imports()
        from training.schema import TrainingPresetConfig

        try:
            preset = TrainingPresetConfig.model_validate(data)
            path = Path(preset_meta["path"])
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(
                    preset.model_dump(by_alias=True),
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            self.status_label.setText(self.TR("save_success"))
            self.refresh_presets()
        except Exception as e:
            QMessageBox.critical(self, self.TR("yaml_error"), str(e))

    def refresh_runs(self):
        self._ensure_training_imports()
        from training.runs_manager import RunsManager

        manager = RunsManager(project_root=Path(self._project_root()))
        if self.target_level is not None:
            runs = manager.list_runs(level=self.target_level, sub_level=self.target_sub_level)
        else:
            runs = manager.list_runs()

        self.run_list.clear()
        for run in runs:
            item = QListWidgetItem(run.display_label)
            item.setData(Qt.ItemDataRole.UserRole, run)
            self.run_list.addItem(item)
        if self.run_list.count() > 0:
            self.run_list.setCurrentRow(0)

    def _on_run_selected(self, current, _previous):
        self.checkpoint_combo.clear()
        if current is None:
            self._update_eval_run_hints(None)
            return
        run = current.data(Qt.ItemDataRole.UserRole)
        for ckpt in run.checkpoints:
            label = ckpt.name if ckpt.step is None else f"{ckpt.name} (step {ckpt.step})"
            self.checkpoint_combo.addItem(label, ckpt.name)
        if self.checkpoint_combo.count() > 0:
            self.checkpoint_combo.setCurrentIndex(0)
        self._update_eval_run_hints(run)

    def _selected_run(self):
        item = self.run_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def launch_eval(self):
        run = self._selected_run()
        if run is None:
            return
        if self._eval_process is not None and self._eval_process.poll() is None:
            QMessageBox.information(self, self.TR("eval_model"), self.TR("eval_already_running"))
            return

        ckpt = self.checkpoint_combo.currentData() or "best_agent"
        cmd_args = [
            "eval",
            "--run", str(run.path),
            "--checkpoint", ckpt,
            "--num-envs", str(self.spin_eval_num_envs.value()),
            "--episodes", str(self.spin_eval_episodes.value()),
        ]
        if not self.check_eval_headless.isChecked():
            cmd_args.extend([
                "--enable-window",
                "--window-envs", str(self.spin_eval_window_envs.value()),
            ])
        try:
            self._eval_process = self._spawn_launcher(*cmd_args)
            self._update_eval_ui_state()
            algo = (run.algorithm or "PPO").upper()
            self.status_label.setText(
                f"{self.TR('eval_started')} [{algo}]: {run.name} / {ckpt} (PID {self._eval_process.pid})"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _update_eval_ui_state(self):
        running = self._eval_process is not None and self._eval_process.poll() is None
        self.btn_stop_eval.setEnabled(running)
        self.btn_eval.setEnabled(not running)
        if not running:
            self._eval_process = None

    def _stop_eval(self, silent: bool = False) -> None:
        proc = self._eval_process
        if proc is None or proc.poll() is not None:
            self._eval_process = None
            self._update_eval_ui_state()
            return

        self._kill_subprocess(proc)
        self._eval_process = None
        self._update_eval_ui_state()
        if not silent:
            self.status_label.setText(self.TR("eval_stopped"))

    def stop_eval(self):
        self._stop_eval()

    def _update_tb_ui_state(self):
        running = self._tb_process is not None and self._tb_process.poll() is None
        self.btn_stop_tb.setEnabled(running)
        self.btn_open_tb.setEnabled(running)
        if running and self._tb_active_run:
            self.status_label.setText(
                f"{self.TR('tb_started')}: http://localhost:{self._tb_port} ({self._tb_active_run})"
            )
        elif not running:
            self._tb_process = None
            self._tb_active_run = None

    def _kill_subprocess(self, proc: subprocess.Popen | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        except Exception:
            pass

    def _stop_tensorboard(self, silent: bool = False) -> None:
        proc = self._tb_process
        if proc is None or proc.poll() is not None:
            self._tb_process = None
            self._tb_active_run = None
            self._update_tb_ui_state()
            return

        self._kill_subprocess(proc)
        self._tb_process = None
        self._tb_active_run = None
        self._update_tb_ui_state()
        if not silent:
            self.status_label.setText(self.TR("tb_stopped"))

    def stop_tensorboard(self):
        self._stop_tensorboard()

    def open_tensorboard_browser(self):
        if self._tb_process is None or self._tb_process.poll() is not None:
            return
        QDesktopServices.openUrl(QUrl(f"http://localhost:{self._tb_port}"))

    def launch_tensorboard(self):
        run = self._selected_run()
        if run is None:
            return
        try:
            if self._tb_process is not None and self._tb_process.poll() is None:
                if self._tb_active_run == run.name:
                    self.open_tensorboard_browser()
                    return
                self.status_label.setText(self.TR("tb_switching"))
                self._stop_tensorboard(silent=True)

            self._ensure_training_imports()
            from training.runtime_env import ensure_runtime_env, resolve_tensorboard_command

            ensure_runtime_env()
            cmd = resolve_tensorboard_command(str(run.path), self._tb_port)
            popen_kwargs = {
                "cwd": self._project_root(),
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.PIPE,
            }
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self._tb_process = subprocess.Popen(cmd, **popen_kwargs)
            self._tb_active_run = run.name
            self._update_tb_ui_state()
            self.open_tensorboard_browser()
        except Exception as e:
            self._tb_process = None
            self._tb_active_run = None
            self._update_tb_ui_state()
            QMessageBox.critical(self, "Error", str(e))

    def _update_train_ui_state(self):
        running = self._train_process is not None and self._train_process.poll() is None
        self.btn_stop_train.setEnabled(running)
        self.btn_start_train.setEnabled(not running)
        if not running:
            self._train_process = None

    def _stop_training(self, silent: bool = False) -> None:
        proc = self._train_process
        if proc is None or proc.poll() is not None:
            self._train_process = None
            self._update_train_ui_state()
            return

        self._kill_subprocess(proc)
        self._train_process = None
        self._update_train_ui_state()
        if not silent:
            self.status_label.setText(self.TR("train_stopped"))
            self.refresh_runs()

    def stop_training(self):
        self._stop_training()

    def hideEvent(self, event):
        self._stop_tensorboard(silent=True)
        self._stop_training(silent=True)
        self._stop_eval(silent=True)
        super().hideEvent(event)

    def open_run_folder(self):
        run = self._selected_run()
        if run is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(run.path)))

    def launch_train(self):
        preset_id = self.preset_combo.currentData()
        if not preset_id:
            return
        cmd_args = [
            "train",
            "--preset", preset_id,
            "--num-envs", str(self.spin_train_envs.value()),
            "--mode", "custom",
        ]
        resume = self.input_resume.text().strip()
        if resume:
            cmd_args.extend(["--resume", resume])
        if not self.check_headless.isChecked():
            cmd_args.extend([
                "--enable-window",
                "--window-envs", str(self.spin_window_envs.value()),
            ])
        try:
            self._train_process = self._spawn_launcher(*cmd_args)
            self._update_train_ui_state()
            self.status_label.setText(f"{self.TR('train_started')} (PID {self._train_process.pid})")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def retranslate_ui(self):
        self.title.setText(self.TR("exp_hub_title"))
        self.btn_back.setText(self.TR("back_main"))
        self.tabs.setTabText(0, self.TR("tab_preset"))
        self.tabs.setTabText(1, self.TR("tab_runs"))
        self.tabs.setTabText(2, self.TR("tab_launch"))
        self.btn_reload_preset.setText(self.TR("reload_preset"))
        self.btn_save_preset.setText(self.TR("save_preset"))
        self.lbl_preset_editor.setText(self.TR("preset_editor"))
        self.btn_refresh_runs.setText(self.TR("refresh"))
        self.btn_eval.setText(self.TR("eval_model"))
        self.btn_stop_eval.setText(self.TR("stop_eval"))
        self.lbl_eval_window_envs.setText(self.TR("window_envs"))
        run = self._selected_run()
        self._update_eval_run_hints(run)
        self.btn_tb.setText(self.TR("launch_tb"))
        self.btn_stop_tb.setText(self.TR("stop_tb"))
        self.btn_open_tb.setText(self.TR("open_tb_browser"))
        self.btn_open_folder.setText(self.TR("open_folder"))
        self.btn_start_train.setText(self.TR("start_train"))
        self.btn_stop_train.setText(self.TR("stop_train"))
        self.lbl_window_envs.setText(self.TR("window_envs"))
        lang = getattr(self.main_app, "global_config", {}).get("language", "zh")
        self.preset_editor.set_language(lang)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_presets()
        self.refresh_runs()
        self._update_tb_ui_state()
        self._update_train_ui_state()
        self._update_eval_ui_state()
