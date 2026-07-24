from pathlib import Path
import os
import numpy as np
import pickle
import json
import yaml

from script.game_config import GameConfig



class Trainer_base:
    def __init__(self):
        self.agent = None
        self.loaded_config = None
        self.preset_path = None


    def _init_battle_tracking(self) -> None:
        detector = getattr(self.env.game.reward_calculator, "episode_end_detector", None)
        winner = getattr(detector, "winner", None) if detector is not None else None
        if winner is None:
            self.winner_array_prev = np.zeros(0, dtype=np.int32)
            self.battle_results = {}
            return

        self.winner_array_prev = np.zeros(int(winner.shape[0]), dtype=np.int32)
        self.battle_results = {}
        for name in self.env.game.name_list:
            if name and name not in self.battle_results:
                self.battle_results[name] = 0

    def update_winner(self):
        detector = getattr(self.env.game.reward_calculator, "episode_end_detector", None)
        if detector is None:
            return

        winner_array_new = detector.winner.numpy()
        if self.winner_array_prev.shape != winner_array_new.shape:
            self.winner_array_prev = np.zeros_like(winner_array_new)

        new_winner = winner_array_new - self.winner_array_prev
        winners = np.nonzero(new_winner)[0]
        name_list = self.env.game.name_list

        for winner in winners.tolist():
            if winner < 0 or winner >= len(name_list):
                continue
            winner_name = name_list[winner]
            if not winner_name:
                continue
            print("winner_name: ", winner_name)
            self.battle_results.setdefault(winner_name, 0)
            self.battle_results[winner_name] += 1

        print("update_winner - battle_results: ", self.battle_results)
        self.winner_array_prev = winner_array_new.copy()

    def save_config_pickle(self, model_cfg, train_cfg, level_cfg, path):
        path = str(path)
        with open(path + "/model_cfg.pkl", 'wb') as f:
            pickle.dump(model_cfg, f)

        with open(path + "/train_cfg.pkl", 'wb') as f:
            pickle.dump(train_cfg, f)
            
        with open(path + "/level_cfg.yaml", "w", encoding="utf-8") as f:
            yaml.dump(level_cfg, f, default_flow_style=False, sort_keys=False)

    def save_run_config(self, config_path, level_cfg):
        """Save full run artifacts: preset.yaml, run_manifest.json, and legacy pickles."""
        from training.runs_manager import RunsManager

        os.makedirs(config_path, exist_ok=True)
        meta = self.loaded_config.meta if self.loaded_config else None

        preset_data = self.loaded_config.preset.model_dump(by_alias=True) if self.loaded_config else None
        manifest = None
        if meta is not None:
            from training.runs_manager import RunsManager as RM
            manifest = RM.build_run_manifest(
                preset_id=meta.id,
                policy_module=meta.policy_module,
                trainer_module=meta.trainer_module,
                algorithm=meta.algorithm,
                num_envs=self.env.num_envs,
                level=meta.level,
                sub_level=meta.sub_level,
                resume_from=getattr(self, "_resume_from", None),
            )

        if preset_data and manifest:
            RunsManager.save_run_artifacts(
                Path(config_path),
                preset_data=preset_data,
                manifest=manifest,
                model_cfg=self.model_cfg,
                train_cfg=self.train_cfg,
                level_cfg=level_cfg,
            )
        else:
            self.save_config_pickle(
                model_cfg=self.model_cfg,
                train_cfg=self.train_cfg,
                level_cfg=level_cfg,
                path=config_path,
            )


    def load_config_from_checkpoint(self, path):
        file_path = Path(path).resolve()
        config_path = file_path.parent.parent / "config"

        preset_yaml = config_path / "preset.yaml"
        if preset_yaml.exists():
            from training.loader import TrainingPresetLoader
            loaded = TrainingPresetLoader.load_from_dict(
                yaml.safe_load(open(preset_yaml, "r", encoding="utf-8")),
                preset_path=preset_yaml,
            )
            level_cfg_path = config_path / "level_cfg.yaml"
            return loaded.model_cfg, loaded.train_cfg, str(level_cfg_path), loaded

        manifest_path = config_path / "run_manifest.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        from training.runs_manager import RunsManager

        model_obs_type = "state_based"
        model_cfg_path = config_path / "model_cfg.pkl"
        if model_cfg_path.exists():
            with open(model_cfg_path, "rb") as f:
                probe_model_cfg = pickle.load(f)
            model_obs_type = getattr(probe_model_cfg, "model_obs_type", model_obs_type)

        manifest = RunsManager.infer_run_metadata(
            config_path.parent,
            manifest,
            model_obs_type=model_obs_type,
        )

        train_cfg_path = config_path / "train_cfg.pkl"
        level_cfg_path = config_path / "level_cfg.yaml"

        if not model_cfg_path.exists():
            raise FileNotFoundError(f"File not found: {model_cfg_path}")
        if not train_cfg_path.exists():
            raise FileNotFoundError(f"File not found: {train_cfg_path}")

        with open(model_cfg_path, "rb") as f:
            model_cfg = pickle.load(f)
        with open(train_cfg_path, "rb") as f:
            train_cfg = pickle.load(f)

        from training.loader import TrainingPresetLoader
        loaded = TrainingPresetLoader.from_legacy_pickle(model_cfg, train_cfg, manifest)
        return model_cfg, train_cfg, str(level_cfg_path), loaded

    def load_config_pickle(self, path):
        model_cfg, train_cfg, level_cfg_path, _ = self.load_config_from_checkpoint(path)
        return model_cfg, train_cfg, level_cfg_path



