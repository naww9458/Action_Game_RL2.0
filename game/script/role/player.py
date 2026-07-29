
from typing import Literal, Optional, Tuple, TYPE_CHECKING

from script.role.base_role import BaseRole, BaseRoleModel
from script.role.controller_utils import (
    CONTROLLER_CHOICES,
    PlayerController,
    normalize_controller,
)

if TYPE_CHECKING:
    from script.simulate.physics_manager import PhysicsManager

__all__ = [
    "Player",
    "PlayerModel",
    "PlayerController",
    "CONTROLLER_CHOICES",
    "normalize_controller",
]


class PlayerModel(BaseRoleModel):
    type: Literal["player"] = "player"
    name: str = "Player_1"
    controller: PlayerController = "Human"
    team_id: int = 1
    health: float = 5.0


class Player(BaseRole):
    role_key = "player"
    model_cls = PlayerModel
    path = "player_configs"
    container = "list"


    def __init__(self, configs, **kwargs):
        super().__init__(**kwargs)
        self._player_configs_cache = list(configs or [])

        self.setup(configs=configs)

    def update_index_rl_and_bot(self, index_rl_players_gpu, num_rl_players, is_rl_player_mask_gpu, index_bot_players_gpu, num_bot_players, is_bot_player_mask_gpu):
        self.index_rl_players_gpu = index_rl_players_gpu
        self.num_rl_players = num_rl_players
        self.is_rl_player_mask_gpu = is_rl_player_mask_gpu

        self.index_bot_players_gpu = index_bot_players_gpu
        self.num_bot_players = num_bot_players
        self.is_bot_player_mask_gpu = is_bot_player_mask_gpu

        for ability in self.abilities_instance_list:
            ability.update_index_bot(index_rl_players_gpu=index_rl_players_gpu, num_rl_players=num_rl_players, index_bot_players_gpu=index_bot_players_gpu, num_bot_players=num_bot_players)

    def resolve_follow_targets(
        self,
        role_index: int,
        physics_manager: "PhysicsManager",
    ) -> Optional[Tuple[int, int]]:
        """
        Map a player role index to (role_object_id, camera_body_index).

        Uses configured follow-body prim when available; otherwise the role's
        primary object index (first body of the articulation root).
        """
        if role_index < 0 or role_index >= len(self.index_obj_role):
            return None

        role_object_id = int(self.index_obj_role[role_index])
        # Fallback: historically some levels used role_object_id as body index.
        # We will override it when we can resolve a body prim suffix via metadata.
        camera_body_index = role_object_id

        # Cache follow mapping because it can be called every frame.
        if not hasattr(self, "_follow_target_cache") or self._follow_target_cache is None:
            self._follow_target_cache = {}
        cache_key = int(role_index)
        cached = self._follow_target_cache.get(cache_key)
        if cached is not None:
            return cached

        env_idx = 0
        env_map = getattr(self, "index_obj_role_to_env_mapping", None) or []
        if role_index < len(env_map):
            try:
                env_idx = int(env_map[role_index])
            except Exception:
                env_idx = 0

        # Map local body indices (from path_body_map) -> global body_q indices.
        if not hasattr(self, "_follow_env_body_indices_cache"):
            self._follow_env_body_indices_cache = None
        if self._follow_env_body_indices_cache is None and getattr(
            physics_manager, "model", None
        ) is not None and getattr(physics_manager.model, "body_world", None) is not None:
            try:
                body_world_np = physics_manager.model.body_world.numpy()
                max_world = int(body_world_np.max()) if body_world_np.size > 0 else 0
                env_to_body_indices = []
                for w in range(max_world + 1):
                    env_to_body_indices.append((body_world_np == w).nonzero()[0])
                self._follow_env_body_indices_cache = env_to_body_indices
            except Exception:
                self._follow_env_body_indices_cache = None

        params = BaseRole._object_game_params[role_object_id]
        pattern = str(params.get("pattern", "default"))
        task_name = None
        if role_index < len(self._player_configs_cache):
            task_name = (self._player_configs_cache[role_index].get("object") or {}).get(
                "control_task"
            )

        from script.role.abilities.articulation_control_config.joint_config_registry import (
            resolve_follow_body_prim_suffix,
        )
        from script.role.objects.tool_anchor import find_body_prim_path

        suffix = resolve_follow_body_prim_suffix(pattern, task_name)
        meta = (physics_manager.object_metadata_by_role or {}).get(role_object_id, {})
        path_body_map = meta.get("path_body_map") or {}
        chosen_local_body_idx: Optional[int] = None
        if path_body_map:
            # Collect candidate body local indices.
            # Usually path_body_map indices correspond to rigid-body local indices inside the template.
            try:
                local_candidates = [int(v) for v in path_body_map.values()]
            except Exception:
                local_candidates = []
            if local_candidates:
                chosen_local_body_idx = min(local_candidates)

        if suffix and path_body_map:
            try:
                body_path = find_body_prim_path(path_body_map, str(suffix))
                local_body_idx = int(path_body_map[body_path])
                chosen_local_body_idx = local_body_idx
            except KeyError:
                pass

        if chosen_local_body_idx is not None:
            # Prefer mapping local->global using body_world lists.
            env_list = None
            if (
                self._follow_env_body_indices_cache is not None
                and 0 <= env_idx < len(self._follow_env_body_indices_cache)
            ):
                env_list = self._follow_env_body_indices_cache[env_idx]

            if env_list is not None and 0 <= chosen_local_body_idx < len(env_list):
                camera_body_index = int(env_list[chosen_local_body_idx])
            else:
                # If we cannot map, assume local body index is already usable.
                camera_body_index = int(chosen_local_body_idx)

        result = (role_object_id, camera_body_index)
        self._follow_target_cache[cache_key] = result
        return result

    def rl_action(self, actions, **kwargs):
        if self.num_rl_players > 0:
            for ability in self.abilities_instance_list:
                # if ability.action_shape_offset is None:
                #     raise ValueError(f"{ability.ability_name}.action_shape_offset cannot be None")
                
                ability.rl_action(actions=actions, **kwargs)

    def bot_action(self, **kwargs):
        if self.num_bot_players > 0:
            for ability in self.abilities_instance_list:
                ability.bot_action(**kwargs)
