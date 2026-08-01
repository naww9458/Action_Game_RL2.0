import warp as wp

from dataclasses import dataclass
from typing import Optional, Tuple

from .ability import Ability
from script.game_config import GameConfig
from script.role.abilities.articulation_control_config.profile_registry import (
    resolve_ability_generated_object_pattern,
)
from script.role.objects.object_template.loader import (
    get_object_template,
    load_template_control_config,
)
from utils.warp_math import sigmoid, quat_local_x

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.levels.levels import Levels


@wp.func
def _find_expired_bullet_slot(
    owner_idx: int,
    owner_mapping_gpu: wp.array(dtype=wp.int32, ndim=2),
    expired_steps: wp.array(dtype=wp.int32),
    num_slots: int,
) -> int:
    """Return the first expired (free) generated-object slot for ``owner_idx``."""
    for i in range(num_slots):
        local_idx = owner_mapping_gpu[owner_idx, i]
        if local_idx != -1 and expired_steps[local_idx] <= 0:
            return local_idx
    return -1


@wp.func
def _resolve_view_obj_idx(
    player_idx: int,
    num_objects_env: int,
    view_object_indices: wp.array(dtype=int),
    count_per_world: int,
) -> int:
    """Map a role object index to its view slot within a world, or -1."""
    local_idx = player_idx % num_objects_env
    for i in range(count_per_world):
        if view_object_indices[i] == local_idx:
            return i
    return -1


@wp.func
def _resolve_bullet_view_obj_idx(
    bullet_local_idx: int,
    bullet_view_object_indices: wp.array(dtype=int),
    bullet_count_per_world: int,
) -> int:
    """Map a generated-object local index to its view slot, or -1."""
    for i in range(bullet_count_per_world):
        if bullet_view_object_indices[i] == bullet_local_idx:
            return i
    return -1


@dataclass
class ShootFireConfig:
    """Per-owner firing parameters for the Shoot ability.

    Sources (highest priority first):
      1. The role config's ``abilities.Shoot`` dict (environment level YAML).
      2. The tool template's ``control_configs.yaml`` ``shoot`` section
         (for tool owners, e.g. turret_110mm).
      3. The global ``abilities_default_cfg.yaml`` ``Shoot`` section.
    """

    spawn_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    forward_force: float = 0.0
    recoil_force: float = 0.0
    cooldown_steps: int = 0  # 0 -> fall back to the ability-global cooldown
    speed: float = 0.0  # 0 -> fall back to the ability-global speed


class Shoot(Ability):
    """
    Bot 的射擊頻率是隨機化的，具體冷卻時間為 cooldown +- 50% (範圍内隨機)
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.seeds = None
        self.offset = None
        self.fire_intent_buffer = None
        self._generated_object_view_ctx = None
        # Lazy 1-element flag written by _fire_gpu only when a shot actually
        # spawns; read back on the host so callers (e.g. turret recoil) can tell
        # a real muzzle fire from an early-out (cooldown / no free slot).
        self._fire_result_gpu = None
        # Per-owner firing configs (muzzle offset / forces / cooldown / speed),
        # keyed by the owner role-object index (player or tool).
        self._owner_fire_configs: dict[int, ShootFireConfig] = {}
        # Global fallback built from abilities_default_cfg.yaml's Shoot section.
        self._default_fire_config = self._build_default_fire_config()

    def _build_default_fire_config(self) -> ShootFireConfig:
        """Build the global fallback firing config from abilities_default_cfg.yaml."""
        global_cfg = None
        if Ability._default_configs is not None:
            global_cfg = Ability._default_configs.root.get(self.ability_name)
        if global_cfg is None:
            return ShootFireConfig()
        raw = getattr(global_cfg, "model_dump", lambda: {})()
        if not isinstance(raw, dict):
            return ShootFireConfig()
        return self._fire_config_from_dict(raw) or ShootFireConfig()

    def _fire_config_from_dict(self, cfg: dict) -> Optional[ShootFireConfig]:
        """Build a ShootFireConfig from an ability-config dict (ability overrides).

        Accepts the field names used in both the ability config
        (``abilities_default_cfg.yaml``) and the tool control configs
        (``control_configs.yaml``), e.g. ``forward_force_n``,
        ``recoil_force_n``, ``projectile_generation_point_offset``,
        ``cooldown_s`` and ``speed``.
        """
        if not isinstance(cfg, dict) or not cfg:
            return None
        offset = cfg.get("projectile_generation_point_offset")
        cooldown_s = cfg.get("cooldown_s")
        speed = cfg.get("speed")
        fwd = cfg.get("forward_force_n")
        rec = cfg.get("recoil_force_n")
        if (
            offset is None
            and cooldown_s is None
            and speed is None
            and fwd is None
            and rec is None
        ):
            return None
        if isinstance(offset, (list, tuple)) and len(offset) >= 3:
            spawn_offset = (float(offset[0]), float(offset[1]), float(offset[2]))
        else:
            spawn_offset = (0.0, 0.0, 0.0)
        return ShootFireConfig(
            spawn_offset=spawn_offset,
            forward_force=float(fwd) if fwd is not None else 0.0,
            recoil_force=float(rec) if rec is not None else 0.0,
            cooldown_steps=(
                int(float(cooldown_s) * GameConfig.FPS_ACTION)
                if cooldown_s is not None
                else 0
            ),
            speed=float(speed) if speed is not None else 0.0,
        )

    def _merge_fire_config(
        self, base: ShootFireConfig, overrides: dict
    ) -> ShootFireConfig:
        """Overlay a role's ability-config dict on top of a base firing config."""
        merged = self._fire_config_from_dict(overrides)
        if merged is None:
            return base
        return ShootFireConfig(
            spawn_offset=(
                merged.spawn_offset
                if merged.spawn_offset != (0.0, 0.0, 0.0)
                else base.spawn_offset
            ),
            forward_force=merged.forward_force or base.forward_force,
            recoil_force=merged.recoil_force or base.recoil_force,
            cooldown_steps=merged.cooldown_steps or base.cooldown_steps,
            speed=merged.speed or base.speed,
        )

    def register_role_ability_config(
        self, role_object_index: int, ability_cfg: dict
    ) -> None:
        """Register per-owner firing config from a role's ``abilities.Shoot`` dict."""
        super().register_role_ability_config(role_object_index, ability_cfg)
        fire_cfg = self._fire_config_from_dict(ability_cfg)
        if fire_cfg is not None:
            existing = self._owner_fire_configs.get(int(role_object_index))
            self._owner_fire_configs[int(role_object_index)] = (
                self._merge_fire_config(existing, ability_cfg)
                if existing is not None
                else fire_cfg
            )

    def configure_from_generated_object_config(self, object_key: str, config: dict) -> None:
        object_config = dict(config.get("object") or {})
        self.ability_generated_object_name = object_key
        self.generated_object_pattern = resolve_ability_generated_object_pattern(object_config)
        # Who owns this generated object pool: "player" (default) or "tool".
        # Used by AbilityGeneratedObject.update_owner to build the owner mapping
        # and by Levels to emit collision filter pairs.
        self.owner_role_type = str(config.get("owner_role_type") or "player")
        self.collision_filter_owner_bodies = list(
            config.get("collision_filter_owner_bodies") or []
        )

    def configure_from_tool_configs(self, tool_configs, level: "Levels") -> None:
        """Configure Shoot for tool owners (e.g. turret_110mm).

        Reads per-pattern shoot parameters (muzzle offset, forward/recoil force,
        cooldown) from the tool template's control_configs.yaml, then overlays
        the tool role's ``abilities.Shoot`` dict (level config). Per-pattern
        side-effects (e.g. recoil) are applied by the tool's own action module,
        not by the ability.
        """
        if not tool_configs:
            return
        tools = getattr(level, "tools", None)
        if tools is None:
            return
        for tool_index, tool_cfg in enumerate(tool_configs):
            if tool_index >= len(tools.index_obj_role):
                continue
            tool_role_id = int(tools.index_obj_role[tool_index])
            fire_cfg = self._load_tool_fire_config(tool_cfg)
            role_override = self._role_ability_configs.get(tool_role_id)
            if role_override:
                fire_cfg = self._merge_fire_config(fire_cfg, role_override)
            self._owner_fire_configs[tool_role_id] = fire_cfg

        if self.generated_object_pattern:
            self._generated_object_view_ctx = self.resolve_pattern_view(
                self.generated_object_pattern
            )
        self._configured = True

    def _read_template_shoot_section(self, pattern: str) -> dict:
        """Read the tool's ``shoot`` section from its template control_configs.yaml.

        File discovery is delegated to ``load_template_control_config`` (config-
        driven via the template's ``control_config_path``); only the section
        layout is parsed here.
        """
        shoot_cfg: dict = {}
        if not pattern:
            return shoot_cfg
        template = get_object_template(pattern)
        if template is None:
            return shoot_cfg
        raw = load_template_control_config(pattern)
        template_id = str(template.get("id") or pattern)
        section = raw.get(template_id)
        if not isinstance(section, dict):
            section = raw
        sub_shoot = section.get("shoot") if isinstance(section, dict) else None
        if isinstance(sub_shoot, dict):
            shoot_cfg = sub_shoot
        return shoot_cfg

    def _load_tool_fire_config(self, tool_cfg: dict) -> ShootFireConfig:
        """Build a firing config from the tool's template ``shoot`` section."""
        pattern = str((tool_cfg.get("object") or {}).get("pattern") or "")
        return self._fire_config_from_dict(self._read_template_shoot_section(pattern)) or ShootFireConfig()

    def get_owner_fire_config(self, owner_obj_idx: int) -> Optional[ShootFireConfig]:
        """Return the firing config for an owner, or the global fallback."""
        cfg = self._owner_fire_configs.get(int(owner_obj_idx))
        if cfg is not None:
            return cfg
        return self._default_fire_config

    def configure_from_player_configs_post_indices(self, level: "Levels") -> None:
        super().configure_from_player_configs_post_indices(level)
        pattern = self.generated_object_pattern
        self._generated_object_view_ctx = (
            self.resolve_pattern_view(pattern) if pattern else None
        )

    def fire_from_aim_action(
        self,
        physics_manager,
        owner_obj_idx: int,
        spawn_pos_world,
        barrel_forward_dir,
    ) -> bool:
        """Launch a projectile from a world position/direction (tool aim path).

        Called by a tool's aim action (e.g. Turret110mmAimAction) when the
        human fires. Reads per-owner firing config (muzzle force / cooldown /
        speed). Returns True when a projectile was actually spawned; the caller
        may then apply its own per-pattern side-effects (e.g. recoil).
        """
        if not getattr(self, "_configured", False):
            return False

        config = self.get_owner_fire_config(owner_obj_idx)
        if config is None:
            config = self._default_fire_config

        bullet_ctx = self._generated_object_view_ctx
        if bullet_ctx is None or not bullet_ctx.valid:
            return False

        cooldown_steps = config.cooldown_steps if config.cooldown_steps > 0 else self.cooldown
        bullet_speed = config.speed if config.speed > 0.0 else float(self.speed)

        # Reset the host-readable "actually fired" flag before the launch; the
        # kernel sets it to 1 only when a projectile really spawns.
        if self._fire_result_gpu is None:
            self._fire_result_gpu = wp.zeros(
                1, dtype=wp.int32, device=physics_manager.device
            )
        self._fire_result_gpu.zero_()

        wp.launch(
            kernel=self._fire_gpu,
            dim=1,
            inputs=[
                self.articulation_body.control_mask_gpus[self.generated_object_pattern],
                self.articulation_body.control_pos_gpus[self.generated_object_pattern],
                self.articulation_body.control_rot_gpus[self.generated_object_pattern],
                self.articulation_body.control_vel_gpus[self.generated_object_pattern],
                self.articulation_body.control_force_gpus[self.generated_object_pattern],
                self.articulation_body.view_object_indices_gpus[self.generated_object_pattern],
                bullet_ctx.count_per_world,
                self.ability_generated_object.expired_steps,
                self.ability_generated_object.index_obj_role_gpu,
                self.ability_generated_object.default_expired_step_list_gpu,
                self.owner_mapping_gpu,
                self.cooldown_ability_owners,
                wp.int32(cooldown_steps),
                wp.int32(owner_obj_idx),
                wp.vec3(*spawn_pos_world),
                wp.vec3(*barrel_forward_dir),
                wp.float32(config.forward_force),
                wp.float32(bullet_speed),
                self.articulation_body.num_objects_env,
                self._fire_result_gpu,
            ],
            device=physics_manager.device,
        )
        # Read the flag back (numpy() syncs); True only when a shot actually fired.
        return int(self._fire_result_gpu.numpy()[0]) == 1

    @wp.kernel
    def _fire_gpu(
        bullet_control_mask: wp.array3d(dtype=wp.int32),
        bullet_control_pos: wp.array3d(dtype=wp.vec3),
        bullet_control_rot: wp.array3d(dtype=wp.quat),
        bullet_control_vel: wp.array3d(dtype=wp.vec3),
        bullet_control_force: wp.array3d(dtype=wp.vec3),

        bullet_view_object_indices: wp.array(dtype=int),
        bullet_count_per_world: int,

        expired_steps: wp.array(dtype=wp.int32),
        index_obj_role_gpu: wp.array(dtype=wp.int32),
        default_expired_step_list_gpu: wp.array(dtype=wp.int32),
        owner_mapping_gpu: wp.array(dtype=wp.int32, ndim=2),

        cooldown_ability_owners: wp.array(dtype=wp.int32),
        cooldown: wp.int32,

        owner_obj_idx: wp.int32,
        spawn_pos_world: wp.vec3,
        barrel_forward_dir: wp.vec3,
        forward_force: wp.float32,
        bullet_speed: wp.float32,

        num_objects_env: int,
        fire_result: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()

        # --- Cooldown check ---
        if cooldown_ability_owners[owner_obj_idx] != 0:
            return

        # --- Find an available (expired) bullet slot ---
        available_bullet_local_idx = _find_expired_bullet_slot(
            owner_obj_idx,
            owner_mapping_gpu,
            expired_steps,
            owner_mapping_gpu.shape[1],
        )
        if available_bullet_local_idx == -1:
            return

        # --- Bullet index resolution ---
        bullet_idx = index_obj_role_gpu[available_bullet_local_idx]
        b_world = bullet_idx // num_objects_env
        b_local_idx = bullet_idx % num_objects_env
        b_obj_idx = _resolve_bullet_view_obj_idx(
            b_local_idx, bullet_view_object_indices, bullet_count_per_world
        )
        if b_obj_idx == -1:
            return

        # --- Spawn the projectile at the muzzle ---
        cooldown_ability_owners[owner_obj_idx] = cooldown
        expired_steps[available_bullet_local_idx] = default_expired_step_list_gpu[
            available_bullet_local_idx
        ]

        bullet_control_pos[b_world, b_obj_idx, 0] = spawn_pos_world
        bullet_control_rot[b_world, b_obj_idx, 0] = wp.quat(0.0, 0.0, 0.0, 1.0)

        if bullet_speed > 0.0:
            bullet_control_vel[b_world, b_obj_idx, 0] = barrel_forward_dir * bullet_speed

        bullet_control_force[b_world, b_obj_idx, 0] = barrel_forward_dir * forward_force

        # Position + velocity control bits (1 | 4 = 5)
        bullet_control_mask[b_world, b_obj_idx, 0] = (
            bullet_control_mask[b_world, b_obj_idx, 0] | 5
        )

        # A projectile really spawned — let the host distinguish this from
        # cooldown/slot early-outs (e.g. to gate turret recoil).
        fire_result[0] = 1

    def _shoot_ctx(self, controller: str):
        player_ctx = self._view_ctx(controller)
        bullet_ctx = self._generated_object_view_ctx
        if player_ctx is None or bullet_ctx is None or not bullet_ctx.valid:
            return None
        return player_ctx, bullet_ctx

    def human_control_interface(self, keyboard_keys, mouse_buttons, index_human_player_gpu: wp.array, **kwargs):
        if not getattr(self, "_configured", False):
            return
        
        physics_manager = self.physics_manager

        if self._is_pressed(self._keyboard_action, self._mouse_action, keyboard_keys, mouse_buttons) == 1:
            ctxs = self._shoot_ctx("human")
            if ctxs is None:
                return
            player_ctx, bullet_ctx = ctxs
            pattern = player_ctx.pattern

            wp.launch(
                kernel=self.human_action_gpu,
                dim=1, 
                inputs=[
                    self.articulation_body.control_mask_gpus[self.generated_object_pattern],
                    self.articulation_body.control_pos_gpus[self.generated_object_pattern],
                    self.articulation_body.control_rot_gpus[self.generated_object_pattern],
                    self.articulation_body.control_vel_gpus[self.generated_object_pattern],
                    physics_manager.state_0.body_q,
                    index_human_player_gpu,
                    self.owner_mapping_gpu,
                    self.ability_generated_object.expired_steps,
                    self.ability_generated_object.index_obj_role_gpu,
                    self.ability_generated_object.default_expired_step_list_gpu,
                    self.cooldown_ability_owners,
                    self.speed,
                    self.cooldown,
                    
                    self.articulation_body.view_object_indices_gpus[pattern],
                    self.articulation_body.view_body_local_indices_gpus[pattern],
                    player_ctx.count_per_world,
                    player_ctx.bodies_per_object,
                    
                    self.articulation_body.view_object_indices_gpus[self.generated_object_pattern],
                    bullet_ctx.count_per_world,
                    
                    self.articulation_body.num_objects_env,
                    self.articulation_body.num_rigid_bodies_env,
                ],
                device=physics_manager.device
            )

    @wp.kernel
    def human_action_gpu(
        bullet_control_mask: wp.array3d(dtype=wp.int32),
        bullet_control_pos: wp.array3d(dtype=wp.vec3),
        bullet_control_rot: wp.array3d(dtype=wp.quat),
        bullet_control_vel: wp.array3d(dtype=wp.vec3),
        body_q: wp.array(dtype=wp.transform), 
        index_human_player_gpu: wp.array(dtype=wp.int32),
        owner_mapping_gpu: wp.array(dtype=wp.int32, ndim=2),
        expired_steps: wp.array(dtype=wp.int32),
        index_obj_role_gpu: wp.array(dtype=wp.int32),
        default_expired_step_list_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        speed: wp.float32,
        cooldown: wp.int32,

        player_view_object_indices: wp.array(dtype=int),
        player_view_body_local_indices: wp.array(dtype=int),
        player_count_per_world: int,
        player_bodies_per_object: int,

        bullet_view_object_indices: wp.array(dtype=int),
        bullet_count_per_world: int,

        num_objects_env: int,
        num_body_object_env: int,
    ):
        tid = wp.tid()
        index_player = index_human_player_gpu[tid]

        if cooldown_ability_owners[index_player] != 0:
            return

        available_bullet_local_idx = _find_expired_bullet_slot(
            index_player,
            owner_mapping_gpu,
            expired_steps,
            owner_mapping_gpu.shape[1],
        )
        if available_bullet_local_idx == -1:
            return

        # Player transformation lookup
        p_world = index_player // num_objects_env
        p_obj_idx = _resolve_view_obj_idx(
            index_player,
            num_objects_env,
            player_view_object_indices,
            player_count_per_world,
        )
        if p_obj_idx == -1:
            return

        p_local_tid = p_obj_idx * player_bodies_per_object
        p_local_body_idx = player_view_body_local_indices[p_local_tid]
        p_global_body_idx = p_world * num_body_object_env + p_local_body_idx

        # 取得 Bot 自己的資訊
        tf = body_q[p_global_body_idx]
        forward = quat_local_x(tf.q)

        # 計算子彈初始位置與速度
        bullet_pos = tf.p
        bullet_vel = forward * speed

        # Bullet mapping
        bullet_idx = index_obj_role_gpu[available_bullet_local_idx]
        b_world = bullet_idx // num_objects_env
        b_local_idx = bullet_idx % num_objects_env
        b_obj_idx = _resolve_bullet_view_obj_idx(
            b_local_idx, bullet_view_object_indices, bullet_count_per_world
        )
        if b_obj_idx == -1:
            return

        # 套用物理變更
        cooldown_ability_owners[index_player] = cooldown
        expired_steps[available_bullet_local_idx] = default_expired_step_list_gpu[
            available_bullet_local_idx
        ]

        bullet_control_pos[b_world, b_obj_idx, 0] = bullet_pos
        bullet_control_rot[b_world, b_obj_idx, 0] = tf.q
        bullet_control_vel[b_world, b_obj_idx, 0] = bullet_vel
        bullet_control_mask[b_world, b_obj_idx, 0] = (
            bullet_control_mask[b_world, b_obj_idx, 0] | 7
        )


    def rl_action(self, actions, **kwargs):
        if getattr(self, "num_rl_players", 0) <= 0 or not getattr(self, "_configured", False):
            return
            
        physics_manager = self.physics_manager

        ctxs = self._shoot_ctx("rl")
        if ctxs is None:
            return
        player_ctx, bullet_ctx = ctxs
        pattern = player_ctx.pattern

        wp.launch(
            kernel=self.rl_action_gpu,
            dim=self.num_rl_players, 
            inputs=[
                self.articulation_body.control_mask_gpus[self.generated_object_pattern],
                self.articulation_body.control_pos_gpus[self.generated_object_pattern],
                self.articulation_body.control_rot_gpus[self.generated_object_pattern],
                self.articulation_body.control_vel_gpus[self.generated_object_pattern],
                physics_manager.state_0.body_q,
                actions,
                self.action_shape_offset,
                self.fire_intent_buffer, 
                self.index_rl_players_gpu,
                self.owner_mapping_gpu,
                self.ability_generated_object.expired_steps,
                self.ability_generated_object.index_obj_role_gpu,
                self.ability_generated_object.default_expired_step_list_gpu,
                self.cooldown_ability_owners,
                self.speed,
                self.cooldown,
                
                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.view_body_local_indices_gpus[pattern],
                player_ctx.count_per_world,
                player_ctx.bodies_per_object,
                
                self.articulation_body.view_object_indices_gpus[self.generated_object_pattern],
                bullet_ctx.count_per_world,
                
                self.articulation_body.num_objects_env,
                self.articulation_body.num_rigid_bodies_env,
            ],
            device=physics_manager.device
        )

    @wp.kernel
    def rl_action_gpu(
        bullet_control_mask: wp.array3d(dtype=wp.int32),
        bullet_control_pos: wp.array3d(dtype=wp.vec3),
        bullet_control_rot: wp.array3d(dtype=wp.quat),
        bullet_control_vel: wp.array3d(dtype=wp.vec3),
        body_q: wp.array(dtype=wp.transform), 
        actions: wp.array2d(dtype=wp.float32), 
        action_shape_offset: wp.int32,
        fire_intent_buffer: wp.array(dtype=wp.float32), 
        index_bot_players_gpu: wp.array(dtype=wp.int32),
        owner_mapping_gpu: wp.array(dtype=wp.int32, ndim=2),
        expired_steps: wp.array(dtype=wp.int32),
        index_obj_role_gpu: wp.array(dtype=wp.int32),
        default_expired_step_list_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        speed: wp.float32,
        cooldown: wp.int32,

        player_view_object_indices: wp.array(dtype=int),
        player_view_body_local_indices: wp.array(dtype=int),
        player_count_per_world: int,
        player_bodies_per_object: int,

        bullet_view_object_indices: wp.array(dtype=int),
        bullet_count_per_world: int,

        num_objects_env: int,
        num_body_object_env: int,
    ):
        tid = wp.tid()

        raw_fire_input = actions[tid][action_shape_offset]
        
        # 計算連續的發射意圖
        fire_intent = sigmoid(raw_fire_input * 10.0)
        fire_intent_buffer[tid] = fire_intent
    
        if raw_fire_input < 0.0:   
            return

        index_bot = index_bot_players_gpu[tid]

        if cooldown_ability_owners[index_bot] != 0:
            return

        available_bullet_local_idx = _find_expired_bullet_slot(
            index_bot,
            owner_mapping_gpu,
            expired_steps,
            owner_mapping_gpu.shape[1],
        )
        if available_bullet_local_idx == -1:
            return

        # Player transformation lookup
        p_world = index_bot // num_objects_env
        p_obj_idx = _resolve_view_obj_idx(
            index_bot,
            num_objects_env,
            player_view_object_indices,
            player_count_per_world,
        )
        if p_obj_idx == -1:
            return

        p_local_tid = p_obj_idx * player_bodies_per_object
        p_local_body_idx = player_view_body_local_indices[p_local_tid]
        p_global_body_idx = p_world * num_body_object_env + p_local_body_idx

        # 獲取發射者狀態
        tf = body_q[p_global_body_idx]
        forward = quat_local_x(tf.q)

        bullet_pos = tf.p
        bullet_vel = forward * speed

        # Bullet mapping
        bullet_idx = index_obj_role_gpu[available_bullet_local_idx]
        b_world = bullet_idx // num_objects_env
        b_local_idx = bullet_idx % num_objects_env
        b_obj_idx = _resolve_bullet_view_obj_idx(
            b_local_idx, bullet_view_object_indices, bullet_count_per_world
        )
        if b_obj_idx == -1:
            return

        cooldown_ability_owners[index_bot] = cooldown
        expired_steps[available_bullet_local_idx] = default_expired_step_list_gpu[available_bullet_local_idx]

        bullet_control_pos[b_world, b_obj_idx, 0] = bullet_pos
        bullet_control_rot[b_world, b_obj_idx, 0] = tf.q
        bullet_control_vel[b_world, b_obj_idx, 0] = bullet_vel
        bullet_control_mask[b_world, b_obj_idx, 0] = bullet_control_mask[b_world, b_obj_idx, 0] | 7


    def bot_action(self, **kwargs):
        if getattr(self, "num_bot_players", 0) <= 0 or not getattr(self, "_configured", False):
            return
            
        physics_manager = self.physics_manager

        ctxs = self._shoot_ctx("bot")
        if ctxs is None:
            return
        player_ctx, bullet_ctx = ctxs
        pattern = player_ctx.pattern

        wp.launch(
            kernel=self.bot_action_gpu,
            dim=self.num_bot_players, 
            inputs=[
                self.articulation_body.control_mask_gpus[self.generated_object_pattern],
                self.articulation_body.control_pos_gpus[self.generated_object_pattern],
                self.articulation_body.control_rot_gpus[self.generated_object_pattern],
                self.articulation_body.control_vel_gpus[self.generated_object_pattern],
                physics_manager.state_0.body_q,
                self.index_bot_players_gpu,
                self.owner_mapping_gpu,
                self.ability_generated_object.expired_steps,
                self.ability_generated_object.index_obj_role_gpu,
                self.ability_generated_object.default_expired_step_list_gpu,
                self.cooldown_ability_owners,
                
                self.seeds,
                self.offset,
                self.speed, 
                self.cooldown,
                
                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.view_body_local_indices_gpus[pattern],
                player_ctx.count_per_world,
                player_ctx.bodies_per_object,
                
                self.articulation_body.view_object_indices_gpus[self.generated_object_pattern],
                bullet_ctx.count_per_world,
                
                self.articulation_body.num_objects_env,
                self.articulation_body.num_rigid_bodies_env,
            ],
            device=physics_manager.device
        )

    @wp.kernel
    def bot_action_gpu(
        bullet_control_mask: wp.array3d(dtype=wp.int32),
        bullet_control_pos: wp.array3d(dtype=wp.vec3),
        bullet_control_rot: wp.array3d(dtype=wp.quat),
        bullet_control_vel: wp.array3d(dtype=wp.vec3),
        body_q: wp.array(dtype=wp.transform), 
        index_bot_players_gpu: wp.array(dtype=wp.int32),
        owner_mapping_gpu: wp.array(dtype=wp.int32, ndim=2),
        expired_steps: wp.array(dtype=wp.int32),
        index_obj_role_gpu: wp.array(dtype=wp.int32),
        default_expired_step_list_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        
        seeds: wp.array(dtype=wp.int32),
        offset: wp.array(dtype=wp.int32),
        speed: wp.float32,
        cooldown: wp.int32,

        player_view_object_indices: wp.array(dtype=int),
        player_view_body_local_indices: wp.array(dtype=int),
        player_count_per_world: int,
        player_bodies_per_object: int,

        bullet_view_object_indices: wp.array(dtype=int),
        bullet_count_per_world: int,

        num_objects_env: int,
        num_body_object_env: int,
    ):
        tid = wp.tid()
        index_bot = index_bot_players_gpu[tid]

        if cooldown_ability_owners[index_bot] != 0:
            return

        available_bullet_local_idx = _find_expired_bullet_slot(
            index_bot,
            owner_mapping_gpu,
            expired_steps,
            owner_mapping_gpu.shape[1],
        )
        if available_bullet_local_idx == -1:
            return

        # Player transformation lookup
        p_world = index_bot // num_objects_env
        p_obj_idx = _resolve_view_obj_idx(
            index_bot,
            num_objects_env,
            player_view_object_indices,
            player_count_per_world,
        )
        if p_obj_idx == -1:
            return

        p_local_tid = p_obj_idx * player_bodies_per_object
        p_local_body_idx = player_view_body_local_indices[p_local_tid]
        p_global_body_idx = p_world * num_body_object_env + p_local_body_idx

        # 取得 Bot 自己的資訊
        tf = body_q[p_global_body_idx]
        forward = quat_local_x(tf.q)

        bullet_pos = tf.p
        bullet_vel = forward * speed

        # Bullet mapping
        bullet_idx = index_obj_role_gpu[available_bullet_local_idx]
        b_world = bullet_idx // num_objects_env
        b_local_idx = bullet_idx % num_objects_env
        b_obj_idx = _resolve_bullet_view_obj_idx(
            b_local_idx, bullet_view_object_indices, bullet_count_per_world
        )
        if b_obj_idx == -1:
            return

        rng = wp.rand_init(seeds[tid], offset=offset[tid])
        cooldown_ability_owners[index_bot] = cooldown + wp.randi(rng, 0, cooldown/2)
        offset[tid] += 1
        expired_steps[available_bullet_local_idx] = default_expired_step_list_gpu[available_bullet_local_idx]

        bullet_control_pos[b_world, b_obj_idx, 0] = bullet_pos
        bullet_control_rot[b_world, b_obj_idx, 0] = tf.q
        bullet_control_vel[b_world, b_obj_idx, 0] = bullet_vel
        bullet_control_mask[b_world, b_obj_idx, 0] = bullet_control_mask[b_world, b_obj_idx, 0] | 7


    def setup_keymapping(self):
        super().setup_keymapping(self.__class__.__name__)
        self._keyboard_action = self.control_keys["keyboard"].get("action", [])
        self._mouse_action = self.control_keys["mouse"].get("action", [])

    def update_index_bot(self, index_rl_players_gpu, num_rl_players, index_bot_players_gpu, num_bot_players):
        super().update_index_bot(index_rl_players_gpu=index_rl_players_gpu, num_rl_players=num_rl_players, index_bot_players_gpu=index_bot_players_gpu, num_bot_players=num_bot_players)
        self.fire_intent_buffer = wp.zeros(self.num_rl_players, dtype=wp.float32, device=self.physics_manager.device)

        self.num_bullets = len(self.index_ability_generated_object_gpu)
        self.hitted_bullet = wp.zeros(self.num_bullets, dtype=wp.int32, device=self.physics_manager.device)

        self.setup_bot_random_state()
        
    def reset(self):
        return super().reset()