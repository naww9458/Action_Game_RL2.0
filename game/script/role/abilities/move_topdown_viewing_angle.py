import warp as wp

from .ability import Ability
from script.game_config import GameConfig

class Move_topdown_viewing_angle(Ability):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.seeds = None
        self.random_offset = None
        self.forces_bot = None
        self.last_reset_bot_xy_force = None
        self.reset_bot_xy_force_cooldown = wp.int32(int(0.3 * GameConfig.FPS_ACTION))

        cfg = Ability._default_configs.root.get(self.ability_name)
        self.decay = wp.float32(float(getattr(cfg, "decay", 0.8)) if cfg else 0.8)

    def apply_ability_config_overrides(self, overrides) -> None:
        super().apply_ability_config_overrides(overrides)
        if isinstance(overrides, dict) and "decay" in overrides:
            self.decay = wp.float32(float(overrides["decay"]))

    def human_control_interface(self, keyboard_keys, mouse_buttons, look_yaw, index_human_player_gpu, **kwargs):
        if look_yaw is None or not getattr(self, "_configured", False):
            return None
        
        move_f = self._is_pressed(self.keyboard_front, self.mouse_front, keyboard_keys, mouse_buttons)
        move_l = self._is_pressed(self.keyboard_left, self.mouse_left, keyboard_keys, mouse_buttons)
        move_b = self._is_pressed(self.keyboard_back, self.mouse_back, keyboard_keys, mouse_buttons)
        move_r = self._is_pressed(self.keyboard_right, self.mouse_right, keyboard_keys, mouse_buttons)

        local_x = wp.float32(move_f - move_b)
        local_y = wp.float32(move_l - move_r)

        ctx = self._view_ctx("human")
        if ctx is None:
            return
        pattern = ctx.pattern
        bodies_per_object = ctx.bodies_per_object

        wp.launch(
            kernel=self.human_control_interface_gpu,
            dim=1, 
            inputs=[
                self.articulation_body.control_force_gpus[pattern],
                self.articulation_body.control_vel_gpus[pattern],
                self.articulation_body.control_mask_gpus[pattern],
                self.physics_manager.state_0.body_qd,
                index_human_player_gpu,
                self.cooldown_ability_owners,

                look_yaw,
                local_x,
                local_y,

                self.force,
                self.speed,
                self.decay,

                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.num_objects_env,
                ctx.count_per_world,
                self.articulation_body.view_body_local_indices_gpus[pattern],
                self.articulation_body.num_rigid_bodies_env,
                bodies_per_object,
            ],
            device=self.physics_manager.device
        )

    @wp.kernel
    def human_control_interface_gpu(
        control_force: wp.array3d(dtype=wp.vec3),  
        control_vel: wp.array3d(dtype=wp.vec3),
        control_mask: wp.array3d(dtype=wp.int32),
        body_qd: wp.array(dtype=wp.spatial_vector),
        index_human_player_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        
        look_yaws: wp.float32, 
        local_x: wp.float32,
        local_y: wp.float32,

        force: float,
        speed: float,
        decay: float,

        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        view_body_local_indices: wp.array(dtype=int),
        num_body_object_env: int,
        bodies_per_object: int,
    ):
        if cooldown_ability_owners[0] != 0:
            return

        player_idx = index_human_player_gpu[0]
        world = player_idx // num_objects_env
        local_idx = player_idx % num_objects_env

        obj_idx = wp.int32(-1)
        for i in range(count_per_world):
            if view_object_indices[i] == local_idx:
                obj_idx = i
                break

        if obj_idx == -1:
            return

        # 定位 Root Body (body_in_obj_idx = 0)
        local_tid = obj_idx * bodies_per_object
        local_body_idx = view_body_local_indices[local_tid]
        global_body_idx = world * num_body_object_env + local_body_idx

        # 角度轉換 (度 -> 弧度)
        theta = look_yaws * (3.14159265 / 180.0)
        
        cos_t = wp.cos(theta)
        sin_t = wp.sin(theta)

        # 2D 旋轉變換 (Local to World)
        world_x = (local_x * cos_t - local_y * sin_t) * force
        world_y = (local_x * sin_t + local_y * cos_t) * force

        control_force[world, obj_idx, 0] += wp.vec3(world_x, world_y, 0.0)

        # 速度衰減邏輯：當超過上限時乘以衰減係數，保持轉向流暢性
        curr_qd = body_qd[global_body_idx]
        curr_v_2d_sq = curr_qd[0]*curr_qd[0] + curr_qd[1]*curr_qd[1]
        
        if curr_v_2d_sq > speed * speed:
            control_vel[world, obj_idx, 0] = wp.vec3(curr_qd[0] * decay, curr_qd[1] * decay, curr_qd[2])
            control_mask[world, obj_idx, 0] = control_mask[world, obj_idx, 0] | 4


    def rl_action(self, actions, **kwargs):
        if getattr(self, "num_rl_players", 0) <= 0 or not getattr(self, "_configured", False):
            return

        ctx = self._view_ctx("rl")
        if ctx is None:
            return
        pattern = ctx.pattern
        bodies_per_object = ctx.bodies_per_object

        action_shape_offset = self.action_shape_offset if self.action_shape_offset is not None else 0

        wp.launch(
            kernel=self.rl_action_gpu,
            dim=self.num_rl_players, 
            inputs=[
                self.articulation_body.control_force_gpus[pattern],
                self.articulation_body.control_vel_gpus[pattern],
                self.articulation_body.control_mask_gpus[pattern],
                self.physics_manager.state_0.body_q,
                self.physics_manager.state_0.body_qd,
                self.index_rl_players_gpu,
                self.cooldown_ability_owners,

                actions,
                action_shape_offset,

                self.force,
                self.speed,
                self.decay,
                
                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.num_objects_env,
                ctx.count_per_world,
                self.articulation_body.view_body_local_indices_gpus[pattern],
                self.articulation_body.num_rigid_bodies_env,
                bodies_per_object,
            ],
            device=self.physics_manager.device
        )

    @wp.kernel
    def rl_action_gpu(
        control_force: wp.array3d(dtype=wp.vec3),  
        control_vel: wp.array3d(dtype=wp.vec3),
        control_mask: wp.array3d(dtype=wp.int32),
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        index_rl_players_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        
        actions: wp.array2d(dtype=wp.float32),
        action_shape_offset: wp.int32,

        force: float,
        speed: float,
        decay: float,

        view_object_indices_gpus: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        view_body_local_indices_gpus: wp.array(dtype=int),
        num_body_object_env: int,
        bodies_per_object: int,
    ):
        tid = wp.tid()
        player_idx = index_rl_players_gpu[tid]

        # Warp AD: early return / break zeros grads for the whole launch.
        active = wp.where(cooldown_ability_owners[player_idx] == 0, 1.0, 0.0)

        local_body_idx = view_body_local_indices_gpus[0]
        global_body_idx = tid * num_body_object_env + local_body_idx

        # 獲取四元數
        tf = body_q[global_body_idx]
        bot_rot = wp.normalize(tf.q)

        action_x = actions[tid, action_shape_offset]
        action_y = actions[tid, action_shape_offset + 1]

        dir_local = wp.vec3(action_x, action_y, 0.0)

        # 使用四元數進行旋轉 (Local -> World)
        dir_world = wp.quat_rotate(bot_rot, dir_local)

        # Index with tid only — runtime (world, obj_idx) scatters drop value grads in Warp AD.
        # TODO 一個環境只能有一個 RL Agent
        control_force[tid, 0, 0] += wp.vec3(
            dir_world[0] * force * active,
            dir_world[1] * force * active,
            0.0,
        )

        # 速度衰減邏輯：當超過上限時乘以衰減係數，保持轉向流暢性
        curr_qd = body_qd[global_body_idx]
        curr_v_2d_sq = curr_qd[0]*curr_qd[0] + curr_qd[1]*curr_qd[1]
        overspeed = wp.where(curr_v_2d_sq > speed * speed, 1.0, 0.0) * active
        # Blend toward decayed velocity without branching (mask bit still discrete).
        # TODO 一個環境只能有一個 RL Agent
        control_vel[tid, 0, 0] = wp.vec3(
            curr_qd[0] * (1.0 - overspeed + overspeed * decay),
            curr_qd[1] * (1.0 - overspeed + overspeed * decay),
            curr_qd[2],
        )
        if overspeed > 0.5:
            # TODO 一個環境只能有一個 RL Agent
            control_mask[tid, 0, 0] = control_mask[tid, 0, 0] | 4


    def bot_action(self, **kwargs) -> float:
        if getattr(self, "num_bot_players", 0) <= 0 or not getattr(self, "_configured", False):
            return
            
        ctx = self._view_ctx("bot")
        if ctx is None:
            return
        pattern = ctx.pattern
        bodies_per_object = ctx.bodies_per_object

        wp.launch(
            kernel=self.bot_action_gpu,
            dim=self.num_bot_players, 
            inputs=[
                self.articulation_body.control_force_gpus[pattern],
                self.articulation_body.control_vel_gpus[pattern],
                self.articulation_body.control_mask_gpus[pattern],
                self.physics_manager.state_0.body_qd,
                self.index_bot_players_gpu,
                self.cooldown_ability_owners,
                self.seeds,
                self.random_offset,
                self.forces_bot,
                self.last_reset_bot_xy_force,
                self.reset_bot_xy_force_cooldown,

                self.force,
                self.speed,
                self.decay,

                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.num_objects_env,
                ctx.count_per_world,
                self.articulation_body.view_body_local_indices_gpus[pattern],
                self.articulation_body.num_rigid_bodies_env,
                bodies_per_object,
            ],
            device=self.physics_manager.device
        )

    @wp.kernel
    def bot_action_gpu(
        control_force: wp.array3d(dtype=wp.vec3),  
        control_vel: wp.array3d(dtype=wp.vec3),
        control_mask: wp.array3d(dtype=wp.int32),
        body_qd: wp.array(dtype=wp.spatial_vector),
        index_bot_players_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        seeds: wp.array(dtype=wp.int32),
        random_offset: wp.array(dtype=wp.int32),
        forces_bot: wp.array(dtype=wp.vec3),
        last_reset_bot_xy_force: wp.array(dtype=wp.int32),
        reset_bot_xy_force_cooldown: wp.int32,
        force: wp.float32,
        speed: wp.float32,
        decay: wp.float32,

        view_object_indices_gpus: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        view_body_local_indices_gpus: wp.array(dtype=int),
        num_body_object_env: int,
        bodies_per_object: int,
    ):
        tid = wp.tid()
        player_idx = index_bot_players_gpu[tid]

        if cooldown_ability_owners[player_idx] != 0:
            return

        world = player_idx // num_objects_env
        local_idx = player_idx % num_objects_env

        obj_idx = wp.int32(-1)
        for i in range(count_per_world):
            if view_object_indices_gpus[i] == local_idx:
                obj_idx = i
                break

        if obj_idx == -1:
            return

        # 定位 Root Body
        local_tid = obj_idx * bodies_per_object
        local_body_idx = view_body_local_indices_gpus[local_tid]
        global_body_idx = world * num_body_object_env + local_body_idx

        # 每隔一段時間重置 bot_x_force
        if last_reset_bot_xy_force[tid] == 0 :
            rng = wp.rand_init(seeds[tid], offset=random_offset[tid])
            random_offset[tid] += 1

            # 生成 -1.0 到 1.0 之間的隨機數
            forces_bot[tid] = wp.vec3(wp.randf(rng, -1.0, 1.0)*force, wp.randf(rng, -1.0, 1.0)*force, 0.0)

            last_reset_bot_xy_force[tid] = reset_bot_xy_force_cooldown+1
        
        last_reset_bot_xy_force[tid] -= 1
        control_force[world, obj_idx, 0] += forces_bot[tid]

        # 速度衰減邏輯：當超過上限時乘以衰減係數，保持轉向流暢性
        curr_qd = body_qd[global_body_idx]
        curr_v_2d_sq = curr_qd[0]*curr_qd[0] + curr_qd[1]*curr_qd[1]
        
        if curr_v_2d_sq > speed * speed:
            control_vel[world, obj_idx, 0] = wp.vec3(curr_qd[0] * decay, curr_qd[1] * decay, curr_qd[2])
            control_mask[world, obj_idx, 0] = control_mask[world, obj_idx, 0] | 4

    def setup_keymapping(self):
        super().setup_keymapping(self.__class__.__name__)
        self.keyboard_front = self.control_keys["keyboard"].get("front", [])
        self.keyboard_left = self.control_keys["keyboard"].get("left", [])
        self.keyboard_back = self.control_keys["keyboard"].get("back", [])
        self.keyboard_right = self.control_keys["keyboard"].get("right", [])

        self.mouse_front = self.control_keys["mouse"].get("front", [])
        self.mouse_left = self.control_keys["mouse"].get("left", [])
        self.mouse_back = self.control_keys["mouse"].get("back", [])
        self.mouse_right = self.control_keys["mouse"].get("right", [])

    def update_index_bot(self, index_rl_players_gpu, num_rl_players, index_bot_players_gpu, num_bot_players):
        super().update_index_bot(index_rl_players_gpu=index_rl_players_gpu, num_rl_players=num_rl_players, index_bot_players_gpu=index_bot_players_gpu, num_bot_players=num_bot_players)
        self.forces_bot = wp.zeros(self.num_bot_players, dtype=wp.vec3, device=self.physics_manager.device)
        self.last_reset_bot_xy_force = wp.zeros(shape=self.num_bot_players, dtype=wp.int32, device=self.physics_manager.device)
        self.setup_bot_random_state(offset_attr="random_offset")

    def reset(self):
        return super().reset()