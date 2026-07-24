import numpy as np
import warp as wp

from .ability import Ability
from script.game_config import GameConfig


class Jump(Ability):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        
        # 定義世界坐標系的「向上」向量
        # 2D 常用 [0, 1, 0] (Y 軸向上)
        # 3D 常用 [0, 0, 1] (Z 軸向上)
        self.up_axis = np.array([0.0, 1.0, 0.0]) 

        self.seeds = None
        self.offset = None


    def human_control_interface(self, keyboard_keys, mouse_buttons, index_human_player_gpu: wp.array, **kwargs):
        if not getattr(self, "_configured", False):
            return

        physics_manager = self.physics_manager

        if self._is_pressed(self._keyboard_action, self._mouse_action, keyboard_keys, mouse_buttons) == 1:
            ctx = self._view_ctx("human")
            if ctx is None:
                return
            pattern = ctx.pattern
            bodies_per_object = ctx.bodies_per_object

            wp.launch(
                kernel=self.human_action_gpu, 
                dim=1, 
                inputs=[
                    self.articulation_body.control_force_gpus[pattern],
                    index_human_player_gpu, 
                    self.cooldown_ability_owners,
                    self.force,
                    self.cooldown,

                    self.articulation_body.view_object_indices_gpus[pattern],
                    self.articulation_body.num_objects_env,
                    ctx.count_per_world,
                    self.articulation_body.view_body_local_indices_gpus[pattern],
                    self.articulation_body.num_rigid_bodies_env,
                    bodies_per_object,
                ],
                device=physics_manager.device
            )

    @wp.kernel
    def human_action_gpu(
        control_force: wp.array3d(dtype=wp.vec3),  
        index_human_player_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        
        force: wp.float32,
        cooldown: wp.int32,

        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        view_body_local_indices: wp.array(dtype=int),
        num_body_object_env: int,
        bodies_per_object: int,
    ):
        player_idx = index_human_player_gpu[0]

        if cooldown_ability_owners[player_idx] != 0:
            return

        world = player_idx // num_objects_env
        local_idx = player_idx % num_objects_env

        obj_idx = wp.int32(-1)
        for i in range(count_per_world):
            if view_object_indices[i] == local_idx:
                obj_idx = i
                break

        if obj_idx == -1:
            return

        cooldown_ability_owners[player_idx] = cooldown
        control_force[world, obj_idx, 0] += wp.vec3(0.0, 0.0, force)


    def rl_action(self, actions, **kwargs):
        if getattr(self, "num_rl_players", 0) <= 0 or not getattr(self, "_configured", False):
            return

        physics_manager = self.physics_manager

        ctx = self._view_ctx("rl")
        if ctx is None:
            return
        pattern = ctx.pattern
        bodies_per_object = ctx.bodies_per_object
        
        offset = self.action_shape_offset if self.action_shape_offset is not None else 0

        wp.launch(
            kernel=self.rl_action_gpu, 
            dim=self.num_rl_players, 
            inputs=[
                self.articulation_body.control_force_gpus[pattern],
                actions,
                offset,
                self.index_rl_players_gpu,     
                self.cooldown_ability_owners,
                self.force,
                self.cooldown,

                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.num_objects_env,
                ctx.count_per_world,
                self.articulation_body.view_body_local_indices_gpus[pattern],
                self.articulation_body.num_rigid_bodies_env,
                bodies_per_object,
            ],
            device=physics_manager.device
        )

    @wp.kernel
    def rl_action_gpu(
        control_force: wp.array3d(dtype=wp.vec3),
        actions: wp.array2d(dtype=wp.float32), 
        action_shape_offset: wp.int32,
        index_rl_players_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        force: wp.float32,
        cooldown: wp.int32,

        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        view_body_local_indices: wp.array(dtype=int),
        num_body_object_env: int,
        bodies_per_object: int,
    ):
        tid = wp.tid()
        player_idx = index_rl_players_gpu[tid]

        if cooldown_ability_owners[player_idx] != 0:
            return

        jump_intent = actions[tid, action_shape_offset]
        
        # 假設大於 0.5 時觸發跳躍
        if jump_intent > 0.5:
            world = player_idx // num_objects_env
            local_idx = player_idx % num_objects_env

            obj_idx = wp.int32(-1)
            for i in range(count_per_world):
                if view_object_indices[i] == local_idx:
                    obj_idx = i
                    break

            if obj_idx != -1:
                cooldown_ability_owners[player_idx] = cooldown
                control_force[world, obj_idx, 0] += wp.vec3(0.0, 0.0, force)


    def bot_action(self, **kwargs):
        if getattr(self, "num_bot_players", 0) <= 0 or not getattr(self, "_configured", False):
            return
            
        physics_manager = self.physics_manager

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
                self.index_bot_players_gpu,
                self.cooldown_ability_owners,
                
                self.seeds,
                self.offset,
                self.force,
                self.cooldown,

                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.num_objects_env,
                ctx.count_per_world,
                self.articulation_body.view_body_local_indices_gpus[pattern],
                self.articulation_body.num_rigid_bodies_env,
                bodies_per_object,
            ],
            device=physics_manager.device
        )

    @wp.kernel
    def bot_action_gpu(
        control_force: wp.array3d(dtype=wp.vec3),
        index_bot_players_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        
        seeds: wp.array(dtype=wp.int32),
        offset: wp.array(dtype=wp.int32),
        force: wp.float32,
        cooldown: wp.int32,

        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        view_body_local_indices: wp.array(dtype=int),
        num_body_object_env: int,
        bodies_per_object: int,
    ):
        tid = wp.tid()
        player_idx = index_bot_players_gpu[tid]

        if cooldown_ability_owners[player_idx] != 0:
            return

        rng = wp.rand_init(seeds[tid], offset=offset[tid])
        offset[tid] += 1

        # bot 隨機一定機率起跳 (e.g. 1% 機率 / tick)
        if wp.randf(rng, 0.0, 1.0) < 0.01:
            world = player_idx // num_objects_env
            local_idx = player_idx % num_objects_env

            obj_idx = wp.int32(-1)
            for i in range(count_per_world):
                if view_object_indices[i] == local_idx:
                    obj_idx = i
                    break

            if obj_idx != -1:
                cooldown_ability_owners[player_idx] = cooldown
                control_force[world, obj_idx, 0] += wp.vec3(0.0, 0.0, force)

    def setup_keymapping(self):
        super().setup_keymapping(self.__class__.__name__)
        self._keyboard_action = self.control_keys["keyboard"].get("action", [])
        self._mouse_action = self.control_keys["mouse"].get("action", [])

    def update_index_bot(self, index_rl_players_gpu, num_rl_players, index_bot_players_gpu, num_bot_players):
        super().update_index_bot(index_rl_players_gpu=index_rl_players_gpu, num_rl_players=num_rl_players, index_bot_players_gpu=index_bot_players_gpu, num_bot_players=num_bot_players)

        import numpy as np
        seed = GameConfig.SEED
        seeds_np = np.arange(seed, seed+self.num_bot_players + 1, dtype=np.int32)
        self.seeds = wp.array(seeds_np, dtype=wp.int32, device=self.physics_manager.device)
        self.offset = wp.zeros(shape=self.num_bot_players, dtype=wp.int32, device=self.physics_manager.device)
        
    def reset(self):
        return super().reset()