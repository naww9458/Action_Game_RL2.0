"""
Level 6-2 — MPM 求解器測試關卡。

在長方體容器中放置 MPM 粒子，並以 coupled [mujoco, mpm] 求解器模擬：
- mujoco: 玩家 (unitree_g1) 與容器剛體平台
- mpm:    容器內的顆粒材質

環境重置時，粒子位置由 DeformableBody 重置，MPM 專屬的內部變形狀態
(彈性應變/塑性體積/應力/變形梯度) 在此一併清除，避免殘留應變造成彈跳。
"""

import warp as wp

from script.levels.levels import Level_Default
from script.simulate.solvers.mpm import reset_mpm_particle_state


class Level6_2(Level_Default):
    def reset_env(self, terminated, current_step):
        super().reset_env(terminated=terminated, current_step=current_step)

        # 重置 MPM 粒子的內部變形狀態 (安全空操作當無 mpm 屬性時)
        reset_mpm_particle_state(self.physics_manager.state_0)
