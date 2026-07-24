import gymnasium
import torch
import warp as wp

from torch.optim import Adam
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model

from typing import Any, Mapping, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from game.skrl_script.wrapperSKRL import WarpEnv


APG_DEFAULT_CONFIG = {
    "learning_rate": 1e-3, 
    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}

class APG(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        requires_grad: bool = True,
        env: 'WarpEnv' = None,
    ) -> None:

        self.num_envs = cfg["num_envs"]
        self.horizon = cfg["horizon"]
        self.requires_grad = requires_grad
        self.env = env
        self.num_agents_total = self.env.num_agents_each_env * self.num_envs

        super().__init__(models=models, memory=memory, observation_space=observation_space, action_space=action_space, device=device, cfg=cfg)

        self.policy = self.models.get("policy", None)
        self.optimizer = Adam(self.policy.parameters(), lr=cfg["learning_rate"])

        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["optimizer"] = self.optimizer

        self._device_type = torch.device(device).type
        self._mixed_precision = self.cfg["mixed_precision"]

        self.episode_total_grad = 0.0
        self.total_loss_wp = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=self.requires_grad)
        self.actions_pt_list = []
        self.actions_wp_list = []

        self.rnn_states = {"rnn": [None, None]}


    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, _, self.rnn_states = self.policy.compute({"states": states, "rnn_states": self.rnn_states}, role="policy")

        return actions, self.rnn_states

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:


        self.actions_pt_list.append(actions)

        # 從環境獲取帶梯度的 Warp Array
        actions_wp = infos["actions_wp"]
        rewards_wp = infos["rewards_wp"]
        infos = {} # TODO

        if actions_wp is not None:
            self.actions_wp_list.append(actions_wp)

        # 3. 在 Warp Tape 內部累積 Loss
        wp.launch(kernel=self.compute_loss_kernel, 
                  dim=self.num_agents_total, 
                  inputs=[rewards_wp, self.total_loss_wp, float(self.num_agents_total), float(self.horizon)])

        super().record_transition(states=states, actions=actions, rewards=rewards, next_states=next_states, terminated=terminated, truncated=truncated, infos=infos, timestep=timestep, timesteps=timesteps)


    def pre_interaction(): 
        pass

    def post_interaction(self, timestep: int, timesteps: int, tape: wp.Tape) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # self.set_mode("train")
        self._update(timestep, timesteps, tape)

        # 因爲 wp.Tape 的特殊性，評估模式可能導致 cudnn RNN backward can only be called in training mode 的報錯 
        # self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)


    def track_data(self, tag: str, value: float) -> None:
        super().track_data(tag, value)

    def _update(self, timestep: int, timesteps: int, tape: wp.Tape) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        :param tape: Warp tape
        :type tape: wp.Tape
        """

        # === 解析梯度反向傳播 (Warp 端) ===
        # 從 total_loss_wp 往回計算，得出物理環境對每一階段 actions_wp 的需求梯度
        tape.backward(loss=self.total_loss_wp)

# # =============================================================================
#         # Authoritative APG signals: actions_wp.grad / policy grads.
#         # After tape.backward, assignment/zero_ adjoints often leave
#         # step_total_rewards_rl_diff.grad and control_* buffer .grad at 0 even
#         # when the chain is healthy — do not treat those as proof of a break.
#         print("=============================================================================")
#         action_grad_sum = 0.0
#         for wp_array in self.actions_wp_list:
#             if wp_array.grad is not None:
#                 action_grad_sum += float(wp.to_torch(wp_array.grad).abs().sum().item())
#         print(f"動作梯度總和 (actions_wp, 權威指標): {action_grad_sum}")

#         import numpy as np
#         if self.env.game.reward_calculator.step_total_rewards_rl_diff.grad is not None:
#             print(f"獎勵層梯度總和: {np.abs(self.env.game.reward_calculator.step_total_rewards_rl_diff.grad.numpy()).sum()}")
#         else:
#             print("獎勵層沒有梯度！ (檢查 step_total_rewards_rl_diff.requires_grad)")

#         # # 2. 檢查物理座標是否有梯度 (這是你目前獎勵的來源)
#         # q_grad = wp.to_torch(self.env.game.physics_manager.state_0.body_q.grad)
#         # print(f"物理座標梯度總和 (僅供參考): {q_grad.abs().sum().item()}")
        
#         # qd_grad = wp.to_torch(self.env.game.physics_manager.state_0.body_qd.grad)
#         # print(f"物理速度梯度總和 (僅供參考): {qd_grad.abs().sum().item()}")
        
#         # bv_grad = wp.to_torch(self.env.game.articulation_body.control_vel_gpus["RL_player_default"].grad)
#         # print(f"綫速度緩衝區梯度總和 (僅供參考): {bv_grad.abs().sum().item()}")
#         # bo_grad = wp.to_torch(self.env.game.articulation_body.control_omega_gpus["RL_player_default"].grad)
#         # print(f"角速度緩衝區梯度總和 (僅供參考): {bo_grad.abs().sum().item()}")


#         f_grad = wp.to_torch(self.env.game.physics_manager.state_0.body_f.grad)
#         print(f"物理力和扭矩梯度總和 (僅供參考): {f_grad.abs().sum().item()}")

#         bf_grad = wp.to_torch(self.env.game.articulation_body.control_force_gpus["RL_player_default"].grad)
#         print(f"力緩衝區梯度總和 (僅供參考): {bf_grad.abs().sum().item()}")
#         bt_grad = wp.to_torch(self.env.game.articulation_body.control_torque_gpus["RL_player_default"].grad)
#         print(f"扭矩緩衝區梯度總和 (僅供參考): {bt_grad.abs().sum().item()}")
#         print("=============================================================================")
# # =============================================================================


        # === 收集物理梯度並橋接回 PyTorch ===
        # 我們把 Warp 算出的動作梯度，當作 PyTorch 反向傳播的起點
        valid_actions_pt = []
        valid_actions_grad = []
        
        has_nan = False
        i = 0
        for pt_tensor, wp_array in zip(self.actions_pt_list, self.actions_wp_list):
            i += 1
            # wp_array.grad 儲存了 Warp 算出來的梯度，轉回 PyTorch Tensor
            grad_tensor = wp.to_torch(wp_array.grad).clone()

            # 檢查 NaN
            if not torch.isfinite(grad_tensor).all():
                has_nan = True
                print(f"[Warning] NaN detected at step {i}. Discarding this horizon...")
                break # 直接放棄這個 Horizon 的所有數據，不更新網路

            # # 梯度歸一化 (確保物理環境傳來的力道大小穩定)
            # grad_mean = grad_tensor.mean()
            # grad_std = grad_tensor.std() + 1e-8
            # grad_tensor = (grad_tensor - grad_mean) / grad_std


            # 梯度裁剪 (必須放在歸一化之後！防止極端特異值破壞分佈)
            grad_tensor = torch.clamp(grad_tensor, -1.0, 1.0)

            valid_actions_pt.append(pt_tensor)
            valid_actions_grad.append(grad_tensor)

        # 執行 PyTorch 的反向傳播
        self.optimizer.zero_grad()

        if not has_nan:
            # 將 Warp 算好的梯度傳給 PyTorch，讓 PyTorch 繼續往網路權重反向傳播
            torch.autograd.backward(tensors=valid_actions_pt, grad_tensors=valid_actions_grad)
            
            # # (可選) 全局網路梯度裁剪，保護 RNN 不發生梯度爆炸
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

            # 統計網路層的真實梯度大小 (用於 TensorBoard 或 Log)
            for name, param in self.policy.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.abs().mean().item()
                    self.episode_total_grad += grad_norm

            # 更新神經網路權重
            self.optimizer.step()

        # === 截斷 RNN 隱藏狀態的梯度圖連接 ===
        # 我們需要保留 rnn_states 的數值，但切斷它與上一段計算圖的聯繫
        if self.rnn_states["rnn"][0] is not None:
            self.rnn_states["rnn"] = [s.detach() for s in self.rnn_states["rnn"]]
        
        # 清除 Tape 和 Loss 緩存，釋放顯存
        tape.zero()
        self.total_loss_wp.zero_()
        self.actions_pt_list = []
        self.actions_wp_list = []

    def reset(self):
        self.episode_total_grad = 0.0
        self.rnn_states = {"rnn": [None, None]}

    @wp.kernel
    def compute_loss_kernel(rewards: wp.array(dtype=wp.float32), 
                            loss: wp.array(dtype=wp.float32), 
                            num_agents_total: float, 
                            horizon: float):
        tid = wp.tid()
        # 我們的目標是最大化累積 Reward，所以 Loss 是負的 Reward。
        # 除以 num_agents_total * horizon 以計算所有環境和時間步的平均 Loss，穩定梯度
        wp.atomic_add(loss, 0, -rewards[tid] / (num_agents_total * horizon))











