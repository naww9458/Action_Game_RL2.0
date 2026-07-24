import torch
import torch.nn as nn
import numpy as np
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# =============================================================================
# Policy 模型 (Actor)
# =============================================================================
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, 
                 hidden_size=256, **kwargs):
        # 注意：移除了 num_envs 和 num_layers 參數，純 MLP 不需要這些
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=False, reduction="mean")

        self.hidden_size = hidden_size

        # 使用純 MLP (3 層)
        self.encoder = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )
        self.action_layer = nn.Linear(hidden_size, self.num_actions)
        
        self._init_weights()
        
        # 將 log_std 初始值調小，避免初期隨機動作過於劇烈 (-0.5 改為 -1.5)
        self.log_std_unconstrained = nn.Parameter(torch.fill_(torch.empty(self.num_actions), -2.5))

    def _init_weights(self):
        # 初始化 Encoder (使用 Orthogonal 正交初始化)
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0) 
                nn.init.constant_(m.bias, 0)

        # 動作層初始化：使用極小權重，讓 Agent 初期動作趨於平穩
        nn.init.orthogonal_(self.action_layer.weight, gain=0.01)
        nn.init.constant_(self.action_layer.bias, 0)

    def get_specification(self):
        # 不再使用 RNN，回傳空字典
        return {}

    def compute(self, inputs, role):
        states = inputs["states"]
        
        # 直接通過 MLP (修正了你原本代碼錯寫的 self.net)
        x = self.encoder(states)
        mean_actions = self.action_layer(x)
        
        # 限制 log_std，防止極端探索導致崩潰
        log_std = torch.clamp(self.log_std_unconstrained, -3, 0)
        
        # 回傳空字典 {} 代替 rnn_states
        return mean_actions, log_std, {}


# =============================================================================
# Value 模型 (Critic)
# =============================================================================
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, 
                 hidden_size=256, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self)

        self.hidden_size = hidden_size

        # Critic 的 Encoder 結構與 Actor 完全對稱 (3 層 MLP)
        self.encoder = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )

        self.value_layer = nn.Linear(hidden_size, 1)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0) 
                nn.init.constant_(m.bias, 0)
        
        # Value 層的 gain 通常設為 1.0
        nn.init.orthogonal_(self.value_layer.weight, gain=1.0)
        nn.init.constant_(self.value_layer.bias, 0)

    def get_specification(self):
        # 不再使用 RNN，回傳空字典
        return {}

    def compute(self, inputs, role):
        states = inputs["states"]

        # 移除所有 LSTM 處理邏輯，直接通過 MLP
        x = self.encoder(states)
        value = self.value_layer(x)

        if not torch.isfinite(value).all():
            raise RuntimeError("Value contains NaN!")

        # 回傳空字典 {} 代替 rnn_states
        return value, {}