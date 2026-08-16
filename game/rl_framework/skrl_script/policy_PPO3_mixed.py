import torch
import torch.nn as nn
import numpy as np

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space

# =============================================================================
# Policy 模型 (Actor)
# =============================================================================
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=256, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=False, reduction="mean")

        self.hidden_size = hidden_size
        self.num_layers = 1

        # 獲取通道數 (不論是 1 還是 3)
        visual_shape = observation_space.spaces["visual"].shape
        # 判斷是 (C, H, W) 還是 (H, W, C)
        # 如果最後一個維度是 1 或 3，通常是 HWC
        if visual_shape[-1] in [1, 3]:
            self.in_channels = visual_shape[-1]
            self.h, self.w = visual_shape[0], visual_shape[1]
        else:
            self.in_channels = visual_shape[0]
            self.h, self.w = visual_shape[1], visual_shape[2]

        # 1. 影像特徵提取 (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        
        # 自動計算 CNN 輸出維度
        with torch.no_grad():
            # 模擬 (Batch, C, H, W)
            dummy_input = torch.zeros(1, self.in_channels, self.h, self.w)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        # 2. 狀態特徵提取 (MLP)
        state_dim = observation_space.spaces["state"].shape[0]
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ELU(),
        )

        # 3. LSTM 層
        self.lstm = nn.LSTM(input_size=cnn_out_dim + 64, 
                            hidden_size=hidden_size, 
                            num_layers=self.num_layers, 
                            batch_first=True)

        self.action_layer = nn.Linear(hidden_size, self.num_actions)
        self.log_std_unconstrained = nn.Parameter(torch.fill_(torch.empty(self.num_actions), -1.5))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name: nn.init.constant_(param, 0)
        nn.init.orthogonal_(self.action_layer.weight, gain=0.01)

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": 1,
                "sizes":[(self.num_layers, self.hidden_size), (self.num_layers, self.hidden_size)]
            }
        }

    def compute(self, inputs, role):
        # --- 【關鍵修復：將攤平的 Tensor 還原為字典】 ---
        if not isinstance(inputs["states"], dict):
            obs_dict = unflatten_tensorized_space(self.observation_space, inputs["states"])
        else:
            obs_dict = inputs["states"]
        
        visual_obs = obs_dict["visual"].to(torch.float32) / 255.0
        state_obs = obs_dict["state"]
        
        # --- 【關鍵修復：處理維度順序】 ---
        # 如果輸入是 (Batch, H, W, C) -> 轉為 (Batch, C, H, W)
        if visual_obs.dim() == 4:
            if visual_obs.shape[-1] == self.in_channels: # 檢查最後一維是否為通道
                visual_obs = visual_obs.permute(0, 3, 1, 2)

        rnn_states = inputs.get("rnn_states", {"rnn":[None, None]})["rnn"]
        h, c = rnn_states[0], rnn_states[1]

        # --- 特徵提取 ---
        if visual_obs.dim() == 5:
            b, t, ch, h_img, w_img = visual_obs.shape
            v_flat = visual_obs.reshape(-1, ch, h_img, w_img) # 使用 reshape 或 view
            v_features = self.cnn(v_flat)
            v_features = v_features.view(b, t, -1)
            s_features = self.state_encoder(state_obs)
            combined = torch.cat([v_features, s_features], dim=-1)
        else:
            v_features = self.cnn(visual_obs)
            s_features = self.state_encoder(state_obs)
            combined = torch.cat([v_features, s_features], dim=-1)

        # --- LSTM 處理 ---
        if h is None or c is None:
            if combined.dim() == 2:
                x, (h_new, c_new) = self.lstm(combined.unsqueeze(1))
                x = x.squeeze(1)
            else:
                x, (h_new, c_new) = self.lstm(combined)
        else:
            h = h.transpose(0, 1).contiguous()
            c = c.transpose(0, 1).contiguous()
            if combined.dim() == 2:
                x, (h_new, c_new) = self.lstm(combined.unsqueeze(1), (h, c))
                x = x.squeeze(1)
            else:
                x, (h_new, c_new) = self.lstm(combined, (h, c))

        h_new = h_new.transpose(0, 1)
        c_new = c_new.transpose(0, 1)

        mean_actions = self.action_layer(x)
        log_std = torch.clamp(self.log_std_unconstrained, -3, 0)

        return mean_actions, log_std, {"rnn": [h_new, c_new]}

# =============================================================================
# Value 模型 (Critic)
# =============================================================================
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=256, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self)

        self.hidden_size = hidden_size
        self.num_layers = 1

        visual_shape = observation_space.spaces["visual"].shape
        if visual_shape[-1] in [1, 3]:
            self.in_channels, self.h, self.w = visual_shape[-1], visual_shape[0], visual_shape[1]
        else:
            self.in_channels, self.h, self.w = visual_shape[0], visual_shape[1], visual_shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.h, self.w)
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        state_dim = observation_space.spaces["state"].shape[0]
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.ELU())

        self.lstm = nn.LSTM(input_size=cnn_out_dim + 64, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True)
        self.value_layer = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name: nn.init.constant_(param, 0)
        nn.init.orthogonal_(self.value_layer.weight, gain=1.0)

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": 1,
                "sizes":[(self.num_layers, self.hidden_size), (self.num_layers, self.hidden_size)]
            }
        }


    def compute(self, inputs, role):
        # --- 【關鍵修復：將攤平的 Tensor 還原為字典】 ---
        if not isinstance(inputs["states"], dict):
            obs_dict = unflatten_tensorized_space(self.observation_space, inputs["states"])
        else:
            obs_dict = inputs["states"]
            
        visual_obs = obs_dict["visual"].to(torch.float32) / 255.0
        state_obs = obs_dict["state"]
        
        # --- 【關鍵修復：處理維度順序】 ---
        if visual_obs.dim() == 4:
            if visual_obs.shape[-1] == self.in_channels:
                visual_obs = visual_obs.permute(0, 3, 1, 2)
        elif visual_obs.dim() == 5:
            if visual_obs.shape[-1] == self.in_channels:
                visual_obs = visual_obs.permute(0, 1, 4, 2, 3)

        rnn_states = inputs.get("rnn_states", {"rnn": [None, None]})["rnn"]
        h, c = rnn_states[0], rnn_states[1]

        if visual_obs.dim() == 5:
            b, t, ch, h_img, w_img = visual_obs.shape
            v_features = self.cnn(visual_obs.reshape(-1, ch, h_img, w_img)).view(b, t, -1)
            s_features = self.state_encoder(state_obs)
            combined = torch.cat([v_features, s_features], dim=-1)
        else:
            v_features = self.cnn(visual_obs)
            s_features = self.state_encoder(state_obs)
            combined = torch.cat([v_features, s_features], dim=-1)

        if h is None or c is None:
            if combined.dim() == 2:
                x, (h_new, c_new) = self.lstm(combined.unsqueeze(1))
                x = x.squeeze(1)
            else:
                x, (h_new, c_new) = self.lstm(combined)
        else:
            h = h.transpose(0, 1).contiguous()
            c = c.transpose(0, 1).contiguous()
            if combined.dim() == 2:
                x, (h_new, c_new) = self.lstm(combined.unsqueeze(1), (h, c))
                x = x.squeeze(1)
            else:
                x, (h_new, c_new) = self.lstm(combined, (h, c))

        h_new = h_new.transpose(0, 1)
        c_new = c_new.transpose(0, 1)

        value = self.value_layer(x)
        return value, {"rnn":[h_new, c_new]}
    

