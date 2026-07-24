import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# =============================================================================
# Policy 模型 (Actor)
# =============================================================================
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=256, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=False, reduction="mean")

        self.hidden_size = hidden_size
        self.num_layers = 1 # 通常一層 LSTM 就夠了

        # 1. 特徵提取層
        self.encoder = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
        )

        # 2. LSTM 層
        self.lstm = nn.LSTM(input_size=hidden_size, 
                            hidden_size=hidden_size, 
                            num_layers=self.num_layers, 
                            batch_first=True)

        # 3. 動作輸出層
        self.action_layer = nn.Linear(hidden_size, self.num_actions)
        self.log_std_unconstrained = nn.Parameter(torch.fill_(torch.empty(self.num_actions), -1.5))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        nn.init.orthogonal_(self.action_layer.weight, gain=0.01)

    def get_specification(self):
        # 修正為 skrl 要求的 RNN 規格格式
        # 這樣 skrl 才會在記憶體中正確分配 (batch_size, num_layers, hidden_size) 的張量
        return {
            "rnn": {
                "sequence_length": 1,
                "sizes":[(self.num_layers, self.hidden_size),  # h 的形狀
                          (self.num_layers, self.hidden_size)]  # c 的形狀
            }
        }

    def compute(self, inputs, role):
        states = inputs["states"]
        rnn_states = inputs.get("rnn_states", {"rnn":[None, None]})["rnn"]
        
        # 獲取隱狀態
        h, c = rnn_states[0], rnn_states[1]

        # 防禦性截斷
        states = torch.nan_to_num(torch.clamp(states, -10.0, 10.0), 0.0)
        x = self.encoder(states)

        # 處理 None 的情況 (剛開始訓練或 skrl 未初始化時)
        if h is None or c is None:
            if x.dim() == 2:
                x, (h_new, c_new) = self.lstm(x.unsqueeze(1))
                x = x.squeeze(1)
            else:
                x, (h_new, c_new) = self.lstm(x)
        else:
            # 【關鍵】skrl 傳來的形狀是 (batch_size, num_layers, hidden_size)
            # PyTorch LSTM 要求 (num_layers, batch_size, hidden_size)
            h = h.transpose(0, 1).contiguous()
            c = c.transpose(0, 1).contiguous()
            
            if x.dim() == 2:
                x, (h_new, c_new) = self.lstm(x.unsqueeze(1), (h, c))
                x = x.squeeze(1)
            else:
                x, (h_new, c_new) = self.lstm(x, (h, c))

        # 【關鍵】傳回 skrl 之前，要把形狀轉回 (batch_size, num_layers, hidden_size)
        h_new = h_new.transpose(0, 1)
        c_new = c_new.transpose(0, 1)

        # # 【核心修改】使用 tanh 限制均值輸出在 [-1, 1] 之間，防止計算 log_prob 時數值爆炸
        # mean_actions = torch.tanh(self.action_layer(x))
        # # 【修改】放寬 log_std 的截斷範圍到 [-20, 2]（skrl 默認值），-3 太小容易導致數值敏感
        # log_std = torch.clamp(self.log_std_unconstrained, -20, 2)

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

        # 1. 特徵提取
        self.encoder = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
        )
        
        # 2. LSTM (與 Policy 保持一致)
        self.lstm = nn.LSTM(input_size=hidden_size, 
                            hidden_size=hidden_size, 
                            num_layers=self.num_layers, 
                            batch_first=True)

        # 3. 輸出一個標量 Value
        self.value_layer = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        # LSTM 初始化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        # Value 層初始化
        nn.init.orthogonal_(self.value_layer.weight, gain=1.0)
        nn.init.constant_(self.value_layer.bias, 0)

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": 1,
                "sizes":[(self.num_layers, self.hidden_size),  
                          (self.num_layers, self.hidden_size)]  
            }
        }

    def compute(self, inputs, role):
        states = inputs["states"]
        rnn_states = inputs.get("rnn_states", {"rnn": [None, None]})["rnn"]
        
        h, c = rnn_states[0], rnn_states[1]

        states = torch.nan_to_num(torch.clamp(states, -10.0, 10.0), 0.0)
        x = self.encoder(states)

        # 處理 None
        if h is None or c is None:
            if x.dim() == 2:
                x, (h_new, c_new) = self.lstm(x.unsqueeze(1))
                x = x.squeeze(1)
            else:
                x, (h_new, c_new) = self.lstm(x)
        else:
            # 形狀轉換
            h = h.transpose(0, 1).contiguous()
            c = c.transpose(0, 1).contiguous()
            
            if x.dim() == 2:
                x, (h_new, c_new) = self.lstm(x.unsqueeze(1), (h, c))
                x = x.squeeze(1)
            else:
                x, (h_new, c_new) = self.lstm(x, (h, c))

        # 轉回 skrl 需要的形狀
        h_new = h_new.transpose(0, 1)
        c_new = c_new.transpose(0, 1)

        value = self.value_layer(x)
        # 對 Value 進行截斷防護
        # value = torch.clamp(value, -100, 100)

        return value, {"rnn":[h_new, c_new]}







