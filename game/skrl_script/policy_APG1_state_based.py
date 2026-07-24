import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model

# =============================================================================
# Policy 模型 (Actor)
# =============================================================================
class Policy(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=256, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self) # 確定性輸出

        self.hidden_size = hidden_size
        self.num_layers = 1

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

        # 初始權重調小，防止第一步力矩過大導致物理崩潰
        nn.init.orthogonal_(self.action_layer.weight, gain=0.001) 
        self.to(device) 

        print(f"Policy model initialized on device: {device}")
        print("self.action_layer.weight device check: ", self.action_layer.weight.device)


    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": 1,
                "sizes":[(self.num_layers, self.hidden_size), (self.num_layers, self.hidden_size)]
            }
        }

    def compute(self, inputs, role):
        states = inputs["states"]
        # 安全地獲取 rnn_states
        rnn_info = inputs.get("rnn_states", {"rnn": [None, None]})
        rnn_states = rnn_info.get("rnn", [None, None])
        h, c = rnn_states[0], rnn_states[1]

        states = torch.clamp(states, -10.0, 10.0)
        x = self.encoder(states)

        # LSTM 需要 (Batch, Seq, Hidden)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if h is None:
            # 第一步：讓 LSTM 自動初始化 h, c (均為 0)
            x, (h_new, c_new) = self.lstm(x)
        else:
            # 後續步：將外部儲存的 (Batch, L, H) 轉回 PyTorch 要求的 (L, B, H)
            # 這裡檢查一下，如果已經是 (L, B, H) 就不轉置 (防禦性編程)
            if h.shape[0] == states.shape[0] and h.shape[1] == self.num_layers:
                h = h.transpose(0, 1).contiguous()
                c = c.transpose(0, 1).contiguous()
            
            x, (h_new, c_new) = self.lstm(x, (h, c))

        x = x.squeeze(1)

        # 將回傳的 (L, B, H) 轉置為 (Batch, L, H) 供外部儲存
        h_out = h_new.transpose(0, 1).contiguous()
        c_out = c_new.transpose(0, 1).contiguous()

        raw_actions = self.action_layer(x)
        actions = torch.tanh(raw_actions) 

        # 這裡請確保與 trainer_APG.py 的解包數量一致
        # 如果 trainer 是寫 actions_pt, rnn_states = policy.compute(...)
        # 則這裡回傳 (actions, None, {"rnn": [h_out, c_out]})
        return actions, None, {"rnn": [h_out, c_out]}


