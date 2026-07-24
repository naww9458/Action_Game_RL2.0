import time
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

class TensorboardRecorder:
    def __init__(self, is_use_timestamp: bool, project_name="RL_Project", task_name="Benchmark", log_dir="runs"):
        """
        初始化記錄器
        :param project_name: 項目名稱 (例如: "Newton_Physics")
        :param task_name: 任務名稱 (例如: "PPO_with_Graph")
        :param log_dir: 存放日誌的根目錄
        """
        # 生成具體的實驗路徑: runs/2026-04-28_18-00_Newton_Physics_PPO_with_Graph
        timestamp = ""
        if is_use_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M") + "_"
        self.base_log_dir = os.path.join(log_dir, f"{timestamp}{project_name}_{task_name}")
        self.writer = None
        
        print(f"[Recorder] Tensorboard log directory: {self.base_log_dir}")

    def change_run(self, sub_run_name: str):
        """
        切換子實驗（例如不同的 num_envs）
        :param sub_run_name: 子路徑名稱，例如 "env_64", "env_128"
        """
        # 如果當前有正在運行的 writer，先關閉它（確保數據寫入硬盤）
        if self.writer is not None:
            self.writer.close()
        
        final_path = os.path.join(self.base_log_dir, sub_run_name)
        self.writer = SummaryWriter(log_dir=final_path)
        print(f"[Recorder] Switched to sub-run: {sub_run_name}")

    def record_scalars(self, step, metrics_dict, prefix=""):
        """
        通用記錄方法：記錄自定義字典中的所有數值
        :param step: 當前全局步數 (Global Step)
        :param metrics_dict: 包含數據的字典, 例如 {"FPS": 2800, "Reward": 1.5}
        :param prefix: 標籤前綴, 例如 "Performance/"
        """
        for key, value in metrics_dict.items():
            tag = f"{prefix}{key}" if prefix else key
            self.writer.add_scalar(tag, value, step)

    def record_performance(self, step, num_envs, fps):
        """
        自動計算並記錄性能指標 (FPS, SPS, Timestamp)
        :param step: 當前步數
        :param num_envs: 環境數量
        :param interval_steps: 距離上次記錄過了多少步
        :return: 計算出的 FPS，方便同時在終端打印
        """
        
        # SPS (Samples Per Second)
        sps = num_envs * fps
        
        # 封裝成字典進行記錄
        perf_metrics = {
            "FPS": fps,
            "SPS": sps,
        }
        
        self.record_scalars(step, perf_metrics, prefix="Performance/")

    def close(self):
        """關閉 Writer"""
        self.writer.close()
        print("[Recorder] Recorder closed.")