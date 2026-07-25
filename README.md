# Action_Game_RL

## 🚧 專案狀態

目前狀態：開發中（Early Development）

本專案仍在快速迭代，架構、API、資料格式及部分功能都可能在未來版本中發生變動，目前不保證向後相容。


## 一句話介紹

基於 Newton Physics 和 NVIDIA Warp 建構的 GPU 加速物理模擬與強化學習沙箱。


## 簡介

最終目標：打造 AI 版 TABS（Totally Accurate Battle Simulator）

目前定位：
    本專案定位於娛樂性物理模擬與強化學習實驗平台。並非以 Sim-to-Real 或取代 IsaacLab、mjlab 為目標，而是希望提供一個更容易修改、更具互動性，並適合實驗各種 AI 想法的 Sandbox。
    加入 mjlab 的部分只是因爲這是現成已經能拿來用的方案。對大部分人來説，這可以成爲一個互動性很高的物理模擬器，對於對强化學習有興趣的人來説，這能成爲一個學習工具。

注意：專案中的環境是基於 Newton 物理引擎，建議使用 Nvidia GPU 運行。

（Newton 建立於 NVIDIA Warp 之上，並由 Disney Research、Google DeepMind 與 NVIDIA 等團隊共同開發。）

<div align="center">
    <img src="docs\images\UI_main.png" style="width: 40%; height: auto;">
    <p><i>主界面</i></p>
    <img src="docs\images\Viewer_and_object_inspector.png" style="width: 40%; height: auto;">
    <p><i>同時運行 100 個并行環境</i></p>
</div>

## 快速開始：
1. 安裝 uv Python Package Manager。
2. 安裝 CUDA 12.8 版本
3. 運行指令 uv sync (如果你的 CUDA 版本不是 12.8，建議在運行指令前將 pyproject.toml 中 pytorch 對應的 CUDA 版本修改為你的版本)
4. 運行 game\run_UI.py

啓動 Tensorboard 查看訓練日志（可通過指令，也可通過 UI 快速啓動）：
tensorboard --logdir [dir]


## 文件：

- [ROADMAP](/docs/roadmap.md)
- [Benchmark] 即將加入
- [系統架構](/docs/architecture.md)
- [已完成功能](/docs/features.md)

---

## 🙏 致謝：

本專案在開發過程中參考、使用或改編了部分開源專案的設計與程式碼，包括：

### 程式碼與框架

* **[Newton Physics](https://github.com/newton-physics/newton)** — GPU 物理模擬引擎 (Apache-2.0)。
* **[mjlab](https://github.com/mujocolab/mjlab)** — GPU 機器人學習框架 (Apache-2.0)。
* **[skrl](https://github.com/Toni-SM/skrl)** — 強化學習框架 (MIT)。
* **[NVIDIA Warp](https://github.com/NVIDIA/warp)** — 可微分運算與 GPU 計算框架 (Apache-2.0)。

### 模型與資源（Assets）

- **[Action_Game_RL_Assets](https://github.com/naww9458/Action_Game_RL_Assets.git)** — 提供本專案部分 USD 模型、機器人模型、布料及其他模擬資源 (Apache-2.0)。

### 本專案改編之程式碼

部分原始碼參考或改編自以下專案：

- skrl
- mjlab

感謝各專案作者與所有貢獻者的開源分享。若本專案中有引用、改編或使用了您的開源成果而未能正確標示來源，歡迎提出 Issue 或與我聯絡，我會在確認後盡快補充相關致謝或標示。

## ❤️ 支持

如果你想支持我繼續開發這個開源專案:

- Patreon: https://patreon.com/naww9458

非常感謝你的支持!
