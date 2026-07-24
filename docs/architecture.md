# 系統架構

本專案主要由三個部分組成：

1. 環境與訓練配置編輯介面
2. 環境初始化與訓練循環
3. 執行期間除錯與環境檢視工具

後續章節將介紹各模組的職責與相互關係，並搭配流程圖說明資料流與控制流程。

<div align="center">
    <img src="images\object_initialization_order.png" style="width: 40%; height: auto;">
    <p><i>物件建立順序</i></p>
    <img src="images\env_test_runtime_architecture.png" style="width: 40%; height: auto;">
    <p><i>環境測試流程</i></p>
    <img src="images\rl_training_flow.png" style="width: 40%; height: auto;">
    <p><i>强化學習訓練流程</i></p>
</div>

---

# 1. 環境與訓練配置編輯介面

本模組基於 **PyQt6** 開發，主要負責建立、修改與管理專案中的各種設定檔，包括環境配置、訓練配置及測試配置。

此模組主要目的是提升開發效率與使用體驗，不直接參與強化學習訓練流程或物理模擬。

主要功能包括：

- 環境建立與編輯
- 訓練配置管理
- 測試環境啟動
- TensorBoard 啟動
- 系統設定

---

# 2. 環境初始化與訓練循環

此模組為整個專案的核心，負責建立環境、初始化物理世界、管理遊戲邏輯、執行強化學習訓練，以及維護所有執行期間的資料。

## Game

Game 為整個模擬流程的入口，負責初始化環境、建立所有管理器，並控制每一幀模擬流程。

主要包含下列子模組：

### Level

負責建立與管理環境中的所有物件，包含角色、平台、實體、技能生成物件等。

### Physics Manager

負責與物理引擎互動，管理物理世界初始化、模擬步進、以及物件同步。

目前支援的求解器：

- Newton XPBD
- Newton VBD
- MuJoCo

### RewardCalculator

負責計算強化學習訓練所需的 Reward 和 Termination。

### GameConfig

負責載入與管理遊戲、環境及訓練相關的靜態設定。

### Ability

技能系統。

負責定義玩家可使用的各種技能，並管理技能初始化、冷卻、執行與相關物件生成。

### ArticulationBody

關節體封裝。

主要用於管理剛體特別是具有關節結構的物件，例如機器人、機械臂等。

### DeformableBody

可變形物體封裝。

主要用於管理 Soft Body、Cloth 等可變形物件。

### Role

角色系統。

負責定義玩家、平臺、實體和技能生成物件在内的不同角色，以及角色控制流程。

---

## Trainer

負責與強化學習框架（目前為 skrl）整合，管理：

- Policy
- Algorithm
- Checkpoint
- TensorBoard
- 訓練流程

---

## WarpEnv

WarpEnv 為專案與強化學習框架之間的橋樑。

負責將 Game 封裝成符合 RL Framework 使用的 Environment Interface，提供 Observation、Action、Reward、Reset 等介面。

---

# 3. 執行期間除錯與環境檢視工具

本模組主要用於開發與除錯，不影響正式訓練流程。

## CustomViewerGL

即時顯示環境狀態。

主要負責：

- 場景渲染
- 相機控制
- 物件選取
- 環境可視化

---

## ObjectInspector

執行期間物件檢查工具。

可在模擬執行期間查看或修改物件資訊，包括：

- Position
- Rotation
- Velocity
- Force
- Joint State
- RL Action
- Commands
- Controls

主要用於環境除錯與模型分析。