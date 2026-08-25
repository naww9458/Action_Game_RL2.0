# Bug List

1. (✅️fixed) `g1_and_wheeled_armor`（舊 Level 8.1）中無法通過 object_inspector 控制 Unitree G1 機器人
2. (✅️fixed) object_inspector 的 RL Action 頁面顯示混亂
3. (✅️fixed) `g1_and_wheeled_armor`（舊 Level 8.1）中，wheeled_armored_vehicle_basic 的碰撞箱有問題
4. (✅️fixed) 當通過 UI 實驗中心啓動訓練後最小化窗口可能中斷訓練
5. (✅️fixed) 重置環境后應解除車體和炮塔的連接
6. (✅️fixed) 無法通過 object_inspector 控制 tools (無法控制炮塔轉向，炮管升降)
7. (✅️fixed) 應該在 object template 中添加 Ability
8. (✅️fixed) `vbd_clothing`（舊 Level 6.0）中 reset 環境錯誤
9. 環境編輯界面 3D 預覽區域無法顯示 USD 模型
10. 核心 ``MjcfModel``（``game/script/role/objects/mjcf.py``）把 ``file_name`` / ``file_path_or_source`` 默認成 G1 的 ``g1.xml`` 路徑，違反隔離原則（核心 loader 不應綁定具體物件）。``UsdModel`` 已用空字符串默認。尚未決定如何與 ``object_template`` 裏的 MJCF 資源聲明兼容（Pydantic 字段默認 vs 模板 YAML 覆蓋）。
11. Unitree G1 平地行走訓練測試問題：
    穩定性不足，雖然能根據指令完成行走/轉向等動作但抖動幅度較大