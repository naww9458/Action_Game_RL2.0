
1. 環境和訓練配置編輯以及啟動介面
    主頁：
        一切的開始，你可以在這裏選擇或者新增環境，又或者去編輯/訓練/測試頁面。

    系統設置：
        你可以在這裏修改字體大小/語言/項目目錄等（目前僅限此界面）

    環境編輯頁面：
        你可以在主頁選擇環境後到這裏進行編輯。
        左邊選擇或者新增物件
        中間修改物件屬性
        右邊有 3D 預覽區域但目前只能顯示長方體和球體（未能顯示 USD 等模型）

    實驗中心：
        你可以在主頁選擇環境後到這裏進行編輯訓練配置/評估/訓練模型，或是啓動 Tensorboard 來查看訓練日志

    測試頁面：
        你可以在主頁選擇環境後到這裏進行環境測試，參數介紹：
        1. 渲染模式：是否要再測試環境的時候啓用顯示窗口
            可選項：window/headless
        
        2. 觀察類型：用於測試觀察空間的輸出
            可選項：state_based

        3. 最大步數：環境一回合的最大模擬步數
            可選項：整數數字 1 到 100000

        4. 環境數量：同時啓動多少并行運行的環境
            可選項：整數數字 1 到 10000

        5. 需要梯度：是否需要梯度，啓用自動微分的必選項。
            注意：在本專案中自動微分暫時和 CUDA Graph 處於衝突狀態，如果暫時不需要自動微分功能建議關閉此選項來使用 CUDA Graph 加速，可大幅提升幀率。

        6. 鎖定幀率：是否將環境模擬速度和現實時間同步。


2. 環境初始化以及訓練循環的運行模組
## 求解器
    xpbd：本專案中唯一完整强化學習訓練的環境所使用的求解器，在本專案中目前只能進行剛體模擬

    vbd：本專案中唯一能模擬軟體（soft body）已經布料的求解器，但剛體可能因爲質量/密度過大而陷入地面

    mujoco：本專案中唯一能模擬 Unitree G1 在内的關節體的求解器

## 環境物件角色
### 玩家：
    主要分爲 Human / Bot / RL Agent 三種角色，是環境中唯一能自主行動的角色分類
        1. Human：由人類玩家親自操控的角色
        2. Bot：由固定硬編碼脚本控制的角色
        3. RL Agent：由通過强化學習訓練出來的人工智能模型控制的角色

    技能：
        1. articulation_body_control
            設計目的是用於通用關節體控制，包括機器人機器狗等。

        2. articulation_body_control_rl_assisted
            設計目的是爲了方便人類控制多關節模型，包括機器人機器狗等。

        3. jump
            設計目的是爲了控制極度簡化的角色物件（僅爲一個球體），更多是用於訓練模型高維決策而不是身體控制。
            控制方式是直接給予一個基於世界坐標系的向上的力，讓物件飛起來

        4. move_topdown_viewing_angle
            設計目的是爲了控制極度簡化的角色物件（僅爲一個球體），更多是用於訓練模型高維決策而不是身體控制。
            控制方式是直接給予一個基於玩家角色坐標系的前後左右的力，讓物件動起來

        5. turning_topdown_viewing_angle
            設計目的是爲了控制極度簡化的角色物件（僅爲一個球體），更多是用於訓練模型高維決策而不是身體控制。
            Human 的控制方式是直接根據鼠標移動來修改物件面對的方向。
            Bot 的控制方式是直接把面對方向鎖定在目標上
            RL Agent 的控制方式是讓模型輸出由算法轉換成扭矩來進行叠加，保證可微分的同時讓物件轉起來

        6. shoot
            設計目的是爲了控制極度簡化的角色物件（僅爲一個球體），更多是用於訓練模型高維決策而不是身體控制。
            使用技能後查找一個處於休眠狀態的 “子彈”，將其傳送到玩家物件的位置后根據玩家物件面對的方向直接高速飛過去。

        7. tool_attachment
            設計目的是讓 **玩家（載具）** 在運行時掛載/拆卸可分離工具（如炮塔），并在掛載後用 **相機視角** 驅動炮塔瞄準。
            掛在 **玩家** 的能力列表上（不是 Tool 角色本身）；關卡需在 `tool_configs` 中配置工具實例，且關卡需有 `mount_joint_registry` 才會生效。
            Human 操作：
                - 跟隨載具角色且靠近未掛載炮塔時，畫面中央顯示掛載提示
                - 按 **U** 切換掛載/拆卸（再次按 U 可從載具上拆下）
                - 掛載後，鼠標/相機 yaw/pitch 通過 PD 扭矩驅動炮塔水平偏航與俯仰（MuJoCo 關卡使用 weld 約束驅動偏航）
            Bot / RL：`rl_action` 與 `bot_action` 尚未實現；attach 離散動作空間已在配置中預留。

    角色範例：
        1. Unitree G1：宇树科技推出的其中一款機器人，本專案部署了 mjlab 平地行走任務訓練出來的模型，可通過 articulation_body_control_rl_assisted 來控制其行走

        2. 輪式裝甲車（`wheeled_armored_vehicle_basic`）：本人自製六輪低面數模型，采用差速轉向，通過 `articulation_body_control` 控制。可作為 **工具宿主（host）**，車體上預留 `hull_mount_anchor` 掛載點。

        3. 110 毫米火炮炮塔（`turret_110mm`）：見下方 **Tool 角色** 小節；與輪式裝甲車模塊化組合，未掛載時為場景中獨立浮置關節體，掛載後跟隨車體並可由 `tool_attachment` 瞄準。

        4. 史萊姆：沒有專屬模型文件，采用全粒子加流體模擬，以一個主粒子爲核心通過内聚力吸引下屬粒子，并通過數個範圍内自主移動的分粒子實現觸手/形變能力 (未實現)

### 工具（Tool）：
    新增角色類型，用於 **可從載具上拆裝的模塊化裝備**（炮塔、武器等）。在關卡 YAML 的 `tool_configs` 中按 **list** 配置（每個工具一條），與 `player_configs` / `platform_configs` 并列。

    與玩家的關係：
        - **Player**：宿主（host），負責駕駛與掛載操作；需配置 `Tool_attachment` 能力
        - **Tool**：被掛載物，關卡加載時通常以 **FREE 關節浮置** 於場景中，靠近宿主後由 `Tool_attachment` 啟用 mount joint / weld 約束完成掛載

    統一 ID 規則：
        - 所有角色物件以 **物件 ID** 作為唯一識別：list 容器角色（player/platform/tool）用 `id` 欄位；
          dict 容器角色（entity/ability_generated_object）的 dict key 是 **物件子角色**
          （`object_sub_role`），代表這類會批量生成、位置由計算得出的物件類別，
          而非唯一的物件 ID（因此不與欄位 `id` 同步）
        - `name` 只是可重複的顯示名稱，不再作為識別符

    掛載相關配置（可在關卡 `tool_configs` 或 `object_template/<pattern>/template.yaml` 中定義）：
        - `mount_anchor_name` / `host_anchor_name`：工具與車體 USD 錨點 prim 名
        - `host_body_prim_suffix` / `tool_base_body_prim_suffix`：錨點所屬剛體後綴
        - `host_player_index`：綁定哪個玩家索引為宿主（可省略以自動嘗試）
        - `host_player_id`：以 player_configs 的物件 ID（`id` 欄位）明確指定宿主（優先於 `host_player_index`）。
          指定後工具的初始坐標/旋轉/速度/角速度可省略——它會在宿主生成點出生，掛載時再吸附到宿主身上。
        - `internal_joint_names` / `pitch_joint_name`：工具內部關節；`pitch_joint_name` 指定相機瞄準俯仰 DOF
        - `aim_control`：瞄準 PD 增益、死區、扭矩上限等
        - `start_attached`：設為 `true` 時，環境啟動（及每次 reset 後）工具自動掛載到宿主上，不需按 U。
          取消此選項後，工具的初始姿態欄位會重新出現（因為工具仍需以浮置姿態生成）

    物件範例：110 毫米火炮炮塔（`turret_110mm`）
        - 資產：`turret_110mm.usdc`，模板路徑 `game/script/role/objects/object_template/turret_110mm/`
        - 錨點：工具側 `turret_base_anchor`，車體側 `hull_mount_anchor`（在輪式裝甲車模板中定義）
        - 俯仰關節：USD 葉子名 `RevoluteJoint`（須在配置中與 `pitch_joint_name` 一致）

    運行時模塊（代碼概要）：
        - `setup_tool_mount_joints`：關卡初始化時解析錨點、預建禁用狀態的 mount joint，寫入 `MountJointRegistry`
        - `Tool_attachment.human_control_interface`：近距離檢測、U 鍵掛載/拆卸、掛載後 `apply_attached_aim`
        - 無 `tool_configs` 的關卡不創建 registry，無額外開銷

### 平臺：
    主要用於地板/墻壁等不應該被移動的靜態環境物品

### 實體：
    主要用於箱子/磚塊等可被被動移動的環境物品

### 技能生成物件：
    主要用於和技能聯動，因爲物理引擎特性建議在初始化時一次性生成所有物件，因此這個角色的作用是當玩家尚未動用技能是處於休眠狀態，在必要時激活。
    代表物件：子彈，聯動技能：射擊。


3. 環境運行時顯示環境情況的介面（包含顯示物件位置/速度/姿態以及調整對應值的功能）

    屬性分類規則（環境編輯頁面與運行時顯示介面一致，依物件欄位性質分為三大類）：
        - 角色屬性：對該角色**所有物件**（無論 object template 為何）都通用的欄位，
          且不進入 object template。包括 `type` / `id` / `name` / `color` / 初始坐標 /
          初始旋轉 / 初始速度 / 初始角速度 / `controller` / `team_id` / `health`；
          dict 容器角色的「物件子角色」（`object_sub_role`）等同於 list 角色的 `id`，
          也是角色屬性；以及 Tool 專屬的「連接宿主」（`host_player_id`）、
          「啟動時自動掛載」（`start_attached`）等每物件自身的設定。
        - 物件屬性：與特定物件/模板相關、會進入 object template 並可透過
          "Add Object Template" 批量套用預設參數的欄位。該下拉框位於環境編輯頁面
          的「物件屬性」區段頂部，與模型屬性緊貼、和角色屬性明顯區分。包括
          `abilities`（能力）、
          掛載介面（`mount_anchor_name` / `host_anchor_name` / `host_body_prim_suffix` /
          `tool_base_body_prim_suffix` / `mount_joint_type` 等）、`possess_offset`、
          `separation` 等。不同物件類型（例如標準球體 vs Unitree G1）這些欄位可能不同。
        - 模型屬性：加入物理引擎時輸入的參數，作為「物件屬性」最底部的次級區域。
          例如剛體球體的半徑/質量/摩擦力/彈性，或 Unitree G1 的模型路徑等
          （即 `object` 欄位內的內容）。

    環境顯示介面：
        通過鼠標點擊環境顯示介面中對應物件來選擇

    物件屬性顯示/調整界面：
        上方：選擇物件以及環境

        中下方：分爲五個不同的頁面
        1. Bodies
            這裏顯示了你選擇的物件目前能顯示和修改的所有物理屬性（位置/旋轉/綫速度/角速度/綫性力/扭矩）
            注意：修改這裏的屬性可能導致不可預料的後果，如果重置環境無效便需要徹底重啓環境。

        2. Joints
            當物件有對應關節的時候才會解鎖這一頁，
            這裏顯示了你選擇的關節能顯示以及修改的物理屬性（角度/角速度/扭矩）

        3. RL Action
            此頁面主要用於測試模型輸出到物件反應是否符合預期

        4. Commands
            當模型支持 Commands 輸入（比如控制 Unitree G1 行走的模型）則可以通過此頁面發出指令

        5. Controls
            記錄控制按鍵控制/修改環境重力加速度

<div align="center">
    <img src="images\object_inspector_Controls.png">
    <p><i>物件屬性顯示/調整界面的 Controls 頁面</i></p>
</div>
