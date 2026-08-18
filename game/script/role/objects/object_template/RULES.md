# object_template 規則

本檔適用於 `object_template/` 下**每一個物件資料夾**。環境 YAML、USD/MJCF、Ability、Reward、`policy_bundle` 不讀、不合併、不覆寫各物件資料夾內部的檔案佈局。缺哪個檔、檔在根目錄還是 `models/<version>/`，由該物件自己的 `register.py` 決定。

## 資料夾

每個物件一個資料夾，內含 `template.yaml`（必要）。可選 `register.py`：

- `register()` — 註冊關節載入器、policy 索引等
- `prepare_object(object_cfg)` — 物理載入前，把本物件的完整表寫進 object config（沒有則跳過）
- `setup(environment)` — 物理建好後接 runtime（本物件不在場景裡則 no-op）

## 根目錄 vs `models/<version>/`

**根目錄**：這個物件所有變體都必然相同的東西（USD/MJCF 資產路徑、能力列表、馬達硬體、本物件共用程式）。

**`models/<version>/`**：滿足任一條就放進來：

1. 該版本缺少此資訊/組件就無法運行
2. 可能有兩個版本完全一樣，但**不是所有版本必然一樣**

有控制策略的物件用 `policy_versions.yaml` 把版本 id 指到 `models/<version>/control_policy.yaml`。沒有版本差異的物件不必建 `models/`。

## 完整表，禁止合併

版本或根目錄裡的表（碰撞、task、command 含義等）是**完整來源**。`prepare_object` 整份寫上 object，不與環境 YAML 拼鍵。場景只寫 `pattern`、`control_policy_version` 等指標，不要把整表抄進環境 YAML。USD/MJCF 只套用 object 上已有的欄位，不回查 template 資料夾。

讀檔順序由該物件自己實作：先 `models/<version>/`，沒有再讀該物件根目錄同名檔。外面不要做 overlay。搬進 `models/<version>/` 的檔案**沿用原檔名**，只改路徑。

## 有 policy 版本的物件：`models/<version>/` 標準檔名

| 檔案                   | 用途                                                                                             |
|------------------------|-------------------------------------------------------------------------------------------------|
| `control_policy.yaml`  | 觀測/動作/command、checkpoint、`history_len`；`observation.obs_actor` / `observation.obs_critic` |
| `control_configs.yaml` | 初始姿態、感測器、command 採樣等 task 專屬表（沿用原檔名，只換資料夾）                               |
| `obs_actor.py`         | 必須 export `create_obs_actor`（policy / actor 觀測與 command runtime）                          |
| `obs_critic.py`        | 可選；export `create_obs_critic`（非對稱 critic 觀測；無則對稱 critic）                            |
| `collision.yaml`       | 可選；該版本完整碰撞表（由 `prepare_object` 整份寫上 object）                                      |
| `*.pt`                 | 權重 checkpoint                                                                                 |

`control_policy.yaml` 的 `observation` 區塊用 **`obs_actor`**、**`obs_critic`** 作為註冊 id（不是 `provider` / `critic_provider`）。對應模組檔名固定為 **`obs_actor.py`**、**`obs_critic.py`**。

沒有 policy 版本的物件（例如純車輛、靶標）仍用根目錄的 `control_configs.yaml` 等既有慣例；不必強制 `obs_actor` / `obs_critic` 檔名。

## 新增版本

1. 複製 `models/<既有版本>/` 為 `models/<新 id>/` 並改該副本
2. 在該物件的 `policy_versions.yaml` 加一筆
3. 訓練 preset / 環境 YAML 只改 `control_policy_version`
