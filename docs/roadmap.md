
   1. (✅️finished) 嘗試合并保存模型訓練的相關 config 文件

   2. (✅️finished) 為 Balancing_Ball_RL_V6 中的 Train 和 Optuna 創建單獨的 Python 文件 

   3. (✅️finished) 嘗試合并 play_with_model.py 的評估和 Train 中的評估

   4. (✅️finished) 改進 RL\levels 中的 config.py 文件，其中 render_mode="human" 適用於在本地電腦測試模型，render_mode="headless" 適用於在 Google Colab 訓練模型

   5. (✅️finished) 改進獎勵函數，應將按照距離給予獎勵換成判斷是否正在接近來給予獎勵或者懲罰

   6. (✅️finished) 當前技能冷卻是 time.time() 控制，當游戲在不同環境下以不同速度運行，會導致技能在非預期情況下冷卻完畢，太早或者太慢都會產生錯誤數據。
      應該修改爲由 step 數控制，比如當一個技能冷卻時間是一秒，那麽實際冷卻就是 1 * FPS

   7. 改進加載不同 Level 的 config 的方式

   8. (✅️finished) 解決 Level 3 模型訓練局部最優的問題

   9. (✅️finished) 添加 Level 4 用於對抗式訓練 (Level 4 將被設計成在一個沒有障礙物的正方形場地中進行 1v1 俯視角射擊游戲)

   10. (✅️finished) 把執行動作數據改成字典，技能名為 Key，把按鍵輸入轉化成 action value 移動到技能類類中，借此優化人類玩家的輸入邏輯

   11. (✅️finished) 添加鍵盤按鍵映射，技能的 json 配置文件中儲存鍵盤按鈕名稱，key_mapping 把案件名稱映射到 pygame 對應的按鍵編號

   12. (✅️finished) 加入鼠標按鍵映射

   13. (✅️finished) 加入 GameConfig 類，改進初始化以及解耦

   14. (✅️finished) 添加俯視角移動以及射擊技能

   15. (✅️finished) (已經完成但又發現對於單機訓練似乎沒必要) 修改游戲架構為 Client-Server 來應用對抗式訓練

   16. (✅️finished) 評估使用 dict 的地方是否需要換成 NamedTuple
         結果：暫時不需要

   17. (✅️finished) 渲染切換到 ModernGL，然後和 Pygame 進行性能測試對比。(單環境，渲染模式：human，圖像大小：1000 x 1000)
         測試結果  【Pygame FPS：+-1700】
                  【ModernGL FPS：+-3400 (需要關閉文字顯示，不然 FPS 鎖定在 390 左右)】

   18. (✅️finished) 完成對 Level 4 的訓練 commit ID: 24165eb982ded4c3ddfe67769e54b4cd74402f36
         測試結果：
            總測試回合數: 5
            總測試 step 數: 18140
            平均每回合 step 數: 3628.0
            獎勵:
               玩家 RL_player0 總獎勵為: 1662.8585468601682, 平均獎勵: 332.57170937203364
         
   19. (✅️finished) 把物理引擎從 Pymunk 切換到 NVIDIA Newton-warp, 項目也將更名爲 Action_Game_RL

   20. (✅️finished) 把窗口顯示和人類輸入處理從 pygame 切換成 moderngl-window
         原因：pygame 的窗口坐標 0,0 在左上角，但是 ViewerViser 0,0 是在場地中間，萬一刷新了 ViewerViser 的網站就可能要花很長時間去找

   21. (✅️finished) 把 moderngl-window 的攝像機固定到對應角色中心，攝像機朝向和該角色朝向綁定
            (從 moderngl-window 換成了 newton 内建的 viewerGL)

   22. (✅️finished) 完成 Newton 的碰撞判定

   23. (✅️finished) 根據 Newton 的碰撞判定修復好獎勵函數

   24. (✅️finished) moderngl-window FPS 過低，需要檢查優化 (60 FPS)

   25. (✅️finished) 嘗試使用 CUDA Graphs 來優化 PhysicsManager 的 simulate()，目前 solver.step 占據率超過 80% 運行時間
            測試結果(FPS)：使用 CUDA Graphs：MAX = 2572, MIN = 1586
                          不使用 CUDA Graphs：MAX = 1028, MIN = 673

   26. (✅️finished) Bot 的主要動作由生成從 CPU 移動到 wp.kernel 中，這是目前我的物理引擎環境在同時模擬 200, 100, 80, 70 個并行環境下的 FPS (4080S GPU)
            200 個環境：477
            100 個環境：1640
            80 個環境：2470
            70 個環境：3125
          200 個環境的時候 FPS 只有 477，目前懷疑是因爲部分 wp.kernel 中存在過多 for loop 導致

   27. (✅️finished) 徹底重構底層的初始化，構建邏輯，從 OOP 轉換成「OOP 負責管理，DOP 負責計算」, 順便加入多環境啓動

   28. (✅️finished) 加入視綫檢測防止數據被其他環境的物件污染

   29. (✅️finished) 包裝成環境

   30. (✅️finished) 目前訓練的模型在 40 并行環境，學習率為 1e-4 訓練 10000 步并且鎖定俯仰角能獲得最大命中率，問題是只能逆時針旋轉以及旋轉過程不順滑，如果目標跑右邊就只能往左邊轉一圈回來而不是往右轉
       接下來先增加 Custom Viewer GL 窗口顯示的信息，增加可控制角色範圍用於 debug
       詳細信息：優化了性能，添加 FPS 顯示以及使用顔色區分人類玩家 RL 玩家以及 Bot 玩家。實際上并沒有增加可控制角色範圍

   31. 嘗試優化 SKRL 訓練過程，目前 CPU 占用過高 GPU 使用率過低

   32. (✅️finished) 目前認爲訓練出有正常瞄準能力的模型必須要 1e-4 學習率加上 2000 到 5000 迭代的原因是重置環境后角色位置固定，嘗試讓角色位置和方向隨機化

   33. (✅️finished) 實現 “可破壞/倒塌” 障礙物，可想象為多個正方體或者長方體堆叠的墻壁

   34. 實現閃避或者速度獎勵，具體實現方式跟情況決定，用於訓練模型躲避子彈

   35. (✅️finished) 完善 sub level，sub level 1 以及以後的關卡是基於 sub level 0 的設置上額外添加東西，因此除了加載 sub level 1 的設置外還會加載 sub level 0 的設置
   
   36. (✅️finished) 通過設置 Collision Group / Mask 優化碰撞計算，這是目前我的物理引擎環境在同時模擬 300, 80 個并行環境下的 FPS (4080S GPU)
            相關參數：rollouts = 512, learning_epochs = 8, mini_batches = 2
            數據來源：運行 train.py 后觀察終端機右下角數據，例如：[00:27<02:32, 557.19it/s]
            注意1：下方測試數據的最低值大多爲剛更新完模型的時候，因爲更新模型會暫停訓練，因此剛更新完繼續訓練的時候 FPS 最低
            注意2：環境數量設置為 300 的時候運行 test_env.py FPS 能穩定保持 2800 以上 GPU 使用率為 99%, 而運行 train.py 的時候 GPU 使用率大多為 80% 以下, skrl 或許存在很多 CPU 操作

            優化前：
            300 個環境：50-90 FPS
            80 個環境：200-400 FPS

            優化後：
            300 個環境：300 - 600 FPS
            80 個環境：400 - 700 FPS

   37. (✅️finished) 完善以及優化 nvdiffrast 渲染，正式移除 moderngl
            注意：test_game_without_render.py 脚本不包含 MLP，CNN 計算 以及模型更新，因此實際訓練速度會在降低。
            測試脚本：test_game_without_render.py  測試結果：672 - 687 FPS (53,760+ Samples per second)
            相關參數：render_mode = "headless"
                     level = 4 
                     sub_level = 0 
                     sub_steps = 4
                     model_obs_type = "mixed"
                     obs_width = 80
                     obs_height = 45
                     num_env = 80

            測試脚本：test_game_without_render.py  測試結果：230 - 231 FPS (69,000+ Samples per second)
            相關參數：render_mode = "headless"
                     level = 4 
                     sub_level = 0 
                     sub_steps = 4
                     model_obs_type = "mixed"
                     obs_width = 300
                     obs_height = 45
                     num_env = 80

            

   38. (✅️finished) 通過 nvdiffrast 渲染圖象用於混合觀察空間訓練

   39. (✅️finished) 對雖有訓練相關的函數進行可微改造，嘗試使用自動微分

   40. (✅️finished) 加入 APG 訓練

   41. (✅️finished) 完善 APG 訓練（tensorboard 監控，權重參數 checkpoints 保存）

   42. (✅️finished) 加入 PPO + APG 訓練 

   43. 添加在每次訓練結束後保存該次訓練的所有相關設置

   44. (✅️finished) 把 Newton 物理引擎版本從 0.1.3 升級為 0.2.2 並應用新功能 (實際上更新到 1.2.0 版本)

   45. (✅️finished) 添加軟體，流體以及 USD 等其他各種模型

   46. (✅️finished) 添加方便用於編輯環境配置文件的 UI

   47. 改進 UI 界面，將 Level 以及相關名稱並統一換成 “環境” 或 “environment”，并將部分環境改成基礎範例環境

   47. 重構初始化相關的代碼，讓系統能統一處理剛體，關節體（Unitree G1 模型）以及軟體 （暫時是不包含流體）
            a. (✅️finished) 環境初始化
            b. (✅️finished) 環境重置
            c. (需要測試) 技能系統
            d. (需要測試) 獎勵函數

   48. 訓練强化學習模型操控 Unitree G1 行走 (暫時擱置，) 
            a. (✅️finished) 加載 mjlab 平地行走任務訓練的模型控制 Unitree G1 機器人行走
            b. 

   49. 實現更多種類角色（人類模型，動物模型，以及通過算法聚合流體的史萊姆）
            a. (✅️finished) Unitree G1 (無手部控制，代表機器人角色分類)
            b. (✅️finished) 輪式裝甲車 (代表模塊化改裝設計，局内切換模塊，基礎功能完成)

   50. (✅️finished) 多求解器協同模擬（MuJoCo 用於機器人，VBD/XPBD 用於剛體/軟體，MPM 用於流體）
            目前情況：MPM 求解器模擬速度過慢 ( 不確定具體原因但 Nvidia 官方 Example 幀率也不高 )，因此暫時不做相關應用等待後續發展
                     軟體球 ( VBD ) 和 Unitree G1 ( MuJoCo ) 能在同一環境中模擬但還沒想到實際可行的應用方案

   51. (✅️finished) 移除基於 nvdiffrast 的 Renderer（暫時）。
       原因：為保持本專案 Apache-2.0 授權的一致性，暫時移除依賴 NVIDIA Source Code License (1-Way Commercial) 的實作。
       未來將評估以其他相容授權的 Renderer 或自行實作替代方案。

   52. (✅️finished) 將 Newton 版本從 1.2 升級到 1.4 

   53. 修正誤區：動態增減剛體并非不現實，動態加索引才不現實，可嘗試製作破壞效果

   54. 

   55. 

   56. 

   57. 

   58. 


