import sys
import torch
import time

from skrl_script.wrapperSKRL import WarpEnv
from training.loader import TrainingPresetLoader
from utils.fps_calculator import fpsCalculator

DEVICE = "cuda:0"

if __name__ == '__main__':

    device = DEVICE

    loaded = TrainingPresetLoader.load("level5_0_ppo_state_based")
    model_cfg = loaded.model_cfg
    train_cfg = loaded.train_cfg
    enable_window=True

    fps_calculator = fpsCalculator()

    # "cuda_graph", "differentiation"
    env = WarpEnv(num_envs=2, device=device, model_cfg=model_cfg, train_cfg=train_cfg, level_config_path=None, is_training=True, step_mode="cuda_graph", enable_window=enable_window)

    tested_steps = 0
    num_step_for_test = 6000

    render_fps_counter = 0
    render_fps_timer = time.time() 

    # actions1 = torch.Tensor([[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       # 左腿
    #                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       # 右腿
    #                            0.0, 0.0, 0.0,                      # 腰部

    #                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 左臂（肩膀到手肘）

    #                             # 左手（有部分手指要設置成 -1 才會卷起）
    #                            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  

    #                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 右臂（肩膀到手肘）

    #                             # 右手（有部分手指要設置成 1 才會卷起）
    #                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0   
    #                          ],

    #                          [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       # 左腿
    #                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       # 右腿
    #                            0.0, 0.0, 0.0,                      # 腰部
    #                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 左臂（肩膀到手肘）
    #                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # 左手（只有拇指會動，第 3，4，5，6 個值不知道做什麽的）
    #                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 右臂（肩膀到手肘）
    #                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0   # 右手（只有食指會動）
    #                          ],
    #                          ], device='cpu').to(device=device)


    actions1 = torch.Tensor([[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       # 左腿
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       # 右腿
                               0.0, 0.0, 0.0,                      # 腰部

                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 左臂（肩膀到手肘）

                                # 左手（有部分手指要設置成 -1 才會卷起）
                            #    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  

                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 右臂（肩膀到手肘）

                                # 右手（有部分手指要設置成 1 才會卷起）
                            #    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0   
                             ],
                             ], device='cpu').to(device=device)

    try:
        while tested_steps < num_step_for_test: 

            
            actions = actions1
                                        
            env.step(actions)

            tested_steps += 1

            if env.game.render_mode != "window":
                fps_calculator.update()
                if fps_calculator.updated_step:
                    print("FPS: ", fps_calculator.fps_current)
            
                
            if enable_window:
                env.render()
            
            env.reset()

    except Exception as e:
        print(f"An exception occurred: {e}")
        print("print traceback: ")
        import traceback
        traceback.print_exc()
    finally:
        # Perform final cleanup
        env.close()
        print("Process exited safely")
        sys.exit(0)



