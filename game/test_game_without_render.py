import sys
from utils.fps_calculator import fpsCalculator
from script.game import Game

DEVICE = "cuda:0"

if __name__ == '__main__':

    device = DEVICE

    # render_mode="window"
    render_mode="headless" 
    # render_mode="server"

    player_configs = None
    platform_configs = None
    environment_configs = None
    fps_calculator = fpsCalculator()

    game = Game(render_mode=render_mode, 
                model_obs_type="state_based", # "game_screen", "state_based", "mixed"
                obs_width=400,
                obs_height=225,
                device=device,
                max_episode_step=3000,
                player_configs=player_configs,
                platform_configs=platform_configs,
                environment_configs=environment_configs,
                num_env=1,
                level=8, 
                sub_level=0,
                capture_per_second=None,
                
                requires_grad=False,
            )

    tested_steps = 0
    num_step_for_test = 6000
    
    try:
        while tested_steps < num_step_for_test: 
            if not game.game_over:

                actions = None
                data = game.step_CUDA_Graph(actions)
                # data = game.step_Diff(actions)

                tested_steps += 1
            else:
                game.reset()

            
            if render_mode != "window":
                fps_calculator.update(print_fps_when_update=True)

    except Exception as e:
        print(f"發生異常: {e}")
        print("print traceback: ")
        import traceback
        traceback.print_exc()
    finally:
        # 執行最後的清理（見下文）
        game.physics_manager.cleanup() 
        print("進程已安全退出")
        sys.exit(0)



