import sys
from utils.fps_calculator import fpsCalculator
from script.game import Game

if __name__ == '__main__':

    # render_mode="window"
    render_mode="headless" 
    # render_mode="server"

    player_configs = None
    platform_configs = None
    environment_configs = None
    fps_calculator = fpsCalculator()

    game = Game(render_mode=render_mode, 
                model_obs_type="state_based", # "game_screen", "state_based", "mixed"
                obs_width=80,
                obs_height=45,
                max_episode_step=300000,
                player_configs=player_configs,
                platform_configs=platform_configs,
                environment_configs=environment_configs,
                num_env=80,
                level=5, 
                sub_level=1,
                capture_per_second=None,

                requires_grad=True,
            )

    tested_steps = 0
    num_step_for_test = 6000
    
    try:
        while tested_steps < num_step_for_test: 
            if not game.game_over:

                actions = None
                data = game.step_Diff(actions)

                tested_steps += 1
            else:
                game.reset()

            
            if render_mode != "window":
                fps_calculator.update()

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



