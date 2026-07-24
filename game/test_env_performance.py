
import sys
import torch
import time

from skrl_script.wrapperSKRL import WarpEnv
from skrl_script.get_config_and_model import get_config_and_model

from utils.fps_calculator import fpsCalculator
from utils.tensorboard_recorder import TensorboardRecorder


DEVICE = "cuda:0"

def test_env(algorithm: str, level: int, sub_level: int, obs_type: str, num_envs: int, step_mode: str, enable_window: bool, num_step_for_test: int, recorder, device):
    model_cfg, train_cfg, Policy_cls, Value_cls = get_config_and_model(algorithm=algorithm, level=level, sub_level=sub_level, obs_type=obs_type)

    fps_calculator = fpsCalculator()

    # "cuda_graph", "differentiation"
    env = WarpEnv(num_envs=num_envs, device=device, model_cfg=model_cfg, train_cfg=train_cfg, level_config_path=None, is_training=True, step_mode=step_mode, enable_window=enable_window)

    low = env.action_space.low[0] # TODO Hard code
    high = env.action_space.high[0] # TODO Hard code
    action_shape = env.action_space.shape[0]

    actions = torch.empty((env.game.players.num_rl_players, action_shape), device, dtype=torch.float32) # TODO Hard code


    tested_steps = 0
    while tested_steps < num_step_for_test: 

        actions.uniform_(low, high)
        env.step(actions)

        tested_steps += 1

        if env.game.render_mode != "window":
            fps_calculator.update()

            if fps_calculator.updated_step:
                fps_recorder.record_performance(step=tested_steps, num_envs=num_envs, fps=fps_calculator.fps_current)


        if enable_window:
            env.render()

        env.reset()

    # Perform final cleanup
    env.close()
    recorder.close()




if __name__ == '__main__':
    device = DEVICE

    # "state_based", "mixed"
    obs_type="state_based"

    # tested_num_env = [1, 32, 64, 128, 256, 512, 1024]
    tested_num_env = 256

    # "cuda_graph", "differentiation"
    step_mode="cuda_graph"
    num_step_for_test=100000

    fps_recorder=TensorboardRecorder(is_use_timestamp=False, project_name="Action_Game_RL", task_name=f"FPS_testing/{step_mode}/{obs_type}", log_dir="fps_testing")

    try:
        fps_recorder.change_run(f"num_env_{tested_num_env}")
        test_env(algorithm="PPO",
                level=4,
                sub_level=0,
                obs_type=obs_type,
                num_envs=tested_num_env,
                step_mode=step_mode,
                enable_window=False,
                num_step_for_test=num_step_for_test,
                recorder=fps_recorder,
                device=device,
                )

    except Exception as e:
        print(f"An exception occurred: {e}")
        print("print traceback: ")
        import traceback
        traceback.print_exc()
    finally:
        print("Process exited safely")
        sys.exit(0)






