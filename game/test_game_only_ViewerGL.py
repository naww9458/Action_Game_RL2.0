import multiprocessing as mp

from queue import Queue
from script.game import Game
from script.simulate.physics_manager import PhysicsManager
from script.custom_viewergl import CustomViewerGL

# import numpy as np
# # 設置門檻為無限大，這樣就不會出現省略號
# np.set_printoptions(threshold=np.inf)

DEVICE = "cuda:0"

if __name__ == '__main__':

    device = DEVICE

    event_is_window_setup_ready = mp.Event()
    event_is_game_logic_keymapping_setup_ready = mp.Event()
    # shm = shared_memory.SharedMemory(create=True, size=data_size)

    physics_manager_state_queue = Queue(maxsize=1)
    human_input_queue = Queue(maxsize=1)

    render_mode="window"
    # render_mode="headless"
    # render_mode="server"

    player_configs = None
    platform_configs = None
    environment_configs = None
    
    viewerGL=CustomViewerGL(event_is_window_setup_ready=event_is_window_setup_ready, 
                            human_input_queue=human_input_queue,
                            follow_body_index=0, 
                           )
    physics_manager = PhysicsManager(device=device, viewerGL=viewerGL)

    game = Game(render_mode=render_mode, 
                model_obs_type="state_based", # "game_screen", "state_based", "mixed"
                obs_width=400,
                obs_height=225,
                device=device,
                physics_manager=physics_manager,
                max_episode_step=3000,
                player_configs=player_configs,
                platform_configs=platform_configs,
                environment_configs=environment_configs,
                num_env=10,
                level=4, 
                sub_level=1,
                capture_per_second=None,
                
                requires_grad=True,
            )
    game.run_game_human(event_is_window_setup_ready=event_is_window_setup_ready, 
                        event_is_game_logic_keymapping_setup_ready=event_is_game_logic_keymapping_setup_ready, 
                        physics_manager_state_queue=physics_manager_state_queue, 
                        human_input_queue=human_input_queue,
                        is_lock_fps=True
                       )


