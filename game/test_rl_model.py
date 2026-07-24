import sys
import time
import torch
import copy

from skrl.resources.preprocessors.torch import RunningStandardScaler

from script.game_config import GameConfig
# Assuming your policy.py contains Policy, Value (for PPO)

from skrl_script.wrapperSKRL import WarpEnv

DEVICE = "cuda:0"

# =============================================================================
# PPO Configuration Wrapper Function
# =============================================================================
def get_ppo_config(env, device):
    from skrl_script.policy_PPO2_state_based import Policy as Policy_PPO, Value as Value_PPO
    # from skrl_script.policy_PPO3_mixed import Policy as Policy_PPO, Value as Value_PPO

    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory

    models = {
        "policy": Policy_PPO(env.observation_space, env.action_space, device=device),
        "value": Value_PPO(env.observation_space, env.action_space, device=device)
    }
    
    # Memory does not need to be large in test mode
    memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)
    

    return models, memory, PPO

# =============================================================================
# APG Configuration Wrapper Function
# =============================================================================
def get_apg_config(env, device):
    from skrl_script.policy_APG1_state_based import Policy as Policy_APG
    
    from game.skrl_script.algorithm.apg.apg import APG

    models = {
        "policy": Policy_APG(env.observation_space, env.action_space, device=device),
    }
    
    return models, APG



if __name__ == '__main__':
    device = DEVICE
    enable_window = True

    memory = None
    num_envs = 2
    checkpoint_path = "./runs/26-04-03_19-17-41-995097_APG/checkpoints/agent_3160.pt" 

    env = None

    if "PPO" in checkpoint_path:
        print("Switching to PPO evaluation mode")

        from training.loader import TrainingPresetLoader
        loaded = TrainingPresetLoader.load("level4_0_ppo_state_based")
        model_cfg = loaded.model_cfg
        train_cfg = loaded.train_cfg
        env = WarpEnv(num_envs=num_envs, device=device, model_cfg=model_cfg, train_cfg=train_cfg, is_training=False, step_mode="CUDA_Graph", enable_window=enable_window)
        models, memory, agent_class = get_ppo_config(env, device)

    elif "APG" in checkpoint_path:
        raise NotImplementedError(
            "Legacy APG config removed. Use rl_launcher.py eval with a saved run checkpoint."
        )
    
    # Usually set num_envs smaller in test mode for better observation

    # 4. Instantiate Agent
    agent = agent_class(models=models, 
                        memory=memory, 
                        cfg=model_cfg.cfg, 
                        observation_space=env.observation_space, 
                        action_space=env.action_space, 
                        device=device)

    # Load weights
    try:
        agent.load(checkpoint_path)
        print(f"Successfully loaded weights: {checkpoint_path}")
    except Exception as e:
        raise ValueError(f"Failed to load, please check if the path and model structure match: {e}")

    agent.set_running_mode("eval") 

    tested_steps = 0
    num_step_for_test = 30000
    
    obs, _ = env.reset()
    target_fps = GameConfig.FPS_ACTION
    frame_duration = 1.0 / target_fps

    try:
        while tested_steps < num_step_for_test: 
            start_time = time.perf_counter()

            with torch.no_grad():
                # skrl's act method returns (actions, log_prob, outputs)
                actions = agent.act(obs, timestep=tested_steps, timesteps=num_step_for_test)[0]

            obs, reward, terminated, truncated, info = env.step(actions)

            # # Optional: print actions of the first environment for debugging
            # if tested_steps % 100 == 0:
            # print(f"Step: {tested_steps} | Action Example: {actions[0].cpu().numpy()}")
            # print(f"Step total reward: {reward[0].item()}")

            tested_steps += 1
            
            if enable_window:
                env.render()
                # Control frame rate to prevent CPU/GPU from running too fast to see the window
                elapsed_time = time.perf_counter() - start_time
                if elapsed_time < frame_duration:
                    time.sleep(frame_duration - elapsed_time)

            env.reset()
            
        print("Test result: ", env.game.total_reward_for_test)
        for key, data in env.game.total_reward_for_test.items():
            if data["total_reward"] == 0 or data["reset_times"] == 0:
                Average_reward = data["total_reward"]
            else:
                Average_reward = data["total_reward"] / data["reset_times"]

            print(f"Player {key} Average reward: {Average_reward}")


    except Exception as e:
        print(f"An exception occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment closed, test finished")
        sys.exit(0)