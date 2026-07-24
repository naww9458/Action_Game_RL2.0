# from rl_launcher import main

# if __name__ == "__main__":
#     raise SystemExit(main(["train", "--preset", "level5_0_ppo_state_based"]))


import torch

# from skrl_script.trainer_PPO_APG import Trainer
from skrl_script.trainer_PPO import Trainer
# from skrl_script.trainer_APG import Trainer

trainer = Trainer(num_envs=4096, is_training=True, level=5, sub_level=0, obs_type="state_based", enable_window=False)
trainer.train_custom()







