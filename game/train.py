# from rl_launcher import main

# if __name__ == "__main__":
#     raise SystemExit(main(["train", "--preset", "flat_walk_skrl_ppo_state_based"]))


import torch

# from rl_framework.skrl_script.trainer_PPO_APG import Trainer
from rl_framework.skrl_script.trainer_PPO import Trainer
# from rl_framework.skrl_script.trainer_APG import Trainer

trainer = Trainer(num_envs=4096, is_training=True, preset_id="flat_walk_skrl_ppo_state_based", enable_window=False)
trainer.train_custom()







