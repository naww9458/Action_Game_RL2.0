# This file contains code adapted from:
# https://github.com/mujocolab/mjlab
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the Apache License 2.0.

"""Generic mjlab joint-position action pipeline for MuJoCo Warp deployment.

Usage
-----
>>> from script.mjlab_components.act import MjlabWarpActionApplier
>>> applier = MjlabWarpActionApplier(
...     device="cuda:0",
...     scale=scale_array,
...     default_joint_pos=offset_array,
...     action_dim=len(scale_array),
... )
>>> applier.launch_to_targets(raw_actions_wp, encoder_bias_wp, targets_wp)
"""

try:
    from .warp_kernels import MjlabWarpActionApplier

    __all__ = ["MjlabWarpActionApplier"]
except ImportError:
    __all__ = []
