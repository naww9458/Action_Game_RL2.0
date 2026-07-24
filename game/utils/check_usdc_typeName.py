from pxr import Usd

# path = "./Action_Game_RL_Assets/assets/final_sphere_combined.usdc"
# path = "./Action_Game_RL_Assets/assets/tang_sword.usdc"
# path = "./Action_Game_RL_Assets/assets/hollow_sphere.usdc"
path = "./Action_Game_RL_Assets/assets/hollow_sphere.usdc"

usd_stage = Usd.Stage.Open(path)
for prim in usd_stage.Traverse():
    print(f"Path: {prim.GetPath()}, Type: {prim.GetTypeName()}")