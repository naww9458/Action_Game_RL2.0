# Wheeled Armored Vehicle Basic — Joint Mapping

Asset: `Action_Game_RL_Assets/assets/wheeled_armored_vehicle_basic.usdc`

- Bodies: 13
- Joints: 13
- Total DOFs: 18
- Left wheel DOFs: 3
- Right wheel DOFs: 3

## Bodies

- `0`: `/root/vehicle_body/vehicle_body`
- `1`: `/root/vehicle_body/Susp_L1/Susp_L1`
- `2`: `/root/vehicle_body/Susp_L1/Wheels_L1/Wheels_L1`
- `3`: `/root/vehicle_body/Susp_L2/Susp_L2`
- `4`: `/root/vehicle_body/Susp_L2/Wheels_L2/Wheels_L2`
- `5`: `/root/vehicle_body/Susp_L3/Susp_L3`
- `6`: `/root/vehicle_body/Susp_L3/Wheels_L3/Wheels_L3`
- `7`: `/root/vehicle_body/Susp_R1/Susp_R1`
- `8`: `/root/vehicle_body/Susp_R1/Wheels_R1/Wheels_R1`
- `9`: `/root/vehicle_body/Susp_R2/Susp_R2`
- `10`: `/root/vehicle_body/Susp_R2/Wheels_R2/Wheels_R2`
- `11`: `/root/vehicle_body/Susp_R3/Susp_R3`
- `12`: `/root/vehicle_body/Susp_R3/Wheels_R3/Wheels_R3`

## Joints

- `joint_1` — FREE, qd=[0:6], dofs=6
- `/root/vehicle_body/Susp_L1/Susp_L1/VB_Susp_L1_RevoluteJoint` — REVOLUTE, qd=[6:7], dofs=1
- `/root/vehicle_body/Susp_L1/Wheels_L1/Wheels_L1/D6Joint` — D6, qd=[7:8], dofs=1
- `/root/vehicle_body/Susp_L2/Susp_L2/VB_Susp_L2_RevoluteJoint` — REVOLUTE, qd=[8:9], dofs=1
- `/root/vehicle_body/Susp_L2/Wheels_L2/Wheels_L2/D6Joint` — D6, qd=[9:10], dofs=1
- `/root/vehicle_body/Susp_L3/Susp_L3/VB_Susp_L3_RevoluteJoint` — REVOLUTE, qd=[10:11], dofs=1
- `/root/vehicle_body/Susp_L3/Wheels_L3/Wheels_L3/D6Joint` — D6, qd=[11:12], dofs=1
- `/root/vehicle_body/Susp_R1/Susp_R1/VB_Susp_R1_RevoluteJoint` — REVOLUTE, qd=[12:13], dofs=1
- `/root/vehicle_body/Susp_R1/Wheels_R1/Wheels_R1/D6Joint` — D6, qd=[13:14], dofs=1
- `/root/vehicle_body/Susp_R2/Susp_R2/VB_Susp_R2_RevoluteJoint` — REVOLUTE, qd=[14:15], dofs=1
- `/root/vehicle_body/Susp_R2/Wheels_R2/Wheels_R2/D6Joint` — D6, qd=[15:16], dofs=1
- `/root/vehicle_body/Susp_R3/Susp_R3/VB_Susp_R3_RevoluteJoint` — REVOLUTE, qd=[16:17], dofs=1
- `/root/vehicle_body/Susp_R3/Wheels_R3/Wheels_R3/D6Joint` — D6, qd=[17:18], dofs=1

## Per-DOF (non-FREE)

| DOF | Joint | Type | Local DOF | Role | Target Mode |
|-----|-------|------|-----------|------|-------------|
| 6 | `/root/vehicle_body/Susp_L1/Susp_L1/VB_Susp_L1_RevoluteJoint` | REVOLUTE | 0 | suspension | POSITION |
| 7 | `/root/vehicle_body/Susp_L1/Wheels_L1/Wheels_L1/D6Joint` | D6 | 0 | wheel_spin_primary | VELOCITY |
| 8 | `/root/vehicle_body/Susp_L2/Susp_L2/VB_Susp_L2_RevoluteJoint` | REVOLUTE | 0 | suspension | POSITION |
| 9 | `/root/vehicle_body/Susp_L2/Wheels_L2/Wheels_L2/D6Joint` | D6 | 0 | wheel_spin_primary | VELOCITY |
| 10 | `/root/vehicle_body/Susp_L3/Susp_L3/VB_Susp_L3_RevoluteJoint` | REVOLUTE | 0 | suspension | POSITION |
| 11 | `/root/vehicle_body/Susp_L3/Wheels_L3/Wheels_L3/D6Joint` | D6 | 0 | wheel_spin_primary | VELOCITY |
| 12 | `/root/vehicle_body/Susp_R1/Susp_R1/VB_Susp_R1_RevoluteJoint` | REVOLUTE | 0 | suspension | POSITION |
| 13 | `/root/vehicle_body/Susp_R1/Wheels_R1/Wheels_R1/D6Joint` | D6 | 0 | wheel_spin_primary | VELOCITY |
| 14 | `/root/vehicle_body/Susp_R2/Susp_R2/VB_Susp_R2_RevoluteJoint` | REVOLUTE | 0 | suspension | POSITION |
| 15 | `/root/vehicle_body/Susp_R2/Wheels_R2/Wheels_R2/D6Joint` | D6 | 0 | wheel_spin_primary | VELOCITY |
| 16 | `/root/vehicle_body/Susp_R3/Susp_R3/VB_Susp_R3_RevoluteJoint` | REVOLUTE | 0 | suspension | POSITION |
| 17 | `/root/vehicle_body/Susp_R3/Wheels_R3/Wheels_R3/D6Joint` | D6 | 0 | wheel_spin_primary | VELOCITY |
