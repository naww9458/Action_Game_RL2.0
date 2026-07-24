# Wheeled Armored Vehicle Basic — Joint Mapping

Asset: `Action_Game_RL_Assets/assets/wheeled_armored_vehicle_basic.usdc`

- Bodies: 13
- Joints: 13
- Total DOFs: 21
- Left wheel DOFs: 3
- Right wheel DOFs: 6

## Body masses (config override)

| Part | Mass (kg) |
|------|-----------|
| Chassis (`vehicle_body`) | 14000 |
| Suspension arm (`Susp_*`) | 200 |
| Wheel (`Wheels_*`) | 300 |

Suspension PD gains are computed from chassis weight in `control_configs.yaml` → `suspension` (default ke ≈ 412000 N·m/rad at g=9.81). Do not use the old hardcoded ke=800.

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
- `/root/vehicle_body/Susp_R1/Wheels_R1/Wheels_R1/D6Joint` — D6, qd=[13:15], dofs=2
- `/root/vehicle_body/Susp_R2/Susp_R2/VB_Susp_R2_RevoluteJoint` — REVOLUTE, qd=[15:16], dofs=1
- `/root/vehicle_body/Susp_R2/Wheels_R2/Wheels_R2/D6Joint` — D6, qd=[16:18], dofs=2
- `/root/vehicle_body/Susp_R3/Susp_R3/VB_Susp_R3_RevoluteJoint` — REVOLUTE, qd=[18:19], dofs=1
- `/root/vehicle_body/Susp_R3/Wheels_R3/Wheels_R3/D6Joint` — D6, qd=[19:21], dofs=2

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
| 13 | `/root/vehicle_body/Susp_R1/Wheels_R1/Wheels_R1/D6Joint` | D6 | 0 | wheel_spin_extra (rotX) | POSITION (locked) |
| 14 | `/root/vehicle_body/Susp_R1/Wheels_R1/Wheels_R1/D6Joint` | D6 | 1 | wheel_spin_primary (rotY) | VELOCITY |
| 15 | `/root/vehicle_body/Susp_R2/Susp_R2/VB_Susp_R2_RevoluteJoint` | REVOLUTE | 0 | suspension | POSITION |
| 16 | `/root/vehicle_body/Susp_R2/Wheels_R2/Wheels_R2/D6Joint` | D6 | 0 | wheel_spin_extra (rotX) | POSITION (locked) |
| 17 | `/root/vehicle_body/Susp_R2/Wheels_R2/Wheels_R2/D6Joint` | D6 | 1 | wheel_spin_primary (rotY) | VELOCITY |
| 18 | `/root/vehicle_body/Susp_R3/Susp_R3/VB_Susp_R3_RevoluteJoint` | REVOLUTE | 0 | suspension | POSITION |
| 19 | `/root/vehicle_body/Susp_R3/Wheels_R3/Wheels_R3/D6Joint` | D6 | 0 | wheel_spin_extra (rotX lock) | POSITION (locked) |
| 20 | `/root/vehicle_body/Susp_R3/Wheels_R3/Wheels_R3/D6Joint` | D6 | 1 | wheel_spin_primary (rotY) | VELOCITY |

All six wheels are symmetric (2 DOF each: locked axis + spin axis).
