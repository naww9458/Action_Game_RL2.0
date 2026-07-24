from contextlib import contextmanager

import warp as wp


@contextmanager
def suspend_tape():
    """Temporarily disable the active Warp tape.

    Discrete resets (teleports / random respawns) must not be recorded when APG
    wraps ``step_Diff`` in ``wp.Tape``; their adjoints are invalid and can fault.
    """
    runtime = wp._src.context.runtime
    tape = runtime.tape
    runtime.tape = None
    try:
        yield
    finally:
        runtime.tape = tape


@wp.func
def sigmoid(x: float):
    """
    可微的 Sigmoid 函數
    """
    return 1.0 / (1.0 + wp.exp(-x))

@wp.func
def safe_length(v: wp.vec3):
    """
    可微的安全長度計算，避免在原點產生 0/0 的 NaN 梯度
    """
    return wp.sqrt(wp.max(wp.length_sq(v), 1e-6))

@wp.func
def safe_length_vec2(v: wp.vec2):
    """
    可微的安全二維長度計算
    """
    return wp.sqrt(wp.max(wp.length_sq(v), 1e-6))

@wp.func
def safe_normalize(v: wp.vec3) -> wp.vec3:
    l = wp.length(v)
    # 防止除零，同時保持可微
    l = wp.max(l, 1e-6)          
    return v / l

@wp.func
def safe_atan2(y: float, x: float) -> float:
    """
    永遠不會產生 NaN 的 atan2，適合所有能力 kernel 使用
    """
    return wp.atan2(y, x + 1e-8)


@wp.func
def calculate_ballistic_aim_dir( 
    diff: wp.vec3, 
    bullet_speed: float, 
    gravity: float
) -> wp.vec3:
    """
    給定相對位置、子彈速度與重力，計算出最佳的世界座標系瞄準方向 (World Aim Direction)
    """
    # TODO 可能没算上速度衰减 damping
    # 1. 計算水平與垂直距離
    horizontal_diff = wp.vec3(diff[0], diff[1], 0.0)
    x = safe_length(horizontal_diff) 
    y = diff[2]
    
    v = bullet_speed
    g = gravity
    
    # 2. 計算判別式
    v2 = v * v
    v4 = v2 * v2
    discriminant = v4 - g * (g * x * x + 2.0 * y * v2)

    # 3. 求解理想物理俯仰角 (Ballistic Pitch)
    root = wp.sqrt(wp.max(1e-6, discriminant))
    tan_theta = (v2 - root) / (g * x)
    ideal_pitch_ballistic = wp.atan(tan_theta)

    # 4. 直射俯仰角 (Direct Pitch)
    ideal_pitch_direct = wp.atan2(y, x)

    # 5. 平滑混合 (避免超出射程時無解)
    reachable_weight = sigmoid(discriminant * 100.0) 
    ideal_pitch = wp.lerp(ideal_pitch_direct, ideal_pitch_ballistic, reachable_weight)

    # 6. 計算最終的三維方向向量 (World Space)
    dir_x = diff[0] / x
    dir_y = diff[1] / x
    cos_p = wp.cos(ideal_pitch)
    sin_p = wp.sin(ideal_pitch)
    
    aim_dir_world = wp.vec3(dir_x * cos_p, dir_y * cos_p, sin_p)
    
    return aim_dir_world

@wp.func
def calculate_ballistic_aim_dir_move( 
    diff: wp.vec3, 
    target_vel: wp.vec3, 
    bullet_speed: float, 
    gravity: float # 這裡傳入 -9.81 或 9.81 都可以，函數內會自動處理
) -> wp.vec3:
    """
    給定相對位置、相對速度、子彈速度與重力，計算出最佳的世界座標系瞄準方向，包含提前量預判。
    """
    v = bullet_speed
    
    # 【關鍵修正】：彈道公式推導本身已內建重力向下，因此 g 必須是正數(純量)
    g = wp.abs(gravity) 
    
    # 初始化：先做一個最粗略的飛行時間猜測 (直線距離 / 子彈速度)
    t = wp.length(diff) / wp.max(v, 1e-6)
    
    future_diff = diff
    ideal_pitch = 0.0
    x = 0.0
    
    for _ in range(3):
        # 1. 預測目標在時間 t 之後的位置
        future_diff = diff + target_vel * t
        
        # 2. 計算新的水平與垂直距離
        horizontal_diff = wp.vec3(future_diff[0], future_diff[1], 0.0)
        x = safe_length(horizontal_diff) 
        y = future_diff[2]
        
        # 3. 計算判別式
        v2 = v * v
        v4 = v2 * v2
        discriminant = v4 - g * (g * x * x + 2.0 * y * v2)

        # 4. 求解理想物理俯仰角 (Ballistic Pitch)
        root = wp.sqrt(wp.max(1e-6, discriminant))
        tan_theta = (v2 - root) / wp.max(1e-6, (g * x))
        ideal_pitch_ballistic = wp.atan(tan_theta)

        # 5. 直射俯仰角 (Direct Pitch)
        ideal_pitch_direct = wp.atan2(y, x)

        # 6. 平滑混合 (避免超出射程時無解)
        reachable_weight = sigmoid(discriminant * 100.0) 
        ideal_pitch = wp.lerp(ideal_pitch_direct, ideal_pitch_ballistic, reachable_weight)
        
        # 7. 更新飛行時間 t（wp.where：避免分支打斷 Warp AD）
        horizontal_speed = v * wp.cos(ideal_pitch)
        t_horiz = x / wp.max(horizontal_speed, 1e-4)
        t_fallback = wp.length(future_diff) / wp.max(v, 1e-6)
        t = wp.where(horizontal_speed > 1e-4, t_horiz, t_fallback)

    # 8. 計算最終的三維方向向量 (World Space)
    dir_x = future_diff[0] / wp.max(1e-6, x)
    dir_y = future_diff[1] / wp.max(1e-6, x)
    
    cos_p = wp.cos(ideal_pitch)
    sin_p = wp.sin(ideal_pitch)
    
    # 注意：這裡算出的 aim_dir_world，如果 sin_p 是正數，代表向量朝向 +Z (往天空看)
    aim_dir_world = wp.vec3(dir_x * cos_p, dir_y * cos_p, -sin_p)
    
    return aim_dir_world


