import warp as wp

@wp.func
def ray_sphere_intersect(ro: wp.vec3, rd: wp.vec3, ce: wp.vec3, r: float) -> float:
    """計算射線與球體的距離"""
    oc = ro - ce
    b = wp.dot(oc, rd)
    c = wp.dot(oc, oc) - r * r
    h = b * b - c
    if h < 0.0:
        return -1.0
    
    t = -b - wp.sqrt(h)
    if t < 0.0:
        return -1.0
    return t

@wp.func
def get_ray_distance_to_blocks(
    chest_pos: wp.vec3,
    ray_dir: wp.vec3,
    index_entities_offset_env: wp.int32, 
    num_entities_each_env: wp.int32, 
    body_q: wp.array(dtype=wp.transform),
    max_ray_dist: float,
    block_radius: float,
) -> float:
    """檢測特定方向的射線距離"""
    closest_dist = max_ray_dist
    
    for j in range(num_entities_each_env):
        block_idx = index_entities_offset_env + j
        block_t = body_q[block_idx] 
        block_pos = wp.transform_get_translation(block_t)
        
        hit_t = ray_sphere_intersect(chest_pos, ray_dir, block_pos, block_radius)
        
        if hit_t > 0.0 and hit_t < closest_dist:
            closest_dist = hit_t
            
    return closest_dist