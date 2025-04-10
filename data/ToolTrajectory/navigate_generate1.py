import os
import numpy as np
import quaternion
from typing import List
import habitat_sim
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_angle_axis
from habitat_sim.utils import viz_utils as vut
import magnum as mn
from tqdm import tqdm
import cv2
import math

def path_to_actions(path, start_rotation, end_rotation, rotation_step=10, move_step=0.1):
    actions = []
    if len(path) < 2:
        return actions
    
    current_rot = start_rotation
    current_pos = np.array(path[0], dtype=np.float64)
    
    for i in range(1, len(path)):
        target_pos = np.array(path[i], dtype=np.float64)
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            continue

        direction_normalized = direction / distance
        
        # 计算当前前向方向（Habitat前向是-z）
        forward = np.quaternion(0, 0, 0, -1)  # 纯四元数表示-z方向
        rotated_forward = current_rot * forward * current_rot.conj()
        current_forward = np.array([rotated_forward.x, rotated_forward.y, rotated_forward.z])
        current_forward[1] = 0  # 投影到水平面
        current_forward = current_forward / np.linalg.norm(current_forward)
        
        direction_normalized[1] = 0
        direction_normalized = direction_normalized / np.linalg.norm(direction_normalized)
        
        # 计算旋转角度和方向
        cross = np.cross(current_forward, direction_normalized)
        dot = np.dot(current_forward, direction_normalized)
        angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        if cross[1] < 0:
            angle_deg = -angle_deg
        
        # 分解旋转步骤
        if abs(angle_deg) > 1e-3:
            rotation_direction = "turn_left" if angle_deg > 0 else "turn_right"
            num_steps = int(np.ceil(abs(angle_deg) / rotation_step))
            for _ in range(num_steps):
                actions.append((rotation_direction,))
        
        # 分解移动步骤
        if distance > 1e-3:
            num_steps = int(np.ceil(distance / move_step))
            for _ in range(num_steps):
                actions.append(("move_forward",))
            
            current_pos = target_pos
    
    return actions

def execute_actions(sim, actions):
    observations = []
    agent = sim.agents[0]
    
    for action in tqdm(actions):
        agent.act(action[0])
        obs = sim.get_sensor_observations()
        rgb = obs["color"][..., :3][..., ::-1]  # BGR转RGB
        observations.append({"color": rgb})
    
    final_state = agent.get_state()
    print("Final position:", final_state.position)
    print("Final rotation:", quat_to_angle_axis(final_state.rotation))
    return observations

def main():
    scene_file = "data/HM3D/00876-mv2HUxq3B53/mv2HUxq3B53.basis.glb"
    navmesh_file = "data/HM3D/00876-mv2HUxq3B53/mv2HUxq3B53.basis.navmesh"
    output_video = "output.mp4"

    # 修正四元数顺序转换
    start_rotation_np = quaternion.from_float_array([0.06694663545449525, 0.0, 0.997756557483499, 0.0])
    start_rotation_mn = mn.Quaternion(
        mn.Vector3(start_rotation_np.x, start_rotation_np.y, start_rotation_np.z),
        start_rotation_np.w
    )
    
    end_rotation_np = quaternion.from_float_array([0.015153784750062174, 0.0, 0.9998851748114626, 0.0])
    end_rotation_mn = mn.Quaternion(
        mn.Vector3(end_rotation_np.x, end_rotation_np.y, end_rotation_np.z),
        end_rotation_np.w
    )
    
    # 模拟器配置
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_file
    backend_cfg.enable_physics = False
    
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [480, 640]
    sensor_spec.hfov = 120.0
    
    agent_cfg = habitat_sim.agent.AgentConfiguration(
        sensor_specifications=[sensor_spec],
        action_space={
            "move_forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.1)),
            "turn_left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=10)),
            "turn_right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=10))
        }
    )
    
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    sim.pathfinder.load_nav_mesh(navmesh_file)
    
    # 计算路径
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array([3.34482479095459, 0.050354525446891785, 9.988510131835938])
    path.requested_end = np.array([-1.5373001098632812, 0.050354525446891785, 16.74457359313965])
    
    if not sim.pathfinder.find_path(path):
        print("Path not found")
        return
    
    actions = path_to_actions(path.points, start_rotation_np, end_rotation_np)
    print(f"Generated {len(actions)} actions")
    
    # 初始化代理
    agent = sim.initialize_agent(0)
    initial_state = habitat_sim.AgentState()
    initial_state.position = np.array([3.34482479095459, 0.050354525446891785, 9.988510131835938])
    initial_state.rotation = habitat_sim.utils.common.quat_to_coeffs(start_rotation_np)
    
    agent.set_state(initial_state)
    
    # 执行动作
    observations = execute_actions(sim, actions)
    
    # 生成视频
    vut.make_video(
        observations=observations,
        primary_obs="color",
        primary_obs_type="color",
        video_file=output_video,
        fps=24,
        open_vid=False
    )
    
    sim.close()

if __name__ == "__main__":
    main()