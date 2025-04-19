# Author: zml
# 该脚本用于在Habitat-sim中生成路径并保存为视频
# 并将路径点转换为动作序列

import os
import numpy as np
import quaternion
from typing import List
import habitat_sim
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_angle_axis
from habitat_sim.utils import viz_utils as vut
import magnum as mn
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import cv2
import math
from show_map import display_map
from src.utils.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)

# 将路径转换为前进、左转、右转动作
def path_to_actions(path, start_rotation, end_rotation, rotation_step=10, move_step=0.1):
    """
    将路径点转换为动作序列
    
    参数:
        path: 路径点列表，每个点是3D坐标 (x,y,z)
        start_rotation: 起始旋转 (np.quaternion类型)
        end_rotation: 结束旋转 (np.quaternion类型)
        rotation_step: 每次旋转的角度 (度)
        move_step: 每次移动的距离

    返回:
        动作列表，每个动作是 ("move_forward", distance) 或 ("turn_left"/"turn_right", angle)
    """
    actions = []
    if len(path) < 2:
        return actions
    
    current_rot = start_rotation
    current_pos = np.array(path[0], dtype=np.float64)
    
    for i in range(1, len(path)):
        target_pos = np.array(path[i], dtype=np.float64)
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:  # 忽略极小距离
            continue

        direction = direction / distance
        
        # 计算目标方向相对于当前旋转的方向
        # 默认前向是-z轴 (habitat坐标系)
        forward = np.quaternion(0, 0, 0, -1)  # 纯四元数表示-z方向
        
        # 使用四元数旋转向量
        rotated_forward = current_rot * forward * current_rot.conj()
        current_forward = np.array([rotated_forward.x, rotated_forward.y, rotated_forward.z])
        
        # 投影到水平面 (忽略y轴)
        current_forward[1] = 0
        direction[1] = 0
        
        # 归一化
        if np.linalg.norm(current_forward) < 1e-6 or np.linalg.norm(direction) < 1e-6:
            continue

        current_forward = current_forward / np.linalg.norm(current_forward)
        direction = direction / np.linalg.norm(direction)

        # 计算旋转角度 (带符号)
        cross = np.cross(current_forward, direction)
        dot = np.dot(current_forward, direction)

        # 计算旋转角度 (度)
        angle_deg = np.degrees(np.arctan2(np.linalg.norm(cross), dot))

        # 确定旋转方向 (使用叉积的y分量符号)
        if cross[1] < 0:
            angle_deg = -angle_deg

        # 分解旋转为多个小步
        if abs(angle_deg) > 1e-3:  # 忽略极小角度
            num_rot_steps = int(np.ceil(abs(angle_deg) / rotation_step))
            actual_step = angle_deg / num_rot_steps

            for _ in range(num_rot_steps):
                if actual_step > 0:
                    actions.append(("turn_right", min(rotation_step, abs(actual_step))))
                else:
                    actions.append(("turn_left", min(rotation_step, abs(actual_step))))
                
                # 更新当前旋转 (使用四元数乘法)
                delta_rot = np.quaternion(np.cos(np.radians(-actual_step)/2), 
                                        0, np.sin(np.radians(-actual_step)/2), 0)
                current_rot = current_rot * delta_rot
        
        # 分解移动为多个小步
        if distance > 1e-3:  # 忽略极小距离
            num_move_steps = int(np.ceil(distance / move_step))
            actual_step = distance / num_move_steps
            
            for _ in range(num_move_steps):
                actions.append(("move_forward", min(move_step, actual_step)))
            
            current_pos = target_pos
    
    return actions
    


# 执行动作并记录视频
def execute_actions(sim, actions):
    observations: List[np.ndarray] = []
    agent = sim.agents[0]
    
    for action, amount in tqdm(actions):
        # 更新动作参数
        agent.agent_config.action_space[action].actuation.amount = amount
        # 执行动作
        agent.act(action)
        
        # 获取观察并写入视频
        obs = sim.get_sensor_observations()
        observations.append({"color": obs["color"]})

    print(agent.get_state().position)

    return observations


def EulerAndQuaternionTransform(intput_data):
    """
        四元素与欧拉角互换
    """
    data_len = len(intput_data)
    angle_is_not_rad = False

    if data_len == 3:
        r = 0
        p = 0
        y = 0
        if angle_is_not_rad: # 180 ->pi
            r = math.radians(intput_data[0]) 
            p = math.radians(intput_data[1])
            y = math.radians(intput_data[2])
        else:
            r = intput_data[0] 
            p = intput_data[1]
            y = intput_data[2]
 
        sinp = math.sin(p/2)
        siny = math.sin(y/2)
        sinr = math.sin(r/2)
 
        cosp = math.cos(p/2)
        cosy = math.cos(y/2)
        cosr = math.cos(r/2)
 
        w = cosr*cosp*cosy + sinr*sinp*siny
        x = sinr*cosp*cosy - cosr*sinp*siny
        y = cosr*sinp*cosy + sinr*cosp*siny
        z = cosr*cosp*siny - sinr*sinp*cosy
        return [x,y,z,w]
 
    elif data_len == 4:
 
        w = intput_data[0] 
        x = intput_data[1]
        y = intput_data[2]
        z = intput_data[3]
 
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - z * x))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
 
        if angle_is_not_rad : # pi -> 180
            r = math.degrees(r)
            p = math.degrees(p)
            y = math.degrees(y)
        return [r,p,y]


def save_map_with_path(sim, path_points, start_rotation, end_rotation, meters_per_pixel=0.1):
    height = sim.pathfinder.get_bounds()[0][1]
    sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    # 世界坐标系的点path_points转到地图坐标系
    path_points_map = []
    for point in path_points:
        x = int((point[0] - sim.pathfinder.get_bounds()[0][0]) / meters_per_pixel)
        y = int((point[2] - sim.pathfinder.get_bounds()[0][2]) / meters_per_pixel)
        path_points_map.append([x, y])
    display_map(sim_topdown_map, key_points=path_points_map, output_path="top_down_map.png")


def save_navigation_video(sim, agent, path_points, start_position, start_rotation, end_position, end_rotation, output_video="navigation.mp4"):
    agent_state = habitat_sim.AgentState()
    actions = path_to_actions(path_points, start_rotation, end_rotation)

    # 设置起始位置和旋转
    agent_state.position = start_position
    agent_state.rotation = start_rotation
    agent.set_state(agent_state)
    obs = sim.get_sensor_observations()
    cv2.imwrite("start_rgb.jpg", obs["color"][..., :3])

    observations = execute_actions(sim, actions)

    if len(observations) < 2:
        print("Not enough frames captured for video!")
        return

    # 创建视频录制器
    vut.make_video(
        observations=observations,
        primary_obs="color",
        primary_obs_type="color",
        video_file=output_video,
        fps=24,
        open_vid=False
    )

def save_obs_on_path(sim, agent, path_points, start_rotation, end_rotation):
    agent_state = habitat_sim.AgentState()
    for i in range(len(path_points)):
        if i == len(path_points) - 1:
            # 最后一个点使用终点旋转
            agent_state.position = path_points[i]
            agent_state.rotation = end_rotation
        elif i == 0:
            # 最后一个点使用终点旋转
            agent_state.position = path_points[i]
            agent_state.rotation = start_rotation
        else:
            # 中间点使用当前点和上一个点之间的方向的反方向
            point = path_points[i]
            prior_point = path_points[i + 1]
            direction = point - prior_point
            direction = direction / np.linalg.norm(direction)
            # 计算当前点的旋转
            forward = np.quaternion(0, 0, 0, -1)
            rotated_forward = start_rotation * forward * start_rotation.conj()
            current_forward = np.array([rotated_forward.x, rotated_forward.y, rotated_forward.z])
            # 投影到水平面 (忽略y轴)
            current_forward[1] = 0
            direction[1] = 0
            # 归一化
            if np.linalg.norm(current_forward) < 1e-6 or np.linalg.norm(direction) < 1e-6:
                continue
            current_forward = current_forward / np.linalg.norm(current_forward)
            direction = direction / np.linalg.norm(direction)
            # 计算旋转角度 (带符号)
            cross = np.cross(current_forward, direction)
            dot = np.dot(current_forward, direction)
            # 计算旋转角度 (度)
            angle_deg = np.degrees(np.arctan2(np.linalg.norm(cross), dot))
            # 确定旋转方向 (使用叉积的y分量符号)
            if cross[1] < 0:
                angle_deg = -angle_deg
            # 更新旋转（四元数）
            delta_rot = np.quaternion(np.cos(np.radians(-angle_deg)/2), 
                                        0, np.sin(np.radians(-angle_deg)/2), 0)
            start_rotation = start_rotation * delta_rot
            # 设置代理状态
            agent_state.position = path_points[i]
            agent_state.rotation = start_rotation

        agent.set_state(agent_state)

        obs = sim.get_sensor_observations()
        cv2.imwrite(f"{i}.jpg", obs["color"][..., :3][..., ::-1])

def main():
    # 配置参数
    scene_file = "data/HM3D/00876-mv2HUxq3B53/mv2HUxq3B53.basis.glb"  # 替换为你的场景GLB文件路径
    navmesh_file = "data/HM3D/00876-mv2HUxq3B53/mv2HUxq3B53.basis.navmesh"  # 替换为你的导航网格文件路径
    output_video = "output.mp4"  # 输出视频文件名
    
    # 起点和终点的位置和旋转(四元数)
    start_position = np.array([3.34482479095459, 0.050354525446891785, 9.988510131835938])  # 替换为实际起点坐标
    start_rotation = quaternion.from_float_array([0.06694663545449525, 0.0, 0.997756557483499, 0.0])  # wxyz

    end_position = np.array([-1.5373001098632812, 0.050354525446891785, 16.74457359313965])  # 替换为实际终点坐标
    end_rotation = quaternion.from_float_array([0.015153784750062174, 0.0, 0.9998851748114626, 0.0])  # wxyz

    # 初始化模拟器配置
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_file
    backend_cfg.enable_physics = False  # 不需要物理引擎

    # 2. 配置RGB传感器
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [480, 640]
    sensor_spec.hfov = 120.0  # 水平视场角
    sensor_spec.position = mn.Vector3(0, 1.5, 0)  # 相对于代理的位置
    sensor_spec.orientation = mn.Vector3(0, 0, 0)
    
    action_space_config = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward", 
            habitat_sim.ActuationSpec(amount=0.1)  # 默认步长
        ),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=10.0)  # 默认旋转角度
        ),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(amount=10.0)  # 默认旋转角度
        )
    }

    # 代理配置
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    agent_cfg.action_space = action_space_config
    
    # 创建模拟器
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    
    # 加载导航网格
    if not os.path.exists(navmesh_file):
        print(f"Navmesh file {navmesh_file} not found!")
        return
    
    sim.pathfinder.load_nav_mesh(navmesh_file)
    
    # 计算最短路径
    path = habitat_sim.ShortestPath()
    path.requested_start = start_position
    path.requested_end = end_position
    
    # 计算最短路径
    found_path = sim.pathfinder.find_path(path)
    if not found_path:
        print("No valid path found between start and end positions!")
        return
    
    print(f"Path found with {len(path.points)} points")
    print(f"Path length: {path.geodesic_distance}")
    
    # 初始化代理
    agent = sim.initialize_agent(0)

    # 使用 path.points 获取路径点
    path_points = path.points

    save_obs_on_path(sim, agent, path_points, start_rotation, end_rotation)
    save_map_with_path(sim, path_points, start_rotation, end_rotation)
    save_navigation_video(sim, agent, path_points, start_position, start_rotation, end_position, end_rotation, output_video)
    
    # 关闭模拟器
    sim.close()


if __name__ == "__main__":
    main()