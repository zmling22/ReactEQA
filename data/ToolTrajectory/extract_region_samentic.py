# 提取每个区域的语义信息
import os
from generator_deerapi import requests_api
from collections import defaultdict
import json
import pickle
import numpy as np
from tqdm import tqdm
import habitat_sim
import math
from src.utils.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
import cv2
import random

def load_focus_point(file_path):
    focus_points = defaultdict(list)
    with open(file_path, "rb") as f:
        object_data = pickle.load(f)

    object_data = object_data["objects"]
    for object in object_data:
        region_id = object["region_id"]
        category_name = object["category_name"]
        if category_name in ["wall", "wall panel", "ceiling", "floor", "window frame", "window", "door", "door frame", 
                             "kitchen countertop item", "kitchen countertop", "kitchen appliance", "appliance, decoration", "bathroom accessory", 
                             "beam", "kitchen counter", "duct"]:
            continue
        focus_points[region_id].append((category_name.replace(" ", "-"), object["bbox"]["center"], object["bbox"]["dimensions"]))

    return focus_points


def create_sim(scene_dir):
    scene_id = scene_dir.split("/")[-1].split("-")[-1]
    scene_file = os.path.join(scene_dir, scene_id + ".basis" + ".glb")
    navmesh_file = os.path.join(scene_dir, scene_id + ".basis" + ".navmesh")
    
    # 配置模拟器参数
    sim_settings = {
            "scene": scene_file,
            "default_agent": 0,
            "sensor_height": 0.0,
            "width": 640,
            "height": 480,
            "hfov": 120,
    }
    sim_cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(sim_cfg)

    pathfinder = sim.pathfinder
    pathfinder.load_nav_mesh(navmesh_file)

    agent = sim.initialize_agent(sim_settings["default_agent"])
    
    return sim, agent

def quat_from_yaw(yaw):
    """从偏航角(yaw)创建四元数"""
    return np.array([0, np.sin(yaw/2), 0, np.cos(yaw/2)])

def set_agent_orientation_towards_point(sim, agent, target_point, radius=5.0, max_attempts=500):
    target_point = pos_normal_to_habitat(target_point)
    pathfinder = sim.pathfinder
    snapped_target = pathfinder.get_random_navigable_point_near(target_point, radius=radius, max_tries=max_attempts)
    snapped_target[1] = target_point[1]

    best_position = None
    best_rotation = None
    min_distance_error = float('inf')
    
    for _ in range(max_attempts):
        # 在目标点周围球面上随机采样
        random_dir = np.random.normal(size=3)
        random_dir[1] = 0  # 保持水平
        random_dir = random_dir / np.linalg.norm(random_dir)
        
        candidate = snapped_target - random_dir * radius
        candidate[1] = target_point[1]  # 保持相同高度
        candidate = pathfinder.get_random_navigable_point_near(candidate, radius=radius, max_tries=max_attempts)
        
        # 检查点是否可导航
        if not pathfinder.is_navigable(candidate):
            # print(f"点{candidate}不可导航")
            continue
            
        # 计算距离误差
        current_distance = np.linalg.norm(candidate - snapped_target)
        distance_error = abs(current_distance - radius)
        
        # 如果找到更精确的距离，更新最佳位置
        if distance_error < min_distance_error:
            min_distance_error = distance_error
            # 计算朝向
            direction = snapped_target - candidate
            direction[1] = 0
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            yaw = np.arctan2(direction[2], direction[0])
            rotation = quat_from_yaw(yaw)
            
            best_position = candidate
            best_rotation = rotation
            
            # 如果误差很小(小于5cm)，提前返回
            if distance_error < 0.05:
                break
    
    # if best_position is None:
    #     raise RuntimeError(f"无法找到距离目标点{distance}米且无遮挡的可导航位置")
    # if best_rotation == np.array([ 0., np.nan,  0., np.nan]):
    #     raise RuntimeError(f"无法找到朝向目标点的可导航位置")

    if best_position is None:
        # print(f"无法找到距离目标点{radius}米且无遮挡的可导航位置")
        return False

    # 5. 设置代理状态
    agent_state = habitat_sim.AgentState()
    agent_state.position = best_position
    agent_state.rotation = best_rotation
    agent.set_state(agent_state)
    return True
    

def sample_point_observation(sim, agent, target_point, radius):
    flag = set_agent_orientation_towards_point(sim, agent, target_point, radius)
    if not flag:
        return None
    
    observation = sim.get_sensor_observations()
    rgb_image = observation["color_sensor"][..., :3][..., ::-1]

    # 判断图像黑色区域是否大于50%
    black_pixels = np.sum(np.all(rgb_image == [0, 0, 0], axis=-1))
    total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
    black_ratio = black_pixels / total_pixels
    if black_ratio > 0.5:
        return None

    return rgb_image


def extract_region_semantic(scene_dir, prompt):
    sim, agent = create_sim(scene_dir)

    scene_id = scene_dir.split("/")[-1].split("-")[-1]
    data = load_focus_point(os.path.join(scene_dir, scene_id + ".objects.pkl"))

    num_region = len(data)
    region_map = []
    for region_id, objs_info in tqdm(data.items()):
        # print(f"正在处理区域 {region_id}，包含 {len(objs_info)} 个采样点")
        sampled_points = objs_info
        image_files = []
        for i, (name, point, dimensions) in enumerate(sampled_points):
            # radius = max([max(dimensions) * 2, 4.0])
            radius = max(dimensions)
            image_file = sample_point_observation(sim, agent, point, radius)
            if image_file is None:
                continue

            save_name = f"{name}_{i}.png"
            save_dir = os.path.join(scene_dir, "region_images")
            save_path = os.path.join(scene_dir, "region_images", f"region_{region_id}")
            if os.path.isdir(save_path):
                cv2.imwrite(os.path.join(save_path, save_name), image_file)
            elif not os.path.exists(save_dir):
                os.makedirs(save_dir)
                os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, save_name), image_file)
            else:
                os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, save_name), image_file)

            if image_file is None:
                # print(f"采样点{point}失败，跳过")
                continue
            image_files.append(image_file)
        if len(image_files) >= 3: 
            image_files = random.sample(image_files, 3)
        if len(image_files) == 0:
            continue

        result = requests_api(image_files, prompt)
        region_name = result["choices"][0]["message"]["content"].strip()
        region_map.append({"region_id": f"region_{region_id}", "region_name": region_name})

    try:
        sim.close()
    except:
        pass

    save_region_map_file = os.path.join(scene_dir, scene_id + ".region.json")

    with open(save_region_map_file, "w") as f:
        json.dump(region_map, f, indent=4)
    

if __name__=="__main__":
    random.seed(42)

    root = "data/HM3D"
    dirs = os.listdir(root)
    dirs.sort()

    prompt_path = "data/ToolTrajectory/prompts/prompt_region_samentic.txt"
    with open(prompt_path, "r") as file:
        prompt = file.read()

    for dir in tqdm(dirs):
        path = os.path.join(root, dir)
        if not os.path.isdir(path):
            continue
        number, scene_id = dir.split("-")

        output_path = os.path.join(path, scene_id + ".region.json")
        # if os.path.exists(output_path):
        #     print(f"文件 {output_path} 已存在，跳过该目录。")
        #     continue

        object_pkl = os.path.join(path, scene_id + ".objects.pkl")
        if os.path.exists(object_pkl):
            extract_region_semantic(os.path.join(root, dir), prompt)
            exit()
        else:
            # print(f"文件 {object_pkl} 不存在，跳过该目录。")
            continue

