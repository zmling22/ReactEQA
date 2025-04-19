# 从指定场景中随机采样 Agent 的位置和姿态，并保存 RGB 图像和状态信息
import habitat_sim
import numpy as np
import random
import os
import cv2
import math
import json
import csv

from tqdm import tqdm
from src.utils.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from habitat_sim.utils.common import quat_from_angle_axis

# 确保场景数据路径正确
scene_dir = "data/HM3D/"

def create_sim(scene):
    scene_file = os.path.join(
            scene_dir, scene, scene[6:] + ".basis" + ".glb"
        )
    navmesh_file = os.path.join(
            scene_dir, scene, scene[6:] + ".basis" + ".navmesh"
        )
    
    # 配置模拟器参数
    sim_settings = {
            "scene": scene_file,
            "default_agent": 0,
            "sensor_height": 1.5,
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

def get_random_image(sim):
    # 随机采样可行走点
    target_position = sim.pathfinder.get_random_navigable_point()
    
    # 随机旋转（水平方向）
    angle = np.random.uniform(0, 2 * np.pi)
    rotation = quat_from_angle_axis(angle, np.array([0, 1, 0]))  # Y轴旋转
    
    # 设置 Agent 状态
    agent_state = habitat_sim.AgentState()
    agent_state.position = target_position
    agent_state.rotation = rotation
    agent.set_state(agent_state)
    
    # 获取观察结果
    observation = sim.get_sensor_observations()
    rgb_image = observation["color_sensor"]  # 形状 [H, W, 3]
    
    return rgb_image, agent_state

def get_random_samples(sim, num_samples=3):
    images = []
    states = []
    for _ in range(num_samples):
        rgb_image, agent_state = get_random_image(sim)
        images.append(rgb_image)
        states.append(agent_state)

    return images, states

if __name__ == "__main__":
    csv_columns = ["scene", "image_path", "position", "rotation"]    
    csv_file_path = 'data/ReactEQA/sample_scene/position.csv'

    finished_samples = []
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                finished_samples.append(row['source_image'])

    csvfile = open(csv_file_path, "a", newline='')
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)

    for scene in tqdm(os.listdir(scene_dir)):
        if not os.path.isdir(os.path.join(scene_dir, scene)):
            continue
        sim, agent = create_sim(scene)
        sampled_images, sampled_states = get_random_samples(sim, 10)

        datas = []
        for i, (img, state) in enumerate(zip(sampled_images, sampled_states)):
            # 保存图片
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            save_root = f"data/ReactEQA/sample_scene/images/{scene}"
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            image_path = os.path.join(save_root, f"{i}.png")
            cv2.imwrite(image_path, img)

            # 保存状态
            pos = state.position
            rot = state.rotation
            rot = [rot.x, rot.y, rot.z, rot.w]
            print(f"Scene: {scene}, Position: {pos}, Rotation: {rot}")
            item = {"scene": scene, "image_path": image_path, "position": pos.tolist(), "rotation": rot}
            datas.append(item)
            writer.writerow(item)
        try:
            sim.close()
        except:
            pass
    json.dump(datas, open("data/ReactEQA/sample_scene/position.json", "w"), indent=4)