# 提取每个区域的语义信息
import os
from collections import defaultdict
import json
import pickle
import numpy as np
from tqdm import tqdm
import habitat_sim
import quaternion
import math
import base64
from src.utils.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
import io
import cv2
import random
import requests
from habitat_sim.utils.common import quat_from_two_vectors
from PIL import Image
from sklearn.cluster import DBSCAN
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load tokenizer and model
model_name = "/mynvme0/models/Qwen2-VL/Qwen2-VL-72B-Instruct-GPTQ-Int4/"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, 
        device_map="auto", 
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).eval()
processor = AutoProcessor.from_pretrained(model_name)


def load_focus_point(file_path):
    focus_points = defaultdict(list)
    with open(file_path, "rb") as f:
        object_data = pickle.load(f)

    object_data = object_data["objects"]
    for object in object_data:
        object_id = object["object_id"]
        region_id = object["region_id"]
        category_name = object["category_name"]
        if category_name in ["wall", "wall panel", "ceiling", "floor", "window frame", "window", "door", "door frame", 
                             "kitchen countertop item", "kitchen countertop", "kitchen appliance", "appliance, decoration", "bathroom accessory", 
                             "beam", "kitchen counter", "duct"]:
            continue
        focus_points[region_id].append((object_id, category_name.replace(" ", "-"), object["bbox"]["center"], object["bbox"]["dimensions"]))

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
            "hfov": 90,
            "enable_physics": True,
    }
    sim_cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(sim_cfg)

    pathfinder = sim.pathfinder
    pathfinder.load_nav_mesh(navmesh_file)

    agent = sim.initialize_agent(sim_settings["default_agent"])
    
    return sim, agent


def look_at(from_pos, to_pos):
    direction = to_pos - from_pos
    direction /= np.linalg.norm(direction)

    # 提取 yaw（水平旋转）和 pitch（上下抬头）
    dx, dy, dz = direction
    yaw = np.arctan2(-dx, -dz)  # 注意摄像头默认朝向 -Z
    pitch = np.arcsin(dy)       # 上下方向

    # 先绕 Y 轴（yaw），再绕 X 轴（pitch）
    q_yaw = quaternion.from_rotation_vector(yaw * np.array([0, 1, 0]))
    q_pitch = quaternion.from_rotation_vector(pitch * np.array([1, 0, 0]))

    return q_yaw * q_pitch


def detect_floor_heights(sim, min_y=0.0, max_y=10.0, y_step=0.1, sample_num=100):
    pathfinder = sim.pathfinder
    min_bounds, max_bounds = pathfinder.get_bounds()
    min_x, min_z = min_bounds[0], min_bounds[2]
    max_x, max_z = max_bounds[0], max_bounds[2]

    candidate_heights = []
    for y in np.arange(min_y, max_y, y_step):
        count = 0
        for _ in range(sample_num):
            x = np.random.uniform(min_x, max_x)
            z = np.random.uniform(min_z, max_z)
            if pathfinder.is_navigable([x, y, z]):
                count += 1
        if count > 0:
            candidate_heights.append(y)

    db = DBSCAN(eps=0.3, min_samples=2).fit(np.array(candidate_heights).reshape(-1, 1))
    labels = db.labels_
    floor_centers = []
    for l in set(labels):
        if l == -1:
            continue
        ys = np.array(candidate_heights)[labels == l]
        floor_centers.append(float(np.mean(ys)))

    floor_centers.sort()
    # print(f"检测到 {len(floor_centers)} 层楼: {floor_centers}")
    return floor_centers


def get_y_offset(point, floor_heights):
    y = point[1]  # 获取点的y坐标(高度)
    
    # 如果楼层高度列表为空，返回-1
    if len(floor_heights) == 1:
        return floor_heights[0] / 2
    
    for i in range(len(floor_heights)):
        if i == 0:
            continue
        if y < floor_heights[i - 1]:
            return floor_heights[i - 1] / 2
        elif y >= floor_heights[i - 1] and y < floor_heights[i]:
            return (floor_heights[i] - floor_heights[i - 1]) / 2 + floor_heights[i - 1]
        else:
            return floor_heights[i]


def is_point_inside_scene(sim, point):
    """
    通过尝试snap点来验证是否在场景内
    :param sim: Habitat Simulator实例
    :param point: 要检查的3D点
    :return: True如果在场景内，False否则
    """
    if not sim.pathfinder.is_loaded:
        return False
    
    try:
        snapped_point = sim.pathfinder.snap_point(point)
        # 如果snap后的点距离原始点太远，可能不在场景内
        return np.linalg.norm(np.array(point) - np.array(snapped_point)) < 2.0
    except:
        # 如果snap失败，通常意味着点不在场景内
        return False


def pil_to_base64(image: Image.Image, format='JPEG') -> str:
    """
    将PIL图像转换为Base64编码字符串
    
    参数:
        image: PIL.Image对象
        format: 输出格式（JPEG/PNG等）
    
    返回:
        Base64编码的字符串
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_str = f"data:image;base64,{img_str}"

    return img_str

def request_vlm(image, prompt):
    image = Image.fromarray(image)
    image = pil_to_base64(image)

    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # print(inputs)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text


def check_img(img, cate_name, color_threshold = 0.1, variance_threshold = 1500):
    # 验证黑色区域占比
    if np.sum(np.all(img == [0, 0, 0], axis=-1)) / (img.shape[0] * img.shape[1]) > color_threshold:
        return False
    
    # 验证颜色方差
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var_gray = np.var(gray)
    if var_gray < variance_threshold:
        return False
    
    # 验证目标包含
    prompt = f"Does this picture include a {cate_name}? Answer with single letter, Y or N."
    
    res = request_vlm(img, prompt)
    
    if res == "N":
        return False
    
    return True


def capture_object_fine_details(sim, cate_name, floor_heights, object_center, radius=1, num_views=6):
    rgbs = []
    object_center = pos_normal_to_habitat(object_center)
    y_offset = get_y_offset(object_center, floor_heights)

    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        # 在 XZ 平面环绕
        cam_pos = object_center + radius * np.array([np.sin(angle), 0, np.cos(angle)])
        cam_pos[1] = y_offset
        if is_point_inside_scene(sim, cam_pos):
            rotation = look_at(cam_pos, object_center)

            # 设置 agent 状态
            state = habitat_sim.AgentState()
            state.position = cam_pos
            state.rotation = rotation
            sim.get_agent(0).set_state(state)
        
            # 采图
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"][..., :3][..., ::-1]

            # 筛选图片
            if check_img(rgb, cate_name):
                rgbs.append(rgb)
            
    return rgbs


def extract_region_semantic(scene_dir):
    sim, agent = create_sim(scene_dir)

    scene_id = scene_dir.split("/")[-1].split("-")[-1]
    data = load_focus_point(os.path.join(scene_dir, scene_id + ".objects.pkl"))

    floor_heights = detect_floor_heights(sim)

    region_map = []
    for region_id, objs_info in tqdm(data.items()):
        image_files = []
        for _, (id, name, target_point, dimensions) in enumerate(objs_info):
            radius = max(dimensions)

            image_files = capture_object_fine_details(sim, name, floor_heights, target_point, radius)

            if len(image_files) == 0:
                continue
            for j, image_file in enumerate(image_files):
                if image_file is None:
                    continue
                save_name = f"{id}_{name}_{j}.png"
                save_dir = os.path.join(scene_dir, "objects_rgb")
                save_path = os.path.join(scene_dir, "objects_rgb", f"region_{region_id}")

                if os.path.isdir(save_path):
                    cv2.imwrite(os.path.join(save_path, save_name), image_file)
                elif not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path, save_name), image_file)
                else:
                    os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path, save_name), image_file)
    try:
        sim.close()
    except:
        pass

if __name__=="__main__":
    random.seed(42)

    root = "data/HM3D"
    dirs = os.listdir(root)
    dirs.sort()

    for dir in tqdm(dirs):
        path = os.path.join(root, dir)
        if not os.path.isdir(path):
            continue
        number, scene_id = dir.split("-")

        # output_path = os.path.join(path, scene_id + ".region.json")
        # if os.path.exists(output_path):
        #     print(f"文件 {output_path} 已存在，跳过该目录。")
        #     continue

        object_pkl = os.path.join(path, scene_id + ".objects.pkl")
        if os.path.exists(object_pkl):
            extract_region_semantic(os.path.join(root, dir))
        else:
            # print(f"文件 {object_pkl} 不存在，跳过该目录。")
            continue
