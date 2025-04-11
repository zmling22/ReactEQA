import os
import json
import numpy as np
import cv2
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from grounded_sam import build_grounded_sam, grounded_sam_inference
from sklearn.cluster import DBSCAN

SCENE_ID = "wcojb4TFT35"
BASE_DIR = f"/home/hanshengliang/hm3d-data/val-data/hm3d-val-habitat-v0.2/00802-{SCENE_ID}"

INPUT_GLB = os.path.join(BASE_DIR, f"{SCENE_ID}.basis.glb")
OUTPUT_GLB = os.path.join(BASE_DIR, f"{SCENE_ID}.semantic.glb")
OUTPUT_TXT = os.path.join(BASE_DIR, f"{SCENE_ID}.semantic.txt")
OUTPUT_JSON = os.path.join(BASE_DIR, f"{SCENE_ID}.semantic.json")

RGB_DIR = os.path.join(BASE_DIR, "rgb")
DEPTH_DIR = os.path.join(BASE_DIR, "depth")
POSE_DIR = os.path.join(BASE_DIR, "pose")
INTRINSIC_PATH = os.path.join(BASE_DIR, "intrinsic.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 工具函数 =================
def load_intrinsics(path):
    with open(path, 'r') as f:
        K = np.array(json.load(f)["intrinsic_matrix"]).reshape(3, 3)
    return K

def depth_to_points(depth, K):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth / 1000.0
    x = (i - K[0, 2]) * z / K[0, 0]
    y = (j - K[1, 2]) * z / K[1, 1]
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)

def rgb_to_hex(rgb):
    return ''.join(f'{c:02X}' for c in rgb)

def generate_semantic_txt(path, instance_to_class, class_to_color):
    lines = []
    for iid, cname in instance_to_class.items():
        color = class_to_color[cname]
        hex_color = rgb_to_hex(color)
        lines.append(f"{iid},{hex_color},{cname},0\n")
    with open(path, 'w') as f:
        f.writelines(lines)
    print(f"语义映射 TXT 已保存：{path}")

def save_colored_pointcloud(points_list, colors_list, filename="output_colored_global_pointcloud.ply"):
    all_points = np.vstack(points_list)
    all_colors = np.vstack(colors_list) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"着色点云已保存为：{filename}")

def save_instance_metadata_json(path, instance_metadata_list):
    with open(path, "w") as f:
        json.dump({"objects": instance_metadata_list}, f, indent=4)
    print(f"目标信息 JSON 已保存：{path}")



def merge_duplicate_instances_with_metadata(global_points, instance_to_class, class_to_color, iou_threshold=0.3):
    def get_aabb(points):
        min_xyz = np.min(points, axis=0)
        max_xyz = np.max(points, axis=0)
        center = (min_xyz + max_xyz) / 2
        dimensions = max_xyz - min_xyz
        return {
            "min": min_xyz.tolist(),
            "max": max_xyz.tolist(),
            "center": center.tolist(),
            "dimensions": dimensions.tolist()
        }

    def box_iou(box1, box2):
        min1, max1 = np.array(box1["min"]), np.array(box1["max"])
        min2, max2 = np.array(box2["min"]), np.array(box2["max"])
        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        inter_dims = np.maximum(inter_max - inter_min, 0)
        inter_vol = np.prod(inter_dims)
        vol1 = np.prod(max1 - min1)
        vol2 = np.prod(max2 - min2)
        union_vol = vol1 + vol2 - inter_vol
        return inter_vol / union_vol if union_vol > 0 else 0

    def filter_by_dbscan(points, eps=0.05, min_samples=10):
        if len(points) < min_samples:
            return points
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        if np.all(labels == -1):
            return points
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        main_label = unique[np.argmax(counts)]
        return points[labels == main_label]

    merged_points, merged_colors, merged_metadata = [], [], []
    used = [False] * len(global_points)

    for i in range(len(global_points)):
        if used[i]:
            continue
        cls_name = instance_to_class[i]
        if cls_name not in class_to_color:
            continue
        pts = global_points[i]
        bbox = get_aabb(pts)

        merged_idxs = [i]
        used[i] = True

        for j in range(i + 1, len(global_points)):
            if used[j] or instance_to_class[j] != cls_name:
                continue
            iou = box_iou(bbox, get_aabb(global_points[j]))
            if iou > iou_threshold:
                merged_idxs.append(j)
                used[j] = True

        merged_pts = np.vstack([global_points[k] for k in merged_idxs])
        filtered_pts = filter_by_dbscan(merged_pts)  # << 加了这步

        color = class_to_color[cls_name]
        merged_clr = np.tile(color, (len(filtered_pts), 1))
        merged_bbox = get_aabb(filtered_pts)

        merged_points.append(filtered_pts)
        merged_colors.append(merged_clr)

        metadata = {
            "object_id": len(merged_points) - 1,
            "category_name": cls_name,
            "region_id": 0,
            "color": color,
            "bbox": merged_bbox,
            "vertex_count": len(filtered_pts)
        }
        merged_metadata.append(metadata)

    return merged_points, merged_colors, merged_metadata

# ================= 主流程 =================
def main():
    with open("obj_cls.json", "r", encoding="utf-8") as f:
        TARGET_CLASSES = json.load(f)  # 类名 -> 颜色（RGB）
    GROUNDING_CATEGORIES = list(TARGET_CLASSES.keys())
    TEXT_PROMPT = " . ".join(GROUNDING_CATEGORIES)

    grounded_model = build_grounded_sam(device=DEVICE)
    K = load_intrinsics(INTRINSIC_PATH)

    instance_to_class = {}
    instance_id = 0
    global_points = []

    rgb_files = sorted(os.listdir(RGB_DIR))
    for rgb_file in tqdm(rgb_files):
        stem = os.path.splitext(rgb_file)[0]
        rgb_path = os.path.join(RGB_DIR, rgb_file)
        depth_path = os.path.join(DEPTH_DIR, stem + ".png")
        pose_path = os.path.join(POSE_DIR, stem + ".txt")

        image = cv2.imread(rgb_path)[:, :, ::-1].copy()
        depth = cv2.imread(depth_path, -1)
        pose = np.loadtxt(pose_path)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes, masks, phrases = grounded_sam_inference(image_tensor, TEXT_PROMPT, grounded_model)
        

        if boxes is None or len(boxes) == 0:
            print(f"No objects detected in {rgb_file}")
            continue

        pts = depth_to_points(depth, K)
        #import pdb;pdb.set_trace()
        pts_world = (pose[:3, :3] @ pts.T + pose[:3, 3:4]).T

        for j, mask in enumerate(masks):
            mask_np = mask
            if mask_np.shape != depth.shape:
                mask_np = cv2.resize(mask_np.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

            idxs = mask_np.reshape(-1).astype(bool)
            if idxs.sum() < 100:
                continue

            inst_pts = pts_world[idxs]
            phrase = phrases[j].lower()
            matched_class = next((c for c in GROUNDING_CATEGORIES if c in phrase), phrase)

            instance_to_class[instance_id] = matched_class
            global_points.append(inst_pts)
            instance_id += 1

    # 合并实例并生成 metadata
    global_points, global_colors, instance_metadata = merge_duplicate_instances_with_metadata(
        global_points, instance_to_class, TARGET_CLASSES, iou_threshold=0.3
    )

    # 保存输出
    save_colored_pointcloud(global_points, global_colors)
    generate_semantic_txt(OUTPUT_TXT, {m["object_id"]: m["category_name"] for m in instance_metadata}, TARGET_CLASSES)
    save_instance_metadata_json(OUTPUT_JSON, instance_metadata)

if __name__ == '__main__':
    main()
