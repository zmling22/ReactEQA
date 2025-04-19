import os
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm
from sklearn.cluster import DBSCAN, KMeans
from collections import defaultdict
from grounded_sam import build_grounded_sam, grounded_sam_inference
import time
from cuml.cluster import DBSCAN as cuDBSCAN
import cupy as cp
from sklearn.cluster import DBSCAN as cpuDBSCAN

def pose_habitat_to_normal(pose):
    return np.dot(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), pose)

def pose_normal_to_tsdf(pose):
    return np.dot(pose, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))

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

def save_instance_metadata_json(path, instance_metadata_list):
    with open(path, "w") as f:
        json.dump({"objects": instance_metadata_list}, f, indent=4)
    print(f"目标信息 JSON 已保存：{path}")


def merge_duplicate_instances_with_metadata(global_points_raw, instance_to_class, class_to_color, iou_threshold=0.3):
    def get_aabb_torch(points):
        bbox_min = points.min(dim=0).values
        bbox_max = points.max(dim=0).values
        center = (bbox_min + bbox_max) / 2
        dimensions = bbox_max - bbox_min
        return {"min": bbox_min, "max": bbox_max, "center": center, "dimensions": dimensions}

    def fast_box_iou_torch(b1, b2):
        inter_min = torch.max(b1["min"], b2["min"])
        inter_max = torch.min(b1["max"], b2["max"])
        inter_dims = torch.clamp(inter_max - inter_min, min=0)
        inter_vol = inter_dims.prod()
        vol1 = (b1["max"] - b1["min"]).prod()
        vol2 = (b2["max"] - b2["min"]).prod()
        union_vol = vol1 + vol2 - inter_vol
        return inter_vol / union_vol if union_vol > 0 else torch.tensor(0.0, device='cuda')

    def filter_by_dbscan_gpu(points_np, eps=0.05, min_samples=10, max_pts=20000):
        if points_np.shape[0] < min_samples:
            return points_np
        if points_np.shape[0] > max_pts:
            idx = np.random.choice(points_np.shape[0], max_pts, replace=False)
            points_np = points_np[idx]
        points_cp = cp.asarray(points_np.astype(np.float32))
        clustering = cuDBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(points_cp).get()
        if np.all(labels == -1):
            return points_np
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        main_label = unique[np.argmax(counts)]
        return points_np[labels == main_label]

    def filter_by_dbscan_cpu(points_np, eps=0.05, min_samples=10):
        if points_np.shape[0] < min_samples:
            return points_np
        clustering = cpuDBSCAN(eps=eps, min_samples=min_samples).fit(points_np)
        labels = clustering.labels_
        if np.all(labels == -1):
            return points_np
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        main_label = unique[np.argmax(counts)]
        return points_np[labels == main_label]

    def filter_by_dbscan_with_fallback(points_np, eps=0.05, min_samples=10):
        try:
            return filter_by_dbscan_gpu(points_np, eps, min_samples)
        except Exception as e:
            print(f"cuML DBSCAN fallback to CPU due to: {e}")
            return filter_by_dbscan_cpu(points_np, eps, min_samples)

    grouped = defaultdict(list)
    for idx, (points_np, floor_id) in enumerate(global_points_raw):
        cls_name = instance_to_class[idx]
        grouped[(cls_name, floor_id)].append((idx, points_np))

    merged_points, merged_colors, merged_metadata = [], [], []
    instance_id = 0

    for (cls_name, floor_id), items in grouped.items():
        if cls_name not in class_to_color:
            continue

        pts_list = [torch.tensor(pts, device='cuda') for _, pts in items]
        bboxes = [get_aabb_torch(pts) for pts in pts_list]
        used = [False] * len(items)

        for i in range(len(items)):
            if used[i]:
                continue
            merged_idxs = [i]
            used[i] = True
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                iou = fast_box_iou_torch(bboxes[i], bboxes[j])
                if iou.item() > iou_threshold:
                    merged_idxs.append(j)
                    used[j] = True

            merged_pts = torch.cat([pts_list[k] for k in merged_idxs], dim=0)
            merged_pts_np = merged_pts.cpu().numpy()
            filtered_pts = filter_by_dbscan_with_fallback(merged_pts_np)

            if filtered_pts.shape[0] == 0:
                continue

            bbox = get_aabb_torch(torch.tensor(filtered_pts, device='cuda'))
            color = class_to_color[cls_name]
            merged_points.append(filtered_pts)
            merged_colors.append(np.tile(color, (len(filtered_pts), 1)))
            merged_metadata.append({
                "object_id": instance_id,
                "category_name": cls_name,
                "floor_id": floor_id,
                "color": color,
                "bbox": {
                    "min": bbox["min"].tolist(),
                    "max": bbox["max"].tolist(),
                    "center": bbox["center"].tolist(),
                    "dimensions": bbox["dimensions"].tolist()
                },
                "vertex_count": len(filtered_pts)
            })
            instance_id += 1

    return merged_points, merged_colors, merged_metadata


def process_scene(SCENE_ID, BASE_DIR, MAX_INSTANCES=300):
    start_time_total = time.time()  

    OUTPUT_TXT = os.path.join(BASE_DIR, f"{SCENE_ID}.semantic.txt")
    OUTPUT_JSON = os.path.join(BASE_DIR, f"{SCENE_ID}.semantic.json")
    RGB_DIR = os.path.join(BASE_DIR, "rgb")
    DEPTH_DIR = os.path.join(BASE_DIR, "depth")
    POSE_DIR = os.path.join(BASE_DIR, "pose")
    INTRINSIC_PATH = os.path.join(BASE_DIR, "intrinsic.json")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        with open(os.path.join(BASE_DIR, "semantic_floor_map.json"), "r") as f:
            semantic_floor_map = json.load(f)
    except FileNotFoundError:
        print(f"跳过 {SCENE_ID}：缺少 semantic_floor_map.json")
        return

    with open("obj_cls.json", "r", encoding="utf-8") as f:
        TARGET_CLASSES = json.load(f)
    GROUNDING_CATEGORIES = list(TARGET_CLASSES.keys())
    TEXT_PROMPT = " . ".join(GROUNDING_CATEGORIES)

    t0 = time.time()
    grounded_model = build_grounded_sam(device=DEVICE)
    K = load_intrinsics(INTRINSIC_PATH)
    print(f"模型加载与内参读取耗时：{time.time() - t0:.2f}s")

    instance_to_class = {}
    instance_id = 0
    global_points = []

    rgb_files = sorted(os.listdir(RGB_DIR))

    t1 = time.time()
    for rgb_file in tqdm(rgb_files, desc=f"Processing {SCENE_ID}"):
        stem = os.path.splitext(rgb_file)[0]
        rgb_path = os.path.join(RGB_DIR, rgb_file)
        depth_path = os.path.join(DEPTH_DIR, stem + ".png")
        pose_path = os.path.join(POSE_DIR, stem + ".txt")

        image = cv2.imread(rgb_path)[:, :, ::-1].copy()
        depth = cv2.imread(depth_path, -1)
        pose = np.loadtxt(pose_path)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        t_inf = time.time()
        boxes, masks, phrases = grounded_sam_inference(image_tensor, TEXT_PROMPT, grounded_model)
        print(f"单帧目标检测耗时：{time.time() - t_inf:.2f}s")

        if boxes is None or len(boxes) == 0:
            continue

        pts = depth_to_points(depth, K)
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        T_normal = pose_habitat_to_normal(pose)
        T_tsdf = pose_normal_to_tsdf(T_normal)
        pts_world = (T_tsdf @ pts_h.T).T[:, :3]
        floor_id = semantic_floor_map.get(stem, 1)

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
            global_points.append((inst_pts, floor_id))
            instance_id += 1
    print(f"所有图像处理总耗时：{time.time() - t1:.2f}s")

    t2 = time.time()
    if len(global_points) > MAX_INSTANCES:
        pass
    print(f"实例采样耗时：{time.time() - t2:.2f}s")

    t3 = time.time()
    global_points, global_colors, instance_metadata = merge_duplicate_instances_with_metadata(
        global_points, instance_to_class, TARGET_CLASSES, iou_threshold=0.3
    )
    print(f"合并实例耗时：{time.time() - t3:.2f}s")

    t4 = time.time()
    generate_semantic_txt(OUTPUT_TXT, {m["object_id"]: m["category_name"] for m in instance_metadata}, TARGET_CLASSES)
    save_instance_metadata_json(OUTPUT_JSON, instance_metadata)
    print(f"文件保存耗时：{time.time() - t4:.2f}s")
    print(f"总耗时：{time.time() - start_time_total:.2f}s for scene {SCENE_ID}")

def get_scene_list(base_data_dir):
    return sorted([
        d for d in os.listdir(base_data_dir)
        if os.path.isdir(os.path.join(base_data_dir, d)) and "-" in d
    ])

def run_batch(base_data_dir, batch_id=0, total_batch=8, max_instances=300):
    all_scenes = get_scene_list(base_data_dir)

    unprocessed_scenes = []
    for scene_folder in all_scenes:
        scene_id = scene_folder.split("-")[-1]
        scene_path = os.path.join(base_data_dir, scene_folder)
        semantic_json_path = os.path.join(scene_path, f"{scene_id}.semantic.json")
        if not os.path.exists(semantic_json_path):
            unprocessed_scenes.append(scene_folder)
        else:
            print(f"⏭️  跳过 {scene_id}（已存在 semantic.json）")

    batch_scenes = [
        scene for idx, scene in enumerate(unprocessed_scenes)
        if idx % total_batch == batch_id
    ]

    print(f"当前是第 {batch_id + 1}/{total_batch} 批次，需要处理 {len(batch_scenes)} 个场景")

    for scene_folder in tqdm(batch_scenes, desc=f"[GPU{batch_id}] Processing scenes"):
        try:
            scene_id = scene_folder.split("-")[-1]
            scene_path = os.path.join(base_data_dir, scene_folder)
            process_scene(scene_id, scene_path, MAX_INSTANCES=max_instances)
        except Exception as e:
            print(f"处理 {scene_folder} 失败：{e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-id', type=int, default=0, help='当前批次编号，从0开始')
    parser.add_argument('--total-batch', type=int, default=8, help='总批次数，默认8')
    parser.add_argument('--base-dir', type=str, required=True, help='HM3D 场景数据根目录')
    parser.add_argument('--max-instances', type=int, default=300, help='每个场景最多合并的实例数')
    args = parser.parse_args()

    run_batch(args.base_dir, args.batch_id, args.total_batch, args.max_instances)