# 提取语义分割结果中的目标信息保存为pkl

import trimesh
import numpy as np
from collections import defaultdict
import json
import numpy as np
import open3d as o3d
import pickle
import os
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
from analysis_object_color import get_all_colors
import colorsys

def get_dominant_color(colors, max_clusters=5):
    # 归一化颜色
    colors = np.array(colors)
    if colors.max() > 1.0:
        colors = colors / 255.0
    # 合并重复颜色并计算权重
    unique_colors = np.unique(colors, axis=0)

    n_unique = len(unique_colors)

    if n_unique == 1:
        dominant_color = unique_colors[0]
        dominant_color = (dominant_color * 255).astype(int)
        return dominant_color

    if n_unique <= 3:
        counts = np.array([np.sum(np.all(colors == uc, axis=1)) for uc in unique_colors])
        dominant_color = unique_colors[np.argmax(counts)]
        dominant_color = (dominant_color * 255).astype(int)
        return dominant_color
    
    max_clusters = min(max_clusters, n_unique - 1)
    if max_clusters < 1:
        dominant_color = np.mean(colors, axis=0)
        dominant_color = (dominant_color * 255).astype(int)
        return dominant_color
    
    # 自动选择最佳聚类数（肘部法则）
    inertias = []
    possible_k = range(1, max_clusters + 1)
    for k in possible_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(colors)
        inertias.append(kmeans.inertia_)

    if len(inertias) <= 2:  # 数据量过小
        best_k = 1
    else:
        # 计算二阶差分，找到第一个负值最大的点
        second_derivatives = np.diff(np.diff(inertias))
        if len(second_derivatives) == 0:  # 防止空数组
            best_k = 1
        else:
            best_k = possible_k[np.argmin(second_derivatives) + 1]

    if best_k == 1:
        dominant_color = np.mean(colors, axis=0)
        dominant_color = (dominant_color * 255).astype(int)
        return dominant_color
    
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans.fit(colors)
    
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]

    dominant_color = (dominant_color * 255).astype(int)  # 转换为整数RGB值
    return dominant_color


def bbox_intersects(a_min, a_max, b_min, b_max):
    """检查两个 bounding box 是否相交"""
    return not (
        (a_max[0] < b_min[0]) or (a_min[0] > b_max[0]) or
        (a_max[1] < b_min[1]) or (a_min[1] > b_max[1]) or
        (a_max[2] < b_min[2]) or (a_min[2] > b_max[2])
    )


def calculate_texture_average_in_bbox(scene, bbox_min, bbox_max):
    # 创建 bounding box
    bbox = trimesh.primitives.Box(extents=bbox_max - bbox_min)
    bbox.apply_translation((bbox_min + bbox_max) / 2)
    
    # 收集 bounding box 内的网格
    meshes_in_bbox = []
    
    # 遍历场景中的所有几何体
    for name, geometry in scene.geometry.items():
        geom_min, geom_max = geometry.bounds
        if bbox_intersects(geom_min, geom_max, bbox_min, bbox_max):
            meshes_in_bbox.append(geometry)
    
    if not meshes_in_bbox:
        print("No meshes found in the bounding box.")
        return None
    
    # 计算贴图平均颜色
    all_colors = []
    
    for mesh in meshes_in_bbox:
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'material'):
            continue
        
        material = mesh.visual.material
        if not hasattr(material, 'baseColorTexture'):
            continue
        
        # 加载贴图
        texture = material.baseColorTexture
        if texture is None:
            continue
        
        # 确保贴图数据是 PIL.Image
        if isinstance(texture, np.ndarray):
            img = Image.fromarray(texture)
        elif isinstance(texture, Image.Image):
            img = texture
        else:
            continue
        
        # 转换为 numpy 数组并提取所有像素颜色
        img_array = np.array(img)
        if img_array.ndim == 3:  # RGB 或 RGBA
            if img_array.shape[2] == 4:  # RGBA，忽略 alpha 通道
                img_array = img_array[:, :, :3]
            # 将所有像素颜色添加到列表
            height, width, _ = img_array.shape
            for y in range(height):
                for x in range(width):
                    r, g, b = img_array[y, x]
                    all_colors.append([float(r), float(g), float(b)])
    
    if not all_colors:
        print("No valid textures found in the bounding box.")
        return None
    
    return all_colors

def get_dominant_color_simple(colors):
    """通过频率统计获取最显著颜色（适用于离散颜色）"""
    color_counter = Counter(map(tuple, colors))  # 将颜色转换为可哈希的元组
    dominant_color = color_counter.most_common(1)[0][0]  # 取频率最高的颜色
    return list(dominant_color)

def get_dominant_color_kmeans(colors, k=3, n_samples=1000):
    """
    通过K-Means聚类获取最显著颜色
    :param colors: 形状为 (n, 3) 的RGB颜色列表
    :param k: 聚类数量（默认3类）
    :param n_samples: 随机采样数量（避免计算过载）
    """
    colors = np.array(colors)
    if len(colors) > n_samples:
        colors = colors[np.random.choice(len(colors), n_samples, replace=False)]
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(colors)
    
    # 获取最大簇的中心
    counts = np.bincount(kmeans.labels_)
    dominant_idx = np.argmax(counts)
    dominant_color = kmeans.cluster_centers_[dominant_idx].astype(int)
    
    return dominant_color.tolist()

def get_dominant_color_hsv(colors, min_s=0.5, min_v=0.5):
    """
    基于HSV空间筛选高饱和度、高明度的颜色
    :param min_s: 最小饱和度阈值（0-1）
    :param min_v: 最小明度阈值（0-1）
    """
    dominant_colors = []
    for color in colors:
        r, g, b = [x/255.0 for x in color]  # 归一化到[0,1]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s >= min_s and v >= min_v:
            dominant_colors.append(color)
    
    # 如果没有颜色满足条件，返回原始频率最高的颜色
    if not dominant_colors:
        return get_dominant_color_simple(colors)
    
    # 从候选颜色中再次通过K-Means提取
    return get_dominant_color_kmeans(dominant_colors, k=1)

def hex_to_rgb(hex_color):
    """
    将十六进制颜色代码转换为RGB元组
    支持格式: "#RRGGBB" 或 "RRGGBB"
    """
    # 去除#号（如果存在）
    hex_color = hex_color.lstrip('#')
    
    # 转换为RGB整数值
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return (r, g, b)


def vis_pc(vertices, colors):
    """
    可视化点云
    :param vertices: 顶点坐标
    :param colors: 颜色
    """
    assert len(vertices) == len(colors), "顶点和颜色数量不匹配"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    colors_normalized = colors.astype(np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
    
    o3d.io.write_point_cloud("colored_cloud.ply", pcd)

def parse_color_mapping(txt_path):
    color_to_category = {}
    with open(txt_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('HM3D'):
                parts = line.strip("'\n").split(",")
                if len(parts) >= 4:
                    obj_id, color, category_name, region_id = parts[0], hex_to_rgb(parts[1]), parts[2], parts[3]
                    # 格式: R G B category_id category_name
                    r, g, b = color
                    obj_id = int(obj_id)
                    color_to_category[(r, g, b)] = {
                        'obj_id': obj_id,
                        'region_id': int(region_id),
                        'name': category_name.strip("\"")
                    }
    return color_to_category

def find_matching_color(target_color, color_mapping, tolerance=5):
    """
    在颜色映射表中查找最接近的颜色
    :param target_color: 目标颜色 (R,G,B)
    :param color_mapping: 颜色映射字典
    :param tolerance: 颜色匹配容差
    :return: 匹配的颜色键或None
    """
    # 首先尝试精确匹配
    exact_match = tuple(target_color)
    if exact_match in color_mapping:
        return exact_match
    
    # 如果没有精确匹配，则寻找容差范围内的颜色
    for color in color_mapping:
        if all(abs(t - c) <= tolerance for t, c in zip(target_color, color)):
            return color
    return None


def get_vertex_colors(mesh):
    uv = mesh.visual.uv.copy()  # 复制UV避免修改原数据
    
    # 修正UV坐标系（V轴翻转）
    uv[:, 1] = 1 - uv[:, 1]  # PIL的V轴原点在左上，GLB通常在左下
    
    # 处理纹理
    if hasattr(mesh.visual, "material"):
        material = mesh.visual.material
        if hasattr(material, "baseColorTexture"):
            # 获取纹理尺寸
            texture = material.baseColorTexture
            width, height = texture.size
            
            # 处理UV平铺（将超出[0,1]的UV映射回有效范围）
            uv = uv % 1.0
            
            # 采样纹理
            uv_pixels = (uv * [width - 1, height - 1]).astype(int)
            uv_pixels = np.clip(uv_pixels, 0, [width-1, height-1])
            
            # 转换为RGB数组
            texture_array = np.array(texture)[..., :3]  # 确保只取RGB通道
            return texture_array[uv_pixels[:, 1], uv_pixels[:, 0]]  # 注意y,x顺序
    return np.full((len(uv), 3), 0.5)  # 默认灰色（无纹理时）

def extract_objects_from_glb(scene_glb, glb_path, color_mapping):
    scene = trimesh.load(glb_path, force='scene')
    org_scene = trimesh.load(scene_glb, force='scene')

    # 按颜色分组顶点
    color_groups = defaultdict(list)
    for name, mesh in tqdm(scene.geometry.items()):
        vertices = mesh.vertices.copy()
        colors = get_vertex_colors(mesh)
        
        # 按面片展开（保持顶点-颜色对应）
        if hasattr(mesh, 'faces'):
            faces = mesh.faces.reshape(-1)
            vertices = vertices[faces]
            colors = colors[faces]

        if colors is not None:
            # 将顶点按颜色分组
            for v, c in zip(vertices, colors):
                # 转换为整数颜色值
                int_color = tuple(np.round(c).astype(int))
                # 查找匹配的颜色类别
                matched_color = find_matching_color(int_color, color_mapping)
                if matched_color is not None:
                    color_groups[matched_color].append(v)


    # 为每个颜色区域计算bounding box
    objects = []
    for color, vertices in tqdm(color_groups.items()):
        if color in color_mapping:
            vert_array = np.array(vertices)
            bbox_min = vert_array.min(axis=0)
            bbox_max = vert_array.max(axis=0)

            objects.append({
                'object_id': color_mapping[color]['obj_id'],
                'category_name': color_mapping[color]['name'],
                'region_id': color_mapping[color]['region_id'],
                'color': list(color),
                'bbox': {
                    'min': bbox_min.tolist(),
                    'max': bbox_max.tolist(),
                    'center': ((bbox_min + bbox_max) / 2).tolist(),
                    'dimensions': (bbox_max - bbox_min).tolist()
                },
                'vertex_count': len(vertices)
            })

    # 提取物体主色调
    # all_colors = get_all_colors(org_scene, objects)
    # for object, all_color in zip(objects, all_colors):
    #     dominant_color = get_dominant_color_hsv(all_color)
    #     print(dominant_color)
    #     # dominant_color = get_dominant_color(all_color)
    #     object["dominant_color"] = dominant_color

    return objects

def save_to_json(objects, output_path):
    result = {
        'metadata': {
            'object_count': len(objects)
        },
        'objects': objects
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

def save_to_pkl(objects, output_path):
    result = {
        'metadata': {
            'object_count': len(objects)
        },
        'objects': objects
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

def extract(scene_glb, semantic_glb, semantic_txt, output_path):
    # 1. 解析颜色-类别映射
    color_mapping = parse_color_mapping(semantic_txt)
    # 2. 从GLB提取物体信息
    objects = extract_objects_from_glb(scene_glb, semantic_glb, color_mapping)
    # 3. 保存为PKL
    postfix = output_path.split(".")[-1]
    if postfix == "json":
        save_to_json(objects, output_path)
    elif postfix == "pkl":
        save_to_pkl(objects, output_path)

    print(f"成功提取 {len(objects)} 个物体信息并保存到 {output_path}")

# 主程序
if __name__ == "__main__":
    root = "data/HM3D"
    dirs = os.listdir(root)
    dirs.sort()
    for dir in tqdm(dirs):
        path = os.path.join(root, dir)
        if not os.path.isdir(path):
            continue
        number, scene_id = dir.split("-")

        scene_glb = os.path.join(path, scene_id + ".glb")
        semantic_glb = os.path.join(path, scene_id + ".semantic.glb")
        semantic_txt = os.path.join(path, scene_id + ".semantic.txt")

        output_path = os.path.join(path, scene_id + ".objects.pkl")
        # if os.path.exists(output_path):
        #     print(f"文件 {output_path} 已存在，跳过该目录。")
        #     continue

        if os.path.exists(semantic_glb) and os.path.exists(semantic_txt):
            extract(scene_glb, semantic_glb, semantic_txt, output_path)
            exit()
        else:
            print(f"文件 {semantic_glb} 或 {semantic_txt} 不存在，跳过该目录。")
