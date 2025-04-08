import trimesh
import numpy as np
from collections import defaultdict
import json
import numpy as np
import open3d as o3d
import pickle
import os
from tqdm import tqdm

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

def extract_objects_from_glb(glb_path, color_mapping):
    scene = trimesh.load(glb_path, force='scene')
    
    # 按颜色分组顶点
    color_groups = defaultdict(list)
    for name, mesh in scene.geometry.items():
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
    for color, vertices in color_groups.items():
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
    
    return objects

def save_to_json(objects, output_path):
    result = {
        'metadata': {
            'source_glb': 'HkseAnWCgqk.semantic.glb',
            'source_txt': 'HkseAnWCgqk.semantic.txt',
            'object_count': len(objects)
        },
        'objects': objects
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

def save_to_pkl(objects, output_path):
    result = {
        'metadata': {
            'source_glb': 'HkseAnWCgqk.semantic.glb',
            'source_txt': 'HkseAnWCgqk.semantic.txt',
            'object_count': len(objects)
        },
        'objects': objects
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

def extract(semantic_glb, semantic_txt):
    # 1. 解析颜色-类别映射
    color_mapping = parse_color_mapping(semantic_txt)
    # 2. 从GLB提取物体信息
    objects = extract_objects_from_glb(semantic_glb, color_mapping)
    # 3. 保存为PKL
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
        semantic_glb = os.path.join(path, scene_id + ".semantic.glb")
        semantic_txt = os.path.join(path, scene_id + ".semantic.txt")

        output_path = os.path.join(path, scene_id + ".objects.pkl")
        if os.path.exists(output_path):
            print(f"文件 {output_path} 已存在，跳过该目录。")
            continue

        if os.path.exists(semantic_glb) and os.path.exists(semantic_txt):
            extract(semantic_glb, semantic_txt)
        else:
            print(f"文件 {semantic_glb} 或 {semantic_txt} 不存在，跳过该目录。")
