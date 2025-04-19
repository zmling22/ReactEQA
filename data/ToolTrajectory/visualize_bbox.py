# 可视化3D Bounding Box
# 该脚本使用trimesh库加载GLB文件，并在每个物体上绘制3D Bounding Box

import trimesh
import numpy as np
import pickle
import json
from tqdm import tqdm
from PIL import Image
from extract_object import get_vertex_colors

def bake_texture_to_vertex_color(mesh):
    """将纹理颜色烘焙到顶点颜色"""
    if not hasattr(mesh.visual, 'material') or not hasattr(mesh.visual.material, 'baseColorTexture'):
        return mesh
    
    texture = mesh.visual.material.baseColorTexture
    uv = mesh.visual.uv
    
    if uv is None:
        return mesh
    
    # 将纹理转换为numpy数组
    texture_array = np.array(texture)
    tex_height, tex_width = texture_array.shape[:2]
    
    # 转换UV坐标到纹理像素坐标
    u_coords = np.clip(uv[:, 0], 0, 1) * (tex_width - 1)
    v_coords = (1 - np.clip(uv[:, 1], 0, 1)) * (tex_height - 1)  # V坐标翻转
    
    # 采样纹理颜色
    vertex_colors = texture_array[
        v_coords.astype(int),
        u_coords.astype(int)
    ]
    
    # 创建新网格副本并应用颜色
    baked_mesh = mesh.copy()
    baked_mesh.visual.vertex_colors = vertex_colors
    return baked_mesh


def visualize(scene_path, objects_path, output_path, unexcepts=[]):
    # 1. 加载 GLB 场景
    scene = trimesh.load(scene_path, force='scene')

    # 按颜色分组顶点
    for name, mesh in scene.geometry.items():

        if not hasattr(mesh, 'faces'):
            continue

        # 获取顶点和颜色
        faces = mesh.faces.reshape(-1)
        vertices = mesh.vertices[faces]
        colors = get_vertex_colors(mesh)[faces]

        # 去重处理
        unique_verts, inverse = np.unique(
            vertices, axis=0, return_inverse=True
        )
        
        # 计算平均颜色
        unique_colors = np.zeros((len(unique_verts), 3), dtype=np.uint8)
        for idx in range(len(unique_verts)):
            vertex_colors = colors[inverse == idx]
            if len(vertex_colors) > 0:
                unique_colors[idx] = np.mean(vertex_colors, axis=0)

        # 正确重建面片（确保索引不越界）
        num_faces = len(mesh.faces)
        faces_new = inverse.reshape(-1, 3)  # 使用去重后的索引

        # 验证数据
        assert faces_new.max() < len(unique_verts), "面片索引错误！"

        # 创建新网格
        new_mesh = trimesh.Trimesh(
            vertices=unique_verts,
            faces=faces_new,
            vertex_colors=unique_colors
        )
        
        scene.geometry[name] = new_mesh

    if objects_path.endswith('.json'):
        # 2. 加载 JSON 对象数据
        with open(objects_path, 'r') as f:
            data = json.load(f)
    elif objects_path.endswith('.pkl'):
        # 2. 加载 PKL 对象数据
        with open(objects_path, 'rb') as f:
            data = pickle.load(f)

    for obj in tqdm(data['objects']):
        bbox = obj['bbox']
        name = obj['category_name']
        color = obj['color']
        dominant_color = obj['dominant_color']

        if name in unexcepts:
            continue

        # 创建 Bounding Box
        bbox_min = np.array(bbox['min'])
        bbox_max = np.array(bbox['max'])
        bbox_center = np.array(bbox['center'])
        bbox_dimensions = np.array(bbox['dimensions'])

        # 创建 Bounding Box 可视化
        vertices = np.array([
            [bbox_min[0], bbox_min[1], bbox_min[2]],  # 0
            [bbox_max[0], bbox_min[1], bbox_min[2]],  # 1
            [bbox_max[0], bbox_max[1], bbox_min[2]],  # 2
            [bbox_min[0], bbox_max[1], bbox_min[2]],  # 3
            [bbox_min[0], bbox_min[1], bbox_max[2]],  # 4
            [bbox_max[0], bbox_min[1], bbox_max[2]],  # 5
            [bbox_max[0], bbox_max[1], bbox_max[2]],  # 6
            [bbox_min[0], bbox_max[1], bbox_max[2]]   # 7
        ])

        # # 旋转绕 Y 轴 90 度
        # theta = np.radians(270)
        # rotation_matrix = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(theta), -np.sin(theta)],
        #     [0, np.sin(theta),  np.cos(theta)]
        # ])
        # # vertices = vertices - bbox_center
        # vertices = vertices @ rotation_matrix.T
        # # vertices = vertices + bbox_center

        edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
        ])
        bbox_lines = []
        for edge in edges:
            # 每条边转换为圆柱（segment 是两点的坐标）
            line = trimesh.creation.cylinder(
                radius=0.01,  # 边线粗细
                segment=vertices[edge]  # 必须是 (2, 3) 的数组
            )
            bbox_lines.append(line)
        bbox_wireframe = trimesh.util.concatenate(bbox_lines)
        bbox_wireframe.visual.face_colors = color + [100] # 半透明

        # 添加到场景
        scene.add_geometry(bbox_wireframe)

    scene.export(output_path)

if __name__ == "__main__":
    scene_path = "data/HM3D/00006-HkseAnWCgqk/HkseAnWCgqk.glb"
    # scene_path = "data/HM3D/00006-HkseAnWCgqk/HkseAnWCgqk.glb"
    objects_path = "data/HM3D/00006-HkseAnWCgqk/HkseAnWCgqk.objects.pkl"
    output_path = "HkseAnWCgqk.ply"
    visualize(scene_path, objects_path, output_path)