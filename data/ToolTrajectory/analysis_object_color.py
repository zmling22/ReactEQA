from multiprocessing import Pool, cpu_count
import trimesh
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm

# 全局变量（只读）
global_geometries = None

def init_worker(shared_geometries):
    global global_geometries
    global_geometries = shared_geometries

def clean_global():
    global global_geometries
    global_geometries = None

def get_faces_in_bbox(mesh, bbox):
    bbox_min, bbox_max = bbox.bounds
    face_mins = mesh.vertices[mesh.faces].min(axis=1)
    face_maxs = mesh.vertices[mesh.faces].max(axis=1)
    overlap = np.all((face_maxs >= bbox_min) & (face_mins <= bbox_max), axis=1)
    return np.where(overlap)[0].tolist()

def extract_colors_from_uv(mesh, faces_in_bbox, texture):
    if not faces_in_bbox:
        return np.empty((0, 3))
    uv_coords = mesh.visual.uv[mesh.faces[faces_in_bbox]].reshape(-1, 2)
    h, w = texture.shape[:2]
    pixel_coords = (uv_coords * [w, h]).astype(int)
    valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < w) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < h)
    return texture[pixel_coords[valid, 1], pixel_coords[valid, 0]][:, :3]

def process_item(item):
    bbox_min = np.array(item["bbox"]["min"])
    bbox_max = np.array(item["bbox"]["max"])
    bbox = trimesh.primitives.Box(extents=bbox_max - bbox_min)
    bbox.apply_translation((bbox_min + bbox_max) / 2)

    all_colors = []
    for mesh, texture in global_geometries:
        faces_in_bbox = get_faces_in_bbox(mesh, bbox)
        colors = extract_colors_from_uv(mesh, faces_in_bbox, texture)
        if len(colors) > 0:
            all_colors.append(colors)

    return np.vstack(all_colors) if all_colors else None

def preprocess_scene(scene):
    geometries = []
    for name, mesh in scene.geometry.items():
        if not hasattr(mesh.visual, 'uv') or not hasattr(mesh.visual, 'material'):
            continue
        material = mesh.visual.material
        if not hasattr(material, 'baseColorTexture') or material.baseColorTexture is None:
            continue
        if isinstance(material.baseColorTexture, Image.Image):
            texture = np.array(material.baseColorTexture)
        else:
            texture = np.array(Image.open(material.baseColorTexture))
        geometries.append((mesh, texture))
    return geometries

def get_all_colors(scene, data):
    geometries = preprocess_scene(scene)
    # 启动进程池，初始化全局共享数据
    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(geometries,)) as pool:
        results = list(tqdm(pool.imap(process_item, data), total=len(data)))
    clean_global()
    return results

if __name__ == "__main__":
    glb_path = "data/HM3D/00006-HkseAnWCgqk/HkseAnWCgqk.glb"
    object_pkl = "data/HM3D/00006-HkseAnWCgqk/HkseAnWCgqk.objects.pkl"

    scene = trimesh.load(glb_path)

    data = pickle.load(open(object_pkl, "rb"))["objects"]
    results = get_all_colors(scene, data)
    for r in results:
        print(r)