import os
import numpy as np
import habitat_sim
import habitat_sim.utils.common as utils
import cv2
import json
from tqdm import tqdm
from sklearn.cluster import DBSCAN


#配置路径
SCENE_ID = "wcojb4TFT35"
BASE_DIR = "val-data/hm3d-val-habitat-v0.2/00802-wcojb4TFT35"
SCENE_PATH = os.path.join(BASE_DIR, f"{SCENE_ID}.basis.glb")
NAVMESH_PATH = os.path.join(BASE_DIR, f"{SCENE_ID}.basis.navmesh")

RGB_OUT = os.path.join(BASE_DIR, "rgb")
DEPTH_OUT = os.path.join(BASE_DIR, "depth")
POSE_OUT = os.path.join(BASE_DIR, "pose")
INTRINSIC_PATH = os.path.join(BASE_DIR, "intrinsic.json")

os.makedirs(RGB_OUT, exist_ok=True)
os.makedirs(DEPTH_OUT, exist_ok=True)
os.makedirs(POSE_OUT, exist_ok=True)

IMG_WIDTH = 640
IMG_HEIGHT = 480
XY_STEP = 2.0
ANGLES = [0, 90, 180, 270]


# ===== 工具函数 =====
def quat_to_rot_matrix(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,           2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])


def make_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE_PATH
    sim_cfg.enable_physics = False
    sim_cfg.load_semantic_mesh = False

    sensor_spec_rgb = habitat_sim.CameraSensorSpec()
    sensor_spec_rgb.uuid = "color"
    sensor_spec_rgb.resolution = [IMG_HEIGHT, IMG_WIDTH]
    sensor_spec_rgb.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec_rgb.position = [0.0, 1.0, 0.0]  # 降低相机高度，避免跨层视野

    sensor_spec_depth = habitat_sim.CameraSensorSpec()
    sensor_spec_depth.uuid = "depth"
    sensor_spec_depth.resolution = [IMG_HEIGHT, IMG_WIDTH]
    sensor_spec_depth.sensor_type = habitat_sim.SensorType.DEPTH
    sensor_spec_depth.position = [0.0, 1.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec_rgb, sensor_spec_depth]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    sim.pathfinder.load_nav_mesh(NAVMESH_PATH)
    return sim


def save_intrinsics():
    fx = fy = 320.0
    cx = 320.0
    cy = 240.0
    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    with open(INTRINSIC_PATH, 'w') as f:
        json.dump({"intrinsic_matrix": K}, f, indent=2)


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
    print(f"检测到 {len(floor_centers)} 层楼: {floor_centers}")
    return floor_centers


def sample_points_on_floor(sim, floor_y, step=0.5):
    pathfinder = sim.pathfinder
    min_pt, max_pt = pathfinder.get_bounds()
    x_vals = np.arange(min_pt[0], max_pt[0], step)
    z_vals = np.arange(min_pt[2], max_pt[2], step)

    points = []
    for x in x_vals:
        for z in z_vals:
            pt = [x, floor_y, z]
            if pathfinder.is_navigable(pt):
                points.append(pt)
    return points


def collect_data():
    sim = make_sim()
    save_intrinsics()

    floor_ys = detect_floor_heights(sim, min_y=-3.0, max_y=10.0, y_step=0.1)
    all_points = []
    for y in floor_ys:
        floor_points = sample_points_on_floor(sim, y, step=XY_STEP)
        all_points.extend(floor_points)

    print(f"📸 开始逐点采样，共 {len(all_points)} 个点，每点 {len(ANGLES)} 视角")

    frame_id = 0
    for pos in tqdm(all_points):
        for angle in ANGLES:
            agent_state = sim.agents[0].get_state()
            agent_state.position = np.array(pos)
            angle_rad = np.deg2rad(angle)
            agent_state.rotation = utils.quat_from_angle_axis(angle_rad, np.array([0, 1, 0]))
            sim.agents[0].set_state(agent_state)

            obs = sim.get_sensor_observations()
            color_img = obs["color"][:, :, :3][:, :, ::-1]
            depth_img = obs["depth"] * 1000.0
            import pdb;pdb.set_trace()
            cv2.imwrite(os.path.join(RGB_OUT, f"{frame_id:05d}.png"), color_img)
            cv2.imwrite(os.path.join(DEPTH_OUT, f"{frame_id:05d}.png"), depth_img.astype(np.uint16))

            sensor_state = sim.agents[0].get_state().sensor_states["color"]
            T = np.eye(4)
            T[:3, :3] = quat_to_rot_matrix(sensor_state.rotation)
            T[:3, 3] = sensor_state.position
            np.savetxt(os.path.join(POSE_OUT, f"{frame_id:05d}.txt"), T)

            frame_id += 1

    print(f"共生成 {frame_id} 帧 RGB-D + Pose")
    sim.close()


if __name__ == '__main__':
    collect_data()
