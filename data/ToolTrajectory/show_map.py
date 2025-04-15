# 将场景的导航网格可视化为自顶向下的地图
# 该脚本使用Habitat-sim的API获取自顶向下的地图，并使用matplotlib进行可视化

import numpy as np
import os
import imageio
import habitat_sim
import matplotlib.pyplot as plt
import cv2


def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


def display_map(topdown_map, key_points=None, output_path="./top_down_map.png"):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for idx, point in enumerate(key_points):
            if idx == 0:
                color  = "green"
            else:
                color = "blue"
            plt.plot(point[0], point[1], marker="o", markersize=8, color=color, alpha=0.8, label=f"point {idx}")
            # plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    # 绘制点和点之间的连线
    if key_points is not None and len(key_points) > 1:
        for i in range(len(key_points) - 1):
            plt.plot(
                [key_points[i][0], key_points[i + 1][0]],
                [key_points[i][1], key_points[i + 1][1]],
                color="red",
                linewidth=2,
            )
    plt.savefig(
        os.path.join(output_path),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show(block=False)
    return ax

if __name__ == "__main__":
    scene_file = "data/HM3D/00876-mv2HUxq3B53/mv2HUxq3B53.basis.glb"  # 替换为你的场景GLB文件路径
    navmesh_file = "data/HM3D/00876-mv2HUxq3B53/mv2HUxq3B53.basis.navmesh"  # 替换为你的导航网格文件路径
    output_path = "./"
    display = True

    # @markdown ###Configure Example Parameters:
    # @markdown Configure the map resolution:
    # 定义地图的分辨率
    meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
    # @markdown ---
    # @markdown Customize the map slice height (global y coordinate):
    custom_height = False  # @param {type:"boolean"}
    # 定义地图的高度，如果取消设置将设为地图的最低点
    height = 1  # @param {type:"slider", min:-10, max:10, step:0.1}
    # @markdown If not using custom height, default to scene lower limit.
    # @markdown (Cell output provides scene height range from bounding box for reference.)

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_file
    backend_cfg.enable_physics = False  # 不需要物理引擎

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # 从sim.pathfinder.get_bounds()[0][1]获取地图的最低点
    print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
    if not custom_height:
        # get bounding box minumum elevation for automatic height
        height = sim.pathfinder.get_bounds()[0][1]

    # 如果pathfinder已经加载
    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
        # This map is a 2D boolean array
        # 直接调用sim.pathfinder.get_topdown_view获取自顶向下视图
        # get_topdown_view的api接受两个参数，分别是分辨率meters_per_pixel和水平切片高度height；
        sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
        display_map(sim_topdown_map)

        # if display:
        #     # @markdown Alternatively, you can process the map using the Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-api/blob/master/habitat/utils/visualizations/maps.py)
        #     # 同样地可以调用habitat_lab中的maps模块中的get_topdown_map函数
        #     hablab_topdown_map = maps.get_topdown_map(
        #         sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        #     )
        #     recolor_map = np.array(
        #         [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        #     )
        #     hablab_topdown_map = recolor_map[hablab_topdown_map]
        #     print("Displaying the raw map from get_topdown_view:")
        #     display_map(sim_topdown_map)
        #     print("Displaying the map from the Habitat-Lab maps module:")
        #     display_map(hablab_topdown_map)

        #     # easily save a map to file:
        #     map_filename = os.path.join(output_path, "top_down_map.png")
        #     imageio.imsave(map_filename, hablab_topdown_map)
