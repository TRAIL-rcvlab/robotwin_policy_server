import numpy as np
import torch
import hydra
import dill
import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))

sys.path.append('/workspace/third_party/gello_software_bp')


from hydra import initialize, compose
from datetime import datetime
from omegaconf import OmegaConf
from dp3_policy import DP3


from dataclasses import dataclass
from typing import Optional, Tuple

import tyro
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera
import torch
from collections import deque
from PIL import Image
import cv2
import time


def rgb_depth_to_pointcloud(
    rgb_bgr: np.ndarray,          # (H, W, 3), uint8, BGR order (as in your code)
    depth_raw: np.ndarray,          # (H, W), float32, in meters
    fx: float, fy: float,
    cx: float, cy: float,
    depth_scale: float = 1.0,
    k_points: int = 1024,
    min_depth: float = 0.0,
    max_depth: float = float('inf'),
    seed: int = 42
) -> np.ndarray:  # (k_points, 6) → [x,y,z,r,g,b], rgb ∈ [0,1]
    """
    Convert aligned RGB (BGR uint8) and depth (meters) to downsampled colored point cloud.
    """
    depth_m = depth_raw * depth_scale
    # 检查 depth_m 是否有3个维度，并且最后一个维度是1
    if depth_m.ndim == 3 and depth_m.shape[2] == 1:
        # 压缩掉最后一个维度，使其从 (H, W, 1) 变为 (H, W)
        depth_m = np.squeeze(depth_m, axis=2)

    H, W = depth_m.shape  # 现在 depth_m.shape 应该是 (H, W)

    if rgb_bgr.shape[:2] != (H, W):
        # Resize depth to match RGB if needed (or vice versa)
        # In your case, you likely ensure they match, but we resize depth to RGB size
        depth_m = cv2.resize(depth_m, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Step 1: Filter depth by valid range
    depth_valid = depth_m.copy()
    if np.isfinite(min_depth):
        depth_valid[depth_valid < min_depth] = 0.0
    if np.isfinite(max_depth):
        depth_valid[depth_valid > max_depth] = 0.0

    # Step 2: Project to 3D
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    z = depth_valid.astype(np.float32)
    valid = np.isfinite(z) & (z > 0.0)  # (H, W)

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (H*W, 3)
    valid_flat = valid.reshape(-1)

    # Step 3: Get RGB (convert BGR → RGB, then to [0,1])
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8 → RGB
    rgb_flat = rgb_rgb.reshape(-1, 3).astype(np.float32) / 255.0  # (H*W, 3), [0,1]

    # Step 4: Keep only valid points
    xyz_valid = xyz[valid_flat]
    rgb_valid = rgb_flat[valid_flat]

    # Step 5: Downsample to k_points
    rng = np.random.default_rng(seed)
    N = xyz_valid.shape[0]
    if N == 0:
        return np.zeros((k_points, 6), dtype=np.float32)
    if N >= k_points:
        idx = rng.choice(N, size=k_points, replace=False)
    else:
        extra = rng.choice(N, size=k_points - N, replace=True)
        idx = np.concatenate([np.arange(N), extra])
        rng.shuffle(idx)
    xyz_k = xyz_valid[idx]
    rgb_k = rgb_valid[idx]

    # Step 6: Concatenate → (K, 6)
    pc = np.concatenate([xyz_k, rgb_k], axis=1).astype(np.float32)  # (K, 6)
    return pc

def encode_obs(observation):
    depth_scale = 0.0002500000118743628
    fx, fy, cx, cy = 601.6535034179688, 601.79345703125, 325.8245849609375, 237.5863494873047
    k_points=1024
    obs = dict()

    # print(observation["base_depth"].min(), observation["base_depth"].max())
    depth_mm = observation["base_depth"]
    rgb_bgr = observation["base_rgb"]  # (H, W, 3), uint8, BGR

    # 生成点云
    pointcloud = rgb_depth_to_pointcloud(
        rgb_bgr=rgb_bgr,
        depth_raw=depth_mm,
        fx=fx, fy=fy, cx=cx, cy=cy,
        depth_scale=depth_scale,
        k_points=1024,
        min_depth=0.1,   # 可选：过滤太近的噪声
        max_depth=2.0,   # 可选：过滤太远的点
        seed=42
    )  # shape: (1024, 6)

    # observation["base_rgb"] = observation["base_rgb"][:,:,[2,1,0]] # RGB to BGR
    obs['point_cloud'] = pointcloud
    
    position = observation["joint_positions"].astype(np.float32)
    if position[-1] > 0.5:
        position[-1] = 1.0  
    else:
        position[-1] = 0.0

    obs["agent_pos"] = position
    return obs

def get_model(usr_args): 
    # ckpt_file = f"{usr_args['ckpt_path']}/{usr_args['checkpoint_num']}.ckpt"
    config_path = "./3D-Diffusion-Policy/diffusion_policy_3d/config"
    config_name = f"{usr_args['config_name']}.yaml"
    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"
    
    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.policy.use_pc_color = usr_args['use_rgb']
    OmegaConf.set_struct(cfg, True)

    return DP3(cfg, usr_args)

def reset_model(model):
    model.env_runner.reset_obs()

def resize_img(image, size=(320,240)):
    # print(image.shape)
    image = Image.fromarray(image)
    image = np.array(image.resize(size, Image.BILINEAR))
    # image = np.transpose(np.array(image), (1,2,0))
    # print(image.shape)
    return image 

@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5001
    base_camera_port: int = 5000
    hostname: str = "10.27.50.231" # 主要修改这个
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "/home/landau/gello_software/bc_data"
    task_name: str = 'default' 
    bimanual: bool = False
    verbose: bool = False

def main(args):
    import yaml
    yaml_file = 'deploy_policy.yml'  # 可以是相对路径或绝对路径
    with open(yaml_file, 'r', encoding='utf-8') as file:
        usr_args = yaml.safe_load(file)  # 使用 safe_load 更安全
    model = get_model(usr_args)

    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)
    count = 0
    reset_model(model)

    # inference loop
    while True: 
        observation = env.get_obs()
        # show_image = cv2.cvtColor(observation['base_rgb'], cv2.COLOR_RGB2BGR)
        # success = cv2.imwrite(f'data/outputs/img/saved_image{count}.jpg', show_image)
        count+=1
        obs = encode_obs(observation)
        actions = model.get_action(obs)
        for act in actions:
            import copy
            ori_g = copy.deepcopy(act[-1])
            act[-1] = 0.0 if act[-1] > 0.5 else 1.0 # 0.12/0.7 are min./max. experimental gripper joint values
            print(f"{ori_g=}, gripper : {act[-1]}, {observation['joint_positions'][-1]}, {observation['gripper_position']}");
            # act[-1] = 0.0
            env.step(act)
            observation = env.get_obs()
            obs = encode_obs(observation)
            model.update_obs(obs)

# def main(args):
#     import h5py
#     import yaml
#     yaml_file = 'deploy_policy.yml'  # 可以是相对路径或绝对路径
#     with open(yaml_file, 'r', encoding='utf-8') as file:
#         usr_args = yaml.safe_load(file)  # 使用 safe_load 更安全
    
#     print("正在加载模型...")
#     model = get_model(usr_args)

#     # --- [MODIFICATION START] ---
#     # 移除机器人/ZMQ/Env，从 HDF5 加载
    
#     # 1. --- 在此处配置你要测试的文件和帧 ---
#     HDF5_FILE_PATH = '/workspace/data_real/move_banana_to_box_dp3/demo_clean/data/0.hdf5' # 替换为你的 HDF5 文件路径
#     FRAME_INDEX = 300  # 替换为你想测试的帧编号
#     # ------------------------------------

#     print(f"正在从 {HDF5_FILE_PATH} 加载第 {FRAME_INDEX} 帧...")
    
#     # 2. --- 从 HDF5 加载数据以构建 observation 字典 ---
#     observation = {}
#     try:
#         with h5py.File(HDF5_FILE_PATH, 'r') as f:
#             # 加载 RGB (JPEG 字节)
#             rgb_bytes = f['/observation/front_camera/rgb'][FRAME_INDEX]
#             # 从内存缓冲区解码
#             rgb_np_array = np.frombuffer(rgb_bytes, dtype=np.uint8)
#             # cv2.imdecode 默认读取为 BGR，这符合 encode_obs 的预期
#             observation['base_rgb'] = cv2.imdecode(rgb_np_array, cv2.IMREAD_COLOR) 
            
#             # 加载 Depth
#             observation['base_depth'] = f['/observation/front_camera/depth_raw'][FRAME_INDEX]
            
#             # 加载 Velocities (这是你的模型 'agent_pos' 期望的输入, 7-dim)
#             observation['joint_positions'] = f['/joint_action/velocities'][FRAME_INDEX]

#     except Exception as e:
#         print(f"!! 严重错误: 从 HDF5 加载数据失败: {e}")
#         print("!! 请检查 HDF5_FILE_PATH, FRAME_INDEX, 和 HDF5 数据集键 (keys) 是否正确。")
#         return

#     print("HDF5 数据加载成功:")
#     print(f"  base_rgb shape: {observation['base_rgb'].shape}")
#     print(f"  base_depth shape: {observation['base_depth'].shape}")
#     print(f"  joint_positions: {observation['joint_positions']}")

#     # 3. --- 在单帧上运行模型 ---
    
#     # 重置模型状态 (例如，对于 diffusion, 这是必须的)
#     reset_model(model) 

#     print("\n正在调用 encode_obs(observation)...")
#     obs = encode_obs(observation)
#     print(f"  obs['agent_pos'] (已处理): {obs['agent_pos']}")
    
#     print("正在调用 model.get_action(obs)...")
#     actions = model.get_action(obs)
    
#     print(f"\n--- 🚀 模型预测完成 ---")
#     print(f"预测的动作块 (Action Chunk) shape: {actions.shape}")

#     first_predicted_action = actions[0]
#     model_gripper_output = first_predicted_action[-1] # 模型输出 (接近 0.0 或 1.0)
    
#     if model_gripper_output > 0.5: # 模型想要 "1.0" (张开)
#         final_gripper_cmd = 0.12
#         gripper_decision = f"(Open) (模型原始输出: {model_gripper_output:.4f} -> 映射到 {final_gripper_cmd})"
#     else: # 模型想要 "0.0" (闭合)
#         final_gripper_cmd = 0.7
#         gripper_decision = f"(Close) (模型原始输出: {model_gripper_output:.4f} -> 映射到 {final_gripper_cmd})"

#     print(f"\n第一个预测的动作 (Raw): {first_predicted_action}")
#     print(f"  - 预测的 7-DoF 速度: {first_predicted_action[:-1]}")
#     print(f"  - 预测的夹爪指令: {gripper_decision}")

if __name__ == '__main__':
    main(tyro.cli(Args))