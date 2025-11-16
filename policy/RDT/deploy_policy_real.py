import numpy as np
import torch

import sys, os
from model import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append('/workspace/third_party/gello_software_bp')



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
import random

# intrcution list
INSTRUCTION_STR = [
    "Move the banana to the box.",
    "Place the banana inside the box",
    "Put the banana in the box",
    "Deposit the banana into the box"
]

def get_random_instruction_path(directory_path):
    """
    从指定目录中随机选择一个文件，并返回其完整路径。

    参数:
    directory_path (str or Path): 要搜索的目录路径。

    返回:
    str: 一个随机选择的文件的完整路径，如果目录为空或不存在则返回 None。
    """
    try:
        dir_path = Path(directory_path)

        # 检查路径是否存在并且是一个目录
        if not dir_path.is_dir():
            print(f"错误: 路径 '{dir_path}' 不是一个有效的目录。")
            return None

        # 列出目录中的所有文件（排除了子目录）
        # 我们假设指令文件都是.pt文件，就像截图中显示的那样
        files = [f for f in dir_path.glob('*.pt') if f.is_file()]

        if not files:
            print(f"错误: 在 '{dir_path}' 中没有找到 .pt 文件。")
            return None

        # 随机选择一个文件
        random_file_path = random.choice(files)

        # 绝对路径
        return str(random_file_path.resolve())

    except Exception as e:
        print(f"发生意外错误: {e}")
        return None

def encode_obs(observation):
    obs = {}
    obs["head_camera"] = observation["base_rgb"]
    position = observation["joint_positions"].astype(np.float32)
    if position[-1] > 0.5:
        position[-1] = 1.0  
    else:
        position[-1] = 0.0

    obs["agent_pos"] = position
    return obs

def fill_obs(obs, left_arm, left_gripper, black_img):
    # 填充状态
    obs["right_camera"] = black_img
    obs["left_camera"] = black_img
    obs["agent_pos"] = np.concatenate([left_arm, left_gripper, obs["agent_pos"]], axis=0)
    return obs

def get_model(usr_args): 
    model_name = usr_args["ckpt_setting"]
    checkpoint_id = usr_args["checkpoint_id"]
    left_arm_dim, right_arm_dim, rdt_step = (
        usr_args["left_arm_dim"],   # 7
        usr_args["right_arm_dim"],
        usr_args["rdt_step"],
    )
    rdt = RDT(
        os.path.join(
            parent_directory,
            f"checkpoints/{model_name}/checkpoint-{checkpoint_id}/pytorch_model/mp_rank_00_model_states.pt",
        ),
        usr_args["task_name"],
        left_arm_dim,
        right_arm_dim,
        rdt_step,
    )
    return rdt

def reset_model(model):
    model.reset_obsrvationwindows()

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

    lang_dir = f"/workspace/policy/RDT/processed_data/{usr_args['task_name']}-{usr_args['task_config']}-{usr_args['expert_data_num']}/episode_0/instructions"

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

    # 构造虚拟左臂
    left_arm = np.zeros(7, dtype=np.float32)
    left_gripper = np.array([0.0], dtype=np.float32)
    # 虚拟手腕图像：生成纯黑图像
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)  # 黑色 640x480

    # inference loop
    while True: 
        observation = env.get_obs()
        # show_image = cv2.cvtColor(observation['base_rgb'], cv2.COLOR_RGB2BGR)
        # success = cv2.imwrite(f'nihao/saved_image{count}.jpg', show_image)
        count+=1
        obs = encode_obs(observation)
        obs = fill_obs(obs, left_arm, left_gripper, black_img)
        # instruction = get_random_instruction_path(lang_dir)
        input_rgb_arr, input_state = [
            obs["head_camera"],
            obs["right_camera"],
            obs["left_camera"],
        ], obs["agent_pos"]

        if (model.observation_window
                is None):  # Force an update of the observation at the first frame to avoid an empty observation window
            instruction_str = random.choice(INSTRUCTION_STR)
            model.set_language_instruction(instruction_str)
            model.update_observation_window(input_rgb_arr, input_state)

        actions = model.get_action()[:model.rdt_step, :]  # Get Action according to observation chunk
        for act in actions:
            print(f"gripper : {act[-1]}");
            act[-1] = 0 if act[-1] > 0.5 else 1.0 # 0.12/0.7 are min./max. experimental gripper joint values
            env.step(act[-8:])
            observation = env.get_obs()
            obs = encode_obs(observation)
            obs = fill_obs(obs, left_arm, left_gripper, black_img)
            input_rgb_arr, input_state = [
                obs["head_camera"],
                obs["right_camera"],
                obs["left_camera"],
            ], obs["agent_pos"]
            model.update_observation_window(input_rgb_arr, input_state)

if __name__ == '__main__':
    main(tyro.cli(Args))