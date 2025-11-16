import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # left_gripper, left_arm = (
        #     root["/joint_action/left_gripper"][()],
        #     root["/joint_action/left_arm"][()],
        # )
        # right_gripper, right_arm = (
        #     root["/joint_action/right_gripper"][()],
        #     root["/joint_action/right_arm"][()],
        # )
        left_gripper = root["/joint_action/gripper"][()]
        position = root["/joint_action/positions"][()]
        # vector = root["/joint_action/velocities"][()]
        pointcloud = root["/pointcloud"][()]

    # return left_gripper, left_arm, right_gripper,right_arm, vector, pointcloud
    return left_gripper, position, pointcloud


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = "/workspace/data_real/" + str(task_name) + "/" + str(task_config) + "/data"

    total_count = 0

    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    point_cloud_arrays = []
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = (
        [],
        [],
        [],
        [],
    )

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"{current_ep}.hdf5")
        (
            left_gripper_all,
            # left_arm_all,
            # right_gripper_all,
            # right_arm_all,
            position,
            pointcloud_all,
        ) = load_hdf5(load_path)

        for j in range(0, left_gripper_all.shape[0]):

            pointcloud = pointcloud_all[j]
            position_val = position[j]
            gripper_val = left_gripper_all[j]
            # 确保夹爪值是一个数组 (例如, 将标量 0.7 变为 [0.7])
            if gripper_val.ndim == 0:
                gripper_val = np.expand_dims(gripper_val, axis=0)
            joint_state = np.concatenate([position_val, gripper_val]).astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                point_cloud_arrays.append(pointcloud)
                state_arrays.append(joint_state)
            if j != 0:
                joint_action_arrays.append(joint_state)

        current_ep += 1
        total_count += left_gripper_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

    print()
    try:
        episode_ends_arrays = np.array(episode_ends_arrays)
        state_arrays = np.array(state_arrays)
        point_cloud_arrays = np.array(point_cloud_arrays)
        joint_action_arrays = np.array(joint_action_arrays)
    
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        joint_chunk_size = (100, joint_action_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1])
        zarr_data.create_dataset(
            "point_cloud",
            data=point_cloud_arrays,
            chunks=point_cloud_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "state",
            data=state_arrays,
            chunks=state_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "action",
            data=joint_action_arrays,
            chunks=joint_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends_arrays,
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
    except ZeroDivisionError as e:
        print("If you get a `ZeroDivisionError: division by zero`, check that `data/pointcloud` in the task config is set to true.")
        raise 
    except Exception as e:
        print(f"An unexpected error occurred ({type(e).__name__}): {e}")
        raise

if __name__ == "__main__":
    main()
