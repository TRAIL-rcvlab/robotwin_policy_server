import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '..'))
sys.path.append(os.path.join(parent_directory, '../..'))

from typing import Dict
import torch
import numpy as np
import copy
from tqdm import tqdm
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import pdb


class RobotDataset(BaseDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
    ):
        super().__init__()
        self.task_name = task_name
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        zarr_path = os.path.join(parent_directory, zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=["state", "action", "point_cloud"])  # 'img'
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # <--- 3. 在 __init__ 的末尾添加这一行 ---
        # self._compute_chunk_weights()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][
            :,
        ].astype(np.float32)  # (agent_posx2, block_posex3)
        point_cloud = sample["point_cloud"][
            :,
        ].astype(np.float32)  # (T, 1024, 6)

        data = {
            "obs": {
                "point_cloud": point_cloud,  # T, 1024, 6
                "agent_pos": agent_pos,  # T, D_pos
            },
            "action": sample["action"].astype(np.float32),  # T, D_action
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    # def _compute_chunk_weights(self):
    #     """
    #     遍历所有由 self.sampler 定义的有效数据块,
    #     并为包含 gripper 状态转变的块分配更高的权重。
    #     """
    #     print("[Oversampling] Calculating chunk weights for gripper transitions...")

    #     # --- 定义超参数 ---
    #     # 夹爪动作在你 action 向量中的索引
    #     GRIPPER_ACTION_INDEX = -1 
    #     # 判定为“转变”的阈值 (e.g., 0.7 -> 0.35, 变化量 > 0.1)
    #     TRANSITION_THRESHOLD = 0.1
    #     # 为包含转变的 chunk 增加的权重 (例如，提高 20 倍的学习概率)
    #     OVERSAMPLE_WEIGHT = 20.0  
        
    #     # --- 开始计算权重 ---
        
    #     # 1. 获取所有 'action' 数据
    #     # (TotalSteps, ActionDim)
    #     all_actions = self.replay_buffer['action']
        
    #     # 2. 获取 self.sampler 定义的所有有效 chunk 的起始索引
    #     # 这是连接 __getitem__(idx) 和原始数据的关键
    #     # len(self.sampler.indices) == len(self)
    #     chunk_start_indices = self.sampler.indices

    #     chunk_weights = []
        
    #     # 3. 遍历所有可能的 chunk 起始点
    #     for i in tqdm(range(len(chunk_start_indices)), desc="Calculating chunk weights"):
    #         # 获取这个 chunk 在 'all_actions' 数组中的起始和结束索引
    #         # start_idx = int(chunk_start_indices[i])
    #         start_idx_arr = chunk_start_indices[i]
            
    #         if start_idx_arr.size == 0:
    #             # This chunk is invalid (e.g., empty array), assign default weight and skip.
    #             chunk_weights.append(1.0)
    #             continue

    #         # Take the first element of the array, flatten it, and cast to int.
    #         start_idx = int(start_idx_arr.flat[0])
    #         end_idx = start_idx + self.horizon # self.horizon 是你的 chunk 长度
            
    #         # 安全检查：确保索引在边界内
    #         if end_idx > len(all_actions):
    #             # 这种情况不应该发生, 但作为保险
    #             chunk_weights.append(1.0)
    #             continue

    #         # 4. 提取这个 chunk 的 action 数据
    #         # (horizon, ActionDim)
    #         action_chunk = all_actions[start_idx:end_idx]

    #         # 5. 检查 chunk 长度是否足够计算差异
    #         if len(action_chunk) < 2:
    #             chunk_weights.append(1.0)
    #             continue
                
    #         # 6. 提取夹爪动作
    #         # (horizon,)
    #         gripper_actions = action_chunk[:, GRIPPER_ACTION_INDEX]
            
    #         # 7. 计算动作差异并检查是否存在“转变”
    #         # np.diff 计算 [a[1]-a[0], a[2]-a[1], ...]
    #         diffs = np.abs(np.diff(gripper_actions))
            
    #         if np.any(diffs > TRANSITION_THRESHOLD):
    #             # 这是一个关键 chunk，提高它的权重
    #             chunk_weights.append(OVERSAMPLE_WEIGHT)
    #         else:
    #             # 这是一个普通 chunk
    #             chunk_weights.append(1.0)

    #     # --- 8. 将权重保存为类的属性 ---
    #     self.chunk_weights = torch.tensor(chunk_weights, dtype=torch.float)
        
    #     num_transitions = sum(w > 1.0 for w in chunk_weights)
    #     print(f"[Oversampling] Weight calculation complete.")
    #     print(f"Found {num_transitions} transition chunks out of {len(chunk_weights)} total chunks.")
    #     print(f"These key chunks will be sampled {OVERSAMPLE_WEIGHT}x more often.")
