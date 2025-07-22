from typing import Dict
import torch
import numpy as np
import copy
import einops
from src.common.pytorch_util import dict_apply
from src.common.replay_buffer import ReplayBuffer
from src.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from src.model.common.normalizer import LinearNormalizer
from src.dataset.base_dataset import BaseImageDataset
from src.common.normalize_util import get_image_range_normalizer

import torch
import einops
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from torch.nn.utils.rnn import pad_sequence
from src.common.pytorch_util import dict_apply
from src.dataset.base_dataset import BaseImageDataset


class LiberoGoalDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
        ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'agentview', 'robotview'])
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['staticview'] = get_image_range_normalizer()
        normalizer['gripperview'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # Stack image views and normalize
        agentview = sample['agentview']
        robotview = sample['robotview']
        # images = np.stack([agentview, robotview], axis=1)  # (T, 2, H, W, C)
        # images = images.astype(np.float32) / 255.0
        staticview = agentview.astype(np.float32) / 255.0
        gripperview = robotview.astype(np.float32) / 255.0

        staticview = einops.rearrange(staticview, 't h w c -> t c h w')
        gripperview = einops.rearrange(gripperview, 't h w c -> t c h w')
        
        data = {
            'obs': {
                # 'image': images,                  # (T, 2, H, W, C)
                'staticview': staticview,
                'gripperview': gripperview,
                'agent_pos': sample['state'][:].astype(np.float32)  # (T, 2)
            },
            'action': sample['action'].astype(np.float32)  # (T, action_dim)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
