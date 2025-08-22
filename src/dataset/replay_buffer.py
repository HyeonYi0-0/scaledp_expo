import collections
from typing import Optional, Union, Any

import gym
import gym.spaces
import numpy as np
import os
import pickle
import glob
import re

from tqdm import tqdm
from src.dataset.mdp_dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()
    
def _slice_replay_dict(dataset_dict, start, end):
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[start:end]
    elif isinstance(dataset_dict, dict):
        return {k: _slice_replay_dict(v, start, end) for k, v in dataset_dict.items()}
    else:
        raise TypeError()

def _get_item_from_replay_dict(dataset_dict, index):
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[index]
    elif isinstance(dataset_dict, dict):
        return {k: _get_item_from_replay_dict(v, index) for k, v in dataset_dict.items()}
    else:
        raise TypeError()

def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        save_dir: str = "./data/robomimic/online/lowdim",
        save_interval: int = 1000,
        start_training: int = 5000,
    ):
        self.save_dir = save_dir
        self.save_interval = save_interval
        self._insert_count = 0
        self._start_training = start_training

        os.makedirs(self.save_dir, exist_ok=True)
        
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict, auto_save: bool = True):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        
        # auto save
        if auto_save:
            self._insert_count += 1
            if (self._start_training >= 0) and (self._insert_count >= self._start_training) and (self._insert_count % self.save_interval) == 0:
                self.save_chunk()
                
    def save_chunk(self):
        start = max(0, self._insert_index - self.save_interval)
        end = self._insert_index
        filename = os.path.join(self.save_dir, f"chunk_{end}.pkl")

        state = {
            "observations": _slice_replay_dict(self.dataset_dict["observations"], start, end),
            "next_observations": _slice_replay_dict(self.dataset_dict["next_observations"], start, end),
            "actions": self.dataset_dict["actions"][start:end],
            "rewards": self.dataset_dict["rewards"][start:end],
            "masks": self.dataset_dict["masks"][start:end],
            "dones": self.dataset_dict["dones"][start:end],
        }
        with open(filename, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_all_chunks(self, folder: str):
        """Load all chunk files from a folder (resume mode)."""
        # chunk 파일 목록 정렬 (숫자 기준 정렬)
        files = glob.glob(os.path.join(folder, "chunk_*.pkl"))
        files.sort(key=lambda x: int(re.search(r"chunk_(\d+)\.pkl", x).group(1)))

        total_loaded = 0
        for filename in files:
            with open(filename, "rb") as f:
                state = pickle.load(f)

            n = len(state["actions"])
            for i in range(n):
                data_dict = {
                    "observations": _get_item_from_replay_dict(state["observations"], i),
                    "next_observations": _get_item_from_replay_dict(state["next_observations"], i),
                    "actions": state["actions"][i],
                    "rewards": state["rewards"][i],
                    "masks": state["masks"][i],
                    "dones": state["dones"][i],
                }
                # 버퍼에 삽입
                self.insert(data_dict, auto_save=False)

            total_loaded += n
            print(f"[Online Replay Buffer] Loaded {n} transitions from {filename}")

        print(f"[Online Replay Buffer] Resume complete, total {total_loaded} transitions loaded.")
                
    def save_to_pickle(self):
        filename = os.path.join(self.save_dir, f"replay_buffer_{len(self)}.pkl")
        state = {
            "dataset_dict": self.dataset_dict,
            "_size": self._size,
            "_capacity": self._capacity,
            "_insert_index": self._insert_index,
        }
        with open(filename, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Online Replay Buffer] Saved {filename}")

    def load_from_pickle(self, filename: str):
        with open(filename, "rb") as f:
            state = pickle.load(f)
        self.dataset_dict = state["dataset_dict"]
        self._size = state["_size"]
        self._capacity = state["_capacity"]
        self._insert_index = state["_insert_index"]
        print(f"[Online Replay Buffer] Resumed from {filename}")

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # PyTorch version - simplified iterator without device placement
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
            
    def init_replay_buffer_from_lowdim_demo_data(
        self,
        demo_data: Any,
    ) -> "ReplayBuffer":    
        episode_len = len(demo_data.episode_ends)

        # Parse episode boundaries
        for idx in tqdm(range(episode_len), desc="Processing episodes"):
            episode = demo_data.get_episode(idx)
            episode_steps = len(episode["obs"])
            for t in range(episode_steps - 1):
                obs = episode["obs"][t]
                next_obs = episode["obs"][t + 1]

                action = episode["action"][t]
                if t == (episode_steps - 2):
                    reward = 1.0
                    done = True
                    mask = 0.0
                else :
                    reward = 0.0
                    done = False
                    mask = 1.0 

                # Insert into buffer
                self.insert(
                    dict(
                        observations=obs,
                        actions=action,
                        rewards=reward,
                        masks=mask,
                        dones=done,
                        next_observations=next_obs,
                    )
                )