import copy
from typing import Iterable, Optional, Tuple, Any, Dict

import gym
import os
import pickle
import numpy as np
from gym.spaces import Box
from tqdm import tqdm

from src.dataset.mdp_dataset import DatasetDict, _sample
from src.dataset.replay_buffer import ReplayBuffer


class MemoryEfficientReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        save_dir: str = "./data/robomimic/online/lowdim",
        save_interval: int = 1000,
        start_training: int = 5000,
    ):
        self.pixel_keys = pixel_keys

        observation_space = copy.deepcopy(observation_space)
        self._num_stack = None
        for pixel_key in self.pixel_keys:
            pixel_obs_space = observation_space.spaces[pixel_key]
            if self._num_stack is None:
                self._num_stack = pixel_obs_space.shape[1]
                print(f"Number of stacked frames for {pixel_key}: {self._num_stack}")
            else:
                assert self._num_stack == pixel_obs_space.shape[1]
            self._unstacked_dim_size = pixel_obs_space.shape[-2]
            low = pixel_obs_space.low[..., 0, :, :]
            high = pixel_obs_space.high[..., 0, :, :]
            unstacked_pixel_obs_space = Box(
                low=low, high=high, dtype=pixel_obs_space.dtype
            )
            observation_space.spaces[pixel_key] = unstacked_pixel_obs_space

        next_observation_space_dict = copy.deepcopy(observation_space.spaces)
        for pixel_key in self.pixel_keys:
            next_observation_space_dict.pop(pixel_key)
        next_observation_space = gym.spaces.Dict(next_observation_space_dict)

        self._first = True
        self._is_correct_index = np.full(capacity, False, dtype=bool)

        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space=next_observation_space,
            save_dir=save_dir,
            save_interval=save_interval,
            start_training=start_training,
        )

    def insert(self, data_dict: DatasetDict):
        self._insert_count += 1
        
        if self._insert_index == 0 and self._capacity == len(self) and not self._first:
            indxs = np.arange(len(self) - self._num_stack, len(self))
            for indx in indxs:
                element = super().sample(1, indx=indx)
                self._is_correct_index[self._insert_index] = False
                super().insert(element, auto_save=False)

        data_dict = data_dict.copy()
        
        data_dict["observations"] = data_dict["observations"].copy()
        data_dict["next_observations"] = data_dict["next_observations"].copy()
        
        for pixel_key in self.pixel_keys:
            if data_dict["observations"][pixel_key].shape[-1] != self._num_stack:
                data_dict["observations"][pixel_key] = np.moveaxis(data_dict["observations"][pixel_key], 1, -1)
                # print(data_dict["observations"][pixel_key].shape)
            if data_dict["next_observations"][pixel_key].shape[-1] != self._num_stack:
                data_dict["next_observations"][pixel_key] = np.moveaxis(data_dict["next_observations"][pixel_key], 1, -1)
                # print(data_dict["next_observations"][pixel_key].shape)

        obs_pixels = {}
        next_obs_pixels = {}
        for pixel_key in self.pixel_keys:
            obs_pixels[pixel_key] = data_dict["observations"].pop(pixel_key)
            next_obs_pixels[pixel_key] = data_dict["next_observations"].pop(pixel_key)

        if self._first:
            for i in range(self._num_stack):
                for pixel_key in self.pixel_keys:
                    data_dict["observations"][pixel_key] = obs_pixels[pixel_key][..., i]
                    # print(f"Stacking {pixel_key} at index {self._insert_index}: {data_dict['observations'][pixel_key].shape}")

                self._is_correct_index[self._insert_index] = False
                super().insert(data_dict, auto_save=False)

        for pixel_key in self.pixel_keys:
            data_dict["observations"][pixel_key] = next_obs_pixels[pixel_key][..., -1]

        self._first = data_dict["dones"]

        self._is_correct_index[self._insert_index] = True
        super().insert(data_dict, auto_save=False)

        for i in range(self._num_stack):
            indx = (self._insert_index + i) % len(self)
            self._is_correct_index[indx] = False
        
        # auto save
        if (self._start_training >= 0) and (self._insert_count >= self._start_training) and (self._insert_count % self.save_interval) == 0:
            self.save_to_pickle()
            
    def save_to_pickle(self):
        filename = os.path.join(self.save_dir, f"replay_buffer_{len(self)}.pkl")
        state = {
            "dataset_dict": self.dataset_dict,
            "_size": self._size,
            "_capacity": self._capacity,
            "_insert_index": self._insert_index,
            "_is_correct_index": self._is_correct_index,
            "_first": self._first,
            "pixel_keys": self.pixel_keys,
            "_num_stack": self._num_stack,
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
        self._is_correct_index = state["_is_correct_index"]
        self._first = state["_first"]
        self.pixel_keys = state["pixel_keys"]
        self._num_stack = state["_num_stack"]
        print(f"[Online Replay Buffer] Resumed from {filename}")

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        pack_obs_and_next_obs: bool = False,
    ) -> Dict[str, Any]:
        """Samples from the replay buffer.

        Args:
            batch_size: Minibatch size.
            keys: Keys to sample.
            indx: Take indices instead of sampling.
            pack_obs_and_next_obs: whether to pack img and next_img into one image.
                It's useful when they have overlapping frames.

        Returns:
            A frozen dictionary.
        """

        if indx is None:
            if hasattr(self.np_random, "integers"):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

            for i in range(batch_size):
                while not self._is_correct_index[indx[i]]:
                    if hasattr(self.np_random, "integers"):
                        indx[i] = self.np_random.integers(len(self))
                    else:
                        indx[i] = self.np_random.randint(len(self))
        else:
            raise NotImplementedError()

        if keys is None:
            keys = self.dataset_dict.keys()
        else:
            assert "observations" in keys

        keys = list(keys)
        keys.remove("observations")

        batch = super().sample(batch_size, keys, indx)
        batch = dict(batch)  # Convert to mutable dict

        obs_keys = self.dataset_dict["observations"].keys()
        obs_keys = list(obs_keys)
        for pixel_key in self.pixel_keys:
            obs_keys.remove(pixel_key)

        batch["observations"] = {}
        for k in obs_keys:
            batch["observations"][k] = _sample(
                self.dataset_dict["observations"][k], indx
            )

        for pixel_key in self.pixel_keys:
            obs_pixels = self.dataset_dict["observations"][pixel_key]
            obs_pixels = np.lib.stride_tricks.sliding_window_view(
                obs_pixels, self._num_stack + 1, axis=0
            )
            obs_pixels = obs_pixels[indx - self._num_stack]

            if pack_obs_and_next_obs:
                batch["observations"][pixel_key] = obs_pixels
            else:
                batch["observations"][pixel_key] = obs_pixels[..., :-1]
                if "next_observations" in keys:
                    batch["next_observations"][pixel_key] = obs_pixels[..., 1:]

        return batch

    def init_replay_buffer_from_demo_data(
        self,
        demo_data: Any,
    ) -> "MemoryEfficientReplayBuffer":
        # Load zarr dataset
        root = demo_data
        data = root["/data"]
        meta = root["/meta"]
        episode_ends = meta["episode_ends"][:]

        # Parse episode boundaries
        start_idx = 0
        for end_idx in tqdm(episode_ends, desc="Processing episodes"):
            for t in range(start_idx, end_idx-1):  # skip last index (no next_obs)
                obs = {
                    "robot0_eef_pos": data["robot0_eef_pos"][t],
                    "robot0_eef_quat": data["robot0_eef_quat"][t],
                    "robot0_gripper_qpos": data["robot0_gripper_qpos"][t],
                }
                next_obs = {
                    "robot0_eef_pos": data["robot0_eef_pos"][t + 1],
                    "robot0_eef_quat": data["robot0_eef_quat"][t + 1],
                    "robot0_gripper_qpos": data["robot0_gripper_qpos"][t + 1],
                }

                for key in self.pixel_keys:
                    obs[key] = data[key][t]
                    next_obs[key] = data[key][t + 1]

                action = data["action"][t]
                if t == (end_idx-2) :
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

            start_idx = end_idx