#! /usr/bin/env python
import gym
import tqdm

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from src.workspace.base_workspace import BaseWorkspace
from src.policy.ScaleDP_lowdim_policy import ScaleDiffusionTransformerLowdimPolicy
from src.policy.EXPO import Expo
from src.policy.fast_EXPO import fast_Expo, fast_Lowdim_Expo
from src.dataset.base_dataset import BaseLowdimDataset
from src.env_runner.base_lowdim_runner import BaseLowdimRunner 
from src.common.checkpoint_util import TopKCheckpointManager
from src.common.json_logger import JsonLogger
from src.common.pytorch_util import dict_apply, optimizer_to
from src.model.diffusion.ema_model import EMAModel
from src.model.common.lr_scheduler import get_scheduler
from src.dataset import MemoryEfficientReplayBuffer, ReplayBuffer

import time

OmegaConf.register_new_resolver("eval", eval, replace=True)

class finetuneScaleDPLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: ScaleDiffusionTransformerLowdimPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: ScaleDiffusionTransformerLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        
        if cfg.training.debug:
            cfg.start_training = 0
            cfg.global_max_steps = 1
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.utd_ratio = 10
            cfg.log_interval = 1
            cfg.online_save_dir = cfg.offline_save_dir
            cfg.online_buffer_file_name = cfg.offline_buffer_file_name

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        action_repeat = cfg.action_repeat

        # resume training
        if cfg.training.resume:
            # lastest_ckpt_path = self.get_checkpoint_path()
            resume_ckpt_path = pathlib.Path(cfg.training.resume, cfg.training.ckpt_name)
            if resume_ckpt_path.is_file():
                print(f"Resuming from checkpoint {resume_ckpt_path}")
                self.load_checkpoint(path=resume_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        
        # configure rollout evaluation env
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure Online RL env
        env = env_runner.get_environment()
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
        env.seed(cfg.seed)
        # configure RL replay buffer
        offline_replay_buffer_size = cfg.replay_buffer_size
        offline_replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, offline_replay_buffer_size,
            start_training=-1, save_dir=cfg.offline_save_dir
        )
        offline_replay_buffer_iterator = offline_replay_buffer.get_iterator(
            sample_args={
                "batch_size": (cfg.batch_size // 2) * cfg.utd_ratio,
            }
        )
        offline_replay_buffer.seed(cfg.seed)
        file_path = os.path.join(offline_replay_buffer.save_dir, cfg.offline_buffer_file_name)
        if os.path.isfile(file_path):
            offline_replay_buffer.load_from_pickle(file_path)
            print(f"Loaded offline replay buffer from {file_path}")
        else:
            print(f"Offline replay buffer not found at {file_path}, starting from scratch.")
            offline_replay_buffer.init_replay_buffer_from_lowdim_demo_data(
                demo_data=dataset.replay_buffer, 
            )
            offline_replay_buffer.save_to_pickle()
        print(f"Replay buffer size: {len(offline_replay_buffer)}")
        
        online_replay_buffer_size = cfg.replay_buffer_size if cfg.training.debug else cfg.global_max_steps // action_repeat
        online_replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, online_replay_buffer_size,
            start_training=cfg.start_training, save_dir=cfg.online_save_dir, save_interval=cfg.save_interval
        )
        online_replay_buffer_iterator = online_replay_buffer.get_iterator(
            sample_args={
                "batch_size": (cfg.batch_size // 2) * cfg.utd_ratio,
            }
        )
        online_replay_buffer.seed(cfg.seed+1)
        # if file exists
        file_path = os.path.join(online_replay_buffer.save_dir, cfg.online_buffer_file_name)
        if os.path.isfile(file_path):
            online_replay_buffer.load_from_pickle(file_path)
            print(f"Loaded online replay buffer from {file_path}")
        else:
            print(f"Online replay buffer not found at {file_path}, starting from scratch.")
            # online_replay_buffer.init_replay_buffer_from_lowdim_demo_data(
            #     demo_data=dataset.replay_buffer, 
            # )
        print(f"Replay buffer size: {len(online_replay_buffer)}")
        
        # Set normalizer
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(cfg.global_max_steps // action_repeat),
            num_cycles=cfg.training.num_cycles,
            last_epoch=-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            
        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure Agent (Edit, on-the-fly)
        kwargs = dict(cfg.agent)
        kwargs.pop('ckpt_name')
        agent = fast_Lowdim_Expo.create(
            seed=cfg.seed,
            observation_space=env.observation_space,
            action_space=env.action_space,
            **kwargs,
        )
        
        start_step = 0
        if cfg.training.resume:
            agent_ckpt_path = pathlib.Path(cfg.training.resume, cfg.agent.ckpt_name)
            if os.path.isfile(agent_ckpt_path):
                print(f"Loading agent from {agent_ckpt_path}")
                checkpoint = torch.load(agent_ckpt_path, map_location=cfg.agent.device)
                start_step = checkpoint.get('step', 0)
                print("start step: ", start_step)
                agent.load_state_dict(checkpoint['agent_state_dict'])
            else:
                print(f"Agent checkpoint not found at {agent_ckpt_path}, starting from scratch.")

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # training loop
        observation, done = env.reset(), False
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            step_log = dict()
            train_losses = list()
            for step_t in tqdm.tqdm(
                range(start_step, cfg.global_max_steps // action_repeat),
                smoothing=0.1,
            ):
                policy = self.model if cfg.training.use_ema else self.ema_model
                policy.eval() 
                action, agent = agent.on_the_fly(base_policy=policy, obs=observation)
                # Reshape action from (1, 1, 7) to (1, 7)
                if isinstance(action, np.ndarray) and action.ndim == 3:
                    action = action.squeeze(0)  # Remove the middle dimension
                policy.train()
                
                next_observation, reward, done, info = env.step(action)
                
                mask = 1.0 if not done or "TimeLimit.truncated" in info else 0.0

                online_replay_buffer.insert(
                    dict(
                        observations=observation,
                        actions=action,
                        rewards=reward,
                        masks=mask,
                        dones=done,
                        next_observations=next_observation,
                    )
                )
                observation = next_observation

                if done:
                    observation, done = env.reset(), False

                if step_t >= cfg.start_training:
                    online_batch = next(online_replay_buffer_iterator)
                    offline_batch = next(offline_replay_buffer_iterator)
                    policy = self.model if cfg.training.use_ema else self.ema_model
                    policy.eval()
                    # start_time = time.time()
                    agent, mini_batch, update_info = agent.update(policy, 
                                                                  online_data=online_batch, 
                                                                  offline_data=offline_batch, 
                                                                  utd_ratio=cfg.utd_ratio)
                    # end_time = time.time()
                    # print(f"Agent update time: {end_time - start_time:.4f} seconds")
                    policy.train()
                    
                    # Base policy update (finetuning)
                    # batch = dict_apply(mini_batch, lambda x: torch.from_numpy(x).to(device, non_blocking=True))

                    # compute loss
                    raw_loss = self.model.compute_loss(mini_batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if step_t % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'base_lr': lr_scheduler.get_last_lr()[0],
                        "step": step_t * action_repeat,
                    }

                    # ============ evaluate Base policy =============
                    policy = self.model
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    policy.eval()

                    if (step_t % cfg.training.rollout_every) == 0:   
                        runner_log = env_runner.run(policy)
                        # log all
                        step_log.update(runner_log)
                    
                    # checkpoint
                    if (step_t % cfg.training.checkpoint_every) == 0:
                        # checkpointing
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint(agent=agent, step_log=step_log)
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value
                        
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(agent=agent, step_log=step_log, path=topk_ckpt_path)

                    # ============ train Base policy =============
                    policy.train()

                    if step_t % cfg.log_interval == 0:
                        # replace train_loss with log_interval average
                        train_loss = np.mean(train_losses)
                        step_log['base_train_loss'] = train_loss

                        # agent logging
                        for k, v in update_info.items():
                            if isinstance(v, dict):
                                for sub_k, sub_v in v.items():
                                    step_log[f"agent_{k}_{sub_k}"] = sub_v
                            else:
                                step_log[f"agent_{k}"] = v
                        
                        for k, v in step_log.items():
                            if v is None:
                                print(f"Warning: {k} is None, removing from step_log")
                                step_log.pop(k)

                        wandb_run.log(step_log)
                        json_logger.log(step_log)
                        
                        step_log = dict()
                        train_losses = list()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = finetuneScaleDPLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()