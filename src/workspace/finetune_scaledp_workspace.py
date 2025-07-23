#! /usr/bin/env python
import dmcgym
import gym
import tqdm

from src.dataset import MemoryEfficientReplayBuffer, ReplayBuffer

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
from src.policy.ScaleDP_hybrid_image_policy import ScaleDiffusionTransformerHybridImagePolicy
from src.policy.EXPO import Expo
from src.dataset.base_dataset import BaseImageDataset
# from src.env_runner.base_image_runner import BaseImageRunner
from src.common.checkpoint_util import TopKCheckpointManager
from src.common.json_logger import JsonLogger
from src.common.pytorch_util import dict_apply, optimizer_to
from src.model.diffusion.ema_model import EMAModel
from src.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

PLANET_ACTION_REPEAT = {
    "cartpole-swingup-v0": 8,
    "reacher-easy-v0": 4,
    "cheetah-run-v0": 4,
    "finger-spi-n-0": 2,
    "ball_in_cup-catch-v0": 4,
    "walker-walk-v0": 2,
}

class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: ScaleDiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: ScaleDiffusionTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        # train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        
        # TODO: Initialize replay buffer with offline dataset

        # # configure validation dataset
        # val_dataset = dataset.get_validation_dataset()
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(cfg.max_steps // action_repeat),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)

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
        
        action_repeat = cfg.action_repeat or PLANET_ACTION_REPEAT.get(cfg.env_name, 2)

        def wrap(env):
            if "quadruped" in cfg.env_name:
                camera_id = 2
            else:
                camera_id = 0
            
            return env, None
            # return wrap_pixels(
            #     env,
            #     action_repeat=action_repeat,
            #     image_size=cfg.image_size,
            #     num_stack=cfg.num_stack,
            #     camera_id=camera_id,
            # )

        env = gym.make(cfg.env_name)
        # env, pixel_keys = wrap(env)
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
        if cfg.save_video:
            # TODO: 
            # env = WANDBVideo(env)
            pass
        env.seed(cfg.seed)

        eval_env = gym.make(cfg.env_name)
        eval_env, _ = wrap(eval_env)
        eval_env.seed(cfg.seed + 42)

        kwargs = dict(cfg.agent)
        model_cls = kwargs.pop("model_cls")
        agent = Expo.create(
            cfg.seed,
            env.observation_space,
            env.action_space,
            # pixel_keys=pixel_keys,
            **kwargs,
        )
        
        replay_buffer_size = cfg.replay_buffer_size or cfg.max_steps // action_repeat
        if cfg.memory_efficient_replay_buffer:
            replay_buffer = MemoryEfficientReplayBuffer(
                env.observation_space, env.action_space, replay_buffer_size
            )
            replay_buffer_iterator = replay_buffer.get_iterator(
                sample_args={
                    "batch_size": cfg.batch_size * cfg.utd_ratio,
                    "pack_obs_and_next_obs": True,
                }
            )
        else:
            replay_buffer = ReplayBuffer(
                env.observation_space, env.action_space, replay_buffer_size
            )
            replay_buffer_iterator = replay_buffer.get_iterator(
                sample_args={
                    "batch_size": cfg.batch_size * cfg.utd_ratio,
                }
            )

        replay_buffer.seed(cfg.seed)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        
        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        observation, done = env.reset(), False
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            step_log = dict()
            train_losses = list()
            for i in tqdm.tqdm(
                range(1, cfg.max_steps // action_repeat + 1),
                smoothing=0.1,
                disable=not cfg.tqdm,
            ):
                if i < cfg.start_training:
                    action = env.action_space.sample()
                else:
                    action, agent = agent.sample_actions(observation)
                next_observation, reward, done, info = env.step(action)

                if not done or "TimeLimit.truncated" in info:
                    mask = 1.0
                else:
                    mask = 0.0

                replay_buffer.insert(
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
                    for k, v in info["episode"].items():
                        decode = {"r": "return", "l": "length", "t": "time"}
                        wandb.log({f"training/{decode[k]}": v}, step=i * action_repeat)

                if i >= cfg.start_training:
                    batch = next(replay_buffer_iterator)
                    agent, update_info, mini_batch = agent.update(batch, cfg.utd_ratio)
                    
                    batch = dict_apply(mini_batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
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
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }

                    if i % cfg.log_interval == 0:
                        # replace train_loss with log_interval average
                        train_loss = np.mean(train_losses)
                        step_log['train_loss'] = train_loss
                        for k, v in update_info.items():
                            wandb.log({f"training/{k}": v}, step=i * action_repeat)
                        
                        step_log = dict()
                        train_losses = list()

                if i % cfg.eval_interval == 0:   
                    # TODO:                  
                    # eval_info = evaluate(
                    #     agent,
                    #     eval_env,
                    #     num_episodes=cfg.eval_episodes,
                    #     save_video=cfg.save_video,
                    # )
                    # for k, v in eval_info.items():
                    #     wandb.log({f"evaluation/{k}": v}, step=i * action_repeat)

                    if cfg.save_dir is not None:
                        checkpoint_path = os.path.join(cfg.save_dir, f"checkpoint_{i * action_repeat}.pt")
                        torch.save({
                            'agent_state_dict': agent.state_dict() if hasattr(agent, 'state_dict') else agent,
                            'step': i * action_repeat,
                            'optimizer_state_dict': getattr(agent, 'optimizer', {}).state_dict() if hasattr(getattr(agent, 'optimizer', {}), 'state_dict') else {}
                        }, checkpoint_path)
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()