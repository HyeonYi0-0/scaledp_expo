#! /usr/bin/env python
import dmcgym
import gym
import tqdm

from src.dataset import MemoryEfficientReplayBuffer, init_replay_buffer_from_demo_data

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
from src.env_runner.base_image_runner import BaseImageRunner
from src.common.checkpoint_util import TopKCheckpointManager
from src.common.json_logger import JsonLogger
from src.common.pytorch_util import dict_apply, optimizer_to
from src.model.diffusion.ema_model import EMAModel
from src.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class finetuneScaleDPWorkspace(BaseWorkspace):
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

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        action_repeat = cfg.action_repeat

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

        # configure RL replay buffer
        replay_buffer_size = cfg.replay_buffer_size or cfg.global_max_steps // action_repeat
        replay_buffer = MemoryEfficientReplayBuffer(
            env.observation_space, env.action_space, replay_buffer_size
        )
        replay_buffer_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": cfg.batch_size * cfg.utd_ratio,
                "pack_obs_and_next_obs": True,
            }
        )
        replay_buffer.seed(cfg.seed)

        replay_buffer = init_replay_buffer_from_demo_data(
            demo_data=dataset.replay_buffer, 
            replay_buffer=replay_buffer, 
            pixel_keys=("agentview_image", "robot0_eye_in_hand_image")
        )
        
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

        # configure rollout evaluation env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure Online RL env
        env = env_runner.get_environment()
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
        env.seed(cfg.seed)

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
        agent = Expo.create_pixels(
            cfg.seed,
            env.observation_space,
            env.action_space,
            **kwargs,
        )

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
            for i in tqdm.tqdm(
                range(1, cfg.global_max_steps // action_repeat + 1),
                smoothing=0.1,
            ):
                if i < cfg.start_training:
                    action = env.action_space.sample()
                else:
                    action, agent = agent.sample_actions(observation)
                next_observation, reward, done, info = env.step(action)

                mask = 1.0 if not done or "TimeLimit.truncated" in info else 0.0

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
                    # TODO: Check Validation
                    for k, v in info["episode"].items():
                        decode = {"r": "return", "l": "length", "t": "time"}
                        wandb.log({f"agent/{decode[k]}": v}, step=i * action_repeat)

                if i >= cfg.start_training:
                    batch = next(replay_buffer_iterator)
                    agent, update_info, mini_batch = agent.update(batch, cfg.utd_ratio)
                    
                    # Base policy update (finetuning)
                    batch = dict_apply(mini_batch, lambda x: x.to(device, non_blocking=True))

                    # compute loss
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if i % cfg.training.gradient_accumulate_every == 0:
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
                        'base/train_loss': raw_loss_cpu,
                        'base/lr': lr_scheduler.get_last_lr()[0]
                    }

                    # ============ evaluate Base policy =============
                    policy = self.model
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    policy.eval()

                    if (i % cfg.training.rollout_every) == 0:   
                        runner_log = env_runner.run(policy)
                        # log all
                        step_log.update(runner_log)
                    
                    # checkpoint
                    if (i % cfg.training.checkpoint_every) == 0:
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

                        agent_ckpt_dir = os.path.join(os.path.dirname(topk_ckpt_path), "agent")

                        if not os.path.exists(agent_ckpt_dir):
                            os.makedirs(agent_ckpt_dir, exist_ok=True)
                        
                        file_name = os.path.basename(topk_ckpt_path)
                        name, _ = os.path.splitext(file_name)
                        agent_ckpt_path = os.path.join(agent_ckpt_dir, f"{name}.pt")
                        if not os.path.isfile(agent_ckpt_path): 
                            torch.save({
                                'agent_state_dict': agent.state_dict() if hasattr(agent, 'state_dict') else agent,
                                'step': i * action_repeat,
                                'optimizer_state_dict': getattr(agent, 'optimizer', {}).state_dict() if hasattr(getattr(agent, 'optimizer', {}), 'state_dict') else {}
                            }, agent_ckpt_path)

                    # ============ train Base policy =============
                    policy.train()

                    if i % cfg.log_interval == 0:
                        # replace train_loss with log_interval average
                        train_loss = np.mean(train_losses)
                        step_log['base/train_loss'] = train_loss

                        # agent logging
                        for k, v in update_info.items():
                            step_log[f"agent/{k}"] = v
                        
                        step_log = dict()
                        train_losses = list()

                        wandb_run.log(step_log, step=(i*action_repeat))
                        json_logger.log(step_log)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = finetuneScaleDPWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()