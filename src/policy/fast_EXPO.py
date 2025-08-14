"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Any, Union, Type
import copy

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from src.distributions import TanhNormal
from src.model import MLP, Ensemble, StateActionValue, subsample_ensemble
from src.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import src.model.vision.crop_randomizer as dmvc
from src.common.pytorch_util import dict_apply, replace_submodules
from src.real_world.real_inference_util import get_real_obs_dict
from tqdm import tqdm

DataType = Union[np.ndarray, torch.Tensor, Dict[str, "DataType"]]
DatasetDict = Dict[str, DataType]

def _sample_actions(actor_network, observations: torch.Tensor, actions: torch.Tensor, clip_beta: float = 0.1) -> torch.Tensor:
    with torch.no_grad():
        dist = actor_network(observations, actions)
        actions = dist.sample()
    if clip_beta is not None and clip_beta > 0:
        actions = torch.clamp(actions, -clip_beta, clip_beta)
    return actions, dist

def _eval_actions(actor_network, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():   
        # Use the actor network's get_mode method for deterministic actions
        if hasattr(actor_network, 'get_mode'):
            actions = actor_network.get_mode(observations)
        else:
            # Fallback: use mean of the distribution
            dist = actor_network(observations, actions)
            if hasattr(dist, 'mode'):
                actions = dist.mode()
            else:
                # For Independent distributions, get the mean
                actions = dist.base_dist.loc
                if hasattr(actor_network, 'squash_tanh') and actor_network.squash_tanh:
                    actions = torch.tanh(actions)
    return actions

def to_tensor(v, device):
    return torch.as_tensor(v, dtype=torch.float32).pin_memory().to(device=device, non_blocking=True)

def _prepare_batch(batch, base_np, edit_np, next_base_np, next_edit_np, i, batch_size, device):
    start_idx = batch_size * i
    end_idx   = batch_size * (i + 1)

    return {
        "observations": {k: to_tensor(v[start_idx:end_idx], device) for k, v in edit_np.items()},
        "next_observations": {k: to_tensor(v[start_idx:end_idx], device) for k, v in next_edit_np.items()},
        "base_obs": {k: to_tensor(v[start_idx:end_idx], device) for k, v in base_np.items()},
        "next_base_obs": {k: to_tensor(v[start_idx:end_idx], device) for k, v in next_base_np.items()},
        "actions": to_tensor(batch["actions"][i], device),
        "rewards": to_tensor(batch["rewards"][i], device),
        "masks":   to_tensor(batch["masks"][i], device)
    }

class Agent:
    def __init__(self, actor, device='cpu'):
        self.actor = actor
        self.device = device

    def eval_actions(self, observations: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, 'Agent']:
        obs_tensor = torch.from_numpy(observations).float().to(self.device) 
        actions_tensor = torch.from_numpy(actions).float().to(self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        actions = _eval_actions(self.actor, obs_tensor, actions_tensor)
        return actions.cpu().numpy(), self

    def sample_actions(self, observations: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, 'Agent']:
        obs_tensor = torch.from_numpy(observations).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).float().to(self.device)
        if obs_tensor.dim() == 1:   
            obs_tensor = obs_tensor.unsqueeze(0)
        actions, _ = _sample_actions(self.actor, obs_tensor, actions_tensor)
        return actions.cpu().numpy(), self

class EditPolicy(nn.Module):
    def __init__(self, base_cls, input_dim, **kwargs):
        super().__init__()
        # Extract hidden_dims from kwargs to avoid duplication
        hidden_dims = kwargs.pop('hidden_dims', [256, 256, 256])
        
        # Create the base network with proper input dimension
        self.base_network = base_cls(hidden_dims=[input_dim] + list(hidden_dims), **kwargs)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        inputs = torch.cat([observations, actions], dim=-1)
        outputs = self.base_network(inputs, *args, **kwargs)
        return outputs

class Temperature(nn.Module):
    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(initial_temperature)))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

class fast_Expo(Agent):
    def __init__(self,
                 shape_meta, 
                 actor, 
                 critic, 
                 target_critic, 
                 actor_obs_encoder,
                 critic_obs_encoder,
                 temp, 
                 optimizer_actor, 
                 optimizer_critic, 
                 optimizer_temp,
                 clip_beta: float,
                 n_samples: int,
                 tau: float, 
                 discount: float, 
                 target_entropy: float,
                 num_qs: int = 2,
                 num_min_qs: Optional[int] = None, 
                 backup_entropy: bool = True, 
                 device='cpu'):
        
        super().__init__(actor, device)
        self.shape_meta = shape_meta
        self.actor_obs_encoder = actor_obs_encoder
        self.critic_obs_encoder = critic_obs_encoder
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.optimizer_temp = optimizer_temp
        self.tau = tau
        self.discount = discount
        self.target_entropy = target_entropy
        self.num_qs = num_qs
        self.num_min_qs = num_min_qs or num_qs
        self.backup_entropy = backup_entropy
        self.clip_beta = clip_beta
        self.n_samples = n_samples
        
    def _get_observation_encoder(shape_meta: dict,
                                 crop_shape: Optional[Tuple[int, int]] = (76, 76),
                                 obs_encoder_group_norm=False,
                                 eval_fixed_crop=False,
                                 task_name: str = 'square',
                                 dataset_type: str = 'ph',
                                 device: str = 'cpu'
                                ) -> Optional[nn.Module]:
        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            _type = attr.get('type', 'low_dim')
            if _type == 'rgb':
                obs_config['rgb'].append(key)
            elif _type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {_type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name=task_name,
            dataset_type=dataset_type)

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config
            # config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet50Conv"
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device=device,
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )
            
        return obs_encoder
        
    @classmethod
    def create_pixels(cls,
               seed: int,
               shape_meta: dict,
               observation_space: gym.Space,
               action_space: gym.Space,
               edit_policy_lr: float = 3e-4,
               critic_lr: float = 3e-4,
               temp_lr: float = 3e-4,
               hidden_dims: Sequence[int] = (256, 256, 256),
               discount: float = 0.99,
               crop_shape: Optional[Tuple[int, int]] = (76, 76),
               obs_encoder_group_norm=False,
               eval_fixed_crop=False,
               task_name: str = 'square',
               dataset_type: str = 'ph', 
               clip_beta: float = 0.1,
               n_samples: int = 8,
               tau: float = 0.005,
               num_qs: int = 2,
               num_min_qs: Optional[int] = None,
               critic_dropout_rate: Optional[float] = None,
               critic_layer_norm: bool = False,
               target_entropy: Optional[float] = None,
               init_temperature: float = 1.0,
               backup_entropy: bool = True,
               device: str = 'cpu'):
        """
        PyTorch implementation of Soft-Actor-Critic
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        action_dim = action_space.shape[-1]

        if target_entropy is None:
            target_entropy = -action_dim / 2
            
        # Create obs encoder
        actor_obs_encoder = cls._get_observation_encoder(shape_meta=shape_meta,
                                                   crop_shape=crop_shape,
                                                   obs_encoder_group_norm=obs_encoder_group_norm,
                                                   eval_fixed_crop=eval_fixed_crop,
                                                   task_name=task_name,
                                                   dataset_type=dataset_type,
                                                   device=device)
        critic_obs_encoder = copy.deepcopy(actor_obs_encoder)

        # Create actor network  
        obs_feature_dim = actor_obs_encoder.output_shape()[0]
        actor_hidden_dims = list(hidden_dims)
        actor_base_cls = partial(MLP, hidden_dims=actor_hidden_dims, activate_final=True)
        actor_base_cls = partial(EditPolicy, base_cls=actor_base_cls, input_dim=obs_feature_dim + action_dim)
        actor = TanhNormal(actor_base_cls, action_dim).to(device)
        optim_groups = [{'params': actor.parameters()}, {'params': actor_obs_encoder.parameters()}]
        optimizer_actor = optim.Adam(optim_groups, lr=edit_policy_lr)

        # Create critic ensemble
        critic_hidden_dims = list(hidden_dims)
        critic_base_cls = partial(MLP, 
                                activate_final=True,
                                dropout_rate=critic_dropout_rate,
                                use_layer_norm=critic_layer_norm)
        
        # Create critic ensemble using Ensemble wrapper
        critic = Ensemble(
            StateActionValue,
            num=num_qs,
            base_cls=critic_base_cls,
            input_dim=obs_feature_dim + action_dim,
            hidden_dims=critic_hidden_dims
        ).to(device)

        # Create target critic (deep copy)
        target_critic = copy.deepcopy(critic)

        # Create optimizer
        optim_groups = [{'params': critic.parameters()}, {'params': critic_obs_encoder.parameters()}]
        optimizer_critic = optim.Adam(optim_groups, lr=critic_lr)

        # Create temperature
        temp = Temperature(init_temperature).to(device)
        optimizer_temp = optim.Adam(temp.parameters(), lr=temp_lr)

        return cls(
                  shape_meta=shape_meta,
                  actor=actor, 
                  critic=critic,
                  target_critic=target_critic,
                  actor_obs_encoder=actor_obs_encoder,
                  critic_obs_encoder=critic_obs_encoder,
                  temp=temp,
                  optimizer_actor=optimizer_actor,
                  optimizer_critic=optimizer_critic,
                  optimizer_temp=optimizer_temp,
                  clip_beta=clip_beta,
                  n_samples=n_samples,
                  tau=tau,
                  discount=discount,
                  target_entropy=target_entropy,
                  num_qs=num_qs,
                  num_min_qs=num_min_qs,
                  backup_entropy=backup_entropy,
                  device=device)

    # @classmethod
    # def create(cls,
    #            seed: int,
    #            observation_space: gym.Space,
    #            action_space: gym.Space,
    #            edit_policy_lr: float = 3e-4,
    #            critic_lr: float = 3e-4,
    #            temp_lr: float = 3e-4,
    #            hidden_dims: Sequence[int] = (256, 256, 256),
    #            discount: float = 0.99,
    #            tau: float = 0.005,
    #            num_qs: int = 2,
    #            num_min_qs: Optional[int] = None,
    #            critic_dropout_rate: Optional[float] = None,
    #            critic_layer_norm: bool = False,
    #            target_entropy: Optional[float] = None,
    #            init_temperature: float = 1.0,
    #            backup_entropy: bool = True,
    #            device: str = 'cpu'):
    #     """
    #     PyTorch implementation of Soft-Actor-Critic
    #     """
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)

    #     action_dim = action_space.shape[-1]
    #     obs_dim = observation_space.shape[-1]

    #     if target_entropy is None:
    #         target_entropy = -action_dim / 2

    #     # Create actor network  
    #     actor_hidden_dims = [obs_dim] + list(hidden_dims)
    #     actor_base_cls = partial(MLP, hidden_dims=actor_hidden_dims, activate_final=True)
    #     actor = TanhNormal(actor_base_cls, action_dim).to(device)
    #     optimizer_actor = optim.Adam(actor.parameters(), lr=edit_policy_lr)

    #     # Create critic ensemble
    #     critic_hidden_dims = list(hidden_dims)
    #     critic_base_cls = partial(MLP, 
    #                             activate_final=True,
    #                             dropout_rate=critic_dropout_rate,
    #                             use_layer_norm=critic_layer_norm)
        
    #     critic_networks = []
    #     for _ in range(num_qs):
    #         critic_net = StateActionValue(critic_base_cls, 
    #                                     input_dim=obs_dim + action_dim,
    #                                     hidden_dims=critic_hidden_dims).to(device)
    #         critic_networks.append(critic_net)
        
    #     critic = nn.ModuleList(critic_networks)
        
    #     # Create target critic (deep copy)
    #     target_critic = copy.deepcopy(critic)
        
    #     # Create optimizers
    #     critic_params = []
    #     for net in critic:
    #         critic_params.extend(net.parameters())
    #     optimizer_critic = optim.Adam(critic_params, lr=critic_lr)

    #     # Create temperature
    #     temp = Temperature(init_temperature).to(device)
    #     optimizer_temp = optim.Adam(temp.parameters(), lr=temp_lr)

    #     return cls(actor=actor, 
    #               critic=critic,
    #               target_critic=target_critic,
    #               temp=temp,
    #               optimizer_actor=optimizer_actor,
    #               optimizer_critic=optimizer_critic,
    #               optimizer_temp=optimizer_temp,
    #               tau=tau,
    #               discount=discount,
    #               target_entropy=target_entropy,
    #               num_qs=num_qs,
    #               num_min_qs=num_min_qs,
    #               backup_entropy=backup_entropy,
    #               device=device)
    
    @torch.inference_mode()
    def on_the_fly(self, base_policy, obs, sub_obs=None):
        # 1) obs의 value가 numpy인 경우
        if sub_obs is None:
            np_obs_dict, edit_obs_dict = get_real_obs_dict(env_obs=obs, shape_meta=self.shape_meta)
            base_obs = {k: to_tensor(v, device=self.device) for k, v in np_obs_dict.items()}
            edit_obs = {k: to_tensor(v, device=self.device) for k, v in edit_obs_dict.items()}
        else:
            base_obs = obs
            edit_obs = sub_obs
        B, To = next(iter(base_obs.values())).shape[:2]

        # 2) base_policy 예측을 배치 차원으로 확장
        base_obs = dict_apply(base_obs, lambda x: x.repeat((self.n_samples), *[1]*(x.dim() - 1)))  # [n_samples*B, To, obs_dim]
        action_base = base_policy.predict_action(base_obs)['action']  # [n_samples*B, To, action_dim]

        # 3) actor 샘플도 벡터화
        edit_obs = dict_apply(edit_obs, lambda x: x.repeat((self.n_samples), *[1] * (x.dim() - 1)))  # [n_samples*B, To, obs_dim]
        obs_feat = self.actor_obs_encoder(edit_obs).reshape(self.n_samples, B, To, -1) # [n_samples, B, To, obs_feat_dim]
        base_action = action_base.reshape(self.n_samples, B, To, -1)  # [n_samples, B, To, action_dim]
        
        # delta_actions = []
        # for feature, action in zip(obs_feat, base_action):
        #     delta_action, _ = _sample_actions(self.actor, feature, action, clip_beta=self.clip_beta)
        #     delta_actions.append(delta_action)
        # delta_actions = torch.stack(delta_actions, dim=0).reshape(self.n_samples*B, To, -1)  # [n_samples*B, To, action_dim]
        delta_actions, _ = _sample_actions(self.actor, obs_feat, base_action, clip_beta=self.clip_beta)
        delta_actions = delta_actions.reshape(self.n_samples*B, To, -1)
        edited_actions = action_base + delta_actions  # shape: [n_samples*B, To, action_dim]
        actions = torch.cat([edited_actions, action_base], dim=0)  # [2*n_samples*B, To, action_dim]

        # 4) critic 평가를 한번에
        encoded_obs = self.critic_obs_encoder(edit_obs).unsqueeze(1).repeat(1, To, 1).repeat(2, 1, 1) # [n_samples*B, To, obs_feat_dim]
        critics = subsample_ensemble(self.target_critic.networks, self.num_min_qs, self.num_qs, device=self.device)
        qs = torch.stack([net(encoded_obs, actions) for net in critics], dim=0)
        target_Q, _ = qs.min(dim=0)
        
        # target_Q = target_Q.reshape(self.n_samples*2, B, -1)
        # best_idx = target_Q.mean(dim=1).argmax()
        target_Q = target_Q.reshape(B, self.n_samples*2, -1) # [B, 2*n_samples, 1]
        best_idx = target_Q.squeeze(-1).argmax(dim=1)  # (B,)

        # actions = actions.reshape(self.n_samples*2, B, To, -1)
        # best_action = actions[best_idx]
        actions = actions.reshape(B, self.n_samples*2, To, -1) # [B, 2*n_samples, To, action_dim]
        best_action = actions[torch.arange(B, device=self.device), best_idx]

        # best_dist = dists[(best_idx%self.n_samples)]
        
        best = None
        eps = 1e-6
        if sub_obs is None:
            best = np.clip(best_action.cpu().numpy(), -1.0+eps, 1.0-eps)
        else:
            best = torch.clamp(best_action, -1.0+eps, 1.0-eps)

        return best, self

    def update_actor(self, batch: DatasetDict) -> Tuple['fast_Expo', Dict[str, float]]:
        observations = batch["observations"]
        actions = batch["actions"]
        base_obs_np = batch["base_obs"]
        B, To = next(iter(base_obs_np.values())).shape[:2]
        
        self.optimizer_actor.zero_grad()
        
        # Sample actions from current policy
        obs_feature = self.actor_obs_encoder(observations)
        obs_feature = obs_feature.reshape(B, To, -1)
        
        edit_actions, dist = _sample_actions(self.actor, obs_feature, actions, clip_beta=self.clip_beta)
        
        log_probs = dist.log_prob(edit_actions)
        
        # Get Q-values from all critics
        q_values = []
        encoded_obs = self.critic_obs_encoder(observations).reshape(B, To, -1)
        for critic_net in self.critic.networks:
            q_val = critic_net(encoded_obs, (actions + edit_actions))  
            q_values.append(q_val)
        
        # Average Q-values
        q_values = torch.stack(q_values, dim=0).mean(dim=0)
        
        # Actor loss: maximize Q - temperature * log_prob
        temperature = self.temp()
        
        actor_loss = (temperature * log_probs - q_values).mean()
        
        actor_loss.backward()
        
        # Gradient clipping to prevent explosion
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self.actor_obs_encoder.parameters(), max_norm=1.0)
        
        self.optimizer_actor.step()
        
        return self, {
            "edit_loss": actor_loss.item(),
            "entropy": -log_probs.mean().item(),
        }

    def update_temperature(self, entropy: float) -> Tuple['fast_Expo', Dict[str, float]]:
        self.optimizer_temp.zero_grad()
        
        temperature = self.temp()
        temp_loss = temperature * (entropy - self.target_entropy)
        
        temp_loss.backward()
        self.optimizer_temp.step()
        
        return self, {
            "temperature": temperature.item(),
            "temperature_loss": temp_loss.item()
        }

    def update_critic(self, policy, batch: DatasetDict) -> Tuple['fast_Expo', Dict[str, float]]:
        observations = batch["observations"]
        next_observations = batch["next_observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        masks = batch["masks"]
        next_base_obs = batch["next_base_obs"]
        
        B, To = next(iter(next_base_obs.values())).shape[:2]
        
        self.optimizer_critic.zero_grad()
        
        with torch.no_grad():
            # Sample next actions from policy
            next_actions, _ = self.on_the_fly(base_policy=policy, obs=next_base_obs, sub_obs=next_observations)
            
            # Get target Q-values (use subset for REDQ if specified)
            target_q_values = []
            critics_to_use = subsample_ensemble(self.target_critic.networks, self.num_min_qs, self.num_qs, device=self.device)
            encoded_next_obs = self.critic_obs_encoder(next_observations).reshape(B, To, -1)
            for target_critic_net in critics_to_use:
                target_q = target_critic_net(encoded_next_obs, next_actions)
                target_q_values.append(target_q)
                
            # target_q_values = torch.stack([
            #     critic(encoded_next_obs, next_actions)
            #     for critic in critics_to_use
            # ], dim=0)

            target_q = torch.stack(target_q_values, dim=0).min(dim=0)[0]
            # target_q = target_q_values.min(dim=0)[0]
            
            # Compute target
            rewards = rewards.view(B, 1)
            masks = masks.view(B, 1)
            target_q = rewards + self.discount * masks * target_q
            
            # if self.backup_entropy:
            #     # Mistake: next_dist should be the distribution of edit actions(a^) not next_actions(a+a^)
            #     # + Expo does not use backup entropy
            #     # Backup entropy must be False for this to work correctly
            #     # TODO: Fix this in the future
            #     next_log_probs = next_dist.log_prob(next_actions)
            #     temperature = self.temp()
            #     target_q -= self.discount * masks * temperature * next_log_probs
        
        # Compute critic loss
        critic_losses = []
        q_predictions = []
        encoded_obs = self.critic_obs_encoder(observations).reshape(B, To, -1)
        for critic_net in self.critic.networks:
            q_pred = critic_net(encoded_obs, actions)
            critic_loss = F.mse_loss(q_pred, target_q)
            critic_losses.append(critic_loss)
            q_predictions.append(q_pred)
        
        total_critic_loss = sum(critic_losses)
        
        total_critic_loss.backward()
        
        # Gradient clipping for critic networks
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self.critic_obs_encoder.parameters(), max_norm=1.0)
        
        self.optimizer_critic.step()
        
        # Soft update target networks
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                # target_param.data.copy_((1 - self.tau) * param.data + self.tau * target_param.data)

        return self, {
            "critic_loss": total_critic_loss.item(),
            "q": torch.stack(q_predictions, dim=0).mean().item()
        }
        
    def merge_batch(self, online_data, offline_data, batch_size=128, utd_ratio=20):
        merged_batch = {}

        def merge(a, b):
            a_reshaped = a.reshape(utd_ratio, batch_size, *a.shape[1:])
            b_reshaped = b.reshape(utd_ratio, batch_size, *b.shape[1:])
            return np.stack([a_reshaped, b_reshaped], axis=1).reshape(utd_ratio, batch_size * 2, *a.shape[1:])

        for k, v in online_data.items():
            if isinstance(v, dict):
                merged_batch[k] = {sub_k: merge(v[sub_k], offline_data[k][sub_k]) for sub_k in v.keys()}
            else:
                merged_batch[k] = merge(v, offline_data[k])

        return merged_batch
    
    

    def update(self, policy, online_data: DatasetDict, offline_data: DatasetDict, utd_ratio: int):
        new_agent = self
        
        assert len(online_data["actions"]) % utd_ratio == 0, "Online data length must be divisible by utd_ratio"

        batch_size = len(online_data["actions"]) // utd_ratio
        batch = self.merge_batch(online_data, offline_data, batch_size=batch_size, utd_ratio=utd_ratio)

        batch["observations"] = dict_apply(batch["observations"], lambda x: x.reshape(-1, *x.shape[2:]))
        batch["next_observations"] = dict_apply(batch["next_observations"], lambda x: x.reshape(-1, *x.shape[2:]))

        base_np, edit_np = get_real_obs_dict(env_obs=batch["observations"], shape_meta=self.shape_meta)
        next_base_np, next_edit_np = get_real_obs_dict(env_obs=batch["next_observations"], shape_meta=self.shape_meta)

        # observations = {k: to_tensor(v, self.device) for k, v in edit_np.items()}

        # stream_transfer = torch.cuda.Stream(device=self.device)
        # stream_compute = torch.cuda.current_stream(device=self.device)

        # GPU에 미리 올린 데이터 저장 공간
        # gpu_batches = [None] * utd_ratio

        batch_size *= 2
        # 1. 먼저 첫 번째 배치를 전송 시작
        # with torch.cuda.stream(stream_transfer):
        #     gpu_batches[0] = _prepare_batch(batch, base_np, edit_np, next_base_np, next_edit_np, 0, batch_size, self.device)

        for i in range(utd_ratio):
            start_idx = batch_size * i
            end_idx = batch_size * (i + 1)

            mini_batch = {
                "observations": {k: to_tensor(v[start_idx:end_idx], self.device) for k, v in edit_np.items()},
                "next_observations": {k: to_tensor(v[start_idx:end_idx], self.device) for k, v in next_edit_np.items()},
                "base_obs": {k: to_tensor(v[start_idx:end_idx], self.device) for k, v in base_np.items()},
                "next_base_obs": {k: to_tensor(v[start_idx:end_idx], self.device) for k, v in next_base_np.items()},
                "actions": to_tensor(batch['actions'][i], self.device),
                "rewards": to_tensor(batch["rewards"][i], self.device),
                "masks": to_tensor(batch["masks"][i], self.device)
            }
            
            # mini_batch["actions"] = to_tensor(batch['actions'][i], self.device)
            # mini_batch["rewards"] = to_tensor(batch["rewards"][i], self.device)
            # mini_batch["masks"] = to_tensor(batch["masks"][i], self.device)
            
            # if i + 1 < utd_ratio:
            #     with torch.cuda.stream(stream_transfer):
            #         gpu_batches[i+1] = _prepare_batch(batch, base_np, edit_np, next_base_np, next_edit_np, i+1, batch_size, self.device)

            # 현재 batch로 연산 실행 (기본 stream)
            # stream_compute.wait_stream(stream_transfer)
            # mini_batch = gpu_batches[i]
            # mini_batch = _prepare_batch(batch, base_np, edit_np, next_base_np, next_edit_np, i, batch_size, self.device)

            new_agent, critic_info = new_agent.update_critic(policy, mini_batch)
        
        # Update actor once
        new_agent, actor_info = new_agent.update_actor(mini_batch)
        # Update temperature once
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        
        batch = {
            "obs": mini_batch["base_obs"],
            "action": mini_batch["actions"],
        }

        return new_agent, batch, {**actor_info, **critic_info, **temp_info}

    def state_dict(self):
        """Return state dict for saving model"""
        return {
            'actor': self.actor.state_dict(),
            'critic': [net.state_dict() for net in self.critic.networks],
            'target_critic': [net.state_dict() for net in self.target_critic.networks],
            'temp': self.temp.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'optimizer_temp': self.optimizer_temp.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.actor.load_state_dict(state_dict['actor'])
        for i, net_state in enumerate(state_dict['critic']):
            self.critic.networks[i].load_state_dict(net_state)
        for i, net_state in enumerate(state_dict['target_critic']):
            self.target_critic.networks[i].load_state_dict(net_state)
        self.temp.load_state_dict(state_dict['temp'])
        self.optimizer_actor.load_state_dict(state_dict['optimizer_actor'])
        self.optimizer_critic.load_state_dict(state_dict['optimizer_critic'])
        self.optimizer_temp.load_state_dict(state_dict['optimizer_temp'])