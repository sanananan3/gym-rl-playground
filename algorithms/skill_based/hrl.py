import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.normal import Normal
import scipy
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from experiments.util import Logger

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def log_prob_from_dist(dist: Normal, act: Tensor) -> Tensor:
    return dist.log_prob(act).sum(axis=-1)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, act_dim), nn.Tanh()
        )
        self.log_std = torch.nn.Parameter(torch.as_tensor(
            -0.5 * np.ones(act_dim, dtype=np.float32)
        ))

    def get_distribution(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def forward(self, obs, act = None):
        dist = self.get_distribution(obs)
        log_prob = None
        if act is not None:
            log_prob = log_prob_from_dist(dist, act)
        return dist, log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs: Tensor):
        return torch.squeeze(self.critic(obs), -1)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)
        
    def forward(self, obs):
        with torch.no_grad():
            dist = self.actor.get_distribution(obs)
            action = dist.sample()
            log_prob = log_prob_from_dist(dist, action)
            value = self.critic(obs)
        
        return action.numpy(), log_prob.numpy(), value.numpy()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

class Buffer:
    def __init__(self, obs_dim, act_dim, size, gamma = 0.99, lam = 0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # adv, ret: 각각 actor, critic 업데이트에 사용.
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # size: 한 epoch당 step 수와 동일. epoch 마지막에 model 업데이트할 때 가득찬 버퍼를 사용. 
        self.size = size
        self.ptr = 0
        self.path_start_idx = 0
        # gae 계산을 위한 파라미터
        self.gamma = gamma
        self.lam = lam
        
    def store(self, obs, act, rew, val, logp):
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.val_buf[idx] = val
        self.logp_buf[idx] = logp
        self.ptr += 1
        
    def finish_path(self, last_val=0):
        """
        한 episode (trajectory) 끝날 때 call되는 함수
        - actor 업데이트할 때 필요한 GAE 계산하여 adv 버퍼에 저장.
        - critic 업데이트할 때 필요한 Return = discounted sum of rewards 계산하여 ret 버퍼에 저장.
        """
        
        path_slice = slice(self.path_start_idx, self.ptr)
        # episode가 terminate된 것이 아니고 truncate됐을 때: 마지막 val != 0이므로 마지막 val을 제대로 넣어줌. 
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # gae 계산
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # return (rewards-to-go) 계산
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
        
    def get(self):
        """
        epoch이 끝나서 이제 모델 업데이트하기 직전에 탐험 data 불러오기 위해 call되는 함수
        """
        assert self.ptr == self.size
        self.ptr, self.path_start_idx = 0, 0
        
        # advantage normalization
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        
        data = dict(
            obs = self.obs_buf,
            act = self.act_buf,
            ret = self.ret_buf,
            adv = self.adv_buf,
            logp = self.logp_buf
        )
        return {k: torch.tensor(v) for k, v in data.items()}
    
    
class HierarchicalPPOAgent:
    def __init__(self, args, ckpt_path=None):        
        # initialize policy
        self.meta_model = ActorCritic(args.obs_dim, args.goal_dim).apply(init_weights)
        self.model = ActorCritic(args.obs_dim, args.act_dim).apply(init_weights)
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # initialize buffer
        self.buf = Buffer(args.obs_dim, args.act_dim, args.max_step, args.gamma, args.lam)

        # initialize optimizer
        self.actor_optim = optim.Adam(self.model.actor.parameters(), lr=args.actor_lr)
        self.critic_optim = optim.Adam(self.model.critic.parameters(), lr=args.critic_lr)
        
        self.clip_ratio = args.clip_ratio
        self.target_kl = args.target_kl
        self.max_epoch = args.max_epoch
        self.max_step = args.max_step
        self.train_iter = args.train_iter

    def set_logger(self, log_dir, logger: Logger):
        self.log_dir = log_dir
        self.logger = logger
        
    def compute_loss_actor(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        _, logp = self.model.actor(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        
        loss = -(torch.min(ratio * adv, clip_adv)).mean()
        approx_kl = (logp_old - logp).mean().item()
        
        return loss, approx_kl
        
    def compute_loss_critic(self, data):
        obs, ret = data['obs'], data['ret']
        
        return ((self.model.critic(obs) - ret)**2).mean()
    
    def update(self):
        data = self.buf.get()
        
        actor_l_old, _ = self.compute_loss_actor(data)
        critic_l_old = self.compute_loss_critic(data)
        
        # Policy Update
        for i in range(self.train_iter):
            self.actor_optim.zero_grad()
            policy_loss, kl = self.compute_loss_actor(data)
            # Early Stopping
            if kl > 1.5 * self.target_kl:
                break
            policy_loss.backward()
            self.actor_optim.step()
        
        # Value Update
        for _ in range(self.train_iter):
            self.critic_optim.zero_grad()
            value_loss = self.compute_loss_critic(data)
            value_loss.backward()
            self.critic_optim.step()
            
        self.logger.add(StopIter=i)
        self.logger.add(LossPi=actor_l_old.item())
        self.logger.add(LossV=critic_l_old.item())
    
    def train(self, env: gym.Env):
        max_ret = float('-INF')
        ep_ret = 0
        ep_len = 0

        self.model.train()
        obs, _ = env.reset()

        for epoch in tqdm(range(self.max_epoch), desc='epoch'):
            # Data collection
            for step in tqdm(range(self.max_step), desc='step'):
                act, logp, val = self.model(torch.as_tensor(obs, dtype=torch.float32))
                
                next_obs, rew, term, trunc, _ = env.step(act)
                ep_ret += rew
                ep_len += 1
                
                self.buf.store(obs, act, rew, val, logp)
                obs = next_obs
                
                episode_ended = (term | trunc)
                epoch_ended = (step == self.max_step)
                # episode가 종료되었거나 강제로 잘라야 할 경우
                if episode_ended or epoch_ended:
                    if epoch_ended:
                        _, _, val = self.model(torch.as_tensor(obs, dtype=torch.float32))
                    else:
                        val = 0
                    self.buf.finish_path(val)
                    
                    if episode_ended:
                        self.logger.add(EpRet=ep_ret)
                        self.logger.add(EpLen=ep_len)
                    
                    obs, _ = env.reset()
                    ep_ret = 0
                    ep_len = 0
                    
            # Model update
            self.update()
            
            # Save model
            current_ret = np.mean(self.logger.epoch_dict['EpRet'])
            if current_ret > max_ret:
                # 기존에 저장된 체크포인트가 있다면 삭제 후 저장
                if max_ret != float('-INF'):
                    os.remove(self.log_dir / f'{max_ret}.ckpt')
                torch.save(self.model.state_dict(), self.log_dir / f'{current_ret}.ckpt')
                max_ret = current_ret

            self.logger.log('StopIter')
            self.logger.log('LossPi')
            self.logger.log('LossV')
            self.logger.log('EpRet')
            self.logger.log('EpLen')
            self.logger.flush()

    def test(self, env: gym.Env):
        term = trunc = False        

        self.model.eval()
        obs, _ = env.reset()

        while not (term or trunc):
            # sample deterministic action
            act, logp, val = self.model(torch.as_tensor(obs, dtype=torch.float32))

            obs, _, term, trunc, _ = env.step(act)
            env.render()