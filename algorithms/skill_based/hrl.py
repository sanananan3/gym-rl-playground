import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions import Categorical
import scipy
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from experiments.util import Logger

FINAL_GOAL = None

"""
1. Model Structure 
"""

# return log probablity -> use ppo loss calculation 
def log_prob_from_dist(dist, act):
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

class MetaActorCritic(nn.Module):
    """
        For High Level Policy 
    """
    def __init__(self, obs_dim, goal_dim, act_mask_dim, rnn_hidden_dim=64):
        super().__init__()

        # Meta Actor
        self.rnn = nn.GRU(obs_dim+goal_dim, rnn_hidden_dim, batch_first=True)
        self.hidden = None

        self.fc_subgoal = nn.Sequential(
            nn.Linear(rnn_hidden_dim, goal_dim), nn.Tanh()          # continuous action space
        )
        self.log_std_subgoal = torch.nn.Parameter(torch.as_tensor(
            -0.5 * np.ones(goal_dim, dtype=np.float32)
        ))
        self.fc_act_mask = nn.Sequential(
            nn.Linear(rnn_hidden_dim, act_mask_dim), nn.Softmax(dim=-1)        # discrete action space
        )

        # Meta Critic
        self.fc_critic = nn.Linear(rnn_hidden_dim, 1)
        
    def get_dist_subgoal(self, rnn_hidden):
        subgoal = self.fc_subgoal(rnn_hidden)
        std_subgoal = torch.exp(self.log_std_subgoal)
        return Normal(subgoal, std_subgoal)
    
    def get_dist_act_mask(self, rnn_hidden):
        act_mask = self.fc_act_mask(rnn_hidden)
        return Categorical(act_mask)
    
    def forward(self, obs, goal): # input : observation on environment + entire goal 
        with torch.no_grad():
            # rnn_input: (batch, seq=1, rnn_hidden_dim)
            rnn_input = torch.cat([obs, goal], dim=-1).unsqueeze(1)
            rnn_output, self.hidden = self.rnn(rnn_input, self.hidden)

            dist_subgoal = self.get_dist_subgoal(self.hidden)
            subgoal = dist_subgoal.sample()
            subgoal_log_prob = log_prob_from_dist(dist_subgoal, subgoal)

            dist_act_mask = self.get_dist_act_mask(self.hidden)
            act_mask = dist_act_mask.sample()
            act_mask_log_prob = log_prob_from_dist(dist_act_mask, act_mask)

            value = self.fc_critic(rnn_output.squeeze(1))
        
        return subgoal.numpy(), act_mask.item(), subgoal_log_prob.numpy(), act_mask_log_prob.numpy(), \
                self.hidden.numpy(), value.numpy()
    
    def reset_hidden_state(self, batch_size=1):
        self.hidden = torch.zeros(1, batch_size, self.rnn.hidden_size)

class ActorCritic(nn.Module):
    """
        For Low Level Policy
    """
    def __init__(self, obs_dim, goal_dim, act_dim):
        super().__init__()
        
        self.actor = Actor(obs_dim+goal_dim, act_dim)
        self.critic = Critic(obs_dim+goal_dim)
        
    def forward(self, obs, goal): # input: observation on environment + subgoal
        with torch.no_grad():
            actor_input = torch.cat([obs, goal], dim=-1)
            dist = self.actor.get_distribution(actor_input)
            action = dist.sample()
            log_prob = log_prob_from_dist(dist, action)
            value = self.critic(actor_input)
        
        return action.numpy(), log_prob.numpy(), value.numpy()


"""
2. Training Buffer : store data for agent to train, and when done training (per 1 epoch), update model using data  
"""
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Buffer:
    def __init__(self, obs_dim, act_dim, size, gamma = 0.99, lam = 0.95):

        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)

        # For Meta Actor-Critic 
        self.logp_subgoal_buf = np.zeros(size, dtype=np.float32)
        self.logp_embsel_buf = np.zeros(size, dtype=np.float32)

        # For Actor-Critic 
        self.logp_buf = np.zeros(size, dtype=np.float32)

        # adv, ret: 각각 actor, critic 업데이트에 사용.
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        
        # size: 한 epoch당 step 수와 동일. epoch 마지막에 model 업데이트할 때 가득 찬 버퍼를 사용. 
        self.size = size
        self.ptr = 0
        self.path_start_idx = 0

        # GAE (generalized advantage estimation) 계산을 위한 파라미터
        self.gamma = gamma # discount factor 
        self.lam = lam # GAE parameter 
        
    def store(self, obs, act, rew, val, logp=None, logp_subgoal=None, logp_embsel=None):
        # store data from environmnet to buffer 
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.val_buf[idx] = val

        if logp is not None:
            self.logp_buf[idx] = logp

        if logp_subgoal is not None and logp_embsel is not None:   
            self.logp_subgoal_buf[idx] = logp_subgoal
            self.logp_embsel_buf[idx] = logp_embsel

        self.ptr += 1
        
    def finish_path(self, last_val=0):
        """
         episode (trajectory) 단위로 실행. 하나의 episode가 끝날 때마다 call되는 함수
        - actor 업데이트할 때 필요한 GAE 계산하여 adv 버퍼에 저장.
        - critic 업데이트할 때 필요한 Return = discounted sum of rewards 계산하여 ret 버퍼에 저장.
        """
        
        # define slice about current trajectory's start to finish 
         
        path_slice = slice(self.path_start_idx, self.ptr)
        # episode가 terminate된 것이 아니고 truncate됐을 때: 마지막 val != 0이므로 마지막 val을 제대로 넣어줌. 
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # 1) gae 계산 -> to use update actor 
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # 2) return (rewards-to-go) 계산
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
        
    def get(self):
        """
        epoch(여러 개의 Episode를 모아서 모델을 한 번 업데이트하는 과정)이 끝나서 이제 모델 업데이트하기 직전에 탐험 data 불러오기 위해 call되는 함수
        """
        self.ptr, self.path_start_idx = 0, 0
        
        # advantage normalization
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        
        data = dict(
            obs = self.obs_buf,
            act = self.act_buf,
            ret = self.ret_buf,
            adv = self.adv_buf,
        )
        
        if hasattr(self, 'logp_buf'):
            data['logp'] = self.logp_buf

        if hasattr(self, 'logp_subgoal_buf') and hasattr(self, 'logp_embsel_buf'):  
            data['logp_subgoal'] = self.logp_subgoal_buf
            data['logp_embsel'] = self.logp_embsel_buf

        return {k: torch.tensor(v) for k, v in data.items()}
    
    

"""
3. PPO Agent
"""
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
        
class HierarchicalPPOAgent:
    
    def __init__(self, args, ckpt_path=None):        

        # initialize policy
        self.meta_model = MetaActorCritic(args.obs_dim, args.goal_dim, args.act_mask_dim).apply(init_weights)
        self.model = ActorCritic(args.obs_dim, args.act_dim).apply(init_weights)

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # initialize buffer
        self.buf = Buffer(args.obs_dim, args.act_dim, args.max_step, args.gamma, args.lam)
        self.meta_buf = Buffer(args.obs_dim, args.goal_dim, args.max_step, args.gamma, args.lam)

        # initialize optimizer

        # 1) initalize meta policy optimizer 
        self.meta_optim = optim.Adam(self.meta_model.parameters(), lr=args.meta_lr)

        # 2) initalize low-level policy optimizer
        self.actor_optim = optim.Adam(self.model.actor.parameters(), lr=args.actor_lr)
        self.critic_optim = optim.Adam(self.model.critic.parameters(), lr=args.critic_lr)
        
        self.clip_ratio = args.clip_ratio
        self.target_kl = args.target_kl
        self.max_epoch = args.max_epoch
        self.max_step = args.max_step
        self.train_iter = args.train_iter

        self.act_dim = args.act_dim

    def set_logger(self, log_dir, logger: Logger):
        self.log_dir = log_dir
        self.logger = logger
        


    # Calculate Loss Function 

    def compute_loss_meta(self, data):

        """
        Compute PPO Loss for High-Level Policy. this function returns two outputs : Subgoal (g_LL), Embodiment Selector (e)
        1. subgoal : continuous action -> normal distribution 
        2. embodiment distribution : discrete action -> categorical distribution

        => combine 1,2 for backpropagation

        """

        # 1. data load from buffer 
        obs, subgoal, adv, logp_old_subgoal, logp_old_embsel = data['obs'], data['act'], data['adv'], data['logp_subgoal'], data['logp_embsel']
        
        # 2. run high-level policy

        dist_subgoal = self.meta_model.get_dist_subgoal(obs)
        dist_embsel = self.meta_model.get_dist_act_mask(obs)

        subgoal_sample = dist_subgoal.rsample() # continuous action -> use rsample (reparameterization trick) 
        embsel_sample = dist_embsel.sample() # discrete action -> use sample (softmax)


        # 3. calculate log probability (for PPO Loss)

        logp_subgoal = log_prob_from_dist(dist_subgoal, subgoal_sample)
        logp_embsel = log_prob_from_dist(dist_embsel, embsel_sample)

        # 4. caculate PPO loss 

        ratio_subgoal = torch.exp(logp_subgoal - logp_old_subgoal)
        ratio_embsel = torch.exp(logp_embsel - logp_old_embsel)

        clip_adv_subgoal = torch.clamp(ratio_subgoal, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        clip_embsel = torch.clamp(ratio_embsel, 1-self.clip_ratio, 1+self.clip_ratio) * adv

        loss_subgoal = -(torch.min(ratio_subgoal * adv, clip_adv_subgoal)).mean()
        loss_embsel = -(torch.min(ratio_embsel * adv, clip_embsel)).mean()

        # 5. combine two loss for backpropagation 
        meta_loss = loss_subgoal + loss_embsel

        # 6. KL Divergence 
        apporx_kl_subgoal = (logp_old_subgoal - logp_subgoal).mean().item() 
        approx_kl_emb_sel = (logp_old_embsel - logp_embsel).mean().item()

        return meta_loss, apporx_kl_subgoal, approx_kl_emb_sel
        

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
        
        return ((self.model.critic(obs) - ret)**2).mean() # MSE Loss (train to minimize difference between value (predicted by critic network) and return-to-go)
    
    def update(self):

        # epoch 단위로 업데이트 

        data = self.buf.get() # get data from buffer
        
        actor_l_old, _ = self.compute_loss_actor(data)
        critic_l_old = self.compute_loss_critic(data)
        
        # Policy Update - Actor 
        for i in range(self.train_iter):
            self.actor_optim.zero_grad()
            policy_loss, kl = self.compute_loss_actor(data)
            # Early Stopping
            if kl > 1.5 * self.target_kl: 
                break
            policy_loss.backward()
            self.actor_optim.step()
        
        # Value Update - Critic 
        for _ in range(self.train_iter):
            self.critic_optim.zero_grad()
            value_loss = self.compute_loss_critic(data)
            value_loss.backward()
            self.critic_optim.step()

        # High-Level Policy Update
        meta_data = self.meta_buf.get()
        meta_loss_old, _, _ = self.compute_loss_meta(meta_data)

        for i in range(self.train_iter):
            self.meta_optim.zero_grad()
            meta_loss, kl_subgoal, kl_emb_sel = self.compute_loss_meta(meta_data)   
            if kl_subgoal > 1.5 * self.target_kl or kl_emb_sel > 1.5 * self.target_kl:
                break
            meta_loss.backward()
            self.meta_optim.step()

        self.logger.add(StopIter=i)
        self.logger.add(LossPi=actor_l_old.item())
        self.logger.add(LossV=critic_l_old.item())
        self.logger.add(LossMeta=meta_loss_old.item())

    
    def train(self, env: gym.Env):
        max_ret = float('-INF')
        ep_ret = 0
        ep_len = 0
        subep_len = 0
        ext_rew = 0

        self.model.train()
        obs, info = env.reset()
        FINAL_GOAL = info['goal']

        # 여러 번의 에피소드로 구성된 epoch 이 max_epoch만큼 반복
        for epoch in tqdm(range(self.max_epoch), desc='epoch'):
            # Data collection
            for step in tqdm(range(self.max_step), desc='step'):
                # 새로운 subgoal로 출발해야할때: High-level policy 쿼리하여 subgoal, action mask 샘플링
                if subep_len == 0:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    goal_t = torch.as_tensor(FINAL_GOAL, dtype=torch.float32)
                    self.meta_model.reset_hidden_state()
                    subgoal, act_mask, subgoal_logp, embsel_logp, hidden, meta_val = self.meta_model(obs_t, goal_t)

                    a_mask = np.ones(self.act_dim)
                    sg_mask = np.ones_like(subgoal)
                    if act_mask == 0:
                        a_mask[3:] = 0
                        sg_mask[3:] = 0
                
                # Low-level policy로 action 샘플링
                subgoal_t = torch.as_tensor(subgoal, dtype=torch.float32)
                act, logp, val = self.model(obs_t, subgoal_t)
                
                next_obs, rew, term, trunc, info = env.step(act * a_mask)
                ext_rew += rew
                ep_ret += rew
                ep_len += 1
                subep_len += 1
                
                # 현재 subgoal 달성 또는 시간 초과 시
                dist = np.linalg.norm(next_obs - subgoal)
                if dist < 0.05 or subep_len >= 50:
                    self.meta_buf.store(obs, subgoal, ext_rew, meta_val, subgoal_logp, embsel_logp)
                    subep_len = 0
                    ext_rew = 0

                in_rew = np.linalg.norm((obs - subgoal) * sg_mask) - np.linalg.norm((next_obs - subgoal) * sg_mask)
                self.buf.store(obs, act, in_rew, val, logp)
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
                    self.meta_buf.finish_path(meta_val)
                    
                    if episode_ended:
                        self.logger.add(EpRet=ep_ret)
                        self.logger.add(EpLen=ep_len)
                    
                    obs, _ = env.reset()
                    ep_ret = 0
                    ep_len = 0

            # Model update
            self.update()
            self.buf.clear()
            self.meta_buf.clear()

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