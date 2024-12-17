import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
import wandb

import gymnasium as gym
import numpy as np
import torch

from algorithms import *
from mjc_env import *
from .util import Logger

cur_dir = Path(os.path.dirname(__file__))

AGENT = {
    'ppo': PPOAgent
}

ENV = {
    'mjc-door-open': DoorOpenEnv,
    'mjc-ball': BallBalanceEnv
}

def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.env.startswith('mjc'):
        env = ENV[args.env](render_mode="rgb_array")
    else:
        env = gym.make(args.env, continuous=True, render_mode="rgb_array")
    state, info = env.reset(seed=seed)

    args.train.obs_dim = env.observation_space.shape[0]
    args.train.act_dim = env.action_space.shape[0]
    args.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    agent = AGENT[args.agent](args.train)
    
    # init wandb logger
    wandb.init(project="RL", name=args.run, config=dict(args))
    wandb.watch(agent.model)
    # configure model checkpoint save and log dir
    log_dir = Path(cur_dir / "log" / args.run)
    os.makedirs(log_dir, exist_ok=True)
    
    agent.set_logger(log_dir = log_dir, logger = Logger())
    
    agent.train(env)
    
    env.close()
    
if __name__ == "__main__":
    config_name = sys.argv[1]
    conf = OmegaConf.load(cur_dir / "config" / f"{config_name}.yaml")
    conf.merge_with_cli()
    
    main(conf)