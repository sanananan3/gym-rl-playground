import os
import sys
import re
from glob import glob
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import torch
import gymnasium as gym

from algorithms import *
from mjc_env import *

cur_dir = Path(os.path.dirname(__file__))

AGENT = {
    'ppo': PPOAgent,
    'hrl' : HierarchicalPPOAgent
    
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
        env = ENV[args.env](episode_len=1000, render_mode="human")
    else:
        env = gym.make(args.env, continuous=True, render_mode="human")

    state, _ = env.reset(seed=seed)

    args.train.obs_dim = env.observation_space.shape[0]
    args.train.act_dim = env.action_space.shape[0]
    args.train.device = "cuda" if torch.cuda.is_available() else "cpu"


    # =============== load highest reward checkpoint from log dir ==================
    
    ckpt_list = glob(str(cur_dir / "log" / args.run / "*.ckpt"))
    
    if not ckpt_list :
        raise FileNotFoundError("No checkpoint file found")

    def get_return(ckpt_path):
        match = re.search(r'(\d+\.\d+).ckpt$', os.path.basename(ckpt_path))
        return float(match.group(1)) if match else float('-inf')

    best_ckpt = max(ckpt_list, key=get_return)

    print("Using best model checkpoint: ", {best_ckpt})

    # ============================================================================= 

    agent = AGENT[args.agent](args.train, ckpt_path=best_ckpt)
    agent.test(env)
    
if __name__ == "__main__":
    config_name = sys.argv[1]
    conf = OmegaConf.load(cur_dir / "config" / f"{config_name}.yaml")
    conf.merge_with_cli()
    
    main(conf)