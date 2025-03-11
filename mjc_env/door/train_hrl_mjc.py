from time import time 
from collections import deque, OrderedDict
import random 
import numpy as np
import argparse 
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium import make 

import torch 
from torch.utils.tensorboard import SummaryWriter

import sys 
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from mjc_env.door.door_open_env import FrankaDoorEnv
from utils.utils import *
from utils.args import * 
from utils.logging import logger
from rl.PPO import PPO, MetaPolicy, Policy, RolloutStorage, AsyncRolloutStorage



def main(): 

    parser = argparse.ArgumentParser()
    add_ppo_args(parser)
    add_env_args(parser)
    add_common_args(parser)
    add_hrl_args(parser)
    args = parser.parse_args()

    ckpt_folder, ckpt_path, start_epoch, start_env_step, summary_folder, log_file = \
        set_up_experiment_folder(args.experiment_folder, args.checkpoint_index) # ckpt/logdir , -1 
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    logger.add_filehandler(log_file)

    if not args.eval_only:  
        writer = SummaryWriter(log_dir=summary_folder)
    else:
        writer = None # when eval only, tensorboard writer is not needed


    num_envs = 1

    def make_env():
        def _init():
            env = FrankaDoorEnv(episode_len=5000, render_mode="human")
            return env 
        return _init


    train_env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    print("Number of environments:", train_env.num_envs)  # 병렬 환경 개수 확인

    eval_env = FrankaDoorEnv(episode_len=5000, render_mode="human")

    logger.info(train_env.observation_space)
    logger.info(train_env.action_space)

   

    action_dim = train_env.action_space.shape[0]

    action_mask_choices = torch.zeros(2, action_dim)
    action_mask_choices[0, 0:2] = 1.0
    action_mask_choices[1, :] = 1.0 

    print("[INFO] action dim : ", action_dim) # 7 
    print("[INFO] action mask choices : ", action_mask_choices)
    print("[INFO] observation space : ", train_env.observation_space)

    meta_observation_space = train_env.observation_space
    
    sensor_space = train_env.observation_space.spaces["sensor"] # Box(3,)

    subgoal_space = gym.spaces.Box(low=-2.0, high=2.0, shape=sensor_space.shape, dtype=np.float32)
    subgoal_mask_choices = torch.zeros(2, sensor_space.shape[0])
    subgoal_mask_choices[0,0:2] = 1.0
    subgoal_mask_choices[1, :] = 1.0

    rollout_observation_space = train_env.observation_space.spaces.copy()
    rollout_observation_space["subgoal"] = subgoal_space 
    rollout_observation_space["subgoal_mask"] = gym.spaces.Box(low=0, high=1, shape=subgoal_space.shape, dtype = np.float32)
    rollout_observation_space["action_mask"] = gym.spaces.Box(low=0, high=1, shape=(action_dim,), dtype = np.float32)
    rollout_observation_space = gym.spaces.Dict(rollout_observation_space)
    
    observation_space = rollout_observation_space 

    sensor_normalizer = np.array([[-3. , -3., 0], [3., 6., 1.3]])

    sensor_magnitude = (sensor_normalizer[1]-sensor_normalizer[0]) /2.0
    min_stddev = np.array(args.subgoal_min_std_dev) / sensor_magnitude 
    initial_stddev = np.array(args.subgoal_init_std_dev) / sensor_magnitude 
    subgoal_tolerance = torch.tensor(min_stddev, dtype=torch.float32)

    device = torch.device("cpu")


    # 1. high-level poilicy's network and agent 
    meta_actor_critic = MetaPolicy(
        observation_space = meta_observation_space,
        subgoal_space = subgoal_space,
        use_action_masks =args.use_action_masks,
        action_masks_dim = action_mask_choices.shape[0],
        hidden_size = args.hidden_size,
        initial_stddev = initial_stddev,
        min_stddev = min_stddev,
        stddev_transform = torch.nn.functional.softplus,
    )

    meta_actor_critic.to(device)

    meta_agent = PPO(
        meta_actor_critic, 
        args.clip_param,
        args.ppo_epoch, 
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.meta_lr,
        eps = args.eps,
        max_grad_norm = args.max_grad_norm,
        is_meta_agent=True,
        normalized_advantage = args.meta_agent_normalize_advantage
    )


    # 2. low-level policy's network and agent 

    actor_critic = Policy(
        observation_space = observation_space, 
        action_space = train_env.action_space,
        hidden_size = args.hidden_size, 
        initial_stddev=args.action_init_std_dev,
        min_stddev=args.action_min_std_dev,
        stddev_anneal_schedule=args.action_std_dev_anneal_schedule,
        stddev_transform=torch.nn.functional.softplus,
    )

    actor_critic.to(device)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        is_meta_agent=False,
        normalized_advantage=True,
    )

    load_pretrained_ll_policy = False

    if load_pretrained_ll_policy: 
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(ckpt["state_dict"])
        logger.info("loaded checkpoint: {}".format(ckpt_path)) # ckpt

    elif ckpt_path is not None:

        # load previously trained ckpt 
        ckpt = torch.load(ckpt_path, map_location = device)
        agent.load_state_dict(ckpt["state_dict"])
        logger.info("loaded checkpoint : {}".format(ckpt_path)) 

        # load previously trained meta ckpt 

        ckpt_path = os.path.joint(os.path.dirname(ckpt_path), os.path.basename(ckpt_path).replace("ckpt", "meta_ckpt"))
        ckpt = torch.load(ckpt_path, map_location = device)
        meta_agent.load_state_dict(ckpt["state_dict"])
        logger.info("loaded checkpoint : {}".format(ckpt_path))

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in agent.parameters())
        )
    )
        logger.info(
            "meta agent number of parameters: {}".format(
                sum(param.numel() for param in meta_agent.parameters())
        )
        )


            
    """
    eval_only => main function is immediately returned      
    """

    # if args.eval_only: 

    #     evaluate(args,
    #              eval_envs,
    #              meta_actor_critic,
    #              actor_critic,
    #              action_mask_choices,
    #              subgoal_mask_choices,
    #              subgoal_tolerance,
    #              device,
    #              writer,
    #              update=0,
    #              count_steps=0,
    #              eval_only=True)
    #     return

    observations = [train_env.reset()]

    batch = batch_obs(observations)

    for key in batch.keys():
        batch[key] = batch[key].squeeze(0)  # (1, N) → (N,)

    meta_rollouts = AsyncRolloutStorage(
        args.num_steps,
        num_envs,
        meta_observation_space,
        subgoal_space,
        args.hidden_size,
    )

    for sensor in meta_rollouts.observations:
        meta_rollouts.observations[sensor][0].copy_(batch[sensor])
    meta_rollouts.to(device)

    
    rollouts = RolloutStorage(
        args.num_steps,
        num_envs,
        rollout_observation_space,
        train_env.action_space,
        args.hidden_size,
    )

    for sensor in rollouts.observations:
        if sensor in batch:
            rollouts.observations[sensor][0].copy_(batch[sensor])
    rollouts.to(device)
    

    episode_rewards = torch.zeros(num_envs, 1, device=device)
    episode_success_rates = torch.zeros(num_envs, 1, device=device)
    episode_lengths = torch.zeros(num_envs, 1, device=device)
    episode_collision_steps = torch.zeros(num_envs, 1, device=device)
    episode_total_energy_costs = torch.zeros(num_envs, 1, device=device)
    episode_avg_energy_costs = torch.zeros(num_envs, 1, device=device)
    episode_stage_open_doors = torch.zeros(num_envs, 1, device=device)
    episode_stage_to_targets = torch.zeros(num_envs, 1, device=device)

    episode_counts = torch.zeros(num_envs, 1, device=device)
    current_episode_reward = torch.zeros(num_envs, 1, device=device)

    subgoal_rewards = torch.zeros(num_envs, 1, device=device)
    subgoal_success_rates = torch.zeros(num_envs, 1, device=device)
    subgoal_lengths = torch.zeros(num_envs, 1, device=device)
    subgoal_counts = torch.zeros(num_envs, 1, device=device)   
    current_subgoal_reward = torch.zeros(num_envs, 1, device=device)


    # window (sliding window)

    window_episode_reward = deque()
    window_episode_success_rates = deque()
    window_episode_lengths = deque()
    window_episode_collision_steps = deque()
    window_episode_total_energy_costs = deque()
    window_episode_avg_energy_costs = deque()
    window_episode_stage_open_doors = deque()
    window_episode_stage_to_targets = deque()
    window_episode_counts = deque()

    window_subgoal_reward = deque()
    window_subgoal_success_rates = deque()
    window_subgoal_lengths = deque()
    window_subgoal_counts = deque()

    current_subgoals = torch.zeros(batch["sensor"].shape, device=device)
    current_subgoal_log_probs = torch.zeros(num_envs, 1, device=device)
    current_meta_values = torch.zeros(num_envs, 1, device=device)
    current_subgoals_steps = torch.zeros(num_envs, 1, device=device)
    current_subgoals_cumulative_rewards = torch.zeros(num_envs, 1, device=device)
    original_subgoals = torch.zeros(batch["sensor"].shape, device=device)

    current_subgoal_masks = torch.zeros(batch["sensor"].shape, device=device)
    current_action_masks = torch.zeros(num_envs, action_dim, device=device)
    current_action_mask_indices = torch.zeros(num_envs, 1, dtype=torch.long, device=device)
    current_action_mask_log_probs = torch.zeros(num_envs, 1, device=device)
    next_meta_recurrent_hidden_states = torch.zeros(num_envs, args.hidden_size, device=device)

    t_start = time()
    env_time = 0
    pth_time = 0
    count_steps = start_env_step


    for update in range(start_epoch, args.num_updates):

        update_lr(agent.optimizer, 
                  args.lr,
                  update,
                  args.num_updates,
                  args.use_linear_lr_decay,
                  0
                  )
        update_lr(meta_agent.optimizer, 
                  args.meta_lr,
                  update,
                  args.num_updates,
                  args.use_linear_lr_decay, 
                  args.freeze_lr_n_updates)
        
        agent.clip_param = args.clip_param * (1.0 - update / args.num_updates)


        for step in range(args.num_steps):

            t_sample_action = time()

            with torch.no_grad():
                step_observation = {
                    k: v[step] for k, v in rollouts.observations.items()
                }

                meta_step_observations = {
                    k: v[meta_rollouts.valid_steps, meta_rollouts.env_indices] for k, v in meta_rollouts.observations.items()
                }

                (
                    meta_values,
                    subgoals,
                    subgoal_log_probs,
                    action_mask_indices,
                    action_mask_log_probs,
                    meta_recurrent_hidden_states,
                ) = meta_actor_critic.act(
                    meta_step_observations,
                    meta_rollouts.recurrent_hidden_states[meta_rollouts.valid_steps, meta_rollouts.env_indices],
                    meta_rollouts.masks[meta_rollouts.valid_steps, meta_rollouts.env_indices],
                )


                if args.use_action_masks:
                    action_masks = action_mask_choices.index_select(0, action_mask_indices.squeeze(1))
                    subgoal_masks = subgoal_mask_choices.index_select(0, action_mask_indices.squeeze(1))
                          
                else:
                    action_masks = torch.ones_like(current_action_masks)
                    subgoal_masks = torch.ones_like(current_subgoal_masks)


                

                should_use_new_subgoals = (current_subgoals_steps == 0.0).float()
                current_meta_values = should_use_new_subgoals * meta_values + \
                                      (1 - should_use_new_subgoals) * current_meta_values
                current_subgoals = should_use_new_subgoals * subgoals + \
                                   (1 - should_use_new_subgoals) * current_subgoals
                current_subgoal_log_probs = should_use_new_subgoals * subgoal_log_probs + \
                                            (1 - should_use_new_subgoals) * current_subgoal_log_probs
                original_subgoals = should_use_new_subgoals * subgoals + \
                                    (1 - should_use_new_subgoals) * original_subgoals
                current_subgoal_masks = should_use_new_subgoals * subgoal_masks.float() + \
                                        (1 - should_use_new_subgoals) * current_subgoal_masks
                current_action_masks = should_use_new_subgoals * action_masks + \
                                       (1 - should_use_new_subgoals) * current_action_masks
                current_action_mask_indices = should_use_new_subgoals.long() * action_mask_indices + \
                                              (1 - should_use_new_subgoals.long()) * current_action_mask_indices
                current_action_mask_log_probs = should_use_new_subgoals * action_mask_log_probs + \
                                                (1 - should_use_new_subgoals) * current_action_mask_log_probs
                next_meta_recurrent_hidden_states = should_use_new_subgoals * meta_recurrent_hidden_states + \
                                                    (1 - should_use_new_subgoals) * next_meta_recurrent_hidden_states

                current_subgoals *= current_subgoal_masks

                ideal_next_state = step_observation["sensor"] + current_subgoals

                roll = step_observation["auxiliary_sensor"][:, 9] * np.pi
                pitch = step_observation["auxiliary_sensor"][:, 10] * np.pi
                yaw = step_observation["auxiliary_sensor"][:, 49] * np.pi
                current_subgoals_rotated = rotate_torch_vector(current_subgoals, roll, pitch, yaw)
                current_subgoals_observation = current_subgoals_rotated

                # mask observation and add current subgoal
                step_observation["subgoal"] = current_subgoals_observation
                step_observation["subgoal_mask"] = current_subgoal_masks
                step_observation["action_mask"] = current_action_masks

                rollouts.observations["subgoal"][step] = current_subgoals_observation
                rollouts.observations["subgoal_mask"][step] = current_subgoal_masks
                rollouts.observations["action_mask"][step] = current_action_masks


                (
                    values,
                    actions,
                    action_log_probs,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    step_observation,
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                    update=update,
                )

                actions_masked = actions * current_action_masks


            pth_time += time() - t_sample_action

            t_step_env = time()

            actions_np = actions_masked.cpu().numpy()

            outputs = train_env.step(actions_np)
      
            observations, rewards, dones, infos = outputs
            observations, rewards, dones, infos = [[x] for x in (observations, rewards, dones, infos)]
           
            
            next_obs = [info[0]["last_observation"] if done else obs for obs, done, info in 
                        zip(observations, dones, infos)]
            
            env_time += time() - t_step_env 

            t_updates_stats = time()


            batch = batch_obs(observations)

            for key in batch.keys():
                batch[key] = batch[key].squeeze(0)  # (1, N) → (N,)
            
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            next_obs_batch = batch_obs(next_obs)

            for key in next_obs_batch.keys():
                next_obs_batch[key] = next_obs_batch[key].squeeze(0)  # (1, N) → (N,)
            

            for sensor in next_obs_batch:
                next_obs_batch[sensor] = next_obs_batch[sensor].to(device)

            rewards = torch.tensor(rewards, dtype=torch.float32, device = device)




            masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype = torch.float,
                device = device
            )


            success_masks = torch.tensor(
                [[float(info["success"])] if done and "success" in info else [0.0]
                 for done, info in zip(dones, infos)],
                 dtype = torch.float,
                 device = device
            )
            lengths = torch.tensor(
                [[float(info["episode_length"])] if done and "episode_length" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )
            collision_steps = torch.tensor(
                [[float(info["collision_step"])] if done and "collision_step" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )
            total_energy_cost = torch.tensor(
                [[float(info["energy_cost"])] if done and "energy_cost" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )
            avg_energy_cost = torch.tensor(
                [[float(info["energy_cost"]) / float(info["episode_length"])]
                 if done and "energy_cost" in info and "episode_length" in info
                 else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )
            stage_open_door = torch.tensor(
                [[float(info["stage"] >= 1)] if done and "stage" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )
            stage_to_target = torch.tensor(
                [[float(info["stage"] >= 2)] if done and "stage" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )
            collision_rewards = torch.tensor(
                [[float(info["collision_reward"])] if "collision_reward" in info else [0.0] for info in infos],
                dtype=torch.float,
                device=device
            )

            current_episode_reward += rewards 
            episode_rewards += (1-masks) * current_episode_reward
            episode_success_rates += success_masks
            episode_lengths += lengths
            episode_collision_steps += collision_steps
            episode_total_energy_costs += total_energy_cost
            episode_avg_energy_costs += avg_energy_cost
            episode_stage_open_doors += stage_open_door
            episode_stage_to_targets += stage_to_target 
            episode_counts += 1 - masks 
            current_episode_reward *= masks  # reset current episode reward if done

            current_subgoals_steps += 1 
            current_subgoals_cumulative_rewards += rewards 

            subgoals_diff = (ideal_next_state - next_obs_batch["sensor"]) * current_subgoal_masks 
            subgoals_distance = torch.abs(subgoals_diff)
       
            subgoals_achieved = torch.all(subgoals_distance < subgoal_tolerance, dim=1, keepdim=True)
            subgoals_done = (
                    subgoals_achieved  # subgoals achieved
                    | (current_subgoals_steps == args.time_scale)  # subgoals time up
                    | (1.0 - masks).byte()  # episode is done
            )
            subgoals_done = subgoals_done.float() # false 

            subgoals_achieved = subgoals_achieved.float()

            penalty_prob_th  = 0.5 
            current_subgoals_penalty = args.subgoal_failed_penalty * subgoals_done * (1.0 - subgoals_achieved) * (torch.rand(num_envs, 1, device=device) < penalty_prob_th).float() # 1.0 or 0.0

            meta_rollouts.insert(
                subgoals_done,
                batch,
                next_meta_recurrent_hidden_states,
                original_subgoals,
                current_subgoal_log_probs,
                current_action_mask_indices,
                current_action_mask_log_probs,
                current_meta_values,
                current_subgoals_cumulative_rewards + current_subgoals_penalty, 
                masks,
            )

            prev_potential = ideal_next_state - step_observation["sensor"]

            prev_potential = torch.norm(prev_potential * current_subgoal_masks, dim=1, keepdim=True)
            current_potential = ideal_next_state - next_obs_batch["sensor"]
            current_potential = torch.norm(current_potential * current_subgoal_masks, dim=1, keepdim=True)

            intrinsic_reward = 0.0
            intrinsic_reward += (prev_potential - current_potential) * args.intrinsic_reward_scaling

            intrinsic_reward += subgoals_achieved.float() * args.subgoal_achieved_reward
            intrinsic_reward += collision_rewards * args.extrinsic_collision_reward_weight

            intrinsic_reward += rewards * args.extrinsic_reward_weight

            current_subgoal_reward += intrinsic_reward 
            subgoal_rewards += subgoals_done * current_subgoal_reward
            subgoal_success_rates += subgoals_achieved
            subgoal_lengths += subgoals_done * current_subgoals_steps 
            subgoal_counts += subgoals_done
            current_subgoal_reward *= (1-subgoals_done) # reset current subgoal reward if done

            rollouts.insert(
                batch,
                recurrent_hidden_states,
                actions,
                action_log_probs,
                values,
                intrinsic_reward,
                1-subgoals_done,
            )


            current_subgoals = (ideal_next_state - next_obs_batch["sensor"]) * current_subgoal_masks 
            current_subgoals_steps = (1-subgoals_done) * current_subgoals_steps
            current_subgoals_cumulative_rewards = (1-subgoals_done) * current_subgoals_cumulative_rewards

            count_steps += num_envs 
            pth_time += time() - t_updates_stats


        # 1 step is terminated 

        if len(window_episode_reward) == args.perf_window_size:
            window_episode_reward.popleft()
            window_episode_success_rates.popleft()
            window_episode_lengths.popleft()
            window_episode_collision_steps.popleft()
            window_episode_total_energy_costs.popleft()
            window_episode_avg_energy_costs.popleft()
            window_episode_stage_open_doors.popleft()
            window_episode_stage_to_targets.popleft()
            window_episode_counts.popleft()
            window_subgoal_reward.popleft()
            window_subgoal_success_rates.popleft()
            window_subgoal_lengths.popleft()
            window_subgoal_counts.popleft()
        
        window_episode_reward.append(episode_rewards.clone())
        window_episode_reward.append(episode_rewards.clone())
        window_episode_success_rates.append(episode_success_rates.clone())
        window_episode_lengths.append(episode_lengths.clone())
        window_episode_collision_steps.append(episode_collision_steps.clone())
        window_episode_total_energy_costs.append(episode_total_energy_costs.clone())
        window_episode_avg_energy_costs.append(episode_avg_energy_costs.clone())
        window_episode_stage_open_doors.append(episode_stage_open_doors.clone())
        window_episode_stage_to_targets.append(episode_stage_to_targets.clone())
        window_episode_counts.append(episode_counts.clone())
        window_subgoal_reward.append(subgoal_rewards.clone())
        window_subgoal_success_rates.append(subgoal_success_rates.clone())
        window_subgoal_lengths.append(subgoal_lengths.clone())
        window_subgoal_counts.append(subgoal_counts.clone())

        t_update_model = time()

        with torch.no_grad():
            last_meta_observation = {
                k: v[meta_rollouts.valid_steps, meta_rollouts.env_indices]
                for k, v in meta_rollouts.observations.items()
            }

            next_meta_value = meta_actor_critic.get_value(
                last_meta_observation,
                meta_rollouts.recurrent_hidden_states[meta_rollouts.valid_steps, meta_rollouts.env_indices],
                meta_rollouts.masks[meta_rollouts.valid_steps, meta_rollouts.env_indices]
            ).detach()


            last_observation = {
                k:v[-1].clone() for k, v in rollouts.observations.items()
            }

            roll = rollouts.observations["auxiliary_sensor"][-1][:, 9] * np.pi
            pitch = rollouts.observations["auxiliary_sensor"][-1][:, 10] * np.pi
            yaw = rollouts.observations["auxiliary_sensor"][-1][:, 34] * np.pi
            current_subgoals_rotated = rotate_torch_vector(current_subgoals, roll, pitch, yaw)
            current_subgoals_observation = current_subgoals_rotated 

            last_observation["subgoal"] = current_subgoals_observation
            last_observation["subgoal_mask"] = current_subgoal_masks
            last_observation["action_mask"] = current_action_masks

            next_value = actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        meta_rollouts.compute_returns(
            next_meta_value, args.use_gae, args.meta_gamma, args.tau
        )

        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma, args.tau
        )
    
        meta_value_loss, subgoal_loss, meta_dist_entropy = meta_agent.update(meta_rollouts)
        value_loss, action_loss, dist_entropy = agent.update(rollouts, update = update)

        meta_rollouts.after_update()
        rollouts.after_update()
        pth_time += time() - t_update_model 

        # log stats
        if update > 0 and update % args.log_interval == 0:
            logger.info(
                "update: {}\tenv_steps: {}\tenv_steps_per_sec: {:.3f}\tenv-time: {:.3f}s\tpth-time: {:.3f}s".format(
                    update, count_steps, count_steps / (time() - t_start), env_time, pth_time
                )
            )
            logger.info(
                "update: {}\tenv_steps: {}\tvalue_loss: {:.3f}\taction_loss: {:.3f}\tdist_entropy: {:.3f}".format(
                    update, count_steps, value_loss, action_loss, dist_entropy
                )
            )
            logger.info(
                "update: {}\tenv_steps: {}\tmeta_value_loss: {:.3f}\tsubgoal_loss: {:.3f}\t"
                "meta_dist_entropy: {:.3f}".format(
                    update, count_steps, meta_value_loss, subgoal_loss, meta_dist_entropy
                )
            )
            writer.add_scalar("time/env_step_per_second", count_steps / (time() - t_start), global_step=update)
            writer.add_scalar("time/env_time_per_update", env_time / update, global_step=update)
            writer.add_scalar("time/pth_time_per_update", pth_time / update, global_step=update)
            writer.add_scalar("time/env_steps_per_update", count_steps / update,
                              global_step=update)
            writer.add_scalar("losses/value_loss", value_loss, global_step=update)
            writer.add_scalar("losses/action_loss", action_loss, global_step=update)
            writer.add_scalar("losses/dist_entropy", dist_entropy, global_step=update)
            writer.add_scalar("losses/meta_value_loss", meta_value_loss, global_step=update)
            writer.add_scalar("losses/subgoal_loss", subgoal_loss, global_step=update)
            writer.add_scalar("losses/meta_dist_entropy", meta_dist_entropy, global_step=update)

            window_rewards = (window_episode_reward[-1] - window_episode_reward[0]).sum()
            window_success_rates = (window_episode_success_rates[-1] - window_episode_success_rates[0]).sum()
            window_lengths = (window_episode_lengths[-1] - window_episode_lengths[0]).sum()
            window_collision_steps = (window_episode_collision_steps[-1] - window_episode_collision_steps[0]).sum()
            window_total_energy_costs = (window_episode_total_energy_costs[-1] - window_episode_total_energy_costs[0]).sum()
            window_avg_energy_costs = (window_episode_avg_energy_costs[-1] - window_episode_avg_energy_costs[0]).sum()
            window_stage_open_doors = (window_episode_stage_open_doors[-1] - window_episode_stage_open_doors[0]).sum()
            window_stage_to_targets = (window_episode_stage_to_targets[-1] - window_episode_stage_to_targets[0]).sum()

            window_counts = (window_episode_counts[-1] - window_episode_counts[0]).sum()
            if window_counts > 0:
                reward_mean = (window_rewards / window_counts).item()
                success_rate_mean = (window_success_rates / window_counts).item()
                lengths_mean = (window_lengths / window_counts).item()
                collision_steps_mean = (window_collision_steps / window_counts).item()
                total_energy_costs_mean = (window_total_energy_costs / window_counts).item()
                avg_energy_costs_mean = (window_avg_energy_costs / window_counts).item()
                stage_open_doors_mean = (window_stage_open_doors / window_counts).item()
                stage_to_targets_mean = (window_stage_to_targets / window_counts).item()

                logger.info(
                    "average window size {}\treward: {:.3f}\tsuccess_rate: {:.3f}\tepisode length: {:.3f}\t"
                    "collision_step: {:.3f}\ttotal_energy_cost: {:.3f}\tavg_energy_cost: {:.3f}\t"
                    "stage_open_door: {:.3f}\tstage_to_target: {:.3f}".format(
                        len(window_episode_reward),
                        reward_mean,
                        success_rate_mean,
                        lengths_mean,
                        collision_steps_mean,
                        total_energy_costs_mean,
                        avg_energy_costs_mean,
                        stage_open_doors_mean,
                        stage_to_targets_mean,
                    )
                )
                writer.add_scalar("train/updates/reward", reward_mean, global_step=update)
                writer.add_scalar("train/updates/success_rate", success_rate_mean, global_step=update)
                writer.add_scalar("train/updates/episode_length", lengths_mean, global_step=update)
                writer.add_scalar("train/updates/collision_step", collision_steps_mean, global_step=update)
                writer.add_scalar("train/updates/total_energy_cost", total_energy_costs_mean, global_step=update)
                writer.add_scalar("train/updates/avg_energy_cost", avg_energy_costs_mean, global_step=update)
                writer.add_scalar("train/updates/stage_open_door", stage_open_doors_mean, global_step=update)
                writer.add_scalar("train/updates/stage_to_target", stage_to_targets_mean, global_step=update)

                writer.add_scalar("train/env_steps/reward", reward_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/success_rate", success_rate_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/episode_length", lengths_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/collision_step", collision_steps_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/total_energy_cost", total_energy_costs_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/avg_energy_cost", avg_energy_costs_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/stage_open_door", stage_open_doors_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/stage_to_target", stage_to_targets_mean, global_step=count_steps)
            else:
                logger.info("No episodes finish in current window")

            window_rewards = (window_subgoal_reward[-1] - window_subgoal_reward[0]).sum()
            window_success_rates = (window_subgoal_success_rates[-1] - window_subgoal_success_rates[0]).sum()
            window_lengths = (window_subgoal_lengths[-1] - window_subgoal_lengths[0]).sum()
            window_counts = (window_subgoal_counts[-1] - window_subgoal_counts[0]).sum()

            if window_counts > 0:
                reward_mean = (window_rewards / window_counts).item()
                success_rate_mean = (window_success_rates / window_counts).item()
                lengths_mean = (window_lengths / window_counts).item()
                logger.info(
                    "window_size: {}\tsubgoal_reward: {:.3f}\tsubgoal_success_rate: {:.3f}\tsubgoal_length: {:.3f}".format(
                        len(window_subgoal_reward),
                        reward_mean,
                        success_rate_mean,
                        lengths_mean,
                    )
                )
                writer.add_scalar("train/updates/subgoal_reward", reward_mean, global_step=update)
                writer.add_scalar("train/updates/subgoal_success_rate", success_rate_mean, global_step=update)
                writer.add_scalar("train/updates/subgoal_length", lengths_mean, global_step=update)
                writer.add_scalar("train/env_steps/subgoal_reward", reward_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/subgoal_success_rate", success_rate_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/subgoal_length", lengths_mean, global_step=count_steps)
            else:
                logger.info("No subgoals finish in current window")

        # checkpoint model 

        if update > 0 and update % args.checkpoint_interval == 0 :
            checkpoint = {"state_dict" : agent.state_dict()}
            torch.save(
                checkpoint,
                os.path.join(
                    ckpt_folder, 
                    "ckpt.{}.pth".format(update),
                ),
            )

            checkpoint = {"state_dict" : meta_agent.state_dict()}
            torch.save(
                checkpoint, 
                os.path.join(
                    ckpt_folder,
                    "meta_ckpt.{}.pth".format(update),
                ),
            )



if __name__ == "__main__":
    main()
