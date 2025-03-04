import os
import re
from pathlib import Path
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco

"""
각 관절의 토크 제어를 직접 학습하는 환경 (low-level policy)
"""

GEOM = 5

cur_dir = Path(os.path.dirname(__file__))

class DoorOpenEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 200,
    }

    def __init__(self, episode_len=500, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

        MujocoEnv.__init__(
            self, 
            os.path.abspath(cur_dir / "scene" / "mobile_fr3.xml"), 
            5, 
            observation_space=observation_space, 
            **kwargs
        )
        self.step_number = 0
        self.episode_len = episode_len
        # 충돌 판정에서 제외할 geom id: door_handle, finger 관련된 ids
        self.excluded_geom_ids = self._find_geom_ids()
        # Franka Home Pose 반영
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qpos[3:10] += np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])
        print("Initialized DoorOpenEnv")

    def step(self, a):
        act = self._scale_act(a)
        self.do_simulation(act, self.frame_skip)
        self.step_number += 1

        obs = self._get_obs(act)
        rew, term = self._get_rew_done(obs)
        trunc = self.step_number >= self.episode_len
        
        return obs, rew, term, trunc, {}
    
    def reset_model(self):
        self.step_number = 0
        
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self, prev_act = np.zeros(7)):
        hand_xmat = self.data.body("hand").xmat.reshape(3, 3)
        latch_xmat = self.data.body("latch").xmat.reshape(3, 3)
        hand_y = hand_xmat[:, 1] / np.linalg.norm(hand_xmat[:, 1])
        latch_y = latch_xmat[:, 1] / np.linalg.norm(latch_xmat[:, 1])
        hand_z = hand_xmat[:, 2] / np.linalg.norm(hand_xmat[:, 2])
        latch_z = latch_xmat[:, 2] / np.linalg.norm(latch_xmat[:, 2])
        obs = np.concatenate([
                            self.data.qvel[3:10],         # (7,)
                            self.data.body("latch").xpos - self.data.body("hand").xpos,    # (3,)
                            [np.dot(hand_y, latch_y)],    # (1,)
                            [np.dot(hand_z, latch_z)],    # (1,)
                            prev_act,                     # (7,)
                        ])
        return obs
    
    def _get_rew_done(self, obs):
        # EE와 목표 지점 사이의 거리
        dist = np.linalg.norm(self.data.body("hand").xpos - self.data.body("latch").xpos)
        rew_dist = (1 / (1 + dist))
        rew_dist = 2 * rew_dist if dist < 1.0 else rew_dist
        # EE와 목표 지점의 정렬
        rew_align = (obs[10] + 2 * obs[11]) / 3
        rew_align = 2 * rew_align if rew_align > 0.7 else rew_align
        # 관절 속도 패널티
        pen_qvel = abs(self.data.qvel[3:10]).sum()
        # 충돌 패널티
        is_collided = self._process_collision()
        pen_collision = 1 if is_collided else 0
        
        rew = rew_dist + rew_align - pen_qvel * 0.001 - pen_collision * 100
        
        success = (dist < 0.7) & (rew_align > 1.8)
        done = success | is_collided
        
        return rew, done
    
    def _scale_act(self, act):
        # Scale the action [-1, 1] to the range of the actuators [low, high]
        ctrl_range = self.model.actuator_ctrlrange
        low, high = ctrl_range[:, 0], ctrl_range[:, 1]
        
        act = low + (act + 1.0) * 0.5 * (high - low)
        return act
    
    def _process_collision(self):
        # 충돌 판정
        for contact in self.data.contact:
            if contact.geom1 and contact.geom2 and (contact.geom1 not in self.excluded_geom_ids) and (contact.geom2 not in self.excluded_geom_ids):
                geom1_name = mujoco.mj_id2name(self.model, GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(self.model, GEOM, contact.geom2)
                # print(f"Collision between {geom1_name} and {geom2_name} detected!")
                return True
        return False
            
        
    def _find_geom_ids(self):
        geom_ids = []
        # 문 손잡이는 잡아야 하므로 충돌 감지에서 제외
        handle_id = mujoco.mj_name2id(self.model, GEOM, "door_handle")
        geom_ids.append(handle_id)
        # 손가락은 문을 열기 위해 사용되므로 충돌 감지에서 제외
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, GEOM, geom_id)
            if geom_name and re.match("^finger", geom_name):
                geom_ids.append(geom_id)
                
        return geom_ids