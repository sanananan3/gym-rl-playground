import os
import re
from pathlib import Path
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
from scipy.spatial.transform import Rotation as R
from mjc_env.door.util import rotate_quaternion, get_quaternion_difference

"""
강화학습의 출력 (action) 이 End Effector의 pose의 변화량 (=ee를 움직일 방향을 예측하는 방식)인 Inverse Dynamics 환경 (high-level policy)
End-Effector의 waypoint를 학습하는 환경 
"""

GEOM = 5

cur_dir = Path(os.path.dirname(__file__))

class MetaDoorOpenEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(self, episode_len=500, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        observation_space = Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        self.idv_action_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32) # end effector pose 
                                                                                            # (x, y, z, qw, qx, qy, qz)

        MujocoEnv.__init__(
            self, 
            os.path.abspath(cur_dir / "scene2" / "mobile_fr3.xml"), 
            10, 
            observation_space=observation_space, 
            **kwargs
        )
        self.step_number = 0
        self.episode_len = episode_len
        # 충돌 판정에서 제외할 geom id: door_handle, finger 관련된 ids
        self.excluded_geom_ids = self._find_geom_ids()
        # Franka Reach Pose 반영
        self.init_qpos = self.data.qpos.ravel().copy()
        # self.init_qpos[3:10] += np.array([0.3295, -0.0929, -0.3062, -2.4366, 1.4139, 2.7500, 0.6700])
        self.init_qpos[3:10] += np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854])
        
        self.success_duration = 0
        print("Initialized DoorOpenEnv")
        
    def _set_action_space(self):
        self.action_space = self.idv_action_space
        return self.action_space

    def step(self, a):
        """
        a: desired pose of the end effector
        """
        self.data.mocap_pos[0] += a[:3] # end-effector's position change (x, y, z)
        self.data.mocap_quat[0] = a[3:] # end-effector's rotation change (qw, qx, qy, qz)
        mujoco.mj_step(self.model, self.data, self.frame_skip) # mujoco simulation update 
        self.step_number += 1

        obs = self._get_obs()
        
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

    def _get_obs(self):

        print("[INFO] observation data : ", self.data)
        current_ee_pos = self.data.body("hand").xpos
        current_ee_quat = R.from_matrix(self.data.body("hand").xmat.reshape(3, 3)).as_quat()
        
        target_pos = self.data.body("latch_axis").xpos
        target_pos[1] -= 0.05                                           # 손잡이 잡는 위치 조정
        target_quat = R.from_matrix(self.data.body("latch_axis").xmat.reshape(3, 3)).as_quat()
        target_quat = rotate_quaternion(target_quat, [0, 1, 0], 180)    # 손잡이 잡는 z 방향 조정
        
        obs = np.concatenate([
                            current_ee_pos,     # (3,)
                            current_ee_quat,    # (4,)
                            target_pos,         # (3,)
                            target_quat         # (4,)                            
                        ])
        return obs
    
    def convert_mjc_obs(obs):
        """ convert mjc observation (flat array) tp iGibson-style dictionary """

        obs_dict = {
            "sensor" : obs[:3].reshape(1, -1), # end-effector position 
            "auxiliary_sensor": obs[3:7].reshape(1, -1),

        }
    
    def _get_rew_done(self, obs):
        # EE와 목표 지점 사이의 거리
        dist = np.linalg.norm(obs[:3] - obs[7:10])
        rew_dist = (1 / (1 + dist))
        # EE와 목표 방향 사이의 각도
        angle = get_quaternion_difference(obs[3:7], obs[10:])
        rew_angle = (1 / (1 + angle))
        # 관절 속도 패널티
        pen_qvel = -abs(self.data.qvel[3:10]).sum()
        # 충돌 패널티
        is_collided = self._process_collision()
        pen_collision = -1 if is_collided else 0
        
        # 성공 판정: 두 그리퍼가 서로 닿지 않으면서 그리퍼 안의 터치센서 활성화
        success = np.any(self.data.sensordata) & ~ np.all(self.data.qpos[10:11] < 0.01)
        self.success_duration = self.success_duration + 1 if success else 0

        rew = success * 10 + rew_dist + rew_angle - pen_qvel * 0.001 - pen_collision * 10
        done = (self.success_duration > 20)
        
        return rew, done
    
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