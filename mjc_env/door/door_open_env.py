import os
import re
from pathlib import Path
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
import mujoco
from scipy.spatial.transform import Rotation as R
from mjc_env.door.util import rotate_quaternion, get_quaternion_difference
from collections import OrderedDict

"""
policy가 학습되는 최종 환경 (goal space와 동일)
"""

GEOM = 5 # geom id is 5

cur_dir = Path(os.path.dirname(__file__))



class FrankaDoorEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(self, episode_len=5000, **kwargs):
        
        utils.EzPickle.__init__(self, **kwargs)



        observation_space = Dict({
            "sensor": Box(low=-np.inf, high=np.inf, shape=(3,),dtype=np.float32),
            "auxiliary_sensor": Box(low=-np.inf, high=np.inf, shape=(51, ), dtype=np.float32)
        })

        self.idv_action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32) # end effector pose => action space 
                                                                                            # (x, y, z, qw, qx, qy, qz)

        MujocoEnv.__init__(
            self, 
            os.path.abspath(cur_dir / "scene3" / "mobile_fr3.xml"), 
            10, 
            observation_space=observation_space, 
            **kwargs
        )

        print("[INFO] self.dt : ", self.dt)

        # self.episode_len = episode_len
        # 충돌 판정에서 제외할 geom id: door_handle, finger 관련된 ids
        self.excluded_geom_ids = self._find_geom_ids()


         # Franka Reach Pose 반영
        self.init_qpos = self.data.qpos.ravel().copy()

        self.init_qpos[6:13] += np.array([0, -0.7854, 0, -2.3562, 0, 1.5708, 0.7854]) # arm_joint position 
      
        # self.success_duration = 0
        print("Initialized DoorOpenEnv")

        self.stage = 0
        self.prev_stage = self.stage

        self.stage_get_to_door_handle = 0 # going to door handle 
        self.stage_open_door = 1 # opening the door with grasping the handle 
        self.stage_get_to_door_handle_after_open = 2 # going to target position after opening the door 

        self.door_handle_dist_thres =  0.05
        self.cid = None 
        self.door_angle = 1.8


        # termination condition 

        self.dist_tol = 0.05
        self.max_step = 1000

        # reward 

        self.reward_type = 'dense'
        self.success_reward = 50.0
        self.slack_reward = -0.01
        self.death_z_thresh = 0.5

        # reward weight 

        self.potential_reward_weight = 2.0

        self.electricity_reward_weight = -0.001
        self.stall_torque_reward_weight = 0.0
        self.collision_reward_weight = -0.01

        # discount factor
        self.discount_factor = 0.99

    def _set_action_space(self):
        self.action_space = self.idv_action_space
        return self.action_space

    def step(self, action):
        """
        MuJoCo step function
        """

        action =np.squeeze(action)

        print("action: ", action)

        # 1. stage_get_to_door_handle 
        dist = np.linalg.norm(self.data.body("latch").xpos - self.data.body("hand").xpos)

        self.prev_stage = self.stage 

        if self.stage == self.stage_get_to_door_handle and dist < self.door_handle_dist_thres:
            self.stage = self.stage_open_door
            print("[STAGE] Open Door")

        # 2. stage_open_door 
        door_angle = self.data.qpos[self.model.joint("hinge").qposadr] # real simulation's door angle 

        # self.door_angle = 1.8 
        if self.stage == self.stage_open_door and door_angle > self.door_angle:
            self.stage = self.stage_get_to_door_handle_after_open
            print("[STAGE] Get to Door Handle After Open")

        # 3. If the door is illegally open, then reset the door angle 

        if door_angle < -0.002: 
            self.data.qpos[self.model.joint("hinge").qposadr] = 0.0  # reset

        # 4. apply action
        self.data.mocap_pos[0] += action[:3] # end-effector's position change (x, y, z)
        self.data.mocap_quat[0] = action[3:] # end-effector's rotation change (qw, qx, qy, qz)

        # 5. run simulation 
        mujoco.mj_step(self.model, self.data, self.frame_skip) # mujoco simulation update 

        # 6. get state 
        state = self._get_state()


        collision_links = self._process_collision()

        # 7. get reward 
        info = {}
        reward, info = self._get_reward(collision_links, action, info)

        # 8. get termination 
        done, info = self._get_termination(collision_links, info)

        if done:
            info["last_observation"]= state
            state = self.reset()

        return state, reward, done, info
    


    
    def reset_model(self):
        
        self.stage = 0
        self.prev_state = self.stage 

        self.initial_potential = self.get_potential()
        self.potential = self.initial_potential

        self.current_step = 0 
        self.collision_step = 0
        self.energy_cost = 0.0
        

        qpos = self.init_qpos 
        qvel = self.init_qvel 
        

        self.set_state(qpos, qvel)

        self.target_pos = self.data.body("target").xpos
        
        return self._get_state()


    def _get_state(self):
        
        state = OrderedDict()

        # 1. sensor 

        current_ee_pos = self.data.body("hand").xpos # arm_x, arm_y, arm_z (world)
        state["sensor"] = np.array(np.concatenate([current_ee_pos]))

        # 2. auxilary sensor 

        base_pos = self.data.body("base").xpos # x, y, z
        base_vel = self.data.qvel[:3] # vel_x, vel_y, vel_z

        # for i in range(self.model.nv):
        #     joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        #     print(f"qvel[{i}] corresponds to joint: {joint_name}")

        base_rot_mat = self.data.body('base').xmat.reshape(3,3)
        base_euler = R.from_matrix(base_rot_mat).as_euler("xyz") 
        roll, pitch, yaw = base_euler 

        ee_pos = self.data.body("hand").xpos 
        ee_pos_local =  np.array(ee_pos - base_pos) # from world to local 

        arm_joints = np.array([self.data.qpos[self.model.joint(f"joint{i+1}").qposadr] for i in range(7) ]) # 2D (3,1)
        arm_joints = arm_joints.squeeze() # 1D 

        arm_joints_vel = np.array([self.data.qvel[self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")]] for i in range(7)])
        
        base_x_vel = self.data.qvel[self.model.joint("base_x_slide_joint").dofadr]
        base_y_vel = self.data.qvel[self.model.joint("base_y_slide_joint").dofadr]
        base_z_vel = self.data.qvel[self.model.joint("base_z_slide_joint").dofadr]

        wheel1_state = np.array([base_x_vel, base_y_vel, base_z_vel]) # 2D (3,1)
        wheel2_state = np.array([base_x_vel, base_y_vel, base_z_vel]) # 2D (3,1)

        wheel1_state = wheel1_state.squeeze() # 1D
        wheel2_state = wheel2_state.squeeze() # 1D
        door_angle = self.data.qpos[self.model.joint("hinge").qposadr]  # 이미 float 값일 가능성이 높음
        door_angle = np.array(door_angle, dtype=np.float32).flatten()  # (1,)로 변환
        door_cos = np.array([np.cos(door_angle)], dtype=np.float32).flatten()  # (1,)
        door_sin = np.array([np.sin(door_angle)], dtype=np.float32).flatten()  # (1,)

        has_door_handle_in_hand = np.array([1 if self._check_grasping("door_handle") else 0])

        target_pos = self.data.body("target").xpos # previous, latch_axis 
        door_pos_local = self.data.body("door").xpos - base_pos 

        target_pos_local = target_pos - base_pos


        has_collision = np.array([1 if self._process_collision() else 0])

        state["auxiliary_sensor"] = np.array(np.concatenate([
            base_pos, 
            ee_pos_local,
            base_vel,
            [roll,pitch],
            wheel1_state,
            wheel2_state,
            arm_joints,
            arm_joints_vel,
            self.data.qvel[3:6],
            [yaw, np.cos(yaw), np.sin(yaw)],
            door_angle, door_cos, door_sin, has_door_handle_in_hand,
            target_pos, 
            door_pos_local,
            target_pos_local,
            has_collision
        ]))
        
        return state
    



    def get_potential(self):

        def l2_distance(a,b):
            return np.sqrt(np.sum((a-b)**2))
        
        door_angle = self.data.qpos[self.model.joint("hinge").qposadr]
        door_handle_pos = self.data.body("latch").xpos
        ee_pos = self.data.body("hand").xpos

        if self.stage == self.stage_get_to_door_handle:
            potential = l2_distance(door_handle_pos, ee_pos)

        elif self.stage == self.stage_open_door:
            potential = -door_angle 

        elif self.stage == self.stage_get_to_door_handle_after_open:
            potential = l2_distance(self.target_pos, ee_pos)

        return potential 
    

    def _get_reward(self, collision_links=[], action=None, info={}):

        reward = 0.0

        def l2_distance(a,b):
            return np.sqrt(np.sum((a-b)**2))
        
        # 1. potential reward 

        if self.stage != self.prev_stage: 
            # advance to the next stage 
            self.potential = self.get_potential()

        else: 
            new_potential = self.get_potential()
            potential_reward = self.potential - new_potential 
            reward += potential_reward * self.potential_reward_weight
            self.potential = new_potential

        # 2. electricity reward 

        base_moving = np.any(np.abs(action[:2]) >= 0.01)
        arm_moving = np.any(np.abs(action[2:]) >= 0.01)
        electricity_reward = float(base_moving) + float(arm_moving)

        self.energy_cost += electricity_reward
        reward += electricity_reward * self.electricity_reward_weight

        # 3. collision penalty
        collision_reward = float(collision_links)
        self.collision_step += int(collision_reward)
        reward += collision_reward * self.collision_reward_weight
        info["collision_reward"] = collision_reward * self.collision_reward_weight

        # 4. goal reached 

        if l2_distance(self.target_pos, self.data.body("hand").xpos) < self.dist_tol:
            reward += self.success_reward

        return reward, info 
        

    def _get_termination(self, collision_links = [], info= {}):

        self.current_step += 1
        done = False 

        def l2_distance (a,b):
            return np.sqrt(np.sum((a-b)**2))
        
        if l2_distance(self.target_pos, self.data.body("hand").xpos) < self.dist_tol:
            print("GOAL")
            done = True 
            info["success"] = True

        elif self.data.qpos[self.model.joint("base_z_hinge_joint").qposadr] > self.death_z_thresh:
            print("DEATH")
            done = True 
            info["success"] = False 

        elif self.current_step >= self.max_step:
            done = True 
            info["success"] = False

            
        if done:
            info['episode_length'] = self.current_step
            info['collision_step'] = self.collision_step
            info['energy_cost'] = self.energy_cost
            info['stage'] = self.stage 

        return done, info
            
        

    def _process_collision(self):
        # 충돌 판정
        for contact in self.data.contact:
            if contact.geom1 and contact.geom2 and (contact.geom1 not in self.excluded_geom_ids) and (contact.geom2 not in self.excluded_geom_ids):
                geom1_name = mujoco.mj_id2name(self.model, GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(self.model, GEOM, contact.geom2)

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
    
    def _check_grasping(self, object_name):

        for contact in self.data.contact:
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if geom1 == "hand" and geom2 == object_name:
                return True 
            
        return False
        

    
    def set_subgoal(self, ideal_next_state): 

        def set_position(self, pos):

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "subgoal")

            if body_id == -1 :
                raise ValueError("No such body : subgoal")
            
            self.model.body_pos[body_id] = np.array(pos)
            mujoco.mj_forward(self.model, self.data)
            
        # obs_avg = (self.observation_normalizer['sensor'][1] + self.observation_normalizer['sensor'][0]) / 2.0
        # obs_mag = (self.observation_nomalizer['sensor'][1] - self.observation_normalizer['sensor'][0]) / 2.0
        # ideal_next_state = (ideal_next_state * obs_mag) + obs_avg

        set_position(self, ideal_next_state)



    def set_subgoal_type(self, only_base = True):
        
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "subgoal_ee")

        if only_base :
            self.model.geom_rgba[geom_id] = np.array([0, 0, 0, 0.0])

        else : 
            self.model.geom_rgba[geom_id] = np.array([0, 0, 0, 1.0])


        mujoco.mj_forward(self.model, self.data) # maybe annotated later ...  ? 