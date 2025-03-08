import numpy as np
from scipy.spatial.transform import Rotation as R
# from util import rotate_quaternion, interpolate_quaternion

import sys 
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from mjc_env.door.door_open_env_v1 import MetaDoorOpenEnv
from mjc_env.door.door_open_env_v0 import DoorOpenEnv
from mjc_env.door.util import rotate_quaternion, interpolate_quaternion

env = MetaDoorOpenEnv(render_mode="human")
obs, _ = env.reset()
    
done = False

while not done:
    
    # 현재 위치와 목표 위치 사이의 거리 계산
    desired_pos_act = env.data.body('latch_axis').xpos - env.data.body('hand').xpos
    desired_pos_act[1] -= 0.05
    # 목표 방향 (latch axis와 z는 반대, y는 동일 방향 = y축 기준 180도 회전) 계산
    desired_quat = R.from_matrix(env.data.body("latch_axis").xmat.reshape(3, 3)).as_quat()
    desired_quat = rotate_quaternion(desired_quat, [0, 1, 0], 180)

    # 현재 방향과 목표 방향 사이 slerp (구면 선형 보간)하여 목표 방향까지 1/2 정도 회전
    # - 너무 빠르면 물리적으로 불가능한 움직임이 생길 수 있음

    if env.data.mocap_quat.shape[0] > 0:
        current_quat = np.roll(env.data.mocap_quat[0], shift=-1)
    else: 
        print("warning : mocap_quat is empty")
        current_quat = np.array([1, 0, 0, 0])

        
    interp_quat = interpolate_quaternion(current_quat, desired_quat, 2)
    desired_quat_act = np.roll(interp_quat[1], shift=1)

    desired_act = np.append(desired_pos_act, desired_quat_act)
    obs, rew, term, trunc, _ = env.step(desired_act)

    done = term | trunc
    env.render()
        