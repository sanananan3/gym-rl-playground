from door_open_env_v1 import DoorOpenEnv
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import pyquaternion as pyq

env = DoorOpenEnv(render_mode="human")
obs, _ = env.reset()

def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements

def interpolate_quaternion(q1, q2, steps):
    """
    Interpolates between two quaternions q1 and q2 over a specified number of steps.
    """
    q1 = np.asarray(q1) / np.linalg.norm(q1)
    q2 = np.asarray(q2) / np.linalg.norm(q2)

    slerp = Slerp([0, 1], R.from_quat([q1, q2]))
    times = np.linspace(0, 1, steps)
    interpolated_rotations = slerp(times)

    return interpolated_rotations.as_quat()
    
done = False
while not done:
    # 현재 위치와 목표 위치 사이의 거리 계산
    desired_pos_act = env.data.body('latch_axis').xpos - env.data.body('hand').xpos
    desired_pos_act[1] -= 0.05
    # 목표 방향 (latch axis와 z는 반대, y는 동일 방향 = y축 기준 180도 회전) 계산
    desired_quat = R.from_matrix(env.data.body("latch_axis").xmat.reshape(3, 3)).as_quat()
    desired_quat = np.roll(rotate_quaternion(np.roll(desired_quat, shift=1), [0, 1, 0], 180), shift=-1)
    current_quat = np.roll(env.data.mocap_quat[0], shift=-1)
    # 현재 방향과 목표 방향 사이 slerp (구면 선형 보간)하여 목표 방향까지 1/2 정도 회전
    # - 너무 빠르면 물리적으로 불가능한 움직임이 생길 수 있음
    interp_quat = interpolate_quaternion(current_quat, desired_quat, 2)
    desired_quat_act = np.roll(interp_quat[1], shift=1)

    desired_act = np.append(desired_pos_act, desired_quat_act)
    obs, rew, term, trunc, _ = env.step(desired_act)

    done = term | trunc
    env.render()
        